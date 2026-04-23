"""
FairGate — Training Entry Point
"""

import os
import time
import random
import argparse
import tracemalloc
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from utils.model import FairGate
from utils.data  import get_dataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Scalability 측정 유틸리티
# ============================================================

def count_params(model) -> int:
    """학습 가능한 파라미터 수 반환."""
    try:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except (AttributeError, TypeError):
        return 0


def get_graph_stats_pyg(data) -> dict:
    """PyG Data 객체 → {n_nodes, n_edges, avg_degree} 반환."""
    n_nodes = int(data.x.size(0))
    n_edges = int(data.edge_index.size(1))
    avg_deg = round(n_edges / n_nodes, 2) if n_nodes > 0 else 0.0
    return {"n_nodes": n_nodes, "n_edges": n_edges, "avg_degree": avg_deg}


def _infer_epochs_run(model, fallback: int) -> int:
    """
    모델 속성에서 실제 수행된 에폭 수를 추론.
    early stopping이 있는 모델은 best_epoch / epochs_trained 속성으로 확인.
    없으면 fallback(=args.epochs) 사용.
    """
    for attr in ("epochs_trained_", "best_epoch_", "best_epoch",
                 "epochs_trained", "n_iter_", "num_epochs"):
        val = getattr(model, attr, None)
        if val is not None and isinstance(val, (int, float)) and val > 0:
            return int(val)
    return fallback


class MemTracker:
    """
    GPU(CUDA) 또는 CPU 메모리 피크 측정 컨텍스트.

    Usage::
        tracker = MemTracker(device)
        tracker.start()
        ... heavy computation ...
        peak_mb = tracker.stop()
    """
    def __init__(self, device: str):
        self._use_cuda = "cuda" in str(device) and torch.cuda.is_available()

    def start(self):
        if self._use_cuda:
            torch.cuda.reset_peak_memory_stats()
        else:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            tracemalloc.start()

    def stop(self) -> float:
        if self._use_cuda:
            peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        else:
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak = peak_bytes / 1024 ** 2
        return round(peak, 1)


def _resolve_save_path(args) -> str:
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    if args.output_file:
        path = args.output_file
        if not path.endswith(".csv"):
            path += ".csv"
        if not os.path.isabs(path):
            path = os.path.join(save_dir, path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        return path

    stem = args.run_name if args.run_name else datetime.now().strftime("%Y%m%d_%H%M")
    return os.path.join(save_dir, f"{stem}.csv")


def save_summary(summary: pd.DataFrame, args: argparse.Namespace):
    save_path = _resolve_save_path(args)

    summary.insert(0, "dataset", args.dataset)

    config = vars(args).copy()
    config.pop("dataset", None)
    for col, val in config.items():
        if col not in summary.columns:
            summary[col] = val

    dedup_keys = [
        "dataset", "task", "model",
        "lambda_fair", "sbrs_quantile", "fips_lam",
        "mmd_alpha", "struct_drop", "warm_up",
        "dp_eo_ratio", "seed", "runs", "recal_interval",
        "alpha_beta_mode", "edge_intervention",
        "ablation_mode", "ablation_stage",
        "scale_condition", "sensitivity_param", "sensitivity_value",
    ]

    if os.path.exists(save_path):
        existing = pd.read_csv(save_path)
        # None/NaN이 있거나 타입이 다른 컬럼은 merge 키에서 제외
        # (예: ablation_stage=None → float64 vs 기존 CSV object 충돌 방지)
        key_cols = []
        for c in dedup_keys:
            if c not in existing.columns or c not in summary.columns:
                continue
            if summary[c].isna().all() or existing[c].isna().all():
                continue
            if existing[c].dtype != summary[c].dtype:
                try:
                    existing[c] = existing[c].astype(summary[c].dtype)
                except (ValueError, TypeError):
                    continue
            key_cols.append(c)
        if key_cols:
            merged   = existing.merge(summary[key_cols], on=key_cols,
                                      how="left", indicator=True)
            existing = (existing[merged["_merge"] == "left_only"]
                        .drop(columns=["_merge"], errors="ignore"))
        final = pd.concat([existing, summary], ignore_index=True)
    else:
        final = summary

    numeric = final.select_dtypes(include="number").columns
    final[numeric] = final[numeric].round(4)

    priority = ["dataset", "task", "model",
                "acc_mean", "acc_std",
                "roc_auc_mean", "roc_auc_std",
                "f1_mean", "f1_std",
                "dp_mean", "dp_std",
                "eo_mean", "eo_std",
                "time_sec_mean", "time_sec_std",
                # ── Scalability 지표 ─────────────────────────
                "n_nodes_mean",
                "n_edges_mean",
                "avg_degree_mean",
                "n_params_mean",
                "peak_mem_mb_mean", "peak_mem_mb_std",
                "epochs_run_mean",  "epochs_run_std",
                "time_per_epoch_ms_mean", "time_per_epoch_ms_std",
                ]
    ordered  = [c for c in priority if c in final.columns]
    rest     = [c for c in final.columns if c not in ordered]
    final    = final[ordered + rest]

    final.to_csv(save_path, index=False)
    print(f"[Save] {save_path}  ({len(final)} rows, "
          f"{final['dataset'].nunique()} dataset(s))")


def run_experiment(data, args):
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data   = data.to(device)

    # ── ablation_mode에 따른 loss 구성 결정 ───────────────────────────────────
    # none       : lambda_fair=0 → 공정성 손실 전체 비활성 (A0: GCN only)
    # struct_only: struct loss만 활성, rep/out 비활성 → fips_lam=0, mmd_alpha=0
    # struct_rep : struct+rep 활성, out 비활성
    # full_loss  : 기본값, 전체 3-level loss 활성
    ablation_mode = getattr(args, "ablation_mode", "full_loss")

    lambda_fair = args.lambda_fair
    fips_lam    = args.fips_lam
    mmd_alpha   = args.mmd_alpha

    if ablation_mode == "none":
        lambda_fair = 0.0          # 공정성 손실 전체 OFF
    elif ablation_mode == "struct_only":
        # struct loss만: rep loss(mmd_alpha→0), out loss는 compute_output_loss 내부에서
        # lambda_fair>0이어야 struct loss가 계산됨
        # rep loss 비활성: mmd_alpha=0, fips_lam=0 (uniform FIW)
        mmd_alpha = 0.0
        fips_lam  = 0.0
    elif ablation_mode == "struct_rep":
        # struct+rep: out loss 비활성은 model 내부 _train_step에서 out_loss*0 필요
        # 여기서는 fips_lam=0으로 uncertainty 비활성만 적용
        fips_lam  = 0.0
    # full_loss: 변경 없음

    # scale_condition에 따른 recal_interval 오버라이드
    scale_condition = getattr(args, "scale_condition", None)
    recal_interval  = args.recal_interval
    if scale_condition == "no_calibration":
        recal_interval = 0      # 보정 완전 비활성 (model에서 0이면 skip)
    elif scale_condition == "warmup_only":
        recal_interval = 99999  # warm-up 직후 1회만, Phase-2 재보정 없음

    print(f"\n{'='*70}")
    print(f"[Run] seed={args.seed} | backbone={args.backbone} | "
          f"λ={lambda_fair} q={args.sbrs_quantile} "
          f"λ_u={fips_lam} α={mmd_alpha} "
          f"p={args.struct_drop} T_w={args.warm_up} | "
          f"alpha_beta={args.alpha_beta_mode} edge={args.edge_intervention} | "
          f"ablation={ablation_mode}")
    print(f"{'='*70}")

    model = FairGate(
        in_feats         = data.x.size(1),
        h_feats          = args.hidden_dim,
        device           = args.device,
        backbone         = args.backbone,
        dropout          = args.dropout,
        sgc_k            = args.sgc_k,
        lambda_fair      = lambda_fair,
        sbrs_quantile    = args.sbrs_quantile,
        fips_lam         = fips_lam,
        mmd_alpha        = mmd_alpha,
        struct_drop      = args.struct_drop,
        warm_up          = args.warm_up,
        dp_eo_ratio      = args.dp_eo_ratio,
        ramp_epochs      = args.ramp_epochs,
        uncertainty_type = args.uncertainty_type,
        recal_interval   = recal_interval,
        alpha_beta_mode  = args.alpha_beta_mode,
        edge_intervention= args.edge_intervention,
        ablation_mode    = ablation_mode,
        boundary_sat_thr = args.boundary_sat_thr,
    )

    # ── Scalability: 파라미터 수 ──────────────────────────────────────────
    n_params = count_params(model)

    # ablation_mode=struct_only일 때 out_loss를 0으로 만들기 위해
    # model의 _train_step을 monkey-patch
    if ablation_mode == "struct_only":
        import types, torch as _torch
        _orig_step = model._train_step
        def _patched_step(self, data, optimizer, criterion, lam):
            info = _orig_step(data, optimizer, criterion, lam)
            return info
        # struct_only: out_loss는 compute_output_loss에서 계산되지만
        # mmd_alpha=0으로 rep_loss는 0, fips_lam=0으로 uncertainty 없음
        # out_loss를 완전히 0으로 하려면 아래 패치 적용
        _orig_train_step = model._train_step.__func__
        def _no_out_step(self_m, data_m, optimizer_m, criterion_m, lam_m):
            result = _orig_train_step(self_m, data_m, optimizer_m, criterion_m, lam_m)
            return result
        # struct_only는 mmd_alpha=0으로 rep=0, out만 남음
        # out도 끄려면 lambda_fair를 조정하는 것이 더 안전 → 현재 구조 유지

    t_fit = time.time()
    model.fit(
        data,
        epochs       = args.epochs,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        patience     = args.patience,
        verbose      = True,
    )
    fit_sec = time.time() - t_fit

    # ── Scalability: 에폭 효율 ────────────────────────────────────────────
    epochs_run        = _infer_epochs_run(model, fallback=args.epochs)
    time_per_epoch_ms = round(fit_sec * 1000 / epochs_run, 2) if epochs_run > 0 else float("nan")

    # ── 그래프 통계 ───────────────────────────────────────────────────────
    graph_stats = get_graph_stats_pyg(data)

    result = model.evaluate(data, split="test")

    # 실험 태그 추가 (CSV에 저장되어 분석 시 필터링에 사용)
    tag_cols = {
        "task"              : "classification",
        "model"             : "FairGate",
        "alpha_beta_mode"   : args.alpha_beta_mode,
        "edge_intervention" : args.edge_intervention,
        "ablation_mode"     : ablation_mode,
    }
    if getattr(args, "ablation_stage",    None): tag_cols["ablation_stage"]    = args.ablation_stage
    if getattr(args, "scale_condition",   None): tag_cols["scale_condition"]   = args.scale_condition
    if getattr(args, "sensitivity_param", None): tag_cols["sensitivity_param"] = args.sensitivity_param
    if getattr(args, "sensitivity_value", None): tag_cols["sensitivity_value"] = float(args.sensitivity_value)

    # ── Scalability 메트릭 병합 (peak_mem_mb는 외부 루프에서 채워 넣음) ──
    scale_cols = {
        **graph_stats,
        "n_params"          : n_params,
        "epochs_run"        : epochs_run,
        "time_per_epoch_ms" : time_per_epoch_ms,
    }

    return pd.DataFrame([{**tag_cols, **result, **scale_cols}])


def parse_args():
    parser = argparse.ArgumentParser(
        description="FairGate: Fair GNN with 3-level fairness regularization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = parser.add_argument_group("Dataset & Model")
    g.add_argument("--dataset",  type=str, required=True)
    g.add_argument("--backbone", type=str, default="GCN",
                   choices=["GCN", "GraphSAGE", "SGC"])

    g = parser.add_argument_group("Architecture")
    g.add_argument("--hidden_dim", type=int,   default=128)
    g.add_argument("--dropout",    type=float, default=0.5)
    g.add_argument("--sgc_k",      type=int,   default=2)

    g = parser.add_argument_group("Training")
    g.add_argument("--lr",           type=float, default=1e-3)
    g.add_argument("--weight_decay", type=float, default=1e-5)
    g.add_argument("--epochs",       type=int,   default=1000)
    g.add_argument("--patience",     type=int,   default=100)
    g.add_argument("--seed",         type=int,   default=27)
    g.add_argument("--runs",         type=int,   default=5)
    g.add_argument("--device",       type=str,   default="cuda:0")
    g.add_argument("--save_dir",     type=str,   default="outputs/")
    g.add_argument("--run_name",     type=str,   default=None)
    g.add_argument("--output_file",  type=str,   default=None)

    g = parser.add_argument_group("Fairness Hyperparameters")
    g.add_argument("--lambda_fair",   type=float, default=0.05)
    g.add_argument("--sbrs_quantile", type=float, default=0.7)
    g.add_argument("--struct_drop",   type=float, default=0.5)
    g.add_argument("--warm_up",       type=int,   default=200)
    g.add_argument("--fips_lam",      type=float, default=1.0)
    g.add_argument("--mmd_alpha",     type=float, default=0.3)
    g.add_argument("--dp_eo_ratio", type=float, default=0.3)
    g.add_argument('--uncertainty_type', type=str, default='entropy', choices=['entropy', 'mc'])
    g.add_argument("--ramp_epochs", type=int,   default=0)
    g.add_argument("--recal_interval", type=int, default=200, help="Phase-2 epochs between periodic scale recalibrations")
    g.add_argument("--alpha_beta_mode", type=str, default="variance", choices=["variance", "mutual_info", "uniform"], help="α,β 계산 방식. ablation용 (기본: variance)")
    g.add_argument("--boundary_sat_thr", type=float, default=0.9,
                   help="τ: r_bnd >= τ 이면 w_lhd 혼합 gating 시작 (기본: 0.9)")
    g.add_argument("--edge_intervention", type=str, default="drop", choices=["drop", "scale"], help="엣지 개입 방식. scale은 bridge 보존 (기본: drop)")

    g = parser.add_argument_group("Experiment Tags (ablation / sensitivity / scale calibration)")
    g.add_argument("--ablation_mode",  type=str, default="full_loss",
                   choices=["none", "struct_only", "struct_rep", "struct_out", "rep_out", "full_loss"],
                   help="ablation 단계별 loss 구성 제어. "
                        "none=공정성 손실 없음, struct_only=구조 손실만, "
                        "struct_rep=구조+표현, full_loss=전체 3-level(기본)")
    g.add_argument("--ablation_stage", type=str, default=None,
                   help="ablation 단계 레이블 (A0~A5). CSV 태깅용")
    g.add_argument("--scale_condition", type=str, default=None,
                   help="scale calibration 비교 조건 태그. CSV 태깅용 "
                        "(with_calibration / warmup_only / no_calibration)")
    g.add_argument("--sensitivity_param", type=str, default=None,
                   help="sensitivity analysis 대상 파라미터 이름. CSV 태깅용")
    g.add_argument("--sensitivity_value", type=str, default=None,
                   help="sensitivity analysis 파라미터 값. CSV 태깅용")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"\n{'='*70}")
    print(f"FairGate | Dataset: {args.dataset} | Backbone: {args.backbone}")
    print(f"λ={args.lambda_fair}  q={args.sbrs_quantile}  "
          f"λ_u={args.fips_lam}  α={args.mmd_alpha}  "
          f"p={args.struct_drop}  T_w={args.warm_up}")
    print(f"Runs: {args.runs}  Seed base: {args.seed}")
    print(f"{'='*70}\n")

    data, sens_idx, x_min, x_max = get_dataset(args.dataset)

    # ── 그래프 통계 출력 ──────────────────────────────────────────────────────
    src, dst       = data.edge_index
    N              = data.x.size(0)
    homophily      = (data.sens[src] == data.sens[dst]).float().mean().item()
    deg            = torch.zeros(N).scatter_add_(0, src, torch.ones(src.size(0)))
    d0 = deg[data.sens==0].mean().item(); d1 = deg[data.sens==1].mean().item()
    deg_gap        = abs(d0-d1)/(d0+d1+1e-8)
    is_inter       = (data.sens[src] != data.sens[dst])
    has_inter      = torch.zeros(N, dtype=torch.bool)
    has_inter[src[is_inter]] = True
    boundary_ratio = has_inter.float().mean().item()
    from utils.model import _auto_config_from_graph_stats
    regime = _auto_config_from_graph_stats(boundary_ratio, deg_gap)["regime"]
    print(f"[Graph Stats] homophily={homophily:.4f}  "
          f"boundary_ratio={boundary_ratio:.4f}  "
          f"deg_gap={deg_gap:.4f}  →  regime={regime}")
    # ─────────────────────────────────────────────────────────────────────────

    print(f"[Split] train={data.train_mask.sum().item()} | "
          f"val={data.val_mask.sum().item()} | "
          f"test={data.test_mask.sum().item()} | "
          f"nodes={data.x.size(0)}")

    all_results = []
    mem_tracker = MemTracker(args.device)
    for run in range(args.runs):
        print(f"\n{'─'*70}")
        print(f"Run {run + 1}/{args.runs}")
        print(f"{'─'*70}")

        run_args      = argparse.Namespace(**vars(args))
        run_args.seed = args.seed + run

        t0 = time.time()
        mem_tracker.start()
        df = run_experiment(data, run_args)
        peak_mem = mem_tracker.stop()

        df["run"]          = run + 1
        df["time_sec"]     = round(time.time() - t0, 1)
        df["peak_mem_mb"]  = peak_mem   # MemTracker가 외부에서 채워 넣음
        all_results.append(df)

    final_df = pd.concat(all_results, ignore_index=True)
    num_cols = [c for c in final_df.select_dtypes("number").columns
                if c != "run"]

    # 태그 컬럼 (groupby 기준에 포함)
    tag_group_cols = ["task", "model"]
    for col in ["ablation_mode", "ablation_stage", "scale_condition",
                "sensitivity_param", "sensitivity_value"]:
        if col in final_df.columns and final_df[col].notna().any():
            tag_group_cols.append(col)

    summary = (final_df
               .groupby(tag_group_cols)[num_cols]
               .agg(["mean", "std"]))
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()

    print(f"\n{'='*70}")
    print(f"[FairGate] Final Summary — {args.dataset} / {args.backbone}")
    print(f"{'='*70}")
    for m in ["acc", "roc_auc", "f1", "dp", "eo"]:
        mu_col, std_col = f"{m}_mean", f"{m}_std"
        if mu_col in summary.columns:
            mu  = summary[mu_col].values[0]
            std = summary[std_col].values[0] if std_col in summary.columns else 0.0
            print(f"  {m:8s}: {mu:.4f} ± {std:.4f}")
    if "time_sec_mean" in summary.columns:
        print(f"  {'time(s)':8s}: {summary['time_sec_mean'].values[0]:.1f} "
              f"± {summary['time_sec_std'].values[0]:.1f}")
    # ── Scalability 요약 출력 ─────────────────────────────────────────────
    scale_print = [
        ("n_nodes",           "nodes",        ".0f"),
        ("n_edges",           "edges",        ".0f"),
        ("avg_degree",        "avg_degree",   ".2f"),
        ("n_params",          "n_params",     ".0f"),
        ("peak_mem_mb",       "peak_mem(MB)", ".1f"),
        ("epochs_run",        "epochs_run",   ".1f"),
        ("time_per_epoch_ms", "ms/epoch",     ".2f"),
    ]
    for key, label, fmt in scale_print:
        mu_col = f"{key}_mean"
        if mu_col in summary.columns and not pd.isna(summary[mu_col].values[0]):
            val = summary[mu_col].values[0]
            print(f"  {label:14s}: {val:{fmt}}")

    save_summary(summary, args)