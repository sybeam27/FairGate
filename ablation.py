"""
ablation.py — FairGate Ablation Study

기본 GCN에서 시작해 구성 요소를 하나씩 추가하며 각 설계 선택의 기여를 검증한다.

Ablation 단계:
    A0  GCN              기본 GCN (공정성 제약 없음)
    A1  +L_struct        + 구조 레벨 fairness loss
    A2  +L_rep           + 표현 레벨 fairness loss
    A3  +L_out           + 출력 레벨 fairness loss  (= 3-level, 균일 가중치)
    A4  +FIW-Struct      + FIW 1단계: boundary-only 구조 가중치 (uncertainty 없음)
    A5  +FIW-Full        + FIW 2단계: uncertainty 기반 가중치 조정 (최종 FairGate)

Usage:
    # 단일 데이터셋
    python ablation.py --dataset pokec_z --run_name ablation_v1

    # 여러 데이터셋
    python ablation.py --dataset pokec_z pokec_n german credit recidivism pokec_z_g pokec_n_g\\
                       --stages A1 --run_name exp_a1
    python ablation.py --dataset pokec_z pokec_n german credit recidivism pokec_z_g pokec_n_g\\
                       --stages A2 --run_name exp_a2

    # 특정 단계만
    python ablation.py --dataset income --stages A0 --run_name exp_gcn

    # dry_run
    python ablation.py --dataset pokec_z --dry_run
"""

import copy
import os
import time
import random
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (
    FairGate,
    _build_backbone,
    _get_fair_idx,
    compute_fiw_weights,
    compute_structural_loss,
    compute_output_loss,
    RepresentationLoss,
    _WEIGHT_MIN,
    _WEIGHT_MAX,
    _REFRESH_INTERVAL,
)
from utils.data    import get_dataset
from utils.metrics import evaluate_pyg_model

DEVICE = "cuda:1"

# ============================================================
# Ablation stage definitions
# ============================================================

STAGES = ["A0", "A1", "A2", "A3", "A4", "A5"]

STAGE_LABELS = {
    "A0": "GCN",
    "A1": "GCN + L_struct",
    "A2": "GCN + L_struct + L_rep",
    "A3": "GCN + L_struct + L_rep + L_out  (3-level, uniform w)",
    "A4": "FairGate - FIW-Struct  (boundary-only, no uncertainty)",
    "A5": "FairGate - Full  (hierarchical FIW)",
}

# A0~A3: 어떤 loss 를 켤지
STAGE_LOSS_FLAGS = {
    "A0": dict(use_struct=False, use_rep=False, use_out=False),
    "A1": dict(use_struct=True,  use_rep=False, use_out=False),
    "A2": dict(use_struct=True,  use_rep=True,  use_out=False),
    "A3": dict(use_struct=True,  use_rep=True,  use_out=True),
    "A4": dict(use_struct=True,  use_rep=True,  use_out=True),
    "A5": dict(use_struct=True,  use_rep=True,  use_out=True),
}

# A0~A3: 균일 가중치 사용  A4: FIW 1단계(구조만)  A5: FIW 완전
STAGE_FIW_MODE = {
    "A0": "uniform",
    "A1": "uniform",
    "A2": "uniform",
    "A3": "uniform",
    "A4": "struct_only",   # uncertainty 없이 boundary-only FIW
    "A5": "full",          # 계층적 FIW (최종 모델)
}


# ============================================================
# AblationModel: FairGate 설계를 단계별로 제어
# ============================================================

class AblationModel:
    """
    FairGate 구성 요소를 단계별로 켜고 끄는 ablation wrapper.

    stage별 동작:
        A0  task loss만 학습
        A1  + L_struct (균일 w)
        A2  + L_struct + L_rep (균일 w)
        A3  + L_struct + L_rep + L_out (균일 w)  ← 3-level without FIW
        A4  A3와 동일한 loss + FIW 1단계 (boundary-only, uncertainty 없음)
        A5  A3와 동일한 loss + FIW 완전 (boundary-first → uncertainty-second)
    """

    def __init__(
        self,
        stage,
        in_feats,
        h_feats,
        device,
        backbone        = "GCN",
        dropout         = 0.5,
        sgc_k           = 2,
        lambda_fair     = 0.05,
        sbrs_quantile   = 0.7,
        fips_lam        = 1.0,
        mmd_alpha       = 0.3,
        struct_drop     = 0.5,
        warm_up         = 200,
        dp_eo_ratio     = 0.3,
        ramp_epochs     = 0,
        uncertainty_type= "entropy",
    ):
        assert stage in STAGES, f"Unknown stage: {stage}"

        self.stage            = stage
        self.device           = device
        self.lambda_fair      = lambda_fair
        self.sbrs_quantile    = sbrs_quantile
        self.fips_lam         = fips_lam
        self.struct_drop      = struct_drop
        self.warm_up          = warm_up
        self.dp_eo_ratio      = dp_eo_ratio
        self.ramp_epochs      = ramp_epochs
        self.uncertainty_type = uncertainty_type

        self.use_struct = STAGE_LOSS_FLAGS[stage]["use_struct"]
        self.use_rep    = STAGE_LOSS_FLAGS[stage]["use_rep"]
        self.use_out    = STAGE_LOSS_FLAGS[stage]["use_out"]
        self.fiw_mode   = STAGE_FIW_MODE[stage]

        self.name = f"{backbone}/FairGate-{stage}"

        self.model      = _build_backbone(backbone, in_feats, h_feats,
                                          dropout=dropout, sgc_k=sgc_k).to(device)
        self._rep_loss  = RepresentationLoss(mmd_alpha=mmd_alpha)
        self._scales    = {"struct": 1.0, "rep": 1.0, "out": 1.0}
        self._node_w    = None

    # ── 가중치 초기화 ─────────────────────────────────────────

    def _make_uniform_weight(self, data):
        """A0~A3: 모든 노드에 동일 가중치 1.0 부여"""
        N = data.x.size(0)
        return torch.ones(N, device=data.x.device)

    def _init_node_weights(self, data, model=None):
        """
        stage별 노드 가중치 초기화.
        warm-up 시점(model=None)과 FIW 갱신 시점(model=backbone) 모두 처리.
        """
        if self.fiw_mode == "uniform":
            self._node_w = self._make_uniform_weight(data)

        elif self.fiw_mode == "struct_only":
            # FIW 1단계: boundary-only, uncertainty 없음 (fips_lam=0 으로 전달)
            self._node_w, meta = compute_fiw_weights(
                data,
                model=None,           # uncertainty 미사용
                sbrs_quantile=self.sbrs_quantile,
                fips_lam=0.0,         # uncertainty 반영 강도 0
                uncertainty_type=self.uncertainty_type,
            )
            if model is not None:
                # warm-up 후에도 uncertainty 없이 구조 신호만 갱신
                self._node_w, meta = compute_fiw_weights(
                    data,
                    model=None,
                    sbrs_quantile=self.sbrs_quantile,
                    fips_lam=0.0,
                    uncertainty_type=self.uncertainty_type,
                )

        elif self.fiw_mode == "full":
            # FIW 완전: warm-up 전에는 구조만, 이후 uncertainty 포함
            self._node_w, meta = compute_fiw_weights(
                data,
                model=model,
                sbrs_quantile=self.sbrs_quantile,
                fips_lam=self.fips_lam if model is not None else 0.0,
                uncertainty_type=self.uncertainty_type,
            )

    # ── Scale calibration ────────────────────────────────────

    @torch.no_grad()
    def _calibrate_scales(self, data, criterion):
        if not any([self.use_struct, self.use_rep, self.use_out]):
            return  # A0: fairness loss 없으므로 calibration 불필요

        self.model.eval()
        labels    = data.y.float()
        idx_train = data.train_mask.nonzero(as_tuple=False).view(-1)
        idx_fair  = _get_fair_idx(data)
        sens      = data.sens

        out, h   = self.model(data, return_hidden=True)
        task_val = criterion(out[idx_train], labels[idx_train]).item()

        def _s(loss_fn):
            v = loss_fn().item()
            return task_val / (v + 1e-8)

        if self.use_struct:
            self._scales["struct"] = _s(lambda: compute_structural_loss(
                self.model, data, self._node_w, self.struct_drop))
        if self.use_rep:
            self._scales["rep"] = _s(lambda: self._rep_loss(
                h, sens, self._node_w, idx_fair))
        if self.use_out:
            self._scales["out"] = _s(lambda: compute_output_loss(
                torch.sigmoid(out), labels, sens,
                self._node_w, idx_fair, self.dp_eo_ratio))

        self.model.train()
        print(
            f"[{self.name}] Scale calibrated | task={task_val:.4f} | "
            f"struct×{self._scales['struct']:.2f} "
            f"rep×{self._scales['rep']:.2f} "
            f"out×{self._scales['out']:.2f}"
        )

    # ── Train step ───────────────────────────────────────────

    def _train_step(self, data, optimizer, criterion, lam):
        self.model.train()
        optimizer.zero_grad()

        labels    = data.y.float()
        idx_train = data.train_mask.nonzero(as_tuple=False).view(-1)
        idx_fair  = _get_fair_idx(data)
        sens      = data.sens

        out, h    = self.model(data, return_hidden=True)
        task_loss = criterion(out[idx_train], labels[idx_train])

        struct_loss = rep_loss = out_loss = task_loss.new_tensor(0.0)

        if lam > 0.0:
            if self.use_struct:
                struct_loss = compute_structural_loss(
                    self.model, data, self._node_w, self.struct_drop)
            if self.use_rep:
                rep_loss = self._rep_loss(h, sens, self._node_w, idx_fair)
            if self.use_out:
                out_loss = compute_output_loss(
                    torch.sigmoid(out), labels, sens,
                    self._node_w, idx_fair, self.dp_eo_ratio)

        total = task_loss + lam * (
            self._scales["struct"] * struct_loss
            + self._scales["rep"]  * rep_loss
            + self._scales["out"]  * out_loss
        )

        total.backward()
        optimizer.step()

        return dict(
            total=float(total),
            task=float(task_loss),
            struct=float(struct_loss),
            rep=float(rep_loss),
            out=float(out_loss),
        )

    # ── Val score ────────────────────────────────────────────

    def _val_score(self, result):
        acc = float(result.get("acc", 0.0))
        dp  = abs(float(result.get("dp", 0.0)))
        eo  = abs(float(result.get("eo", 0.0)))
        return acc - self.dp_eo_ratio * dp - (1.0 - self.dp_eo_ratio) * eo

    # ── Fit ──────────────────────────────────────────────────

    def fit(self, data, epochs=1000, lr=1e-3, weight_decay=0.0,
            patience=100, verbose=True, print_interval=50):

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay or 0.0)
        criterion = nn.BCEWithLogitsLoss()

        # Phase 1: warm-up
        self._init_node_weights(data, model=None)
        lam_warmup = 0.0 if self.stage == "A0" else 0.0  # warm-up은 항상 task only

        if verbose:
            print(f"[{self.name}] Phase 1: warm-up {self.warm_up} epochs...")
        for _ in range(self.warm_up):
            self._train_step(data, optimizer, criterion, lam=lam_warmup)

        # 전환점
        if any([self.use_struct, self.use_rep, self.use_out]):
            if verbose:
                print(f"[{self.name}] Updating node weights & calibrating scales...")
            self._init_node_weights(data, model=self.model)
            self._calibrate_scales(data, criterion)
        self.model.train()

        # Phase 2
        best_score = -float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        counter    = 0
        remaining  = epochs - self.warm_up

        if verbose:
            print(f"[{self.name}] Phase 2: main training {remaining} epochs...")

        for epoch in range(remaining):
            # FIW refresh (A4, A5만)
            if self.fiw_mode in ("struct_only", "full"):
                if epoch > 0 and epoch % _REFRESH_INTERVAL == 0:
                    self._init_node_weights(data, model=self.model)
                    self.model.train()

            if self.ramp_epochs > 0 and epoch < self.ramp_epochs:
                lam = self.lambda_fair * (epoch + 1) / self.ramp_epochs
            else:
                lam = self.lambda_fair if self.stage != "A0" else 0.0

            info       = self._train_step(data, optimizer, criterion, lam)
            val_result = evaluate_pyg_model(
                self.model, data, split="val", task_type="classification")
            score      = self._val_score(val_result)

            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(self.model.state_dict())
                counter    = 0
            else:
                counter += 1

            if verbose and (epoch == 0 or (epoch + 1) % print_interval == 0):
                tr = evaluate_pyg_model(
                    self.model, data, split="train", task_type="classification")
                print(
                    f"[{self.name}] Ep {epoch + self.warm_up + 1:04d} | "
                    f"Total {info['total']:.4f} "
                    f"Task {info['task']:.4f} "
                    f"Struct {info['struct']:.4f} "
                    f"Rep {info['rep']:.4f} "
                    f"Out {info['out']:.4f} | "
                    f"Train {tr} | Val {val_result} | Score {score:.4f}"
                )

            if counter >= patience:
                if verbose:
                    print(f"[{self.name}] Early stopping at "
                          f"epoch {epoch + self.warm_up + 1}.")
                break

        self.model.load_state_dict(best_state)
        if verbose:
            print(f"[{self.name}] Done. Best val score: {best_score:.4f}")

    @torch.no_grad()
    def evaluate(self, data, split="test"):
        return evaluate_pyg_model(
            self.model, data, split=split, task_type="classification")


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_save_path(args) -> str:
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    stem = args.run_name if args.run_name else datetime.now().strftime("%Y%m%d_%H%M")
    return os.path.join(save_dir, f"{stem}_ablation.csv")


def save_summary(rows: list, args) -> str:
    """
    train.py 스타일로 run 결과를 mean/std summary로 저장.
    - group key: dataset, stage, stage_label
    - 동일 설정의 기존 행은 교체
    """
    save_path = _resolve_save_path(args)
    df = pd.DataFrame(rows)

    if df.empty:
        # 빈 결과도 저장 경로만 반환
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        pd.DataFrame().to_csv(save_path, index=False)
        return save_path

    # run별 raw 결과 -> summary
    num_cols = [c for c in df.select_dtypes(include="number").columns if c != "run"]

    summary = (
        df.groupby(["dataset", "stage", "stage_label"])[num_cols]
          .agg(["mean", "std"])
          .reset_index()
    )
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    # 실행 설정 추가 (train.py와 유사)
    config = vars(args).copy()
    config.pop("dataset", None)   # dataset은 summary key로 이미 존재
    config.pop("stages", None)    # stage도 summary key로 이미 존재
    config.pop("dry_run", None)
    for col, val in config.items():
        summary[col] = val

    dedup_keys = [
        "dataset", "stage", "stage_label",
        "backbone",
        "hidden_dim", "dropout", "sgc_k",
        "lr", "weight_decay", "epochs", "patience",
        "seed", "runs",
        "lambda_fair", "sbrs_quantile", "fips_lam",
        "mmd_alpha", "struct_drop", "warm_up",
        "dp_eo_ratio", "uncertainty_type", "ramp_epochs",
    ]

    if os.path.exists(save_path):
        existing = pd.read_csv(save_path)
        key_cols = [c for c in dedup_keys if c in existing.columns and c in summary.columns]

        if key_cols:
            merged = existing.merge(summary[key_cols], on=key_cols, how="left", indicator=True)
            existing = existing[merged["_merge"] == "left_only"].drop(columns=["_merge"], errors="ignore")
            final = pd.concat([existing, summary], ignore_index=True)
        else:
            final = pd.concat([existing, summary], ignore_index=True)
    else:
        final = summary

    # train.py 스타일 우선 컬럼
    priority = [
        "dataset", "stage", "stage_label",
        "acc_mean", "acc_std",
        "roc_auc_mean", "roc_auc_std",
        "f1_mean", "f1_std",
        "dp_mean", "dp_std",
        "eo_mean", "eo_std",
        "time_sec_mean", "time_sec_std",
    ]
    ordered = [c for c in priority if c in final.columns]
    rest = [c for c in final.columns if c not in ordered]
    final = final[ordered + rest]

    numeric = final.select_dtypes(include="number").columns
    final[numeric] = final[numeric].round(4)

    final.to_csv(save_path, index=False)
    print(f"[Save] {save_path}  ({len(final)} rows, {final['dataset'].nunique()} dataset(s))")
    return save_path


def print_summary(rows: list, dataset: str):
    df = pd.DataFrame(rows)
    df = df[df["dataset"] == dataset]

    print(f"\n{'='*75}")
    print(f"  Ablation Summary — {dataset}")
    print(f"{'='*75}")
    print(f"  {'Stage':<6}  {'Label':<45}  {'Acc':>6}  {'AUC':>6}  "
          f"{'DP':>6}  {'EO':>6}")
    print(f"  {'-'*70}")

    for stage in STAGES:
        sub = df[df["stage"] == stage]
        if sub.empty:
            continue
        mu  = sub[["acc", "roc_auc", "dp", "eo"]].mean()
        std = sub[["acc", "roc_auc", "dp", "eo"]].std()
        label = STAGE_LABELS[stage][:44]
        print(
            f"  {stage:<6}  {label:<45}  "
            f"{mu['acc']:.4f}  {mu['roc_auc']:.4f}  "
            f"{mu['dp']:.4f}  {mu['eo']:.4f}"
        )
    print(f"{'='*75}")


# ============================================================
# Experiment runner
# ============================================================

def run_stage(stage, dataset, data, args) -> list:
    """단일 stage를 args.runs 회 실행 → run별 결과 list 반환"""
    results = []

    for run in range(args.runs):
        seed = args.seed + run
        set_seed(seed)

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        data   = data.to(device)

        model = AblationModel(
            stage          = stage,
            in_feats       = data.x.size(1),
            h_feats        = args.hidden_dim,
            device         = args.device,
            backbone       = args.backbone,
            dropout        = args.dropout,
            sgc_k          = args.sgc_k,
            lambda_fair    = args.lambda_fair,
            sbrs_quantile  = args.sbrs_quantile,
            fips_lam       = args.fips_lam,
            mmd_alpha      = args.mmd_alpha,
            struct_drop    = args.struct_drop,
            warm_up        = args.warm_up,
            dp_eo_ratio    = args.dp_eo_ratio,
            ramp_epochs    = args.ramp_epochs,
            uncertainty_type = args.uncertainty_type,
        )

        t0 = time.time()
        model.fit(
            data,
            epochs       = args.epochs,
            lr           = args.lr,
            weight_decay = args.weight_decay,
            patience     = args.patience,
            verbose      = True,
        )
        elapsed = round(time.time() - t0, 1)

        metrics = model.evaluate(data, split="test")
        row = {
            "dataset":     dataset,
            "stage":       stage,
            "stage_label": STAGE_LABELS[stage],
            "run":         run + 1,
            "seed":        seed,
            "time_sec":    elapsed,
            **metrics,
        }
        results.append(row)

        print(
            f"  [{stage}] Run {run+1}/{args.runs} | "
            f"acc={metrics.get('acc', 0):.4f}  "
            f"auc={metrics.get('roc_auc', 0):.4f}  "
            f"dp={metrics.get('dp', 0):.4f}  "
            f"eo={metrics.get('eo', 0):.4f}  "
            f"({elapsed}s)"
        )

    return results


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="FairGate Ablation Study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = parser.add_argument_group("Experiment")
    g.add_argument("--dataset",  nargs="+", required=True,
                   help="데이터셋 이름 (복수 지정 가능)")
    g.add_argument("--stages",   nargs="+", default=STAGES,
                   choices=STAGES,
                   help="실행할 ablation 단계 (기본: 전체)")
    g.add_argument("--run_name", type=str, default=None,
                   help="결과 파일 이름 prefix")
    g.add_argument("--save_dir", type=str, default="outputs/")
    g.add_argument("--dry_run",  action="store_true",
                   help="단계 목록만 출력하고 실행하지 않음")

    g = parser.add_argument_group("Model")
    g.add_argument("--backbone",  type=str, default="GCN",
                   choices=["GCN", "GraphSAGE", "SGC"])
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
    g.add_argument("--device",       type=str,   default=DEVICE)

    g = parser.add_argument_group("FairGate Hyperparameters")
    g.add_argument("--lambda_fair",   type=float, default=0.05)
    g.add_argument("--sbrs_quantile", type=float, default=0.7)
    g.add_argument("--fips_lam",      type=float, default=1.0)
    g.add_argument("--mmd_alpha",     type=float, default=0.3)
    g.add_argument("--struct_drop",   type=float, default=0.5)
    g.add_argument("--warm_up",       type=int,   default=200)
    g.add_argument("--dp_eo_ratio",   type=float, default=0.3)
    g.add_argument("--uncertainty_type", type=str, default="entropy",
                   choices=["entropy", "mc"])
    g.add_argument("--ramp_epochs",   type=int,   default=0)

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    args = parse_args()

    print(f"\n{'='*75}")
    print(f"  FairGate Ablation Study")
    print(f"  datasets : {args.dataset}")
    print(f"  stages   : {args.stages}")
    print(f"  backbone : {args.backbone}  hidden={args.hidden_dim}")
    print(f"  runs={args.runs}  seed={args.seed}  epochs={args.epochs}")
    print(f"{'='*75}")

    if args.dry_run:
        print("\n[Dry run] Ablation stages to run:")
        for stage in args.stages:
            print(f"  {stage}  {STAGE_LABELS[stage]}")
        for ds in args.dataset:
            print(f"\n  Dataset: {ds}")
            for stage in args.stages:
                print(f"    {stage}: {STAGE_LABELS[stage]}")
        print()
        import sys; sys.exit(0)

    all_rows = []

    for dataset in args.dataset:
        print(f"\n{'─'*75}")
        print(f"  Dataset: {dataset}")
        print(f"{'─'*75}")

        data, sens_idx, x_min, x_max = get_dataset(dataset)
        print(f"  [Split] train={data.train_mask.sum().item()} | "
              f"val={data.val_mask.sum().item()} | "
              f"test={data.test_mask.sum().item()} | "
              f"nodes={data.x.size(0)}")

        for stage in args.stages:
            print(f"\n  ── Stage {stage}: {STAGE_LABELS[stage]} ──")
            rows = run_stage(stage, dataset, data, args)
            all_rows.extend(rows)

            # 단계별 중간 저장
            save_path = save_summary(all_rows, args)
            print(f"  [Saved] {save_path}")

        print_summary(all_rows, dataset)

    # 최종 저장
    save_path = save_summary(all_rows, args)
    print(f"\n[Final] Results saved → {save_path}")