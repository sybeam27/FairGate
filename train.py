"""
FairGate — Training Entry Point
"""

import os
import time
import random
import argparse
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
    ]

    if os.path.exists(save_path):
        existing  = pd.read_csv(save_path)
        key_cols  = [c for c in dedup_keys
                     if c in existing.columns and c in summary.columns]
        merged    = existing.merge(summary[key_cols], on=key_cols,
                                   how="left", indicator=True)
        existing  = (existing[merged["_merge"] == "left_only"]
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
                "time_sec_mean", "time_sec_std"]
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

    print(f"\n{'='*70}")
    print(f"[Run] seed={args.seed} | backbone={args.backbone} | "
          f"λ={args.lambda_fair} q={args.sbrs_quantile} "
          f"λ_u={args.fips_lam} α={args.mmd_alpha} "
          f"p={args.struct_drop} T_w={args.warm_up} | "
          f"alpha_beta={args.alpha_beta_mode} edge={args.edge_intervention}")
    print(f"{'='*70}")

    model = FairGate(
        in_feats         = data.x.size(1),
        h_feats          = args.hidden_dim,
        device           = args.device,
        backbone         = args.backbone,
        dropout          = args.dropout,
        sgc_k            = args.sgc_k,
        lambda_fair      = args.lambda_fair,
        sbrs_quantile    = args.sbrs_quantile,
        fips_lam         = args.fips_lam,
        mmd_alpha        = args.mmd_alpha,
        struct_drop      = args.struct_drop,
        warm_up          = args.warm_up,
        dp_eo_ratio      = args.dp_eo_ratio,
        ramp_epochs      = args.ramp_epochs,
        uncertainty_type = args.uncertainty_type,
        recal_interval   = args.recal_interval,
        alpha_beta_mode  = args.alpha_beta_mode,
        edge_intervention= args.edge_intervention,
    )

    model.fit(
        data,
        epochs       = args.epochs,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        patience     = args.patience,
        verbose      = True,
    )

    result = model.evaluate(data, split="test")
    return pd.DataFrame([{
        "task"             : "classification",
        "model"            : "FairGate",
        "alpha_beta_mode"  : args.alpha_beta_mode,
        "edge_intervention": args.edge_intervention,
        **result,
    }])


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
    g.add_argument("--fips_lam",      type=float, default=1.0)
    g.add_argument("--mmd_alpha",     type=float, default=0.3)
    g.add_argument("--struct_drop",   type=float, default=0.5)
    g.add_argument("--warm_up",       type=int,   default=200)
    g.add_argument("--dp_eo_ratio", type=float, default=0.3)
    g.add_argument('--uncertainty_type', type=str, default='entropy',
                   choices=['entropy', 'mc'])
    g.add_argument("--ramp_epochs", type=int,   default=0)
    g.add_argument("--recal_interval", type=int, default=200,
                   help="Phase-2 epochs between periodic scale recalibrations")
    g.add_argument("--alpha_beta_mode", type=str, default="variance",
                   choices=["variance", "mutual_info", "uniform"],
                   help="α,β 계산 방식. ablation용 (기본: variance)")
    g.add_argument("--edge_intervention", type=str, default="drop",
                   choices=["drop", "scale"],
                   help="엣지 개입 방식. scale은 bridge 보존 (기본: drop)")

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
    for run in range(args.runs):
        print(f"\n{'─'*70}")
        print(f"Run {run + 1}/{args.runs}")
        print(f"{'─'*70}")

        run_args      = argparse.Namespace(**vars(args))
        run_args.seed = args.seed + run

        t0             = time.time()
        df             = run_experiment(data, run_args)
        df["run"]      = run + 1
        df["time_sec"] = round(time.time() - t0, 1)
        all_results.append(df)

    final_df = pd.concat(all_results, ignore_index=True)
    num_cols = [c for c in final_df.select_dtypes("number").columns
                if c != "run"]

    # backbone → model 로 변경됐으므로 groupby도 model 사용
    summary = (final_df
               .groupby(["task", "model"])[num_cols]
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

    save_summary(summary, args)