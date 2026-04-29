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

from utils.model_dualhead import FairGate, _auto_config_from_graph_stats
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

    summary = summary.copy()
    summary.insert(0, "dataset", args.dataset)

    # Attach all runtime/config arguments to the result row.
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
        "gating_mode_override", "fiw_weight_mode",
        "fiw_adaptive", "adaptive_probe_epochs",
        "adaptive_eta", "adaptive_auc_tol",
        "ablation_mode", "ablation_stage",
        "scale_condition",
        "lambda_uq", "uq_width_penalty", "use_uq_weighted_loss",
        "sensitivity_param", "sensitivity_value",
    ]

    # Keep only keys that exist in the new summary.
    # For old CSV files, missing key columns are added as NA sentinels below.
    key_cols = [c for c in dedup_keys if c in summary.columns]

    # Normalize metadata-like key columns before merge.
    # This prevents pandas merge errors such as:
    # "You are trying to merge on float64 and object columns for key 'scale_condition'."
    NA_SENTINEL = "__NA__"

    def _normalize_key_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        df = df.copy()
        for c in cols:
            if c not in df.columns:
                df[c] = NA_SENTINEL

            # Use string normalization for robust deduplication across old/new CSVs.
            # Numeric-looking keys are also converted to strings, so 0 and 0.0 can
            # still differ if the original logs differ; we normalize simple .0 forms below.
            s = df[c].where(pd.notna(df[c]), NA_SENTINEL).astype(str)

            # Normalize common null string variants.
            s = s.replace({
                "nan": NA_SENTINEL,
                "NaN": NA_SENTINEL,
                "None": NA_SENTINEL,
                "<NA>": NA_SENTINEL,
                "": NA_SENTINEL,
            })

            # Make boolean strings consistent.
            s = s.replace({
                "True": "true",
                "False": "false",
                "TRUE": "true",
                "FALSE": "false",
            })

            # Normalize numeric strings like "200.0" -> "200" where safe.
            def _strip_trailing_zero(x: str) -> str:
                if x == NA_SENTINEL:
                    return x
                try:
                    f = float(x)
                    if f.is_integer():
                        return str(int(f))
                    return str(f)
                except Exception:
                    return x

            df[c] = s.map(_strip_trailing_zero)

        return df

    if os.path.exists(save_path):
        existing = pd.read_csv(save_path)

        # Make sure old and new frames both contain all dedup keys used for this run.
        existing = _normalize_key_columns(existing, key_cols)
        summary_norm = _normalize_key_columns(summary, key_cols)

        # Use normalized keys only for deduplication.
        merged = existing.merge(
            summary_norm[key_cols].drop_duplicates(),
            on=key_cols,
            how="left",
            indicator=True,
        )

        existing = existing.loc[merged["_merge"] == "left_only"].copy()
        existing = existing.drop(columns=["_merge"], errors="ignore")

        # Concat the original summary, not only normalized summary,
        # so numeric result/config columns remain numeric before final rounding.
        # But normalized key columns are copied back to avoid future dtype conflicts.
        for c in key_cols:
            summary[c] = summary_norm[c]

        final = pd.concat([existing, summary], ignore_index=True, sort=False)
    else:
        summary = _normalize_key_columns(summary, key_cols)
        final = summary

    # Round numeric result columns.
    numeric = final.select_dtypes(include="number").columns
    final[numeric] = final[numeric].round(4)

    priority = [
        "dataset", "task", "model",
        "acc_mean", "acc_std",
        "roc_auc_mean", "roc_auc_std",
        "f1_mean", "f1_std",
        "dp_mean", "dp_std",
        "eo_mean", "eo_std",
        "time_sec_mean", "time_sec_std",
        "selected_fiw_policy",
        "selected_gating_mode",
        "selected_alpha_beta_mode",
        "selected_fiw_weight_mode",
    ]
    ordered = [c for c in priority if c in final.columns]
    rest = [c for c in final.columns if c not in ordered]
    final = final[ordered + rest]

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    final.to_csv(save_path, index=False)

    print(
        f"[Save] {save_path}  ({len(final)} rows, "
        f"{final['dataset'].nunique()} dataset(s))"
    )

def run_experiment(data, args):
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data   = data.to(device)

    print(f"\n{'='*70}")
    print(f"[Run] seed={args.seed} | backbone={args.backbone} | "
          f"λ={args.lambda_fair} q={args.sbrs_quantile} "
          f"λ_u={args.fips_lam} α={args.mmd_alpha} "
          f"p={args.struct_drop} T_w={args.warm_up} | "
          f"alpha_beta={args.alpha_beta_mode} edge={args.edge_intervention} "
          f"gate={args.gating_mode_override} wmode={args.fiw_weight_mode} "
          f"adaptive={args.fiw_adaptive} probe={args.adaptive_probe_epochs} "
          f"unc={args.uncertainty_type} lambda_uq={args.lambda_uq} "
          f"uq_width={args.uq_width_penalty} uq_weighted={args.use_uq_weighted_loss} "
          f"ablation={args.ablation_mode} scale_cond={args.scale_condition}")
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
        gating_mode_override = args.gating_mode_override,
        fiw_weight_mode = args.fiw_weight_mode,
        fiw_adaptive = args.fiw_adaptive,
        adaptive_probe_epochs = args.adaptive_probe_epochs,
        adaptive_eta = args.adaptive_eta,
        adaptive_auc_tol = args.adaptive_auc_tol,
        ablation_mode = args.ablation_mode,
        disable_scale_calibration = (args.scale_condition == "no_calibration"),
        lambda_uq = args.lambda_uq,
        uq_width_penalty = args.uq_width_penalty,
        use_uq_weighted_loss = args.use_uq_weighted_loss,
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
        "gating_mode_override": args.gating_mode_override,
        "fiw_weight_mode": args.fiw_weight_mode,
        "fiw_adaptive": args.fiw_adaptive,
        "adaptive_probe_epochs": args.adaptive_probe_epochs,
        "adaptive_eta": args.adaptive_eta,
        "adaptive_auc_tol": args.adaptive_auc_tol,
        "ablation_mode": args.ablation_mode,
        "scale_condition": args.scale_condition,
        "lambda_uq": args.lambda_uq,
        "uq_width_penalty": args.uq_width_penalty,
        "use_uq_weighted_loss": args.use_uq_weighted_loss,
        "selected_gating_mode": model.gating_mode_override,
        "selected_alpha_beta_mode": model.alpha_beta_mode,
        "selected_fiw_weight_mode": model.fiw_weight_mode,
        "selected_fiw_policy": (
            (model._adaptive_choice or {}).get("cand", {}).get("name", "fixed")
        ),
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
    g.add_argument("--struct_drop",   type=float, default=0.5)
    g.add_argument("--warm_up",       type=int,   default=200)
    g.add_argument("--fips_lam",      type=float, default=1.0)
    g.add_argument("--mmd_alpha",     type=float, default=0.3)
    g.add_argument("--dp_eo_ratio", type=float, default=0.3)
    g.add_argument('--uncertainty_type', type=str, default='entropy', choices=['entropy', 'mc', 'dual'])
    g.add_argument('--lambda_uq', type=float, default=0.0,
                   help='Dual-head uncertainty auxiliary loss coefficient. 0 disables UQ loss.')
    g.add_argument('--uq_width_penalty', type=float, default=0.05,
                   help='Penalty on prediction interval width for dual-head UQ.')
    g.add_argument('--use_uq_weighted_loss', action='store_true',
                   help='Weight dual-head UQ loss by FIW node weights. Use only after unweighted UQ is stable.')
    g.add_argument("--ramp_epochs", type=int,   default=0)
    g.add_argument("--recal_interval", type=int, default=200, help="Phase-2 epochs between periodic scale recalibrations")
    g.add_argument("--alpha_beta_mode", type=str, default="variance", choices=["variance", "mutual_info", "uniform", "bnd_only", "deg_only", "random"], help="α,β 계산 방식. ablation용 (기본: variance)")
    g.add_argument("--edge_intervention", type=str, default="drop", choices=["drop", "scale"], help="엣지 개입 방식. scale은 bridge 보존 (기본: drop)")
    g.add_argument("--gating_mode_override", type=str, default="adaptive",
                   choices=["none", "score", "adaptive", "boundary", "degree",
                            "boundary_degree", "loss", "random"],
                   help="FIW gating policy. 'adaptive' uses graph-regime-aware gating.")
    g.add_argument("--fiw_weight_mode", type=str, default="continuous_uncert",
                   choices=["uniform", "struct_only", "continuous_uncert",
                            "binary_mean", "matched_random_perm"],
                   help="FIW weighting mode")
    g.add_argument("--fiw_adaptive", action="store_true",
                   help="Enable validation-based Adaptive FIW policy selection after warm-up")
    g.add_argument("--adaptive_probe_epochs", type=int, default=20,
                   help="Probe epochs per FIW candidate during adaptive selection")
    g.add_argument("--adaptive_eta", type=float, default=1.0,
                   help="AUC-preserving penalty coefficient for adaptive FIW selection")
    g.add_argument("--adaptive_auc_tol", type=float, default=0.005,
                   help="Allowed validation AUC drop before applying adaptive penalty")

    g = parser.add_argument_group("Ablation & Analysis")
    g.add_argument("--boundary_sat_thr",  type=float, default=0.9,
                   help="Boundary saturation threshold for FIW gating mode switch")
    g.add_argument("--ablation_mode",     type=str,   default="full_loss",
                   choices=["none", "struct_only", "struct_rep", "struct_out",
                            "rep_out", "full_loss"],
                   help="Ablation mode for 3-level loss components")
    g.add_argument("--ablation_stage",    type=str,   default=None,
                   help="Ablation stage label (A0/A3/A5/R1.../F0/F1...) for logging")
    g.add_argument("--scale_condition",   type=str,   default=None,
                   help="Scale calibration condition label for logging")
    g.add_argument("--sensitivity_param", type=str,   default=None,
                   help="Parameter name being swept in sensitivity analysis")
    g.add_argument("--sensitivity_value", type=float, default=None,
                   help="Parameter value being swept in sensitivity analysis")


    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"\n{'='*70}")
    print(f"FairGate | Dataset: {args.dataset} | Backbone: {args.backbone}")
    print(f"λ={args.lambda_fair}  q={args.sbrs_quantile}  "
          f"λ_u={args.fips_lam}  α={args.mmd_alpha}  "
          f"p={args.struct_drop}  T_w={args.warm_up}  "
          f"gate={args.gating_mode_override}  wmode={args.fiw_weight_mode}  "
          f"adaptive={args.fiw_adaptive}  unc={args.uncertainty_type}  "
          f"lambda_uq={args.lambda_uq}")
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

    # Preserve selected Adaptive FIW policy across runs.
    # If different seeds choose different policies, record the most frequent one.
    for col in [
        "selected_fiw_policy",
        "selected_gating_mode",
        "selected_alpha_beta_mode",
        "selected_fiw_weight_mode",
    ]:
        if col in final_df.columns:
            vals = final_df[col].dropna().mode()
            summary[col] = vals.iloc[0] if len(vals) else None

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