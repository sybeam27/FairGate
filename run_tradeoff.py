"""
run_tradeoff.py — FairGate Accuracy-Fairness Tradeoff Analysis

설계 목표:
    "FairGate의 각 구성요소(3-level loss, FIW gating)가
     정확도-공정성 tradeoff frontier 자체를 단계적으로 개선한다"

실험 구조:
    [Main Figure] 설계 C — 구성요소별 frontier 비교
        A0 (GCN)       : lambda 스윕 → frontier (하한)
        A3 (3-level loss, uniform FIW) : lambda 스윕 → frontier (중간)
        A5 (Full FairGate): lambda 스윕 → frontier (최상)
        → frontier가 A0 < A3 < A5 순으로 우측 상단으로 이동
        → 각 구성요소의 기여를 tradeoff 공간에서 직접 시각화

    [Sub Figure] 설계 A — FIW gating 유무 frontier 비교
        A5 (Full, FIW ON) vs R4 (w/o FIW, uniform)
        → 선별적 gating 6개 데이터셋 (100% gating 제외)
        → "FIW가 같은 정확도에서 더 높은 공정성을 달성"

Lambda 스윕 범위: [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
    → 0.0: 공정성 압력 없음 (정확도 최대)
    → 0.40: 강한 공정성 압력 (공정성 최대)
    → 이 범위에서 각 stage의 (AUC, DP+EO) 점군으로 frontier 형성

실행:
    python run_tradeoff.py --mode main          # 설계 C (A0/A3/A5 스윕)
    python run_tradeoff.py --mode sub           # 설계 A (FIW ON/OFF 스윕)
    python run_tradeoff.py --mode both          # 둘 다
    python run_tradeoff.py --dry_run            # 명령어 확인
    python run_tradeoff.py --analyze_only       # 시각화만 재실행
"""

import os, sys, argparse, subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DEVICE = "cuda:1"

# ── 데이터셋 ──────────────────────────────────────────────────────────────────
# Main: 대표 3개 regime (논문 본문) + 나머지 (appendix)
MAIN_DATASETS_PAPER = ["pokec_z", "credit", "german"]     # 본문 메인 figure
MAIN_DATASETS_APPENDIX = [
    "pokec_z_g", "pokec_n", "pokec_n_g",
    "recidivism", "nba", "income",
]
MAIN_DATASETS_ALL = MAIN_DATASETS_PAPER + MAIN_DATASETS_APPENDIX

# Sub (FIW ON/OFF): 100% gating 제외 (FIW 기여 측정 가능한 데이터셋)
SUB_DATASETS = ["pokec_z", "pokec_n", "credit", "recidivism", "nba", "income"]

# Lambda 스윕 범위
LAMBDA_SWEEP = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

FAIRGATE_CONFIGS = {
    "pokec_z"   : dict(sbrs_quantile=0.90, struct_drop=0.5, warm_up=400),
    "pokec_z_g" : dict(sbrs_quantile=0.90, struct_drop=0.5, warm_up=100),
    "pokec_n"   : dict(sbrs_quantile=0.50, struct_drop=0.5, warm_up=400),
    "pokec_n_g" : dict(sbrs_quantile=0.80, struct_drop=0.5, warm_up=400),
    "credit"    : dict(sbrs_quantile=0.50, struct_drop=0.7, warm_up=200),
    "recidivism": dict(sbrs_quantile=0.90, struct_drop=0.2, warm_up=100),
    "income"    : dict(sbrs_quantile=0.50, struct_drop=0.7, warm_up=200),
    "german"    : dict(sbrs_quantile=0.95, struct_drop=0.2, warm_up=100),
    "nba"       : dict(sbrs_quantile=0.50, struct_drop=0.3, warm_up=200),
}

FIXED = dict(
    backbone="GCN", hidden_dim=128, dropout=0.5, sgc_k=2,
    lr=1e-3, weight_decay=1e-5, epochs=500, patience=501,
    runs=5, seed=27, mmd_alpha=0.3,
    dp_eo_ratio=0.3, uncertainty_type="entropy",
    ramp_epochs=0, recal_interval=200,
    alpha_beta_mode="variance", edge_intervention="drop",
    boundary_sat_thr=0.9,
)

# ── Stage 정의 ─────────────────────────────────────────────────────────────────
# Main figure용: A0, A3, A5
MAIN_STAGES = {
    "A0": dict(
        desc="GCN only",
        fips_lam=0.0,
        ablation_mode="none",
        color="#9CA3AF", ls=":", lw=1.8,   # 회색 점선
        label="GCN only (A0)",
    ),
    "A3": dict(
        desc="3-level loss + uniform FIW",
        fips_lam=0.0,               # uncertainty 비활성 = uniform FIW
        ablation_mode="full_loss",
        color="#F59E0B", ls="--", lw=2.0,  # 주황 점선
        label="3-level loss (A3)",
    ),
    "A5": dict(
        desc="Full FairGate (3-level loss + hierarchical FIW)",
        fips_lam=1.0,
        ablation_mode="full_loss",
        color="#2563EB", ls="-",  lw=2.4,  # 파란 실선
        label="Full FairGate (A5)",
    ),
}

# Sub figure용: A5 vs R4
SUB_STAGES = {
    "A5": dict(
        desc="Full FairGate (FIW ON)",
        fips_lam=1.0,
        ablation_mode="full_loss",
        color="#2563EB", ls="-",  lw=2.4,
        label="FIW ON (Full FairGate)",
    ),
    "R4": dict(
        desc="w/o FIW (uniform gating)",
        fips_lam=0.0,
        ablation_mode="full_loss",
        color="#DC2626", ls="--", lw=2.0,
        label="FIW OFF (uniform)",
    ),
}

DATASET_DISPLAY = {
    "pokec_z":"Pokec-Z", "pokec_z_g":"Pokec-Z (g)",
    "pokec_n":"Pokec-N", "pokec_n_g":"Pokec-N (g)",
    "german":"German",   "credit":"Credit",
    "recidivism":"Recidivism", "nba":"NBA", "income":"Income",
}


# ── 명령어 생성 ────────────────────────────────────────────────────────────────

def build_cmd(dataset, stage, stage_cfg, lambda_val, output_file):
    cfg = FAIRGATE_CONFIGS[dataset]
    return [
        sys.executable, "-m", "utils.train",
        "--dataset",           dataset,
        "--backbone",          FIXED["backbone"],
        "--output_file",       output_file,
        "--run_name",          f"tradeoff_{stage}_lam{lambda_val}_{dataset}",
        "--device",            DEVICE,
        "--hidden_dim",        str(FIXED["hidden_dim"]),
        "--dropout",           str(FIXED["dropout"]),
        "--sgc_k",             str(FIXED["sgc_k"]),
        "--lr",                str(FIXED["lr"]),
        "--weight_decay",      str(FIXED["weight_decay"]),
        "--epochs",            str(FIXED["epochs"]),
        "--patience",          str(FIXED["patience"]),
        "--runs",              str(FIXED["runs"]),
        "--seed",              str(FIXED["seed"]),
        "--lambda_fair",       str(lambda_val),
        "--sbrs_quantile",     str(cfg["sbrs_quantile"]),
        "--struct_drop",       str(cfg["struct_drop"]),
        "--warm_up",           str(cfg["warm_up"]),
        "--fips_lam",          str(stage_cfg["fips_lam"]),
        "--mmd_alpha",         str(FIXED["mmd_alpha"]),
        "--dp_eo_ratio",       str(FIXED["dp_eo_ratio"]),
        "--uncertainty_type",  FIXED["uncertainty_type"],
        "--ramp_epochs",       str(FIXED["ramp_epochs"]),
        "--recal_interval",    str(FIXED["recal_interval"]),
        "--alpha_beta_mode",   FIXED["alpha_beta_mode"],
        "--edge_intervention", FIXED["edge_intervention"],
        "--boundary_sat_thr",  str(FIXED["boundary_sat_thr"]),
        "--ablation_mode",     stage_cfg["ablation_mode"],
        "--ablation_stage",    stage,
        "--sensitivity_param", "lambda_fair",
        "--sensitivity_value", str(lambda_val),
    ]


def run_cmd(cmd, dry_run, log_path):
    if dry_run:
        print("    $ " + " ".join(cmd[-20:])); return True
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
        lf.write(proc.stdout)
        for ln in proc.stdout.strip().splitlines()[-3:]:
            print(f"      {ln}")
        if proc.returncode != 0:
            print(f"    [WARN] 실패 — {log_path}"); return False
    return True


# ── Pareto frontier 계산 ───────────────────────────────────────────────────────

def pareto_frontier(points):
    """
    (fair, auc) 점군에서 Pareto frontier 추출.
    fair 낮고 auc 높은 점들이 frontier.
    """
    pts = sorted(points, key=lambda p: p[0])  # fair 기준 정렬
    frontier = []
    best_auc = -np.inf
    for fair, auc in pts:
        if auc > best_auc:
            frontier.append((fair, auc))
            best_auc = auc
    return frontier


# ── 시각화: Main Figure (A0/A3/A5 frontier) ───────────────────────────────────

def plot_main(df: pd.DataFrame, datasets: list, output_dir: str,
              suffix: str = "paper"):
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2))
    if n == 1: axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, ds in zip(axes, datasets):
        ax.set_facecolor("#FAFAFA")

        for stage, scfg in MAIN_STAGES.items():
            sub = df[(df["dataset"] == ds) & (df["ablation_stage"] == stage)]
            if sub.empty: continue

            points = list(zip(
                sub["dp_mean"] + sub["eo_mean"],
                sub["roc_auc_mean"]
            ))
            frontier = pareto_frontier(points)
            fx = [p[0] for p in frontier]
            fy = [p[1] for p in frontier]

            # Pareto frontier 곡선
            ax.plot(fx, fy,
                    color=scfg["color"], linestyle=scfg["ls"],
                    linewidth=scfg["lw"], zorder=4,
                    label=scfg["label"])

            # 개별 점
            for fair, auc in points:
                ax.scatter(fair, auc,
                           color=scfg["color"], s=40, zorder=5,
                           alpha=0.6, edgecolors="white", linewidths=0.8)

            # lambda=0.0 (원점, 최대 정확도) 강조
            lam0 = sub[sub["sensitivity_value"] == 0.0]
            if not lam0.empty:
                ax.scatter(
                    lam0.iloc[0]["dp_mean"] + lam0.iloc[0]["eo_mean"],
                    lam0.iloc[0]["roc_auc_mean"],
                    color=scfg["color"], s=90, zorder=6,
                    edgecolors="black", linewidths=1.2,
                    marker="D"
                )

        # A5의 최적점 (best lambda) 강조 ★
        sub5 = df[(df["dataset"] == ds) & (df["ablation_stage"] == "A5")]
        if not sub5.empty:
            sub5 = sub5.copy()
            sub5["fair"] = sub5["dp_mean"] + sub5["eo_mean"]
            # 정확도 손실 최소화하면서 공정성 최대 = frontier 상의 elbow
            best = sub5.loc[sub5["fair"].idxmin()]
            ax.scatter(best["fair"], best["roc_auc_mean"],
                       color="#2563EB", s=180, marker="*", zorder=8,
                       edgecolors="black", linewidths=0.8)

        ax.set_xlabel(r"$\Delta\mathrm{DP}+\Delta\mathrm{EO}$ ($\downarrow$)",
                      fontsize=10)
        ax.set_ylabel("AUC ($\\uparrow$)", fontsize=10)
        ax.set_title(DATASET_DISPLAY.get(ds, ds), fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.15, linestyle="--")
        for sp in ax.spines.values(): sp.set_linewidth(0.6)

    # 공통 범례
    legend_els = [
        Line2D([0],[0], color=scfg["color"], linestyle=scfg["ls"],
               linewidth=scfg["lw"], label=scfg["label"])
        for scfg in MAIN_STAGES.values()
    ] + [
        Line2D([0],[0], color="none", marker="D", markerfacecolor="gray",
               markeredgecolor="black", markersize=8, linewidth=0,
               label=r"$\lambda_{\mathrm{fair}}=0$ (no fairness pressure)"),
        Line2D([0],[0], color="none", marker="*", markerfacecolor="#2563EB",
               markeredgecolor="black", markersize=12, linewidth=0,
               label="Best fairness point (A5)"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.92, edgecolor="#D1D5DB",
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "FairGate: Accuracy–Fairness Tradeoff\n"
        "Each component progressively improves the Pareto frontier",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    for ext in ["pdf", "png"]:
        path = os.path.join(output_dir, f"tradeoff_main_{suffix}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"[Saved] {path}")
    plt.close()


# ── 시각화: Sub Figure (FIW ON vs OFF) ────────────────────────────────────────

def plot_sub(df: pd.DataFrame, datasets: list, output_dir: str):
    n = len(datasets)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 4.0 * nrows))
    axes = axes.flatten() if n > 1 else [axes]
    fig.patch.set_facecolor("white")

    for ax, ds in zip(axes, datasets):
        ax.set_facecolor("#FAFAFA")

        for stage, scfg in SUB_STAGES.items():
            sub = df[(df["dataset"] == ds) & (df["ablation_stage"] == stage)]
            if sub.empty: continue

            points = list(zip(
                sub["dp_mean"] + sub["eo_mean"],
                sub["roc_auc_mean"]
            ))
            frontier = pareto_frontier(points)
            fx = [p[0] for p in frontier]
            fy = [p[1] for p in frontier]

            ax.plot(fx, fy, color=scfg["color"], linestyle=scfg["ls"],
                    linewidth=scfg["lw"], label=scfg["label"], zorder=4)
            for fair, auc in points:
                ax.scatter(fair, auc, color=scfg["color"], s=40,
                           alpha=0.6, zorder=5, edgecolors="white", linewidths=0.8)

            # lambda별 레이블 (주요 lambda만)
            for _, row in sub[sub["sensitivity_value"].isin([0.0, 0.20, 0.40])].iterrows():
                fair = row["dp_mean"] + row["eo_mean"]
                ax.annotate(
                    f"λ={row['sensitivity_value']:.2f}",
                    xy=(fair, row["roc_auc_mean"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=6.5, color=scfg["color"], alpha=0.8,
                )

        ax.set_xlabel(r"$\Delta\mathrm{DP}+\Delta\mathrm{EO}$ ($\downarrow$)",
                      fontsize=9.5)
        ax.set_ylabel("AUC ($\\uparrow$)", fontsize=9.5)
        ax.set_title(DATASET_DISPLAY.get(ds, ds), fontsize=10.5,
                     fontweight="bold")
        ax.grid(True, alpha=0.15, linestyle="--")
        for sp in ax.spines.values(): sp.set_linewidth(0.6)

    # 빈 subplot 제거
    for ax in axes[len(datasets):]:
        ax.set_visible(False)

    legend_els = [
        Line2D([0],[0], color=scfg["color"], linestyle=scfg["ls"],
               linewidth=scfg["lw"], label=scfg["label"])
        for scfg in SUB_STAGES.values()
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=2,
               fontsize=9.5, framealpha=0.92, edgecolor="#D1D5DB",
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "FIW Gating Effect: FIW ON vs OFF\n"
        "Selective gating improves the frontier on datasets with non-trivial gating",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    for ext in ["pdf", "png"]:
        path = os.path.join(output_dir, f"tradeoff_sub_fiw.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"[Saved] {path}")
    plt.close()


# ── 실험 실행 ──────────────────────────────────────────────────────────────────

def run_experiments(stages_dict, datasets, lambda_sweep,
                    output_file, log_dir, dry_run):
    plans = [
        (stage, scfg, ds, lam)
        for stage, scfg in stages_dict.items()
        for ds in datasets
        for lam in lambda_sweep
        if ds in FAIRGATE_CONFIGS
    ]
    total = len(plans)
    print(f"  실험 수: {total}  "
          f"({len(stages_dict)} stages × {len(datasets)} datasets "
          f"× {len(lambda_sweep)} lambdas)")

    for i, (stage, scfg, ds, lam) in enumerate(plans, 1):
        print(f"\n  [{i:3d}/{total}] {ds:<14} stage={stage}  λ={lam}")
        cmd = build_cmd(ds, stage, scfg, lam, output_file)
        log = os.path.join(log_dir, stage, f"{ds}_lam{lam}.log")
        run_cmd(cmd, dry_run, log)


# ── 메인 ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="FairGate Tradeoff Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", type=str, default="both",
                   choices=["main", "sub", "both"],
                   help="main: A0/A3/A5 frontier / sub: FIW ON vs OFF")
    p.add_argument("--datasets_main", nargs="+",
                   default=MAIN_DATASETS_PAPER,
                   help="Main figure 데이터셋 (기본: 논문 대표 3개)")
    p.add_argument("--datasets_sub",  nargs="+",
                   default=SUB_DATASETS,
                   help="Sub figure 데이터셋 (기본: 선별 6개)")
    p.add_argument("--lambda_sweep",  nargs="+", type=float,
                   default=LAMBDA_SWEEP)
    p.add_argument("--output_file",   type=str,
                   default="analysis/exp_tradeoff.csv")
    p.add_argument("--output_dir",    type=str,
                   default="outputs/figures/tradeoff")
    p.add_argument("--log_dir",       type=str,
                   default="logs/tradeoff")
    p.add_argument("--dry_run",       action="store_true")
    p.add_argument("--analyze_only",  action="store_true",
                   help="실험 없이 시각화만 재실행")
    p.add_argument("--paper_only",    action="store_true",
                   help="논문 대표 3개 데이터셋만 (appendix 제외)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    lam_str = str(args.lambda_sweep)
    print(f"\n{'='*65}")
    print(f"FairGate Tradeoff Analysis  [mode={args.mode}]")
    print(f"  lambda sweep : {lam_str}")
    if args.mode in ("main", "both"):
        print(f"  Main datasets: {args.datasets_main}")
    if args.mode in ("sub", "both"):
        print(f"  Sub datasets : {args.datasets_sub}")
        print(f"  (100% gating 제외 — FIW 기여 측정 가능한 데이터셋)")
    print(f"{'='*65}")

    # ── 실험 실행 ────────────────────────────────────────────────────────────
    if not args.analyze_only:
        if args.mode in ("main", "both"):
            print(f"\n[Main] A0/A3/A5 frontier 실험")
            ds_main = (MAIN_DATASETS_PAPER if args.paper_only
                       else args.datasets_main)
            run_experiments(
                MAIN_STAGES, ds_main,
                args.lambda_sweep,
                args.output_file.replace(".csv", "_main.csv"),
                os.path.join(args.log_dir, "main"),
                args.dry_run,
            )
        if args.mode in ("sub", "both"):
            print(f"\n[Sub] FIW ON vs OFF 실험")
            run_experiments(
                SUB_STAGES, args.datasets_sub,
                args.lambda_sweep,
                args.output_file.replace(".csv", "_sub.csv"),
                os.path.join(args.log_dir, "sub"),
                args.dry_run,
            )

    # ── 시각화 ───────────────────────────────────────────────────────────────
    if not args.dry_run:
        print(f"\n{'='*65}")
        print("시각화")
        print(f"{'='*65}")

        # Main figure
        main_file = args.output_file.replace(".csv", "_main.csv")
        if os.path.exists(main_file) and args.mode in ("main", "both"):
            df_main = pd.read_csv(main_file)

            # 논문 본문용 (대표 3개)
            plot_main(df_main, MAIN_DATASETS_PAPER,
                      args.output_dir, suffix="paper")

            # Appendix용 (전체)
            if not args.paper_only and len(args.datasets_main) > 3:
                plot_main(df_main, args.datasets_main,
                          args.output_dir, suffix="full")

        # Sub figure
        sub_file = args.output_file.replace(".csv", "_sub.csv")
        if os.path.exists(sub_file) and args.mode in ("sub", "both"):
            df_sub = pd.read_csv(sub_file)
            plot_sub(df_sub, args.datasets_sub, args.output_dir)

    print(f"\n완료. 결과: {args.output_dir}/")


if __name__ == "__main__":
    main()