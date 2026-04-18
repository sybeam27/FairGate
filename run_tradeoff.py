"""
run_tradeoff.py — Accuracy–Fairness Tradeoff Curve (Pareto Frontier)

lambda_fair를 0에서 0.5까지 sweep하여
FairGate vs 대표 baseline들의 Pareto frontier를 비교한다.

출력:
    outputs/exp_tradeoff.csv
    outputs/analysis/tradeoff_plot.py   (시각화 스크립트)
    outputs/analysis/tradeoff.tex

실행:
    python run_tradeoff.py
    python run_tradeoff.py --datasets pokec_z german --dry_run
"""

import os, sys, argparse, subprocess
import numpy as np
import pandas as pd

DEVICE = "cuda:1"

DEFAULT_DATASETS = ["pokec_z", "german", "nba", "credit"]

FAIRGATE_CONFIGS = {
    # ── Pokec 계열 ──────────────────────────────────────────────────────────
    "pokec_z":    dict(sbrs_quantile=0.9, struct_drop=0.5, warm_up=400),
    "pokec_z_g":  dict(sbrs_quantile=0.6, struct_drop=0.5, warm_up=100),
    "pokec_n":    dict(sbrs_quantile=0.5, struct_drop=0.5, warm_up=400),
    "pokec_n_g":  dict(sbrs_quantile=0.8, struct_drop=0.5, warm_up=400),
    # ── 소규모 그래프 ────────────────────────────────────────────────────────
    "credit":     dict(sbrs_quantile=0.5, struct_drop=0.7, warm_up=200),
    "recidivism": dict(sbrs_quantile=0.9, struct_drop=0.2, warm_up=100),
    "income":     dict(sbrs_quantile=0.5, struct_drop=0.7, warm_up=200),
    "german":     dict(sbrs_quantile=0.9, struct_drop=0.2, warm_up=100),
    "nba":        dict(sbrs_quantile=0.5, struct_drop=0.3, warm_up=200),
}

FIXED = dict(
    backbone="GCN", hidden_dim=128, dropout=0.5, sgc_k=2,
    lr=1e-3, weight_decay=1e-5, epochs=500, patience=501,
    runs=3, seed=27, fips_lam=1.0, mmd_alpha=0.3,
    dp_eo_ratio=0.3, uncertainty_type="entropy",
    ramp_epochs=0, recal_interval=200,
    alpha_beta_mode="variance", edge_intervention="drop",
)

# lambda_fair sweep 범위
LAMBDA_GRID = [0.0, 0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


def build_cmd(dataset: str, lambda_fair: float, output_file: str) -> list:
    cfg = FAIRGATE_CONFIGS[dataset]
    run_name = f"tradeoff_lf{str(lambda_fair).replace('.','p')}_{dataset}"
    return [
        sys.executable, "-m", "utils.train",
        "--dataset",           dataset,
        "--backbone",          FIXED["backbone"],
        "--output_file",       output_file,
        "--run_name",          run_name,
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
        "--lambda_fair",       str(lambda_fair),
        "--sbrs_quantile",     str(cfg["sbrs_quantile"]),
        "--struct_drop",       str(cfg["struct_drop"]),
        "--warm_up",           str(cfg["warm_up"]),
        "--fips_lam",          str(FIXED["fips_lam"]),
        "--mmd_alpha",         str(FIXED["mmd_alpha"]),
        "--dp_eo_ratio",       str(FIXED["dp_eo_ratio"]),
        "--uncertainty_type",  FIXED["uncertainty_type"],
        "--ramp_epochs",       str(FIXED["ramp_epochs"]),
        "--recal_interval",    str(FIXED["recal_interval"]),
        "--alpha_beta_mode",   FIXED["alpha_beta_mode"],
        "--edge_intervention", FIXED["edge_intervention"],
    ]


def run_cmd(cmd, dry_run, log_path):
    if dry_run:
        print("    $ " + " ".join(cmd)); return True
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


def write_plot_script(output_dir: str, datasets: list):
    DNAME = {"pokec_z":"Pokec-Z","german":"German","nba":"NBA",
             "credit":"Credit","recidivism":"Recidivism","income":"Income"}
    code = f'''"""
tradeoff_plot.py — Accuracy–Fairness Pareto Frontier 시각화
실행: python {output_dir}/tradeoff_plot.py
"""
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

df = pd.read_csv("outputs/exp_tradeoff.csv")
if "model" in df.columns:
    df = df[df["model"] == "FairGate"]

DATASETS = {datasets}
DNAME = {DNAME}

fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.5*len(DATASETS), 4))
if len(DATASETS) == 1:
    axes = [axes]

for ax, ds in zip(axes, DATASETS):
    sub = df[df["dataset"] == ds].copy()
    if sub.empty:
        ax.set_title(DNAME.get(ds, ds)); continue

    sub = sub.sort_values("lambda_fair")
    sub["fair"] = sub["dp_mean"] + sub["eo_mean"]

    # FairGate tradeoff curve
    ax.plot(sub["fair"], sub["roc_auc_mean"],
            "o-", color="#2563EB", linewidth=2, markersize=6,
            label="FairGate", zorder=3)

    # lambda 값 레이블 (일부만)
    for _, row in sub.iterrows():
        if row["lambda_fair"] in [0.0, 0.05, 0.20, 0.50]:
            ax.annotate(f"λ={{row['lambda_fair']}}",
                        (row["fair"], row["roc_auc_mean"]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color="#2563EB")

    # Pareto frontier 강조 (AUC 기준 non-dominated points)
    pareto = []
    pts = list(zip(sub["fair"].values, sub["roc_auc_mean"].values))
    for i, (f, a) in enumerate(pts):
        dominated = any(f2 <= f and a2 >= a and (f2 < f or a2 > a)
                        for j, (f2, a2) in enumerate(pts) if j != i)
        if not dominated:
            pareto.append((f, a))
    if pareto:
        pareto = sorted(pareto)
        px, py = zip(*pareto)
        ax.plot(px, py, "--", color="#DC2626", linewidth=1.5,
                alpha=0.7, label="Pareto frontier")

    ax.set_xlabel("ΔDP + ΔEO (↓)", fontsize=10)
    ax.set_ylabel("AUC (↑)", fontsize=10)
    ax.set_title(DNAME.get(ds, ds), fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("FairGate: Accuracy–Fairness Tradeoff",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out = os.path.join("{output_dir}", "tradeoff_plot.pdf")
plt.savefig(out, bbox_inches="tight", dpi=150)
print(f"[Saved] {{out}}")
plt.show()
'''
    path = os.path.join(output_dir, "tradeoff_plot.py")
    with open(path, "w") as f:
        f.write(code)
    print(f"[Saved] {path}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--datasets",    nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--lambda_grid", nargs="+", type=float, default=LAMBDA_GRID)
    p.add_argument("--output_file", type=str,  default="exp_tradeoff.csv")
    p.add_argument("--output_dir",  type=str,  default="outputs/analysis")
    p.add_argument("--log_dir",     type=str,  default="logs/tradeoff")
    p.add_argument("--dry_run",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    total = len(args.datasets) * len(args.lambda_grid)

    print(f"\n{'='*65}")
    print(f"Accuracy–Fairness Tradeoff Curve")
    print(f"  datasets : {args.datasets}")
    print(f"  λ grid   : {args.lambda_grid}")
    print(f"  total    : {total} runs")
    print(f"{'='*65}")

    step = 0
    for lf in args.lambda_grid:
        for ds in args.datasets:
            step += 1
            cmd = build_cmd(ds, lf, args.output_file)
            log_path = os.path.join(args.log_dir,
                                    f"{ds}_lf{str(lf).replace('.','p')}.log")
            print(f"  [{step:2d}/{total}] {ds:<14}  lambda_fair={lf}")
            run_cmd(cmd, args.dry_run, log_path)

    if not args.dry_run:
        write_plot_script(args.output_dir, args.datasets)

    print(f"\n완료. 결과: {args.output_file}")
    print(f"시각화: python {args.output_dir}/tradeoff_plot.py")


if __name__ == "__main__":
    main()
