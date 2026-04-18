"""
analyze_scale_calibration.py — Scale Calibration 효과 분석

학습 과정에서 scale calibration이 loss 균형 및 공정성에 미치는 영향을 분석.

비교:
    A) with_calibration    : FairGate 기본 (warm_up 후 보정 + periodic 재보정)
    B) no_calibration      : 보정 없이 raw loss 사용 (recal_interval=99999)
    C) warmup_only         : warm_up 직후 1회만 보정 (recal_interval=99999, 초기 보정만)

학습 곡선(epoch별 loss, DP, EO)을 저장하기 위해
train.py에 --log_training 플래그 사용.

출력:
    outputs/exp_scale_*.csv
    outputs/analysis/scale_calibration_stats.csv
    outputs/analysis/scale_calibration_plot.py

실행:
    python analyze_scale_calibration.py
    python analyze_scale_calibration.py --datasets pokec_z german nba
    python analyze_scale_calibration.py --dry_run
"""

import os, sys, argparse, subprocess
import pandas as pd

DEVICE = "cuda:1"

DEFAULT_DATASETS = ["pokec_z", "german", "nba", "credit", "recidivism"]

FAIRGATE_CONFIGS = {
    "pokec_z":    dict(lambda_fair=0.05, sbrs_quantile=0.7, struct_drop=0.5, warm_up=200),
    "pokec_n":    dict(lambda_fair=0.20, sbrs_quantile=0.7, struct_drop=0.5, warm_up=400),
    "pokec_n_g": dict(lambda_fair=0.01, sbrs_quantile=0.8, struct_drop=0.5, warm_up=400),  # 변경
    "pokec_z_g": dict(lambda_fair=0.20, sbrs_quantile=0.6, struct_drop=0.5, warm_up=100),  # 변경
    # "german":    dict(lambda_fair=0.20, sbrs_quantile=0.9, struct_drop=0.2, warm_up=100),  # 변경
    "german":     dict(lambda_fair=0.15, sbrs_quantile=0.7, struct_drop=0.3, warm_up=100),
    "credit":     dict(lambda_fair=0.10, sbrs_quantile=0.8, struct_drop=0.2, warm_up=100),
    "income":     dict(lambda_fair=0.20, sbrs_quantile=0.5, struct_drop=0.7, warm_up=200),
    "nba":       dict(lambda_fair=0.15, sbrs_quantile=0.8, struct_drop=0.2, warm_up=200),  # 유지
    "recidivism": dict(lambda_fair=0.07, sbrs_quantile=0.7, struct_drop=0.3, warm_up=100),
}

FIXED = dict(
    backbone="GCN", hidden_dim=128, dropout=0.5, sgc_k=2,
    lr=1e-3, weight_decay=1e-5, epochs=500, patience=501,
    runs=3, seed=27, fips_lam=1.0, mmd_alpha=0.3,
    dp_eo_ratio=0.3, uncertainty_type="entropy",
    ramp_epochs=0, alpha_beta_mode="variance", edge_intervention="drop",
)

# 비교 조건 정의
CONDITIONS = {
    "with_calibration": dict(
        recal_interval=200,     # periodic 재보정 (기본값)
        desc="Periodic recalibration (default)",
    ),
    "warmup_only": dict(
        recal_interval=99999,   # warm-up 직후 1회만 (Phase 2 재보정 없음)
        desc="Warm-up calibration only",
    ),
    "no_calibration": dict(
        recal_interval=0,       # 보정 완전 비활성 (train.py에서 0이면 skip)
        desc="No scale calibration",
    ),
}


def build_cmd(dataset: str, condition: str, output_file: str) -> list:
    cfg   = FAIRGATE_CONFIGS[dataset]
    cond  = CONDITIONS[condition]
    run_name = f"scale_{condition}_{dataset}"

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
        "--lambda_fair",       str(cfg["lambda_fair"]),
        "--sbrs_quantile",     str(cfg["sbrs_quantile"]),
        "--struct_drop",       str(cfg["struct_drop"]),
        "--warm_up",           str(cfg["warm_up"]),
        "--fips_lam",          str(FIXED["fips_lam"]),
        "--mmd_alpha",         str(FIXED["mmd_alpha"]),
        "--dp_eo_ratio",       str(FIXED["dp_eo_ratio"]),
        "--uncertainty_type",  FIXED["uncertainty_type"],
        "--ramp_epochs",       str(FIXED["ramp_epochs"]),
        "--recal_interval",    str(cond["recal_interval"]),
        "--alpha_beta_mode",   FIXED["alpha_beta_mode"],
        "--edge_intervention", FIXED["edge_intervention"],
        "--scale_condition",   condition,   # 분석용 태그
    ]


def run_cmd(cmd, dry_run, log_path):
    if dry_run:
        print("    $ " + " ".join(cmd)); return True
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
        lf.write(proc.stdout)
        for ln in proc.stdout.strip().splitlines()[-4:]:
            print(f"      {ln}")
        if proc.returncode != 0:
            print(f"    [WARN] 실패 — {log_path}"); return False
    return True


def analyze(output_dir: str, datasets: list):
    all_dfs = []
    for cond in CONDITIONS:
        fpath = f"outputs/exp_scale_{cond}.csv"
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath)
        df["scale_condition"] = cond
        all_dfs.append(df)

    if not all_dfs:
        print("[WARN] 결과 파일 없음"); return

    df_all = pd.concat(all_dfs, ignore_index=True)
    DNAME = {"pokec_z":"Pokec-Z","german":"German","nba":"NBA",
             "credit":"Credit","recidivism":"Recidivism"}

    print(f"\n{'='*70}")
    print(f"{'Dataset':<18} {'Condition':<22} {'AUC':>7} {'ΔDP':>7} {'ΔEO':>7}")
    print(f"{'='*70}")
    for ds in datasets:
        sub = df_all[df_all["dataset"]==ds]
        for cond in CONDITIONS:
            row = sub[sub["scale_condition"]==cond]
            if row.empty: continue
            r = row.iloc[0]
            print(f"{DNAME.get(ds,ds):<18} {cond:<22} "
                  f"{r.get('roc_auc_mean',0):>7.4f} "
                  f"{r.get('dp_mean',0):>7.4f} "
                  f"{r.get('eo_mean',0):>7.4f}")
        print()

    # with_calibration 대비 개선량
    print("[with_calibration 대비 ΔDP+ΔEO 개선량]")
    df_all["fair"] = df_all["dp_mean"] + df_all["eo_mean"]
    for ds in datasets:
        sub = df_all[df_all["dataset"]==ds]
        base = sub[sub["scale_condition"]=="with_calibration"]["fair"].values
        if len(base) == 0: continue
        for cond in ["warmup_only","no_calibration"]:
            comp = sub[sub["scale_condition"]==cond]["fair"].values
            if len(comp) == 0: continue
            delta = comp[0] - base[0]
            sign  = "+" if delta > 0 else ""
            print(f"  {DNAME.get(ds,ds):<18} vs {cond:<22}: {sign}{delta:.4f}")

    csv_path = os.path.join(output_dir, "scale_calibration_stats.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"\n[Saved] {csv_path}")


def write_plot_script(output_dir: str, datasets: list):
    DNAME = {"pokec_z":"Pokec-Z","german":"German","nba":"NBA",
             "credit":"Credit","recidivism":"Recidivism"}
    cond_labels = {
        "with_calibration": "Periodic recal.",
        "warmup_only"     : "Warmup only",
        "no_calibration"  : "No calibration",
    }
    code = f'''"""
scale_calibration_plot.py — Scale Calibration 효과 시각화
실행: python {output_dir}/scale_calibration_plot.py
"""
import os, pandas as pd, numpy as np, matplotlib.pyplot as plt

df = pd.read_csv(os.path.join("{output_dir}", "scale_calibration_stats.csv"))
df["fair"] = df["dp_mean"] + df["eo_mean"]
DATASETS = {datasets}
DNAME = {DNAME}
COND_LABELS = {cond_labels}
CONDS = list(COND_LABELS.keys())
COLORS = ["#2563EB","#F59E0B","#DC2626"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# (1) ΔDP+ΔEO bar chart
ax = axes[0]
x = np.arange(len(DATASETS))
w = 0.25
for i, (cond, color) in enumerate(zip(CONDS, COLORS)):
    vals = []
    for ds in DATASETS:
        sub = df[(df["dataset"]==ds) & (df["scale_condition"]==cond)]
        vals.append(sub["fair"].values[0] if not sub.empty else 0)
    ax.bar(x + (i-1)*w, vals, w, label=COND_LABELS[cond], color=color, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([DNAME.get(d,d) for d in DATASETS], rotation=25, ha="right", fontsize=9)
ax.set_ylabel("ΔDP + ΔEO (↓)")
ax.set_title("Scale calibration effect on fairness")
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.3)

# (2) AUC bar chart
ax = axes[1]
for i, (cond, color) in enumerate(zip(CONDS, COLORS)):
    vals = []
    for ds in DATASETS:
        sub = df[(df["dataset"]==ds) & (df["scale_condition"]==cond)]
        vals.append(sub["roc_auc_mean"].values[0] if not sub.empty else 0)
    ax.bar(x + (i-1)*w, vals, w, label=COND_LABELS[cond], color=color, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([DNAME.get(d,d) for d in DATASETS], rotation=25, ha="right", fontsize=9)
ax.set_ylabel("AUC (↑)")
ax.set_title("Scale calibration effect on accuracy")
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.3)

plt.suptitle("FairGate Scale Calibration Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
out = os.path.join("{output_dir}", "scale_calibration_plot.pdf")
plt.savefig(out, bbox_inches="tight", dpi=150)
print(f"[Saved] {{out}}")
plt.show()
'''
    path = os.path.join(output_dir, "scale_calibration_plot.py")
    with open(path, "w") as f:
        f.write(code)
    print(f"[Saved] {path}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--datasets",   nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS.keys()),
                   choices=list(CONDITIONS.keys()))
    p.add_argument("--output_dir", type=str, default="outputs/analysis")
    p.add_argument("--log_dir",    type=str, default="logs/scale_calibration")
    p.add_argument("--dry_run",    action="store_true")
    p.add_argument("--analyze_only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    total = len(args.datasets) * len(args.conditions)

    print(f"\n{'='*65}")
    print(f"Scale Calibration Effect Analysis")
    print(f"  conditions : {args.conditions}")
    print(f"  datasets   : {args.datasets}")
    print(f"  total      : {total} runs")
    print(f"{'='*65}")

    if not args.analyze_only:
        step = 0
        for cond in args.conditions:
            output_file = f"exp_scale_{cond}.csv"
            print(f"\n{'─'*65}")
            print(f"  [Condition: {cond}]  {CONDITIONS[cond]['desc']}")
            print(f"{'─'*65}")
            for ds in args.datasets:
                step += 1
                cmd = build_cmd(ds, cond, output_file)
                log_path = os.path.join(args.log_dir, cond, f"{ds}.log")
                print(f"  [{step:2d}/{total}] {ds:<14}  {cond}")
                run_cmd(cmd, args.dry_run, log_path)

    if not args.dry_run:
        analyze(args.output_dir, args.datasets)
        write_plot_script(args.output_dir, args.datasets)


if __name__ == "__main__":
    main()
