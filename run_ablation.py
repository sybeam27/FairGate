"""
run_ablation.py — FairGate Ablation Study

두 가지 ablation 방식을 지원한다.

[순차 추가] --mode sequential
    A0: GCN only / A1: +Lstruct / A2: +Lrep / A3: +Lout / A4: +FIW(bnd) / A5: Full

[제거 ablation] --mode removal  ← 추가
    A5에서 구성요소를 하나씩 제거 → 각 구성요소의 필요성 검증
    A5: Full FairGate (기준)
    R1: w/o L_struct   (rep + out + FIW)
    R2: w/o L_rep      (struct + out + FIW)
    R3: w/o L_out      (struct + rep + FIW)
    R4: w/o FIW        (3-level loss, uniform FIW)
    A0: GCN only       (하한선)

실행:
    python run_ablation.py --mode removal
    python run_ablation.py --mode both
    python run_ablation.py --mode removal --datasets pokec_z german nba credit
    python run_ablation.py --mode removal --dry_run
"""

import os, sys, argparse, subprocess
import pandas as pd

DEVICE = "cuda:1"

ALL_DATASETS = [
    "pokec_z", "pokec_z_g", "pokec_n", "pokec_n_g",
    "german", "credit", "recidivism", "nba", "income",
]


FAIRGATE_CONFIGS = {
    # ── Pokec 계열 ──────────────────────────────────────────────────────────
    "pokec_z":    dict(lambda_fair=0.10, sbrs_quantile=0.9, struct_drop=0.5, warm_up=400),
    "pokec_z_g":  dict(lambda_fair=0.20, sbrs_quantile=0.6, struct_drop=0.5, warm_up=100),
    "pokec_n":    dict(lambda_fair=0.15, sbrs_quantile=0.5, struct_drop=0.5, warm_up=400),
    "pokec_n_g":  dict(lambda_fair=0.01, sbrs_quantile=0.8, struct_drop=0.5, warm_up=400),
    # ── 소규모 그래프 ────────────────────────────────────────────────────────
    "credit":     dict(lambda_fair=0.20, sbrs_quantile=0.5, struct_drop=0.7, warm_up=200),
    "recidivism": dict(lambda_fair=0.10, sbrs_quantile=0.9, struct_drop=0.2, warm_up=100),
    "income":     dict(lambda_fair=0.20, sbrs_quantile=0.5, struct_drop=0.7, warm_up=200),
    "german":     dict(lambda_fair=0.20, sbrs_quantile=0.9, struct_drop=0.2, warm_up=100),
    "nba":        dict(lambda_fair=0.40, sbrs_quantile=0.5, struct_drop=0.3, warm_up=200),
}

FIXED = dict(
    backbone="GCN", hidden_dim=128, dropout=0.5, sgc_k=2,
    lr=1e-3, weight_decay=1e-5, epochs=500, patience=501,
    runs=5, seed=27, fips_lam=1.0, mmd_alpha=0.3,
    dp_eo_ratio=0.3, uncertainty_type="entropy",
    ramp_epochs=0, recal_interval=200,
    alpha_beta_mode="variance", edge_intervention="drop",
)

# ── 순차 추가 ──────────────────────────────────────────────────────────────────
SEQUENTIAL_STAGES = {
    "A0": dict(desc="GCN only",                 lambda_fair=0.0, fips_lam=0.0, ablation_mode="none"),
    "A1": dict(desc="+L_struct",                                 fips_lam=0.0, ablation_mode="struct_only"),
    "A2": dict(desc="+L_struct+L_rep",                          fips_lam=0.0, ablation_mode="struct_rep"),
    "A3": dict(desc="+full 3-level (uniform FIW)",              fips_lam=0.0, ablation_mode="full_loss"),
    "A4": dict(desc="+FIW boundary (fips=0)",                   fips_lam=0.0, ablation_mode="full_loss"),
    "A5": dict(desc="Full FairGate",                                           ablation_mode="full_loss"),
}

# ── 제거 ablation ──────────────────────────────────────────────────────────────
# model.py에 추가된 mode:
#   struct_out = L_struct + L_out (rep 제거)
#   rep_out    = L_rep   + L_out (struct 제거)
REMOVAL_STAGES = {
    "A5": dict(desc="Full FairGate (기준)",                                     ablation_mode="full_loss"),
    "R1": dict(desc="w/o L_struct  (rep+out+FIW)",                             ablation_mode="rep_out"),
    "R2": dict(desc="w/o L_rep     (struct+out+FIW)",                          ablation_mode="struct_out"),
    "R3": dict(desc="w/o L_out     (struct+rep+FIW)",                          ablation_mode="struct_rep"),
    "R4": dict(desc="w/o FIW       (full loss, uniform FIW)", fips_lam=0.0,   ablation_mode="full_loss"),
    "A0": dict(desc="GCN only (하한선)",  lambda_fair=0.0, fips_lam=0.0,       ablation_mode="none"),
}


def build_cmd(dataset, stage, stage_cfg, output_file):
    cfg = FAIRGATE_CONFIGS[dataset].copy()
    lf  = stage_cfg.get("lambda_fair", cfg["lambda_fair"])
    fl  = stage_cfg.get("fips_lam",    FIXED["fips_lam"])
    am  = stage_cfg.get("ablation_mode", "full_loss")

    return [
        sys.executable, "-m", "utils.train",
        "--dataset",           dataset,
        "--backbone",          FIXED["backbone"],
        "--output_file",       output_file,
        "--run_name",          f"ablation_{stage}_{dataset}",
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
        "--lambda_fair",       str(lf),
        "--sbrs_quantile",     str(cfg["sbrs_quantile"]),
        "--struct_drop",       str(cfg["struct_drop"]),
        "--warm_up",           str(cfg["warm_up"]),
        "--fips_lam",          str(fl),
        "--mmd_alpha",         str(FIXED["mmd_alpha"]),
        "--dp_eo_ratio",       str(FIXED["dp_eo_ratio"]),
        "--uncertainty_type",  FIXED["uncertainty_type"],
        "--ramp_epochs",       str(FIXED["ramp_epochs"]),
        "--recal_interval",    str(FIXED["recal_interval"]),
        "--alpha_beta_mode",   FIXED["alpha_beta_mode"],
        "--edge_intervention", FIXED["edge_intervention"],
        "--ablation_mode",     am,
        "--ablation_stage",    stage,
    ]


def run_cmd(cmd, dry_run, log_path):
    if dry_run:
        print("    $ " + " ".join(cmd)); return True
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
        lf.write(proc.stdout)
        for ln in proc.stdout.strip().splitlines()[-5:]:
            print(f"      {ln}")
        if proc.returncode != 0:
            print(f"    [WARN] 실패 — 로그: {log_path}"); return False
    return True


def analyze_removal(output_file, output_dir):
    """A5 기준 각 구성요소 제거 시 DP+EO 변화 분석 + LaTeX 저장"""
    if not os.path.exists(output_file):
        print("[WARN] 결과 파일 없음"); return

    df = pd.read_csv(output_file)
    df = df[df["ablation_stage"].isin(REMOVAL_STAGES.keys())]
    if df.empty:
        print("[WARN] removal stage 결과 없음"); return

    DNAME = {"pokec_z":"Pokec-Z","pokec_z_g":"Pokec-Z (g)","pokec_n":"Pokec-N",
             "pokec_n_g":"Pokec-N (g)","german":"German","credit":"Credit",
             "recidivism":"Recidivism","nba":"NBA","income":"Income"}
    stages = ["A5","R1","R2","R3","R4","A0"]
    datasets = [d for d in ALL_DATASETS if d in df["dataset"].unique()]

    # 콘솔 출력
    print(f"\n{'='*80}")
    print("제거 Ablation: A5 대비 DP+EO 변화 (↑ = 악화, 클수록 해당 구성요소 중요)")
    print(f"{'='*80}")
    header = f"{'Dataset':<14}" + "".join(f"  {s:>10}" for s in stages)
    print(header)
    print("-"*80)

    for ds in datasets:
        sub = df[df["dataset"]==ds]
        vals = {}
        for s in stages:
            sr = sub[sub["ablation_stage"]==s]
            if not sr.empty:
                vals[s] = round(sr.iloc[0]["dp_mean"] + sr.iloc[0]["eo_mean"], 4)
        base = vals.get("A5")
        row  = f"{DNAME.get(ds,ds):<14}"
        for s in stages:
            if s not in vals:
                row += f"  {'—':>10}"
            elif s == "A5" or base is None:
                row += f"  {vals[s]:>10.4f}"
            else:
                delta = vals[s] - base
                mark  = "↑" if delta > 0.005 else ""
                row  += f"  {vals[s]:>7.4f}{mark:>3}"
        print(row)

    # LaTeX
    stage_labels = {
        "A5": r"\textbf{Full}",
        "R1": r"w/o $\mathcal{L}_{\mathrm{s}}$",
        "R2": r"w/o $\mathcal{L}_{\mathrm{r}}$",
        "R3": r"w/o $\mathcal{L}_{\mathrm{o}}$",
        "R4": r"w/o FIW",
        "A0": r"GCN",
    }
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{제거 ablation 결과 ($\Delta\mathrm{DP}{+}\Delta\mathrm{EO}$). "
        r"Full FairGate(A5)에서 구성요소를 하나씩 제거했을 때의 공정성 변화. "
        r"$\uparrow$: A5 대비 공정성 악화.}",
        r"\label{tab:ablation_removal}",
        r"\setlength{\tabcolsep}{5pt}", r"\renewcommand{\arraystretch}{1.2}",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{l" + "r"*len(stages) + "}",
        r"\toprule",
        r"\textbf{Dataset} & " + " & ".join(stage_labels[s] for s in stages) + r" \\",
        r"\midrule",
    ]
    for ds in datasets:
        sub = df[df["dataset"]==ds]
        vals = {}
        for s in stages:
            sr = sub[sub["ablation_stage"]==s]
            if not sr.empty:
                vals[s] = sr.iloc[0]["dp_mean"] + sr.iloc[0]["eo_mean"]
        base = vals.get("A5")
        row  = DNAME.get(ds, ds)
        for s in stages:
            if s not in vals:
                row += " & —"
            elif s == "A5":
                row += f" & \\textbf{{{vals[s]:.4f}}}"
            else:
                delta = (vals[s] - base) if base else 0
                sup   = r"$^{\uparrow}$" if delta > 0.005 else ""
                row  += f" & {vals[s]:.4f}{sup}"
        lines.append(row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]

    tex_path = os.path.join(output_dir, "ablation_removal.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[Saved] {tex_path}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--datasets",    nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    p.add_argument("--mode",        type=str,  default="both",
                   choices=["sequential","removal","both"])
    p.add_argument("--output_file", type=str,  default="analysis/exp_ablation.csv")
    p.add_argument("--output_dir",  type=str,  default="outputs/analysis")
    p.add_argument("--log_dir",     type=str,  default="logs/ablation")
    p.add_argument("--dry_run",     action="store_true")
    p.add_argument("--analyze_only",action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 실행할 stage 결정 (both면 A5 중복 제거)
    if args.mode == "sequential":
        stages = SEQUENTIAL_STAGES
    elif args.mode == "removal":
        stages = REMOVAL_STAGES
    else:
        merged = {**SEQUENTIAL_STAGES}
        for k, v in REMOVAL_STAGES.items():
            if k not in merged:
                merged[k] = v
        stages = merged

    total = len(args.datasets) * len(stages)
    print(f"\n{'='*65}")
    print(f"FairGate Ablation Study  [mode={args.mode}]")
    print(f"  stages   : {list(stages.keys())}")
    print(f"  datasets : {args.datasets}")
    print(f"  total    : {total} runs")
    print(f"{'='*65}")

    if not args.analyze_only:
        step = 0
        for stage, scfg in stages.items():
            print(f"\n{'─'*65}")
            print(f"  [Stage {stage}] {scfg['desc']}")
            print(f"{'─'*65}")
            for ds in args.datasets:
                step += 1
                cmd = build_cmd(ds, stage, scfg, args.output_file)
                log = os.path.join(args.log_dir, stage, f"{ds}.log")
                print(f"  [{step:2d}/{total}] {ds:<14}  stage={stage}")
                run_cmd(cmd, args.dry_run, log)

    if not args.dry_run:
        print("\n[분석 실행]")
        if args.mode in ("removal","both"):
            analyze_removal(args.output_file, args.output_dir)

    print(f"\n완료. 결과: {args.output_file}")


if __name__ == "__main__":
    main()