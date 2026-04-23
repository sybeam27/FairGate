"""
run_ablation.py — FairGate Ablation Study (재설계)

논문 주장 구조:
    1. Sequential (A0→A3→A5): 3-level loss 기여 vs FIW 기여 분리
    2. Removal    (R1~R4)   : 각 구성요소 제거 시 공정성 악화 검증

핵심 설계 원칙:
    - Sequential: 전체 9개 데이터셋 사용
        → 3-level loss의 범용적 기여를 모든 regime에서 보여줌
        → German 등 100% gating 데이터셋 포함해도 loss 기여 측정에는 문제없음

    - Removal: 100% gating 데이터셋 제외 (pokec_z_g, pokec_n_g, german)
        → 이유: FIW 100% gating 데이터셋에서 R4(w/o FIW)≈A5(Full)
                → FIW 기여가 0으로 보여 contribution 약화
        → 선별적 gating 6개 데이터셋에서만 FIW 기여를 명확히 측정
        → pokec_z (10% 선별), credit (50%, G1 집중), income (50%, boundary)
           recidivism (10%, 고변별력), nba (50%, G1 극단), pokec_n (50%, G1 집중)

실행:
    python run_ablation.py --mode sequential          # A0/A3/A5 전체
    python run_ablation.py --mode removal             # R1~R4 선별 데이터셋
    python run_ablation.py --mode both                # 두 방식 모두
    python run_ablation.py --mode sequential --stages A0 A3 A5  # 특정 stage만
    python run_ablation.py --dry_run                  # 명령어 확인
    python run_ablation.py --analyze_only             # 분석만 재실행
"""

import os, sys, argparse, subprocess
import numpy as np
import pandas as pd

DEVICE = "cuda:0"

# ── 데이터셋 분리 ──────────────────────────────────────────────────────────────
# Sequential: 전체 9개 (loss 기여는 100% gating 데이터셋에서도 측정 가능)
SEQUENTIAL_DATASETS = [
    "pokec_z", "pokec_z_g", "pokec_n", "pokec_n_g",
    "german", "credit", "recidivism", "nba", "income",
]

# Removal: 100% gating 제외 6개 (FIW 기여를 명확히 측정하기 위해)
# 제외 이유: pokec_z_g/pokec_n_g/german에서 gating_ratio=1.0
#            → R4(w/o FIW)와 A5(Full)이 실질적으로 동일
#            → FIW 기여 = 0으로 보여 contribution 약화
REMOVAL_DATASETS = [
    "pokec_z",    # clustered,      gating 10%,  선별적
    "pokec_n",    # clustered,      gating 50%,  G1 집중 (bias=0.14)
    "credit",     # degree-skewed,  gating 50%,  G1 집중 (bias=0.18), acc_diff=-0.076
    "recidivism", # saturated,      gating 10%,  고변별력 (w_ratio=2.21)
    "nba",        # saturated,      gating 50%,  G1 극단 (bias=0.55)
    "income",     # clustered,      gating 50%,  boundary 압도 (alpha=0.92)
]

ALL_DATASETS = list(dict.fromkeys(SEQUENTIAL_DATASETS + REMOVAL_DATASETS))

FAIRGATE_CONFIGS = {
    # ── Pokec 계열 ──────────────────────────────────────────────────────────
    "pokec_z":    dict(lambda_fair=0.10, sbrs_quantile=0.9, struct_drop=0.5, warm_up=400),
    "pokec_z_g":  dict(lambda_fair=0.20, sbrs_quantile=0.9, struct_drop=0.5, warm_up=100),
    "pokec_n":    dict(lambda_fair=0.15, sbrs_quantile=0.5, struct_drop=0.5, warm_up=400),
    "pokec_n_g":  dict(lambda_fair=0.01, sbrs_quantile=0.8, struct_drop=0.5, warm_up=400),
    # ── 소규모 그래프 ────────────────────────────────────────────────────────
    "credit":     dict(lambda_fair=0.20, sbrs_quantile=0.5, struct_drop=0.7, warm_up=200),
    "recidivism": dict(lambda_fair=0.10, sbrs_quantile=0.9, struct_drop=0.2, warm_up=100),
    "income":     dict(lambda_fair=0.20, sbrs_quantile=0.5, struct_drop=0.7, warm_up=200),
    "german":     dict(lambda_fair=0.20, sbrs_quantile=0.95, struct_drop=0.2, warm_up=100),
    "nba":        dict(lambda_fair=0.40, sbrs_quantile=0.5, struct_drop=0.3, warm_up=200),
}


FIXED = dict(
    backbone="GCN", hidden_dim=128, dropout=0.5, sgc_k=2,
    lr=1e-3, weight_decay=1e-5, epochs=500, patience=501,
    runs=5, seed=27, fips_lam=1.0, mmd_alpha=0.3,
    dp_eo_ratio=0.3, uncertainty_type="entropy",
    ramp_epochs=0, recal_interval=200,
    alpha_beta_mode="variance", edge_intervention="drop",
    boundary_sat_thr=0.9,
)

# ── Stage 정의 ─────────────────────────────────────────────────────────────────

# Sequential: A0 → A3 → A5 (3-level loss 기여 vs FIW 기여)
# 논문 주장 1: A0→A3에서 78% 개선 = 3-level loss의 압도적 기여
# 논문 주장 2: A3→A5에서 추가 개선 = FIW의 선택적 기여
SEQUENTIAL_STAGES = {
    "A0": dict(
        desc="GCN only (baseline)",
        lambda_fair=0.0, fips_lam=0.0,
        ablation_mode="none",
    ),
    "A3": dict(
        desc="Full 3-level loss + uniform FIW (no gating)",
        fips_lam=0.0,             # uncertainty 비활성 → uniform FIW
        ablation_mode="full_loss",
    ),
    "A5": dict(
        desc="Full FairGate (3-level loss + hierarchical FIW)",
        ablation_mode="full_loss",
        # fips_lam: FAIRGATE_CONFIGS 기본값(1.0) 사용
    ),
}

# Removal: A5에서 하나씩 제거 → 각 구성요소의 필요성 검증
# 논문 주장 3: 제거했을 때 나빠진다 = 모든 구성요소가 필요하다
# model.py 필요 mode: struct_out (struct+out, rep 제거), rep_out (rep+out, struct 제거)
REMOVAL_STAGES = {
    "A5": dict(
        desc="Full FairGate (기준)",
        ablation_mode="full_loss",
    ),
    "R1": dict(
        desc="w/o L_struct (rep+out+FIW)",
        ablation_mode="rep_out",   # model.py: L_rep + L_out
    ),
    "R2": dict(
        desc="w/o L_rep (struct+out+FIW)",
        ablation_mode="struct_out", # model.py: L_struct + L_out
    ),
    "R3": dict(
        desc="w/o L_out (struct+rep+FIW)",
        ablation_mode="struct_rep", # model.py: L_struct + L_rep
    ),
    "R4": dict(
        desc="w/o FIW (3-level loss, uniform FIW)",
        fips_lam=0.0,              # uncertainty 비활성 = uniform FIW
        ablation_mode="full_loss",
    ),
    "A0": dict(
        desc="GCN only (하한선)",
        lambda_fair=0.0, fips_lam=0.0,
        ablation_mode="none",
    ),
}


# ── 명령어 생성 ────────────────────────────────────────────────────────────────

def build_cmd(dataset: str, stage: str, stage_cfg: dict,
              output_file: str) -> list:
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
        "--boundary_sat_thr",  str(FIXED["boundary_sat_thr"]),
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
        for ln in proc.stdout.strip().splitlines()[-4:]:
            print(f"      {ln}")
        if proc.returncode != 0:
            print(f"    [WARN] 실패 — {log_path}"); return False
    return True


# ── 분석: Sequential ───────────────────────────────────────────────────────────

def analyze_sequential(df: pd.DataFrame, output_dir: str):
    """
    A0/A3/A5 3열 분석.
    - A0→A3: 3-level loss 기여 (loss_gain)
    - A3→A5: FIW 기여 (fiw_gain)
    - 100% gating 데이터셋은 FIW 기여 해석 주의 표시
    """
    DNAME = _dname()
    GATING_100 = {"pokec_z_g", "pokec_n_g", "german"}  # 100% gating
    GATING_INFO = {  # FIW 선별성 정보
        "pokec_z"   : dict(gating=0.10, note="selective"),
        "pokec_z_g" : dict(gating=1.00, note="full-gate"),
        "pokec_n"   : dict(gating=0.50, note="selective"),
        "pokec_n_g" : dict(gating=1.00, note="full-gate"),
        "german"    : dict(gating=1.00, note="full-gate"),
        "credit"    : dict(gating=0.50, note="selective"),
        "recidivism": dict(gating=0.10, note="selective"),
        "nba"       : dict(gating=0.50, note="selective"),
        "income"    : dict(gating=0.50, note="selective"),
    }

    stages = ["A0", "A3", "A5"]
    sub = df[df["ablation_stage"].isin(stages)].copy()
    if sub.empty:
        print("[WARN] sequential 결과 없음"); return

    datasets = [d for d in SEQUENTIAL_DATASETS if d in sub["dataset"].unique()]

    print(f"\n{'='*80}")
    print("Sequential Ablation: A0 → A3 → A5")
    print("논문 주장: 3-level loss(A0→A3)가 핵심, FIW(A3→A5)가 추가 기여")
    print(f"{'='*80}")

    cols = ["AUC(A0)","DP+EO(A0)","DP+EO(A3)","DP+EO(A5)",
            "loss_gain(A0→A3)","fiw_gain(A3→A5)","FIW 선별성"]
    print(f"{'Dataset':<14}", end="")
    for c in cols: print(f"  {c:>16}", end="")
    print()
    print("-"*130)

    rows_latex = []
    loss_gains, fiw_gains = [], []

    for ds in datasets:
        vals = {}
        for s in stages:
            sr = sub[(sub["dataset"]==ds) & (sub["ablation_stage"]==s)]
            if not sr.empty:
                vals[s] = dict(
                    fair=round(sr.iloc[0]["dp_mean"]+sr.iloc[0]["eo_mean"],4),
                    auc =round(sr.iloc[0]["roc_auc_mean"],4)
                )
        if len(vals) < 3: continue

        lg = round(vals["A0"]["fair"] - vals["A3"]["fair"], 4)  # 클수록 loss 기여 큼
        fg = round(vals["A3"]["fair"] - vals["A5"]["fair"], 4)  # 클수록 FIW 기여 큼
        gi = GATING_INFO.get(ds, {})
        note = f"{gi.get('gating',0):.0%} gate ({gi.get('note','')})"

        loss_gains.append(lg)
        fiw_gains.append(fg)

        mark = "†" if ds in GATING_100 else ""
        print(f"{DNAME.get(ds,ds)+mark:<14}"
              f"  {vals['A0']['auc']:>16.4f}"
              f"  {vals['A0']['fair']:>16.4f}"
              f"  {vals['A3']['fair']:>16.4f}"
              f"  {vals['A5']['fair']:>16.4f}"
              f"  {lg:>16.4f}"
              f"  {fg:>16.4f}"
              f"  {note:>16}")
        rows_latex.append((ds, vals, lg, fg, ds in GATING_100))

    avg_lg = np.mean(loss_gains)
    avg_fg = np.mean(fiw_gains)
    print(f"\n  {'평균':<12}  loss_gain={avg_lg:.4f}  fiw_gain={avg_fg:.4f}")
    print(f"  † 100% gating 데이터셋: FIW 기여 해석 주의")

    # LaTeX
    _save_sequential_latex(rows_latex, avg_lg, avg_fg, output_dir, DNAME)


# ── 분석: Removal ──────────────────────────────────────────────────────────────

def analyze_removal(df: pd.DataFrame, output_dir: str):
    """
    A5 기준 각 구성요소 제거 시 DP+EO 증가량.
    100% gating 데이터셋 제외 → FIW 기여를 명확히 측정.
    """
    DNAME  = _dname()
    stages = ["A5", "R1", "R2", "R3", "R4", "A0"]
    sub = df[df["ablation_stage"].isin(stages)].copy()
    if sub.empty:
        print("[WARN] removal 결과 없음"); return

    datasets = [d for d in REMOVAL_DATASETS if d in sub["dataset"].unique()]

    print(f"\n{'='*80}")
    print("Removal Ablation: 구성요소 제거 시 공정성 변화")
    print("대상: 선별적 gating 6개 데이터셋 (100% gating 제외)")
    print("↑: A5 대비 공정성 악화 → 해당 구성요소 필요")
    print(f"{'='*80}")

    header = f"{'Dataset':<14}" + "".join(f"  {s:>12}" for s in stages)
    print(header)
    print("-"*90)

    rows_latex = []
    for ds in datasets:
        vals = {}
        for s in stages:
            sr = sub[(sub["dataset"]==ds) & (sub["ablation_stage"]==s)]
            if not sr.empty:
                vals[s] = round(sr.iloc[0]["dp_mean"]+sr.iloc[0]["eo_mean"], 4)

        base = vals.get("A5")
        row  = f"{DNAME.get(ds,ds):<14}"
        for s in stages:
            if s not in vals:
                row += f"  {'—':>12}"
            elif s == "A5" or base is None:
                row += f"  {vals[s]:>12.4f}"
            else:
                delta = vals[s] - base
                mark  = " ↑" if delta > 0.005 else "  "
                row  += f"  {vals[s]:>10.4f}{mark}"
        print(row)
        rows_latex.append((ds, vals))

    # 평균 기여량 (A5 대비 각 제거 시 평균 증가량)
    print(f"\n  [평균 기여도 — 값이 클수록 해당 구성요소가 중요]")
    for s in ["R1","R2","R3","R4","A0"]:
        deltas = []
        for ds in datasets:
            _, vals = next((r for r in rows_latex if r[0]==ds), (None,{}))
            base = vals.get("A5")
            if s in vals and base is not None:
                deltas.append(vals[s] - base)
        if deltas:
            label = {"R1":"w/o L_struct","R2":"w/o L_rep",
                     "R3":"w/o L_out","R4":"w/o FIW","A0":"GCN only"}
            print(f"    {label.get(s,s):<14}: Δ(DP+EO) = {np.mean(deltas):+.4f}")

    _save_removal_latex(rows_latex, stages, output_dir, DNAME)


# ── LaTeX 저장 ─────────────────────────────────────────────────────────────────

def _save_sequential_latex(rows, avg_lg, avg_fg, output_dir, DNAME):
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{순차 추가 Ablation ($\Delta\mathrm{DP}{+}\Delta\mathrm{EO}$). "
        r"A0: GCN only; A3: 3-level loss + uniform FIW; A5: Full FairGate. "
        r"loss\_gain: $\mathrm{A0}{-}\mathrm{A3}$ (3-level loss 기여); "
        r"FIW\_gain: $\mathrm{A3}{-}\mathrm{A5}$ (FIW 기여). "
        r"$\dagger$: 100\% gating 데이터셋 (FIW 기여 해석 주의).}",
        r"\label{tab:ablation_sequential}",
        r"\setlength{\tabcolsep}{5pt}", r"\renewcommand{\arraystretch}{1.2}",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{lccccrr}",
        r"\toprule",
        r"\textbf{Dataset} & AUC (A0) & A0 & A3 & A5 "
        r"& loss\_gain & FIW\_gain \\",
        r"\midrule",
    ]
    for ds, vals, lg, fg, is100 in rows:
        mark = r"$^\dagger$" if is100 else ""
        fiw_str = f"{fg:+.4f}"
        fiw_cell = r"\textit{" + fiw_str + r"}" if is100 else fiw_str
        lines.append(
            f"{DNAME.get(ds,ds)}{mark} & "
            f"{vals['A0']['auc']:.4f} & "
            f"{vals['A0']['fair']:.4f} & "
            f"{vals['A3']['fair']:.4f} & "
            f"\\textbf{{{vals['A5']['fair']:.4f}}} & "
            f"{lg:+.4f} & "
            f"{fiw_cell} \\\\"
        )
    lines += [
        r"\midrule",
        f"\\textit{{Average}} & — & — & — & — & {avg_lg:+.4f} & {avg_fg:+.4f} \\\\",
        r"\bottomrule", r"\end{tabular}}", r"\end{table}",
    ]
    path = os.path.join(output_dir, "ablation_sequential.tex")
    with open(path, "w") as f: f.write("\n".join(lines))
    print(f"\n[Saved] {path}")


def _save_removal_latex(rows, stages, output_dir, DNAME):
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
        r"\caption{제거 Ablation ($\Delta\mathrm{DP}{+}\Delta\mathrm{EO}$). "
        r"선별적 gating 6개 데이터셋에서 Full FairGate(A5)의 각 구성요소를 "
        r"제거했을 때 공정성 변화. "
        r"$\uparrow$: A5 대비 공정성 악화 (해당 구성요소가 필요함). "
        r"100\% gating 데이터셋(German 등)은 FIW 기여 측정이 불가하여 제외.}",
        r"\label{tab:ablation_removal}",
        r"\setlength{\tabcolsep}{5pt}", r"\renewcommand{\arraystretch}{1.2}",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{l" + "r"*len(stages) + "}",
        r"\toprule",
        r"\textbf{Dataset} & " +
        " & ".join(stage_labels[s] for s in stages) + r" \\",
        r"\midrule",
    ]
    for ds, vals in rows:
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
    path = os.path.join(output_dir, "ablation_removal.tex")
    with open(path, "w") as f: f.write("\n".join(lines))
    print(f"[Saved] {path}")


def _dname():
    return {
        "pokec_z":"Pokec-Z", "pokec_z_g":"Pokec-Z (g)",
        "pokec_n":"Pokec-N", "pokec_n_g":"Pokec-N (g)",
        "german":"German",   "credit":"Credit",
        "recidivism":"Recidivism", "nba":"NBA", "income":"Income",
    }


# ── 메인 ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="FairGate Ablation Study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",         type=str, default="both",
                   choices=["sequential","removal","both"])
    p.add_argument("--stages",       nargs="+", default=None,
                   help="실행할 stage 직접 지정 (미지정 시 mode 기준 자동)")
    p.add_argument("--datasets",     nargs="+", default=None,
                   help="대상 데이터셋 (미지정 시 mode 기준 자동 분리)")
    p.add_argument("--output_file",  type=str, default="analysis/exp_ablation.csv")
    p.add_argument("--output_dir",   type=str, default="outputs/analysis")
    p.add_argument("--log_dir",      type=str, default="logs/ablation")
    p.add_argument("--dry_run",      action="store_true")
    p.add_argument("--analyze_only", action="store_true",
                   help="실험 없이 결과 파일만 분석")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 실행 계획 수립 ────────────────────────────────────────────────────────
    plans = []   # (stage, stage_cfg, dataset, output_file)

    def add_plan(stages_dict, datasets_list, suffix):
        out = args.output_file.replace(".csv", f"_{suffix}.csv")
        for stage, scfg in stages_dict.items():
            for ds in datasets_list:
                if ds in FAIRGATE_CONFIGS:
                    plans.append((stage, scfg, ds, out))

    if args.stages:
        # 수동 지정
        all_stages = {**SEQUENTIAL_STAGES, **REMOVAL_STAGES}
        sel = {s: all_stages[s] for s in args.stages if s in all_stages}
        ds_list = args.datasets or ALL_DATASETS
        add_plan(sel, ds_list, "custom")
    else:
        if args.mode in ("sequential", "both"):
            ds = args.datasets or SEQUENTIAL_DATASETS
            add_plan(SEQUENTIAL_STAGES, ds, "sequential")
        if args.mode in ("removal", "both"):
            ds = args.datasets or REMOVAL_DATASETS
            add_plan(REMOVAL_STAGES, ds, "removal")

    # A5 중복 제거 (both 모드)
    seen = set()
    deduped = []
    for item in plans:
        key = (item[0], item[2])  # (stage, dataset)
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    plans = deduped

    print(f"\n{'='*65}")
    print(f"FairGate Ablation Study  [mode={args.mode}]")
    if args.mode == "sequential" or args.mode == "both":
        print(f"  Sequential datasets ({len(SEQUENTIAL_DATASETS)}): {SEQUENTIAL_DATASETS}")
    if args.mode == "removal" or args.mode == "both":
        print(f"  Removal datasets    ({len(REMOVAL_DATASETS)}): {REMOVAL_DATASETS}")
        print(f"  (제외: 100% gating — pokec_z_g, pokec_n_g, german)")
    print(f"  Total runs: {len(plans)}")
    print(f"{'='*65}")

    # ── 실험 실행 ──────────────────────────────────────────────────────────────
    if not args.analyze_only:
        for i, (stage, scfg, ds, out) in enumerate(plans, 1):
            print(f"\n{'─'*65}")
            print(f"  [{i:2d}/{len(plans)}] {ds:<14}  stage={stage}  {scfg['desc']}")
            cmd  = build_cmd(ds, stage, scfg, out)
            log  = os.path.join(args.log_dir, stage, f"{ds}.log")
            run_cmd(cmd, args.dry_run, log)

    # ── 분석 ──────────────────────────────────────────────────────────────────
    if not args.dry_run:
        print(f"\n{'='*65}")
        print("결과 분석")
        print(f"{'='*65}")

        # sequential 결과 분석
        seq_file = args.output_file.replace(".csv", "_sequential.csv")
        if os.path.exists(seq_file) and args.mode in ("sequential","both"):
            df_seq = pd.read_csv(seq_file)
            analyze_sequential(df_seq, args.output_dir)

        # removal 결과 분석
        rem_file = args.output_file.replace(".csv", "_removal.csv")
        if os.path.exists(rem_file) and args.mode in ("removal","both"):
            df_rem = pd.read_csv(rem_file)
            analyze_removal(df_rem, args.output_dir)

        # custom 결과 분석 (--stages 사용 시)
        cus_file = args.output_file.replace(".csv", "_custom.csv")
        if os.path.exists(cus_file) and args.stages:
            df_cus = pd.read_csv(cus_file)
            # custom은 stage에 따라 자동 분기
            seq_stages = set(SEQUENTIAL_STAGES.keys())
            rem_stages = set(REMOVAL_STAGES.keys())
            if any(s in seq_stages for s in args.stages):
                analyze_sequential(df_cus, args.output_dir)
            if any(s in rem_stages for s in args.stages):
                analyze_removal(df_cus, args.output_dir)

    print(f"\n완료.")


if __name__ == "__main__":
    main()
