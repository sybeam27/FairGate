"""
run_fiw_ablation.py — FIW Gating 효과성 Ablation

FIW의 핵심 기여를 3가지 비교 실험으로 정량화한다.

=== 설계 원칙 ===
"FIW가 선별하지 않는 것(uniform)"이 아니라
"다른 방식으로 선별하는 것"과 비교해야
FIW 선별 기준의 우월성을 증명할 수 있다.

=== 비교 실험 구조 ===

[Tier 1] Gating 선별 기준 비교 — "무엇으로 고를 것인가"
    F0: No gating       (uniform, fips_lam=0) ← 현재 R4와 동일. 기준선
    F1: Random gating   (동일 비율 랜덤 선택) ← 핵심 비교
    F2: Boundary-only   (alpha=1.0, beta=0.0) ← boundary만
    F3: Degree-only     (alpha=0.0, beta=1.0) ← degree만
    F5: Full FIW        (variance alpha/beta + uncertainty) ← 현재 A5

    기대: F5 < F1 → "FIW 선별이 랜덤보다 효과적"
          F5 < F2, F3 → "혼합 신호가 단일 신호보다 효과적"

[Tier 2] Uncertainty modulation 비교 — "선별 후 어떻게 가중치 줄 것인가"
    F4: Struct-only     (gating O, uncertainty X, fips_lam=0)
    F5: Full FIW        (gating O, uncertainty O, fips_lam=1.0)

    기대: F5 < F4 → "uncertainty modulation이 추가 기여한다"

=== 데이터셋 ===
선별적 gating이 유효한 6개만 사용 (100% gating 제외)
pokec_z, pokec_n, credit, recidivism, nba, income

=== 실행 ===
    python run_fiw_ablation.py                     # 전체
    python run_fiw_ablation.py --tier 1            # Gating 기준만
    python run_fiw_ablation.py --tier 2            # Uncertainty만
    python run_fiw_ablation.py --dry_run           # 확인
    python run_fiw_ablation.py --analyze_only      # 분석만
"""

import os, sys, argparse, subprocess
import numpy as np
import pandas as pd

DEVICE = "cuda:1"

# ── 데이터셋 ──────────────────────────────────────────────────────────────────
# 100% gating 제외: FIW 선별 효과가 측정 가능한 데이터셋만
DATASETS = [
    "pokec_z",     # clustered,      gating 10%,  w_ratio=1.68
    "pokec_n",     # clustered,      gating 50%,  G1 집중 bias=0.14
    "credit",      # degree-skewed,  gating 50%,  acc_diff=-0.076
    "recidivism",  # saturated,      gating 10%,  w_ratio=2.21
    "nba",         # saturated,      gating 50%,  G1 극단 bias=0.55
    "income",      # clustered,      gating 50%,  boundary 압도 alpha=0.92
]

FAIRGATE_CONFIGS = {
    "pokec_z"   : dict(lambda_fair=0.10, sbrs_quantile=0.90, struct_drop=0.5, warm_up=400),
    "pokec_n"   : dict(lambda_fair=0.15, sbrs_quantile=0.50, struct_drop=0.5, warm_up=400),
    "credit"    : dict(lambda_fair=0.20, sbrs_quantile=0.50, struct_drop=0.7, warm_up=200),
    "recidivism": dict(lambda_fair=0.10, sbrs_quantile=0.90, struct_drop=0.2, warm_up=100),
    "nba"       : dict(lambda_fair=0.40, sbrs_quantile=0.50, struct_drop=0.3, warm_up=200),
    "income"    : dict(lambda_fair=0.20, sbrs_quantile=0.50, struct_drop=0.7, warm_up=200),
}

FIXED = dict(
    backbone="GCN", hidden_dim=128, dropout=0.5, sgc_k=2,
    lr=1e-3, weight_decay=1e-5, epochs=500, patience=501,
    runs=5, seed=27, mmd_alpha=0.3,
    dp_eo_ratio=0.3, uncertainty_type="entropy",
    ramp_epochs=0, recal_interval=200,
    edge_intervention="drop", boundary_sat_thr=0.9,
)

# ── Stage 정의 ─────────────────────────────────────────────────────────────────
#
# FIW 작동 방식:
#   gating_signal로 상위 k% 노드 gate
#   gate된 노드에 struct_score * (1 + fips_lam * uncertainty) 가중치 부여
#
# alpha_beta_mode:
#   "variance" : Var(w_bnd)/Var(w_deg) 비율로 alpha/beta 자동 결정 (Full FIW)
#   "uniform"  : alpha=beta=0.5 (경계/degree 동등)
#   "bnd_only" : alpha=1.0, beta=0.0 (boundary 신호만)  ← 추가 필요
#   "deg_only" : alpha=0.0, beta=1.0 (degree 신호만)    ← 추가 필요
#   "random"   : 동일 비율 랜덤 선택                    ← 추가 필요

FIW_STAGES = {
    # ── Tier 1: Gating 선별 기준 비교 ─────────────────────────────────────
    "F0": dict(
        desc="No gating — uniform weight (baseline)",
        tier=1,
        fips_lam=0.0,
        alpha_beta_mode="variance",
        gating_mode_override=None,   # fips_lam=0이면 gating 없음
        ablation_mode="full_loss",
        color="#9CA3AF", label="No gating (F0)",
        # 현재 R4와 동일 — 비교 기준선
    ),
    "F1": dict(
        desc="Random gating — same ratio, random selection",
        tier=1,
        fips_lam=1.0,
        alpha_beta_mode="random",    # ← model.py에 추가 필요
        gating_mode_override="random",
        ablation_mode="full_loss",
        color="#F87171", label="Random gating (F1)",
        # FIW와 동일 비율(sbrs_quantile)로 랜덤 선택
        # "선별 기준"만 다르고 "선별 개수"는 같음 → 직접 비교 가능
    ),
    "F2": dict(
        desc="Boundary-only gating — alpha=1.0, beta=0.0",
        tier=1,
        fips_lam=1.0,
        alpha_beta_mode="bnd_only",  # ← model.py에 추가 필요
        gating_mode_override=None,
        ablation_mode="full_loss",
        color="#FB923C", label="Boundary-only (F2)",
    ),
    "F3": dict(
        desc="Degree-only gating — alpha=0.0, beta=1.0",
        tier=1,
        fips_lam=1.0,
        alpha_beta_mode="deg_only",  # ← model.py에 추가 필요
        gating_mode_override=None,
        ablation_mode="full_loss",
        color="#A78BFA", label="Degree-only (F3)",
    ),
    # ── Tier 2: Uncertainty modulation 비교 ───────────────────────────────
    "F4": dict(
        desc="FIW gating, no uncertainty (struct-score only)",
        tier=2,
        fips_lam=0.0,              # uncertainty 비활성
        alpha_beta_mode="variance", # gating은 FIW 방식 유지
        gating_mode_override=None,
        ablation_mode="full_loss",
        color="#34D399", label="Struct-only FIW (F4)",
        # F0와 차이: F0는 gating 자체가 없고, F4는 gating은 하되 uncertainty 없음
        # → model.py 수정 필요: fips_lam=0이어도 gating은 수행
    ),
    "F5": dict(
        desc="Full FIW — boundary+degree gating + uncertainty",
        tier=1,  # Tier 1과 Tier 2 모두의 기준점
        fips_lam=1.0,
        alpha_beta_mode="variance",
        gating_mode_override=None,
        ablation_mode="full_loss",
        color="#2563EB", label="Full FIW (F5)",
        # 현재 A5와 동일
    ),
}


# ── model.py 수정 가이드 ───────────────────────────────────────────────────────
MODEL_PATCH_GUIDE = """
=== model.py 수정 필요 사항 ===

1. alpha_beta_mode에 새 옵션 추가:

   현재: assert alpha_beta_mode in ("variance", "mutual_info", "uniform")
   변경: assert alpha_beta_mode in (
             "variance", "mutual_info", "uniform",
             "bnd_only", "deg_only", "random"
         )

2. compute_fiw_weights() 내 alpha/beta 분기에 추가:

   elif alpha_beta_mode == "bnd_only":
       alpha, beta = 1.0, 0.0

   elif alpha_beta_mode == "deg_only":
       alpha, beta = 0.0, 1.0

   elif alpha_beta_mode == "random":
       # gating_signal을 랜덤으로 대체 (동일 비율 유지)
       alpha, beta = 0.5, 0.5  # weight는 어차피 random gate에서 결정
       # ↓ gate를 random으로 덮어씀
       perm = torch.randperm(N, device=device)
       n_gate = max(1, int(N * (1.0 - sbrs_quantile)))
       gate = torch.zeros(N, dtype=torch.bool, device=device)
       gate[perm[:n_gate]] = True
       gating_mode = "random"

3. F4 (gating O, uncertainty X) 지원:
   현재: fips_lam=0.0이면 uniform weight 반환 (gating 없음)
   변경: fips_lam=0.0이어도 gating은 수행, uncertainty만 0으로

   # L407 부근, gate 계산 이후:
   if gate.sum() > 0:
       if fips_lam == 0.0:
           # F4: struct_score만, uncertainty 없음
           gated_struct = struct_score[gate]
           gated_struct = _minmax(gated_struct)
           weight[gate] = _WEIGHT_MIN + (_WEIGHT_MAX - _WEIGHT_MIN) * gated_struct
       else:
           # Full FIW
           combined = struct_score[gate] * (1.0 + fips_lam * u_n[gate])
           ...

4. train.py alpha_beta_mode choices 확장:
   현재: choices=["variance", "mutual_info", "uniform"]
   변경: choices=["variance", "mutual_info", "uniform",
                  "bnd_only", "deg_only", "random"]
"""


# ── 명령어 생성 ────────────────────────────────────────────────────────────────

def build_cmd(dataset, stage, stage_cfg, output_file):
    cfg = FAIRGATE_CONFIGS[dataset]
    return [
        sys.executable, "-m", "utils.train",
        "--dataset",           dataset,
        "--backbone",          FIXED["backbone"],
        "--output_file",       output_file,
        "--run_name",          f"fiw_ablation_{stage}_{dataset}",
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
        "--fips_lam",          str(stage_cfg["fips_lam"]),
        "--mmd_alpha",         str(FIXED["mmd_alpha"]),
        "--dp_eo_ratio",       str(FIXED["dp_eo_ratio"]),
        "--uncertainty_type",  FIXED["uncertainty_type"],
        "--ramp_epochs",       str(FIXED["ramp_epochs"]),
        "--recal_interval",    str(FIXED["recal_interval"]),
        "--alpha_beta_mode",   stage_cfg["alpha_beta_mode"],
        "--edge_intervention", FIXED["edge_intervention"],
        "--boundary_sat_thr",  str(FIXED["boundary_sat_thr"]),
        "--ablation_mode",     stage_cfg["ablation_mode"],
        "--ablation_stage",    stage,
    ]


def run_cmd(cmd, dry_run, log_path):
    if dry_run:
        print("    $ " + " ".join(cmd[-18:])); return True
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


# ── 분석 ───────────────────────────────────────────────────────────────────────

def analyze(df: pd.DataFrame, output_dir: str):
    df = df.copy()
    df["fair"] = df["dp_mean"] + df["eo_mean"]

    DNAME = {"pokec_z":"Pokec-Z","pokec_n":"Pokec-N","credit":"Credit",
             "recidivism":"Recidivism","nba":"NBA","income":"Income"}
    SLABEL = {s: FIW_STAGES[s]["label"] for s in FIW_STAGES}

    # F5 기준 delta 계산
    f5_vals = {}
    for ds in DATASETS:
        r = df[(df["dataset"]==ds) & (df["ablation_stage"]=="F5")]
        if not r.empty:
            f5_vals[ds] = r.iloc[0]["fair"]

    # ── Tier 1: Gating 기준 비교 ──────────────────────────────────────────
    print(f"\n{'='*75}")
    print("Tier 1: Gating 선별 기준 비교 (F5=Full FIW 기준)")
    print("↑ = FIW 대비 공정성 악화 → FIW가 해당 기준보다 우월")
    print(f"{'='*75}")

    tier1_stages = ["F0","F1","F2","F3","F5"]
    tier1_labels = {s: FIW_STAGES[s]["desc"].split("—")[0].strip()
                    for s in tier1_stages}

    print(f"\n{'Dataset':<14}" +
          "".join(f"  {tier1_labels[s]:>18}" for s in tier1_stages))
    print("-"*105)

    deltas = {s: [] for s in ["F0","F1","F2","F3"]}
    for ds in DATASETS:
        base = f5_vals.get(ds)
        row = f"  {DNAME.get(ds,ds):<12}"
        for s in tier1_stages:
            r = df[(df["dataset"]==ds) & (df["ablation_stage"]==s)]
            if r.empty: row += f"  {'—':>18}"; continue
            val = r.iloc[0]["fair"]
            if s == "F5" or base is None:
                row += f"  {val:>18.4f}"
            else:
                d = val - base
                mark = "↑" if d > 0.005 else ("↓" if d < -0.005 else " ")
                row += f"  {val:>16.4f}{mark} "
                deltas[s].append(d)
        print(row)

    print(f"\n  F5 대비 평균 Δ(DP+EO) — 양수↑ = FIW보다 나쁨:")
    for s in ["F0","F1","F2","F3"]:
        d = deltas[s]
        if not d: continue
        n_worse  = sum(1 for x in d if x > 0.005)
        n_better = sum(1 for x in d if x < -0.005)
        verdict = "✓ FIW 우월" if np.mean(d) > 0.003 else \
                  ("△ 혼재"   if n_worse > 0 else "✗ FIW 우위 없음")
        print(f"    {tier1_labels[s]:<20}: avg={np.mean(d):>+.4f}  "
              f"악화 {n_worse}/6  개선 {n_better}/6  → {verdict}")

    # ── Tier 2: Uncertainty modulation 비교 ──────────────────────────────
    print(f"\n{'='*75}")
    print("Tier 2: Uncertainty modulation 효과 (F4 vs F5)")
    print("F4: gating O, uncertainty X  |  F5: gating O, uncertainty O")
    print(f"{'='*75}")

    tier2_stages = ["F4","F5"]
    print(f"\n{'Dataset':<14}  {'F4 (no uncert)':>16}  {'F5 (full FIW)':>14}  {'Δ(F4→F5)':>10}")
    print("-"*60)
    unc_deltas = []
    for ds in DATASETS:
        r4 = df[(df["dataset"]==ds) & (df["ablation_stage"]=="F4")]
        r5 = df[(df["dataset"]==ds) & (df["ablation_stage"]=="F5")]
        if r4.empty or r5.empty: continue
        v4 = r4.iloc[0]["fair"]
        v5 = r5.iloc[0]["fair"]
        d  = v5 - v4  # 음수면 F5가 더 좋음
        mark = "✓" if d < -0.003 else ("△" if abs(d) <= 0.003 else "✗")
        unc_deltas.append(d)
        print(f"  {DNAME.get(ds,ds):<12}  {v4:>16.4f}  {v5:>14.4f}  {d:>+10.4f} {mark}")

    if unc_deltas:
        print(f"\n  평균 Δ(F4→F5): {np.mean(unc_deltas):+.4f}  "
              f"(음수면 uncertainty modulation이 기여)")

    # ── 논문 메시지 도출 ─────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("논문 주장 강도 평가")
    print(f"{'='*75}")
    print("""
  [주장 1] "FIW 선별이 랜덤보다 효과적이다" (F5 < F1)
    → F1 avg Δ > 0.005이면 강한 주장 가능
    → F1 avg Δ ≈ 0이면 "선별 비율이 중요하고 기준은 부차적" 인정 필요

  [주장 2] "혼합 신호(boundary+degree)가 단일 신호보다 우월" (F5 < F2, F3)
    → F2, F3 모두 avg Δ > 0이면 가장 강한 주장

  [주장 3] "uncertainty modulation이 추가 기여한다" (F5 < F4)
    → Δ(F4→F5) 평균 음수이면 주장 가능

  실험 결과에 따라 약한 주장은 솔직하게 한계로 인정하세요.
  NeurIPS는 과장된 주장보다 정직한 분석을 더 높이 평가합니다.
    """)

    # LaTeX 저장
    _save_latex(df, DATASETS, DNAME, tier1_stages, f5_vals, deltas, output_dir)


def _save_latex(df, datasets, DNAME, stages, f5_vals, deltas, output_dir):
    SLABEL = {
        "F0": "No gate",
        "F1": "Random",
        "F2": "Bnd-only",
        "F3": "Deg-only",
        "F5": r"\textbf{Full FIW}",
    }
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{FIW gating 선별 기준 비교 ($\Delta\mathrm{DP}+\Delta\mathrm{EO}$). "
        r"F0: gating 없음(uniform); F1: 동일 비율 랜덤 선택; "
        r"F2: boundary 신호만; F3: degree 신호만; "
        r"Full FIW: boundary+degree+uncertainty. "
        r"$\uparrow$: Full FIW 대비 공정성 악화.}",
        r"\label{tab:fiw_ablation}",
        r"\setlength{\tabcolsep}{5pt}", r"\renewcommand{\arraystretch}{1.2}",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{l" + "r"*len(stages) + "}",
        r"\toprule",
        r"\textbf{Dataset} & " +
        " & ".join(SLABEL.get(s, s) for s in stages) + r" \\",
        r"\midrule",
    ]
    for ds in datasets:
        base = f5_vals.get(ds)
        row  = DNAME.get(ds, ds)
        for s in stages:
            r = df[(df["dataset"]==ds) & (df["ablation_stage"]==s)]
            if r.empty: row += " & —"; continue
            val = r.iloc[0]["fair"]
            if s == "F5":
                row += f" & \\textbf{{{val:.4f}}}"
            elif base:
                d   = val - base
                sup = r"$^{\uparrow}$" if d > 0.005 else ""
                row += f" & {val:.4f}{sup}"
            else:
                row += f" & {val:.4f}"
        lines.append(row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    path = os.path.join(output_dir, "fiw_ablation.tex")
    with open(path, "w") as f: f.write("\n".join(lines))
    print(f"[Saved] {path}")


# ── 메인 ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="FIW Gating Effectiveness Ablation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tier",         type=int, default=0,
                   choices=[0,1,2],
                   help="0=both, 1=gating 기준, 2=uncertainty modulation")
    p.add_argument("--datasets",     nargs="+", default=DATASETS)
    p.add_argument("--output_file",  type=str,
                   default="analysis/exp_fiw_ablation.csv")
    p.add_argument("--output_dir",   type=str,
                   default="outputs/analysis")
    p.add_argument("--log_dir",      type=str,
                   default="logs/fiw_ablation")
    p.add_argument("--dry_run",      action="store_true")
    p.add_argument("--analyze_only", action="store_true")
    p.add_argument("--print_patch",  action="store_true",
                   help="model.py 수정 가이드 출력")
    return p.parse_args()


def main():
    args = parse_args()

    if args.print_patch:
        print(MODEL_PATCH_GUIDE)
        return

    # tier에 따라 실행할 stage 선택
    if args.tier == 1:
        run_stages = {k: v for k, v in FIW_STAGES.items() if v["tier"] == 1}
    elif args.tier == 2:
        run_stages = {k: v for k, v in FIW_STAGES.items()
                      if v["tier"] in (1, 2) and k in ("F4", "F5")}
    else:
        run_stages = FIW_STAGES

    os.makedirs(args.output_dir, exist_ok=True)
    total = len(run_stages) * len(args.datasets)

    print(f"\n{'='*65}")
    print("FIW Gating Effectiveness Ablation")
    print(f"  Stages  : {list(run_stages.keys())}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Total   : {total} runs")
    print(f"\n  ※ 실행 전 model.py 수정이 필요합니다.")
    print(f"     python run_fiw_ablation.py --print_patch  로 가이드 확인")
    print(f"{'='*65}")

    if not args.analyze_only:
        step = 0
        for stage, scfg in run_stages.items():
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
        if os.path.exists(args.output_file):
            df = pd.read_csv(args.output_file)
            analyze(df, args.output_dir)
        else:
            print(f"\n[WARN] 결과 파일 없음: {args.output_file}")


if __name__ == "__main__":
    main()
