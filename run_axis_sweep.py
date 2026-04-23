"""
run_axis_sweep.py — 축별 one-at-a-time sweep 실험

첨부 문서 기준: 4개 축을 하나씩만 변화시키며 곡선 모양(curvature, robustness)을 분석.
나머지 파라미터는 FAIRGATE_CONFIGS 고정.

4개 축:
    gate_budget      ← sbrs_quantile 변화
    attenuation      ← struct_drop 변화
    fairness_budget  ← lambda_fair 변화
    phase_lag        ← warm_up 변화

추가로 FairGate 고유 파라미터도 스윕:
    boundary_sat_thr ← gating 전환 임계값 τ
    fips_lam         ← uncertainty 증폭 계수

출력:
    outputs/axis_sweep/exp_sweep_{axis}_{dataset}.csv
    outputs/analysis/axis_curvature.csv   ← 곡선 모양 요약
    outputs/analysis/axis_curvature.tex   ← LaTeX 표

실행:
    # 대표 3개 데이터셋 × 4축 전체
    python run_axis_sweep.py

    # 특정 축만
    python run_axis_sweep.py --axes gate_budget attenuation

    # 특정 데이터셋만
    python run_axis_sweep.py --datasets pokec_z german credit

    # dry run
    python run_axis_sweep.py --dry_run

    # 실험 없이 분석만
    python run_axis_sweep.py --analyze_only
"""

import os, sys, argparse, subprocess
import numpy as np
import pandas as pd

DEVICE = "cuda:1"

# 대표 3개 데이터셋: clustered / saturated / degree-skewed
DEFAULT_DATASETS = ["pokec_z", "german", "credit"]

# 전체 9개 (선택적)
ALL_DATASETS = [
    "pokec_z", "pokec_z_g", "pokec_n", "pokec_n_g",
    "german", "credit", "recidivism", "nba", "income",
]

FAIRGATE_CONFIGS = {
    "pokec_z"   : dict(lambda_fair=0.10, sbrs_quantile=0.9, struct_drop=0.5, warm_up=400),
    "pokec_z_g" : dict(lambda_fair=0.20, sbrs_quantile=0.9, struct_drop=0.5, warm_up=100),
    "pokec_n"   : dict(lambda_fair=0.15, sbrs_quantile=0.5, struct_drop=0.5, warm_up=400),
    "pokec_n_g" : dict(lambda_fair=0.01, sbrs_quantile=0.8, struct_drop=0.5, warm_up=400),
    "credit"    : dict(lambda_fair=0.20, sbrs_quantile=0.5, struct_drop=0.7, warm_up=200),
    "recidivism": dict(lambda_fair=0.10, sbrs_quantile=0.9, struct_drop=0.2, warm_up=100),
    "income"    : dict(lambda_fair=0.20, sbrs_quantile=0.5, struct_drop=0.7, warm_up=200),
    "german"    : dict(lambda_fair=0.20, sbrs_quantile=0.7, struct_drop=0.2, warm_up=100),
    "nba"       : dict(lambda_fair=0.40, sbrs_quantile=0.5, struct_drop=0.3, warm_up=200),
}

FIXED = dict(
    backbone="GCN", hidden_dim=128, dropout=0.5, sgc_k=2,
    lr=1e-3, weight_decay=1e-5, epochs=500, patience=501,
    runs=3, seed=27, fips_lam=1.0, mmd_alpha=0.3,
    dp_eo_ratio=0.3, uncertainty_type="entropy",
    ramp_epochs=0, recal_interval=200,
    alpha_beta_mode="variance", edge_intervention="drop",
    boundary_sat_thr=0.9, ablation_mode="full_loss",
)

# ── 4개 축 + FairGate 고유 파라미터 스윕 범위 ────────────────────────────────
AXES = {
    "gate_budget": {
        "param"   : "sbrs_quantile",
        "values"  : [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        "label"   : "Gate budget (1−q)",
        "struct_hypothesis": "homophily↑ → 넓은 gate / boundary_ratio↓ → 좁은 gate",
    },
    "attenuation": {
        "param"   : "struct_drop",
        "values"  : [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
        "label"   : "Attenuation (struct_drop)",
        "struct_hypothesis": "boundary_ratio↑ → 약한 attenuation (r=−0.67 ★★) / "
                             "bridge_criticality: 실측 r=+0.44 (방향 예상과 반대, sweep으로 재확인)",
    },
    "fairness_budget": {
        "param"   : "lambda_fair",
        "values"  : [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
        "label"   : "Fairness budget (lambda_fair)",
        "struct_hypothesis": "deg_gap↑ → 강한 budget / homophily↑ → 강한 budget",
    },
    "phase_lag": {
        "param"   : "warm_up",
        "values"  : [0, 50, 100, 200, 400, 600],
        "label"   : "Phase lag (warm_up)",
        "struct_hypothesis": "bnd↑ → 짧은 warm_up (r=−0.59 ★) / "
                             "lh_std: 실측 r=+0.023 (유의미하지 않음, bridge r=+0.51 ★로 대체 검토)",
    },
    # FairGate 고유 파라미터
    "boundary_sat_thr": {
        "param"   : "boundary_sat_thr",
        "values"  : [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        "label"   : "Saturation threshold (τ)",
        "struct_hypothesis": "boundary_ratio 높은 데이터셋에서 τ 민감도 높음",
    },
}

# 실측값 기준 (analyze_graph_stats.py 실행 결과 반영)
GRAPH_STATS = {
    "pokec_z"   : dict(h=0.9532, bnd=0.3689, deg_gap=0.0833, lh_std=0.1333,
                       bridge=0.1673, regime="clustered"),
    "pokec_z_g" : dict(h=0.4792, bnd=0.8889, deg_gap=0.0240, lh_std=0.2417,
                       bridge=0.0777, regime="mixed"),
    "pokec_n"   : dict(h=0.9559, bnd=0.3070, deg_gap=0.0562, lh_std=0.1296,
                       bridge=0.1450, regime="clustered"),
    "pokec_n_g" : dict(h=0.4889, bnd=0.8861, deg_gap=0.0107, lh_std=0.2427,
                       bridge=0.0737, regime="mixed"),
    "german"    : dict(h=0.8092, bnd=0.9700, deg_gap=0.0491, lh_std=0.1331,
                       bridge=0.0608, regime="saturated"),
    "credit"    : dict(h=0.9600, bnd=0.6768, deg_gap=0.3152, lh_std=0.1114,
                       bridge=0.1820, regime="degree-skewed"),
    "recidivism": dict(h=0.5361, bnd=0.9983, deg_gap=0.0231, lh_std=0.1244,
                       bridge=0.0597, regime="saturated"),
    "nba"       : dict(h=0.7288, bnd=0.9777, deg_gap=0.0957, lh_std=0.1902,
                       bridge=0.0985, regime="saturated"),
    "income"    : dict(h=0.8844, bnd=0.3343, deg_gap=0.1231, lh_std=0.1921,
                       bridge=0.0561, regime="clustered"),
}

DATASET_DISPLAY = {
    "pokec_z":"Pokec-Z", "pokec_z_g":"Pokec-Z(g)", "pokec_n":"Pokec-N",
    "pokec_n_g":"Pokec-N(g)", "german":"German", "credit":"Credit",
    "recidivism":"Recidivism", "nba":"NBA", "income":"Income",
}


# ── 명령어 생성 ────────────────────────────────────────────────────────────────

def build_cmd(dataset: str, axis: str, value, output_file: str) -> list:
    cfg   = FAIRGATE_CONFIGS[dataset].copy()
    param = AXES[axis]["param"]
    cfg[param] = value  # 해당 축만 오버라이드

    run_name = f"sweep_{axis}_{str(value).replace('.','p')}_{dataset}"

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
        "--lambda_fair",       str(cfg.get("lambda_fair", 0.1)),
        "--sbrs_quantile",     str(cfg.get("sbrs_quantile", 0.7)),
        "--struct_drop",       str(cfg.get("struct_drop", 0.5)),
        "--warm_up",           str(int(cfg.get("warm_up", 200))),
        "--fips_lam",          str(FIXED["fips_lam"]),
        "--mmd_alpha",         str(FIXED["mmd_alpha"]),
        "--dp_eo_ratio",       str(FIXED["dp_eo_ratio"]),
        "--uncertainty_type",  FIXED["uncertainty_type"],
        "--ramp_epochs",       str(FIXED["ramp_epochs"]),
        "--recal_interval",    str(FIXED["recal_interval"]),
        "--alpha_beta_mode",   FIXED["alpha_beta_mode"],
        "--edge_intervention", FIXED["edge_intervention"],
        "--boundary_sat_thr",  str(cfg.get("boundary_sat_thr", FIXED["boundary_sat_thr"])),
        "--ablation_mode",     FIXED["ablation_mode"],
        "--sensitivity_param", param,
        "--sensitivity_value", str(value),
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


# ── 곡선 분석 ──────────────────────────────────────────────────────────────────

def compute_curvature(values, scores) -> dict:
    """
    각 축의 곡선 모양 분석:
        score_span      : 성능 변화 폭 (민감도)
        best_val        : 최적값
        robustness_width: 좋은 성능 유지 구간 수
        curvature       : 최적점 주변 곡률 (뾰족하면 ↑)
    """
    vals   = np.array(values, dtype=float)
    scores = np.array(scores, dtype=float)
    valid  = ~np.isnan(scores)
    if valid.sum() < 2:
        return dict(score_span=np.nan, best_val=np.nan,
                    robustness_width=np.nan, curvature=np.nan)

    vals, scores = vals[valid], scores[valid]
    best_idx     = int(np.argmin(scores))   # DP+EO 최소가 최선
    best_val     = float(vals[best_idx])
    score_span   = float(scores.max() - scores.min())

    # robustness_width: best+1% 이내 구간 수
    tol = max(scores.min() * 0.01, 0.005)
    robustness_width = int((scores <= scores.min() + tol).sum())

    # curvature: 이차 다항 근사의 이차 계수 (클수록 뾰족)
    if len(vals) >= 3:
        try:
            coeffs = np.polyfit(vals, scores, 2)
            curvature = abs(float(coeffs[0]))
        except Exception:
            curvature = np.nan
    else:
        curvature = np.nan

    return dict(score_span=round(score_span, 4),
                best_val=round(best_val, 4),
                robustness_width=robustness_width,
                curvature=round(curvature, 4))


# ── 결과 분석 ──────────────────────────────────────────────────────────────────

def analyze(output_dir_sweep: str, axes: list, datasets: list,
            output_dir_analysis: str):
    rows = []
    for axis in axes:
        param = AXES[axis]["param"]
        for ds in datasets:
            fpath = os.path.join(output_dir_sweep, f"exp_sweep_{axis}_{ds}.csv")
            if not os.path.exists(fpath):
                continue
            df = pd.read_csv(fpath)
            df = df[df["dataset"] == ds].copy()
            if "sensitivity_param" in df.columns:
                df = df[df["sensitivity_param"] == param]
            if df.empty:
                continue

            df["fair_sum"] = df["dp_mean"] + df["eo_mean"]
            grp = df.groupby("sensitivity_value")["fair_sum"].mean()

            curve = compute_curvature(grp.index.tolist(), grp.values.tolist())
            gs = GRAPH_STATS.get(ds, {})

            rows.append({
                "axis"     : axis,
                "dataset"  : ds,
                "regime"   : gs.get("regime","?"),
                "h"        : gs.get("h"),
                "bnd"      : gs.get("bnd"),
                "deg_gap"  : gs.get("deg_gap"),
                "lh_std"   : gs.get("lh_std"),
                "bridge"   : gs.get("bridge"),
                **curve,
            })

    if not rows:
        print("[WARN] 분석할 결과 없음")
        return

    summary = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir_analysis, "axis_curvature.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\n[Saved] {csv_path}")

    # 콘솔 출력
    print(f"\n{'='*85}")
    print("축별 곡선 분석 (score_span: 민감도, robust_w: 안정 구간, curvature: 뾰족함)")
    print(f"{'='*85}")
    print(f"{'Axis':<20} {'Dataset':<14} {'regime':<14} "
          f"{'span':>7} {'best':>7} {'robust_w':>9} {'curve':>8}")
    print("-"*85)
    for _, r in summary.iterrows():
        print(f"{r['axis']:<20} {DATASET_DISPLAY.get(r['dataset'],r['dataset']):<14} "
              f"{r['regime']:<14} "
              f"{r['score_span']:>7.4f} {r['best_val']:>7.4f} "
              f"{int(r['robustness_width']) if not np.isnan(r['robustness_width']) else '?':>9} "
              f"{r['curvature']:>8.4f}")

    # 구조→축 상관
    print(f"\n{'='*60}")
    print("score_span ↔ 구조 변수 상관 (민감한 축을 예측하는 구조 변수)")
    print(f"{'='*60}")
    for ax in axes:
        sub = summary[summary["axis"] == ax]
        if len(sub) < 3:
            continue
        print(f"\n  [{AXES[ax]['label']}]")
        print(f"  가설: {AXES[ax]['struct_hypothesis']}")
        for sv in ["h","bnd","deg_gap","lh_std","bridge"]:
            if sv in sub.columns and sub[sv].notna().sum() >= 3:
                r = sub["score_span"].corr(sub[sv])
                if not np.isnan(r):
                    mark = " ★★" if abs(r) > 0.6 else (" ★" if abs(r) > 0.4 else "")
                    print(f"    score_span vs {sv:<10}: r={r:+.3f}{mark}")

    # LaTeX
    _save_curvature_latex(summary, axes, output_dir_analysis)


def _save_curvature_latex(summary, axes, output_dir):
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{축별 one-at-a-time sweep 결과. "
        r"score\_span: 성능 변화 폭, best: 최적값, "
        r"robust\_w: 좋은 성능 유지 구간 수, curvature: 최적점 뾰족함.}",
        r"\label{tab:axis_curvature}",
        r"\setlength{\tabcolsep}{5pt}", r"\renewcommand{\arraystretch}{1.2}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\textbf{Axis} & \textbf{Dataset} & span & best & robust & curve \\",
        r"\midrule",
    ]
    for ax in axes:
        sub = summary[summary["axis"]==ax]
        first = True
        for _, r in sub.iterrows():
            ax_label = AXES[ax]["label"] if first else ""
            first = False
            span = f"{r['score_span']:.4f}" if not np.isnan(r['score_span']) else "—"
            best = f"{r['best_val']:.4f}"   if not np.isnan(r['best_val'])   else "—"
            rw   = str(int(r['robustness_width'])) if not np.isnan(r['robustness_width']) else "—"
            cur  = f"{r['curvature']:.4f}"  if not np.isnan(r['curvature'])  else "—"
            lines.append(f"{ax_label} & {DATASET_DISPLAY.get(r['dataset'],r['dataset'])} "
                         f"& {span} & {best} & {rw} & {cur} \\\\")
        if ax != axes[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = os.path.join(output_dir, "axis_curvature.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[Saved] {path}")


# ── 메인 ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--datasets",      nargs="+", default=DEFAULT_DATASETS,
                   choices=ALL_DATASETS)
    p.add_argument("--axes",          nargs="+", default=list(AXES.keys()),
                   choices=list(AXES.keys()))
    p.add_argument("--output_dir_sweep",    type=str, default="outputs/axis_sweep")
    p.add_argument("--output_dir_analysis", type=str, default="outputs/analysis")
    p.add_argument("--log_dir",       type=str, default="logs/axis_sweep")
    p.add_argument("--dry_run",       action="store_true")
    p.add_argument("--analyze_only",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir_sweep,    exist_ok=True)
    os.makedirs(args.output_dir_analysis, exist_ok=True)

    total = sum(len(AXES[ax]["values"]) for ax in args.axes) * len(args.datasets)
    print(f"\n{'='*65}")
    print(f"FairGate Axis Sweep (one-at-a-time)")
    print(f"  axes     : {args.axes}")
    print(f"  datasets : {args.datasets}")
    print(f"  total    : {total} runs")
    print(f"{'='*65}")

    if not args.analyze_only:
        step = 0
        for axis in args.axes:
            ax_info = AXES[axis]
            print(f"\n{'─'*65}")
            print(f"  [Axis: {axis}]  {ax_info['label']}")
            print(f"  가설: {ax_info['struct_hypothesis']}")
            print(f"  sweep: {ax_info['values']}")
            print(f"{'─'*65}")

            for val in ax_info["values"]:
                for ds in args.datasets:
                    step += 1
                    output_file = os.path.join(
                        args.output_dir_sweep, f"exp_sweep_{axis}_{ds}.csv")
                    cmd = build_cmd(ds, axis, val, output_file)
                    log = os.path.join(args.log_dir, axis,
                                       f"{ds}_{str(val).replace('.','p')}.log")
                    print(f"  [{step:3d}/{total}] {ds:<14}  {axis}={val}")
                    run_cmd(cmd, args.dry_run, log)

    if not args.dry_run:
        print("\n[분석 실행]")
        analyze(args.output_dir_sweep, args.axes, args.datasets,
                args.output_dir_analysis)

    print(f"\n완료.")


if __name__ == "__main__":
    main()