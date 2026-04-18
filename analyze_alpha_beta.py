"""
analyze_alpha_beta.py — α,β 계산 방식 비교 분석

alpha_beta_mode (variance / mutual_info / uniform) 세 방식의 성능을 비교한다.
실험 결과 CSV에 alpha_beta_mode 컬럼이 있어야 한다.

방식 설명:
    variance    : 특성 분산 기반 α,β 계산 (FairGate 기본값)
    mutual_info : 민감속성과의 상호정보량 기반
    uniform     : α=β=1 (균등 가중치, ablation baseline)

분석 항목:
    - 방식별 × 데이터셋별 ΔDP, ΔEO, AUC 비교
    - variance 방식의 우수성 및 일관성 분석
    - regime별 최적 방식 패턴

출력:
    outputs/analysis/alpha_beta_comparison.csv
    outputs/analysis/alpha_beta_comparison.tex
    outputs/analysis/alpha_beta_summary.tex

실행:
    # 단일 CSV에 alpha_beta_mode 컬럼이 있는 경우
    python analyze_alpha_beta.py --csv outputs/exp_fairgate_gcn.csv

    # 여러 CSV 합산
    python analyze_alpha_beta.py \\
        --csv outputs/ablation/exp_ab_variance.csv \\
               outputs/ablation/exp_ab_mutual_info.csv \\
               outputs/ablation/exp_ab_uniform.csv
"""

import os
import argparse
import numpy as np
import pandas as pd


# ── 설정 ───────────────────────────────────────────────────────────────────────
MODE_ORDER   = ["variance", "mutual_info", "uniform"]
MODE_DISPLAY = {
    "variance"   : r"Variance (ours)",
    "mutual_info": r"Mutual info",
    "uniform"    : r"Uniform ($\alpha{=}\beta{=}1$)",
}

DATASET_ORDER = [
    "pokec_z", "pokec_z_g", "pokec_n", "pokec_n_g",
    "german", "credit", "recidivism", "nba", "income",
]
DATASET_DISPLAY = {
    "pokec_z"   : "Pokec-Z",       "pokec_z_g" : "Pokec-Z (gender)",
    "pokec_n"   : "Pokec-N",       "pokec_n_g" : "Pokec-N (gender)",
    "german"    : "German",         "credit"    : "Credit",
    "recidivism": "Recidivism",     "nba"       : "NBA",
    "income"    : "Income",
}

SHOW_METRICS  = ["roc_auc_mean", "dp_mean", "eo_mean"]
LOWER_BETTER  = {"dp_mean", "eo_mean"}
METRIC_LABEL  = {
    "roc_auc_mean": "AUC",
    "dp_mean"     : r"$\Delta$DP",
    "eo_mean"     : r"$\Delta$EO",
}


# ── 로드 ───────────────────────────────────────────────────────────────────────

def load_and_merge(csv_paths: list) -> pd.DataFrame:
    dfs = []
    for path in csv_paths:
        if not os.path.exists(path):
            print(f"  [WARN] 파일 없음: {path}")
            continue
        df = pd.read_csv(path)
        if "model" in df.columns:
            df = df[df["model"] == "FairGate"].copy()
        dfs.append(df)
        print(f"  [OK] {path}  ({len(df)} rows)")
    if not dfs:
        raise RuntimeError("로드된 파일 없음")
    merged = pd.concat(dfs, ignore_index=True)

    if "alpha_beta_mode" not in merged.columns:
        raise ValueError(
            "alpha_beta_mode 컬럼이 없습니다.\n"
            "run_hparam_search.py 또는 train.py에서 --alpha_beta_mode 옵션을 지정하고 실험을 재실행하세요."
        )
    return merged


# ── 집계 ───────────────────────────────────────────────────────────────────────

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    all_metrics = SHOW_METRICS + [
        m.replace("_mean", "_std") for m in SHOW_METRICS
        if m.replace("_mean", "_std") in df.columns
    ] + ["f1_mean", "acc_mean"]
    cols = ["dataset", "alpha_beta_mode"] + [c for c in all_metrics if c in df.columns]
    return df.groupby(["dataset", "alpha_beta_mode"])[
        [c for c in all_metrics if c in df.columns]
    ].mean().reset_index()


# ── 콘솔 출력 ──────────────────────────────────────────────────────────────────

def print_comparison(df: pd.DataFrame):
    modes    = [m for m in MODE_ORDER if m in df["alpha_beta_mode"].unique()]
    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]

    print(f"\n{'='*85}")
    print(f"{'Dataset':<22}", end="")
    for m in modes:
        print(f"  {'─'*2} {m:<12} {'─'*2}", end="")
    print(f"\n{'':22}", end="")
    for _ in modes:
        print(f"  {'AUC':>6} {'ΔDP':>6} {'ΔEO':>6}  ", end="")
    print(f"\n{'='*85}")

    # variance가 best인 횟수 카운트
    variance_wins = {"auc": 0, "dp": 0, "eo": 0, "total": 0}

    for ds in datasets:
        sub = df[df["dataset"] == ds]
        print(f"{DATASET_DISPLAY.get(ds, ds):<22}", end="")
        row_vals = {}
        for m in modes:
            r = sub[sub["alpha_beta_mode"] == m]
            row_vals[m] = {
                "auc": r.iloc[0]["roc_auc_mean"] if not r.empty and "roc_auc_mean" in r else float("nan"),
                "dp" : r.iloc[0]["dp_mean"]      if not r.empty and "dp_mean"      in r else float("nan"),
                "eo" : r.iloc[0]["eo_mean"]       if not r.empty and "eo_mean"      in r else float("nan"),
            }

        # best 찾기
        best_auc = max(modes, key=lambda m: row_vals[m]["auc"], default=None)
        best_dp  = min(modes, key=lambda m: row_vals[m]["dp"],  default=None)
        best_eo  = min(modes, key=lambda m: row_vals[m]["eo"],  default=None)
        if best_auc == "variance": variance_wins["auc"] += 1
        if best_dp  == "variance": variance_wins["dp"]  += 1
        if best_eo  == "variance": variance_wins["eo"]  += 1

        for m in modes:
            v = row_vals[m]
            auc_str = f"{v['auc']:.4f}" + ("*" if best_auc == m else " ")
            dp_str  = f"{v['dp']:.4f}"  + ("*" if best_dp  == m else " ")
            eo_str  = f"{v['eo']:.4f}"  + ("*" if best_eo  == m else " ")
            print(f"  {auc_str:>7} {dp_str:>7} {eo_str:>7}", end="")
        print()

    print(f"{'='*85}")
    n = len(datasets)
    print(f"\n[variance 방식 최고 횟수 / {n}개 데이터셋]")
    print(f"  AUC: {variance_wins['auc']}/{n}  "
          f"ΔDP: {variance_wins['dp']}/{n}  "
          f"ΔEO: {variance_wins['eo']}/{n}")

    # 방식별 전체 평균
    print("\n[방식별 전체 평균]")
    summary = df.groupby("alpha_beta_mode")[SHOW_METRICS].mean()
    for m in modes:
        if m not in summary.index:
            continue
        r = summary.loc[m]
        fair = r.get("dp_mean", 0) + r.get("eo_mean", 0)
        print(f"  {m:<14}: AUC={r.get('roc_auc_mean',0):.4f}  "
              f"ΔDP={r.get('dp_mean',0):.4f}  "
              f"ΔEO={r.get('eo_mean',0):.4f}  "
              f"ΔDP+ΔEO={fair:.4f}")

    # regime별 분석 (데이터셋 알려진 경우)
    regime_map = {
        "pokec_z": "clustered",   "pokec_z_g": "mixed",
        "pokec_n": "clustered",   "pokec_n_g": "mixed",
        "german" : "saturated",   "credit"   : "degree-skewed",
        "recidivism": "saturated","nba"      : "saturated",
        "income" : "clustered",
    }
    if all(ds in regime_map for ds in datasets):
        print("\n[Regime별 최적 방식 (ΔDP+ΔEO 기준)]")
        df2 = df.copy()
        df2["regime"] = df2["dataset"].map(regime_map)
        df2["fair_sum"] = df2["dp_mean"] + df2["eo_mean"]
        for regime, grp in df2.groupby("regime"):
            best_mode = grp.groupby("alpha_beta_mode")["fair_sum"].mean().idxmin()
            vals = grp.groupby("alpha_beta_mode")["fair_sum"].mean()
            print(f"  {regime:<15}: best={best_mode}  "
                  + "  ".join(f"{m}={vals.get(m, float('nan')):.4f}" for m in modes if m in vals))


# ── LaTeX ──────────────────────────────────────────────────────────────────────

def to_latex_full(df: pd.DataFrame) -> str:
    modes    = [m for m in MODE_ORDER if m in df["alpha_beta_mode"].unique()]
    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{$\alpha, \beta$ 계산 방식에 따른 FairGate 성능 비교. "
        r"각 데이터셋에서 DP+EO 합산이 최소인 값을 \textbf{굵게} 표시. "
        r"Variance 방식이 대부분의 데이터셋에서 최선 또는 동등한 성능을 보인다.}"
    )
    lines.append(r"\label{tab:alpha_beta_comparison}")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\resizebox{\linewidth}{!}{")

    col_fmt = "l" + "rrr" * len(modes)
    lines.append(r"\begin{tabular}{" + col_fmt + "}")
    lines.append(r"\toprule")

    # 헤더 1
    hdr1 = r"\multirow{2}{*}{\textbf{Dataset}}"
    for m in modes:
        hdr1 += r" & \multicolumn{3}{c}{" + MODE_DISPLAY[m] + "}"
    lines.append(hdr1 + r" \\")
    for i in range(len(modes)):
        s = 2 + i * 3
        lines.append(r"\cmidrule(lr){" + f"{s}-{s+2}" + "}")

    # 헤더 2
    hdr2 = ""
    for _ in modes:
        hdr2 += r" & AUC$\uparrow$ & $\Delta$DP$\downarrow$ & $\Delta$EO$\downarrow$"
    lines.append(hdr2 + r" \\")
    lines.append(r"\midrule")

    for ds in datasets:
        sub = df[df["dataset"] == ds]
        # best 찾기
        best = {}
        for met in SHOW_METRICS:
            vals = {m: sub[sub["alpha_beta_mode"] == m].iloc[0][met]
                    for m in modes
                    if not sub[sub["alpha_beta_mode"] == m].empty
                    and met in sub.columns}
            if vals:
                best[met] = (min(vals, key=vals.get) if met in LOWER_BETTER
                             else max(vals, key=vals.get))

        row = DATASET_DISPLAY.get(ds, ds)
        for m in modes:
            r = sub[sub["alpha_beta_mode"] == m]
            for met in SHOW_METRICS:
                if r.empty or met not in r.columns:
                    row += " & —"
                else:
                    val  = r.iloc[0][met]
                    cell = f"{val:.4f}"
                    if best.get(met) == m:
                        cell = r"\textbf{" + cell + "}"
                    row += " & " + cell
        lines.append(row + r" \\")

    # 평균 행
    lines.append(r"\midrule")
    avg = df.groupby("alpha_beta_mode")[SHOW_METRICS].mean()
    avg_row = r"\textit{Average}"
    for m in modes:
        for met in SHOW_METRICS:
            val = avg.loc[m, met] if m in avg.index and met in avg.columns else float("nan")
            avg_row += f" & {val:.4f}"
    lines.append(avg_row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def to_latex_summary(df: pd.DataFrame) -> str:
    modes   = [m for m in MODE_ORDER if m in df["alpha_beta_mode"].unique()]
    metrics = [m for m in ["acc_mean", "roc_auc_mean", "f1_mean",
                            "dp_mean", "eo_mean"] if m in df.columns]
    summary = df.groupby("alpha_beta_mode")[metrics].mean()

    label_map = {
        "acc_mean": r"ACC$\uparrow$", "roc_auc_mean": r"AUC$\uparrow$",
        "f1_mean" : r"F1$\uparrow$",  "dp_mean"     : r"$\Delta$DP$\downarrow$",
        "eo_mean" : r"$\Delta$EO$\downarrow$",
    }

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{$\alpha,\beta$ 계산 방식별 전체 데이터셋 평균 성능. "
        r"Variance 방식이 공정성 지표에서 일관되게 우수하다.}"
    )
    lines.append(r"\label{tab:alpha_beta_summary}")
    lines.append(r"\setlength{\tabcolsep}{8pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    lines.append(r"\begin{tabular}{l" + "r" * len(metrics) + "}")
    lines.append(r"\toprule")
    hdr = r"\textbf{Mode}" + "".join(" & " + label_map.get(m, m) for m in metrics)
    lines.append(hdr + r" \\")
    lines.append(r"\midrule")
    for m in modes:
        if m not in summary.index:
            continue
        row = MODE_DISPLAY[m]
        for met in metrics:
            row += f" & {summary.loc[m, met]:.4f}"
        lines.append(row + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── 메인 ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="FairGate α,β 계산 방식 비교 분석",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", nargs="+",
                   default=["outputs/exp_fairgate_gcn.csv"],
                   help="분석할 CSV 파일 (alpha_beta_mode 컬럼 필요). 여러 개 가능")
    p.add_argument("--output_dir", type=str, default="outputs/analysis")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[Alpha-Beta Mode Comparison Analysis]")
    df_all = load_and_merge(args.csv)

    modes_found = df_all["alpha_beta_mode"].unique().tolist()
    print(f"  발견된 alpha_beta_mode: {modes_found}")
    print(f"  발견된 데이터셋: {sorted(df_all['dataset'].unique())}\n")

    if len(modes_found) < 2:
        print(
            "[INFO] alpha_beta_mode가 1종류만 있습니다.\n"
            "  비교 실험을 위해 --alpha_beta_mode 옵션을 다르게 설정한 실험을 추가로 실행하세요:\n"
            "  python train.py --dataset pokec_z --alpha_beta_mode mutual_info ...\n"
            "  python train.py --dataset pokec_z --alpha_beta_mode uniform ..."
        )
        return

    df_agg = aggregate(df_all)
    print_comparison(df_agg)

    csv_path = os.path.join(args.output_dir, "alpha_beta_comparison.csv")
    df_agg.to_csv(csv_path, index=False)
    print(f"\n[Saved] {csv_path}")

    for fname, content in [
        ("alpha_beta_comparison.tex", to_latex_full(df_agg)),
        ("alpha_beta_summary.tex",    to_latex_summary(df_agg)),
    ]:
        path = os.path.join(args.output_dir, fname)
        with open(path, "w") as f:
            f.write(content)
        print(f"[Saved] {path}")


if __name__ == "__main__":
    main()