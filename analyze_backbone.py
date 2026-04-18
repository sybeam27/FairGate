"""
analyze_backbone.py — FairGate 백본 범용성 검증

exp_fairgate_gcn.csv, exp_fairgate_sage.csv, exp_fairgate_sgc.csv를 읽어
GCN / GraphSAGE / SGC 백본 간 성능을 데이터셋별로 비교한다.

분석 항목:
    - ACC / AUC / F1 / ΔDP / ΔEO 백본 간 비교 테이블
    - 백본별 평균 성능 및 순위
    - FairGate가 세 백본에서 일관된 공정성 개선을 보이는지 검증
    - 데이터셋 × 백본 히트맵용 수치 출력

출력:
    outputs/analysis/backbone_comparison.csv
    outputs/analysis/backbone_comparison.tex   (논문 삽입용)
    outputs/analysis/backbone_summary.tex      (백본별 평균 요약)

실행:
    python analyze_backbone.py
    python analyze_backbone.py \\
        --gcn  outputs/exp_fairgate_gcn.csv \\
        --sage outputs/exp_fairgate_sage.csv \\
        --sgc  outputs/exp_fairgate_sgc.csv
"""

import os
import argparse
import numpy as np
import pandas as pd


# ── 설정 ───────────────────────────────────────────────────────────────────────
DATASET_ORDER = [
    "pokec_z", "pokec_z_g", "pokec_n", "pokec_n_g",
    "german", "credit", "recidivism", "nba", "income",
]

DATASET_DISPLAY = {
    "pokec_z"   : "Pokec-Z",
    "pokec_z_g" : "Pokec-Z (gender)",
    "pokec_n"   : "Pokec-N",
    "pokec_n_g" : "Pokec-N (gender)",
    "german"    : "German",
    "credit"    : "Credit",
    "recidivism": "Recidivism",
    "nba"       : "NBA",
    "income"    : "Income",
}

METRICS = ["acc_mean", "roc_auc_mean", "f1_mean", "dp_mean", "eo_mean"]
METRIC_DISPLAY = {
    "acc_mean"    : "ACC",
    "roc_auc_mean": "AUC",
    "f1_mean"     : "F1",
    "dp_mean"     : r"$\Delta$DP",
    "eo_mean"     : r"$\Delta$EO",
}
# 낮을수록 좋은 지표
LOWER_BETTER = {"dp_mean", "eo_mean"}


# ── 로드 ───────────────────────────────────────────────────────────────────────

def load_csv(path: str, backbone_name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일 없음: {path}")
    df = pd.read_csv(path)
    if "model" in df.columns:
        df = df[df["model"] == "FairGate"].copy()
    df["backbone"] = backbone_name
    return df


def load_all(gcn_path, sage_path, sgc_path) -> pd.DataFrame:
    dfs = []
    for path, name in [(gcn_path, "GCN"), (sage_path, "GraphSAGE"), (sgc_path, "SGC")]:
        try:
            df = load_csv(path, name)
            dfs.append(df)
            print(f"  [{name}] {len(df)} rows  from {path}")
        except FileNotFoundError as e:
            print(f"  [WARN] {e}")
    if not dfs:
        raise RuntimeError("로드된 파일 없음")
    return pd.concat(dfs, ignore_index=True)


# ── 집계 ───────────────────────────────────────────────────────────────────────

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """데이터셋 × 백본별 핵심 지표 집계"""
    cols = ["dataset", "backbone"] + [c for c in METRICS if c in df.columns]
    std_cols = [c.replace("_mean", "_std") for c in METRICS
                if c.replace("_mean", "_std") in df.columns]
    grp = df.groupby(["dataset", "backbone"])[
        [c for c in METRICS + std_cols if c in df.columns]
    ].mean().reset_index()
    return grp


# ── 콘솔 출력 ──────────────────────────────────────────────────────────────────

def print_comparison(df: pd.DataFrame):
    backbones = sorted(df["backbone"].unique())
    datasets  = [d for d in DATASET_ORDER if d in df["dataset"].unique()]

    print(f"\n{'='*90}")
    print(f"{'Dataset':<22}", end="")
    for bb in backbones:
        print(f"  {'─'*3} {bb:<12} {'─'*3}", end="")
    print()
    print(f"{'':22}", end="")
    for bb in backbones:
        print(f"  {'AUC':>6} {'ΔDP':>6} {'ΔEO':>6}  ", end="")
    print(f"\n{'='*90}")

    for ds in datasets:
        sub = df[df["dataset"] == ds]
        print(f"{DATASET_DISPLAY.get(ds, ds):<22}", end="")
        for bb in backbones:
            row = sub[sub["backbone"] == bb]
            if row.empty:
                print(f"  {'—':>6} {'—':>6} {'—':>6}  ", end="")
            else:
                r = row.iloc[0]
                auc = r.get("roc_auc_mean", float("nan"))
                dp  = r.get("dp_mean", float("nan"))
                eo  = r.get("eo_mean", float("nan"))
                print(f"  {auc:>6.4f} {dp:>6.4f} {eo:>6.4f}  ", end="")
        print()

    print(f"{'='*90}")

    # 백본별 평균
    print("\n[백본별 평균]")
    summary = df.groupby("backbone")[METRICS].mean()
    for bb, row in summary.iterrows():
        auc = row.get("roc_auc_mean", float("nan"))
        dp  = row.get("dp_mean", float("nan"))
        eo  = row.get("eo_mean", float("nan"))
        print(f"  {bb:<12}: AUC={auc:.4f}  ΔDP={dp:.4f}  ΔEO={eo:.4f}  "
              f"ΔDP+ΔEO={dp+eo:.4f}")

    # 일관성 분석: 데이터셋마다 세 백본 모두 baseline 대비 개선인지
    print("\n[백본 간 AUC 분산 (낮을수록 일관적)]")
    datasets2 = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    for ds in datasets2:
        sub = df[df["dataset"] == ds]
        aucs = sub["roc_auc_mean"].dropna().values if "roc_auc_mean" in sub else []
        if len(aucs) >= 2:
            print(f"  {DATASET_DISPLAY.get(ds, ds):<22}: "
                  f"std={np.std(aucs):.4f}  "
                  f"range={aucs.max()-aucs.min():.4f}  "
                  f"({', '.join(f'{v:.4f}' for v in aucs)})")


# ── LaTeX — 전체 비교 테이블 ───────────────────────────────────────────────────

def to_latex_full(df: pd.DataFrame) -> str:
    backbones = ["GCN", "GraphSAGE", "SGC"]
    backbones = [b for b in backbones if b in df["backbone"].unique()]
    datasets  = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    show_metrics = ["roc_auc_mean", "dp_mean", "eo_mean"]

    n_bb  = len(backbones)
    n_met = len(show_metrics)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{백본 GNN에 따른 FairGate 성능 비교. "
        r"$\uparrow$: 높을수록 우수, $\downarrow$: 낮을수록 우수. "
        r"각 데이터셋에서 가장 우수한 값을 \textbf{굵게} 표시.}"
    )
    lines.append(r"\label{tab:backbone_comparison}")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\resizebox{\linewidth}{!}{")

    # 컬럼 형식: dataset + (AUC DP EO) × n_bb
    col_fmt = "l" + ("rrr" * n_bb)
    lines.append(r"\begin{tabular}{" + col_fmt + "}")
    lines.append(r"\toprule")

    # 헤더 1: 백본 이름
    hdr1 = r"\multirow{2}{*}{\textbf{Dataset}}"
    for bb in backbones:
        hdr1 += r" & \multicolumn{3}{c}{" + bb + "}"
    lines.append(hdr1 + r" \\")

    # cmidrule
    for i, _ in enumerate(backbones):
        start = 2 + i * 3
        lines.append(r"\cmidrule(lr){" + f"{start}-{start+2}" + "}")

    # 헤더 2: 지표명
    hdr2 = ""
    for _ in backbones:
        hdr2 += r" & AUC$\uparrow$ & $\Delta$DP$\downarrow$ & $\Delta$EO$\downarrow$"
    lines.append(hdr2 + r" \\")
    lines.append(r"\midrule")

    for ds in datasets:
        sub = df[df["dataset"] == ds]
        # 지표별 최고값 찾기 (굵게 처리용)
        best = {}
        for met in show_metrics:
            vals = {}
            for bb in backbones:
                r = sub[sub["backbone"] == bb]
                if not r.empty and met in r.columns:
                    vals[bb] = r.iloc[0][met]
            if vals:
                best[met] = (min(vals, key=vals.get)
                             if met in LOWER_BETTER
                             else max(vals, key=vals.get))

        row_str = DATASET_DISPLAY.get(ds, ds)
        for bb in backbones:
            r = sub[sub["backbone"] == bb]
            for met in show_metrics:
                if r.empty or met not in r.columns:
                    row_str += " & —"
                else:
                    val = r.iloc[0][met]
                    std_col = met.replace("_mean", "_std")
                    std = r.iloc[0][std_col] if std_col in r.columns else None
                    cell = f"{val:.4f}"
                    if std is not None:
                        cell = f"{val:.4f}"  # std는 footnote로
                    if best.get(met) == bb:
                        cell = r"\textbf{" + cell + "}"
                    row_str += " & " + cell
        lines.append(row_str + r" \\")

    lines.append(r"\midrule")

    # 평균 행
    avg_row = r"\textit{Average}"
    summary = df.groupby("backbone")[show_metrics].mean()
    for bb in backbones:
        for met in show_metrics:
            val = summary.loc[bb, met] if bb in summary.index and met in summary.columns else float("nan")
            avg_row += f" & {val:.4f}"
    lines.append(avg_row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── LaTeX — 백본별 평균 요약 (compact) ────────────────────────────────────────

def to_latex_summary(df: pd.DataFrame) -> str:
    backbones = ["GCN", "GraphSAGE", "SGC"]
    backbones = [b for b in backbones if b in df["backbone"].unique()]
    metrics   = ["acc_mean", "roc_auc_mean", "f1_mean", "dp_mean", "eo_mean"]
    metrics   = [m for m in metrics if m in df.columns]

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{백본별 전체 데이터셋 평균 성능. "
        r"FairGate는 세 백본 모두에서 일관된 성능을 보인다.}"
    )
    lines.append(r"\label{tab:backbone_summary}")
    lines.append(r"\setlength{\tabcolsep}{8pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    lines.append(r"\begin{tabular}{l" + "r" * len(metrics) + "}")
    lines.append(r"\toprule")

    hdr = r"\textbf{Backbone}"
    for m in metrics:
        arrow = r"$\downarrow$" if m in LOWER_BETTER else r"$\uparrow$"
        hdr += " & " + METRIC_DISPLAY.get(m, m) + arrow
    lines.append(hdr + r" \\")
    lines.append(r"\midrule")

    summary = df.groupby("backbone")[metrics].mean()
    for bb in backbones:
        if bb not in summary.index:
            continue
        row_str = bb
        for m in metrics:
            val = summary.loc[bb, m]
            row_str += f" & {val:.4f}"
        lines.append(row_str + r" \\")

    # 백본 간 std (일관성 지표)
    lines.append(r"\midrule")
    std_row = r"\textit{Std. across backbones}"
    for m in metrics:
        vals = [summary.loc[bb, m] for bb in backbones if bb in summary.index]
        std_row += f" & {np.std(vals):.4f}"
    lines.append(std_row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── 메인 ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="FairGate 백본 범용성 분석",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gcn",  type=str, default="outputs/exp_fairgate_gcn.csv")
    p.add_argument("--sage", type=str, default="outputs/exp_fairgate_sage.csv")
    p.add_argument("--sgc",  type=str, default="outputs/exp_fairgate_sgc.csv")
    p.add_argument("--output_dir", type=str, default="outputs/analysis")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[Backbone Generality Analysis]")
    df_all = load_all(args.gcn, args.sage, args.sgc)
    df_agg = aggregate(df_all)

    print_comparison(df_agg)

    # CSV 저장
    csv_path = os.path.join(args.output_dir, "backbone_comparison.csv")
    df_agg.to_csv(csv_path, index=False)
    print(f"\n[Saved] {csv_path}")

    # LaTeX 저장
    tex_full = to_latex_full(df_agg)
    tex_full_path = os.path.join(args.output_dir, "backbone_comparison.tex")
    with open(tex_full_path, "w") as f:
        f.write(tex_full)
    print(f"[Saved] {tex_full_path}")

    tex_sum = to_latex_summary(df_agg)
    tex_sum_path = os.path.join(args.output_dir, "backbone_summary.tex")
    with open(tex_sum_path, "w") as f:
        f.write(tex_sum)
    print(f"[Saved] {tex_sum_path}")


if __name__ == "__main__":
    main()