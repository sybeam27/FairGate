"""
analyze_graph_stats.py — 데이터셋별 그래프 구조 특성 분석

측정 항목:
    - nodes, edges, avg_degree
    - edge_homophily       : 동일 집단 간 엣지 비율 (h ∈ [0,1])
    - boundary_ratio       : inter-group 이웃이 있는 노드 비율
    - deg_gap              : 두 집단 평균 degree의 정규화된 차이
    - sens_ratio           : 민감속성 집단 비율 (0집단 비율)
    - regime               : _auto_config_from_graph_stats 기반 자동 분류
    - fairness_improvement : exp_fairgate.csv 있으면 ΔDP+ΔEO 개선량 자동 연결

출력:
    outputs/graph_stats.csv   — 전체 통계 테이블
    outputs/graph_stats.tex   — LaTeX 표 (논문 삽입용)

실행:
    python analyze_graph_stats.py
    python analyze_graph_stats.py --datasets pokec_z german nba
    python analyze_graph_stats.py --fairgate_csv outputs/exp_fairgate.csv
"""

import os
import sys
import argparse

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data  import get_dataset
from utils.model import _auto_config_from_graph_stats


# ── 대상 데이터셋 ──────────────────────────────────────────────────────────────
ALL_DATASETS = [
    "pokec_z", "pokec_z_g",
    "pokec_n", "pokec_n_g",
    "german", "credit", "recidivism",
    "nba", "income",
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


# ── 통계 계산 ──────────────────────────────────────────────────────────────────

def compute_stats(dataset: str) -> dict:
    data, sens_idx, x_min, x_max = get_dataset(dataset)
    data = data.cpu()

    src, dst = data.edge_index
    N = data.x.size(0)
    E = data.edge_index.size(1)
    sens = data.sens

    # 기본 통계
    deg = torch.zeros(N).scatter_add_(0, src, torch.ones(E))
    avg_deg = float(deg.mean().item())

    # Edge homophily
    same = (sens[src] == sens[dst]).float()
    homophily = float(same.mean().item())

    # Boundary ratio
    is_inter  = (sens[src] != sens[dst])
    has_inter = torch.zeros(N, dtype=torch.bool)
    has_inter[src[is_inter]] = True
    boundary_ratio = float(has_inter.float().mean().item())

    # Degree gap
    d0 = float(deg[sens == 0].mean().item())
    d1 = float(deg[sens == 1].mean().item())
    deg_gap = abs(d0 - d1) / (d0 + d1 + 1e-8)

    # Sensitive attribute ratio (집단 0 비율)
    sens_ratio = float((sens == 0).float().mean().item())

    # Regime
    regime_info = _auto_config_from_graph_stats(boundary_ratio, deg_gap)
    regime = regime_info["regime"]

    # Label imbalance
    labels = data.y.float()
    label_ratio = float((labels == 1).float().mean().item())

    return {
        "dataset"       : dataset,
        "display_name"  : DATASET_DISPLAY[dataset],
        "nodes"         : N,
        "edges"         : E,
        "avg_degree"    : round(avg_deg, 2),
        "homophily"     : round(homophily, 4),
        "boundary_ratio": round(boundary_ratio, 4),
        "deg_gap"       : round(deg_gap, 4),
        "sens_ratio"    : round(sens_ratio, 4),
        "label_ratio"   : round(label_ratio, 4),
        "regime"        : regime,
    }


# ── FairGate 결과 연결 ─────────────────────────────────────────────────────────

def merge_fairgate_results(stats_df: pd.DataFrame, fairgate_csv: str) -> pd.DataFrame:
    if not os.path.exists(fairgate_csv):
        print(f"[INFO] fairgate_csv not found: {fairgate_csv} — skipping merge")
        return stats_df

    fg = pd.read_csv(fairgate_csv)

    # FairGate 행만 필터
    if "model" in fg.columns:
        fg = fg[fg["model"] == "FairGate"]

    # 데이터셋별 DP, EO 평균
    if {"dataset", "dp_mean", "eo_mean"}.issubset(fg.columns):
        fg_agg = (fg.groupby("dataset")[["dp_mean", "eo_mean"]]
                    .mean()
                    .reset_index())
        fg_agg["dp_eo_sum"] = (fg_agg["dp_mean"] + fg_agg["eo_mean"]).round(4)
        stats_df = stats_df.merge(fg_agg[["dataset", "dp_mean", "eo_mean", "dp_eo_sum"]],
                                  on="dataset", how="left")
        print(f"[INFO] Merged FairGate results for "
              f"{fg_agg['dataset'].nunique()} datasets")
    else:
        print(f"[WARN] Expected columns (dataset, dp_mean, eo_mean) not found in {fairgate_csv}")

    return stats_df


# ── LaTeX 출력 ─────────────────────────────────────────────────────────────────

def to_latex(df: pd.DataFrame, has_fairgate: bool) -> str:
    cols_base = [
        "display_name", "nodes", "edges", "avg_degree",
        "homophily", "boundary_ratio", "deg_gap", "sens_ratio", "regime",
    ]
    cols_fg = ["dp_mean", "eo_mean", "dp_eo_sum"] if has_fairgate else []
    cols = [c for c in cols_base + cols_fg if c in df.columns]

    header_map = {
        "display_name"  : "Dataset",
        "nodes"         : r"$|V|$",
        "edges"         : r"$|E|$",
        "avg_degree"    : r"Avg. deg.",
        "homophily"     : r"$h$",
        "boundary_ratio": r"$r_{\mathrm{bnd}}$",
        "deg_gap"       : r"$\delta_{\mathrm{deg}}$",
        "sens_ratio"    : r"$\rho_0$",
        "regime"        : "Regime",
        "dp_mean"       : r"$\Delta\mathrm{DP}$",
        "eo_mean"       : r"$\Delta\mathrm{EO}$",
        "dp_eo_sum"     : r"$\sum$",
    }

    col_fmt = "l" + "r" * (len(cols) - 2) + ("l" if "regime" in cols else "") + \
              ("r" * len(cols_fg))
    col_fmt = "l" + "".join(
        "l" if c in ("display_name", "regime") else "r" for c in cols[1:]
    )

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    caption = (
        r"\caption{데이터셋별 그래프 구조 특성 및 FairGate 공정성 결과. "
        r"$h$: edge homophily, "
        r"$r_{\mathrm{bnd}}$: boundary ratio, "
        r"$\delta_{\mathrm{deg}}$: degree gap, "
        r"$\rho_0$: 민감속성 집단 0 비율. "
        r"Regime은 $r_{\mathrm{bnd}}$와 $\delta_{\mathrm{deg}}$ 기반 자동 분류.}"
    )
    lines.append(caption)
    lines.append(r"\label{tab:graph_stats}")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\resizebox{\linewidth}{!}{")
    lines.append(r"\begin{tabular}{" + col_fmt + "}")
    lines.append(r"\toprule")

    header = " & ".join(header_map.get(c, c) for c in cols) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in ("nodes", "edges"):
                cells.append(f"{int(v):,}")
            elif c in ("avg_degree",):
                cells.append(f"{v:.1f}")
            elif c in ("homophily", "boundary_ratio", "deg_gap",
                       "sens_ratio", "label_ratio",
                       "dp_mean", "eo_mean", "dp_eo_sum"):
                cells.append(f"{v:.4f}" if pd.notna(v) else "—")
            else:
                cells.append(str(v))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── 콘솔 요약 출력 ─────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print(f"\n{'='*80}")
    print(f"{'Dataset':<20} {'Nodes':>8} {'Edges':>10} {'AvgDeg':>7} "
          f"{'Homoph':>7} {'BndRatio':>9} {'DegGap':>7} "
          f"{'SensR':>6}  Regime")
    print(f"{'='*80}")
    for _, row in df.iterrows():
        print(
            f"{row['display_name']:<20} "
            f"{int(row['nodes']):>8,} "
            f"{int(row['edges']):>10,} "
            f"{row['avg_degree']:>7.1f} "
            f"{row['homophily']:>7.4f} "
            f"{row['boundary_ratio']:>9.4f} "
            f"{row['deg_gap']:>7.4f} "
            f"{row['sens_ratio']:>6.4f}  "
            f"{row['regime']}"
        )
    print(f"{'='*80}\n")

    # 패턴 요약
    print("[Regime 분포]")
    print(df.groupby("regime")["display_name"]
            .apply(list)
            .to_string())

    print("\n[homophily 기준 정렬]")
    for _, r in df.sort_values("homophily").iterrows():
        print(f"  {r['display_name']:<22}: h={r['homophily']:.4f}  "
              f"bnd={r['boundary_ratio']:.4f}  deg_gap={r['deg_gap']:.4f}")


# ── 메인 ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="FairGate 데이터셋 그래프 구조 특성 분석",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                   choices=ALL_DATASETS, help="분석할 데이터셋")
    p.add_argument("--fairgate_csv", type=str, default="outputs/exp_fairgate.csv",
                   help="FairGate 실험 결과 CSV (있으면 ΔDP/ΔEO 자동 연결)")
    p.add_argument("--output_dir", type=str, default="outputs/analysis",
                   help="결과 저장 디렉토리")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n[Graph Stats Analysis]  datasets={args.datasets}\n")

    rows = []
    for ds in args.datasets:
        print(f"  Loading {ds}...", end=" ", flush=True)
        try:
            row = compute_stats(ds)
            rows.append(row)
            print(f"OK  (N={row['nodes']:,}  h={row['homophily']:.4f}  "
                  f"regime={row['regime']})")
        except Exception as e:
            print(f"FAIL — {e}")

    df = pd.DataFrame(rows)

    # FairGate 결과 연결
    df = merge_fairgate_results(df, args.fairgate_csv)

    # 콘솔 출력
    print_summary(df)

    # CSV 저장
    csv_path = os.path.join(args.output_dir, "graph_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved] {csv_path}")

    # LaTeX 저장
    has_fg = "dp_mean" in df.columns
    tex = to_latex(df, has_fg)
    tex_path = os.path.join(args.output_dir, "graph_stats.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"[Saved] {tex_path}")

    # 상관관계 분석 (FairGate 결과 있을 때)
    if has_fg and df["dp_eo_sum"].notna().sum() >= 3:
        print("\n[상관관계 분석: 그래프 특성 vs FairGate ΔDP+ΔEO]")
        for col in ["homophily", "boundary_ratio", "deg_gap", "sens_ratio"]:
            if col in df.columns:
                corr = df[col].corr(df["dp_eo_sum"])
                print(f"  {col:<18}: r = {corr:+.4f}")


if __name__ == "__main__":
    main()
