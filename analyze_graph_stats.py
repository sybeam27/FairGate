"""
analyze_graph_stats.py — 데이터셋별 그래프 구조 특성 분석 (확장판)

측정 항목:
    기존:
        nodes, edges, avg_degree, homophily, boundary_ratio,
        deg_gap, sens_ratio, label_ratio, regime

    추가:
        local_homophily_std  : 노드별 local homophily 표준편차 (국소 이질성 분산)
        bridge_criticality   : inter-group 엣지의 구조적 중요도
                               (제거 시 연결 성분 증가 비율의 근사치)
                               = inter-group 엣지 중 "bridge-like" 비율
        inter_group_edge_ratio: 전체 엣지 중 inter-group 비율
        sens_group_size_ratio : 두 집단 크기 비율 (|G0|/|G1|)

구조→축 매핑 (첨부 문서 기준):
        homophily           → gate_budget (높을수록 넓은 gate)
        boundary_ratio      → attenuation_strength (높을수록 약한 attenuation)
        deg_gap             → fairness_budget (높을수록 강한 lambda_fair)
        local_homophily_std → phase_lag (높을수록 긴 warm_up)
        bridge_criticality  → edge_intervention (높을수록 drop 대신 scale)

출력:
    outputs/analysis/graph_stats.csv
    outputs/analysis/graph_stats.tex
    outputs/analysis/structure_axis_regression.csv  ← 신규
    outputs/analysis/structure_axis_regression.tex  ← 신규

실행:
    python analyze_graph_stats.py
    python analyze_graph_stats.py --datasets pokec_z german credit
    python analyze_graph_stats.py --fairgate_csv outputs/exp_fairgate_gcn.csv
    python analyze_graph_stats.py --with_regression  # structure→axis 회귀 포함
"""

import os, sys, argparse
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data  import get_dataset
from utils.model import _auto_config_from_graph_stats


ALL_DATASETS = [
    "pokec_z", "pokec_z_g", "pokec_n", "pokec_n_g",
    "german", "credit", "recidivism", "nba", "income",
]

DATASET_DISPLAY = {
    "pokec_z"   : "Pokec-Z",         "pokec_z_g": "Pokec-Z (gender)",
    "pokec_n"   : "Pokec-N",         "pokec_n_g": "Pokec-N (gender)",
    "german"    : "German",           "credit"   : "Credit",
    "recidivism": "Recidivism",       "nba"      : "NBA",
    "income"    : "Income",
}


# ── 핵심 통계 계산 ──────────────────────────────────────────────────────────────

def compute_stats(dataset: str) -> dict:
    data, sens_idx, x_min, x_max = get_dataset(dataset)
    data = data.cpu()

    src, dst = data.edge_index
    N = data.x.size(0)
    E = data.edge_index.size(1)
    sens = data.sens

    # ── 기본 통계 ──────────────────────────────────────────────────────────────
    deg = torch.zeros(N).scatter_add_(0, src, torch.ones(E))
    avg_deg = float(deg.mean())

    # Edge homophily
    same_edge = (sens[src] == sens[dst]).float()
    homophily = float(same_edge.mean())

    # Boundary ratio
    is_inter  = (sens[src] != sens[dst])
    has_inter = torch.zeros(N, dtype=torch.bool)
    has_inter[src[is_inter]] = True
    boundary_ratio = float(has_inter.float().mean())

    # Degree gap
    d0 = float(deg[sens == 0].mean()) if (sens == 0).any() else 0.0
    d1 = float(deg[sens == 1].mean()) if (sens == 1).any() else 0.0
    deg_gap = abs(d0 - d1) / (d0 + d1 + 1e-8)

    # Sensitive attribute ratio
    sens_ratio  = float((sens == 0).float().mean())
    label_ratio = float((data.y == 1).float().mean()) if hasattr(data, 'y') else 0.5

    # ── 신규 통계 1: local_homophily_std ───────────────────────────────────────
    # 각 노드의 이웃 중 동일 집단 비율 = local homophily
    # 이 값의 표준편차가 크면 "국소적으로 이질성이 들쭉날쭉"
    same_count  = torch.zeros(N).scatter_add_(0, src, same_edge)
    local_h     = same_count / (deg + 1e-8)
    lh_std      = float(local_h.std())
    lh_mean     = float(local_h.mean())

    # ── 신규 통계 2: inter_group_edge_ratio ────────────────────────────────────
    inter_edge_ratio = float(is_inter.float().mean())

    # ── 신규 통계 3: bridge_criticality ────────────────────────────────────────
    # 정확한 bridge 계산은 O(E²)이라 근사치 사용:
    # "degree가 낮은 inter-group 엣지의 비율"
    # → 두 끝점의 min_degree가 작을수록 bridge에 가까움
    # → min_degree <= 2인 inter-group 엣지 비율로 근사
    if is_inter.any():
        inter_src = src[is_inter]
        inter_dst = dst[is_inter]
        min_deg_inter = torch.minimum(deg[inter_src], deg[inter_dst])
        # min_degree <= median_degree * 0.5 인 inter-group 엣지 비율
        med_deg = float(deg.median())
        bridge_like = (min_deg_inter <= max(med_deg * 0.5, 2.0)).float().mean()
        bridge_criticality = float(bridge_like)
    else:
        bridge_criticality = 0.0

    # ── 신규 통계 4: sens_group_size_ratio ────────────────────────────────────
    n0 = int((sens == 0).sum())
    n1 = int((sens == 1).sum())
    group_size_ratio = n0 / (n1 + 1e-8)

    # ── Regime 분류 ────────────────────────────────────────────────────────────
    regime_info = _auto_config_from_graph_stats(boundary_ratio, deg_gap)
    regime      = regime_info["regime"]

    return {
        "dataset"            : dataset,
        "display_name"       : DATASET_DISPLAY[dataset],
        "nodes"              : N,
        "edges"              : E,
        "avg_degree"         : round(avg_deg, 2),
        # 기존 4개 핵심 구조 변수
        "homophily"          : round(homophily, 4),
        "boundary_ratio"     : round(boundary_ratio, 4),
        "deg_gap"            : round(deg_gap, 4),
        "sens_ratio"         : round(sens_ratio, 4),
        # 신규 4개
        "local_homophily_std": round(lh_std, 4),
        "inter_edge_ratio"   : round(inter_edge_ratio, 4),
        "bridge_criticality" : round(bridge_criticality, 4),
        "group_size_ratio"   : round(group_size_ratio, 4),
        # 보조
        "label_ratio"        : round(label_ratio, 4),
        "regime"             : regime,
    }


# ── FairGate 결과 연결 ─────────────────────────────────────────────────────────

def merge_fairgate_results(df: pd.DataFrame, fairgate_csv: str) -> pd.DataFrame:
    if not fairgate_csv or not os.path.exists(fairgate_csv):
        print(f"[INFO] fairgate_csv not found — skipping merge")
        return df
    fg = pd.read_csv(fairgate_csv)
    if "model" in fg.columns:
        fg = fg[fg["model"] == "FairGate"]
    if {"dataset","dp_mean","eo_mean"}.issubset(fg.columns):
        agg = fg.groupby("dataset")[["roc_auc_mean","dp_mean","eo_mean"]].mean().reset_index()
        agg["dp_eo_sum"] = (agg["dp_mean"] + agg["eo_mean"]).round(4)
        df = df.merge(agg, on="dataset", how="left")
        print(f"[INFO] FairGate 결과 연결: {agg['dataset'].nunique()} 데이터셋")
    return df


# ── 구조→축 회귀 분석 ──────────────────────────────────────────────────────────

def structure_axis_regression(stats_df: pd.DataFrame,
                               fairgate_configs: dict,
                               output_dir: str):
    """
    각 데이터셋의 최적 하이퍼파라미터(4축)와 그래프 구조 변수 간 관계를 분석.

    4개 축:
        gate_budget      = 1 - sbrs_quantile  (높을수록 넓은 gate)
        attenuation      = struct_drop
        fairness_budget  = lambda_fair
        phase_lag        = warm_up
    """
    if not fairgate_configs:
        print("[INFO] fairgate_configs 없음 — 회귀 분석 건너뜀")
        return

    struct_vars = ["homophily","boundary_ratio","deg_gap",
                   "local_homophily_std","bridge_criticality","inter_edge_ratio"]
    axis_vars   = ["gate_budget","attenuation","fairness_budget","phase_lag"]

    rows = []
    for ds, cfg in fairgate_configs.items():
        s = stats_df[stats_df["dataset"] == ds]
        if s.empty:
            continue
        s = s.iloc[0]
        rows.append({
            "dataset"        : ds,
            "regime"         : s.get("regime","?"),
            # 구조 변수
            **{v: s.get(v, np.nan) for v in struct_vars},
            # 4개 축 (의미 있는 단위로 변환)
            "gate_budget"    : 1.0 - cfg.get("sbrs_quantile", 0.7),
            "attenuation"    : cfg.get("struct_drop", 0.5),
            "fairness_budget": cfg.get("lambda_fair", 0.1),
            "phase_lag"      : cfg.get("warm_up", 200),
        })

    reg_df = pd.DataFrame(rows)
    if len(reg_df) < 3:
        print("[WARN] 데이터셋 수 부족 — 회귀 건너뜀")
        return

    # 상관 분석
    print(f"\n{'='*70}")
    print("구조 변수 ↔ 최적 축 상관 (Pearson r)")
    print(f"{'='*70}")

    header = f"{'구조변수':<22}"
    for ax in axis_vars:
        header += f"  {ax:>16}"
    print(header)
    print("-"*70)

    corr_rows = []
    for sv in struct_vars:
        row_str = f"{sv:<22}"
        corr_row = {"struct_var": sv}
        for ax in axis_vars:
            valid = reg_df[[sv, ax]].dropna()
            if len(valid) >= 3:
                r = float(np.corrcoef(valid[sv], valid[ax])[0,1])
            else:
                r = float("nan")
            mark = "★★" if abs(r) > 0.6 else ("★" if abs(r) > 0.4 else "  ")
            row_str += f"  {r:>+8.3f}{mark:>6}"
            corr_row[ax] = round(r, 4)
        print(row_str)
        corr_rows.append(corr_row)

    corr_df = pd.DataFrame(corr_rows)
    csv_path = os.path.join(output_dir, "structure_axis_regression.csv")
    corr_df.to_csv(csv_path, index=False)
    print(f"\n[Saved] {csv_path}")

    # 패턴 요약 출력
    print(f"\n{'='*70}")
    print("패턴 요약 (|r| > 0.4인 관계만)")
    print(f"{'='*70}")
    mapping = {
        "gate_budget"    : "gate budget  (1−sbrs_quantile)",
        "attenuation"    : "attenuation  (struct_drop)",
        "fairness_budget": "fairness budget (lambda_fair)",
        "phase_lag"      : "phase lag    (warm_up)",
    }
    for ax in axis_vars:
        strong = [(sv, corr_df.loc[corr_df["struct_var"]==sv, ax].values[0])
                  for sv in struct_vars
                  if not np.isnan(corr_df.loc[corr_df["struct_var"]==sv, ax].values[0])
                  and abs(corr_df.loc[corr_df["struct_var"]==sv, ax].values[0]) > 0.4]
        if strong:
            print(f"\n  [{mapping[ax]}]")
            for sv, r in sorted(strong, key=lambda x: abs(x[1]), reverse=True):
                direction = "↑" if r > 0 else "↓"
                print(f"    {sv:<25}: r={r:+.3f}  "
                      f"({sv} 높을수록 {direction} {ax})")

    # LaTeX 저장
    _save_regression_latex(corr_df, struct_vars, axis_vars, output_dir)


def _save_regression_latex(corr_df, struct_vars, axis_vars, output_dir):
    ax_labels = {
        "gate_budget"    : r"Gate budget",
        "attenuation"    : r"Attenuation",
        "fairness_budget": r"Fairness budget",
        "phase_lag"      : r"Phase lag",
    }
    sv_labels = {
        "homophily"          : r"$h$",
        "boundary_ratio"     : r"$r_{\mathrm{bnd}}$",
        "deg_gap"            : r"$\delta_{\mathrm{deg}}$",
        "local_homophily_std": r"$\sigma_h^{\mathrm{local}}$",
        "bridge_criticality" : r"$\beta_{\mathrm{crit}}$",
        "inter_edge_ratio"   : r"$r_{\mathrm{inter}}$",
    }
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{그래프 구조 변수와 최적 하이퍼파라미터 축 간 Pearson 상관계수. "
        r"$|\cdot| > 0.6$: ★★, $|\cdot| > 0.4$: ★.}",
        r"\label{tab:structure_axis}",
        r"\setlength{\tabcolsep}{7pt}", r"\renewcommand{\arraystretch}{1.2}",
        r"\begin{tabular}{l" + "r"*len(axis_vars) + "}",
        r"\toprule",
        r"\textbf{구조 변수} & " + " & ".join(
            r"\textbf{" + ax_labels[a] + "}" for a in axis_vars) + r" \\",
        r"\midrule",
    ]
    for sv in struct_vars:
        row = sv_labels.get(sv, sv)
        for ax in axis_vars:
            r = corr_df.loc[corr_df["struct_var"]==sv, ax].values[0]
            if np.isnan(r):
                row += " & —"
            else:
                mark = r"$^{\star\star}$" if abs(r) > 0.6 else (
                       r"$^{\star}$"      if abs(r) > 0.4 else "")
                row += f" & {r:+.3f}{mark}"
        lines.append(row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = os.path.join(output_dir, "structure_axis_regression.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[Saved] {path}")


# ── 콘솔 요약 출력 ─────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print(f"\n{'='*95}")
    print(f"{'Dataset':<22} {'h':>6} {'bnd':>6} {'deg_g':>6} "
          f"{'lh_std':>7} {'bridge':>7} {'inter_e':>8}  Regime")
    print(f"{'='*95}")
    for _, row in df.iterrows():
        print(f"{row['display_name']:<22} "
              f"{row['homophily']:>6.4f} "
              f"{row['boundary_ratio']:>6.4f} "
              f"{row['deg_gap']:>6.4f} "
              f"{row.get('local_homophily_std', float('nan')):>7.4f} "
              f"{row.get('bridge_criticality', float('nan')):>7.4f} "
              f"{row.get('inter_edge_ratio', float('nan')):>8.4f}  "
              f"{row['regime']}")
    print(f"{'='*95}")

    print("\n[구조 변수 기준 정렬 — boundary_ratio]")
    for _, r in df.sort_values("boundary_ratio").iterrows():
        print(f"  {r['display_name']:<24}: bnd={r['boundary_ratio']:.4f} "
              f"lh_std={r.get('local_homophily_std', float('nan')):.4f} "
              f"bridge={r.get('bridge_criticality', float('nan')):.4f} "
              f"regime={r['regime']}")


# ── LaTeX 출력 ─────────────────────────────────────────────────────────────────

def to_latex(df: pd.DataFrame, has_fairgate: bool) -> str:
    cols_base = ["display_name","nodes","edges","homophily","boundary_ratio",
                 "deg_gap","local_homophily_std","bridge_criticality","regime"]
    cols_fg   = ["roc_auc_mean","dp_mean","eo_mean"] if has_fairgate else []
    cols = [c for c in cols_base + cols_fg if c in df.columns]

    header_map = {
        "display_name"       : "Dataset",
        "nodes"              : r"$|V|$",
        "edges"              : r"$|E|$",
        "homophily"          : r"$h$",
        "boundary_ratio"     : r"$r_{\mathrm{bnd}}$",
        "deg_gap"            : r"$\delta_{\mathrm{deg}}$",
        "local_homophily_std": r"$\sigma_h$",
        "bridge_criticality" : r"$\beta_{\mathrm{crit}}$",
        "regime"             : "Regime",
        "roc_auc_mean"       : "AUC",
        "dp_mean"            : r"$\Delta\mathrm{DP}$",
        "eo_mean"            : r"$\Delta\mathrm{EO}$",
    }
    col_fmt = "".join("l" if c in ("display_name","regime") else "r" for c in cols)

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{데이터셋별 그래프 구조 특성. "
        r"$h$: edge homophily, "
        r"$r_{\mathrm{bnd}}$: boundary ratio, "
        r"$\delta_{\mathrm{deg}}$: degree gap, "
        r"$\sigma_h$: local homophily 표준편차, "
        r"$\beta_{\mathrm{crit}}$: bridge criticality.}",
        r"\label{tab:graph_stats}",
        r"\setlength{\tabcolsep}{5pt}", r"\renewcommand{\arraystretch}{1.15}",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{" + col_fmt + "}",
        r"\toprule",
        " & ".join(header_map.get(c,c) for c in cols) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row.get(c, float('nan'))
            if c in ("nodes","edges"):
                cells.append(f"{int(v):,}")
            elif c in ("display_name","regime"):
                cells.append(str(v))
            elif pd.isna(v):
                cells.append("—")
            else:
                cells.append(f"{float(v):.4f}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


# ── 메인 ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="FairGate 데이터셋 그래프 구조 특성 분석 (확장판)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--datasets",       nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    p.add_argument("--fairgate_csv",   type=str,  default="outputs/exp_fairgate_gcn.csv")
    p.add_argument("--output_dir",     type=str,  default="outputs/analysis")
    p.add_argument("--with_regression",action="store_true",
                   help="구조→축 회귀 분석 포함 (FAIRGATE_CONFIGS 필요)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n[Graph Stats Analysis]  datasets={args.datasets}\n")

    rows = []
    for ds in args.datasets:
        print(f"  Computing {ds}...", end=" ", flush=True)
        try:
            row = compute_stats(ds)
            rows.append(row)
            print(f"OK  h={row['homophily']:.4f}  "
                  f"bnd={row['boundary_ratio']:.4f}  "
                  f"lh_std={row['local_homophily_std']:.4f}  "
                  f"bridge={row['bridge_criticality']:.4f}  "
                  f"regime={row['regime']}")
        except Exception as e:
            print(f"FAIL — {e}")

    df = pd.DataFrame(rows)
    df = merge_fairgate_results(df, args.fairgate_csv)
    print_summary(df)

    # CSV 저장
    csv_path = os.path.join(args.output_dir, "graph_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved] {csv_path}")

    # LaTeX 저장
    tex_path = os.path.join(args.output_dir, "graph_stats.tex")
    with open(tex_path, "w") as f:
        f.write(to_latex(df, "dp_mean" in df.columns))
    print(f"[Saved] {tex_path}")

    # 구조→축 회귀 분석
    if args.with_regression:
        # FAIRGATE_CONFIGS를 직접 여기서 정의하거나 임포트
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
        structure_axis_regression(df, FAIRGATE_CONFIGS, args.output_dir)

    # 상관관계 분석
    if "dp_eo_sum" in df.columns and df["dp_eo_sum"].notna().sum() >= 3:
        print("\n[구조 특성 vs FairGate ΔDP+ΔEO 상관]")
        for col in ["homophily","boundary_ratio","deg_gap",
                    "local_homophily_std","bridge_criticality"]:
            if col in df.columns:
                r = df[col].corr(df["dp_eo_sum"])
                print(f"  {col:<25}: r={r:+.4f}")


if __name__ == "__main__":
    main()