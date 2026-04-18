"""
analyze_hparam.py — FairGate 하이퍼파라미터 탐색 결과 분석

run_hparam_search.py 실행 후 생성된 outputs/hparam/hparam_*.csv를 읽어
다음을 수행한다:

  1. 데이터셋별 Top-5 config (콘솔 출력)
  2. 파라미터 민감도 분석 (DP+EO range, AUC range)
  3. 파라미터 ↔ 그래프 특성 상관 분석
  4. Regime별 파라미터 패턴 요약
  5. 최종 FAIRGATE_CONFIGS 추천 (baseline 비교 기반)
  6. LaTeX 테이블 저장 (탐색 범위 / 최종 설정 / 민감도)
  7. 시각화 (matplotlib PDF)

출력:
  outputs/analysis/hparam_top5.txt
  outputs/analysis/hparam_sensitivity.csv
  outputs/analysis/hparam_sensitivity.tex
  outputs/analysis/hparam_final.tex
  outputs/analysis/hparam_plot.pdf

실행:
  python analyze_hparam.py
  python analyze_hparam.py --hparam_dir outputs/hparam
  python analyze_hparam.py --baseline_csv outputs/compare/exp_baselines.csv
  python analyze_hparam.py --rep_datasets pokec_z german credit pokec_z_g
"""

import os
import glob
import argparse
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ── 상수 ───────────────────────────────────────────────────────────────────────

HPARAMS = ["lambda_fair", "sbrs_quantile", "struct_drop", "warm_up"]

HPARAM_LABELS = {
    "lambda_fair"  : r"$\lambda_{\mathrm{fair}}$",
    "sbrs_quantile": r"$q_{\mathrm{sbrs}}$",
    "struct_drop"  : r"$p_{\mathrm{struct}}$",
    "warm_up"      : r"$T_{\mathrm{warm}}$",
}

GRAPH_STATS = {
    "pokec_z"   : dict(h=0.953, bnd=0.369, deg_gap=0.083, regime="clustered"),
    "pokec_z_g" : dict(h=0.479, bnd=0.889, deg_gap=0.024, regime="mixed"),
    "pokec_n"   : dict(h=0.956, bnd=0.307, deg_gap=0.056, regime="clustered"),
    "pokec_n_g" : dict(h=0.489, bnd=0.886, deg_gap=0.011, regime="mixed"),
    "german"    : dict(h=0.809, bnd=0.970, deg_gap=0.049, regime="saturated"),
    "credit"    : dict(h=0.960, bnd=0.677, deg_gap=0.315, regime="degree-skewed"),
    "recidivism": dict(h=0.536, bnd=0.998, deg_gap=0.023, regime="saturated"),
    "nba"       : dict(h=0.729, bnd=0.978, deg_gap=0.096, regime="saturated"),
    "income"    : dict(h=0.884, bnd=0.334, deg_gap=0.123, regime="clustered"),
}

DATASET_DISPLAY = {
    "pokec_z"   : "Pokec-Z",       "pokec_z_g": "Pokec-Z (g)",
    "pokec_n"   : "Pokec-N",       "pokec_n_g": "Pokec-N (g)",
    "german"    : "German",         "credit"   : "Credit",
    "recidivism": "Recidivism",     "nba"      : "NBA",
    "income"    : "Income",
}

DATASET_ORDER = [
    "pokec_z", "pokec_z_g", "pokec_n", "pokec_n_g",
    "german", "credit", "recidivism", "nba", "income",
]

REGIME_COLORS = {
    "clustered"     : "#3B82F6",
    "mixed"         : "#8B5CF6",
    "saturated"     : "#EF4444",
    "degree-skewed" : "#F59E0B",
}


# ── 데이터 로드 ────────────────────────────────────────────────────────────────

def load_hparam_data(hparam_dir: str) -> pd.DataFrame:
    pattern = os.path.join(hparam_dir, "hparam_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"hparam_*.csv 파일 없음: {hparam_dir}")

    dfs = []
    for f in sorted(files):
        df = pd.read_csv(f)
        if "fair_sum" not in df.columns:
            df["fair_sum"] = df["dp_mean"] + df["eo_mean"]
        if "balance_score" not in df.columns:
            df["balance_score"] = df["acc_mean"] - df["dp_mean"] - df["eo_mean"]
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[Load] {len(files)}개 파일, {len(df_all)} rows, "
          f"데이터셋: {sorted(df_all['dataset'].unique())}")
    return df_all


def load_baseline(baseline_csv: str | None) -> dict:
    """데이터셋별 baseline 평균 AUC / best DP+EO 반환"""
    if baseline_csv is None or not os.path.exists(baseline_csv):
        return {}
    bl = pd.read_csv(baseline_csv)
    bl["fair"] = bl["dp_mean"] + bl["eo_mean"]
    result = {}
    for ds, grp in bl.groupby("dataset"):
        # EDITS income 이상치 제외
        sub = grp if ds != "income" else grp[grp["model"] != "EDITS"]
        result[ds] = {
            "mean_auc" : sub["roc_auc_mean"].mean(),
            "best_auc" : sub["roc_auc_mean"].max(),
            "best_fair": sub["fair"].min(),
        }
    return result


# ── 최적 config 선정 ───────────────────────────────────────────────────────────

def select_best(df: pd.DataFrame, baseline: dict, ds: str) -> pd.Series:
    """
    baseline이 있으면: AUC >= baseline 평균 조건 내 DP+EO 최소
    baseline 없으면: balance_score 최대
    """
    if ds in baseline:
        thr = baseline[ds]["mean_auc"]
        cands = df[df["roc_auc_mean"] >= thr]
        if cands.empty:
            cands = df
        return cands.loc[cands["fair_sum"].idxmin()]
    return df.loc[df["balance_score"].idxmax()]


# ── 콘솔 출력 ──────────────────────────────────────────────────────────────────

def print_top5(df_all: pd.DataFrame, baseline: dict, output_dir: str):
    lines = []
    for ds in DATASET_ORDER:
        sub = df_all[df_all["dataset"] == ds]
        if sub.empty:
            continue
        sub = sub.copy()
        sub["fair_sum"] = sub["dp_mean"] + sub["eo_mean"]
        sub["balance_score"] = sub["acc_mean"] - sub["dp_mean"] - sub["eo_mean"]
        top5 = sub.nlargest(5, "balance_score")[
            HPARAMS + ["roc_auc_mean", "dp_mean", "eo_mean", "fair_sum", "balance_score"]
        ]
        best = select_best(sub, baseline, ds)

        header = f"\n[{DATASET_DISPLAY.get(ds, ds)}]"
        if ds in baseline:
            bl = baseline[ds]
            header += (f"  BL mean_AUC={bl['mean_auc']:.4f}  "
                       f"BL best_fair={bl['best_fair']:.4f}")
        lines.append(header)
        lines.append(f"  Top-5 (balance_score = acc - dp - eo):")
        lines.append(top5.to_string(index=False))
        lines.append(f"\n  → 채택 config:")
        lines.append(f"     AUC={best['roc_auc_mean']:.4f}  "
                     f"DP={best['dp_mean']:.4f}  EO={best['eo_mean']:.4f}  "
                     f"DP+EO={best['fair_sum']:.4f}")
        cfg_str = "  ".join(f"{p}={best[p]}" for p in HPARAMS)
        lines.append(f"     {cfg_str}")

    text = "\n".join(lines)
    print(text)
    out = os.path.join(output_dir, "hparam_top5.txt")
    with open(out, "w") as f:
        f.write(text)
    print(f"\n[Saved] {out}")


# ── 민감도 분석 ────────────────────────────────────────────────────────────────

def compute_sensitivity(df_all: pd.DataFrame, rep_datasets: list) -> pd.DataFrame:
    rows = []
    for ds in rep_datasets:
        sub = df_all[df_all["dataset"] == ds].copy()
        if sub.empty:
            continue
        sub["fair_sum"] = sub["dp_mean"] + sub["eo_mean"]
        gs = GRAPH_STATS.get(ds, {})
        for p in HPARAMS:
            grp_fair = sub.groupby(p)["fair_sum"].mean()
            grp_auc  = sub.groupby(p)["roc_auc_mean"].mean()
            rows.append({
                "dataset"   : ds,
                "display"   : DATASET_DISPLAY.get(ds, ds),
                "regime"    : gs.get("regime", "?"),
                "h"         : gs.get("h", None),
                "bnd"       : gs.get("bnd", None),
                "deg_gap"   : gs.get("deg_gap", None),
                "param"     : p,
                "fair_range": round(grp_fair.max() - grp_fair.min(), 4),
                "auc_range" : round(grp_auc.max()  - grp_auc.min(),  4),
                "best_val"  : grp_fair.idxmin(),
                "worst_val" : grp_fair.idxmax(),
            })
    return pd.DataFrame(rows)


def print_sensitivity(sens: pd.DataFrame):
    print(f"\n{'='*85}")
    print("파라미터 민감도 (DP+EO range = 해당 파라미터 변화 시 최대–최소 차이)")
    print(f"{'='*85}")
    pivot = sens.pivot_table(index="display", columns="param",
                             values="fair_range", aggfunc="first")
    pivot = pivot.reindex(columns=HPARAMS)
    pivot.columns = [HPARAM_LABELS.get(c, c) for c in pivot.columns]
    print(pivot.to_string())

    print(f"\n{'─'*60}")
    print("데이터셋별 최민감 파라미터:")
    for ds in sens["dataset"].unique():
        sub = sens[sens["dataset"] == ds]
        most = sub.loc[sub["fair_range"].idxmax()]
        print(f"  {DATASET_DISPLAY.get(ds, ds):<18}: "
              f"{HPARAM_LABELS.get(most['param'], most['param'])} "
              f"(range={most['fair_range']:.4f})")


# ── 상관 분석 ──────────────────────────────────────────────────────────────────

def compute_correlations(df_all: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    """데이터셋별 최적 config ↔ 그래프 특성 상관"""
    rows = []
    for ds in DATASET_ORDER:
        sub = df_all[df_all["dataset"] == ds].copy()
        if sub.empty:
            continue
        sub["fair_sum"] = sub["dp_mean"] + sub["eo_mean"]
        sub["balance_score"] = sub["acc_mean"] - sub["dp_mean"] - sub["eo_mean"]
        best = select_best(sub, baseline, ds)
        gs   = GRAPH_STATS.get(ds, {})
        rows.append({
            "dataset"   : ds,
            "h"         : gs.get("h"),
            "bnd"       : gs.get("bnd"),
            "deg_gap"   : gs.get("deg_gap"),
            "regime"    : gs.get("regime", "?"),
            **{p: best[p] for p in HPARAMS},
            "our_auc"   : best["roc_auc_mean"],
            "our_fair"  : best["fair_sum"],
        })

    df_cfg = pd.DataFrame(rows)

    print(f"\n{'='*60}")
    print("파라미터 ↔ 그래프 특성 상관 (Pearson r)")
    print(f"{'='*60}")
    graph_feats = ["h", "bnd", "deg_gap"]
    for p in HPARAMS:
        for gf in graph_feats:
            r = df_cfg[p].corr(df_cfg[gf])
            mark = " ★" if abs(r) > 0.5 else ""
            print(f"  {HPARAM_LABELS.get(p,p):<25} vs {gf:<10}: r={r:+.3f}{mark}")

    print(f"\n{'─'*60}")
    print("Regime별 파라미터 중앙값:")
    for regime, grp in df_cfg.groupby("regime"):
        vals = "  ".join(f"{p}={grp[p].median():.2f}" for p in HPARAMS)
        print(f"  {regime:<15}: {vals}")

    return df_cfg


# ── FAIRGATE_CONFIGS 추천 출력 ─────────────────────────────────────────────────

def print_recommended_configs(df_all: pd.DataFrame, baseline: dict):
    current = {
        "pokec_z"   : dict(lambda_fair=0.05, sbrs_quantile=0.7, struct_drop=0.5, warm_up=200),
        "pokec_z_g" : dict(lambda_fair=0.10, sbrs_quantile=0.7, struct_drop=0.7, warm_up=100),
        "pokec_n"   : dict(lambda_fair=0.20, sbrs_quantile=0.7, struct_drop=0.5, warm_up=400),
        "pokec_n_g" : dict(lambda_fair=0.03, sbrs_quantile=0.7, struct_drop=0.7, warm_up=400),
        "german"    : dict(lambda_fair=0.15, sbrs_quantile=0.7, struct_drop=0.3, warm_up=100),
        "credit"    : dict(lambda_fair=0.10, sbrs_quantile=0.8, struct_drop=0.2, warm_up=100),
        "recidivism": dict(lambda_fair=0.07, sbrs_quantile=0.7, struct_drop=0.3, warm_up=100),
        "nba"       : dict(lambda_fair=0.15, sbrs_quantile=0.8, struct_drop=0.2, warm_up=200),
        "income"    : dict(lambda_fair=0.20, sbrs_quantile=0.5, struct_drop=0.7, warm_up=200),
    }

    print(f"\n{'='*75}")
    print("추천 FAIRGATE_CONFIGS")
    print(f"{'='*75}")
    print("FAIRGATE_CONFIGS = {")
    for ds in DATASET_ORDER:
        sub = df_all[df_all["dataset"] == ds].copy()
        if sub.empty:
            cfg = current.get(ds, {})
            mark = "  # (탐색 결과 없음, 기존값 유지)"
        else:
            sub["fair_sum"] = sub["dp_mean"] + sub["eo_mean"]
            sub["balance_score"] = sub["acc_mean"] - sub["dp_mean"] - sub["eo_mean"]
            best = select_best(sub, baseline, ds)
            cfg = {p: best[p] for p in HPARAMS}
            cur = current.get(ds, {})
            changed = [p for p in HPARAMS
                       if p in cur and abs(float(cfg[p]) - float(cur[p])) > 1e-6]
            mark = f"  # 변경: {changed}" if changed else "  # 변경없음"

        pad = " " * (11 - len(ds))
        print(f'    "{ds}":{pad}dict('
              f'lambda_fair={cfg.get("lambda_fair")}, '
              f'sbrs_quantile={cfg.get("sbrs_quantile")}, '
              f'struct_drop={cfg.get("struct_drop")}, '
              f'warm_up={int(cfg.get("warm_up", 200))}),{mark}')
    print("}")


# ── LaTeX 출력 ─────────────────────────────────────────────────────────────────

def save_latex_sensitivity(sens: pd.DataFrame, output_dir: str):
    rep_order = [d for d in DATASET_ORDER if d in sens["dataset"].unique()]
    lines = []
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{대표 데이터셋별 파라미터 민감도."
        r" 각 파라미터를 탐색 범위에서 변화시킬 때 "
        r"$\Delta\mathrm{DP}{+}\Delta\mathrm{EO}$의 최대--최소 범위(range)와 "
        r"최적값(best)을 정리하였다."
        r" 각 데이터셋에서 가장 큰 range를 굵게 표시하였다.}",
        r"\label{tab:hparam_sensitivity}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{$\lambda_{\mathrm{fair}}$}"
        r"& \multicolumn{2}{c}{$p_{\mathrm{struct}}$}"
        r"& \multicolumn{2}{c}{$q_{\mathrm{sbrs}}$}"
        r"& \multicolumn{2}{c}{$T_{\mathrm{warm}}$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"\textbf{Dataset} & range & best & range & best"
        r" & range & best & range & best \\",
        r"\midrule",
    ]

    for ds in rep_order:
        sub = sens[sens["dataset"] == ds]
        if sub.empty:
            continue
        row_data = {r["param"]: r for _, r in sub.iterrows()}
        max_range_param = sub.loc[sub["fair_range"].idxmax(), "param"]
        dname = DATASET_DISPLAY.get(ds, ds)
        regime = GRAPH_STATS.get(ds, {}).get("regime", "")
        cells = [f"{dname} ({regime})"]
        for p in HPARAMS:
            if p not in row_data:
                cells += ["—", "—"]
                continue
            r = row_data[p]
            range_str = f"{r['fair_range']:.3f}"
            best_str  = str(r["best_val"])
            if p == max_range_param:
                range_str = r"\textbf{" + range_str + "}"
                best_str  = r"\textbf{" + best_str  + "}"
            cells += [range_str, best_str]
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)

    path = os.path.join(output_dir, "hparam_sensitivity.tex")
    with open(path, "w") as f:
        f.write(tex)
    print(f"[Saved] {path}")


def save_latex_final(df_all: pd.DataFrame, baseline: dict, output_dir: str):
    lines = []
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{데이터셋별 최종 하이퍼파라미터 설정."
        r" $\lambda_{\mathrm{fips}}{=}1.0$, $\alpha_{\mathrm{mmd}}{=}0.3$은"
        r" 전 데이터셋 공통이므로 생략하였다.}",
        r"\label{tab:hparam_final}",
        r"\setlength{\tabcolsep}{7pt}",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Dataset}"
        r" & $\lambda_{\mathrm{fair}}$"
        r" & $q_{\mathrm{sbrs}}$"
        r" & $p_{\mathrm{struct}}$"
        r" & $T_{\mathrm{warm}}$"
        r" & Regime \\",
        r"\midrule",
    ]

    for ds in DATASET_ORDER:
        sub = df_all[df_all["dataset"] == ds].copy()
        if sub.empty:
            continue
        sub["fair_sum"] = sub["dp_mean"] + sub["eo_mean"]
        sub["balance_score"] = sub["acc_mean"] - sub["dp_mean"] - sub["eo_mean"]
        best = select_best(sub, baseline, ds)
        regime = GRAPH_STATS.get(ds, {}).get("regime", "?")
        dname = DATASET_DISPLAY.get(ds, ds)
        lines.append(
            f"{dname} & {best['lambda_fair']} & {best['sbrs_quantile']} & "
            f"{best['struct_drop']} & {int(best['warm_up'])} & {regime} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = os.path.join(output_dir, "hparam_final.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[Saved] {path}")


# ── 시각화 ─────────────────────────────────────────────────────────────────────

def plot_all(df_all: pd.DataFrame, sens: pd.DataFrame,
             df_cfg: pd.DataFrame, baseline: dict, output_dir: str):
    rep_datasets = sens["dataset"].unique().tolist()
    n_rep = len(rep_datasets)

    fig = plt.figure(figsize=(18, 22))
    fig.suptitle("FairGate Hyperparameter Analysis", fontsize=15, fontweight="bold", y=0.98)
    gs_root = gridspec.GridSpec(4, 1, figure=fig,
                                hspace=0.48,
                                height_ratios=[1.2, 1.2, 0.9, 1.0])

    # ── Plot 1: lambda_fair 민감도 곡선 (대표 데이터셋) ───────────────────────
    ax_row1 = gridspec.GridSpecFromSubplotSpec(
        1, n_rep, subplot_spec=gs_root[0], wspace=0.35)
    for i, ds in enumerate(rep_datasets):
        ax = fig.add_subplot(ax_row1[i])
        sub = df_all[df_all["dataset"] == ds].copy()
        sub["fair_sum"] = sub["dp_mean"] + sub["eo_mean"]
        grp = sub.groupby("lambda_fair")[["roc_auc_mean", "fair_sum"]].mean()
        regime = GRAPH_STATS.get(ds, {}).get("regime", "")
        color  = REGIME_COLORS.get(regime, "#888")

        ax2 = ax.twinx()
        ax.plot(grp.index, grp["fair_sum"], "o-", color=color,
                linewidth=2, markersize=5, label="ΔDP+ΔEO")
        ax2.plot(grp.index, grp["roc_auc_mean"], "s--", color="gray",
                 linewidth=1.5, markersize=4, alpha=0.7, label="AUC")

        best_lf = grp["fair_sum"].idxmin()
        ax.axvline(best_lf, color=color, alpha=0.4, linewidth=1.5, linestyle=":")

        ax.set_xlabel(r"$\lambda_{\mathrm{fair}}$", fontsize=10)
        if i == 0:
            ax.set_ylabel("ΔDP+ΔEO", fontsize=9, color=color)
        if i == n_rep - 1:
            ax2.set_ylabel("AUC", fontsize=9, color="gray")
        else:
            ax2.set_yticklabels([])

        dname = DATASET_DISPLAY.get(ds, ds)
        ax.set_title(f"{dname}\n({regime})", fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if i == 0:
            ax.legend(lines1 + lines2, labels1 + labels2,
                      fontsize=7, loc="upper right")

    fig.text(0.01, gs_root[0].get_position(fig).y1 + 0.005,
             r"(a) $\lambda_{\mathrm{fair}}$ sensitivity", fontsize=11, fontweight="bold")

    # ── Plot 2: struct_drop 민감도 곡선 ──────────────────────────────────────
    ax_row2 = gridspec.GridSpecFromSubplotSpec(
        1, n_rep, subplot_spec=gs_root[1], wspace=0.35)
    for i, ds in enumerate(rep_datasets):
        ax = fig.add_subplot(ax_row2[i])
        sub = df_all[df_all["dataset"] == ds].copy()
        sub["fair_sum"] = sub["dp_mean"] + sub["eo_mean"]
        grp = sub.groupby("struct_drop")[["roc_auc_mean", "fair_sum"]].mean()
        regime = GRAPH_STATS.get(ds, {}).get("regime", "")
        color  = REGIME_COLORS.get(regime, "#888")

        ax2 = ax.twinx()
        ax.plot(grp.index, grp["fair_sum"], "o-", color=color,
                linewidth=2, markersize=5, label="ΔDP+ΔEO")
        ax2.plot(grp.index, grp["roc_auc_mean"], "s--", color="gray",
                 linewidth=1.5, markersize=4, alpha=0.7, label="AUC")

        best_sd = grp["fair_sum"].idxmin()
        ax.axvline(best_sd, color=color, alpha=0.4, linewidth=1.5, linestyle=":")
        ax.set_xlabel(r"$p_{\mathrm{struct}}$", fontsize=10)
        if i == 0:
            ax.set_ylabel("ΔDP+ΔEO", fontsize=9, color=color)
        if i == n_rep - 1:
            ax2.set_ylabel("AUC", fontsize=9, color="gray")
        else:
            ax2.set_yticklabels([])

        dname = DATASET_DISPLAY.get(ds, ds)
        ax.set_title(f"{dname}\n({regime})", fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.text(0.01, gs_root[1].get_position(fig).y1 + 0.005,
             r"(b) $p_{\mathrm{struct}}$ sensitivity", fontsize=11, fontweight="bold")

    # ── Plot 3: 민감도 히트맵 ─────────────────────────────────────────────────
    ax_heat = fig.add_subplot(gs_root[2])
    pivot = sens.pivot_table(index="display", columns="param",
                             values="fair_range", aggfunc="first")
    pivot = pivot.reindex(columns=HPARAMS)

    ds_order = [DATASET_DISPLAY.get(d, d) for d in rep_datasets]
    pivot = pivot.reindex(ds_order)

    col_labels = [HPARAM_LABELS.get(c, c) for c in pivot.columns]
    im = ax_heat.imshow(pivot.values.astype(float), cmap="YlOrRd", aspect="auto")
    ax_heat.set_xticks(range(len(col_labels)))
    ax_heat.set_xticklabels(col_labels, fontsize=10)
    ax_heat.set_yticks(range(len(pivot.index)))
    ax_heat.set_yticklabels(pivot.index, fontsize=9)
    for r in range(len(pivot.index)):
        for c in range(len(pivot.columns)):
            val = pivot.values[r, c]
            if not np.isnan(val):
                ax_heat.text(c, r, f"{val:.3f}", ha="center", va="center",
                             fontsize=8,
                             color="white" if val > pivot.values.max() * 0.6 else "black")
    plt.colorbar(im, ax=ax_heat, label="ΔDP+ΔEO range", shrink=0.8)
    ax_heat.set_title("(c) Parameter sensitivity heatmap (ΔDP+ΔEO range)",
                      fontsize=11, fontweight="bold", loc="left")

    # ── Plot 4: 그래프 특성 ↔ 최적 파라미터 산점도 ───────────────────────────
    ax_row4 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_root[3], wspace=0.4)

    # (d) h vs lambda_fair
    ax_d = fig.add_subplot(ax_row4[0])
    for _, row in df_cfg.iterrows():
        regime = row.get("regime", "")
        color  = REGIME_COLORS.get(regime, "#888")
        ax_d.scatter(row["h"], row["lambda_fair"], color=color, s=80, zorder=3)
        ax_d.annotate(DATASET_DISPLAY.get(row["dataset"], row["dataset"]),
                      (row["h"], row["lambda_fair"]),
                      textcoords="offset points", xytext=(4, 3), fontsize=7)
    # 회귀선
    if len(df_cfg) >= 3:
        coeffs = np.polyfit(df_cfg["h"], df_cfg["lambda_fair"], 1)
        xfit = np.linspace(df_cfg["h"].min(), df_cfg["h"].max(), 100)
        r_val = df_cfg["h"].corr(df_cfg["lambda_fair"])
        ax_d.plot(xfit, np.polyval(coeffs, xfit), "--", color="#94A3B8",
                  alpha=0.7, label=f"r={r_val:+.3f}")
        ax_d.legend(fontsize=8)
    ax_d.set_xlabel("Edge homophily $h$", fontsize=10)
    ax_d.set_ylabel(r"$\lambda_{\mathrm{fair}}$ (best)", fontsize=10)
    ax_d.set_title(r"(d) $h$ vs $\lambda_{\mathrm{fair}}$",
                   fontsize=11, fontweight="bold", loc="left")
    ax_d.grid(True, alpha=0.3)

    # (e) bnd vs struct_drop
    ax_e = fig.add_subplot(ax_row4[1])
    for _, row in df_cfg.iterrows():
        regime = row.get("regime", "")
        color  = REGIME_COLORS.get(regime, "#888")
        ax_e.scatter(row["bnd"], row["struct_drop"], color=color, s=80, zorder=3)
        ax_e.annotate(DATASET_DISPLAY.get(row["dataset"], row["dataset"]),
                      (row["bnd"], row["struct_drop"]),
                      textcoords="offset points", xytext=(4, 3), fontsize=7)
    if len(df_cfg) >= 3:
        coeffs = np.polyfit(df_cfg["bnd"], df_cfg["struct_drop"], 1)
        xfit = np.linspace(df_cfg["bnd"].min(), df_cfg["bnd"].max(), 100)
        r_val = df_cfg["bnd"].corr(df_cfg["struct_drop"])
        ax_e.plot(xfit, np.polyval(coeffs, xfit), "--", color="#94A3B8",
                  alpha=0.7, label=f"r={r_val:+.3f}")
        ax_e.legend(fontsize=8)
    ax_e.set_xlabel(r"Boundary ratio $r_{\mathrm{bnd}}$", fontsize=10)
    ax_e.set_ylabel(r"$p_{\mathrm{struct}}$ (best)", fontsize=10)
    ax_e.set_title(r"(e) $r_{\mathrm{bnd}}$ vs $p_{\mathrm{struct}}$",
                   fontsize=11, fontweight="bold", loc="left")
    ax_e.grid(True, alpha=0.3)

    # 범례 (regime 색상)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=r)
                       for r, c in REGIME_COLORS.items()]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=4, fontsize=9, title="Regime",
               bbox_to_anchor=(0.5, 0.01))

    out = os.path.join(output_dir, "hparam_plot.pdf")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    print(f"[Saved] {out}")

    out_png = out.replace(".pdf", ".png")
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"[Saved] {out_png}")
    plt.close()


# ── 메인 ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="FairGate 하이퍼파라미터 탐색 결과 분석",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--hparam_dir", type=str, default="outputs/hparam",
                   help="hparam_*.csv 파일이 있는 디렉토리")
    p.add_argument("--baseline_csv", type=str, default=None,
                   help="비교 모델 결과 CSV (있으면 AUC threshold 기반 선정)")
    p.add_argument("--output_dir", type=str, default="outputs/analysis")
    p.add_argument("--rep_datasets", nargs="+",
                   default=["pokec_z", "pokec_z_g", "german", "credit"],
                   help="민감도 분석에 사용할 대표 데이터셋")
    p.add_argument("--no_plot", action="store_true",
                   help="시각화 생략")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("FairGate Hyperparameter Analysis")
    print(f"  hparam_dir  : {args.hparam_dir}")
    print(f"  baseline_csv: {args.baseline_csv}")
    print(f"  rep_datasets: {args.rep_datasets}")
    print(f"  output_dir  : {args.output_dir}")
    print(f"{'='*70}")

    # 1. 데이터 로드
    df_all   = load_hparam_data(args.hparam_dir)
    baseline = load_baseline(args.baseline_csv)

    # 2. Top-5 출력
    print_top5(df_all, baseline, args.output_dir)

    # 3. 민감도 분석
    rep_ds = [d for d in args.rep_datasets if d in df_all["dataset"].unique()]
    sens   = compute_sensitivity(df_all, rep_ds)
    print_sensitivity(sens)

    csv_path = os.path.join(args.output_dir, "hparam_sensitivity.csv")
    sens.to_csv(csv_path, index=False)
    print(f"\n[Saved] {csv_path}")

    # 4. 상관 분석 + 최적 config
    df_cfg = compute_correlations(df_all, baseline)

    # 5. 추천 CONFIGS 출력
    print_recommended_configs(df_all, baseline)

    # 6. LaTeX 저장
    save_latex_sensitivity(sens, args.output_dir)
    save_latex_final(df_all, baseline, args.output_dir)

    # 7. 시각화
    if not args.no_plot:
        print("\n[시각화 생성 중...]")
        plot_all(df_all, sens, df_cfg, baseline, args.output_dir)

    print(f"\n{'='*70}")
    print(f"분석 완료. 결과: {args.output_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()