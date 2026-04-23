"""
analyze_hparam_viz.py — Hyperparameter Analysis Visualization

Generates paper-ready figures.

Figure layout (2x2):
    (a) Core parameter sensitivity — T_warm (3 datasets, DP+EO + AUC)
    (b) Core parameter sensitivity — lambda_fair
    (c) Auxiliary parameter — q_sbrs (role varies by regime)
    (d) Auxiliary parameter — p_struct (relationship with bnd)

Additional figures:
    (e) Span heatmap — parameter x dataset sensitivity summary
    (f) bnd vs span scatter — structure-to-sensitivity

Output:
    outputs/figures/hparam_curves.pdf   <- main paper figure
    outputs/figures/hparam_curves.png
    outputs/figures/hparam_span_heatmap.pdf
    outputs/figures/hparam_span_heatmap.png

Usage:
    python analyze_hparam_viz.py
    python analyze_hparam_viz.py --sweep_dir outputs/axis_sweep
    python analyze_hparam_viz.py --no_heatmap   # main figure only
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── Constants ────────────────────────────────────────────────────────────────

DATASET_DISPLAY = {
    "pokec_z": "Pokec-Z",
    "german" : "German",
    "credit" : "Credit",
}

DATASET_ORDER = ["pokec_z", "german", "credit"]

GRAPH_STATS = {
    "pokec_z": dict(h=0.953, bnd=0.369, deg_gap=0.083, regime="clustered"),
    "german" : dict(h=0.809, bnd=0.970, deg_gap=0.049, regime="saturated"),
    "credit" : dict(h=0.960, bnd=0.677, deg_gap=0.315, regime="degree-skewed"),
}

REGIME_COLORS = {
    "pokec_z": "#3B82F6",    # blue — clustered
    "german" : "#EF4444",    # red  — saturated
    "credit" : "#F59E0B",    # amber — degree-skewed
}

REGIME_MARKERS = {
    "pokec_z": "o",
    "german" : "s",
    "credit" : "^",
}

CONFIGS = {
    "pokec_z": dict(sbrs_quantile=0.9,  struct_drop=0.5, lambda_fair=0.10, warm_up=400),
    "german" : dict(sbrs_quantile=0.95, struct_drop=0.2, lambda_fair=0.20, warm_up=100),
    "credit" : dict(sbrs_quantile=0.5,  struct_drop=0.7, lambda_fair=0.20, warm_up=200),
}

PARAM_INFO = {
    "warm_up"      : dict(label=r"$T_{\mathrm{warm}}$",
                          xlabel=r"$T_{\mathrm{warm}}$ (epochs)",
                          title=r"(a) Phase lag ($T_{\mathrm{warm}}$) — Core"),
    "lambda_fair"  : dict(label=r"$\lambda_{\mathrm{fair}}$",
                          xlabel=r"$\lambda_{\mathrm{fair}}$",
                          title=r"(b) Fairness budget ($\lambda_{\mathrm{fair}}$) — Core"),
    "sbrs_quantile": dict(label=r"$q_{\mathrm{sbrs}}$",
                          xlabel=r"$q_{\mathrm{sbrs}}$",
                          title=r"(c) Gate budget ($q_{\mathrm{sbrs}}$) — Auxiliary"),
    "struct_drop"  : dict(label=r"$p_{\mathrm{struct}}$",
                          xlabel=r"$p_{\mathrm{struct}}$",
                          title=r"(d) Attenuation ($p_{\mathrm{struct}}$) — Auxiliary"),
}

PARAM_ORDER = ["warm_up", "lambda_fair", "sbrs_quantile", "struct_drop"]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_sweep(sweep_dir: str) -> pd.DataFrame:
    pattern = os.path.join(sweep_dir, "exp_sweep_*.csv")
    files   = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No sweep CSV found: {sweep_dir}")
    dfs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    dfs["fair"] = dfs["dp_mean"] + dfs["eo_mean"]
    return dfs


def get_curve(dfs: pd.DataFrame, param: str, ds: str):
    sub = dfs[(dfs["sensitivity_param"] == param) & (dfs["dataset"] == ds)]
    if sub.empty:
        return None, None
    grp_fair = sub.groupby("sensitivity_value")["fair"].mean()
    grp_auc  = sub.groupby("sensitivity_value")["roc_auc_mean"].mean()
    return grp_fair, grp_auc


def compute_span(grp_fair) -> float:
    return float(grp_fair.max() - grp_fair.min()) if grp_fair is not None else np.nan


# ── Main Figure: sensitivity curves (2x2) ────────────────────────────────────

def plot_sensitivity_curves(dfs: pd.DataFrame, output_dir: str):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "Hyperparameter Sensitivity Analysis\n"
        r"(DP+EO: solid / AUC: dashed)",
        fontsize=13, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    for idx, param in enumerate(PARAM_ORDER):
        row, col = divmod(idx, 2)
        ax_fair  = fig.add_subplot(gs[row, col])
        ax_auc   = ax_fair.twinx()

        info = PARAM_INFO[param]
        any_plotted = False

        for ds in DATASET_ORDER:
            grp_fair, grp_auc = get_curve(dfs, param, ds)
            if grp_fair is None:
                continue

            color  = REGIME_COLORS[ds]
            marker = REGIME_MARKERS[ds]
            dname  = DATASET_DISPLAY[ds]
            gs_info = GRAPH_STATS[ds]
            cur_val = CONFIGS[ds].get(param)

            x = grp_fair.index.values.astype(float)
            y_fair = grp_fair.values
            y_auc  = grp_auc.values

            # DP+EO curve (solid line)
            ax_fair.plot(x, y_fair, color=color, marker=marker,
                         linewidth=2, markersize=6, zorder=3,
                         label=f"{dname} ({gs_info['regime']})")

            # AUC curve (dashed, lighter)
            ax_auc.plot(x, y_auc, color=color, marker=marker,
                        linewidth=1.5, markersize=5, linestyle="--",
                        alpha=0.5, zorder=2)

            # Current config marker (vertical dotted line)
            if cur_val is not None and cur_val in grp_fair.index:
                ax_fair.axvline(cur_val, color=color, linestyle=":",
                                linewidth=1.5, alpha=0.6)
                fair_at_cur = grp_fair.loc[cur_val]
                ax_fair.scatter([cur_val], [fair_at_cur],
                                color=color, s=80, zorder=5,
                                edgecolors="white", linewidths=1.5)

            # Best value marker (star)
            best_x = grp_fair.idxmin()
            best_y = grp_fair.min()
            ax_fair.scatter([best_x], [best_y], color=color,
                            s=120, marker="*", zorder=6,
                            edgecolors="black", linewidths=0.8)

            any_plotted = True

        # span annotation (German has the largest span)
        grp_g, _ = get_curve(dfs, param, "german")
        if grp_g is not None:
            span_val = compute_span(grp_g)
            ax_fair.annotate(
                f"span(German)={span_val:.3f}",
                xy=(0.98, 0.04), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=8,
                color="#EF4444",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#EF4444", alpha=0.7)
            )

        ax_fair.set_xlabel(info["xlabel"], fontsize=10)
        ax_fair.set_ylabel(r"$\Delta\mathrm{DP}+\Delta\mathrm{EO}$",
                           fontsize=9, color="black")
        ax_auc.set_ylabel("AUC", fontsize=9, color="gray")
        ax_auc.tick_params(axis="y", labelcolor="gray")
        ax_fair.set_title(info["title"], fontsize=10,
                          fontweight="bold", loc="left")
        ax_fair.grid(True, alpha=0.25, linestyle="--")

        # warm_up: show raw scale (no log)
        if param == "warm_up":
            ax_fair.set_xticks([0, 50, 100, 200, 400, 600])

    # Legend
    legend_lines = [
        Line2D([0], [0], color=REGIME_COLORS[ds], marker=REGIME_MARKERS[ds],
               linewidth=2, markersize=7,
               label=f"{DATASET_DISPLAY[ds]} ({GRAPH_STATS[ds]['regime']}, "
                     f"bnd={GRAPH_STATS[ds]['bnd']:.2f})")
        for ds in DATASET_ORDER
    ]
    legend_lines += [
        Line2D([0], [0], color="gray", linewidth=2, linestyle="-",
               label=r"$\Delta\mathrm{DP}+\Delta\mathrm{EO}$ (left)"),
        Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--",
               alpha=0.5, label="AUC (right)"),
        Line2D([0], [0], color="none", marker="o", markerfacecolor="gray",
               markeredgecolor="white", markersize=9,
               label="Current config"),
        Line2D([0], [0], color="none", marker="*", markerfacecolor="gray",
               markeredgecolor="black", markersize=11,
               label="Best value"),
    ]
    fig.legend(handles=legend_lines, loc="lower center",
               ncol=3, fontsize=8.5, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0))

    plt.subplots_adjust(bottom=0.14)

    for ext in ["pdf", "png"]:
        path = os.path.join(output_dir, f"hparam_curves.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=180)
        print(f"[Saved] {path}")
    plt.close()


# ── Span heatmap ─────────────────────────────────────────────────────────────

def plot_span_heatmap(dfs: pd.DataFrame, output_dir: str):
    param_labels = {
        "warm_up"      : r"$T_{\mathrm{warm}}$",
        "lambda_fair"  : r"$\lambda_{\mathrm{fair}}$",
        "sbrs_quantile": r"$q_{\mathrm{sbrs}}$",
        "struct_drop"  : r"$p_{\mathrm{struct}}$",
    }

    # Compute span matrix
    spans = {}
    for param in PARAM_ORDER:
        spans[param] = {}
        for ds in DATASET_ORDER:
            grp, _ = get_curve(dfs, param, ds)
            spans[param][ds] = compute_span(grp)

    matrix = np.array([[spans[p][ds] for ds in DATASET_ORDER]
                        for p in PARAM_ORDER])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                              gridspec_kw={"width_ratios": [1.4, 1]})

    # ── (a) Heatmap ───────────────────────────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=matrix.max())

    ax.set_xticks(range(len(DATASET_ORDER)))
    ax.set_xticklabels(
        [f"{DATASET_DISPLAY[ds]}\n(bnd={GRAPH_STATS[ds]['bnd']:.2f})"
         for ds in DATASET_ORDER], fontsize=10
    )
    ax.set_yticks(range(len(PARAM_ORDER)))
    ax.set_yticklabels(
        [param_labels[p] for p in PARAM_ORDER], fontsize=11
    )

    # Cell values + highlight max per row
    for i, param in enumerate(PARAM_ORDER):
        max_ds_idx = np.argmax([spans[param][ds] for ds in DATASET_ORDER])
        for j, ds in enumerate(DATASET_ORDER):
            val = spans[param][ds]
            text_color = "white" if val > matrix.max() * 0.6 else "black"
            weight = "bold" if j == max_ds_idx else "normal"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color=text_color, fontweight=weight)

    plt.colorbar(im, ax=ax, label=r"Span ($\Delta\mathrm{DP}+\Delta\mathrm{EO}$)",
                 shrink=0.85)
    ax.set_title("(a) Parameter sensitivity heatmap\n"
                 r"(larger span = more important to tune)",
                 fontsize=10, fontweight="bold", loc="left")

    # ── (b) bnd vs span scatter ──────────────────────────────────────────────
    ax2 = axes[1]
    bnd_vals = [GRAPH_STATS[ds]["bnd"] for ds in DATASET_ORDER]

    for pidx, param in enumerate(PARAM_ORDER):
        span_vals = [spans[param][ds] for ds in DATASET_ORDER]
        ax2.plot(bnd_vals, span_vals,
                 marker=["o","s","^","D"][pidx],
                 linewidth=1.5, markersize=8,
                 label=param_labels[param])

        # Regression line
        coeffs = np.polyfit(bnd_vals, span_vals, 1)
        xfit   = np.linspace(min(bnd_vals) - 0.05, max(bnd_vals) + 0.05, 50)
        r_val  = np.corrcoef(bnd_vals, span_vals)[0, 1]
        line_color = ax2.lines[-1].get_color()
        ax2.plot(xfit, np.polyval(coeffs, xfit), linestyle=":",
                 alpha=0.4, color=line_color)

    ax2.set_xlabel(r"Boundary ratio $r_{\mathrm{bnd}}$", fontsize=10)
    ax2.set_ylabel(r"Span ($\Delta\mathrm{DP}+\Delta\mathrm{EO}$)", fontsize=10)
    ax2.set_title("(b) Structure → sensitivity\n"
                  r"($r_{\mathrm{bnd}}$ vs span)",
                  fontsize=10, fontweight="bold", loc="left")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(fontsize=9, loc="upper left")

    # Dataset labels
    for j, ds in enumerate(DATASET_ORDER):
        ax2.annotate(
            DATASET_DISPLAY[ds],
            xy=(bnd_vals[j], max(spans[p][ds] for p in PARAM_ORDER)),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8, color="gray"
        )

    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = os.path.join(output_dir, f"hparam_span_heatmap.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=180)
        print(f"[Saved] {path}")
    plt.close()


# ── Hierarchy bar chart ──────────────────────────────────────────────────────

def plot_hierarchy_summary(dfs: pd.DataFrame, output_dir: str):
    """Summarize core/auxiliary parameter hierarchy as bar chart."""
    param_labels = {
        "warm_up"      : r"$T_{\mathrm{warm}}$",
        "lambda_fair"  : r"$\lambda_{\mathrm{fair}}$",
        "sbrs_quantile": r"$q_{\mathrm{sbrs}}$",
        "struct_drop"  : r"$p_{\mathrm{struct}}$",
    }
    colors_ds = [REGIME_COLORS[ds] for ds in DATASET_ORDER]

    fig, ax = plt.subplots(figsize=(8, 4))

    n_params = len(PARAM_ORDER)
    n_ds     = len(DATASET_ORDER)
    width    = 0.22
    x        = np.arange(n_params)

    for i, ds in enumerate(DATASET_ORDER):
        spans = []
        for param in PARAM_ORDER:
            grp, _ = get_curve(dfs, param, ds)
            spans.append(compute_span(grp))
        bars = ax.bar(x + i * width, spans, width, label=DATASET_DISPLAY[ds],
                      color=colors_ds[i], alpha=0.85, edgecolor="white")

    ax.set_xticks(x + width)
    ax.set_xticklabels([param_labels[p] for p in PARAM_ORDER], fontsize=12)
    ax.set_ylabel(r"Span ($\Delta\mathrm{DP}+\Delta\mathrm{EO}$)", fontsize=10)
    ax.set_title("Parameter importance by dataset\n"
                 "(larger bar = more important to tune)",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    # Core / auxiliary separator
    ax.axvline(1.5, color="black", linestyle="--", linewidth=1.5, alpha=0.5)
    ax.text(0.5, ax.get_ylim()[1] * 0.95, "Core params",
            ha="center", fontsize=9, color="black",
            bbox=dict(boxstyle="round", fc="#DBEAFE", ec="#3B82F6", alpha=0.7))
    ax.text(2.5, ax.get_ylim()[1] * 0.95, "Auxiliary params",
            ha="center", fontsize=9, color="black",
            bbox=dict(boxstyle="round", fc="#FEF3C7", ec="#F59E0B", alpha=0.7))

    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = os.path.join(output_dir, f"hparam_hierarchy.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=180)
        print(f"[Saved] {path}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="FairGate Hyperparameter Analysis Visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sweep_dir",   type=str, default="outputs/axis_sweep",
                   help="Directory containing exp_sweep_*.csv files")
    p.add_argument("--output_dir",  type=str, default="outputs/figures")
    p.add_argument("--no_heatmap",  action="store_true",
                   help="Skip span heatmap")
    p.add_argument("--no_hierarchy",action="store_true",
                   help="Skip hierarchy bar chart")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n[Load] sweep CSV from: {args.sweep_dir}")
    dfs = load_sweep(args.sweep_dir)
    print(f"  {len(dfs)} rows  |  params: {dfs['sensitivity_param'].unique().tolist()}")
    print(f"  datasets: {dfs['dataset'].unique().tolist()}\n")

    print("[Figure 1] Sensitivity curves (4 params x 3 datasets)")
    plot_sensitivity_curves(dfs, args.output_dir)

    if not args.no_heatmap:
        print("\n[Figure 2] Span heatmap + bnd vs span scatter")
        plot_span_heatmap(dfs, args.output_dir)

    if not args.no_hierarchy:
        print("\n[Figure 3] Parameter hierarchy bar chart")
        plot_hierarchy_summary(dfs, args.output_dir)

    print(f"\nDone. Output: {args.output_dir}/")


if __name__ == "__main__":
    main()