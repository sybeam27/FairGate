"""
analyze_scalability.py — FairGate vs Baseline 확장성 비교 분석

FairGate와 비교 모델들의 학습 시간을 그래프 규모(nodes, edges)와 비교한다.

측정 항목:
    - 노드/엣지 수 vs 학습 시간 (모델별)
    - 노드 당 시간 (time_per_1k_nodes), 엣지 당 시간 (time_per_1k_edges)
    - log-log 회귀 기반 complexity 추정
    - FairGate vs baseline 시간 비율 (overhead 분석)

출력:
    outputs/analysis/scalability.csv
    outputs/analysis/scalability.tex
    outputs/analysis/scalability_plot.py

실행:
    python analyze_scalability.py
    python analyze_scalability.py \\
        --fairgate_csv outputs/exp_fairgate_gcn.csv \\
        --baseline_csv outputs/compare/exp_baselines.csv \\
        --graph_stats  outputs/analysis/graph_stats.csv
"""

import os
import argparse
import numpy as np
import pandas as pd


GRAPH_SIZE_FALLBACK = {
    "pokec_z"   : {"nodes": 67796,  "edges": 1303712, "display_name": "Pokec-Z"},
    "pokec_z_g" : {"nodes": 67796,  "edges": 1303712, "display_name": "Pokec-Z (gender)"},
    "pokec_n"   : {"nodes": 66569,  "edges": 1100663, "display_name": "Pokec-N"},
    "pokec_n_g" : {"nodes": 66569,  "edges": 1100663, "display_name": "Pokec-N (gender)"},
    "german"    : {"nodes": 1000,   "edges": 44484,   "display_name": "German"},
    "credit"    : {"nodes": 30000,  "edges": 2873716, "display_name": "Credit"},
    "recidivism": {"nodes": 18876,  "edges": 642616,  "display_name": "Recidivism"},
    "nba"       : {"nodes": 403,    "edges": 21645,   "display_name": "NBA"},
    "income"    : {"nodes": 14821,  "edges": 100483,  "display_name": "Income"},
}

DATASET_ORDER = [
    "german", "nba", "income", "recidivism",
    "credit", "pokec_z", "pokec_z_g", "pokec_n", "pokec_n_g",
]

DATASET_DISPLAY = {
    "pokec_z": "Pokec-Z",     "pokec_z_g": "Pokec-Z (g)",
    "pokec_n": "Pokec-N",     "pokec_n_g": "Pokec-N (g)",
    "german": "German",        "credit": "Credit",
    "recidivism": "Recidivism","nba": "NBA", "income": "Income",
}


def load_csv(path: str, model_filter: str = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일 없음: {path}")
    df = pd.read_csv(path)
    if model_filter and "model" in df.columns:
        df = df[df["model"] == model_filter].copy()
    if "time_sec_mean" not in df.columns:
        raise ValueError(
            f"time_sec_mean 컬럼 없음: {path}\n"
            f"  → train_baselines.py 수정 후 재실행 필요"
        )
    return df


def load_graph_sizes(graph_stats_csv: str) -> dict:
    if graph_stats_csv and os.path.exists(graph_stats_csv):
        gs = pd.read_csv(graph_stats_csv)
        print(f"[INFO] graph_stats.csv 로드: {graph_stats_csv}")
        result = {}
        for _, row in gs.iterrows():
            result[row["dataset"]] = {
                "nodes"         : int(row["nodes"]),
                "edges"         : int(row["edges"]),
                "display_name"  : row.get("display_name", row["dataset"]),
                "homophily"     : row.get("homophily", None),
                "boundary_ratio": row.get("boundary_ratio", None),
                "regime"        : row.get("regime", None),
            }
        return result
    print("[INFO] graph_stats.csv 없음 — fallback 값 사용")
    return GRAPH_SIZE_FALLBACK


def build_time_df(df: pd.DataFrame, sizes: dict, model_name: str) -> pd.DataFrame:
    grp = df.groupby("dataset")[["time_sec_mean", "time_sec_std"]].mean().reset_index()
    rows = []
    for _, row in grp.iterrows():
        ds = row["dataset"]
        if ds not in sizes:
            continue
        sz = sizes[ds]
        N, E, t = sz["nodes"], sz["edges"], row["time_sec_mean"]
        rows.append({
            "model"            : model_name,
            "dataset"          : ds,
            "display_name"     : sz["display_name"],
            "nodes"            : N,
            "edges"            : E,
            "time_sec_mean"    : round(t, 2),
            "time_sec_std"     : round(row.get("time_sec_std", 0), 2),
            "time_per_1k_nodes": round(t / (N / 1000), 4),
            "time_per_1k_edges": round(t / (E / 1000), 4),
        })
    return pd.DataFrame(rows)


def fit_complexity(df: pd.DataFrame) -> dict:
    results = {}
    for x_col, label in [("nodes", "N"), ("edges", "E")]:
        x = np.log(df[x_col].values.astype(float))
        y = np.log(df["time_sec_mean"].values.astype(float))
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        results[x_col] = {"slope": round(coeffs[0], 3),
                           "r2":    round(r2, 4), "label": label}
    return results


def print_summary(df_all: pd.DataFrame, models: list):
    datasets = [d for d in DATASET_ORDER if d in df_all["dataset"].unique()]
    print(f"\n{'='*95}")
    print(f"{'Dataset':<20} {'Nodes':>8}", end="")
    for m in models:
        print(f"  {m[:12]:>12}", end="")
    print(f"  {'FG/avg':>8}")
    print(f"{'='*95}")
    for ds in datasets:
        sub = df_all[df_all["dataset"] == ds]
        n = sub["nodes"].iloc[0] if not sub.empty else 0
        print(f"{DATASET_DISPLAY.get(ds, ds):<20} {n:>8,}", end="")
        times = {}
        for m in models:
            r = sub[sub["model"] == m]
            t = r["time_sec_mean"].values[0] if not r.empty else float("nan")
            times[m] = t
            print(f"  {t:>12.1f}" if not np.isnan(t) else f"  {'—':>12}", end="")
        fg_t = times.get("FairGate", float("nan"))
        bl_t = [v for k, v in times.items() if k != "FairGate" and not np.isnan(v)]
        if bl_t and not np.isnan(fg_t):
            print(f"  {fg_t/np.mean(bl_t):>8.2f}x")
        else:
            print(f"  {'—':>8}")
    print(f"{'='*95}")

    print("\n[Complexity 추정 (log-log 회귀)]")
    for m in models:
        sub = df_all[df_all["model"] == m].dropna(subset=["time_sec_mean"])
        if len(sub) < 3:
            continue
        comp = fit_complexity(sub)
        best = max(comp.items(), key=lambda x: x[1]["r2"])
        print(f"  {m:<16}: time ~ |{best[1]['label']}|^{best[1]['slope']:.3f}"
              f"  R²={best[1]['r2']:.4f}")


def to_latex(df_all: pd.DataFrame, models: list) -> str:
    datasets = [d for d in DATASET_ORDER if d in df_all["dataset"].unique()]
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{데이터셋별 모델 학습 시간(초) 비교. "
        r"FG/avg: FairGate 시간 대비 baseline 평균 비율. "
        r"학습 시간은 5 runs 평균 $\pm$ 표준편차.}"
    )
    lines.append(r"\label{tab:scalability}")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    lines.append(r"\resizebox{\linewidth}{!}{")
    lines.append(r"\begin{tabular}{l" + "r" * (len(models) + 1) + "}")
    lines.append(r"\toprule")
    hdr = r"\textbf{Dataset}" + "".join(f" & {m}" for m in models) + r" & FG/avg"
    lines.append(hdr + r" \\")
    lines.append(r"\midrule")
    for ds in datasets:
        sub  = df_all[df_all["dataset"] == ds]
        row  = DATASET_DISPLAY.get(ds, ds)
        times = {}
        for m in models:
            r = sub[sub["model"] == m]
            if r.empty:
                row += " & —"; times[m] = float("nan")
            else:
                t, std = r.iloc[0]["time_sec_mean"], r.iloc[0]["time_sec_std"]
                cell = f"{t:.1f} $\\pm$ {std:.1f}"
                if m == "FairGate":
                    cell = r"\textbf{" + cell + "}"
                row += " & " + cell; times[m] = t
        fg_t = times.get("FairGate", float("nan"))
        bl   = [v for k, v in times.items() if k != "FairGate" and not np.isnan(v)]
        row += f" & {fg_t/np.mean(bl):.2f}x" if bl and not np.isnan(fg_t) else " & —"
        lines.append(row + r" \\")
    lines.append(r"\midrule")
    avg_row = r"\textit{Average}"
    for m in models:
        avg_row += f" & {df_all[df_all['model']==m]['time_sec_mean'].mean():.1f}"
    fg_avg = df_all[df_all["model"]=="FairGate"]["time_sec_mean"].mean()
    bl_avg = df_all[df_all["model"]!="FairGate"]["time_sec_mean"].mean()
    avg_row += f" & {fg_avg/bl_avg:.2f}x" if bl_avg > 0 else " & —"
    lines.append(avg_row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def write_plot_script(output_dir: str, models: list):
    code = f'''"""
scalability_plot.py — FairGate vs Baseline 확장성 시각화
실행: python {output_dir}/scalability_plot.py
"""
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(os.path.join("{output_dir}", "scalability.csv"))
MODELS = {models}
COLORS = plt.cm.tab10.colors
MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("Scalability: FairGate vs Baselines", fontsize=13, fontweight="bold")

for ax_idx, (x_col, x_label, title) in enumerate([
    ("nodes", "Number of nodes |V|", "Time vs. Nodes"),
    ("edges", "Number of edges |E|", "Time vs. Edges"),
]):
    ax = axes[ax_idx]
    for i, m in enumerate(MODELS):
        sub = df[df["model"] == m].sort_values(x_col)
        lw = 2.5 if m == "FairGate" else 1.2
        ls = "-" if m == "FairGate" else "--"
        ax.plot(sub[x_col], sub["time_sec_mean"],
                marker=MARKERS[i%len(MARKERS)], color=COLORS[i],
                linewidth=lw, linestyle=ls, markersize=6, label=m)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("Training time (s)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

ax = axes[2]
fg  = df[df["model"]=="FairGate"].set_index("dataset")
bl  = df[df["model"]!="FairGate"].groupby("dataset")["time_sec_mean"].mean()
common = [d for d in fg.index if d in bl.index]
ratios = [fg.loc[d,"time_sec_mean"]/bl[d] for d in common]
dnames = [fg.loc[d,"display_name"] for d in common]
bars = ax.barh(dnames, ratios,
               color=["#2563EB" if r<=2 else "#DC2626" for r in ratios],
               alpha=0.85)
ax.axvline(1.0, color="#9CA3AF", linewidth=1, linestyle="--")
ax.set_xlabel("FairGate time / baseline avg", fontsize=10)
ax.set_title("FairGate overhead", fontsize=11)
ax.grid(True, axis="x", alpha=0.3)
for bar, val in zip(bars, ratios):
    ax.text(bar.get_width()+0.02, bar.get_y()+bar.get_height()/2,
            f"{{val:.2f}}x", va="center", fontsize=8)

plt.tight_layout()
out = os.path.join("{output_dir}", "scalability_plot.pdf")
plt.savefig(out, bbox_inches="tight", dpi=150)
print(f"[Saved] {{out}}")
plt.show()
'''
    path = os.path.join(output_dir, "scalability_plot.py")
    with open(path, "w") as f:
        f.write(code)
    print(f"[Saved] {path}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--fairgate_csv",     type=str, default="outputs/exp_fairgate_gcn.csv")
    p.add_argument("--baseline_csv",     type=str, default="outputs/compare/exp_baselines.csv")
    p.add_argument("--graph_stats",      type=str, default="outputs/analysis/graph_stats.csv")
    p.add_argument("--output_dir",       type=str, default="outputs/analysis")
    p.add_argument("--baseline_models",  nargs="+", default=None,
                   help="포함할 baseline 모델명 (기본: CSV 전체)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n[Scalability Analysis — FairGate vs Baselines]")
    sizes = load_graph_sizes(args.graph_stats)

    fg_df   = load_csv(args.fairgate_csv, model_filter="FairGate")
    fg_time = build_time_df(fg_df, sizes, "FairGate")
    print(f"  FairGate : {len(fg_time)} datasets")

    bl_df = load_csv(args.baseline_csv)
    if args.baseline_models:
        bl_df = bl_df[bl_df["model"].isin(args.baseline_models)]
    bl_models = sorted(bl_df["model"].unique().tolist()) if "model" in bl_df.columns else []
    print(f"  Baselines: {bl_models}")

    bl_frames = [build_time_df(bl_df[bl_df["model"]==m], sizes, m) for m in bl_models]
    df_all  = pd.concat([fg_time] + bl_frames, ignore_index=True)
    models  = ["FairGate"] + bl_models

    print_summary(df_all, models)

    csv_path = os.path.join(args.output_dir, "scalability.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"\n[Saved] {csv_path}")

    tex_path = os.path.join(args.output_dir, "scalability.tex")
    with open(tex_path, "w") as f:
        f.write(to_latex(df_all, models))
    print(f"[Saved] {tex_path}")

    write_plot_script(args.output_dir, models)


if __name__ == "__main__":
    main()