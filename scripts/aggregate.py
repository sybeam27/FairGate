"""
Aggregate WSDM allocation / sweep experiments into outputs/wsdm_report.md.

Inputs (written by utils/train_fairgate.py):
    outputs/wsdm_alloc_per_seed.csv
    outputs/wsdm_sweep_per_seed.csv   (optional)
Optional: log files containing "[Rewire]" lines (--sweep_logs); otherwise the
post-rewiring h / r_b recorded in the per-seed CSV (graph_h, graph_r_b) are used.

Usage:
    python scripts/aggregate.py
    python scripts/aggregate.py --sweep_logs "logs/wsdm_sweep_*.log"
"""

import argparse
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest

warnings.filterwarnings("ignore", message="Sample size too small")

SETTINGS = ["pokec_z", "pokec_n", "pokec_z_g", "pokec_n_g",
            "german", "credit", "income", "recidivism", "nba"]
SATURATED = ["german", "recidivism", "nba"]
NON_SATURATED = [s for s in SETTINGS if s not in SATURATED]
STRATEGIES = ["uniform", "random", "boundary", "degree", "loss", "confidence", "adaptive"]
STRATEGY_LABEL = {"boundary": "fixed (boundary)"}
FIXED_SIGNALS = ["boundary", "degree", "loss"]
METRICS = ["acc", "roc_auc", "dp", "eo"]

# Signal the graph diagnostics select per setting (adaptive branch).
DIAG_CHOICE = {
    "pokec_z": "boundary", "pokec_n": "boundary", "income": "boundary",
    "german": "loss", "recidivism": "loss", "nba": "loss",
    "credit": "degree",
    "pokec_z_g": "boundary", "pokec_n_g": "boundary",   # boundary with a small ℓ mix
}


def strategy_of(row) -> str:
    if str(row.get("fiw_weight_mode", "")) == "uniform":
        return "uniform"
    g = str(row.get("gating_mode_override", ""))
    return "adaptive" if g == "score" else g


def load(path, extra_keys=(), unrewired_only=False):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if unrewired_only and "rewire_frac" in df.columns:
        df = df[df["rewire_frac"].fillna(0.0).astype(float) == 0.0]
    df["strategy"] = df.apply(strategy_of, axis=1)
    keys = ["dataset", "strategy", *extra_keys, "seed_used"]
    df = df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    return df


def fmt(m, s, scale=100.0):
    if pd.isna(m):
        return "–"
    return f"{m * scale:.2f}±{(0.0 if pd.isna(s) else s) * scale:.2f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def setting_table(df):
    strategies = [s for s in STRATEGIES if s in set(df["strategy"])]
    blocks = []
    for metric in METRICS:
        g = df.groupby(["dataset", "strategy"])[metric].agg(["mean", "std", "count"])
        rows = []
        for ds in [d for d in SETTINGS if d in set(df["dataset"])]:
            r = [ds]
            for st in strategies:
                if (ds, st) in g.index:
                    m, s, n = g.loc[(ds, st)]
                    r.append(fmt(m, s) + ("" if n == 5 else f" (n={int(n)})"))
                else:
                    r.append("–")
            rows.append(r)
        blocks.append(f"#### {metric} (×100, mean±std over seeds)\n\n" +
                      md_table(["setting"] + [STRATEGY_LABEL.get(s, s) for s in strategies], rows))
    return "\n\n".join(blocks)


def group_averages(df):
    strategies = [s for s in STRATEGIES if s in set(df["strategy"])]
    per_setting = df.groupby(["dataset", "strategy"])[METRICS].mean()
    rows = []
    for gname, members in [("all", SETTINGS), ("non-saturated", NON_SATURATED),
                           ("saturated (german, recidivism, nba)", SATURATED)]:
        for st in strategies:
            sub = per_setting[per_setting.index.get_level_values("strategy") == st]
            sub = sub[sub.index.get_level_values("dataset").isin(members)]
            if len(sub) == 0:
                continue
            vals = [f"{sub[m].mean() * 100:.2f}" for m in METRICS]
            rows.append([gname, STRATEGY_LABEL.get(st, st), len(sub), *vals])
    return md_table(["group", "strategy", "#settings", *METRICS], rows)


def paired_tests(df):
    if "adaptive" not in set(df["strategy"]):
        return "_No adaptive runs found._"
    strategies = [s for s in STRATEGIES if s in set(df["strategy"]) and s != "adaptive"]
    a = df[df["strategy"] == "adaptive"].set_index(["dataset", "seed_used"])
    a_set = df[df["strategy"] == "adaptive"].groupby("dataset")[["dp", "eo"]].mean()
    rows = []
    for st in strategies:
        b = df[df["strategy"] == st].set_index(["dataset", "seed_used"])
        common = a.index.intersection(b.index)
        b_set = df[df["strategy"] == st].groupby("dataset")[["dp", "eo"]].mean()
        common_set = a_set.index.intersection(b_set.index)
        for m in ["dp", "eo"]:
            d = (a.loc[common, m] - b.loc[common, m]).to_numpy()
            try:
                p_w = wilcoxon(d, alternative="less").pvalue if np.any(d != 0) else float("nan")
            except ValueError:
                p_w = float("nan")
            ds = (a_set.loc[common_set, m] - b_set.loc[common_set, m]).to_numpy()
            wins, losses = int((ds < 0).sum()), int((ds > 0).sum())
            n = wins + losses
            p_s = binomtest(wins, n, 0.5, alternative="greater").pvalue if n > 0 else float("nan")
            rows.append([f"adaptive vs {STRATEGY_LABEL.get(st, st)}", m, len(common),
                         f"{np.mean(d) * 100:+.2f}" if len(d) else "–",
                         "–" if np.isnan(p_w) else f"{p_w:.4f}",
                         f"{wins}/{losses}/{len(ds) - n}",
                         "–" if np.isnan(p_s) else f"{p_s:.4f}"])
    note = ("Wilcoxon: one-sided signed-rank over (setting, seed) pairs, H1: adaptive < other. "
            "Sign test: one-sided binomial test over setting-level means "
            "(wins/losses/ties; ties dropped). Δ = mean(adaptive − other) ×100.")
    return note + "\n\n" + md_table(
        ["comparison", "metric", "#pairs", "Δ", "Wilcoxon p", "W/L/T (settings)", "sign-test p"], rows)


def signal_match(df):
    present = [s for s in FIXED_SIGNALS if s in set(df["strategy"])]
    if len(present) < len(FIXED_SIGNALS):
        return f"_Needs all fixed signals {FIXED_SIGNALS}; found {present}._"
    ps = df[df["strategy"].isin(FIXED_SIGNALS)].groupby(["dataset", "strategy"])[["dp", "eo"]].mean()
    ps["dp_eo"] = ps["dp"] + ps["eo"]
    rows, hits, total = [], 0, 0
    for ds in [d for d in SETTINGS if d in set(df["dataset"])]:
        sub = ps.xs(ds, level="dataset") if ds in ps.index.get_level_values(0) else None
        if sub is None or len(sub) < len(FIXED_SIGNALS):
            continue
        vmin = sub["dp_eo"].min()
        tied = sorted(sub.index[np.isclose(sub["dp_eo"], vmin, atol=1e-9)])
        best = tied[0] if len(tied) == 1 else "tie(" + ",".join(tied) + ")"
        diag = DIAG_CHOICE.get(ds, "?")
        ok = len(tied) == 1 and best == diag
        hits += int(ok); total += 1
        vals = " / ".join(f"{s}={sub.loc[s, 'dp_eo'] * 100:.2f}" for s in FIXED_SIGNALS)
        rows.append([ds, vals, best, diag + (" (+ℓ mix)" if ds.endswith("_g") else ""),
                     "✓" if ok else "✗"])
    return (md_table(["setting", "dp+eo ×100 (boundary / degree / loss)", "best fixed signal",
                      "diagnostic choice", "match"], rows)
            + f"\n\nMatch rate: {hits}/{total} (ties count as no match)")


def parse_rewire_logs(pattern, dataset="pokec_z"):
    """Return {(frac, seed): (h, r_b)} from '[Rewire]' log lines of `dataset`.

    The dataset is taken from the preceding 'FairGate | Dataset: X' header line.
    """
    out = {}
    if not pattern:
        return out
    rx = re.compile(r"\[Rewire\] frac=([\d.]+) seed=(\d+).*after: h=([\d.]+) r_b=([\d.]+)")
    rx_ds = re.compile(r"FairGate \| Dataset: (\S+)")
    for f in glob.glob(pattern):
        cur = None
        with open(f, errors="ignore") as fh:
            for line in fh:
                md = rx_ds.search(line)
                if md:
                    cur = md.group(1)
                m = rx.search(line)
                if m and cur == dataset:
                    out[(float(m.group(1)), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)))
    return out


def sweep_table(df, rewire_logs):
    df = df.copy()
    df["rewire_frac"] = df.get("rewire_frac", 0.0)
    df["rewire_frac"] = df["rewire_frac"].fillna(0.0).astype(float)
    strategies = [s for s in ["uniform", "boundary", "adaptive"] if s in set(df["strategy"])]
    g = df.groupby(["rewire_frac", "strategy"])
    stats = g[["dp", "eo", "roc_auc"]].agg(["mean", "std"])
    graph = df.groupby("rewire_frac")[[c for c in ["graph_h", "graph_r_b"] if c in df.columns]].mean()
    rows = []
    for f in sorted(df["rewire_frac"].unique()):
        rs = int(df.loc[df["rewire_frac"] == f, "rewire_seed"].iloc[0]) if "rewire_seed" in df.columns else 0
        if (f, rs) in rewire_logs:
            h, rb = rewire_logs[(f, rs)]; src = "log"
        elif "graph_h" in graph.columns and f in graph.index:
            h, rb = graph.loc[f, "graph_h"], graph.loc[f, "graph_r_b"]; src = "csv"
        else:
            h, rb, src = float("nan"), float("nan"), "–"
        for st in strategies:
            if (f, st) not in stats.index:
                continue
            r = stats.loc[(f, st)]
            rows.append([f"{f:.1f}", f"{h:.4f}", f"{rb:.4f}", src, STRATEGY_LABEL.get(st, st),
                         fmt(r[("dp", "mean")], r[("dp", "std")]),
                         fmt(r[("eo", "mean")], r[("eo", "std")]),
                         fmt(r[("roc_auc", "mean")], r[("roc_auc", "std")])])
    return md_table(["rewire_frac", "h", "r_b", "h/r_b source", "strategy",
                     "dp ×100", "eo ×100", "auc ×100"], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--sweep_logs", default=None,
                    help="glob of log files with [Rewire] lines (optional)")
    args = ap.parse_args()

    alloc = load(os.path.join(args.outputs, "wsdm_alloc_per_seed.csv"), unrewired_only=True)
    sweep = load(os.path.join(args.outputs, "wsdm_sweep_per_seed.csv"), extra_keys=("rewire_frac",))

    parts = ["# WSDM'27 short — allocation experiments report", ""]
    if alloc is not None:
        parts += ["## 1. Setting × strategy", "",
                  f"Rows: {len(alloc)} (after de-duplicating (dataset, strategy, seed); last kept). "
                  "`fixed (boundary)` = `--gating_mode_override boundary`.", "",
                  setting_table(alloc), "",
                  "## 2. Group averages (×100; mean of per-setting means)", "",
                  group_averages(alloc), "",
                  "## 3. Paired tests: adaptive vs. each strategy", "",
                  paired_tests(alloc), "",
                  "## 4. Best fixed signal vs. diagnostic choice", "",
                  signal_match(alloc), ""]
    else:
        parts += ["_outputs/wsdm_alloc_per_seed.csv not found._", ""]
    if sweep is not None:
        parts += ["## 5. Cross-group rewiring sweep (pokec_z)", "",
                  sweep_table(sweep, parse_rewire_logs(args.sweep_logs)), ""]
    else:
        parts += ["_outputs/wsdm_sweep_per_seed.csv not found._", ""]

    out = os.path.join(args.outputs, "wsdm_report.md")
    os.makedirs(args.outputs, exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(parts))
    print(f"[aggregate] wrote {out}")


if __name__ == "__main__":
    main()
