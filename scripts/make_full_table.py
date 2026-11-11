"""
Build results/full_results.md: per-setting results for baselines and FairGate.

Inputs:
    results/baselines_per_setting.csv   (baseline mean/std per setting)
    results/wsdm_alloc_per_seed.csv     (FairGate per-seed runs)

FairGate             = strategy "adaptive" (gating_mode_override adaptive)
FairGate (uniform φ) = strategy "uniform"  (fiw_weight_mode uniform)
Only un-rewired runs (rewire_frac == 0) are used; duplicate
(dataset, strategy, seed_used) rows keep the last one, as in scripts/aggregate.py.

Usage:
    python scripts/make_full_table.py
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate import SETTINGS, load  # noqa: E402

BASELINES = ["FairGNN", "FairVGNN", "FairWalk", "CrossWalk", "FairGB", "NIFTY", "FairGT"]
FAIRGATE = {"FairGate": "adaptive", "FairGate (uniform φ)": "uniform"}
METRICS = [("Acc", "acc", "acc"), ("AUC", "auc", "roc_auc"),
           ("ΔDP", "dp", "dp"), ("ΔEO", "eo", "eo")]  # (label, baseline col, FairGate col)

NOTE = (
    "> Per-setting results (mean ± std over seeds 27–31). FairGate rows come from "
    "deterministic GPU runs (`run.py --mode fairgate --strict_label_budget`, see "
    "`scripts/run_wsdm.sh`). Baseline rows come from earlier runs with the same data "
    "splits and seeds, using hyperparameters from the original papers or official "
    "implementations. FairEdit and EDITS are omitted (out of GPU memory on Pokec or "
    "single-class collapse)."
)


def fmt(mean, std):
    return f"{mean:.4f}±{std:.3f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    args = p.parse_args()

    base = pd.read_csv(os.path.join(args.results, "baselines_per_setting.csv"))
    base = base.set_index(["setting", "method"])

    fg = load(os.path.join(args.results, "wsdm_alloc_per_seed.csv"), unrewired_only=True)
    fg = fg[fg["strategy"].isin(FAIRGATE.values())]
    g = fg.groupby(["dataset", "strategy"])
    fg_mean, fg_std, fg_n = g.mean(numeric_only=True), g.std(ddof=1, numeric_only=True), g.size()

    columns = BASELINES + list(FAIRGATE)
    means = {}  # (setting, metric label, column) -> mean
    lines = [NOTE, "", "| setting | metric | " + " | ".join(columns) + " |",
             "|---|---|" + "---|" * len(columns)]
    for s in SETTINGS:
        for label, bcol, fcol in METRICS:
            cells = []
            for m in BASELINES:
                mu, sd = base.loc[(s, m), bcol], base.loc[(s, m), f"{bcol}_std"]
                means[(s, label, m)] = mu
                cells.append(fmt(mu, sd))
            for name, st in FAIRGATE.items():
                if fg_n.loc[(s, st)] != 5:
                    raise SystemExit(f"{s}/{st}: expected 5 seeds, got {fg_n.loc[(s, st)]}")
                mu, sd = fg_mean.loc[(s, st), fcol], fg_std.loc[(s, st), fcol]
                means[(s, label, name)] = mu
                cells.append(fmt(mu, sd))
            lines.append(f"| {s} | {label} | " + " | ".join(cells) + " |")
    for label, _, _ in METRICS:
        avg = [sum(means[(s, label, c)] for s in SETTINGS) / len(SETTINGS) for c in columns]
        lines.append(f"| **Average (9 settings)** | {label} | "
                     + " | ".join(f"{v:.4f}" for v in avg) + " |")

    out = os.path.join(args.results, "full_results.md")
    with open(out, "w") as f:
        f.write("# Per-setting results\n\n" + "\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
