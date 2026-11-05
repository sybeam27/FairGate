### FairGate: Multi-level Fairness Intervention via Gating Intensity for Graph Neural Networks

## Requirements

FairGate itself (`utils/`, `run.py --mode fairgate`, `scripts/aggregate.py`):

- Python ≥ 3.9
- `torch`, `torch_geometric`
- `numpy`, `pandas`, `scipy`, `scikit-learn`

Baselines (`algorithms/`, `run.py --mode baselines`) additionally use
`torch_scatter`, `torch_sparse`, `networkx`, `gensim`, `deeprobust`, `tqdm`,
`matplotlib`, `six`, `joblib`, `cytoolz`, `psutil` (depending on the baseline).

## Data

Datasets are read from `./data/` relative to the repository root:

| setting | files |
|---|---|
| `pokec_z`, `pokec_z_g` | `data/pokec/region_job.csv`, `data/pokec/region_job_relationship.txt` |
| `pokec_n`, `pokec_n_g` | `data/pokec/region_job_2.csv`, `data/pokec/region_job_2_relationship.txt` |
| `nba` | `data/NBA/nba.csv`, `data/NBA/nba_relationship.txt` |
| `german` | `data/german/german.csv`, `data/german/german_edges.txt` |
| `credit` | `data/credit/credit.csv`, `data/credit/credit_edges.txt` |
| `recidivism` | `data/bail/bail.csv`, `data/bail/bail_edges.txt` |
| `income` | `data/income/income.csv`, `data/income/income_edges.txt` |

`*_g` settings use gender as the sensitive attribute; the others use region / country / the loader default.

## Running

All 9 settings, 5 seeds, configuration from `FAIRGATE_CONFIGS` in `run.py`:

```bash
python run.py --mode fairgate --device cuda:0 --run_name my_run
```

A single setting directly:

```bash
python -m utils.train_fairgate --dataset german --device cpu --runs 5
```

Outputs:

- `outputs/<run_name>.csv` — one summary row (mean/std over seeds) per configuration.
- `outputs/<run_name>_per_seed.csv` — one row per seed (`seed_used = seed + run index`) with every CLI setting.

Report for the allocation / sweep experiments:

```bash
python scripts/aggregate.py
```

## Allocation strategies and flags

| strategy | flags |
|---|---|
| uniform (no allocation) | `--fiw_weight_mode uniform` |
| random (same budget) | `--gating_mode_override random` |
| fixed boundary | `--gating_mode_override boundary` |
| fixed degree | `--gating_mode_override degree` |
| fixed loss ℓ(v) | `--gating_mode_override loss` |
| confidence | `--gating_mode_override confidence` |
| adaptive (FairGate) | `--gating_mode_override adaptive` (default) |

Other flags:

| flag | meaning |
|---|---|
| `--strict_label_budget` | non-train labels are masked (−1) before the output-level loss, so its EO term uses train labels only |
| `--rewire_frac f --rewire_seed s` | after loading, rewire a fraction `f` of observed intra-group edges into cross-group edges (controlled r_b experiment) |
| `--ablation_mode {full_loss, rep_out, struct_out, struct_rep, out_only, none, ...}` | objective ablation |
| `--fiw_adaptive` | optional validation-based selection among FIW candidates (off by default) |

## Implementation facts

- **Graph diagnostics** (h, boundary ratio r_b, degree gap) are computed from the sensitive attribute s of *all* nodes on the graph *with self-loops*.
- **Adaptive branch order**: saturated (r_b ≥ 0.9 → ℓ(v)) → degree-skewed (deg gap > 0.2 → degree) → h < 0.5 (boundary mixed with ℓ(v)) → clustered (r_b < 0.5 → boundary) → default (variance-mixed boundary/degree).
- **ℓ(v)** is BCE × (0.5 + cross-group neighbor fraction), min-max normalized over labelled *train* nodes only; it is 0 elsewhere. When ℓ(v) is the gating signal, the top-k gate is taken among those train nodes, with k = max(1, round((1 − q)·|train labelled|)). All other signals gate over all nodes.
- **α / β** use the variance rule in every setting (`alpha_beta_mode = variance`). Edge intervention is **drop**: cross-group edges are dropped with probability p_drop, intra-group edges with 0.1·p_drop.
- **σ(v)** is the probability-interval width of the auxiliary uncertainty head, trained with λ_unc = 0.02 and width penalty λ_w = 0.05.
- **Checkpoint selection** uses validation Acc − 0.3·ΔDP − 0.7·ΔEO.
- **NBA** follows the FairGNN protocol, so its validation set equals its test set.
- `_auto_config_from_graph_stats` only names the regime in the log; it does not change training.
