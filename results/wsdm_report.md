# WSDM'27 short — allocation experiments report

## 1. Setting × strategy

Rows: 315 (after de-duplicating (dataset, strategy, seed); last kept). `fixed (boundary)` = `--gating_mode_override boundary`.

#### acc (×100, mean±std over seeds)

| setting | uniform | random | fixed (boundary) | degree | loss | confidence | adaptive |
|---|---|---|---|---|---|---|---|
| pokec_z | 68.89±0.40 | 69.08±0.29 | 68.90±0.18 | 69.35±0.16 | 68.89±0.40 | 69.17±0.52 | 68.90±0.18 |
| pokec_n | 69.25±0.96 | 69.22±0.65 | 69.62±0.75 | 69.43±0.33 | 69.25±0.96 | 69.54±0.72 | 69.62±0.75 |
| pokec_z_g | 69.06±0.37 | 69.38±0.14 | 69.11±0.46 | 69.35±0.27 | 69.06±0.37 | 69.01±0.44 | 69.11±0.46 |
| pokec_n_g | 67.70±3.52 | 67.67±2.05 | 67.88±3.18 | 68.65±1.26 | 67.70±3.52 | 68.92±0.52 | 67.88±3.18 |
| german | 69.12±1.04 | 63.20±9.21 | 64.72±7.81 | 57.60±5.04 | 69.36±0.73 | 66.32±8.74 | 69.36±0.73 |
| credit | 77.52±2.31 | 77.54±1.88 | 78.10±0.77 | 78.50±0.31 | 74.61±1.13 | 77.71±0.82 | 78.50±0.31 |
| income | 81.60±0.07 | 81.10±0.58 | 81.02±0.38 | 81.09±0.66 | 81.22±0.23 | 81.32±0.27 | 81.02±0.38 |
| recidivism | 83.17±0.58 | 83.76±0.70 | 83.29±0.37 | 83.22±0.25 | 82.55±1.66 | 84.07±0.93 | 82.55±1.66 |
| nba | 71.92±1.98 | 73.62±0.61 | 63.66±10.02 | 71.36±1.79 | 67.98±3.50 | 72.49±1.65 | 67.98±3.50 |

#### roc_auc (×100, mean±std over seeds)

| setting | uniform | random | fixed (boundary) | degree | loss | confidence | adaptive |
|---|---|---|---|---|---|---|---|
| pokec_z | 75.76±0.11 | 75.60±0.34 | 75.40±0.49 | 75.72±0.35 | 75.76±0.11 | 75.59±0.08 | 75.40±0.49 |
| pokec_n | 73.72±0.56 | 73.68±0.61 | 73.83±0.82 | 73.59±0.54 | 73.72±0.56 | 73.81±0.38 | 73.83±0.82 |
| pokec_z_g | 74.87±0.77 | 75.52±0.64 | 75.15±0.87 | 75.68±0.77 | 74.87±0.77 | 75.10±0.68 | 75.15±0.87 |
| pokec_n_g | 73.63±0.42 | 73.42±0.41 | 73.30±0.99 | 72.67±1.81 | 73.63±0.42 | 73.55±0.23 | 73.30±0.99 |
| german | 64.75±2.01 | 62.40±3.73 | 62.80±3.99 | 57.99±3.04 | 61.03±2.13 | 64.28±4.72 | 61.03±2.13 |
| credit | 72.44±1.39 | 73.59±0.49 | 72.57±0.73 | 72.78±0.61 | 73.08±0.28 | 73.06±0.88 | 72.78±0.61 |
| income | 77.00±0.46 | 77.11±0.34 | 77.02±0.23 | 76.81±0.27 | 77.08±0.36 | 77.03±0.33 | 77.02±0.23 |
| recidivism | 87.28±0.43 | 86.59±1.33 | 87.55±0.11 | 86.18±0.76 | 85.62±2.15 | 87.49±0.59 | 85.62±2.15 |
| nba | 78.26±0.71 | 78.20±0.50 | 77.06±1.68 | 78.00±0.73 | 77.75±0.78 | 78.02±0.47 | 77.75±0.78 |

#### dp (×100, mean±std over seeds)

| setting | uniform | random | fixed (boundary) | degree | loss | confidence | adaptive |
|---|---|---|---|---|---|---|---|
| pokec_z | 1.71±0.82 | 0.88±0.76 | 1.42±0.69 | 2.32±1.11 | 1.71±0.82 | 0.72±0.69 | 1.42±0.69 |
| pokec_n | 4.21±1.11 | 4.09±1.07 | 4.00±0.52 | 2.48±1.37 | 4.21±1.11 | 4.35±1.02 | 4.00±0.52 |
| pokec_z_g | 5.43±1.75 | 6.98±1.43 | 6.00±2.39 | 7.35±1.88 | 5.43±1.75 | 7.40±1.94 | 6.00±2.39 |
| pokec_n_g | 3.13±1.60 | 2.77±1.29 | 2.77±1.33 | 2.84±1.37 | 3.13±1.60 | 3.45±1.06 | 2.77±1.33 |
| german | 5.23±8.64 | 6.16±4.85 | 4.26±2.57 | 7.54±7.41 | 2.55±1.92 | 4.79±3.83 | 2.55±1.92 |
| credit | 4.82±2.59 | 5.76±2.74 | 3.55±2.25 | 2.72±2.08 | 5.48±2.03 | 4.34±2.38 | 2.72±2.08 |
| income | 4.24±0.47 | 4.83±1.50 | 4.12±1.62 | 4.51±1.29 | 4.41±1.80 | 3.88±0.70 | 4.12±1.62 |
| recidivism | 6.94±0.48 | 6.19±0.68 | 7.60±0.33 | 6.28±0.39 | 6.34±1.60 | 6.75±0.82 | 6.34±1.60 |
| nba | 4.73±1.11 | 4.37±2.38 | 3.83±3.46 | 6.08±2.77 | 3.51±2.64 | 4.11±3.30 | 3.51±2.64 |

#### eo (×100, mean±std over seeds)

| setting | uniform | random | fixed (boundary) | degree | loss | confidence | adaptive |
|---|---|---|---|---|---|---|---|
| pokec_z | 1.09±0.76 | 0.68±0.58 | 0.85±0.50 | 1.82±0.98 | 1.09±0.76 | 0.59±0.35 | 0.85±0.50 |
| pokec_n | 6.32±1.42 | 5.73±1.51 | 6.63±1.29 | 5.15±1.09 | 6.32±1.42 | 6.23±1.38 | 6.63±1.29 |
| pokec_z_g | 0.56±0.84 | 1.98±1.60 | 1.69±1.60 | 2.30±1.63 | 0.56±0.84 | 2.10±1.45 | 1.69±1.60 |
| pokec_n_g | 7.60±2.45 | 6.58±2.37 | 7.18±2.08 | 7.14±2.14 | 7.60±2.45 | 8.12±1.13 | 7.18±2.08 |
| german | 5.34±8.80 | 5.63±3.86 | 3.57±1.90 | 10.40±7.92 | 2.44±1.63 | 2.82±2.49 | 2.44±1.63 |
| credit | 2.45±2.23 | 3.43±2.71 | 1.99±1.05 | 1.08±1.57 | 3.71±1.90 | 2.37±2.07 | 1.08±1.57 |
| income | 2.53±1.08 | 3.48±2.35 | 2.31±1.89 | 3.12±2.92 | 3.10±3.74 | 1.73±1.45 | 2.31±1.89 |
| recidivism | 3.87±1.10 | 3.77±0.40 | 4.56±0.32 | 3.96±0.83 | 3.39±1.10 | 4.27±0.72 | 3.39±1.10 |
| nba | 4.53±2.39 | 7.49±1.30 | 4.76±4.16 | 1.54±1.31 | 5.37±2.85 | 4.91±3.15 | 5.37±2.85 |

## 2. Group averages (×100; mean of per-setting means)

| group | strategy | #settings | acc | roc_auc | dp | eo |
|---|---|---|---|---|---|---|
| all | uniform | 9 | 73.14 | 75.30 | 4.49 | 3.81 |
| all | random | 9 | 72.73 | 75.12 | 4.67 | 4.31 |
| all | fixed (boundary) | 9 | 71.81 | 74.96 | 4.17 | 3.73 |
| all | degree | 9 | 72.06 | 74.38 | 4.68 | 4.06 |
| all | loss | 9 | 72.29 | 74.73 | 4.09 | 3.73 |
| all | confidence | 9 | 73.17 | 75.33 | 4.42 | 3.68 |
| all | adaptive | 9 | 72.77 | 74.65 | 3.72 | 3.44 |
| non-saturated | uniform | 6 | 72.34 | 74.57 | 3.92 | 3.42 |
| non-saturated | random | 6 | 72.33 | 74.82 | 4.22 | 3.65 |
| non-saturated | fixed (boundary) | 6 | 72.44 | 74.55 | 3.64 | 3.44 |
| non-saturated | degree | 6 | 72.73 | 74.54 | 3.70 | 3.43 |
| non-saturated | loss | 6 | 71.79 | 74.69 | 4.06 | 3.73 |
| non-saturated | confidence | 6 | 72.61 | 74.69 | 4.02 | 3.52 |
| non-saturated | adaptive | 6 | 72.50 | 74.58 | 3.51 | 3.29 |
| saturated (german, recidivism, nba) | uniform | 3 | 74.74 | 76.76 | 5.63 | 4.58 |
| saturated (german, recidivism, nba) | random | 3 | 73.52 | 75.73 | 5.57 | 5.63 |
| saturated (german, recidivism, nba) | fixed (boundary) | 3 | 70.56 | 75.80 | 5.23 | 4.30 |
| saturated (german, recidivism, nba) | degree | 3 | 70.73 | 74.06 | 6.63 | 5.30 |
| saturated (german, recidivism, nba) | loss | 3 | 73.30 | 74.80 | 4.13 | 3.73 |
| saturated (german, recidivism, nba) | confidence | 3 | 74.29 | 76.60 | 5.21 | 4.00 |
| saturated (german, recidivism, nba) | adaptive | 3 | 73.30 | 74.80 | 4.13 | 3.73 |

## 3. Paired tests: adaptive vs. each strategy

Wilcoxon: one-sided signed-rank over (setting, seed) pairs, H1: adaptive < other. Sign test: one-sided binomial test over setting-level means (wins/losses/ties; ties dropped). Δ = mean(adaptive − other) ×100.

| comparison | metric | #pairs | Δ | Wilcoxon p | W/L/T (settings) | sign-test p |
|---|---|---|---|---|---|---|
| adaptive vs uniform | dp | 45 | -0.78 | 0.0767 | 8/1/0 | 0.0195 |
| adaptive vs uniform | eo | 45 | -0.37 | 0.3911 | 6/3/0 | 0.2539 |
| adaptive vs random | dp | 45 | -0.96 | 0.0169 | 7/2/0 | 0.0898 |
| adaptive vs random | eo | 45 | -0.87 | 0.0243 | 6/3/0 | 0.2539 |
| adaptive vs fixed (boundary) | dp | 45 | -0.46 | 0.0542 | 4/0/5 | 0.0625 |
| adaptive vs fixed (boundary) | eo | 45 | -0.29 | 0.0584 | 3/1/5 | 0.3125 |
| adaptive vs degree | dp | 45 | -0.96 | 0.0298 | 6/2/1 | 0.1445 |
| adaptive vs degree | eo | 45 | -0.62 | 0.2616 | 5/3/1 | 0.3633 |
| adaptive vs loss | dp | 45 | -0.37 | 0.0449 | 5/1/3 | 0.1094 |
| adaptive vs loss | eo | 45 | -0.29 | 0.1144 | 4/2/3 | 0.3438 |
| adaptive vs confidence | dp | 45 | -0.70 | 0.0238 | 7/2/0 | 0.0898 |
| adaptive vs confidence | eo | 45 | -0.24 | 0.0943 | 5/4/0 | 0.5000 |

## 4. Best fixed signal vs. diagnostic choice

| setting | dp+eo ×100 (boundary / degree / loss) | best fixed signal | diagnostic choice | match |
|---|---|---|---|---|
| pokec_z | boundary=2.28 / degree=4.14 / loss=2.80 | boundary | boundary | ✓ |
| pokec_n | boundary=10.63 / degree=7.62 / loss=10.53 | degree | boundary | ✗ |
| pokec_z_g | boundary=7.70 / degree=9.64 / loss=5.98 | loss | boundary (+ℓ mix) | ✗ |
| pokec_n_g | boundary=9.95 / degree=9.98 / loss=10.73 | boundary | boundary (+ℓ mix) | ✓ |
| german | boundary=7.83 / degree=17.93 / loss=4.99 | loss | loss | ✓ |
| credit | boundary=5.54 / degree=3.80 / loss=9.20 | degree | degree | ✓ |
| income | boundary=6.43 / degree=7.62 / loss=7.51 | boundary | boundary | ✓ |
| recidivism | boundary=12.16 / degree=10.24 / loss=9.73 | loss | loss | ✓ |
| nba | boundary=8.59 / degree=7.62 / loss=8.88 | degree | loss | ✗ |

Match rate: 6/9 (ties count as no match)

## 5. Cross-group rewiring sweep (pokec_z)

| rewire_frac | h | r_b | h/r_b source | strategy | dp ×100 | eo ×100 | auc ×100 |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.9532 | 0.3689 | csv | uniform | 1.71±0.82 | 1.09±0.76 | 75.76±0.11 |
| 0.0 | 0.9532 | 0.3689 | csv | fixed (boundary) | 1.42±0.69 | 0.85±0.50 | 75.40±0.49 |
| 0.0 | 0.9532 | 0.3689 | csv | adaptive | 1.42±0.69 | 0.85±0.50 | 75.40±0.49 |
| 0.1 | 0.8631 | 0.7997 | log | uniform | 1.39±1.00 | 2.50±1.10 | 75.36±0.23 |
| 0.1 | 0.8631 | 0.7997 | log | fixed (boundary) | 1.32±0.75 | 2.20±1.25 | 74.94±0.42 |
| 0.1 | 0.8631 | 0.7997 | log | adaptive | 1.77±0.79 | 3.07±1.30 | 75.10±0.34 |
| 0.2 | 0.7730 | 0.9158 | log | uniform | 3.71±2.33 | 5.56±2.15 | 73.12±1.04 |
| 0.2 | 0.7730 | 0.9158 | log | fixed (boundary) | 3.93±0.86 | 5.85±0.94 | 73.79±0.29 |
| 0.2 | 0.7730 | 0.9158 | log | adaptive | 3.71±2.33 | 5.56±2.15 | 73.12±1.04 |
| 0.3 | 0.6829 | 0.9612 | log | uniform | 0.84±1.08 | 2.87±1.74 | 73.88±0.21 |
| 0.3 | 0.6829 | 0.9612 | log | fixed (boundary) | 1.31±0.40 | 4.06±1.14 | 73.85±0.32 |
| 0.3 | 0.6829 | 0.9612 | log | adaptive | 0.84±1.08 | 2.87±1.74 | 73.88±0.21 |
| 0.4 | 0.5928 | 0.9812 | log | uniform | 2.57±1.52 | 1.13±1.25 | 72.02±0.89 |
| 0.4 | 0.5928 | 0.9812 | log | fixed (boundary) | 3.09±0.84 | 0.76±0.90 | 72.81±0.44 |
| 0.4 | 0.5928 | 0.9812 | log | adaptive | 2.57±1.52 | 1.13±1.25 | 72.02±0.89 |
| 0.5 | 0.5027 | 0.9903 | log | uniform | 6.97±2.86 | 4.27±1.89 | 70.86±0.44 |
| 0.5 | 0.5027 | 0.9903 | log | fixed (boundary) | 7.77±0.49 | 4.97±0.88 | 71.23±0.24 |
| 0.5 | 0.5027 | 0.9903 | log | adaptive | 6.97±2.86 | 4.27±1.89 | 70.86±0.44 |
| 0.6 | 0.4126 | 0.9944 | log | uniform | 0.76±0.46 | 2.48±1.03 | 69.10±1.07 |
| 0.6 | 0.4126 | 0.9944 | log | fixed (boundary) | 1.19±1.04 | 1.98±1.83 | 68.49±1.66 |
| 0.6 | 0.4126 | 0.9944 | log | adaptive | 0.76±0.46 | 2.48±1.03 | 69.10±1.07 |
| 0.8 | 0.2323 | 0.9986 | log | uniform | 5.13±1.24 | 1.07±0.56 | 65.97±0.41 |
| 0.8 | 0.2323 | 0.9986 | log | fixed (boundary) | 3.57±2.21 | 0.79±1.12 | 65.66±0.86 |
| 0.8 | 0.2323 | 0.9986 | log | adaptive | 5.13±1.24 | 1.07±0.56 | 65.97±0.41 |
