# Running the experiments

Five entry points, each a plain script with a `__main__` block — no CLI
wrapper, no Makefile. All of them write into `results/` (see
[results.md](results.md) for what lands where). For the test suite, see
[testing.md](testing.md) instead — this page only covers the five scripts
below.

Run them from the repo root, in this order the first time: `sweep.py` and
`real_data.py` produce the run tables the other three read.

## 1. The synthetic sweep — `sweep.py`

```
.venv/bin/python sweep.py
```

Runs the published synthetic grid (`sweep.SYNTHETIC_SWEEP`, `n_jobs=-1` —
see [experiments.md](experiments.md) for the exact grid), validates the
output against both of
[ADR-0001](adr/0001-pin-mlquantify-0-5-1-and-void-earlier-results.md)'s
defect signatures, and saves it to `results/runs/synthetic.parquet`. Takes a
while — it's the full grid, on every core.

```
.venv/bin/python sweep.py --ablation
```

Runs the distance-measure ablation instead (`sweep.MEASURE_ABLATION_SWEEP`
— see [experiments.md](experiments.md)), and saves to
`results/runs/measure-ablation.parquet`.

## 2. The real-data sweep — `real_data.py`

```
.venv/bin/python real_data.py
```

Fetches (or reads from cache — see [setup.md](setup.md)) all eight binary
datasets, cross-validates a Random Forest on each, runs the published
real-data grid (`real_data.build_spec` — see [experiments.md](experiments.md)
for the exact grid), and saves to `results/runs/real-data.parquet`.

```
.venv/bin/python real_data.py --dataset mushroom
```

Runs one dataset only — useful while iterating, since fetching and
cross-validating all eight takes a while. `--dataset` accepts any key of
`real_data.DATASET_FETCHERS` — see [experiments.md](experiments.md) for the
full list.

```
.venv/bin/python real_data.py --multiclass
```

Runs the multiclass datasets
([#14](https://github.com/luizfernandolj/MoSS/issues/14),
[experiments.md](experiments.md) for which) instead of the eight binary
ones, through `real_data.build_multiclass_spec`. Cannot be combined with
`--dataset`.

## 3. The interactive dashboard — `dashboard.py`

```
.venv/bin/streamlit run dashboard.py
```

Needs `results/runs/synthetic.parquet` already on disk (run `sweep.py`
first). Opens a browser tab with the interactive grid and boxplots,
filterable by reference/bag simulator, reference merging factor, method
simulator and base quantifier.

## 4. The published figures — `export_grid_matplotlib.py`

```
.venv/bin/python export_grid_matplotlib.py
```

Also needs `results/runs/synthetic.parquet`. Renders the fixed (non-interactive)
versions of the grid and two boxplots used in the paper, six files total
under `results/` (PNG and PDF each).

## 5. The statistical report — `statistical_report.py`

```
.venv/bin/python statistical_report.py
```

Needs `results/runs/real-data.parquet` (run `real_data.py` first). Prints
the Friedman test, the Nemenyi critical difference, and the headline
pairwise signed-rank comparison to stdout, and writes the critical-difference
diagram under `results/`. See [analysis.md](analysis.md) for what these
numbers mean and how to read the diagram.
