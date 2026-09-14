# Results: where they land, and what they show

## Run tables (`results/runs/*.parquet`)

Written by `sweep.py` and `real_data.py` ([running.md](running.md)),
git-ignored — regenerate them rather than expecting them after a fresh
clone. Each row is one *run*: one bag, one method, one estimate
(`runs.py`'s own vocabulary — see [CONTEXT.md](../CONTEXT.md) for what a
run, a method and a bag are). `runs.load_labelled(kind)` is how every
downstream script reads one of these tables; raw column names are documented
in `runs.py`.

| File | Produced by | Kind |
| --- | --- | --- |
| `synthetic.parquet` | `sweep.py` | `runs.SYNTHETIC` |
| `real-data.parquet` | `real_data.py` | `runs.REAL_DATA` |
| `measure-ablation.parquet` | `sweep.py --ablation` | `runs.MEASURE_ABLATION` |

## Figures (`results/*.png`, `results/*.pdf`)

Committed — these are the artifacts the paper draws from, produced by
`export_grid_matplotlib.py` and `statistical_report.py`
([running.md](running.md)).

**`grid_matplotlib_stylized.{png,pdf}`** — the published 3×3 grid: rows are
bag simulator (Uniform / MVN / Dirichlet), columns are reference merging
factor (0.25 / 0.5 / 0.75), and each panel plots one line per method (a base
quantifier crossed with a method simulator — [CONTEXT.md](../CONTEXT.md)'s
"method") of mean absolute error against bag merging factor. The vertical
dashed line in each panel marks where the bag's merging factor equals the
panel's reference merging factor — reference and bag drawn at matched
difficulty.

**`boxplot_mtr_backline.{png,pdf}`** — mean-absolute-error distribution
grouped by reference merging factor, one box per method simulator,
restricted to bags no harder than the reference they're matched against
(`bag_merging_factor <= reference_merging_factor`).

**`boxplot_distribution_xaxis.{png,pdf}`** — the same restricted set of
runs, grouped by bag simulator instead of reference merging factor.

**`critical_difference_diagram.{png,pdf}`** — Demsar's (2006)
critical-difference diagram over the real-data ranking: each method placed
on an axis by its average rank across datasets (lower is better), with a
horizontal bar bracketing every *clique* — a maximal group of methods whose
ranks are not far enough apart to call significantly different (the Nemenyi
post-hoc). See [analysis.md](analysis.md) for the statistics behind this
figure.

## The interactive dashboard

`streamlit run dashboard.py` ([running.md](running.md)) draws the same grid
and two boxplots as the figures above, but over the synthetic runs you
choose interactively rather than the fixed published subset — pick the
reference/bag simulator, reference merging factor, and which method
simulators and base quantifiers to include.

## `results/void-mlquantify-0.2.0/`

Archived results from before the mlquantify 0.5.1 port
([ADR-0001](adr/0001-pin-mlquantify-0-5-1-and-void-earlier-results.md)).
Void — kept for reference, not regenerated or read by any script here.
