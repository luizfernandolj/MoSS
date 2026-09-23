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

### Bag size sensitivity

Everything above this point reads the bag_size=100 slice of the real-data
table, published before the sweep grew a bag-size axis (#16) and left exactly
as it was. The figures below instead read all four published bag sizes
(100 / 500 / 1000 / 5000, `sweep.BAG_SIZES`) together, produced by
`bag_size_sensitivity_report.py` from the same `real-data.parquet`
([running.md](running.md)) — the question here is whether a bigger bag
changes what the baseline above already says, not a replacement for it.

**`mae_by_bag_size.{png,pdf}`** — one subplot per dataset: mean absolute
error (every method and repetition averaged together) against bag size, on
a log axis. Error falls as bags grow for six of the eight datasets, most
sharply for `haberman_survival` — the smallest pool in the table, and the
one bootstrap replication (#18) has the most room to help, from an MAE of
0.32 at bag_size=100 down to 0.11 at bag_size=1000. `banknote_authentication`
is the exception: its error keeps falling through bag_size=1000 (0.075 to
0.071) and then rises again at bag_size=5000 (0.105), rather than continuing
to fall. `haberman_survival` and `pima_diabetes` have no bag_size=5000 point
at all — neither pool can fill a bag that big even at #18's
bootstrap-replication cap, so bag_size=5000 produced no valid estimate for
either dataset (`real_data.draw_bag`).

**`critical_difference_diagram_bag_size_{100,500,1000,5000}.{png,pdf}`** —
the same kind of Nemenyi ranking as `critical_difference_diagram.{png,pdf}`
above, computed independently at each bag size rather than pooled into one.
`haberman_survival` and `pima_diabetes` are left out of the bag_size=5000
ranking specifically, for the same reason they have no bag_size=5000 point
above; every other bag size's ranking covers all eight datasets. The ranking
itself is not stable across bag sizes: DyS leads at bag_size 100 and 500
(average rank 2.25 and 2.75), HDy overtakes it by bag_size 1000 (3.31 against
3.50), and by bag_size 5000 HDy and its QuaDapt-All(HDy) counterpart are
tied well clear of the rest (average rank 2.17, against DyS's 4.67) — a
bigger bag does not just shrink every method's error together, it changes
which method the ranking would recommend.

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
