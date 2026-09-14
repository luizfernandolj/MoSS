# Statistical analysis

How the real-data ranking is turned into a checkable statistical claim
rather than a table of means, and how to reproduce it. Methodology only —
no current findings are stated here, so this page doesn't go stale when a
re-run changes the numbers; see the paper draft or a dated report for those.

## Why not just compare means

A table of per-method mean error answers "which number is smaller," not
whether the gap is larger than the datasets' own spread would produce by
chance. `stats.py` runs three tests, each answering a different question,
over the real-data runs' per-(dataset, method) mean absolute error.

## The three tests (`stats.py`)

**Friedman's test** (`stats.friedman`) — the omnibus test: is there a real
difference *anywhere* in the ranking, across datasets (blocks) and methods
(treatments)? `scipy.stats.friedmanchisquare` over each method's per-dataset
mean error. Answering yes is what licenses reading anything into the
pairwise comparison below (Demsar, 2006).

**Nemenyi's post-hoc** (`stats.nemenyi`) — which *pairs* of that ranking
actually differ. Computes one critical difference (via the studentized
range distribution, `alpha=0.05` by default) that every pair of average
ranks is checked against, and groups methods into *cliques*: maximal runs of
adjacent-by-rank methods whose two extremes are still within the critical
difference, i.e. not distinguishable from one another. Ranks, not raw means,
because the datasets being compared across are not on a comparable error
scale, and rank is the one summary that doesn't care.

**A signed-rank test** (`stats.pairwise_signed_rank`) — Wilcoxon's
signed-rank test, paired on dataset, between one named pair of methods.
Nemenyi's critical difference is conservative across every pair at once;
this is the same two methods compared at full power, for the one pair a
report wants to make a claim about by name. `statistical_report.py` names
this pair `HEADLINE_PAIR = ("DyS", "QuaDapt-MVN(DyS)")` — the same pair
[CONTEXT.md](../CONTEXT.md) itself uses to introduce "method": one base
quantifier, with and without an MVN-simulated reference standing in for the
real training scores.

## Reading the critical-difference diagram

`statistical_report.critical_difference_diagram` draws Demsar's (2006)
diagram from a `NemenyiResult`: each method placed on an axis at its average
rank (lower/left is better), and a horizontal bar under every clique the
post-hoc found. Two methods under the same bar are not shown to differ; two
methods under different bars are.

## Reproducing it

```
.venv/bin/python real_data.py          # produces results/runs/real-data.parquet
.venv/bin/python statistical_report.py # prints the three tests, writes the diagram
```

`statistical_report.report` prints the Friedman statistic and p-value, the
Nemenyi critical difference, and the headline pair's signed-rank statistic
and p-value, then writes `results/critical_difference_diagram.{png,pdf}`
(see [results.md](results.md)).

## Reference

Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple
Data Sets. *Journal of Machine Learning Research*, 7, 1–30.
