"""Whether one method's ranking beats another's is a claim, not a mean (#12).

A table of per-method means answers "which number is smaller" and nothing
about whether the gap is more than the datasets' own spread would produce by
chance. Friedman's test (is there a real difference anywhere in the ranking),
Nemenyi's post-hoc (which pairs of that ranking actually differ) and a
signed-rank test (one named pair, at full power rather than Nemenyi's
family-wise-conservative one) are the three ways this module turns "method X
wins" into a number a reader can check.

Every function here takes runs the way ``runs.load_labelled`` returns them —
one row per estimate, carrying ``dataset``, ``method`` and ``absolute_error``
— not a table already collapsed to one row per method. Collapsing across
repetitions and bags into one mean per (dataset, method) is this module's job,
the same reasoning ``grid.py`` gives for a figure panel: a caller that
averaged for itself before calling in would be the same duplication one level
down, and the one place most likely to disagree with this module about which
rows share a mean.

Ranking, not the raw mean, is what Nemenyi's critical difference is built
from (Demsar, 2006): the datasets being compared across are not on a
comparable error scale, and a rank is the one summary that does not care.

Drawing a critical-difference diagram from a :class:`NemenyiResult` is not
this module's job: the cliques a diagram brackets are data, computed here,
but turning them into a picture belongs to ``statistical_report.py``, the
same split ``grid.py`` and ``export_grid_matplotlib.py`` already draw for the
synthetic figure.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

#: Demsar (2006, Table 5)'s critical values are a studentized range quantile
#: at infinite degrees of freedom: the post-hoc's reference distribution
#: assumes an unbounded number of datasets, not the handful at hand, so this
#: is the quantile's own asymptote rather than a stand-in for it.
_INFINITE_DF = np.inf


def _dataset_method_means(labelled_runs):
    """Mean absolute error per (dataset, method): the block/treatment matrix every test below shares."""
    means = (
        labelled_runs.groupby(["dataset", "method"], observed=True)["absolute_error"]
        .mean()
        .unstack("method")
    )
    unscored = means.columns[means.isna().any()].tolist()
    if unscored:
        raise ValueError(
            f"method(s) {unscored} have no valid estimate on at least one "
            "dataset; a ranking comparison needs every method scored on "
            "every dataset"
        )
    return means


def dataset_method_ranks(labelled_runs):
    """Each method's rank within every dataset — the matrix :func:`average_ranks` collapses to one number per method.

    Ties within a dataset share the midpoint rank (pandas' ``"average"``
    method), which is what keeps a tie from silently favouring whichever
    method happens to sort first. Public for a reader that wants the spread
    across datasets rather than the average alone — a rank boxplot, e.g.
    """
    means = _dataset_method_means(labelled_runs)
    return means.rank(axis=1, method="average")


def average_ranks(labelled_runs):
    """Each method's average rank across datasets, 1 best, sorted best-first."""
    return (
        dataset_method_ranks(labelled_runs)
        .mean(axis=0)
        .rename("average_rank")
        .sort_values()
    )


@dataclass(frozen=True)
class FriedmanResult:
    """The omnibus test: do the methods' ranks differ by more than chance across datasets."""

    statistic: float
    p_value: float
    n_datasets: int
    n_methods: int
    average_ranks: pd.Series


def friedman(labelled_runs):
    """Friedman's test across datasets (blocks) and methods (treatments).

    Answers one question only — "is there a real difference somewhere in
    this ranking" — which is what licenses reading anything into the pairwise
    comparisons :func:`nemenyi` makes next (Demsar, 2006).
    """
    means = _dataset_method_means(labelled_runs)
    statistic, p_value = scipy_stats.friedmanchisquare(
        *(means[method] for method in means.columns)
    )
    return FriedmanResult(
        statistic=float(statistic),
        p_value=float(p_value),
        n_datasets=means.shape[0],
        n_methods=means.shape[1],
        average_ranks=average_ranks(labelled_runs),
    )


@dataclass(frozen=True)
class NemenyiResult:
    """Demsar's (2006) post-hoc: which pairs of a Friedman-tested ranking actually differ.

    ``cliques`` names every maximal group of methods, ordered by rank, whose
    two extremes are still not significantly apart — the groups a
    critical-difference diagram draws a bar under. A method with no
    indistinguishable neighbour belongs to no clique: nothing to bracket.
    """

    average_ranks: pd.Series
    critical_difference: float
    alpha: float
    n_datasets: int
    n_methods: int
    cliques: tuple

    def significant(self, method_a, method_b):
        """Whether the two methods' average ranks are farther apart than the critical difference."""
        gap = abs(self.average_ranks[method_a] - self.average_ranks[method_b])
        return gap > self.critical_difference


def nemenyi(labelled_runs, alpha=0.05):
    """The post-hoc comparison :func:`friedman` licenses: one critical difference for every pair.

    ``q_alpha`` is the studentized range distribution's own quantile divided
    by sqrt(2) rather than a copied table (Demsar, 2006, Table 5) — the two
    agree to three decimal places for every method count from 2 to 10, and a
    computed quantile has no upper bound on the method count a table would.
    """
    ranks = average_ranks(labelled_runs)
    n_datasets = _dataset_method_means(labelled_runs).shape[0]
    n_methods = len(ranks)
    q_alpha = scipy_stats.studentized_range.ppf(1 - alpha, n_methods, _INFINITE_DF) / np.sqrt(2)
    critical_difference = float(
        q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
    )
    return NemenyiResult(
        average_ranks=ranks,
        critical_difference=critical_difference,
        alpha=alpha,
        n_datasets=n_datasets,
        n_methods=n_methods,
        cliques=_cliques(ranks, critical_difference),
    )


def _cliques(sorted_ranks, critical_difference):
    """Maximal runs of methods, by rank, whose extremes are still within the critical difference.

    A method may belong to more than one maximal run — Nemenyi's guarantee is
    pairwise, not transitive, so a method can be indistinguishable from each
    of two neighbours that are themselves significantly apart. ``sorted_ranks``
    must already be ascending (:func:`average_ranks`'s own order), since the
    critical difference is only ever checked between a run's two extremes.
    """
    names = list(sorted_ranks.index)
    values = sorted_ranks.to_numpy()
    n = len(values)
    spans = []
    for start in range(n):
        end = start
        while end + 1 < n and values[end + 1] - values[start] <= critical_difference:
            end += 1
        spans.append((start, end))
    maximal = [
        span
        for span in spans
        if span[1] > span[0]
        and not any(
            other != span and other[0] <= span[0] and other[1] >= span[1]
            for other in spans
        )
    ]
    return tuple(tuple(names[start : end + 1]) for start, end in maximal)


@dataclass(frozen=True)
class PairwiseComparison:
    """One named pair, compared at full power rather than Nemenyi's family-wise-conservative one."""

    method_a: str
    method_b: str
    statistic: float
    p_value: float
    n_datasets: int


def pairwise_signed_rank(labelled_runs, method_a, method_b):
    """Wilcoxon's signed-rank test between two methods' per-dataset mean error.

    For the one pair a report wants to make a claim about by name — the
    "headline pair" — rather than every pair :func:`nemenyi` is conservative
    across at once. Paired on dataset: the same datasets score both methods,
    so the test is over their per-dataset difference, not two independent
    samples.

    Filters ``labelled_runs`` to the two named methods before checking
    coverage, so a coverage gap in some third method ``labelled_runs`` happens
    to carry — irrelevant to this pair — never blocks a comparison that has
    nothing to do with it.
    """
    present = set(labelled_runs["method"])
    for method in (method_a, method_b):
        if method not in present:
            raise ValueError(f"unknown method {method!r}; expected one of {sorted(present)}")
    pair_runs = labelled_runs[labelled_runs["method"].isin((method_a, method_b))]
    means = _dataset_method_means(pair_runs)
    statistic, p_value = scipy_stats.wilcoxon(means[method_a], means[method_b])
    return PairwiseComparison(
        method_a=method_a,
        method_b=method_b,
        statistic=float(statistic),
        p_value=float(p_value),
        n_datasets=means.shape[0],
    )
