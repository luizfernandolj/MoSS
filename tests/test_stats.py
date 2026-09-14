"""Friedman, Nemenyi and a signed-rank test on a fixture whose ranking is known (#12).

Every fixture here is built by formula rather than measured, so the expected
average ranks, the Friedman p-value's sign and the Nemenyi cliques are known
ahead of time — the property #12's acceptance criterion asks a fixture prove,
and the same reason ``tests/conftest.py`` builds its run tables in code rather
than loading a committed file.
"""

import numpy as np
import pandas as pd
import pytest

import stats


def _row(*, dataset, method, absolute_error):
    return {"dataset": dataset, "method": method, "absolute_error": absolute_error}


def _frame(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# A fixture with a known ranking: A always best, B always middle, C always
# worst, by a gap (0.1) that swamps the per-dataset noise (at most 0.02).
# ---------------------------------------------------------------------------

DATASETS = ("d1", "d2", "d3", "d4", "d5", "d6")
_BASE_ERROR = {"A": 0.05, "B": 0.15, "C": 0.25}
_REPETITIONS = range(4)


@pytest.fixture
def known_ranking_runs():
    rows = []
    for dataset_idx, dataset in enumerate(DATASETS):
        for method, base_error in _BASE_ERROR.items():
            for repetition in _REPETITIONS:
                noise = 0.005 * ((dataset_idx + repetition) % 3)
                rows.append(
                    _row(
                        dataset=dataset,
                        method=method,
                        absolute_error=base_error + noise,
                    )
                )
    return _frame(rows)


# ---------------------------------------------------------------------------
# average_ranks / _dataset_method_means
# ---------------------------------------------------------------------------


def test_average_ranks_orders_methods_best_first_on_the_known_ranking(known_ranking_runs):
    ranks = stats.average_ranks(known_ranking_runs)

    assert list(ranks.index) == ["A", "B", "C"]
    assert ranks["A"] < ranks["B"] < ranks["C"]


def test_average_ranks_is_one_for_a_method_that_always_wins():
    # No noise at all: A beats B and C on every one of two datasets, so A's
    # average rank is exactly 1.
    runs_ = _frame(
        [
            _row(dataset="d1", method="A", absolute_error=0.1),
            _row(dataset="d1", method="B", absolute_error=0.2),
            _row(dataset="d2", method="A", absolute_error=0.1),
            _row(dataset="d2", method="B", absolute_error=0.2),
        ]
    )

    ranks = stats.average_ranks(runs_)

    assert ranks["A"] == 1.0
    assert ranks["B"] == 2.0


def test_average_ranks_splits_a_tie_at_the_midpoint():
    runs_ = _frame(
        [
            _row(dataset="d1", method="A", absolute_error=0.1),
            _row(dataset="d1", method="B", absolute_error=0.1),
            _row(dataset="d1", method="C", absolute_error=0.3),
        ]
    )

    ranks = stats.average_ranks(runs_)

    assert ranks["A"] == pytest.approx(1.5)
    assert ranks["B"] == pytest.approx(1.5)
    assert ranks["C"] == pytest.approx(3.0)


def test_average_ranks_averages_over_repetitions_before_ranking():
    # B's mean (0.2) is worse than A's (0.1) even though one of B's two
    # repetitions beats A outright — the mean is what gets ranked, not any
    # single run.
    runs_ = _frame(
        [
            _row(dataset="d1", method="A", absolute_error=0.1),
            _row(dataset="d1", method="A", absolute_error=0.1),
            _row(dataset="d1", method="B", absolute_error=0.0),
            _row(dataset="d1", method="B", absolute_error=0.4),
        ]
    )

    ranks = stats.average_ranks(runs_)

    assert ranks["A"] < ranks["B"]


def test_a_method_missing_a_dataset_entirely_is_refused_with_its_name():
    runs_ = _frame(
        [
            _row(dataset="d1", method="A", absolute_error=0.1),
            _row(dataset="d1", method="B", absolute_error=0.2),
            _row(dataset="d2", method="A", absolute_error=0.1),
            # B never scored on d2.
        ]
    )

    with pytest.raises(ValueError, match="B"):
        stats.average_ranks(runs_)


# ---------------------------------------------------------------------------
# friedman
# ---------------------------------------------------------------------------


def test_friedman_finds_a_significant_difference_on_the_known_ranking(known_ranking_runs):
    result = stats.friedman(known_ranking_runs)

    assert result.n_datasets == len(DATASETS)
    assert result.n_methods == 3
    assert result.p_value < 0.01
    assert list(result.average_ranks.index) == ["A", "B", "C"]


def test_friedman_finds_no_difference_under_a_balanced_latin_square():
    # Each dataset cyclically permutes which method comes first, second,
    # third, so every method's rank sum across the three datasets is equal
    # (6) — a real "no difference" case, as opposed to every value being
    # literally identical (which degenerates the tie-corrected statistic to
    # 0/0 rather than exercising this function).
    orders = [("A", "B", "C"), ("B", "C", "A"), ("C", "A", "B")]
    values = (0.1, 0.2, 0.3)
    rows = [
        _row(dataset=f"d{i}", method=method, absolute_error=value)
        for i, order in enumerate(orders)
        for method, value in zip(order, values)
    ]

    result = stats.friedman(_frame(rows))

    assert result.statistic == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)


def test_friedman_needs_at_least_three_methods():
    runs_ = _frame(
        [
            _row(dataset="d1", method="A", absolute_error=0.1),
            _row(dataset="d1", method="B", absolute_error=0.2),
            _row(dataset="d2", method="A", absolute_error=0.1),
            _row(dataset="d2", method="B", absolute_error=0.2),
        ]
    )

    with pytest.raises(ValueError, match="3"):
        stats.friedman(runs_)


# ---------------------------------------------------------------------------
# nemenyi and its critical difference / cliques
# ---------------------------------------------------------------------------


def test_nemenyi_critical_difference_matches_the_studentized_range_formula(known_ranking_runs):
    from scipy.stats import studentized_range

    result = stats.nemenyi(known_ranking_runs, alpha=0.05)

    q_alpha = studentized_range.ppf(0.95, 3, np.inf) / np.sqrt(2)
    expected = q_alpha * np.sqrt(3 * 4 / (6 * len(DATASETS)))
    assert result.critical_difference == pytest.approx(expected)
    assert result.alpha == 0.05
    assert result.n_datasets == len(DATASETS)
    assert result.n_methods == 3


def test_nemenyi_is_conservative_between_neighbours_but_still_separates_the_extremes(
    known_ranking_runs,
):
    # A, B, C rank exactly 1, 2, 3 on every one of six datasets (the noise in
    # the fixture is common to all three methods within a block, so it never
    # changes the order). The critical difference at three methods and six
    # datasets (~1.35) exceeds the gap between neighbours (1.0) but not the
    # gap between the extremes (2.0) — Nemenyi is more conservative than the
    # omnibus Friedman result above, which is the whole reason a report needs
    # both.
    result = stats.nemenyi(known_ranking_runs)

    assert not result.significant("A", "B")
    assert not result.significant("B", "C")
    assert result.significant("A", "C")
    assert result.cliques == (("A", "B"), ("B", "C"))


def test_nemenyi_does_not_separate_methods_within_the_critical_difference():
    # Ranks 1, 1.5, 3 with a critical difference of 1.9: A-B (gap 0.5) and
    # B-C (gap 1.5) are each within it, A-C (gap 2.0) is not — a chain, not
    # one clique of three.
    ranks = pd.Series({"A": 1.0, "B": 1.5, "C": 3.0}).sort_values()

    cliques = stats._cliques(ranks, critical_difference=1.9)

    assert cliques == (("A", "B"), ("B", "C"))


def test_a_clique_of_three_is_reported_once_not_as_three_overlapping_pairs():
    ranks = pd.Series({"A": 1.0, "B": 1.5, "C": 2.0}).sort_values()

    cliques = stats._cliques(ranks, critical_difference=1.0)

    assert cliques == (("A", "B", "C"),)


def test_no_cliques_when_every_method_is_significantly_apart():
    ranks = pd.Series({"A": 1.0, "B": 5.0, "C": 10.0}).sort_values()

    cliques = stats._cliques(ranks, critical_difference=0.5)

    assert cliques == ()


def test_significant_agrees_with_cliques():
    result = stats.NemenyiResult(
        average_ranks=pd.Series({"A": 1.0, "B": 1.5, "C": 3.0}),
        critical_difference=1.9,
        alpha=0.05,
        n_datasets=6,
        n_methods=3,
        cliques=(("A", "B"), ("B", "C")),
    )

    assert not result.significant("A", "B")
    assert not result.significant("B", "C")
    assert result.significant("A", "C")


# ---------------------------------------------------------------------------
# pairwise_signed_rank
# ---------------------------------------------------------------------------


def test_pairwise_signed_rank_is_significant_for_the_known_ranking_s_extremes(
    known_ranking_runs,
):
    comparison = stats.pairwise_signed_rank(known_ranking_runs, "A", "C")

    assert comparison.method_a == "A"
    assert comparison.method_b == "C"
    assert comparison.n_datasets == len(DATASETS)
    assert comparison.p_value < 0.05


def test_pairwise_signed_rank_p_value_is_symmetric(known_ranking_runs):
    a_vs_c = stats.pairwise_signed_rank(known_ranking_runs, "A", "C")
    c_vs_a = stats.pairwise_signed_rank(known_ranking_runs, "C", "A")

    assert a_vs_c.p_value == pytest.approx(c_vs_a.p_value)


def test_pairwise_signed_rank_refuses_an_unknown_method(known_ranking_runs):
    with pytest.raises(ValueError, match="nope"):
        stats.pairwise_signed_rank(known_ranking_runs, "A", "nope")


def test_pairwise_signed_rank_ignores_a_coverage_gap_in_an_unrelated_method(
    known_ranking_runs,
):
    # D is scored on one dataset only — irrelevant to an A-vs-C comparison,
    # and should not block it the way it would block friedman/nemenyi across
    # the whole ranking including D.
    incomplete = pd.concat(
        [known_ranking_runs, _frame([_row(dataset="d2", method="D", absolute_error=0.5)])],
        ignore_index=True,
    )

    comparison = stats.pairwise_signed_rank(incomplete, "A", "C")

    assert comparison.n_datasets == len(DATASETS)
