"""The MAE trend and the per-bag-size ranking, both shaped from fabricated tables (#19).

Every fixture here is built in code, the same reason ``tests/test_stats.py``
and ``tests/test_grid.py`` build theirs that way: no real sweep execution is
needed to prove a pure data-shaping function, and #19's own acceptance
criterion asks for exactly that.
"""

import pandas as pd
import pytest

import bag_size_report


def _row(*, dataset, bag_size, method="DyS", absolute_error):
    return {
        "dataset": dataset,
        "bag_size": bag_size,
        "method": method,
        "absolute_error": absolute_error,
    }


def _frame(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# mae_by_bag_size
# ---------------------------------------------------------------------------


def test_mae_by_bag_size_averages_over_methods_and_repetitions():
    labelled = _frame(
        [
            _row(dataset="d1", bag_size=100, method="A", absolute_error=0.1),
            _row(dataset="d1", bag_size=100, method="B", absolute_error=0.3),
        ]
    )

    mae = bag_size_report.mae_by_bag_size(labelled)

    assert len(mae) == 1
    assert mae["absolute_error"].iloc[0] == pytest.approx(0.2)


def test_mae_by_bag_size_keeps_distinct_datasets_and_bag_sizes_apart():
    labelled = _frame(
        [
            _row(dataset="d1", bag_size=100, absolute_error=0.1),
            _row(dataset="d1", bag_size=500, absolute_error=0.2),
            _row(dataset="d2", bag_size=100, absolute_error=0.3),
        ]
    )

    mae = bag_size_report.mae_by_bag_size(labelled)

    assert len(mae) == 3
    assert set(zip(mae["dataset"], mae["bag_size"])) == {
        ("d1", 100),
        ("d1", 500),
        ("d2", 100),
    }


def test_mae_by_bag_size_is_ordered_by_dataset_then_bag_size():
    labelled = _frame(
        [
            _row(dataset="d2", bag_size=500, absolute_error=0.1),
            _row(dataset="d1", bag_size=500, absolute_error=0.2),
            _row(dataset="d1", bag_size=100, absolute_error=0.3),
        ]
    )

    mae = bag_size_report.mae_by_bag_size(labelled)

    assert mae[["dataset", "bag_size"]].values.tolist() == [
        ["d1", 100],
        ["d1", 500],
        ["d2", 500],
    ]


def test_mae_by_bag_size_does_not_filter_replicated_rows():
    # #18 flags a run whose bag used bootstrap replication; this function
    # takes every row it is handed, replicated or not, rather than dropping
    # any of them by default.
    labelled = _frame(
        [
            _row(dataset="d1", bag_size=5000, method="A", absolute_error=0.1),
            _row(dataset="d1", bag_size=5000, method="B", absolute_error=0.3),
        ]
    )
    labelled["bag_replicated"] = [True, False]

    mae = bag_size_report.mae_by_bag_size(labelled)

    assert mae["absolute_error"].iloc[0] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# rankings_by_bag_size
# ---------------------------------------------------------------------------


def _three_method_rows(dataset, bag_size, errors):
    return [
        _row(dataset=dataset, bag_size=bag_size, method=method, absolute_error=error)
        for method, error in zip(("A", "B", "C"), errors)
    ]


def test_rankings_by_bag_size_does_not_filter_replicated_rows():
    # #18 flags a run whose bag used bootstrap replication; this function
    # takes every row it is handed, replicated or not, rather than dropping
    # any of them by default — the same property asserted for
    # mae_by_bag_size above.
    rows = []
    for dataset in ("d1", "d2", "d3"):
        rows += _three_method_rows(dataset, 100, (0.1, 0.2, 0.3))
    labelled = _frame(rows)
    labelled["bag_replicated"] = [True, False, False] * 3

    rankings = bag_size_report.rankings_by_bag_size(labelled)

    assert len(rankings) == 1
    assert rankings[0].friedman.n_datasets == 3


def test_rankings_by_bag_size_returns_one_ranking_per_bag_size_ordered_ascending():
    rows = []
    for dataset in ("d1", "d2", "d3"):
        rows += _three_method_rows(dataset, 500, (0.1, 0.2, 0.3))
        rows += _three_method_rows(dataset, 100, (0.1, 0.2, 0.3))
    labelled = _frame(rows)

    rankings = bag_size_report.rankings_by_bag_size(labelled)

    assert [ranking.bag_size for ranking in rankings] == [100, 500]


def test_rankings_by_bag_size_computes_each_bag_size_s_ranking_independently():
    # A best, C worst at bag_size 100; the ranking flips entirely by 5000 —
    # a ranking pooled across bag sizes could not show this, since A and C
    # would average out to a tie.
    rows = []
    for dataset in ("d1", "d2", "d3", "d4"):
        rows += _three_method_rows(dataset, 100, (0.05, 0.15, 0.25))
        rows += _three_method_rows(dataset, 5000, (0.25, 0.15, 0.05))
    labelled = _frame(rows)

    rankings = bag_size_report.rankings_by_bag_size(labelled)
    by_bag_size = {ranking.bag_size: ranking for ranking in rankings}

    assert list(by_bag_size[100].nemenyi.average_ranks.index) == ["A", "B", "C"]
    assert list(by_bag_size[5000].nemenyi.average_ranks.index) == ["C", "B", "A"]
    assert by_bag_size[100].friedman.n_datasets == 4
    assert by_bag_size[5000].friedman.n_datasets == 4


def test_rankings_by_bag_size_reaches_stats_with_only_that_bag_size_s_rows(monkeypatch):
    labelled = _frame(
        [
            _row(dataset="d1", bag_size=100, method="A", absolute_error=0.1),
            _row(dataset="d1", bag_size=500, method="A", absolute_error=0.2),
        ]
    )

    calls = []

    def spy_friedman(runs_arg):
        calls.append(("friedman", frozenset(runs_arg["bag_size"])))
        return f"friedman-{frozenset(runs_arg['bag_size'])}"

    def spy_nemenyi(runs_arg, alpha=0.05):
        calls.append(("nemenyi", frozenset(runs_arg["bag_size"]), alpha))
        return f"nemenyi-{frozenset(runs_arg['bag_size'])}"

    monkeypatch.setattr(bag_size_report.stats, "friedman", spy_friedman)
    monkeypatch.setattr(bag_size_report.stats, "nemenyi", spy_nemenyi)

    rankings = bag_size_report.rankings_by_bag_size(labelled, alpha=0.1)
    by_bag_size = {ranking.bag_size: ranking for ranking in rankings}

    assert by_bag_size[100].friedman == "friedman-frozenset({100})"
    assert by_bag_size[100].nemenyi == "nemenyi-frozenset({100})"
    assert ("friedman", frozenset({100})) in calls
    assert ("nemenyi", frozenset({100}), 0.1) in calls
    assert ("friedman", frozenset({500})) in calls
