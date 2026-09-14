"""The real-data report reaches ``runs`` and ``stats`` with the data it is given, not a copy of its own (#12).

Mirrors ``tests/test_renderers.py``'s split: the headline pair's name is
checked against ``runs``' own vocabulary rather than trusted as a literal,
``report`` is driven with spies standing in for ``stats`` so this suite never
depends on any particular numbers coming out of a statistical test, and the
diagram itself is proven against a hand-built :class:`stats.NemenyiResult`
rather than a full sweep.
"""

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

import runs
import stats
import statistical_report as report


# ---------------------------------------------------------------------------
# The headline pair is a live name, not a typo waiting to filter to nothing
# ---------------------------------------------------------------------------


def test_headline_pair_is_spelled_the_way_runs_method_label_would_spell_it():
    base_quantifier, wrapped = report.HEADLINE_PAIR

    assert runs.method_label(base_quantifier, runs.NO_METHOD_SIMULATOR) == base_quantifier
    assert runs.method_label(base_quantifier, runs.MVN) == wrapped


def test_headline_pair_methods_exist_in_the_real_data_vocabulary(results_root, real_data_runs):
    runs.save(real_data_runs, runs.REAL_DATA, root=results_root)
    labelled = runs.load_labelled(runs.REAL_DATA, root=results_root)

    assert set(report.HEADLINE_PAIR) <= set(labelled["method"])


# ---------------------------------------------------------------------------
# report(): reaches stats.friedman/nemenyi/pairwise_signed_rank with the
# frame it is given, and only that frame
# ---------------------------------------------------------------------------


def test_report_reaches_stats_with_the_runs_it_is_given_and_the_named_headline_pair(
    monkeypatch, tmp_path, capsys
):
    labelled_runs = pd.DataFrame(
        {"dataset": ["d1"], "method": ["DyS"], "absolute_error": [0.1]}
    )

    friedman_result = stats.FriedmanResult(
        statistic=1.0,
        p_value=0.5,
        n_datasets=8,
        n_methods=51,
        average_ranks=pd.Series({"DyS": 1.0, "QuaDapt-MVN(DyS)": 2.0}),
    )
    nemenyi_result = stats.NemenyiResult(
        average_ranks=pd.Series({"DyS": 1.0, "QuaDapt-MVN(DyS)": 2.0}),
        critical_difference=0.5,
        alpha=0.05,
        n_datasets=8,
        n_methods=2,
        cliques=(),
    )
    headline_result = stats.PairwiseComparison(
        method_a="DyS",
        method_b="QuaDapt-MVN(DyS)",
        statistic=3.0,
        p_value=0.01,
        n_datasets=8,
    )

    calls = {}

    def spy_friedman(runs_arg):
        calls["friedman"] = runs_arg
        return friedman_result

    def spy_nemenyi(runs_arg):
        calls["nemenyi"] = runs_arg
        return nemenyi_result

    def spy_pairwise(runs_arg, method_a, method_b):
        calls["pairwise"] = (runs_arg, method_a, method_b)
        return headline_result

    monkeypatch.setattr(stats, "friedman", spy_friedman)
    monkeypatch.setattr(stats, "nemenyi", spy_nemenyi)
    monkeypatch.setattr(stats, "pairwise_signed_rank", spy_pairwise)
    monkeypatch.setattr(report, "OUTPUT_CD_DIAGRAM_PNG", str(tmp_path / "cd.png"))
    monkeypatch.setattr(report, "OUTPUT_CD_DIAGRAM_PDF", str(tmp_path / "cd.pdf"))

    returned = report.report(labelled_runs)

    assert calls["friedman"] is labelled_runs
    assert calls["nemenyi"] is labelled_runs
    assert calls["pairwise"] == (labelled_runs, *report.HEADLINE_PAIR)
    assert returned == (friedman_result, nemenyi_result, headline_result)

    printed = capsys.readouterr().out
    assert "Friedman" in printed
    assert "Nemenyi" in printed
    assert "DyS vs QuaDapt-MVN(DyS)" in printed

    assert (tmp_path / "cd.png").exists()
    assert (tmp_path / "cd.pdf").exists()


# ---------------------------------------------------------------------------
# critical_difference_diagram(): draws exactly what a NemenyiResult decided
# ---------------------------------------------------------------------------


def test_critical_difference_diagram_draws_a_stem_and_a_label_per_method_and_a_bar_per_clique():
    nemenyi_result = stats.NemenyiResult(
        average_ranks=pd.Series({"A": 1.0, "B": 2.0, "C": 3.0}),
        critical_difference=0.5,
        alpha=0.05,
        n_datasets=6,
        n_methods=3,
        cliques=(("A", "B"),),
    )

    ax = report.critical_difference_diagram(nemenyi_result)

    # One stem plus one marker point per method, plus one bar per clique.
    assert len(ax.get_lines()) == 2 * nemenyi_result.n_methods + len(nemenyi_result.cliques)
    assert len(ax.texts) == nemenyi_result.n_methods
    assert ax.get_xlim() == pytest.approx((0.5, 3.5))


def test_critical_difference_diagram_draws_no_bar_when_every_method_is_separated():
    nemenyi_result = stats.NemenyiResult(
        average_ranks=pd.Series({"A": 1.0, "B": 5.0, "C": 10.0}),
        critical_difference=0.5,
        alpha=0.05,
        n_datasets=6,
        n_methods=3,
        cliques=(),
    )

    ax = report.critical_difference_diagram(nemenyi_result)

    assert len(ax.get_lines()) == 2 * nemenyi_result.n_methods
