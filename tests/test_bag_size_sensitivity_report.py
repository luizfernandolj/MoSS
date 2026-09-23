"""The bag-size sensitivity script reaches ``bag_size_report`` and ``statistical_report`` with the data it is given, then draws it (#19).

Mirrors ``tests/test_statistical_report.py``'s own split: the shaping
functions are proven separately (``tests/test_bag_size_report.py``), so this
suite only has to prove the script draws what they hand back and writes the
files the ticket asks for — a PNG and a PDF per figure.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

import bag_size_sensitivity_report as report
import stats


# ---------------------------------------------------------------------------
# plot_mae_by_bag_size: one subplot per dataset, drawn from an already-shaped table
# ---------------------------------------------------------------------------


def test_plot_mae_by_bag_size_draws_one_subplot_per_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "OUTPUT_MAE_PNG", str(tmp_path / "mae.png"))
    monkeypatch.setattr(report, "OUTPUT_MAE_PDF", str(tmp_path / "mae.pdf"))

    mae_table = pd.DataFrame(
        [
            {"dataset": "haberman", "bag_size": 100, "absolute_error": 0.2},
            {"dataset": "haberman", "bag_size": 500, "absolute_error": 0.1},
            {"dataset": "wine", "bag_size": 100, "absolute_error": 0.3},
            {"dataset": "wine", "bag_size": 500, "absolute_error": 0.25},
        ]
    )

    fig = report.plot_mae_by_bag_size(mae_table)

    assert len(fig.axes) == 2
    assert [ax.get_title() for ax in fig.axes] == ["haberman", "wine"]
    assert (tmp_path / "mae.png").exists()
    assert (tmp_path / "mae.pdf").exists()


def test_plot_mae_by_bag_size_draws_each_dataset_s_own_points_sorted_by_bag_size(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(report, "OUTPUT_MAE_PNG", str(tmp_path / "mae.png"))
    monkeypatch.setattr(report, "OUTPUT_MAE_PDF", str(tmp_path / "mae.pdf"))

    mae_table = pd.DataFrame(
        [
            {"dataset": "haberman", "bag_size": 500, "absolute_error": 0.1},
            {"dataset": "haberman", "bag_size": 100, "absolute_error": 0.2},
        ]
    )

    fig = report.plot_mae_by_bag_size(mae_table)

    (line,) = fig.axes[0].get_lines()
    assert list(line.get_xdata()) == [100, 500]
    assert list(line.get_ydata()) == pytest.approx([0.2, 0.1])


def test_plot_mae_by_bag_size_defaults_to_every_dataset_the_table_names_sorted(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(report, "OUTPUT_MAE_PNG", str(tmp_path / "mae.png"))
    monkeypatch.setattr(report, "OUTPUT_MAE_PDF", str(tmp_path / "mae.pdf"))

    mae_table = pd.DataFrame(
        [
            {"dataset": "zzz", "bag_size": 100, "absolute_error": 0.1},
            {"dataset": "aaa", "bag_size": 100, "absolute_error": 0.1},
        ]
    )

    fig = report.plot_mae_by_bag_size(mae_table)

    assert [ax.get_title() for ax in fig.axes] == ["aaa", "zzz"]


# ---------------------------------------------------------------------------
# cd_diagram_paths / plot_rankings_by_bag_size
# ---------------------------------------------------------------------------


def test_cd_diagram_paths_names_the_bag_size_in_both_files():
    png_path, pdf_path = report.cd_diagram_paths(500)

    assert "500" in png_path
    assert "500" in pdf_path
    assert png_path.endswith(".png")
    assert pdf_path.endswith(".pdf")


def _nemenyi_result(methods_and_ranks):
    ranks = pd.Series(dict(methods_and_ranks)).sort_values()
    return stats.NemenyiResult(
        average_ranks=ranks,
        critical_difference=0.5,
        alpha=0.05,
        n_datasets=6,
        n_methods=len(ranks),
        cliques=(),
    )


def test_plot_rankings_by_bag_size_prefixes_the_diagram_title_with_its_bag_size(
    monkeypatch, tmp_path
):
    # A diagram's own title (statistical_report.critical_difference_diagram)
    # names only the critical difference and dataset count — nothing that
    # tells two bag sizes' diagrams apart once saved to separate files.
    monkeypatch.setattr(report, "OUTPUT_CD_DIAGRAM_PNG_TEMPLATE", str(tmp_path / "cd_{bag_size}.png"))
    monkeypatch.setattr(report, "OUTPUT_CD_DIAGRAM_PDF_TEMPLATE", str(tmp_path / "cd_{bag_size}.pdf"))

    captured_titles = []
    real_close = report.plt.close

    def spy_close(fig):
        captured_titles.append(fig.axes[0].get_title())
        real_close(fig)

    monkeypatch.setattr(report.plt, "close", spy_close)

    rankings = [
        report.bag_size_report.BagSizeRanking(
            bag_size=500, friedman=None, nemenyi=_nemenyi_result({"A": 1.0, "B": 2.0})
        ),
    ]

    report.plot_rankings_by_bag_size(rankings)

    assert captured_titles[0].startswith("Bag size = 500")


def test_plot_rankings_by_bag_size_writes_one_png_and_pdf_per_bag_size(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "OUTPUT_CD_DIAGRAM_PNG_TEMPLATE", str(tmp_path / "cd_{bag_size}.png"))
    monkeypatch.setattr(report, "OUTPUT_CD_DIAGRAM_PDF_TEMPLATE", str(tmp_path / "cd_{bag_size}.pdf"))

    rankings = [
        report.bag_size_report.BagSizeRanking(
            bag_size=100,
            friedman=None,
            nemenyi=_nemenyi_result({"A": 1.0, "B": 2.0}),
        ),
        report.bag_size_report.BagSizeRanking(
            bag_size=500,
            friedman=None,
            nemenyi=_nemenyi_result({"A": 2.0, "B": 1.0}),
        ),
    ]

    written = report.plot_rankings_by_bag_size(rankings)

    assert written == [
        (str(tmp_path / "cd_100.png"), str(tmp_path / "cd_100.pdf")),
        (str(tmp_path / "cd_500.png"), str(tmp_path / "cd_500.pdf")),
    ]
    for png_path, pdf_path in written:
        assert Path(png_path).exists()
        assert Path(pdf_path).exists()


# ---------------------------------------------------------------------------
# report(): wires bag_size_report's shapes into both plotting calls
# ---------------------------------------------------------------------------


def test_report_prints_which_datasets_a_bag_size_s_ranking_dropped(monkeypatch, capsys):
    # rankings_by_bag_size (#20) drops a dataset with nothing to score at a
    # given bag size rather than failing the ranking outright; a reader
    # running this script needs to see that, not have it happen silently.
    monkeypatch.setattr(report, "plot_mae_by_bag_size", lambda *a, **k: None)
    monkeypatch.setattr(report, "plot_rankings_by_bag_size", lambda rankings: [])
    monkeypatch.setattr(
        report.bag_size_report,
        "rankings_by_bag_size",
        lambda labelled_runs: (
            report.bag_size_report.BagSizeRanking(
                bag_size=5000, friedman=None, nemenyi=None, dropped_datasets=("haberman",)
            ),
            report.bag_size_report.BagSizeRanking(
                bag_size=100, friedman=None, nemenyi=None, dropped_datasets=()
            ),
        ),
    )

    report.report(pd.DataFrame({"dataset": ["d1"], "bag_size": [100], "absolute_error": [0.1]}))

    printed = capsys.readouterr().out
    assert "bag_size=5000" in printed
    assert "haberman" in printed
    assert "bag_size=100" not in printed


def test_report_shapes_with_bag_size_report_and_draws_both_figures(monkeypatch, tmp_path):
    labelled_runs = pd.DataFrame(
        [
            {"dataset": dataset, "bag_size": bag_size, "method": method, "absolute_error": error}
            for bag_size, error_offset in ((100, 0.2), (500, 0.1))
            for dataset in ("d1", "d2", "d3")
            for method, error in (
                ("A", error_offset),
                ("B", error_offset + 0.1),
                ("C", error_offset + 0.2),
            )
        ]
    )

    calls = {}

    def spy_mae_plot(mae_table, datasets=None):
        calls["mae_table"] = mae_table
        return None

    def spy_rankings_plot(rankings):
        calls["rankings"] = rankings
        return []

    monkeypatch.setattr(report, "plot_mae_by_bag_size", spy_mae_plot)
    monkeypatch.setattr(report, "plot_rankings_by_bag_size", spy_rankings_plot)

    mae_table, rankings = report.report(labelled_runs)

    pd.testing.assert_frame_equal(calls["mae_table"], mae_table)
    assert calls["rankings"] == rankings
    assert set(mae_table["bag_size"]) == {100, 500}
