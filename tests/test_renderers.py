"""Every name a renderer filters on must exist in the runs it filters.

This is the regression for the defect that motivated the runs module. The
figure export selected base quantifiers named ``"X"`` and ``"MS"``; the sweep
records them as ``"TX"`` and ``"TMS"``. Neither filter ever matched a row, so
two of the eight methods in the published grid were simply not drawn, and
nothing said so — the plot rendered cleanly with six lines instead of eight.

A filter is only ever wrong relative to the data, so these tests assert
against runs rather than against a second list of names. The fixture is built
from ``runs``' own vocabulary (see ``conftest.py``), so a rename reaches both
sides and a filter left behind fails here.
"""

import matplotlib

matplotlib.use("Agg")

import export_grid_matplotlib as export
import grid
import runs


def _present(results_root, synthetic_runs, column):
    runs.save(synthetic_runs, runs.SYNTHETIC, root=results_root)
    return set(runs.load(runs.SYNTHETIC, root=results_root)[column])


def test_every_base_quantifier_the_export_filters_on_exists_in_the_data(
    results_root, synthetic_runs
):
    present = _present(results_root, synthetic_runs, "base_quantifier")

    assert set(export.PUBLISHED_BASE_QUANTIFIERS) <= present


def test_every_method_simulator_the_export_filters_on_exists_in_the_data(
    results_root, synthetic_runs
):
    present = _present(results_root, synthetic_runs, "method_simulator")

    assert set(export.PUBLISHED_METHOD_SIMULATORS) <= present


def test_the_reference_simulator_the_grid_selects_exists_in_the_data(
    results_root, synthetic_runs
):
    present = _present(results_root, synthetic_runs, "reference_simulator")

    assert export.REFERENCE_SIMULATOR in present


def test_every_bag_simulator_the_grid_rows_select_exists_in_the_data(
    results_root, synthetic_runs
):
    present = _present(results_root, synthetic_runs, "bag_simulator")

    assert set(export.BAG_SIMULATORS) <= present


def test_the_export_filters_select_something(results_root, synthetic_runs):
    # The filters above could each be individually live and still intersect to
    # nothing. What the figure needs is that the conjunction has rows.
    runs.save(synthetic_runs, runs.SYNTHETIC, root=results_root)
    published = runs.load(runs.SYNTHETIC, root=results_root)

    selected = published[
        published["base_quantifier"].isin(export.PUBLISHED_BASE_QUANTIFIERS)
        & published["method_simulator"].isin(export.PUBLISHED_METHOD_SIMULATORS)
    ]

    assert not selected.empty


def test_the_baseline_quantifier_the_dashboard_draws_exists_in_the_data(
    results_root, synthetic_runs
):
    present = _present(results_root, synthetic_runs, "base_quantifier")

    assert runs.BASELINE_QUANTIFIER in present


def test_every_published_quantifier_has_its_own_marker():
    # A missing marker is the quiet version of the same defect: the method is
    # drawn, but under another method's symbol.
    assert set(export.PUBLISHED_BASE_QUANTIFIERS) <= set(export.QUANTIFIER_MARKERS)
    assert len(set(export.QUANTIFIER_MARKERS.values())) == len(export.QUANTIFIER_MARKERS)


def test_every_published_method_simulator_is_drawable():
    assert set(export.PUBLISHED_METHOD_SIMULATORS) <= set(
        export.METHOD_SIMULATOR_COLOURS
    )
    assert set(export.PUBLISHED_METHOD_SIMULATORS) <= set(
        export.METHOD_SIMULATOR_LINESTYLES
    )


# ---------------------------------------------------------------------------
# #13: the export reaches the one grid module rather than keeping its own
# copy of which cell shows which runs.
# ---------------------------------------------------------------------------


def test_the_grid_shape_export_draws_is_grid_s_own_not_a_copy():
    # Identity, not just equality: this fails if a future edit goes back to
    # spelling out ``BAG_SIMULATORS``/``REFERENCE_MERGING_FACTORS`` in
    # ``export_grid_matplotlib.py`` instead of importing them, even if the
    # spelled-out values still happen to match today.
    assert export.BAG_SIMULATORS is grid.BAG_SIMULATORS
    assert export.REFERENCE_MERGING_FACTORS is grid.REFERENCE_MERGING_FACTORS


def test_plot_grid_reaches_grid_panels_with_its_own_constants_and_the_runs_it_is_given(
    monkeypatch, tmp_path, synthetic_runs
):
    # A test calling ``grid.panels`` directly (``tests/test_grid.py``) proves
    # the function itself is correct; it says nothing about whether this
    # renderer actually calls it, or calls it with the right arguments —
    # which is the failure mode #13 exists to end. This drives
    # ``export.plot_grid`` for real and inspects the call it makes.
    monkeypatch.setattr(export, "OUTPUT_PNG", str(tmp_path / "grid.png"))
    monkeypatch.setattr(export, "OUTPUT_PDF", str(tmp_path / "grid.pdf"))

    real_panels = grid.panels
    calls = []

    def spying_panels(*args, **kwargs):
        calls.append((args, kwargs))
        return real_panels(*args, **kwargs)

    monkeypatch.setattr(grid, "panels", spying_panels)

    # The same selection ``export.load_published_runs`` makes: the baseline
    # quantifier and the ``all``/``none`` arms are not among
    # ``PUBLISHED_BASE_QUANTIFIERS``/``PUBLISHED_METHOD_SIMULATORS``, so a
    # published run is never the baseline, and ``plot_grid`` never has to draw
    # one.
    published = synthetic_runs[
        synthetic_runs["base_quantifier"].isin(export.PUBLISHED_BASE_QUANTIFIERS)
        & synthetic_runs["method_simulator"].isin(export.PUBLISHED_METHOD_SIMULATORS)
    ].copy()
    published["method"] = runs.method_labels(published)
    published["absolute_error"] = runs.absolute_error(published)

    export.plot_grid(published)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (published,)
    assert kwargs == {
        "reference_simulator": export.REFERENCE_SIMULATOR,
        "bag_simulators": export.BAG_SIMULATORS,
        "reference_merging_factors": export.REFERENCE_MERGING_FACTORS,
    }
