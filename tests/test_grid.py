"""Which runs land in which panel of the grid: the defect #13 exists to end.

Which cell shows which runs used to be decided twice — once in the Streamlit
script body, once in ``export_grid_matplotlib.plot_grid`` — free to drift the
same way the runs module's two readers drifted (#5). ``grid.panels`` is now
the one place that decides, and every test here asserts against its return
value directly: a tuple of :class:`grid.Panel`, built from a small labelled
table constructed in code rather than loaded from disk. No renderer runs.
"""

import numpy as np
import pandas as pd
import pytest

import grid
import runs


def _row(
    *,
    reference_merging_factor=0.5,
    reference_simulator=runs.UNIFORM,
    bag_simulator=runs.UNIFORM,
    bag_merging_factor=0.5,
    base_quantifier="DyS",
    method_simulator=runs.NO_METHOD_SIMULATOR,
    method=None,
    absolute_error=0.1,
):
    """One labelled run, every field defaulted so a test states only what it varies."""
    return {
        "reference_merging_factor": reference_merging_factor,
        "reference_simulator": reference_simulator,
        "bag_simulator": bag_simulator,
        "bag_merging_factor": bag_merging_factor,
        "base_quantifier": base_quantifier,
        "method_simulator": method_simulator,
        "method": method if method is not None else base_quantifier,
        "absolute_error": absolute_error,
    }


def _frame(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Masking: a panel holds exactly the runs of its own row and column
# ---------------------------------------------------------------------------


def test_a_panel_only_holds_runs_matching_its_bag_simulator_and_reference_factor():
    labelled = _frame(
        [
            _row(bag_simulator=runs.UNIFORM, reference_merging_factor=0.25, method="in"),
            _row(bag_simulator=runs.MVN, reference_merging_factor=0.25, method="wrong_row"),
            _row(bag_simulator=runs.UNIFORM, reference_merging_factor=0.75, method="wrong_col"),
        ]
    )

    (panel,) = grid.panels(
        labelled,
        reference_simulator=runs.UNIFORM,
        bag_simulators=(runs.UNIFORM,),
        reference_merging_factors=(0.25,),
    )

    assert panel.methods["method"].tolist() == ["in"]


def test_a_panel_only_holds_runs_matching_its_reference_simulator():
    labelled = _frame(
        [
            _row(reference_simulator=runs.UNIFORM, method="in"),
            _row(reference_simulator=runs.MVN, method="wrong_reference_simulator"),
        ]
    )

    (panel,) = grid.panels(
        labelled,
        reference_simulator=runs.UNIFORM,
        bag_simulators=(runs.UNIFORM,),
        reference_merging_factors=(0.5,),
    )

    assert panel.methods["method"].tolist() == ["in"]


def test_reference_merging_factor_matching_tolerates_floating_point_noise():
    # 0.1 + 0.2 != 0.3 in floating point; a strict equality mask would drop a
    # cell whose merging factor was only ever nominally 0.3.
    noisy = 0.1 + 0.2
    labelled = _frame([_row(reference_merging_factor=noisy, method="in")])

    (panel,) = grid.panels(
        labelled,
        reference_simulator=runs.UNIFORM,
        bag_simulators=(runs.UNIFORM,),
        reference_merging_factors=(0.3,),
    )

    assert panel.methods["method"].tolist() == ["in"]


# ---------------------------------------------------------------------------
# Baseline split-out and labelling
# ---------------------------------------------------------------------------


def test_the_baseline_quantifier_is_split_into_its_own_frame_and_relabelled():
    labelled = _frame(
        [
            _row(
                base_quantifier=runs.BASELINE_QUANTIFIER,
                method_simulator=runs.NO_METHOD_SIMULATOR,
                method=runs.BASELINE_QUANTIFIER,
            ),
            _row(base_quantifier="DyS", method_simulator=runs.NO_METHOD_SIMULATOR, method="DyS"),
        ]
    )

    (panel,) = grid.panels(
        labelled,
        reference_simulator=runs.UNIFORM,
        bag_simulators=(runs.UNIFORM,),
        reference_merging_factors=(0.5,),
    )

    assert panel.methods["method"].tolist() == ["DyS"]
    assert panel.baseline["method"].tolist() == [grid.BASELINE_LABEL]
    assert grid.BASELINE_LABEL == f"{runs.BASELINE_QUANTIFIER} (baseline)"


def test_a_panel_with_no_baseline_runs_has_an_empty_baseline_frame():
    labelled = _frame([_row(base_quantifier="DyS")])

    (panel,) = grid.panels(
        labelled,
        reference_simulator=runs.UNIFORM,
        bag_simulators=(runs.UNIFORM,),
        reference_merging_factors=(0.5,),
    )

    assert panel.baseline.empty


def test_a_method_simulator_that_happens_to_use_the_baseline_quantifier_is_not_split_out():
    # The baseline is identified by (base quantifier, no method simulator)
    # together, not by base quantifier alone — a candidate-search run on CC
    # (were one ever produced) is not the baseline.
    labelled = _frame(
        [_row(base_quantifier=runs.BASELINE_QUANTIFIER, method_simulator=runs.MVN, method="m")]
    )

    (panel,) = grid.panels(
        labelled,
        reference_simulator=runs.UNIFORM,
        bag_simulators=(runs.UNIFORM,),
        reference_merging_factors=(0.5,),
    )

    assert panel.methods["method"].tolist() == ["m"]
    assert panel.baseline.empty


# ---------------------------------------------------------------------------
# Aggregation: repetitions collapse into one mean point per method and x
# ---------------------------------------------------------------------------


def test_repetitions_are_averaged_into_one_point_per_method_and_x_value():
    labelled = _frame(
        [
            _row(bag_merging_factor=0.4, absolute_error=0.10),
            _row(bag_merging_factor=0.4, absolute_error=0.30),
        ]
    )

    (panel,) = grid.panels(
        labelled,
        reference_simulator=runs.UNIFORM,
        bag_simulators=(runs.UNIFORM,),
        reference_merging_factors=(0.5,),
    )

    assert len(panel.methods) == 1
    assert panel.methods["absolute_error"].iloc[0] == pytest.approx(0.20)


def test_distinct_x_values_are_not_collapsed_together():
    labelled = _frame(
        [
            _row(bag_merging_factor=0.2, absolute_error=0.10),
            _row(bag_merging_factor=0.4, absolute_error=0.30),
        ]
    )

    (panel,) = grid.panels(
        labelled,
        reference_simulator=runs.UNIFORM,
        bag_simulators=(runs.UNIFORM,),
        reference_merging_factors=(0.5,),
    )

    assert sorted(panel.methods["bag_merging_factor"].tolist()) == [0.2, 0.4]


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_methods_are_ordered_by_name_then_by_x_within_a_panel():
    labelled = _frame(
        [
            _row(base_quantifier="TMS", method="TMS", bag_merging_factor=0.6),
            _row(base_quantifier="DyS", method="DyS", bag_merging_factor=0.8),
            _row(base_quantifier="DyS", method="DyS", bag_merging_factor=0.2),
        ]
    )

    (panel,) = grid.panels(
        labelled,
        reference_simulator=runs.UNIFORM,
        bag_simulators=(runs.UNIFORM,),
        reference_merging_factors=(0.5,),
    )

    assert panel.methods[["method", "bag_merging_factor"]].values.tolist() == [
        ["DyS", 0.2],
        ["DyS", 0.8],
        ["TMS", 0.6],
    ]


def test_panels_are_returned_in_the_order_the_caller_names_rows_and_columns():
    labelled = _frame(
        [
            _row(bag_simulator=runs.UNIFORM, reference_merging_factor=0.25),
            _row(bag_simulator=runs.UNIFORM, reference_merging_factor=0.75),
            _row(bag_simulator=runs.MVN, reference_merging_factor=0.25),
            _row(bag_simulator=runs.MVN, reference_merging_factor=0.75),
        ]
    )

    built = grid.panels(
        labelled,
        reference_simulator=runs.UNIFORM,
        bag_simulators=(runs.MVN, runs.UNIFORM),
        reference_merging_factors=(0.75, 0.25),
    )

    assert [(p.bag_simulator, p.reference_merging_factor) for p in built] == [
        (runs.MVN, 0.75),
        (runs.MVN, 0.25),
        (runs.UNIFORM, 0.75),
        (runs.UNIFORM, 0.25),
    ]


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------


def test_downsampling_caps_points_per_method_without_touching_other_methods():
    dense = [
        _row(base_quantifier="DyS", method="DyS", bag_merging_factor=x, absolute_error=x)
        for x in np.linspace(0.0, 1.0, 10)
    ]
    sparse = [_row(base_quantifier="TMS", method="TMS", bag_merging_factor=0.5)]

    (panel,) = grid.panels(
        _frame(dense + sparse),
        reference_simulator=runs.UNIFORM,
        bag_simulators=(runs.UNIFORM,),
        reference_merging_factors=(0.5,),
        max_points_per_method=3,
    )

    assert (panel.methods["method"] == "DyS").sum() == 3
    assert (panel.methods["method"] == "TMS").sum() == 1


def test_downsampling_keeps_the_first_and_last_x_value():
    dense = [
        _row(base_quantifier="DyS", method="DyS", bag_merging_factor=x, absolute_error=x)
        for x in np.linspace(0.0, 1.0, 10)
    ]

    (panel,) = grid.panels(
        _frame(dense),
        reference_simulator=runs.UNIFORM,
        bag_simulators=(runs.UNIFORM,),
        reference_merging_factors=(0.5,),
        max_points_per_method=3,
    )

    x_values = panel.methods["bag_merging_factor"].tolist()
    assert x_values[0] == pytest.approx(0.0)
    assert x_values[-1] == pytest.approx(1.0)


def test_a_panel_under_the_cap_is_left_untouched():
    labelled = _frame([_row(bag_merging_factor=0.5)])

    (panel,) = grid.panels(
        labelled,
        reference_simulator=runs.UNIFORM,
        bag_simulators=(runs.UNIFORM,),
        reference_merging_factors=(0.5,),
        max_points_per_method=1000,
    )

    assert len(panel.methods) == 1


# ---------------------------------------------------------------------------
# Determinism: no hidden state a second call could land differently on
# ---------------------------------------------------------------------------


def test_panels_built_twice_from_the_same_runs_and_arguments_are_identical():
    # ``grid.panels`` carries no state of its own, so this is a narrow claim:
    # the same call twice cannot itself diverge. It is not, by itself, the
    # claim that the dashboard and the figure export agree — that they
    # reach this function at all, and with their own real constants rather
    # than a hand-typed copy of them, is asserted separately against each
    # renderer's actual call site (`tests/test_renderers.py`, #13), since
    # `dashboard.py` is a Streamlit script neither this test file nor any
    # other can import to check directly.
    labelled = _frame(
        [
            _row(bag_simulator=bag_simulator, reference_merging_factor=factor, method="DyS")
            for bag_simulator in grid.BAG_SIMULATORS
            for factor in grid.REFERENCE_MERGING_FACTORS
        ]
    )

    first = grid.panels(
        labelled,
        reference_simulator=runs.UNIFORM,
        bag_simulators=grid.BAG_SIMULATORS,
        reference_merging_factors=grid.REFERENCE_MERGING_FACTORS,
    )
    second = grid.panels(
        labelled,
        reference_simulator=runs.UNIFORM,
        bag_simulators=grid.BAG_SIMULATORS,
        reference_merging_factors=grid.REFERENCE_MERGING_FACTORS,
    )

    assert len(first) == len(second) == len(grid.BAG_SIMULATORS) * len(
        grid.REFERENCE_MERGING_FACTORS
    )
    for panel_a, panel_b in zip(first, second):
        assert panel_a.bag_simulator == panel_b.bag_simulator
        assert panel_a.reference_merging_factor == panel_b.reference_merging_factor
        pd.testing.assert_frame_equal(panel_a.methods, panel_b.methods)
        pd.testing.assert_frame_equal(panel_a.baseline, panel_b.baseline)
