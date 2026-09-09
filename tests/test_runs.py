"""The runs module: one owner for how runs are stored, loaded and named.

The defect these tests stand under is drift. Two readers each carried their
own loading and labelling code, and they disagreed: the figure export filtered
on base quantifiers named ``"X"`` and ``"MS"`` while the sweep records ``"TX"``
and ``"TMS"``, so two methods vanished from the published grid without anyone
noticing. Everything below exists so that the vocabulary has exactly one
definition and a caller cannot spell it wrong in silence.
"""

import pandas as pd
import pytest

import runs
from variables import (
    DATA_SIMULATORS as SWEEP_DATA_SIMULATORS,
    METHOD_SIMULATORS as SWEEP_METHOD_SIMULATORS,
    QUANTIFIERS,
)


# ---------------------------------------------------------------------------
# Schema: two tables sharing an estimator block (ADR-0003)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", runs.KINDS)
def test_both_tables_open_with_the_same_estimator_block(kind):
    columns = runs.columns_for(kind)

    assert columns[: len(runs.ESTIMATOR_COLUMNS)] == runs.ESTIMATOR_COLUMNS


@pytest.mark.parametrize("kind", runs.KINDS)
def test_the_method_simulator_is_present_in_both_tables(kind):
    # It describes the estimator, not the data, so "which simulator does the
    # meta-quantifier use internally" stays comparable on real datasets too.
    assert "method_simulator" in runs.columns_for(kind)


def test_only_the_synthetic_table_carries_the_data_vocabulary():
    # Merging factors and data simulators say how synthetic scores were made
    # and mean nothing for a real dataset — the reason there are two tables.
    data_columns = {
        "reference_simulator",
        "reference_merging_factor",
        "bag_simulator",
        "bag_merging_factor",
    }

    assert data_columns <= set(runs.columns_for(runs.SYNTHETIC))
    assert data_columns.isdisjoint(runs.columns_for(runs.REAL_DATA))


# ---------------------------------------------------------------------------
# Storage: Parquet, and its layout invisible to callers
# ---------------------------------------------------------------------------


def test_synthetic_runs_survive_a_round_trip(results_root, synthetic_runs):
    runs.save(synthetic_runs, runs.SYNTHETIC, root=results_root)

    loaded = runs.load(runs.SYNTHETIC, root=results_root)

    pd.testing.assert_frame_equal(loaded, synthetic_runs)


def test_real_data_runs_survive_a_round_trip(results_root, real_data_runs):
    runs.save(real_data_runs, runs.REAL_DATA, root=results_root)

    loaded = runs.load(runs.REAL_DATA, root=results_root)

    pd.testing.assert_frame_equal(loaded, real_data_runs)


def test_the_two_kinds_do_not_overwrite_each_other(
    results_root, synthetic_runs, real_data_runs
):
    runs.save(synthetic_runs, runs.SYNTHETIC, root=results_root)
    runs.save(real_data_runs, runs.REAL_DATA, root=results_root)

    assert len(runs.load(runs.SYNTHETIC, root=results_root)) == len(synthetic_runs)
    assert len(runs.load(runs.REAL_DATA, root=results_root)) == len(real_data_runs)


def test_everything_written_is_parquet(results_root, synthetic_runs, real_data_runs):
    # The previous single CSV reached 291 MB and had to be split three ways to
    # clear a hosting limit; the split then leaked into every reader.
    runs.save(synthetic_runs, runs.SYNTHETIC, root=results_root)
    runs.save(real_data_runs, runs.REAL_DATA, root=results_root)

    written = [path for path in results_root.rglob("*") if path.is_file()]

    assert written
    assert {path.suffix for path in written} == {".parquet"}


def test_a_caller_never_names_a_file(results_root, synthetic_runs):
    # A kind is the whole of the public vocabulary for storage. If this test
    # ever needs a path to pass, the layout has leaked back out.
    runs.save(synthetic_runs, runs.SYNTHETIC, root=results_root)

    assert not runs.load(runs.SYNTHETIC, root=results_root).empty


def test_loading_a_kind_that_was_never_written_says_so(results_root):
    with pytest.raises(FileNotFoundError):
        runs.load(runs.SYNTHETIC, root=results_root)


def test_an_unknown_kind_is_rejected(results_root):
    with pytest.raises(ValueError, match="kind"):
        runs.load("synthetic-v2", root=results_root)


def test_only_the_requested_columns_are_read(results_root, synthetic_runs):
    runs.save(synthetic_runs, runs.SYNTHETIC, root=results_root)

    loaded = runs.load(
        runs.SYNTHETIC, root=results_root, columns=["base_quantifier", "method_simulator"]
    )

    assert list(loaded.columns) == ["base_quantifier", "method_simulator"]


def test_loading_labelled_attaches_what_every_reader_derives(
    results_root, synthetic_runs
):
    # The dashboard and the figure export each used to derive these for
    # themselves, which is half of the duplication the module exists to end.
    runs.save(synthetic_runs, runs.SYNTHETIC, root=results_root)

    labelled = runs.load_labelled(runs.SYNTHETIC, root=results_root)

    assert list(labelled.columns) == list(runs.columns_for(runs.SYNTHETIC)) + [
        "method",
        "absolute_error",
    ]
    pd.testing.assert_series_equal(
        labelled["method"], runs.method_labels(synthetic_runs), check_names=False
    )
    pd.testing.assert_series_equal(
        labelled["absolute_error"],
        runs.absolute_error(synthetic_runs),
        check_names=False,
    )


def test_a_missing_estimate_stays_missing(results_root, real_data_runs):
    # Haberman cannot fill a size-100 bag at high prevalence. That absence is
    # data (ADR-0005); a round trip must not quietly turn it into a number.
    real_data_runs.loc[0, "estimated_prevalence"] = None

    runs.save(real_data_runs, runs.REAL_DATA, root=results_root)

    assert pd.isna(runs.load(runs.REAL_DATA, root=results_root).loc[0, "estimated_prevalence"])


def test_a_prevalence_vector_needs_no_schema_change(results_root, real_data_runs):
    # Multiclass runs carry a vector where binary runs carry a scalar, in the
    # same columns (ADR-0003).
    multiclass = real_data_runs.assign(
        true_prevalence=[[0.2, 0.3, 0.5]] * len(real_data_runs),
        estimated_prevalence=[[0.1, 0.4, 0.5]] * len(real_data_runs),
        target_prevalence=[[0.2, 0.3, 0.5]] * len(real_data_runs),
    )

    runs.save(multiclass, runs.REAL_DATA, root=results_root)

    loaded = runs.load(runs.REAL_DATA, root=results_root)
    assert list(loaded.loc[0, "estimated_prevalence"]) == [0.1, 0.4, 0.5]


# ---------------------------------------------------------------------------
# Saving rejects what the readers would otherwise have to guess about
# ---------------------------------------------------------------------------


def test_saving_the_wrong_columns_is_rejected(results_root, synthetic_runs):
    with pytest.raises(ValueError, match="column"):
        runs.save(
            synthetic_runs.drop(columns=["bag_merging_factor"]),
            runs.SYNTHETIC,
            root=results_root,
        )


def test_saving_a_real_data_frame_as_synthetic_is_rejected(results_root, real_data_runs):
    with pytest.raises(ValueError, match="column"):
        runs.save(real_data_runs, runs.SYNTHETIC, root=results_root)


def test_saving_an_unknown_method_simulator_is_rejected(results_root, synthetic_runs):
    synthetic_runs.loc[0, "method_simulator"] = "Quadapt_MvN"

    with pytest.raises(ValueError, match="method_simulator"):
        runs.save(synthetic_runs, runs.SYNTHETIC, root=results_root)


def test_saving_an_unknown_base_quantifier_is_rejected(results_root, synthetic_runs):
    synthetic_runs.loc[0, "base_quantifier"] = "X"

    with pytest.raises(ValueError, match="base_quantifier"):
        runs.save(synthetic_runs, runs.SYNTHETIC, root=results_root)


# ---------------------------------------------------------------------------
# The method label
# ---------------------------------------------------------------------------


def test_a_run_with_no_method_simulator_is_labelled_by_its_base_quantifier():
    assert runs.method_label("DyS", runs.NO_METHOD_SIMULATOR) == "DyS"


def test_a_run_with_a_method_simulator_names_both():
    assert runs.method_label("DyS", runs.MVN) == "QuaDapt-MVN(DyS)"


def test_every_method_simulator_has_a_display_spelling():
    labels = {runs.method_label("DyS", name) for name in runs.METHOD_SIMULATORS}

    assert len(labels) == len(runs.METHOD_SIMULATORS)


def test_labels_are_derived_for_a_whole_frame(synthetic_runs):
    labelled = runs.method_labels(synthetic_runs)

    assert len(labelled) == len(synthetic_runs)
    assert set(labelled) == {
        runs.method_label(row.base_quantifier, row.method_simulator)
        for row in synthetic_runs.itertuples()
    }


def test_labelling_an_unknown_method_simulator_is_rejected():
    with pytest.raises(ValueError, match="method_simulator"):
        runs.method_label("DyS", "Quadapt_MvN")


# ---------------------------------------------------------------------------
# Derived quantities the renderers used to read from a stored column
# ---------------------------------------------------------------------------


def test_absolute_error_is_derived_from_the_recorded_prevalences(synthetic_runs):
    error = runs.absolute_error(synthetic_runs)

    expected = (
        synthetic_runs["estimated_prevalence"] - synthetic_runs["true_prevalence"]
    ).abs()
    pd.testing.assert_series_equal(error, expected, check_names=False)


def test_absolute_error_of_a_missing_estimate_is_missing(synthetic_runs):
    synthetic_runs.loc[0, "estimated_prevalence"] = None

    assert pd.isna(runs.absolute_error(synthetic_runs).loc[0])


def test_absolute_error_averages_over_the_classes_of_a_prevalence_vector():
    vectors = pd.DataFrame(
        {
            "true_prevalence": [[0.2, 0.3, 0.5]],
            "estimated_prevalence": [[0.1, 0.4, 0.5]],
        }
    )

    assert runs.absolute_error(vectors).loc[0] == pytest.approx(0.2 / 3)


# ---------------------------------------------------------------------------
# The vocabulary is the sweep's, not a second copy of it
# ---------------------------------------------------------------------------


def test_every_base_quantifier_the_sweep_runs_is_in_the_vocabulary():
    assert set(runs.BASE_QUANTIFIERS) == set(QUANTIFIERS)


def test_every_method_simulator_arm_of_the_sweep_has_a_canonical_name():
    assert set(runs.METHOD_SIMULATOR_NAMES) == set(SWEEP_METHOD_SIMULATORS)
    assert set(runs.METHOD_SIMULATOR_NAMES.values()) == set(runs.METHOD_SIMULATORS)


def test_every_data_simulator_the_sweep_draws_from_has_a_canonical_name():
    assert set(runs.SIMULATOR_NAMES) == set(SWEEP_DATA_SIMULATORS)
    assert set(runs.SIMULATOR_NAMES.values()) == set(runs.SIMULATORS)


def test_the_baseline_quantifier_is_one_the_sweep_runs():
    assert runs.BASELINE_QUANTIFIER in QUANTIFIERS
