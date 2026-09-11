"""The real-data pipeline: pools, cells, missing runs (ADR-0005, #10).

Fabricated pools throughout, never a downloaded dataset or a trained
classifier: everything above :func:`real_data.build_pool` takes a
:class:`real_data.Pool` already built, which is what lets this suite run with
no network (docs/testing.md). ``test_a_pool_is_built_from_out_of_fold_
probabilities`` and its neighbours are the exception — they exercise
``build_pool`` itself, on a tiny frame built in the test rather than fetched.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

import real_data
import runs
import sweep
from tests.score_sets import scored


def pool(name, n, prevalence, **score_kwargs):
    """A pool with a real reference score set but fabricated scores.

    Exercises the bag-drawing and estimator seams without training anything —
    the same reason ``tests/score_sets.py`` exists for the synthetic sweep.
    """
    scores, labels = scored(n, prevalence, **score_kwargs)
    return real_data.Pool(dataset=name, scores=scores, labels=labels)


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def test_a_grid_is_every_dataset_at_every_prevalence():
    cells = real_data.grid(("a", "b"), (0.2, 0.8))

    assert set(cells) == {
        real_data.Cell(dataset="a", target_prevalence=0.2),
        real_data.Cell(dataset="a", target_prevalence=0.8),
        real_data.Cell(dataset="b", target_prevalence=0.2),
        real_data.Cell(dataset="b", target_prevalence=0.8),
    }


def test_the_published_grid_has_twenty_one_prevalences():
    assert len(real_data.TARGET_PREVALENCES) == 21


def test_eight_datasets_are_registered():
    # ADR-0005: eight binary tabular datasets.
    assert len(real_data.DATASET_FETCHERS) == 8


# ---------------------------------------------------------------------------
# Drawing a bag: without replacement, and missing when the pool cannot supply it
# ---------------------------------------------------------------------------


@pytest.fixture
def haberman_shaped_pool():
    """Haberman's own shape (ADR-0005): 306 instances, 81 minority (26.5%).

    Fabricated scores, not a downloaded dataset: this suite is about the grid
    this shape produces, not about Haberman's own posterior scores.
    """
    return pool("haberman_survival", 306, 81 / 306)


def test_a_bag_within_the_pool_s_capacity_is_drawn_without_replacement(
    haberman_shaped_pool,
):
    indices = real_data.draw_bag(haberman_shaped_pool, 0.2, 100, random_state=1)

    assert indices is not None
    assert len(indices) == 100
    assert len(set(indices)) == 100


def test_a_bag_beyond_the_pool_s_capacity_is_missing(haberman_shaped_pool):
    # 81 minority instances cannot fill 90 of a size-100 bag (ADR-0005).
    assert real_data.draw_bag(haberman_shaped_pool, 0.9, 100, random_state=1) is None


def test_a_bag_at_the_pool_s_exact_capacity_is_drawn():
    exact = pool("exact", 200, 0.5)  # 100 of each class

    assert real_data.draw_bag(exact, 0.5, 100, random_state=1) is not None


def test_the_published_grid_s_high_prevalences_are_missing_for_haberman(
    haberman_shaped_pool,
):
    # The exact boundary this shape draws (verified against class_counts):
    # 80% needs 80 of the minority class and 85% needs 85, so the grid's own
    # step is what separates the last satisfiable cell from the first missing
    # one.
    missing = {
        p
        for p in real_data.TARGET_PREVALENCES
        if real_data.draw_bag(haberman_shaped_pool, p, 100, random_state=1) is None
    }

    assert missing == {p for p in real_data.TARGET_PREVALENCES if p > 0.80}


def test_a_seeded_draw_is_the_same_bag_twice(haberman_shaped_pool):
    one = real_data.draw_bag(haberman_shaped_pool, 0.2, 100, random_state=7)
    another = real_data.draw_bag(haberman_shaped_pool, 0.2, 100, random_state=7)

    np.testing.assert_array_equal(one, another)


def test_two_seeds_draw_different_bags(haberman_shaped_pool):
    one = real_data.draw_bag(haberman_shaped_pool, 0.2, 100, random_state=1)
    another = real_data.draw_bag(haberman_shaped_pool, 0.2, 100, random_state=2)

    assert list(one) != list(another)


# ---------------------------------------------------------------------------
# The spec's vocabulary is the runs module's, same as sweep.SweepSpec
# ---------------------------------------------------------------------------


#: Two pools, built once: a plain constant rather than a fixture, the same way
#: ``sweep.SMOKE_SWEEP`` is a module-level value and not something rebuilt per
#: test. Only ``mushroom`` is in the default grid below; ``haberman_survival``
#: is registered but otherwise unused unless a test asks for its cell by name.
TWO_POOLS = {
    "haberman_survival": pool("haberman_survival", 306, 81 / 306),
    "mushroom": pool("mushroom", 400, 0.5),
}

#: The smallest real-data spec that still covers every method — one cell, the
#: same relationship ``sweep.SMOKE_SWEEP`` has to ``test_sweep.py``. A single
#: cell keeps the full (11 base quantifiers) x (5 method-simulator arms) grid
#: to one bag's worth of estimating; ``test_an_unsatisfiable_cell_records_
#: every_method_as_missing`` and its neighbours swap in ``haberman_survival``
#: explicitly, which costs nothing to run since a missing bag skips estimation
#: entirely.
SMOKE_REAL_DATA_SPEC = real_data.RealDataSpec(
    cells=real_data.grid(("mushroom",), (0.4,)),
    pools=TWO_POOLS,
    method_simulators=sweep.METHOD_SIMULATORS,
    base_quantifiers=sweep.BASE_QUANTIFIERS,
    bag_size=100,
    repetitions=1,
    seed=20260911,
)


@pytest.fixture
def smoke_real_data_spec():
    """The module constant above, as a fixture.

    Tests that vary one thing about the spec say so by replacing one field of
    this, the property the spec exists to have.
    """
    return SMOKE_REAL_DATA_SPEC


def test_a_spec_rejects_a_quantifier_the_runs_module_cannot_store(
    smoke_real_data_spec,
):
    with pytest.raises(ValueError, match="base quantifier"):
        dataclasses.replace(smoke_real_data_spec, base_quantifiers={"MS": object})


def test_a_spec_rejects_a_cell_naming_an_unregistered_dataset(smoke_real_data_spec):
    with pytest.raises(ValueError, match="dataset"):
        dataclasses.replace(
            smoke_real_data_spec, cells=real_data.grid(("not-a-pool",), (0.4,))
        )


def test_a_spec_that_would_silently_drop_the_baseline_quantifier_is_rejected(
    smoke_real_data_spec,
):
    # Shares the rule sweep.SweepSpec enforces (sweep.reject_baseline_without_
    # a_home): CC reads no reference score set, so it only runs under the
    # no-method-simulator arm.
    with pytest.raises(ValueError, match=runs.BASELINE_QUANTIFIER):
        dataclasses.replace(
            smoke_real_data_spec,
            method_simulators={
                runs.UNIFORM: smoke_real_data_spec.method_simulators[runs.UNIFORM]
            },
        )


def test_a_spec_rejects_an_unknown_measure(smoke_real_data_spec):
    with pytest.raises(ValueError, match="measure"):
        dataclasses.replace(smoke_real_data_spec, measure="euclidean")


def test_a_spec_with_no_measure_is_accepted(smoke_real_data_spec):
    dataclasses.replace(smoke_real_data_spec, measure=None)


# ---------------------------------------------------------------------------
# What one cell produces
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mushroom_cell():
    """The runs of a single smoke cell, shared by the assertions below.

    Computed once (``scope="module"``): the full grid of base quantifiers and
    method-simulator arms is the same cost ``sweep.py``'s own ``one_cell``
    fixture pays, and recomputing it per assertion would multiply that cost
    by every test below instead of paying it once.
    """
    return real_data.run_cell(SMOKE_REAL_DATA_SPEC.cells[0], SMOKE_REAL_DATA_SPEC)


def test_a_cell_produces_runs_the_results_module_recognises(mushroom_cell):
    assert tuple(mushroom_cell.columns) == runs.columns_for(runs.REAL_DATA)


def test_a_satisfiable_cell_estimates_its_bags_by_every_method(mushroom_cell):
    expected = {
        (base, method)
        for base in SMOKE_REAL_DATA_SPEC.base_quantifiers
        if base != runs.BASELINE_QUANTIFIER
        for method in SMOKE_REAL_DATA_SPEC.method_simulators
    } | {(runs.BASELINE_QUANTIFIER, runs.NO_METHOD_SIMULATOR)}

    assert set(
        zip(mushroom_cell["base_quantifier"], mushroom_cell["method_simulator"])
    ) == expected
    assert mushroom_cell["estimated_prevalence"].notna().all()


def test_the_baseline_quantifier_is_recorded_once_and_not_under_a_simulator(
    mushroom_cell,
):
    baseline = mushroom_cell[mushroom_cell["base_quantifier"] == runs.BASELINE_QUANTIFIER]

    assert set(baseline["method_simulator"]) == {runs.NO_METHOD_SIMULATOR}


def test_each_repetition_draws_a_bag_of_its_own(smoke_real_data_spec):
    spec = dataclasses.replace(smoke_real_data_spec, repetitions=2)

    produced = real_data.run_cell(spec.cells[0], spec)

    assert set(produced["repetition"]) == {1, 2}
    assert (
        produced.groupby(["base_quantifier", "method_simulator"]).size().eq(2).all()
    )


def test_a_run_records_the_dataset_and_target_prevalence_of_its_cell(mushroom_cell):
    assert set(mushroom_cell["dataset"]) == {"mushroom"}
    assert set(mushroom_cell["target_prevalence"]) == {0.4}


def test_an_unsatisfiable_cell_records_every_method_as_missing(smoke_real_data_spec):
    # ADR-0005 exactly: Haberman cannot fill a size-100 bag at 90% positive, so
    # the absence is recorded for every method rather than the cell being
    # skipped.
    cell = real_data.Cell(dataset="haberman_survival", target_prevalence=0.9)
    spec = dataclasses.replace(smoke_real_data_spec, cells=(cell,))

    with pytest.warns(sweep.MissingRunWarning):
        produced = real_data.run_sweep(spec)

    expected = {
        (base, method)
        for base in spec.base_quantifiers
        if base != runs.BASELINE_QUANTIFIER
        for method in spec.method_simulators
    } | {(runs.BASELINE_QUANTIFIER, runs.NO_METHOD_SIMULATOR)}

    assert set(zip(produced["base_quantifier"], produced["method_simulator"])) == expected
    assert produced["estimated_prevalence"].isna().all()
    assert produced["true_prevalence"].isna().all()
    assert len(produced) == len(expected) * spec.repetitions


# ---------------------------------------------------------------------------
# Running the whole spec
# ---------------------------------------------------------------------------


def test_running_the_same_spec_twice_produces_identical_runs(smoke_real_data_spec):
    pd.testing.assert_frame_equal(
        real_data.run_sweep(smoke_real_data_spec),
        real_data.run_sweep(smoke_real_data_spec),
    )


def test_the_number_of_workers_does_not_change_the_runs(smoke_real_data_spec):
    spec = dataclasses.replace(
        smoke_real_data_spec,
        base_quantifiers={"DyS": smoke_real_data_spec.base_quantifiers["DyS"]},
        method_simulators={runs.NO_METHOD_SIMULATOR: None},
    )

    pd.testing.assert_frame_equal(
        real_data.run_sweep(spec), real_data.run_sweep(spec, n_jobs=2)
    )


def test_a_sweep_is_written_through_the_results_module(
    smoke_real_data_spec, results_root
):
    # Not sweep.validate_and_save here: its ADR-0001 defect check is exercised
    # against realistic fixtures in test_sweep.py, and this fixture's
    # cleanly-separated fabricated scores (tests/score_sets.scored) make many
    # methods agree by construction — a false positive for that check, not the
    # defect it exists to catch.
    spec = dataclasses.replace(
        smoke_real_data_spec, cells=(real_data.Cell("mushroom", 0.4),)
    )

    path = runs.save(real_data.run_sweep(spec), runs.REAL_DATA, root=results_root)

    assert path == results_root / "runs" / "real-data.parquet"
    assert not runs.load(runs.REAL_DATA, root=results_root).empty


def test_missing_runs_survive_the_results_module(smoke_real_data_spec, results_root):
    cell = real_data.Cell(dataset="haberman_survival", target_prevalence=0.9)
    spec = dataclasses.replace(smoke_real_data_spec, cells=(cell,))

    with pytest.warns(sweep.MissingRunWarning):
        produced = real_data.run_sweep(spec)
    runs.save(produced, runs.REAL_DATA, root=results_root)

    labelled = runs.load_labelled(runs.REAL_DATA, root=results_root)
    assert labelled["estimated_prevalence"].isna().all()
    assert labelled["absolute_error"].isna().all()


# ---------------------------------------------------------------------------
# The published spec's shape (ADR-0005)
# ---------------------------------------------------------------------------


def test_the_published_spec_draws_bags_of_a_hundred_over_ten_repetitions():
    spec = real_data.build_spec({"mushroom": pool("mushroom", 400, 0.5)}, seed=1)

    assert spec.bag_size == 100
    assert spec.repetitions == 10


def test_the_published_spec_speaks_the_stored_vocabulary():
    spec = real_data.build_spec({"mushroom": pool("mushroom", 400, 0.5)}, seed=1)

    assert set(spec.method_simulators) == set(runs.METHOD_SIMULATORS)
    assert set(spec.base_quantifiers) == set(runs.BASE_QUANTIFIERS)


def test_the_published_spec_covers_every_prevalence_for_every_pool_it_is_given():
    pools = {"mushroom": pool("mushroom", 400, 0.5), "banknote": pool("banknote", 400, 0.5)}
    spec = real_data.build_spec(pools, seed=1)

    assert len(spec.cells) == len(pools) * len(real_data.TARGET_PREVALENCES)


# ---------------------------------------------------------------------------
# Building a pool: the classifier and its out-of-fold scores
# ---------------------------------------------------------------------------
#
# The one seam that reaches scikit-learn rather than a fabricated pool. Still
# no network: the frame below stands in for a fetched dataset's ``(X, y)``.


@pytest.fixture
def raw_frame():
    """A tiny mixed-dtype frame, standing in for a fetched dataset's ``(X, y)``."""
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(
        {"num": rng.normal(size=n), "cat": rng.choice(["a", "b", "?"], size=n)}
    )
    y = pd.Series(rng.choice(["neg", "pos"], size=n, p=[0.7, 0.3]))
    return X, y


def test_a_pool_is_built_from_out_of_fold_probabilities(raw_frame):
    X, y = raw_frame

    built = real_data.build_pool("fake", X, y, cv_folds=3, random_state=0)

    assert built.scores.shape == (len(y), 2)
    assert set(built.labels) <= {0, 1}
    np.testing.assert_allclose(built.scores.sum(axis=1), 1.0)


def test_the_higher_sorted_label_is_positive(raw_frame):
    # The same rule sweep.observed_prevalence reads a drawn bag's prevalence
    # by, so "positive" means the same thing here as in a synthetic run.
    X, y = raw_frame

    built = real_data.build_pool("fake", X, y, cv_folds=3, random_state=0)

    assert set(built.labels[y.to_numpy() == "pos"]) == {1}
    assert set(built.labels[y.to_numpy() == "neg"]) == {0}


def test_a_pool_larger_than_the_cap_is_subsampled(raw_frame):
    X, y = raw_frame

    built = real_data.build_pool("fake", X, y, cap=30, cv_folds=3, random_state=0)

    assert len(built.labels) == 30


def test_a_pool_within_the_cap_is_not_subsampled(raw_frame):
    X, y = raw_frame

    built = real_data.build_pool("fake", X, y, cap=1000, cv_folds=3, random_state=0)

    assert len(built.labels) == len(y)


def test_a_non_binary_target_is_rejected():
    X = pd.DataFrame({"num": [1, 2, 3, 4, 5, 6]})
    y = pd.Series(["a", "b", "c", "a", "b", "c"])

    with pytest.raises(ValueError, match="binary"):
        real_data.build_pool("fake", X, y, cv_folds=3)
