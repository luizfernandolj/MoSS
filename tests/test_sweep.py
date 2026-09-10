"""The sweep as a function of a spec.

Everything the sweep did used to be reachable only by importing a module and
taking its globals: ``from variables import *`` put a 19-value merging-factor
range, seven prevalences and three registries into the sweep's namespace, and
nothing could run a smaller grid without editing that file. The grid is now a
value the caller passes, so a smoke sweep and the published sweep are the same
code with different arguments.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from mlquantify.counting import CC
from mlquantify.matching import DyS

import runs
import sweep
from sweep import (
    BaselineEstimator,
    MetaEstimator,
    ReferenceEstimator,
    ReferenceScoreSet,
    grid,
    run_cell,
    run_sweep,
)
from utils.simulators import MVNSimulator, UniformSimulator


# ---------------------------------------------------------------------------
# The grid is a parameter
# ---------------------------------------------------------------------------


def test_a_grid_is_the_cross_product_of_what_it_is_given():
    cells = grid(
        simulators=(runs.UNIFORM, runs.MVN),
        merging_factors=(0.2, 0.8),
        target_prevalences=(0.4,),
    )

    # Reference simulator × bag simulator × reference merging factor × bag
    # merging factor × target prevalence.
    assert len(cells) == 2 * 2 * 2 * 2 * 1
    assert len(set(cells)) == len(cells)


def test_a_grid_pairs_every_reference_simulator_with_every_bag_simulator():
    cells = grid(
        simulators=(runs.UNIFORM, runs.MVN),
        merging_factors=(0.5,),
        target_prevalences=(0.4,),
    )

    assert {(cell.reference_simulator, cell.bag_simulator) for cell in cells} == {
        (runs.UNIFORM, runs.UNIFORM),
        (runs.UNIFORM, runs.MVN),
        (runs.MVN, runs.UNIFORM),
        (runs.MVN, runs.MVN),
    }


def test_a_spec_rejects_a_cell_naming_a_simulator_it_cannot_draw_from(smoke_spec):
    unreachable = grid(
        simulators=(runs.DIRICHLET,),
        merging_factors=(0.5,),
        target_prevalences=(0.4,),
    )

    with pytest.raises(ValueError, match="simulator"):
        dataclasses.replace(
            smoke_spec,
            cells=unreachable,
            data_simulators={runs.UNIFORM: smoke_spec.data_simulators[runs.UNIFORM]},
        )


# ---------------------------------------------------------------------------
# The spec's vocabulary is the runs module's
# ---------------------------------------------------------------------------


def test_the_published_sweep_speaks_the_stored_vocabulary():
    # Not a translation table that could go stale: the keys the sweep registers
    # its simulators and quantifiers under are the names written to disk, so a
    # rename in either place fails here rather than producing runs no reader
    # can name.
    spec = sweep.SYNTHETIC_SWEEP

    assert set(spec.data_simulators) == set(runs.SIMULATORS)
    assert set(spec.method_simulators) == set(runs.METHOD_SIMULATORS)
    assert set(spec.base_quantifiers) == set(runs.BASE_QUANTIFIERS)


def test_a_spec_naming_a_quantifier_the_runs_module_cannot_store_is_rejected(smoke_spec):
    with pytest.raises(ValueError, match="base quantifier"):
        dataclasses.replace(smoke_spec, base_quantifiers={"MS": object})


def test_a_spec_that_would_silently_drop_the_baseline_quantifier_is_rejected(smoke_spec):
    # It only runs under the no-method-simulator arm, so a spec that registers
    # it without that arm produces no CC runs at all — and would have said
    # nothing about it, which is the quiet half of the defect this seam ends.
    with pytest.raises(ValueError, match=runs.BASELINE_QUANTIFIER):
        dataclasses.replace(
            smoke_spec,
            method_simulators={
                runs.UNIFORM: smoke_spec.method_simulators[runs.UNIFORM]
            },
        )


# ---------------------------------------------------------------------------
# One estimator interface, one adapter per calling convention
# ---------------------------------------------------------------------------
#
# mlquantify asks for a bag's prevalence in three different ways, and the three
# adapters below are the whole of what differs between them. The tests assert
# the convention itself — which inputs an adapter reads — rather than a number
# recomputed the way the code computes it.

#: A bag of 100 rows, 30 of them confidently positive. Every quantifier below
#: should read a prevalence of 0.3 out of it, which is an answer known without
#: running anything.
BAG_PREVALENCE = 0.3


def separated_scores(n, prevalence):
    """Scores so cleanly separated that the true prevalence is unmistakable."""
    n_positive = round(n * prevalence)
    positive = np.tile([0.05, 0.95], (n_positive, 1))
    negative = np.tile([0.95, 0.05], (n - n_positive, 1))
    labels = np.concatenate((np.ones(n_positive, dtype=int), np.zeros(n - n_positive, dtype=int)))
    return np.vstack((positive, negative)), labels


@pytest.fixture
def bag():
    scores, _ = separated_scores(100, BAG_PREVALENCE)
    return scores


@pytest.fixture
def reference():
    scores, labels = separated_scores(400, 0.5)
    return ReferenceScoreSet(scores, labels)


#: Any fixed seed. mlquantify calls its ``MoSS`` seam without one (ADR-0004),
#: so a meta-quantifier's estimate could not be compared to anything until the
#: estimator carried its own — which is what these tests now hand it.
METHOD_SEED = 20260909


def contradicted(reference):
    """The same reference score set with its two classes' scores swapped.

    Every row now says the opposite of what its label says, so a quantifier
    that reads these scores must answer differently and one that does not read
    them must answer the same.
    """
    return ReferenceScoreSet(reference.scores[:, ::-1], reference.labels)


def test_a_baseline_estimator_reads_the_bag_and_nothing_else(bag):
    # CC classifies and counts, so a bag of 30 confident positives in 100 is
    # a prevalence of 0.3 whatever reference exists. It takes none.
    assert BaselineEstimator(CC).estimate(bag) == pytest.approx(BAG_PREVALENCE)


def test_a_reference_estimator_matches_the_bag_against_the_real_scores(bag, reference):
    assert ReferenceEstimator(DyS, reference).estimate(bag) == pytest.approx(
        BAG_PREVALENCE, abs=0.05
    )


def test_a_reference_estimator_changes_its_answer_when_the_reference_does(bag, reference):
    # The reference score set is the distribution it matches against, so
    # replacing it with one that says the opposite must move the estimate.
    assert ReferenceEstimator(DyS, reference).estimate(bag) != ReferenceEstimator(
        DyS, contradicted(reference)
    ).estimate(bag)


def test_a_meta_estimator_reads_the_reference_labels_but_not_its_scores(bag, reference):
    # A meta-quantifier does not match against the real reference score set at
    # all: it simulates candidates and picks one. The reference is there only
    # to say which classes exist, so contradicting its scores changes nothing.
    simulator = UniformSimulator()

    assert MetaEstimator(
        DyS, simulator, reference, METHOD_SEED
    ).estimate(bag) == MetaEstimator(
        DyS, simulator, contradicted(reference), METHOD_SEED
    ).estimate(bag)


def test_a_meta_estimator_draws_with_the_method_simulator_it_was_given(bag, reference):
    # The method simulator is a property of the estimator (CONTEXT.md), so two
    # estimators differing only in it are two different methods. Held on one
    # seed, so that what differs between the two estimates is the simulator
    # rather than the draw.
    one = MetaEstimator(DyS, UniformSimulator(), reference, METHOD_SEED).estimate(bag)
    another = MetaEstimator(DyS, MVNSimulator(), reference, METHOD_SEED).estimate(bag)

    assert one != another


def test_a_seeded_meta_estimator_makes_the_same_estimate_twice(bag, reference):
    # The draw the library never seeds. Two estimators alike in everything,
    # including their seed, are the same method on the same bag, so a
    # difference between them could only come from the candidate score sets
    # they drew — which is the nondeterminism ADR-0004 exists to remove.
    estimator = MetaEstimator(DyS, UniformSimulator(), reference, METHOD_SEED)

    assert estimator.estimate(bag) == estimator.estimate(bag)


def test_two_meta_estimators_on_different_seeds_draw_different_candidates(bag, reference):
    one = MetaEstimator(DyS, UniformSimulator(), reference, 1).estimate(bag)
    another = MetaEstimator(DyS, UniformSimulator(), reference, 2).estimate(bag)

    assert one != another


# ---------------------------------------------------------------------------
# What one cell produces
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def one_cell():
    """The runs of a single smoke cell, shared by the assertions below."""
    return run_cell(sweep.SMOKE_SWEEP.cells[0], sweep.SMOKE_SWEEP)


def test_a_cell_produces_runs_the_results_module_recognises(one_cell):
    assert tuple(one_cell.columns) == runs.columns_for(runs.SYNTHETIC)


def test_a_cell_estimates_its_bags_by_every_method(one_cell, smoke_spec):
    # Every base quantifier under every method simulator, except the baseline,
    # which reads no reference and so has no meta-quantifier arm.
    expected = {
        (base, method)
        for base in smoke_spec.base_quantifiers
        if base != runs.BASELINE_QUANTIFIER
        for method in smoke_spec.method_simulators
    } | {(runs.BASELINE_QUANTIFIER, runs.NO_METHOD_SIMULATOR)}

    assert set(zip(one_cell["base_quantifier"], one_cell["method_simulator"])) == expected


def test_the_baseline_quantifier_is_recorded_once_and_not_under_a_simulator(one_cell):
    # It classifies and counts; no method simulator takes any part in its
    # estimate. Recording it under each arm would file the same number three
    # more times under three methods that did not produce it.
    baseline = one_cell[one_cell["base_quantifier"] == runs.BASELINE_QUANTIFIER]

    assert set(baseline["method_simulator"]) == {runs.NO_METHOD_SIMULATOR}


def test_every_run_records_the_cell_it_came_from(one_cell):
    cell = sweep.SMOKE_SWEEP.cells[0]

    assert set(one_cell["reference_simulator"]) == {cell.reference_simulator}
    assert set(one_cell["bag_simulator"]) == {cell.bag_simulator}
    assert set(one_cell["reference_merging_factor"]) == {cell.reference_merging_factor}
    assert set(one_cell["bag_merging_factor"]) == {cell.bag_merging_factor}
    assert set(one_cell["target_prevalence"]) == {cell.target_prevalence}


def test_a_run_records_the_prevalence_of_the_bag_that_was_actually_drawn(one_cell):
    # A bag of 100 drawn at 0.4 holds exactly 40 positives, so the target and
    # the truth agree here. They stop agreeing on a real dataset that cannot
    # fill the bag at the prevalence asked for (ADR-0005), which is why both
    # are recorded.
    assert set(one_cell["true_prevalence"]) == {0.4}


def test_each_repetition_draws_a_bag_of_its_own(smoke_spec):
    spec = dataclasses.replace(smoke_spec, repetitions=2)

    produced = run_cell(spec.cells[0], spec)

    assert set(produced["repetition"]) == {1, 2}
    assert produced.groupby(["base_quantifier", "method_simulator"]).size().eq(2).all()


# ---------------------------------------------------------------------------
# Where a seed comes from
# ---------------------------------------------------------------------------
#
# A spec used to carry one seed and hand it to every cell, so two cells that
# shared a simulator and a merging factor drew the same scores — the whole grid
# resampling one draw rather than sampling the space it was built to cover.


def cells_differing_in(**field):
    """Two cells alike but for one field, as :func:`seed_for` sees them."""
    base = grid((runs.UNIFORM,), (0.5,), (0.4,))[0]
    return base, dataclasses.replace(base, **field)


@pytest.mark.parametrize(
    "field",
    [
        {"reference_simulator": runs.MVN},
        {"reference_merging_factor": 0.8},
        {"bag_simulator": runs.MVN},
        {"bag_merging_factor": 0.8},
        {"target_prevalence": 0.6},
    ],
)
def test_two_cells_alike_but_for_one_field_derive_different_seeds(smoke_spec, field):
    one, another = cells_differing_in(**field)

    assert sweep.seed_for(smoke_spec, one, 1, sweep.BAG_DRAW) != sweep.seed_for(
        smoke_spec, another, 1, sweep.BAG_DRAW
    )


def test_the_same_cell_derives_the_same_seed_every_time(smoke_spec):
    cell = smoke_spec.cells[0]

    assert sweep.seed_for(smoke_spec, cell, 1, sweep.BAG_DRAW) == sweep.seed_for(
        smoke_spec, cell, 1, sweep.BAG_DRAW
    )


def test_each_repetition_and_each_draw_of_a_cell_gets_a_seed_of_its_own(smoke_spec):
    cell = smoke_spec.cells[0]

    derived = {
        sweep.seed_for(smoke_spec, cell, repetition, draw)
        for repetition in (1, 2, 3)
        for draw in (sweep.REFERENCE_DRAW, sweep.BAG_DRAW, sweep.CANDIDATE_DRAW)
    }

    assert len(derived) == 9


def test_two_specs_on_different_seeds_derive_different_seeds(smoke_spec):
    other = dataclasses.replace(smoke_spec, seed=smoke_spec.seed + 1)

    assert sweep.seed_for(smoke_spec, smoke_spec.cells[0], 1, sweep.BAG_DRAW) != (
        sweep.seed_for(other, other.cells[0], 1, sweep.BAG_DRAW)
    )


def test_a_spec_with_no_seed_derives_none(smoke_spec):
    # Not a seed of its own: ``None`` is how a caller asks for OS entropy, and
    # deriving something from it would take that option away.
    unseeded = dataclasses.replace(smoke_spec, seed=None)

    assert sweep.seed_for(unseeded, unseeded.cells[0], 1, sweep.BAG_DRAW) is None


# ---------------------------------------------------------------------------
# The same spec twice
# ---------------------------------------------------------------------------


def test_running_the_same_spec_twice_produces_identical_runs(smoke_spec):
    # Every arm of the smoke spec, including the meta-quantifier's, whose
    # candidate score sets mlquantify draws through a seam it never passes a
    # seed to (ADR-0004). An unseeded draw there is what disguised ADR-0001's
    # defect, so this is a correctness check and not a convenience.
    pd.testing.assert_frame_equal(run_sweep(smoke_spec), run_sweep(smoke_spec))


def test_the_bags_a_cell_draws_do_not_depend_on_the_reference_before_them(smoke_spec):
    # The baseline quantifier reads no reference score set at all, so its
    # estimate is a function of the bag alone. The bags used to be drawn from
    # the stream the reference had just been drawn from, which made every bag
    # in the sweep a function of how large a reference set preceded it — one
    # draw's parameters reaching into another draw's numbers.
    bags_only = plain_only(smoke_spec, {runs.BASELINE_QUANTIFIER: CC})
    with_a_bigger_reference = dataclasses.replace(
        bags_only, reference_size=bags_only.reference_size * 2
    )

    pd.testing.assert_frame_equal(
        run_sweep(bags_only), run_sweep(with_a_bigger_reference)
    )


def test_the_number_of_workers_does_not_change_the_runs(smoke_spec):
    # The published sweep runs at n_jobs=-1. A seed derived from a cell through
    # Python's own ``hash`` would be salted per process, so every worker would
    # draw its own scores and the property above would hold only in the one
    # configuration nobody runs.
    #
    # One meta-quantifier arm and one base quantifier: what has to cross the
    # process boundary is the derivation, once per draw, and the rest of the
    # registry would only buy the same crossing again at ten times the cost.
    spec = dataclasses.replace(
        smoke_spec,
        base_quantifiers={"DyS": smoke_spec.base_quantifiers["DyS"]},
        method_simulators={runs.UNIFORM: smoke_spec.method_simulators[runs.UNIFORM]},
        cells=grid((runs.UNIFORM,), (0.2, 0.8), (0.4,)),
    )

    pd.testing.assert_frame_equal(run_sweep(spec), run_sweep(spec, n_jobs=2))


# ---------------------------------------------------------------------------
# A failing estimator records a missing run
# ---------------------------------------------------------------------------
#
# ADR-0001's second defect. A caught exception was logged and fell through to
# the row-building code, where the local holding the estimate still held the
# *previous* loop iteration's value — so the failing method was recorded under
# its own name carrying another method's number. About 4.5% of the 3.75M voided
# runs were written that way, and nothing in the file said which.


class FailingQuantifier:
    """A base quantifier that cannot produce an estimate."""

    def aggregate(self, *args, **kwargs):
        raise RuntimeError("this method has no estimate to give")


def plain_only(spec, base_quantifiers):
    """``spec`` with these base quantifiers and no meta-quantifier arm."""
    return dataclasses.replace(
        spec,
        base_quantifiers=base_quantifiers,
        method_simulators={runs.NO_METHOD_SIMULATOR: None},
    )


def test_a_failing_estimator_records_a_missing_run(smoke_spec):
    spec = plain_only(smoke_spec, {"DyS": FailingQuantifier})

    with pytest.warns(sweep.MissingRunWarning):
        produced = run_sweep(spec)

    assert len(produced) == spec.repetitions
    assert produced["estimated_prevalence"].isna().all()


def test_a_failing_run_never_carries_another_method_s_estimate(smoke_spec):
    # The defect exactly: the run after a successful one inherited its number.
    spec = plain_only(
        smoke_spec,
        {"DyS": smoke_spec.base_quantifiers["DyS"], "HDy": FailingQuantifier},
    )

    with pytest.warns(sweep.MissingRunWarning):
        produced = run_sweep(spec)

    estimated = produced.set_index("base_quantifier")["estimated_prevalence"]
    assert pd.notna(estimated.loc["DyS"])
    assert pd.isna(estimated.loc["HDy"])


def test_a_missing_run_still_records_everything_that_was_asked_of_it(smoke_spec):
    # The absence is data (ADR-0005). Which method failed, on which bag, is
    # what a null estimate beside a full data block records — and it is what
    # dropping the row instead would throw away.
    spec = plain_only(smoke_spec, {"DyS": FailingQuantifier})

    with pytest.warns(sweep.MissingRunWarning):
        produced = run_sweep(spec)

    assert produced["true_prevalence"].notna().all()
    assert set(produced["base_quantifier"]) == {"DyS"}


def test_a_missing_run_names_the_method_and_how_many_runs_it_lost(smoke_spec):
    spec = dataclasses.replace(
        plain_only(smoke_spec, {"DyS": FailingQuantifier}), repetitions=4
    )

    with pytest.warns(sweep.MissingRunWarning, match=r"DyS produced no estimate for 4"):
        run_sweep(spec)


def test_one_broken_method_warns_once_however_many_runs_it_loses(smoke_spec):
    # A warning per failure is a warning per run: on the published grid a
    # method broken everywhere would emit millions of identical lines.
    spec = dataclasses.replace(
        plain_only(smoke_spec, {"DyS": FailingQuantifier}),
        cells=grid((runs.UNIFORM,), (0.2, 0.8), (0.4,)),
        repetitions=3,
    )

    with pytest.warns(sweep.MissingRunWarning) as caught:
        run_sweep(spec)

    assert len(caught) == 1


def test_a_failure_inside_a_worker_still_reaches_the_caller(smoke_spec):
    # The published sweep runs with n_jobs=-1. A warning raised inside a cell
    # would land on a loky worker's stderr and never enter the parent's warning
    # machinery, so the one path that matters would be the silent one.
    spec = dataclasses.replace(
        plain_only(smoke_spec, {"DyS": FailingQuantifier}),
        cells=grid((runs.UNIFORM,), (0.2, 0.8), (0.4,)),
    )

    with pytest.warns(sweep.MissingRunWarning):
        run_sweep(spec, n_jobs=2)


def test_a_sweep_runs_its_cells_over_several_processes(smoke_spec):
    # How the published sweep runs, and the only thing that requires the spec
    # to be picklable: it crosses into a worker whole. Deliberately narrow, so
    # that what is being paid for is the worker round trip and not the
    # quantifiers.
    spec = dataclasses.replace(
        smoke_spec,
        base_quantifiers={"DyS": smoke_spec.base_quantifiers["DyS"]},
        method_simulators={runs.NO_METHOD_SIMULATOR: None},
        cells=grid((runs.UNIFORM,), (0.2, 0.8), (0.4,)),
    )

    produced = run_sweep(spec, n_jobs=2)

    assert len(produced) == len(spec.cells) * spec.repetitions
    assert produced["estimated_prevalence"].notna().all()


def test_a_sweep_is_written_through_the_results_module(smoke_spec, results_root):
    # The sweep used to write ``results/results.csv`` in column names of its
    # own, which is how the readers came to disagree about what a run is. It
    # now produces exactly what ``runs.save`` accepts, and ``runs.save``
    # refuses anything a reader would have to guess about.
    runs.save(run_sweep(smoke_spec), runs.SYNTHETIC, root=results_root)

    labelled = runs.load_labelled(runs.SYNTHETIC, root=results_root)

    # Spelled out rather than derived, so that a change to how a method is
    # named has to be made here too.
    assert {"CC", "DyS", "QuaDapt-MVN(DyS)", "QuaDapt-Dirichlet(TMS2)"} <= set(
        labelled["method"]
    )
    assert labelled["absolute_error"].notna().all()


def test_missing_runs_survive_the_results_module(smoke_spec, results_root):
    # A missing estimate has to reach the reader as missing; an error derived
    # from it is missing too, rather than silently zero.
    spec = plain_only(smoke_spec, {"DyS": FailingQuantifier})

    with pytest.warns(sweep.MissingRunWarning):
        runs.save(run_sweep(spec), runs.SYNTHETIC, root=results_root)

    labelled = runs.load_labelled(runs.SYNTHETIC, root=results_root)
    assert labelled["estimated_prevalence"].isna().all()
    assert labelled["absolute_error"].isna().all()
