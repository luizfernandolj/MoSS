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
    CandidateEstimator,
    DefectSignatureError,
    MetaEstimator,
    ReferenceEstimator,
    ReferenceScoreSet,
    collapsed_method_simulator_groups,
    grid,
    run_cell,
    run_measure_ablation,
    run_sweep,
    stale_estimate_rows,
    validate,
)
from tests.score_sets import UNMATCHABLE, FixedSimulator, scored
from utils.meta_quantifier import HISTOGRAM_MEASURES, QuaDaptWithSimulator
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


def test_runs_measures_matches_the_meta_quantifier_s_own_dispatch():
    # Not a translation table either: utils.meta_quantifier dispatches "sord"
    # separately from its histogram measures, so the two lists are asserted to
    # agree rather than one being copied from the other.
    assert set(runs.MEASURES) == set(HISTOGRAM_MEASURES) | {"sord"}


def test_the_published_sweep_runs_ten_repetitions():
    # The re-run this ticket exists to produce (#9): three was the number
    # ADR-0001 voided every run drawn under.
    assert sweep.SYNTHETIC_SWEEP.repetitions == 10


def test_the_published_sweep_does_not_restate_the_library_s_default_measure():
    # ``None`` forwards nothing (:func:`sweep._quadapt_kwargs`), which is what
    # keeps this project from freezing a copy of upstream's own default — the
    # trap ADR-0007 caught the last time it held one (ADR-0012).
    assert sweep.SYNTHETIC_SWEEP.measure is None


def test_the_published_ablation_grid_is_smaller_than_the_published_grid():
    assert len(sweep.MEASURE_ABLATION_GRID) < len(sweep.SYNTHETIC_SWEEP.cells)


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
# What a method reads decides how it is asked for an estimate, and the four
# adapters below are the whole of what differs between them. The tests assert
# the convention itself — which inputs an adapter reads — rather than a number
# recomputed the way the code computes it.

#: A bag of 100 rows, 30 of them confidently positive. Every quantifier below
#: should read a prevalence of 0.3 out of it, which is an answer known without
#: running anything.
BAG_PREVALENCE = 0.3


@pytest.fixture
def bag():
    scores, _ = scored(100, BAG_PREVALENCE)
    return scores


@pytest.fixture
def reference():
    scores, labels = scored(400, 0.5)
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


def test_a_candidate_estimator_reads_the_real_reference_scores(bag, reference):
    # The one meta-quantifier arm that does (ADR-0010). The adapter above is
    # given the same reference and reads only its labels, and the test above
    # asserts exactly that; here contradicting the scores has to move the
    # estimate, because they are one of the candidates being chosen between.
    # The simulated candidates cannot be matched, so whatever the estimate is,
    # they did not produce it.
    simulators = (FixedSimulator(**UNMATCHABLE),)

    assert CandidateEstimator(DyS, simulators, reference, METHOD_SEED).estimate(
        bag
    ) != CandidateEstimator(
        DyS, simulators, contradicted(reference), METHOD_SEED
    ).estimate(bag)


@pytest.fixture
def rejected_reference():
    """A reference score set the candidate search has to reject.

    Both classes at one score, so every mixture of it is the same distribution
    and it matches no bag better than any other (``score_sets.scored``). The
    winning candidate is therefore always a simulated one, which is what makes
    the two tests below about the draw rather than about the real scores.
    """
    scores, labels = scored(400, 0.5, **UNMATCHABLE)
    return ReferenceScoreSet(scores, labels)


def test_a_seeded_candidate_estimator_makes_the_same_estimate_twice(
    bag, rejected_reference
):
    # The same requirement as the arm above, over more draws: this one
    # simulates a candidate per simulator per merging factor rather than one
    # simulator's grid (ADR-0010), and repeating the estimate has to repeat all
    # of them.
    estimator = CandidateEstimator(
        DyS, (UniformSimulator(),), rejected_reference, METHOD_SEED
    )

    assert estimator.estimate(bag) == estimator.estimate(bag)


def test_two_candidate_estimators_on_different_seeds_draw_different_candidates(
    bag, rejected_reference
):
    # What keeps the test above from passing by construction: the estimate
    # moves with the seed, so the candidates it is drawn from are reaching it.
    one = CandidateEstimator(DyS, (UniformSimulator(),), rejected_reference, 1)
    another = CandidateEstimator(DyS, (UniformSimulator(),), rejected_reference, 2)

    assert one.estimate(bag) != another.estimate(bag)


# ---------------------------------------------------------------------------
# The distance measure both meta-quantifier adapters forward (#9, ADR-0012)
# ---------------------------------------------------------------------------
#
# ``measure`` defaults to ``None`` rather than to the library's own default, so
# that leaving it out never freezes a copy of upstream's choice — the trap
# ADR-0007 caught the last time this project held one. These tests are about
# the *wiring*: that a measure handed to an adapter is the one the underlying
# meta-quantifier actually receives, not a recomputation of what either measure
# should produce.


def test_a_meta_estimator_with_no_measure_leaves_the_library_default_in_place(
    bag, reference
):
    simulator = UniformSimulator()

    with_none = MetaEstimator(DyS, simulator, reference, METHOD_SEED).estimate(bag)
    with_default = MetaEstimator(
        DyS, simulator, reference, METHOD_SEED, measure="topsoe"
    ).estimate(bag)

    assert with_none == pytest.approx(with_default)


def test_a_meta_estimator_forwards_its_measure_to_the_meta_quantifier(bag, reference):
    simulator = UniformSimulator()

    direct = QuaDaptWithSimulator(
        DyS(), simulator, METHOD_SEED, measure="hellinger"
    ).aggregate(bag, reference.labels)
    via_estimator = MetaEstimator(
        DyS, simulator, reference, METHOD_SEED, measure="hellinger"
    ).estimate(bag)

    assert via_estimator == pytest.approx(direct[1])


def test_a_candidate_estimator_forwards_its_measure_to_the_meta_quantifier(
    bag, rejected_reference
):
    # utils.meta_quantifier dispatches on ``measure`` itself and raises a clean
    # ValueError for anything it does not recognise — a deterministic proof
    # that the value reached it rather than being silently ignored.
    estimator = CandidateEstimator(
        DyS,
        (UniformSimulator(),),
        rejected_reference,
        METHOD_SEED,
        measure="not-a-real-measure",
    )

    with pytest.raises(ValueError, match="measure"):
        estimator.estimate(bag)


def test_a_spec_rejects_an_unknown_measure(smoke_spec):
    with pytest.raises(ValueError, match="measure"):
        dataclasses.replace(smoke_spec, measure="euclidean")


def test_a_spec_with_no_measure_is_accepted(smoke_spec):
    # ``None`` is the published sweep's own choice (ADR-0012), not an
    # oversight, so it must not be rejected as "unknown".
    dataclasses.replace(smoke_spec, measure=None)


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


def test_the_arm_that_searches_every_candidate_survives_the_worker_boundary(smoke_spec):
    # It holds several simulators where the other arms hold one, and builds a
    # meta-quantifier per estimate that must not carry anything over between
    # them (ADR-0010). Both are properties a single process can hide.
    spec = dataclasses.replace(
        smoke_spec,
        base_quantifiers={"DyS": smoke_spec.base_quantifiers["DyS"]},
        method_simulators={
            runs.ALL_SIMULATORS: smoke_spec.method_simulators[runs.ALL_SIMULATORS]
        },
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


# ---------------------------------------------------------------------------
# The distance-measure ablation (#9, ADR-0012)
# ---------------------------------------------------------------------------


def test_the_published_ablation_spec_excludes_the_baseline_and_its_arm():
    # CC and the no-method-simulator arm never read ``measure``, so running
    # them once per entry of runs.MEASURES would only repeat the same rows.
    spec = sweep.MEASURE_ABLATION_SWEEP

    assert runs.BASELINE_QUANTIFIER not in spec.base_quantifiers
    assert runs.NO_METHOD_SIMULATOR not in spec.method_simulators


@pytest.fixture(scope="module")
def ablation_spec():
    """A spec small enough for this suite, shaped like the published ablation.

    Built from ``sweep.SMOKE_SWEEP`` directly rather than the ``smoke_spec``
    fixture, so this can be module-scoped the way ``ablation_runs`` below
    needs it to be. Excludes the arms that never read ``measure``, the same
    way ``sweep.MEASURE_ABLATION_SWEEP`` does — a spec that kept them would
    still run, just wastefully, repeating their rows once per measure.
    """
    smoke = sweep.SMOKE_SWEEP
    return dataclasses.replace(
        smoke,
        method_simulators={
            name: simulator
            for name, simulator in smoke.method_simulators.items()
            if name != runs.NO_METHOD_SIMULATOR
        },
        base_quantifiers={
            name: quantifier
            for name, quantifier in smoke.base_quantifiers.items()
            if name != runs.BASELINE_QUANTIFIER
        },
    )


@pytest.fixture(scope="module")
def ablation_runs(ablation_spec):
    """The ablation spec's runs, computed once and shared by the tests below."""
    return run_measure_ablation(ablation_spec, measures=("topsoe", "sord"))


def test_running_the_ablation_produces_a_frame_the_results_module_recognises(
    ablation_runs, results_root
):
    assert tuple(ablation_runs.columns) == runs.columns_for(runs.MEASURE_ABLATION)
    runs.save(ablation_runs, runs.MEASURE_ABLATION, root=results_root)


def test_the_ablation_stamps_every_run_with_the_measure_that_produced_it(
    ablation_runs,
):
    assert set(ablation_runs["measure"]) == {"topsoe", "sord"}
    assert set(ablation_runs.groupby("measure").size()) == {len(ablation_runs) // 2}


def test_the_ablation_holds_everything_but_the_measure_fixed(ablation_runs):
    # Two measures on the same spec differ in exactly the one field
    # ``run_measure_ablation`` is meant to vary.
    without_measure = ablation_runs.drop(columns=["measure"])
    topsoe = without_measure[ablation_runs["measure"] == "topsoe"].reset_index(
        drop=True
    )
    sord = without_measure[ablation_runs["measure"] == "sord"].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        topsoe.drop(columns=["estimated_prevalence"]),
        sord.drop(columns=["estimated_prevalence"]),
    )


def test_the_ablation_reuses_run_sweep_s_missing_run_reporting(ablation_spec):
    spec = dataclasses.replace(
        ablation_spec, base_quantifiers={"DyS": FailingQuantifier}
    )

    with pytest.warns(sweep.MissingRunWarning):
        produced = run_measure_ablation(spec, measures=("topsoe",))

    assert produced["estimated_prevalence"].isna().all()


# ---------------------------------------------------------------------------
# Validating a produced sweep against ADR-0001's two defect signatures (#9)
# ---------------------------------------------------------------------------


def cell_row(**overrides):
    """A minimal synthetic run row, everything but ``overrides`` held fixed."""
    row = {
        "base_quantifier": "DyS",
        "method_simulator": runs.UNIFORM,
        "reference_simulator": runs.UNIFORM,
        "reference_merging_factor": 0.5,
        "bag_simulator": runs.UNIFORM,
        "bag_merging_factor": 0.5,
        "target_prevalence": 0.4,
        "true_prevalence": 0.4,
        "estimated_prevalence": 0.37,
        "repetition": 1,
    }
    row.update(overrides)
    return row


def synthetic_frame(rows):
    return pd.DataFrame(rows, columns=list(runs.columns_for(runs.SYNTHETIC)))


def test_a_run_tying_its_predecessor_is_a_stale_estimate_candidate():
    # The defect exactly: a different method's row carrying the number the one
    # before it computed.
    frame = synthetic_frame(
        [
            cell_row(base_quantifier="DyS", estimated_prevalence=0.37),
            cell_row(base_quantifier="HDy", estimated_prevalence=0.37),
        ]
    )

    stale = stale_estimate_rows(frame)

    assert len(stale) == 1
    assert stale.iloc[0]["base_quantifier"] == "HDy"


def test_a_tie_across_a_group_boundary_is_not_a_stale_estimate_candidate():
    # The last row of one (method simulator, cell, repetition) group and the
    # first row of the next describe different bags and different methods
    # entirely; a coincidence there says nothing about the defect, which is
    # one base quantifier's row inheriting the one *before it in its own
    # group*. Different repetitions here, so this is the same cell and method
    # simulator either side of the boundary — the closest a real tie could
    # get to one without being in the same group.
    frame = synthetic_frame(
        [
            cell_row(base_quantifier="DyS", repetition=1, estimated_prevalence=0.37),
            cell_row(base_quantifier="HDy", repetition=2, estimated_prevalence=0.37),
        ]
    )

    assert stale_estimate_rows(frame).empty


def test_a_tie_at_the_merging_extremes_is_a_genuine_tie():
    frame = synthetic_frame(
        [
            cell_row(base_quantifier="DyS", estimated_prevalence=1.0),
            cell_row(base_quantifier="HDy", estimated_prevalence=1.0),
        ]
    )

    assert stale_estimate_rows(frame).empty


def test_a_tie_between_two_threshold_policy_quantifiers_is_a_genuine_tie():
    # MS2 falls back to MS's own thresholds whenever none of its own clear the
    # reliability filter, and TAC and TX can select the same threshold from a
    # sparse candidate set — both real, not the stale-estimate defect.
    frame = synthetic_frame(
        [
            cell_row(base_quantifier="TAC", estimated_prevalence=0.42),
            cell_row(base_quantifier="TX", estimated_prevalence=0.42),
        ]
    )

    assert stale_estimate_rows(frame).empty


def test_two_missing_runs_in_a_row_are_not_a_stale_estimate_candidate():
    frame = synthetic_frame(
        [
            cell_row(base_quantifier="DyS", estimated_prevalence=None),
            cell_row(base_quantifier="HDy", estimated_prevalence=None),
        ]
    )

    assert stale_estimate_rows(frame).empty


def test_a_collapsed_group_of_base_quantifiers_is_flagged():
    # ADR-0001's first defect: every base quantifier under a meta-quantifier
    # computed the same thing because the meta-quantifier never consulted it.
    frame = synthetic_frame(
        [
            cell_row(base_quantifier="DyS", method_simulator=runs.MVN),
            cell_row(base_quantifier="SORD", method_simulator=runs.MVN),
        ]
    )

    collapsed = collapsed_method_simulator_groups(frame)

    assert len(collapsed) == 1


def test_a_real_spread_across_base_quantifiers_is_not_flagged():
    frame = synthetic_frame(
        [
            cell_row(
                base_quantifier="DyS", method_simulator=runs.MVN,
                estimated_prevalence=0.30,
            ),
            cell_row(
                base_quantifier="SORD", method_simulator=runs.MVN,
                estimated_prevalence=0.60,
            ),
        ]
    )

    assert collapsed_method_simulator_groups(frame).empty


def test_the_no_method_simulator_arm_is_exempt_from_the_collapse_check():
    # CC and a reference estimator never share a meta-quantifier or its
    # candidates, so nothing about them can collapse the way ADR-0001 records.
    frame = synthetic_frame(
        [
            cell_row(
                base_quantifier="DyS", method_simulator=runs.NO_METHOD_SIMULATOR,
                estimated_prevalence=0.40,
            ),
            cell_row(
                base_quantifier="CC", method_simulator=runs.NO_METHOD_SIMULATOR,
                estimated_prevalence=0.40,
            ),
        ]
    )

    assert collapsed_method_simulator_groups(frame).empty


def test_validate_raises_on_a_stale_estimate_candidate():
    frame = synthetic_frame(
        [
            cell_row(base_quantifier="DyS", estimated_prevalence=0.37),
            cell_row(base_quantifier="HDy", estimated_prevalence=0.37),
        ]
    )

    with pytest.raises(DefectSignatureError, match="stale-estimate"):
        validate(frame)


def test_validate_raises_on_a_collapsed_group():
    # Close but not identical, and spread across three rows rather than two
    # adjacent ones, so this trips the collapse check and not the stale-tie
    # check first — the two are not the same signature and this proves
    # ``validate`` can report either.
    frame = synthetic_frame(
        [
            cell_row(
                base_quantifier="DyS", method_simulator=runs.MVN,
                estimated_prevalence=0.400,
            ),
            cell_row(
                base_quantifier="HDy", method_simulator=runs.MVN,
                estimated_prevalence=0.401,
            ),
            cell_row(
                base_quantifier="SORD", method_simulator=runs.MVN,
                estimated_prevalence=0.402,
            ),
        ]
    )

    with pytest.raises(DefectSignatureError, match="collapsed-estimate"):
        validate(frame)


def test_validate_passes_the_smoke_sweep(one_cell):
    # The re-run's own regression: the corrected sweep's output, at the size
    # this suite can afford to check on every run. ``one_cell`` is
    # ``SMOKE_SWEEP``'s one cell, which is the whole of what ``run_sweep``
    # would produce for it.
    validate(one_cell)


# ---------------------------------------------------------------------------
# validate_and_save: validate, report, then save, in that order (#9)
# ---------------------------------------------------------------------------


def test_validate_and_save_refuses_to_save_a_defective_frame(results_root):
    frame = synthetic_frame(
        [
            cell_row(base_quantifier="DyS", estimated_prevalence=0.37),
            cell_row(base_quantifier="HDy", estimated_prevalence=0.37),
        ]
    )

    with pytest.raises(DefectSignatureError):
        sweep.validate_and_save(frame, runs.SYNTHETIC, root=results_root)

    # Proof of order, not just of the raise: if saving ran first, a file
    # would exist for a frame that never passed validation.
    with pytest.raises(FileNotFoundError):
        runs.load(runs.SYNTHETIC, root=results_root)


def test_validate_and_save_reports_how_many_runs_are_missing(capsys, results_root):
    frame = synthetic_frame(
        [
            cell_row(base_quantifier="DyS", estimated_prevalence=None),
            cell_row(
                base_quantifier="DyS",
                method_simulator=runs.NO_METHOD_SIMULATOR,
                estimated_prevalence=0.40,
            ),
        ]
    )

    sweep.validate_and_save(frame, runs.SYNTHETIC, root=results_root)

    assert "1 of 2 runs have no estimate" in capsys.readouterr().out


def test_validate_and_save_saves_a_clean_frame_through_the_results_module(
    results_root,
):
    frame = synthetic_frame(
        [cell_row(base_quantifier="DyS", estimated_prevalence=0.37)]
    )

    path = sweep.validate_and_save(frame, runs.SYNTHETIC, root=results_root)

    assert path == results_root / "runs" / "synthetic.parquet"
    assert len(runs.load(runs.SYNTHETIC, root=results_root)) == 1
