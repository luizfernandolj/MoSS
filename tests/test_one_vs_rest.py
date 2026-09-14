"""One-vs-rest decomposition, beyond the two classes every quantifier here is
built for (#14).

Both meta-quantifiers in ``utils/meta_quantifier.py`` reduce an n-class
estimate to n binary ones and recombine them (see that module's own
docstring for why they do it themselves rather than reaching for
mlquantify's fit-based OvR). What is under test here is the reduction itself
— that each class gets its own fair binary sub-problem and that the shares
recombine into one prevalence vector — not the accuracy of any one base
quantifier, which the rest of the suite already covers on two classes.
"""

import numpy as np
import pytest
from mlquantify.matching import DyS

from utils.meta_quantifier import (
    CandidateScoreSet,
    QuaDaptOverCandidates,
    QuaDaptWithSimulator,
    combine_one_vs_rest,
    one_vs_rest_labels,
    one_vs_rest_view,
)
from utils.simulators import DirichletSimulator, MVNSimulator

#: Non-contiguous and not zero-based, so a class used as an array index (or
#: the reverse) cannot come out right by accident (tests/test_candidate_
#: search.py keeps the same discipline for the candidate-search arm).
CLASSES = (2, 5, 9)


# ---------------------------------------------------------------------------
# combine_one_vs_rest: n independent binary answers back into one prevalence
# ---------------------------------------------------------------------------


def test_shares_that_already_sum_to_one_are_returned_unchanged():
    np.testing.assert_allclose(combine_one_vs_rest([0.2, 0.3, 0.5]), [0.2, 0.3, 0.5])


def test_shares_that_do_not_sum_to_one_are_renormalised():
    # Three independent binary sub-problems each confidently answering 0.7
    # does not mean seven-tenths of the bag is three classes over.
    np.testing.assert_allclose(combine_one_vs_rest([0.7, 0.7, 0.7]), [1 / 3, 1 / 3, 1 / 3])


def test_shares_that_are_all_zero_fall_back_to_uniform():
    np.testing.assert_allclose(combine_one_vs_rest([0.0, 0.0, 0.0]), [1 / 3, 1 / 3, 1 / 3])


# ---------------------------------------------------------------------------
# The binary views a class's own sub-problem is built from
# ---------------------------------------------------------------------------


def test_one_vs_rest_view_is_the_class_column_against_its_complement():
    scores = np.array([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])

    np.testing.assert_allclose(one_vs_rest_view(scores, 1), [[0.5, 0.5], [0.9, 0.1]])


def test_one_vs_rest_labels_are_one_for_the_class_and_zero_for_the_rest():
    labels = np.array([2, 4, 4, 2, 7])

    assert list(one_vs_rest_labels(labels, 4)) == [0, 1, 1, 0, 0]


# ---------------------------------------------------------------------------
# QuaDaptWithSimulator, beyond two classes
# ---------------------------------------------------------------------------


@pytest.fixture
def well_separated_multiclass():
    """A reference and a bag, three well-separated classes, a known bag mix.

    Merging factor 0.05: close to the MVN simulator's own floor (ADR-0008's
    ``EPS``), so the classes barely overlap and DyS run through this arm has
    a real chance of recovering the bag's own composition rather than only
    producing *some* vector that sums to one.
    """
    simulator = MVNSimulator()
    reference_scores, reference_labels = simulator(
        2000, [1 / 3, 1 / 3, 1 / 3], 0.05, classes=CLASSES, random_state=1
    )
    bag_scores, _ = simulator(300, [0.2, 0.3, 0.5], 0.05, classes=CLASSES, random_state=2)
    return simulator, reference_scores, reference_labels, bag_scores


def test_quadapt_with_simulator_recovers_a_bag_s_composition_beyond_two_classes(
    well_separated_multiclass,
):
    simulator, _, reference_labels, bag_scores = well_separated_multiclass

    prevalences = QuaDaptWithSimulator(DyS(), simulator, random_state=3).aggregate(
        bag_scores, reference_labels
    )

    assert prevalences.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(prevalences, [0.2, 0.3, 0.5], atol=0.08)


def test_quadapt_with_simulator_reports_the_classes_it_was_asked_about(
    well_separated_multiclass,
):
    simulator, _, reference_labels, bag_scores = well_separated_multiclass

    meta = QuaDaptWithSimulator(DyS(), simulator, random_state=3)
    meta.aggregate(bag_scores, reference_labels)

    assert list(meta.classes_) == sorted(CLASSES)


def test_quadapt_with_simulator_s_binary_path_is_unchanged():
    # Two classes take the single-search path this arm always had; nothing
    # about one-vs-rest should touch it.
    simulator = MVNSimulator()
    _, reference_labels = simulator(400, 0.5, 0.3, classes=(0, 1), random_state=4)
    bag_scores, _ = simulator(100, 0.4, 0.3, classes=(0, 1), random_state=5)

    prevalences = QuaDaptWithSimulator(DyS(), simulator, random_state=6).aggregate(
        bag_scores, reference_labels
    )

    assert prevalences.shape == (2,)
    assert prevalences.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# QuaDaptOverCandidates, beyond two classes
# ---------------------------------------------------------------------------


def test_quadapt_over_candidates_recovers_a_bag_s_composition_beyond_two_classes(
    well_separated_multiclass,
):
    simulator, reference_scores, reference_labels, bag_scores = well_separated_multiclass

    arm = QuaDaptOverCandidates(DyS(), (simulator, DirichletSimulator()), random_state=3)
    prevalences = arm.aggregate(bag_scores, reference_scores, reference_labels)

    assert prevalences.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(prevalences, [0.2, 0.3, 0.5], atol=0.08)


def test_quadapt_over_candidates_searches_once_per_class(well_separated_multiclass):
    # SpyQuantifier-style wiring check (tests/test_candidate_search.py): the
    # base quantifier's aggregate is what has to run once per class, not the
    # arm's own search machinery answering on its behalf.
    simulator, reference_scores, reference_labels, bag_scores = well_separated_multiclass

    class CountingQuantifier:
        calls = 0

        def aggregate(self, predictions, reference_scores, reference_labels):
            CountingQuantifier.calls += 1
            share = reference_labels.mean()
            return np.array([1 - share, share])

    arm = QuaDaptOverCandidates(CountingQuantifier(), (simulator,), random_state=3)
    arm.aggregate(bag_scores, reference_scores, reference_labels)

    assert CountingQuantifier.calls == 3


def test_quadapt_over_candidates_gives_each_class_its_own_real_reference_view(
    well_separated_multiclass,
):
    # The real reference is a candidate for every class's own binary
    # sub-problem, split the same way the module docstring describes.
    simulator, reference_scores, reference_labels, _ = well_separated_multiclass
    classes = sorted(CLASSES)

    arm = QuaDaptOverCandidates(DyS(), (simulator,), random_state=3)

    for i, cls in enumerate(classes):
        candidates = arm.candidates(
            one_vs_rest_view(reference_scores, i),
            one_vs_rest_labels(reference_labels, cls),
            classes=np.array([0, 1]),
        )
        real = candidates[0]
        assert set(np.unique(real.labels)) <= {0, 1}
        assert real.labels.sum() == (np.asarray(reference_labels) == cls).sum()


def test_quadapt_over_candidates_leaves_the_arm_as_it_was_beyond_two_classes(
    well_separated_multiclass,
):
    # tests/test_candidate_search.py asserts the same thing on two classes;
    # one-vs-rest decomposition must not reintroduce state between calls.
    simulator, reference_scores, reference_labels, bag_scores = well_separated_multiclass
    arm = QuaDaptOverCandidates(DyS(), (simulator,), random_state=3)
    before = dict(vars(arm))

    arm.aggregate(bag_scores, reference_scores, reference_labels)

    after = vars(arm)
    assert after.keys() == before.keys()
    assert all(after[name] is before[name] for name in before)


def test_quadapt_over_candidates_draws_different_candidates_for_different_classes(
    well_separated_multiclass,
):
    # A fresh stream per class would hand every class the literal same
    # simulated candidates (same n, same alpha, same merging grid) — the one
    # thing a shared, advancing stream (the module docstring) rules out.
    simulator, reference_scores, reference_labels, bag_scores = well_separated_multiclass
    draws_seen = []

    class RecordingArm(QuaDaptOverCandidates):
        def candidates(self, reference_scores, reference_labels, classes=None, random_state=None):
            built = super().candidates(
                reference_scores, reference_labels, classes, random_state=random_state
            )
            draws_seen.append(built[1].scores.copy())  # the first simulated candidate
            return built

    RecordingArm(DyS(), (simulator,), random_state=3).aggregate(
        bag_scores, reference_scores, reference_labels
    )

    assert len(draws_seen) == len(CLASSES)
    assert not np.array_equal(draws_seen[0], draws_seen[1])
    assert not np.array_equal(draws_seen[1], draws_seen[2])


def test_quadapt_over_candidates_s_binary_path_is_unchanged():
    simulator = MVNSimulator()
    reference_scores, reference_labels = simulator(400, 0.5, 0.3, classes=(0, 1), random_state=4)
    bag_scores, _ = simulator(100, 0.4, 0.3, classes=(0, 1), random_state=5)

    arm = QuaDaptOverCandidates(DyS(), (simulator,), random_state=6)
    prevalences = arm.aggregate(bag_scores, reference_scores, reference_labels)

    assert prevalences.shape == (2,)
    assert prevalences.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CandidateScoreSet keeps its shape through the decomposition
# ---------------------------------------------------------------------------


def test_candidates_beyond_two_classes_are_candidate_score_sets(well_separated_multiclass):
    simulator, reference_scores, reference_labels, _ = well_separated_multiclass
    arm = QuaDaptOverCandidates(DyS(), (simulator,), random_state=3)

    candidates = arm.candidates(
        one_vs_rest_view(reference_scores, 0),
        one_vs_rest_labels(reference_labels, sorted(CLASSES)[0]),
        classes=np.array([0, 1]),
    )

    assert all(isinstance(c, CandidateScoreSet) for c in candidates)
