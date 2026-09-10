"""The meta-quantifier that chooses among candidate score sets.

The other meta-quantifier arms commit to one method simulator and let the
library search its merging grid. This one supplies the candidates itself: the
real reference scores, and every simulator across the grid. What the tests
below assert is the seam that makes that possible at all — a *list* of
candidates, produced without touching the object that produced it. The class
this arm replaces rebound its own ``MoSS`` attribute between candidates and
could not be run twice, let alone in a worker.
"""

import numpy as np
import pytest
from mlquantify.matching import DyS
from mlquantify.meta import QuaDapt

from tests.score_sets import UNMATCHABLE, FixedSimulator, scored
from utils.meta_quantifier import (
    CANDIDATE_PREVALENCE,
    CANDIDATE_SIZE,
    CandidateScoreSet,
    QuaDaptOverCandidates,
)
from utils.simulators import DirichletSimulator, MVNSimulator, UniformSimulator

#: The three simulators in their method role, as the sweep registers them.
SIMULATORS = (UniformSimulator(), MVNSimulator(), DirichletSimulator())

#: Two factors rather than the library's five: what is under test is that the
#: grid is crossed with the simulators, which two values state as well as five
#: and a great deal faster.
MERGING_FACTORS = (0.2, 0.6)


@pytest.fixture
def reference():
    """The real reference score set, balanced as the sweep draws it."""
    return scored(400, 0.5)


def arm(quantifier=None, simulators=SIMULATORS):
    """The arm under test, on a grid small enough to assert about."""
    return QuaDaptOverCandidates(
        quantifier if quantifier is not None else DyS(),
        simulators,
        merging_factors=MERGING_FACTORS,
    )


def test_a_candidate_is_the_size_and_shape_the_library_asks_its_own_seam_for():
    """The guard on the one copy of an upstream choice this arm holds.

    ``CANDIDATE_SIZE`` and ``CANDIDATE_PREVALENCE`` are literals in the middle
    of mlquantify's ``aggregate``, with no parameter to forward, so this arm
    restates them — and a restated upstream choice that no test compares
    against the original is the trap ADR-0007 caught. This is that comparison:
    it drives the library's own meta-quantifier and records what it asks its
    ``MoSS`` seam for.
    """
    asked = []

    class Recording(QuaDapt):
        def MoSS(self, n, alpha, merging_factor, classes=None, random_state=None):
            asked.append((n, alpha))
            return QuaDapt.MoSS(n, alpha, merging_factor, classes, random_state)

    bag_scores, _ = scored(100, 0.3)
    _, reference_labels = scored(400, 0.5)
    Recording(DyS()).aggregate(bag_scores, reference_labels)

    assert set(asked) == {(CANDIDATE_SIZE, CANDIDATE_PREVALENCE)}


def test_the_candidates_are_the_real_reference_and_every_simulator_on_the_grid(
    reference,
):
    reference_scores, reference_labels = reference

    candidates = arm().candidates(reference_scores, reference_labels)

    assert len(candidates) == 1 + len(SIMULATORS) * len(MERGING_FACTORS)


def test_the_real_reference_scores_are_among_the_candidates(reference):
    # At zero merging, in the sense that they are not simulated at all: the
    # arm's question is whether any simulated substitute beats them.
    reference_scores, reference_labels = reference

    candidates = arm().candidates(reference_scores, reference_labels)

    assert any(
        np.array_equal(candidate.scores, reference_scores)
        and np.array_equal(candidate.labels, reference_labels)
        for candidate in candidates
    )


# ---------------------------------------------------------------------------
# The base quantifier produces the estimate
# ---------------------------------------------------------------------------
#
# The class this arm replaces returned the prevalence its own mixture search
# found and never consulted the quantifier it wrapped — ADR-0001's first
# defect, under which every base quantifier computed the same number.


class SpyQuantifier:
    """A base quantifier that answers a known number and remembers its reference."""

    #: Nothing a mixture search would arrive at, so an estimate that matches it
    #: can only have come from here.
    ANSWER = (0.35, 0.65)

    def __init__(self):
        self.reference = None

    def aggregate(self, predictions, reference_scores, reference_labels):
        self.reference = CandidateScoreSet(reference_scores, reference_labels)
        return np.asarray(self.ANSWER, dtype=float)


@pytest.fixture
def bag():
    """A bag of confidently scored rows, 30 of the 100 positive."""
    scores, _ = scored(100, 0.3)
    return scores


def test_the_estimate_is_the_base_quantifier_s(bag, reference):
    spy = SpyQuantifier()

    prevalences = arm(quantifier=spy).aggregate(bag, *reference)

    assert prevalences[1] == pytest.approx(SpyQuantifier.ANSWER[1])


# ---------------------------------------------------------------------------
# Which candidate it hands over
# ---------------------------------------------------------------------------
#
# The two directions of the same search, each with one matchable candidate and
# one that cannot be matched at all. "Closest" is asserted through a candidate
# whose mixture reproduces the bag exactly against one whose mixtures are all
# the same distribution, rather than by recomputing the distances the way the
# code computes them — and the pair is what makes it a choice: a search that
# always returned the real reference passes one of these and fails the other.


def one_simulator(spy, simulator):
    """The arm with a single simulated candidate beside the real reference."""
    return QuaDaptOverCandidates(spy, (simulator,), merging_factors=(0.5,))


def test_a_simulated_candidate_is_handed_over_when_it_is_the_one_that_matches(bag):
    spy = SpyQuantifier()
    scores, labels = scored(400, 0.5, **UNMATCHABLE)

    one_simulator(spy, FixedSimulator(0.95, 0.05)).aggregate(bag, scores, labels)

    # The separated candidate mixes to the bag exactly at its 0.3; the real
    # reference cannot approach it, and is not what the base quantifier got.
    assert not np.array_equal(spy.reference.scores, scores)


def test_the_real_reference_is_handed_over_when_no_simulator_beats_it(bag, reference):
    # The arm's own question. If a simulated substitute always won, "the real
    # scores were better here" would not be an answer the results could carry.
    spy = SpyQuantifier()

    one_simulator(spy, FixedSimulator(**UNMATCHABLE)).aggregate(bag, *reference)

    assert np.array_equal(spy.reference.scores, reference[0])


# ---------------------------------------------------------------------------
# Nothing changes while it estimates
# ---------------------------------------------------------------------------


def test_estimating_leaves_the_arm_exactly_as_it_was(bag, reference):
    # What the rewrite is for. The class this replaces rebound its own ``MoSS``
    # attribute once per simulator and stored the split reference on itself
    # mid-aggregation, so an estimate depended on which estimate ran before it
    # — an assumption a sweep spread over loky workers cannot make.
    meta_quantifier = arm(quantifier=DyS())
    before = dict(vars(meta_quantifier))

    meta_quantifier.aggregate(bag, *reference)

    after = vars(meta_quantifier)
    assert after.keys() == before.keys()
    assert all(after[name] is before[name] for name in before)


# ---------------------------------------------------------------------------
# The classes are the data's
# ---------------------------------------------------------------------------
#
# The class this arm replaces split its reference with ``train_labels == 1``
# and built its answer as ``[1 - prevalence, prevalence]``. On a dataset labelled
# anything but 0 and 1 — which is most of them, and every real one the
# real-data experiment will reach (ADR-0005) — both halves of the reference
# came back empty and the two prevalences named no class in particular.

#: Labels that are neither 0 and 1 nor adjacent, so an index used as a label
#: (or the reverse) cannot come out right by accident.
CLASSES = (3, 7)


def test_the_candidates_carry_the_labels_the_reference_uses(bag):
    reference_scores, reference_labels = scored(400, 0.5, classes=CLASSES)

    candidates = arm().candidates(reference_scores, reference_labels)

    assert all(set(np.unique(c.labels)) == set(CLASSES) for c in candidates)


def test_it_estimates_on_a_reference_whose_labels_are_not_zero_and_one(bag):
    # End to end, and deliberately the direction in which the labels have to be
    # read correctly all the way through: the real reference cannot be matched,
    # so the estimate can only come from the simulated candidate — which has to
    # be drawn under these labels, split by them to be measured, and handed to
    # DyS carrying them. The bag holds 30 positives in 100, an answer known
    # without running anything.
    reference = scored(400, 0.5, **UNMATCHABLE, classes=CLASSES)

    prevalences = one_simulator(DyS(), FixedSimulator(0.95, 0.05)).aggregate(
        bag, *reference
    )

    assert prevalences[1] == pytest.approx(0.3, abs=0.05)
