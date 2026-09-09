"""The contract all three score simulators meet, asserted on all three.

Before this interface existed each simulator had its own signature and its own
way of reading a prevalence, so every caller reshaped the argument itself —
including the three meta-quantifier subclasses, which differed from each other
only in that reshaping. Nothing checked that the reshaping agreed, and it did
not: at a prevalence of 0.8 the MVN and Dirichlet simulators returned 81
positives out of 100 rather than 80, because ``1 - 0.8`` is a hair under 0.2
and their count rule floored it. The counts test below is what that defect
would have run into.

Every test here runs against all three implementations, since the point of the
interface is that a caller need not know which one it holds.
"""

import numpy as np
import pytest

from utils.simulators import (
    DirichletSimulator,
    MVNSimulator,
    ScoreSimulator,
    UniformSimulator,
)
from variables import DATA_SIMULATORS, METHOD_SIMULATORS

#: Every implementation. The uniform one is binary by construction, so the
#: multiclass tests below run on the other two.
SIMULATORS = [UniformSimulator(), MVNSimulator(), DirichletSimulator()]
MULTICLASS_SIMULATORS = [MVNSimulator(), DirichletSimulator()]

all_simulators = pytest.mark.parametrize(
    "simulator", SIMULATORS, ids=lambda s: type(s).__name__
)
multiclass_simulators = pytest.mark.parametrize(
    "simulator", MULTICLASS_SIMULATORS, ids=lambda s: type(s).__name__
)


@all_simulators
def test_score_rows_sum_to_one(simulator):
    scores, _ = simulator(1000, [0.7, 0.3], 0.5, random_state=0)

    assert scores.sum(axis=1) == pytest.approx(1.0)


@multiclass_simulators
def test_score_rows_sum_to_one_with_more_than_two_classes(simulator):
    scores, _ = simulator(900, [0.2, 0.3, 0.5], 0.5, random_state=0)

    assert scores.shape == (900, 3)
    assert scores.sum(axis=1) == pytest.approx(1.0)


@all_simulators
@pytest.mark.parametrize(
    "positive, expected",
    [(0.1, [900, 100]), (0.25, [750, 250]), (0.5, [500, 500]),
     (0.8, [200, 800]), (0.99, [10, 990])],
)
def test_class_counts_match_the_requested_prevalence(simulator, positive, expected):
    # Every share here divides 1000 exactly, so the expected counts are written
    # out rather than recomputed: they are the arithmetic answer, not a
    # restatement of the rounding rule. 0.8 is the one the old rule missed.
    _, labels = simulator(1000, [1 - positive, positive], 0.5, random_state=0)

    assert len(labels) == 1000
    assert list(np.bincount(labels)) == expected


@multiclass_simulators
def test_class_counts_match_the_requested_prevalence_with_more_than_two_classes(simulator):
    _, labels = simulator(1000, [0.2, 0.3, 0.5], 0.5, random_state=0)

    assert list(np.bincount(labels)) == [200, 300, 500]


@all_simulators
def test_counts_that_do_not_divide_are_off_by_at_most_one(simulator):
    prevalence = [2 / 3, 1 / 3]

    _, labels = simulator(100, prevalence, 0.5, random_state=0)

    counts = np.bincount(labels)
    assert counts.sum() == 100
    assert np.abs(counts - np.array(prevalence) * 100).max() <= 1


@all_simulators
def test_a_scalar_prevalence_is_the_positive_class_share(simulator):
    # How mlquantify's meta-quantifier asks for a balanced reference set.
    _, from_scalar = simulator(1000, 0.3, 0.5, random_state=0)
    _, from_vector = simulator(1000, [0.7, 0.3], 0.5, random_state=0)

    assert np.array_equal(from_scalar, from_vector)


@all_simulators
def test_labels_are_the_class_labels_the_caller_asked_for(simulator):
    _, labels = simulator(1000, [0.7, 0.3], 0.5, classes=[4, 7], random_state=0)

    assert set(labels) == {4, 7}
    assert list(np.bincount(labels, minlength=8)[[4, 7]]) == [700, 300]


@all_simulators
def test_class_labels_disagreeing_with_the_prevalence_are_refused(simulator):
    # The old overrides took their class count from the prevalence and ignored
    # ``classes`` entirely, so a disagreement passed unnoticed.
    with pytest.raises(ValueError):
        simulator(1000, [0.2, 0.3, 0.5], 0.5, classes=[0, 1])


@all_simulators
def test_the_same_random_state_draws_the_same_scores(simulator):
    first, _ = simulator(1000, [0.7, 0.3], 0.5, random_state=7)
    again, _ = simulator(1000, [0.7, 0.3], 0.5, random_state=7)

    assert np.array_equal(first, again)


@all_simulators
def test_a_generator_is_drawn_from_and_advanced(simulator):
    # The sweep threads one generator through a whole grid cell, so consecutive
    # calls must not repeat themselves.
    rng = np.random.default_rng(7)

    first, _ = simulator(1000, [0.7, 0.3], 0.5, random_state=rng)
    again, _ = simulator(1000, [0.7, 0.3], 0.5, random_state=rng)

    assert not np.array_equal(first, again)


@all_simulators
def test_accepts_the_call_the_meta_quantifier_makes(simulator):
    # mlquantify 0.5.1 calls its ``MoSS`` seam with exactly these keywords
    # (ADR-0002). A simulator that cannot take them raises on contact.
    scores, labels = simulator(
        n=1000, alpha=0.5, merging_factor=0.4, classes=np.array([0, 1])
    )

    assert scores.shape == (1000, 2)
    assert labels.shape == (1000,)


def test_the_uniform_simulator_is_binary_only():
    with pytest.raises(ValueError):
        UniformSimulator()(900, [0.2, 0.3, 0.5], 0.5)


@pytest.mark.parametrize(
    "simulator",
    [s for s in {**DATA_SIMULATORS, **METHOD_SIMULATORS}.values() if s is not None],
    ids=lambda s: type(s).__name__,
)
def test_every_registered_simulator_is_reached_through_the_one_interface(simulator):
    # The registries used to hold bare functions on one side and meta-quantifier
    # subclasses on the other. Both now hold the same kind of thing.
    assert isinstance(simulator, ScoreSimulator)
