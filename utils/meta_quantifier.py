"""The two meta-quantifiers, each told which method simulators to use.

mlquantify hard-codes the uniform simulator as a ``MoSS`` classmethod on
``QuaDapt``, so the only way to give it a different one used to be a subclass
per simulator. Those three subclasses differed from one another in nothing but
how they reshaped the prevalence before delegating — reshaping the simulators
now do themselves. What is left is one class and one argument.

The second class searches wider: it chooses among candidate score sets drawn
by *every* simulator, and the real reference scores besides, instead of
committing to one. It replaces a class that did the same thing by rebinding
its own ``MoSS`` attribute mid-aggregation and re-implementing the library's
mixture search against an API that no longer exists (ADR-0010).
"""

#: The measures mlquantify searches with a histogram rather than on the raw
#: scores. Its own ``best_mixture`` branches on this same list; the copy that
#: used to live here branched on it twice.
HISTOGRAM_MEASURES = ("hellinger", "topsoe", "probsymm")

from dataclasses import dataclass

import numpy as np
from mlquantify.meta import QuaDapt
from mlquantify.utils import resolve_aggregate_classes, validate_prevalences


class QuaDaptWithSimulator(QuaDapt):
    """``QuaDapt`` whose method simulator is a constructor argument.

    The override exists only to route the library's ``MoSS`` seam to the given
    simulator. It passes the arguments straight through, because the simulators
    take the same ones the library does (ADR-0002) — including ``classes`` and
    ``random_state``, which the old subclasses accepted and dropped on the
    floor.
    """

    #: ``measure``, ``merging_factors`` and ``strategy`` are forwarded rather
    #: than restated, so that their defaults stay the library's. Restating them
    #: here would freeze a copy of upstream's choices that no test compares
    #: against the original — the trap ADR-0007 caught the last time this
    #: project held a copy of library internals.
    def __init__(
        self, quantifier, method_simulator, random_state=None, **quadapt_kwargs
    ):
        super().__init__(quantifier, **quadapt_kwargs)
        self.method_simulator = method_simulator
        self.random_state = random_state
        self._candidate_draws = np.random.default_rng(random_state)

    def aggregate(self, *args, **kwargs):
        """One estimate, one stream.

        The library calls :meth:`MoSS` several times per estimate — once per
        candidate merging factor, then once more at the one it picked — so the
        draws have to differ from one another while the estimate as a whole
        repeats. Restarting the stream here is what makes the second of those
        true: an estimate is reproducible from the estimator that made it,
        however many estimates that estimator has already made.

        Forwarded blind rather than by name, for the reason the class comment
        gives: restating upstream's signature here would freeze a copy of it
        that no test compares against the original.
        """
        self._candidate_draws = np.random.default_rng(self.random_state)
        return super().aggregate(*args, **kwargs)

    def MoSS(self, n, alpha, merging_factor, classes=None, random_state=None):
        """Draw one candidate score set, seeded whether or not the caller says.

        mlquantify calls this seam with no ``random_state`` at all — its own
        ``MoSS`` accepts one and documents it as unused (ADR-0004) — so a
        simulator honouring only what it was passed would draw from OS entropy
        on every candidate, which is the nondeterminism that hid ADR-0001's
        defect. Falling back to this estimator's own stream closes that without
        waiting on the library: the seam is ours to override, and overriding it
        is already why this class exists.

        A ``random_state`` that *is* passed still wins, so the day upstream
        threads one through, it is the caller's seed that governs.
        """
        if random_state is None:
            random_state = self._candidate_draws
        return self.method_simulator(n, alpha, merging_factor, classes, random_state)


#: How large a candidate score set is drawn, and at what prevalence. Both are
#: what mlquantify's own ``aggregate`` asks its ``MoSS`` seam for. They are
#: restated rather than forwarded because upstream writes them as literals in
#: the middle of a method, with no parameter to forward — and a candidate built
#: differently from the library's would not be comparable with the arms that do
#: let the library build them. A copy of an upstream choice is the trap
#: ADR-0007 caught, so ``tests/test_candidate_search.py`` pins these two
#: against the library rather than trusting them.
CANDIDATE_SIZE = 1000
CANDIDATE_PREVALENCE = 0.5


@dataclass(frozen=True)
class CandidateScoreSet:
    """One reference score set among the several a meta-quantifier evaluates.

    The glossary keeps *candidate score set* and *reference score set* apart
    (CONTEXT.md) and so does this project: ``sweep.ReferenceScoreSet`` is the
    one real score set a cell hands every method, and a candidate is one of
    the several the arm below chooses between — of which the real reference is
    only the first.
    """

    scores: np.ndarray
    labels: np.ndarray


class QuaDaptOverCandidates(QuaDapt):
    """``QuaDapt`` choosing among candidate score sets it is given, not one grid.

    The other meta-quantifier arm commits to a single method simulator and
    lets the library search its merging grid. This one searches wider: the real
    reference scores, and every simulator it holds across that same grid. The
    class it replaces did this by rebinding its own ``MoSS`` attribute between
    simulators mid-aggregation, and re-implemented the library's mixture search
    to do it (ADR-0007). Both are gone: :meth:`candidates` returns a list and
    nothing about this object changes while it estimates, which is what makes
    it safe in a worker.

    **Reached through :meth:`aggregate` alone**, as the sweep's estimator seam
    reaches every method (ADR-0009). It subclasses ``QuaDapt`` for one reason:
    to reach the library's mixture search rather than hold a copy of it, which
    is the whole point of the rewrite. It does not inherit a working
    ``fit``/``predict``, because its ``aggregate`` takes the real reference
    scores where the library's takes training labels alone — the difference
    that makes this a different method — and it never sets the ``classes_``
    those inherited methods read. Call one of them and it will say so.
    """

    def __init__(
        self, quantifier, method_simulators, random_state=None, **quadapt_kwargs
    ):
        super().__init__(quantifier, **quadapt_kwargs)
        self.method_simulators = method_simulators
        self.random_state = random_state

    def aggregate(self, predictions, reference_scores, reference_labels, classes=None):
        """Estimate the bag's prevalence from the candidate that matches it best.

        Takes the real reference score set as well as its labels, where the
        library's own ``aggregate`` takes only the labels: the real scores are
        one of the candidates here rather than the thing being replaced.

        What the search picks is a *score set*, not a prevalence. The
        prevalence each candidate's mixture implies is found along the way and
        thrown away, because the base quantifier is what produces the estimate
        — the property whose absence let ADR-0001's defect stand.
        """
        classes = resolve_aggregate_classes(self, classes, reference_labels)
        candidates = self.candidates(reference_scores, reference_labels, classes)

        best = min(
            candidates,
            key=lambda candidate: self._distance_to(predictions, candidate, classes),
        )

        prevalences = self.quantifier.aggregate(predictions, best.scores, best.labels)
        return validate_prevalences(self, prevalences, classes)

    def candidates(self, reference_scores, reference_labels, classes=None):
        """Every score set this arm will choose between, the real one first.

        The real reference is a candidate in its own right — the question this
        arm asks is whether any simulated substitute matches a bag better than
        the scores the classifier actually produced, so an answer of "none of
        them" has to be reachable. The class this replaces said the same thing
        by prepending a zero to the merging grid and treating that first entry
        specially; a score set that was never simulated has no merging factor,
        so the list says it directly instead.

        A list, returned. That is the property the rewrite is for: the object
        is unchanged by the call, so an estimate does not depend on which
        estimate ran before it and a loky worker can hold one safely.

        One generator, opened here and advanced across the draws, is what that
        property buys: the candidates differ from one another, the estimate as
        a whole repeats, and no seeding state outlives the call. The other arm
        cannot do it this way — the library calls its ``MoSS`` seam, so the
        stream has to be reachable from an attribute (ADR-0011).
        """
        classes = resolve_aggregate_classes(self, classes, reference_labels)
        draws = np.random.default_rng(self.random_state)

        return [
            CandidateScoreSet(reference_scores, reference_labels),
            *(
                CandidateScoreSet(
                    *simulator(
                        n=CANDIDATE_SIZE,
                        alpha=CANDIDATE_PREVALENCE,
                        merging_factor=merging_factor,
                        classes=classes,
                        random_state=draws,
                    )
                )
                for simulator in self.method_simulators
                for merging_factor in self._merging_grid()
            ),
        ]

    def _merging_grid(self):
        """The merging factors to draw candidates at.

        Rounded, because upstream's own ``best_mixture`` rounds before it draws
        and this arm has to draw the same candidates the other arms do:
        unrounded, ``np.arange`` puts this arm at 0.30000000000000004 where
        they are at 0.3, which is a difference between methods that nothing in
        the study intends.
        """
        return np.atleast_1d(np.round(self.merging_factors, 2)).astype(float)

    def _distance_to(self, predictions, candidate, classes):
        """How far the bag is from the closest mixture of this candidate.

        mlquantify's mixture search, reached rather than copied: the class it
        replaces held a copy of 0.2.0's, which 0.5.1 then removed from under it
        (ADR-0007). The measure is dispatched here, once — the copy dispatched
        it twice, in two branches that had already begun to differ.

        Named measures only, as upstream has it. Falling back to the histogram
        search for anything unrecognised would answer a question nobody asked
        with a number nothing distinguishes from a real one, which is the shape
        of every defect ADR-0001 records.
        """
        bag = np.asarray(predictions, dtype=float)
        bag = bag[:, 1] if bag.ndim == 2 else bag.ravel()

        positive = candidate.scores[candidate.labels == classes[1]][:, 1]
        negative = candidate.scores[candidate.labels == classes[0]][:, 1]

        if self.measure in HISTOGRAM_MEASURES:
            _, distance = self._histogram_best_mixture(
                bag, positive, negative, self.measure
            )
        elif self.measure == "sord":
            _, distance = self._sord_best_mixture(bag, positive, negative)
        else:
            raise ValueError(
                f"unknown measure {self.measure!r}; expected one of "
                f"{HISTOGRAM_MEASURES + ('sord',)}"
            )
        return distance
