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

Both classes are binary at their core — ``best_mixture``'s histogram and SORD
searches, and ``QuaDaptOverCandidates._distance_to``, all read a two-column
score matrix. Beyond two classes each decomposes one-vs-rest instead: every
class in turn plays the positive share of its own binary sub-problem against
every other class merged into "rest", and the per-class shares renormalise
into one prevalence vector (:func:`one_vs_rest_prevalences`). mlquantify
ships an OvR decomposition of its own (``mlquantify.multiclass.binary_
quantifier``), but it is fit-based: it populates one binary sub-quantifier
per class in ``fit`` and reads them back in ``aggregate``, which this
project's estimator seam has nothing to call — every quantifier here, base or
meta, is reached through ``aggregate`` alone, on scores a classifier already
produced (ADR-0009). Both classes below build the same n-way decomposition
without a ``fit`` step, going straight to a binary ``aggregate`` call per
class — the same shape ``sweep.ReferenceEstimator`` uses for a base
quantifier run through no meta-quantifier at all, which is why the
decomposition's shared loop (:func:`one_vs_rest_prevalences`) lives here
rather than being written out three times.
"""

#: The measures mlquantify searches with a histogram rather than on the raw
#: scores. Its own ``best_mixture`` branches on this same list; the copy that
#: used to live here branched on it twice.
HISTOGRAM_MEASURES = ("hellinger", "topsoe", "probsymm")

from dataclasses import dataclass

import numpy as np
from mlquantify.meta import QuaDapt
from mlquantify.utils import resolve_aggregate_classes, validate_prevalences

# --- One-vs-rest decomposition ----------------------------------------------
#
# Shared by both meta-quantifiers below, and by sweep.ReferenceEstimator: a
# base quantifier run with no meta-quantifier at all is exactly as binary as
# the ones wrapped here, and reads its own reference score set the same way.


def one_vs_rest_view(scores, class_index):
    """Column ``class_index`` of an n-class score matrix, as a binary view.

    A two-column ``[rest, class]`` matrix a quantifier built for two classes
    can read unchanged: everything that is not ``class_index`` collapses into
    "rest", the same reduction one-vs-rest always makes.
    """
    scores = np.asarray(scores, dtype=float)
    positive = scores[:, class_index]
    return np.column_stack((1 - positive, positive))


def one_vs_rest_labels(labels, cls):
    """``labels`` as 0/1: 1 where a row is ``cls``, 0 for every other class."""
    return (np.asarray(labels) == cls).astype(int)


def combine_one_vs_rest(shares):
    """Per-class one-vs-rest shares, renormalised into one prevalence vector.

    Each share comes from its own independent binary sub-problem, so nothing
    makes them sum to one on their own — a bag every sub-problem reads as
    mostly "rest" would have every share low. Renormalising is what turns n
    independent binary answers back into one n-class prevalence, the same way
    upstream's own OvR strategy does (``mlquantify.multiclass._aggregate_ovr``
    by construction, since it too feeds :func:`validate_prevalences` an
    unnormalised per-class dict).
    """
    shares = np.asarray(shares, dtype=float)
    total = shares.sum()
    if total <= 0:
        return np.full(len(shares), 1.0 / len(shares))
    return shares / total


def one_vs_rest_prevalences(classes, positive_share_for):
    """A prevalence vector from n one-vs-rest binary sub-problems, one per class.

    ``positive_share_for(class_index, cls)`` runs that class's own binary
    sub-problem and returns its positive share; called once per entry of
    ``classes``, in order, and the results combined (:func:`combine_one_vs_
    rest`). The one shape this project's three one-vs-rest decompositions
    share — the two meta-quantifiers below, and a plain base quantifier run
    through no meta-quantifier at all (``sweep.ReferenceEstimator``) — even
    though what each does *inside* a class's sub-problem differs: a meta-
    quantifier's own mixture search, a candidate search across simulators, or
    a bare ``aggregate`` call. That difference is exactly what
    ``positive_share_for`` closes over, so this function needs to know
    nothing about it.
    """
    shares = [positive_share_for(i, cls) for i, cls in enumerate(classes)]
    return combine_one_vs_rest(shares)


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

    def aggregate(self, predictions, y_train, classes=None):
        """One estimate, one stream.

        The library calls :meth:`MoSS` several times per estimate — once per
        candidate merging factor, then once more at the one it picked — so the
        draws have to differ from one another while the estimate as a whole
        repeats. Restarting the stream here is what makes the second of those
        true: an estimate is reproducible from the estimator that made it,
        however many estimates that estimator has already made.

        Named rather than forwarded blind (``*args, **kwargs``) as this method
        used to be. That was deliberate, guarding against the ADR-0007 trap:
        restating an upstream *value* here — a default, a literal — freezes a
        copy nothing compares against the original, and it can drift silently.
        Naming ``predictions``/``y_train``/``classes`` restates upstream's
        *signature* instead, which this override already committed to the
        moment it started forwarding ``super().aggregate(...)``'s two
        positional arguments (below, and in every call this project makes
        through ``sweep.MetaEstimator``) — the coupling is not new, only now
        visible in the ``def`` line. If upstream ever changes that signature,
        this raises a ``TypeError`` at the call site, not a value silently out
        of step with its source, which is the failure ADR-0007 exists to
        prevent. One-vs-rest decomposition (module docstring) is why the
        change was worth making: it needs ``classes`` resolved once, up front,
        to know whether there is any decomposing to do at all.
        """
        self._candidate_draws = np.random.default_rng(self.random_state)
        classes = resolve_aggregate_classes(self, classes, y_train)

        if len(classes) <= 2:
            return super().aggregate(predictions, y_train, classes=classes)

        predictions = np.asarray(predictions, dtype=float)
        prevalences = one_vs_rest_prevalences(
            classes,
            lambda i, cls: self._original_aggregate(
                one_vs_rest_view(predictions, i),
                one_vs_rest_labels(y_train, cls),
                classes=np.array([0, 1]),
            )[1],
        )
        self.classes_ = classes
        return validate_prevalences(self, prevalences, classes)

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

        Beyond two classes this is one-vs-rest decomposed (module docstring):
        one binary candidate search per class, each choosing independently
        among that class's own binary view of every candidate, recombined by
        :func:`one_vs_rest_prevalences`. One draw stream is opened here and
        threaded through every class's search — the same "one estimate, one
        stream" property :meth:`QuaDaptWithSimulator.aggregate` keeps — so
        that classes draw different candidates from each other rather than
        the identical ones a fresh stream per class would repeat.
        """
        classes = resolve_aggregate_classes(self, classes, reference_labels)

        if len(classes) <= 2:
            return self._aggregate_binary(
                predictions, reference_scores, reference_labels, classes
            )

        predictions = np.asarray(predictions, dtype=float)
        reference_labels = np.asarray(reference_labels)
        draws = np.random.default_rng(self.random_state)

        prevalences = one_vs_rest_prevalences(
            classes,
            lambda i, cls: self._aggregate_binary(
                one_vs_rest_view(predictions, i),
                one_vs_rest_view(reference_scores, i),
                one_vs_rest_labels(reference_labels, cls),
                np.array([0, 1]),
                random_state=draws,
            )[1],
        )
        return validate_prevalences(self, prevalences, classes)

    def _aggregate_binary(
        self, predictions, reference_scores, reference_labels, classes, random_state=None
    ):
        """The binary search :meth:`aggregate` ran before one-vs-rest existed.

        Its own method so that :meth:`aggregate` can call it once per class:
        every argument a caller varies per class is a parameter here, nothing
        is read off ``self`` but the quantifier and the simulators themselves.
        """
        candidates = self.candidates(
            reference_scores, reference_labels, classes, random_state=random_state
        )
        best = min(
            candidates,
            key=lambda candidate: self._distance_to(predictions, candidate, classes),
        )

        prevalences = self.quantifier.aggregate(predictions, best.scores, best.labels)
        return validate_prevalences(self, prevalences, classes)

    def candidates(self, reference_scores, reference_labels, classes=None, random_state=None):
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

        ``random_state`` overrides ``self.random_state`` when given — how
        :meth:`aggregate`'s one-vs-rest decomposition threads one stream
        through several calls to this method instead of every class reopening
        the same one (and so drawing the same candidates as every other
        class). A single call with no override, as every caller outside this
        class makes, is unaffected.
        """
        classes = resolve_aggregate_classes(self, classes, reference_labels)
        draws = np.random.default_rng(
            self.random_state if random_state is None else random_state
        )

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
