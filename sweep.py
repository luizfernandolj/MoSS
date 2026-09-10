"""The sweep: a spec in, runs out.

A *sweep* is the full cross-product of runs making up one experiment. This
module is the seam that produces it: :func:`run_sweep` takes a
:class:`SweepSpec` and returns runs. A spec is the whole of what it needs — the
grid of cells, the simulators and quantifiers to run in each, and how big a
reference score set and a bag are. Nothing here reads a module global, so a
smoke sweep and the published sweep are this code with different arguments,
which is what makes the sweep testable at all. It used to open with
``from variables import *``, and a smaller grid meant editing that file.

The grid is a sequence of cells rather than the four ranges it happens to be
built from, so a caller can hand-pick cells. The characterization grid does:
it runs matched simulator pairs only, which no cross-product describes.
"""

import dataclasses
import hashlib
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from mlquantify.counting import CC, MS, MS2, T50, TAC, TMAX, TX
from mlquantify.matching import DyS, HDy, SMM, SORD
from mlquantify.utils import get_prev_from_labels

import runs
from utils.meta_quantifier import QuaDaptOverCandidates, QuaDaptWithSimulator
from utils.simulators import (
    DirichletSimulator,
    MVNSimulator,
    ScoreSimulator,
    UniformSimulator,
)

# --- The reference score set -----------------------------------------------


@dataclass(frozen=True)
class ReferenceScoreSet:
    """The labelled score distribution a quantifier matches a bag against."""

    scores: np.ndarray
    labels: np.ndarray


# --- Estimators ------------------------------------------------------------


class Estimator(ABC):
    """One method's estimate of one bag's prevalence.

    A *method* is a base quantifier together with the method simulator, if any,
    of the meta-quantifier wrapping it. What a method reads decides how it is
    asked for an estimate — nothing, the real reference score set, its labels
    alone, or the real scores as one candidate among many — and the adapters
    below are the whole of what differs between them. The sweep holds an
    ``Estimator`` and does not know which it has.
    """

    @abstractmethod
    def estimate(self, bag_scores):
        """The bag's estimated prevalence, as the positive class's share."""


@dataclass(frozen=True)
class BaselineEstimator(Estimator):
    """A base quantifier that reads no reference score set at all.

    It classifies the bag's members and counts them, so it takes the bag and
    nothing else. That is why it appears once per bag rather than once per
    method simulator: a simulator it never consults is not part of the method.
    """

    quantifier: type

    def estimate(self, bag_scores):
        return _positive_share(self.quantifier().aggregate(bag_scores))


@dataclass(frozen=True)
class ReferenceEstimator(Estimator):
    """A base quantifier matching the bag against the real reference scores."""

    quantifier: type
    reference: ReferenceScoreSet

    def estimate(self, bag_scores):
        return _positive_share(
            self.quantifier().aggregate(
                bag_scores, self.reference.scores, self.reference.labels
            )
        )


@dataclass(frozen=True)
class MetaEstimator(Estimator):
    """A meta-quantifier choosing among candidate score sets it draws itself.

    It takes the whole reference score set for symmetry with the adapter above,
    but reads only its labels: the candidates are simulated, so the real
    reference scores are the one thing this method deliberately does not see.
    That substitution is the study's question (CONTEXT.md).

    The seed has no default, unlike everywhere else in this project that takes
    one. This is the draw ADR-0004 was written about — an unseeded one here is
    what let ADR-0001's defect pass for noise — so leaving it out has to be a
    thing someone wrote down, not a thing they forgot.
    """

    quantifier: type
    method_simulator: ScoreSimulator
    reference: ReferenceScoreSet
    random_state: Optional[int]

    def estimate(self, bag_scores):
        meta = QuaDaptWithSimulator(
            self.quantifier(), self.method_simulator, self.random_state
        )
        return _positive_share(meta.aggregate(bag_scores, self.reference.labels))


@dataclass(frozen=True)
class CandidateEstimator(Estimator):
    """A meta-quantifier choosing among candidates from every simulator at once.

    The one meta-quantifier arm that *is* handed the real reference scores.
    The adapter above withholds them on purpose — its method replaces them —
    but this one has them as a candidate beside the simulated ones, so "no
    simulated substitute matched this bag better than the real scores" is an
    answer it can give and the results can carry (ADR-0010).
    """

    quantifier: type
    method_simulators: Tuple[ScoreSimulator, ...]
    reference: ReferenceScoreSet
    random_state: Optional[int]

    def estimate(self, bag_scores):
        meta = QuaDaptOverCandidates(
            self.quantifier(), self.method_simulators, self.random_state
        )
        return _positive_share(
            meta.aggregate(bag_scores, self.reference.scores, self.reference.labels)
        )


def _positive_share(prevalences):
    """The positive class's share of a binary estimate.

    ``prevalences`` is what a quantifier's ``aggregate`` returns: one value per
    class, in class order. Runs record a binary prevalence as this scalar
    (ADR-0003); a multiclass run will carry the vector itself, which is #14.
    """
    return prevalences[1]


def _observed_prevalence(labels):
    """The prevalence of the bag that was actually drawn.

    Not the same as the cell's target prevalence once a bag cannot be filled at
    the prevalence asked for — which cannot happen here, but does on a real
    dataset (ADR-0005), so both are recorded.

    Deliberately not routed through :func:`_positive_share`, which indexes a
    sequence: ``get_prev_from_labels`` returns a *mapping* from class label to
    share, so ``[1]`` there would be a lookup of the label ``1`` that only
    happens to agree while the simulators label their classes 0 and 1.
    """
    by_class = get_prev_from_labels(labels)
    _, positive = sorted(by_class.items())[1]
    return positive


# --- The grid --------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One point of the grid: how the reference score set and the bags are made.

    A cell says nothing about the estimator. Which methods run in it is the
    spec's business, so the same cell describes a plain run and a
    meta-quantifier run alike.
    """

    reference_simulator: str
    reference_merging_factor: float
    bag_simulator: str
    bag_merging_factor: float
    target_prevalence: float


def grid(simulators, merging_factors, target_prevalences):
    """Every cell of the cross-product of these ranges.

    The reference score set and the bags vary independently in both simulator
    and merging factor: whether a quantifier survives a bag drawn from a
    different distribution than its reference is the study's question, so the
    off-diagonal cells are the point rather than filler.
    """
    return tuple(
        Cell(
            reference_simulator=reference_simulator,
            reference_merging_factor=reference_merging_factor,
            bag_simulator=bag_simulator,
            bag_merging_factor=bag_merging_factor,
            target_prevalence=target_prevalence,
        )
        for reference_simulator in simulators
        for bag_simulator in simulators
        for reference_merging_factor in merging_factors
        for bag_merging_factor in merging_factors
        for target_prevalence in target_prevalences
    )


# --- The spec --------------------------------------------------------------


@dataclass(frozen=True)
class SweepSpec:
    """One experiment, entire. Everything :func:`sweep` needs and nothing else.

    Every registry is keyed by the name the runs module stores, not by a
    spelling of its own that would need translating on the way out. A key it
    does not recognise is rejected here rather than at ``runs.save``, so a
    sweep cannot spend hours producing runs no reader can name.
    """

    #: The grid. A sequence of cells, usually from :func:`grid`.
    cells: Sequence[Cell]

    #: Score simulators in their data role: the source of this sweep's own
    #: scores, both the reference set and the bags.
    data_simulators: Mapping[str, ScoreSimulator]

    #: Score simulators in their method role: one simulator per arm, or a
    #: sequence of them for an arm whose meta-quantifier chooses among all
    #: their candidates, or ``None`` for the arm that uses no meta-quantifier
    #: at all. What each shape means to an estimator is ``estimator_for``'s
    #: business and nothing else's.
    method_simulators: Mapping[
        str, Union[None, ScoreSimulator, Tuple[ScoreSimulator, ...]]
    ]

    #: The base quantifiers, by the name their runs are stored under.
    base_quantifiers: Mapping[str, type]

    reference_size: int
    bag_size: int
    repetitions: int

    #: The reference score set is drawn balanced, so that a quantifier's
    #: reference is never itself skewed towards the answer. A field rather than
    #: a module constant even though nothing varies it: a spec that left one
    #: number of the experiment outside itself would not be the whole
    #: experiment, which is the property this type exists to have.
    reference_prevalence: float = 0.5

    #: The root every draw in the sweep is seeded from, cell by cell and
    #: repetition by repetition (:func:`seed_for`) — the experiment's own
    #: simulators and the method simulators inside its meta-quantifiers alike.
    #: Two runs of a spec that carries one produce identical runs.
    #:
    #: ``None`` draws from OS entropy instead, which makes a sweep
    #: irreproducible: it is what a caller asks for deliberately, not a
    #: default anything published should keep.
    seed: Optional[int] = None

    def __post_init__(self):
        runs.reject_unknown(
            "base quantifier", self.base_quantifiers, runs.BASE_QUANTIFIERS
        )
        runs.reject_unknown("data simulator", self.data_simulators, runs.SIMULATORS)
        runs.reject_unknown(
            "method simulator", self.method_simulators, runs.METHOD_SIMULATORS
        )
        for cell in self.cells:
            runs.reject_unknown(
                "cell simulator",
                (cell.reference_simulator, cell.bag_simulator),
                tuple(self.data_simulators),
            )
        self._reject_a_baseline_with_nowhere_to_go()

    def _reject_a_baseline_with_nowhere_to_go(self):
        """The baseline quantifier needs the no-method-simulator arm to run in.

        It reads no reference score set, so :func:`estimator_for` gives it no
        estimator under any other arm — and a spec that registers it without
        that arm would drop it from every cell in silence, which is the same
        quiet mislabelling the rest of this seam exists to end.
        """
        if (
            runs.BASELINE_QUANTIFIER in self.base_quantifiers
            and runs.NO_METHOD_SIMULATOR not in self.method_simulators
        ):
            raise ValueError(
                f"the baseline quantifier {runs.BASELINE_QUANTIFIER!r} reads no "
                f"reference score set, so it only runs under the "
                f"{runs.NO_METHOD_SIMULATOR!r} method simulator, which this spec "
                "does not include"
            )


# --- Seeding ---------------------------------------------------------------

#: The three draws a cell makes, named apart so that no two of them can derive
#: the same seed. The reference set is drawn once for the whole cell; the bag
#: and the meta-quantifier's candidate score sets are drawn once per repetition.
REFERENCE_DRAW = "reference"
BAG_DRAW = "bag"
CANDIDATE_DRAW = "candidates"

#: The repetition a cell-wide draw is attributed to. Repetitions are numbered
#: from one, so nothing else can claim it.
BEFORE_ANY_REPETITION = 0


def seed_for(spec, cell, repetition, draw):
    """The seed for one draw: which cell, which repetition, which of the three.

    Two runs of the same spec derive the same seeds, and two draws that should
    differ derive different ones. The spec used to hand its one ``seed`` to
    every cell, so the grid drew the same reference score set in every cell
    that shared a simulator and a merging factor — 22,743 cells resampling far
    fewer distinct draws, and a sweep whose spread understated the sampling
    variation it was measuring.

    ``None`` in, ``None`` out. A spec with no seed draws from OS entropy, and
    deriving a seed from ``None`` would take that option away.

    Hashed rather than composed by arithmetic because a cell is strings and
    floats, and with blake2b rather than ``hash`` because Python salts string
    hashing per process. Cells cross into loky workers on the published path,
    where a salted hash would seed each worker differently and lose the one
    property this function exists to have.

    Deliberately *not* a function of the base quantifier. Every base quantifier
    in a cell sees the same bag and the same simulated candidates, so the only
    thing varying between their estimates is the quantifier itself — which is
    what makes identical estimates across base quantifiers readable as the
    wiring alarm ADR-0004 asks for rather than as a coincidence of the draw.
    """
    if spec.seed is None:
        return None

    key = (spec.seed, dataclasses.astuple(cell), repetition, draw)
    return int.from_bytes(
        hashlib.blake2b(repr(key).encode(), digest_size=8).digest(), "big"
    )


# --- Running a cell --------------------------------------------------------


def estimator_for(spec, base_quantifier, method_simulator, reference, candidate_seed):
    """The estimator for one (base quantifier, method simulator) pair.

    ``None`` means the pair names no method. The baseline quantifier reads no
    reference score set, so it has no meta-quantifier arm: pairing it with a
    method simulator would file the same number again under a simulator that
    took no part in producing it.

    ``candidate_seed`` is the one both meta-quantifier arms draw their
    candidate score sets from — :func:`seed_for` with :data:`CANDIDATE_DRAW`,
    which the caller has already derived because it belongs to the repetition
    rather than to this pair. The other two adapters draw nothing and ignore
    it.

    Required, with no default, for the reason :class:`MetaEstimator` gives for
    its own seed: a caller who leaves it out gets a meta-quantifier drawing
    from OS entropy, which is the failure this whole change is about, and it
    would go unremarked. ``None`` is still accepted — it is how a spec that
    carries no seed reaches here — but it has to be passed.
    """
    quantifier = spec.base_quantifiers[base_quantifier]

    if base_quantifier == runs.BASELINE_QUANTIFIER:
        if method_simulator != runs.NO_METHOD_SIMULATOR:
            return None
        return BaselineEstimator(quantifier)

    if method_simulator == runs.NO_METHOD_SIMULATOR:
        return ReferenceEstimator(quantifier, reference)

    if method_simulator == runs.ALL_SIMULATORS:
        return CandidateEstimator(
            quantifier,
            spec.method_simulators[method_simulator],
            reference,
            candidate_seed,
        )

    return MetaEstimator(
        quantifier, spec.method_simulators[method_simulator], reference, candidate_seed
    )


def run_cell(cell, spec):
    """Run one cell of the grid and return its runs.

    One reference score set, drawn once and shared by every method — the point
    of the experiment is which method reads it best, so drawing a fresh one per
    method would vary the wrong thing. Then one bag per repetition, and one run
    per method on each.

    Each of those draws is seeded from where it happens rather than from one
    generator threaded through the cell (:func:`seed_for`). A single stream
    made every draw depend on every draw before it: the bags moved when the
    reference score set changed size, and a cell's numbers depended on how many
    repetitions had already run.
    """
    reference = _draw_reference(cell, spec)

    rows = []
    for repetition in range(1, spec.repetitions + 1):
        bag_scores, bag_labels = spec.data_simulators[cell.bag_simulator](
            n=spec.bag_size,
            alpha=cell.target_prevalence,
            merging_factor=cell.bag_merging_factor,
            random_state=seed_for(spec, cell, repetition, BAG_DRAW),
        )
        true_prevalence = _observed_prevalence(bag_labels)
        candidate_seed = seed_for(spec, cell, repetition, CANDIDATE_DRAW)

        for method_simulator in spec.method_simulators:
            for base_quantifier in spec.base_quantifiers:
                estimator = estimator_for(
                    spec, base_quantifier, method_simulator, reference, candidate_seed
                )
                if estimator is None:
                    continue

                rows.append(
                    {
                        "base_quantifier": base_quantifier,
                        "method_simulator": method_simulator,
                        "reference_simulator": cell.reference_simulator,
                        "reference_merging_factor": cell.reference_merging_factor,
                        "bag_simulator": cell.bag_simulator,
                        "bag_merging_factor": cell.bag_merging_factor,
                        "target_prevalence": cell.target_prevalence,
                        "true_prevalence": true_prevalence,
                        "estimated_prevalence": _estimate_or_missing(
                            estimator, bag_scores
                        ),
                        "repetition": repetition,
                    }
                )

    return _frame(rows)


def _estimate_or_missing(estimator, bag_scores):
    """The estimate, or ``None`` if this method could not produce one.

    A failure is recorded rather than raised, because a sweep that stops at the
    first quantifier to dislike a bag never finishes; and recorded rather than
    dropped, because which method failed on which bag is itself a result
    (ADR-0005). What it must never be is a *number*: ADR-0001's second defect
    was a caught exception falling through to a row that then carried the
    previous method's estimate under this method's name.

    Reporting the failure is :func:`run_sweep`'s job, not this one's — see
    :func:`_warn_about_missing_runs` for why it cannot happen here.
    """
    try:
        return estimator.estimate(bag_scores)
    except Exception:
        return None


# --- Running the sweep -----------------------------------------------------


class MissingRunWarning(UserWarning):
    """A method could not estimate some of its bags; those runs have no estimate."""


def run_sweep(spec, n_jobs=1, progress=False):
    """Run every cell of the spec's grid and return the runs as one frame.

    Cells are independent, so ``n_jobs`` spreads them over processes. It
    defaults to one: a smoke sweep is smaller than the cost of starting
    workers, and only the published sweep wants the whole machine.
    """
    frames = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
        delayed(run_cell)(cell, spec) for cell in spec.cells
    )
    if progress:
        frames = tqdm(frames, total=len(spec.cells), desc="sweep", colour="blue")

    frames = list(frames)
    produced = pd.concat(frames, ignore_index=True) if frames else _frame([])
    _warn_about_missing_runs(produced)
    return produced


def _warn_about_missing_runs(produced):
    """Say, once per method, how many of its runs came back without an estimate.

    Raised here rather than where the failure happens, for two reasons that
    both rule the obvious placement out. A warning raised inside a cell never
    reaches a caller: cells run in loky workers, whose warnings land on the
    worker's stderr and never enter the parent's warning machinery, so the
    published path — ``n_jobs=-1`` — was the one place nothing could be caught.
    And a warning per failure is a warning per run: a method broken in every
    cell would have emitted millions of identical lines on the full grid.

    Counting the missing runs off the finished frame fixes both. What it costs
    is the exception itself, which does not survive the worker boundary. That
    is affordable only because the estimator seam makes one method on one bag
    reproducible on its own: the run names the cell and the method, and
    ``estimator_for(spec, ...).estimate(bag_scores)`` raises for anyone who
    wants the traceback.
    """
    missing = produced[produced["estimated_prevalence"].isna()]
    if missing.empty:
        return

    for method, count in missing.groupby(list(runs.ESTIMATOR_COLUMNS)).size().items():
        warnings.warn(
            f"{runs.method_label(*method)} produced no estimate for {count} of "
            "its runs; they are recorded as missing runs",
            MissingRunWarning,
            stacklevel=3,
        )


def _draw_reference(cell, spec):
    """The reference score set every method in this cell is given.

    Drawn once for the whole cell, so it is seeded from the cell and not from
    any repetition in it (:data:`BEFORE_ANY_REPETITION`).
    """
    scores, labels = spec.data_simulators[cell.reference_simulator](
        n=spec.reference_size,
        alpha=spec.reference_prevalence,
        merging_factor=cell.reference_merging_factor,
        random_state=seed_for(spec, cell, BEFORE_ANY_REPETITION, REFERENCE_DRAW),
    )
    return ReferenceScoreSet(scores, labels)


def _frame(rows):
    """Rows as a frame in the results module's column order, even when empty."""
    return pd.DataFrame(rows, columns=list(runs.columns_for(runs.SYNTHETIC)))


# --- The registries --------------------------------------------------------

#: The three score simulators, in their data role. The same three appear below
#: in their method role: they are one set of generators used for two entirely
#: different purposes (CONTEXT.md), and naming the roles apart is the point.
DATA_SIMULATORS = {
    runs.UNIFORM: UniformSimulator(),
    runs.MVN: MVNSimulator(),
    runs.DIRICHLET: DirichletSimulator(),
}

#: What a meta-quantifier draws its candidate score sets with. Three arms name
#: one simulator each; ``all`` names every one of them, because its
#: meta-quantifier chooses among all their candidates and the real reference
#: besides (ADR-0010); and ``none`` names no simulator because it uses no
#: meta-quantifier. ``estimator_for`` is where those three shapes are read.
METHOD_SIMULATORS = {
    **DATA_SIMULATORS,
    runs.ALL_SIMULATORS: tuple(DATA_SIMULATORS.values()),
    runs.NO_METHOD_SIMULATOR: None,
}

BASE_QUANTIFIERS = {
    "DyS": DyS,
    "HDy": HDy,
    "SORD": SORD,
    "SMM": SMM,
    "TAC": TAC,
    "TX": TX,
    "T50": T50,
    "TMAX": TMAX,
    "TMS": MS,
    "TMS2": MS2,
    runs.BASELINE_QUANTIFIER: CC,
}

#: Rounded, because these are stored and grouped on. ``np.arange`` returns
#: 0.15000000000000002 for the third step, and a renderer grouping by merging
#: factor would take that for a category of its own.
MERGING_FACTORS = tuple(round(float(m), 2) for m in np.arange(0.05, 1.0, 0.05))

#: Bracketing the prevalence range, closer together at the extremes where a
#: quantifier's error is largest.
TARGET_PREVALENCES = (0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 0.99)

# --- The specs -------------------------------------------------------------

#: The published synthetic experiment (CONTEXT.md): both the reference and the
#: bags come from data simulators, so score-distribution difficulty is a
#: controlled variable.
SYNTHETIC_SWEEP = SweepSpec(
    cells=grid(tuple(DATA_SIMULATORS), MERGING_FACTORS, TARGET_PREVALENCES),
    data_simulators=DATA_SIMULATORS,
    method_simulators=METHOD_SIMULATORS,
    base_quantifiers=BASE_QUANTIFIERS,
    reference_size=2000,
    bag_size=100,
    repetitions=3,
    #: The published sweep is seeded, so that the runs behind a figure can be
    #: reproduced from the spec that made them rather than only from the file
    #: they were written to. Any value would do; this one is the date the
    #: seeding landed.
    seed=20260910,
)

#: The same sweep, small enough to run in seconds. Every method and every base
#: quantifier still appears, so it exercises all four estimator arms; only the
#: grid and the sample sizes shrink.
SMOKE_SWEEP = SweepSpec(
    cells=grid((runs.UNIFORM,), (0.5,), (0.4,)),
    data_simulators=DATA_SIMULATORS,
    method_simulators=METHOD_SIMULATORS,
    base_quantifiers=BASE_QUANTIFIERS,
    reference_size=400,
    bag_size=100,
    repetitions=1,
    seed=20260909,
)


if __name__ == "__main__":
    print(
        runs.save(
            run_sweep(SYNTHETIC_SWEEP, n_jobs=-1, progress=True), runs.SYNTHETIC
        )
    )
