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

from mlquantify.counting import CC, MS, MS2, T50, TAC, TMAX, TX, ThresholdAdjustment
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

    ``measure`` defaults to ``None``, not to the library's own default, and is
    forwarded only when it is set (:func:`_quadapt_kwargs`). Writing ``"topsoe"``
    here would freeze a copy of upstream's choice that nothing compares against
    the original — the trap ADR-0007 caught the last time this project held one
    (ADR-0012).
    """

    quantifier: type
    method_simulator: ScoreSimulator
    reference: ReferenceScoreSet
    random_state: Optional[int]
    measure: Optional[str] = None

    def estimate(self, bag_scores):
        meta = QuaDaptWithSimulator(
            self.quantifier(),
            self.method_simulator,
            self.random_state,
            **_quadapt_kwargs(self.measure),
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

    ``measure`` is forwarded the same way :class:`MetaEstimator` forwards it —
    see there for why ``None`` is not the library's default spelled out here.
    """

    quantifier: type
    method_simulators: Tuple[ScoreSimulator, ...]
    reference: ReferenceScoreSet
    random_state: Optional[int]
    measure: Optional[str] = None

    def estimate(self, bag_scores):
        meta = QuaDaptOverCandidates(
            self.quantifier(),
            self.method_simulators,
            self.random_state,
            **_quadapt_kwargs(self.measure),
        )
        return _positive_share(
            meta.aggregate(bag_scores, self.reference.scores, self.reference.labels)
        )


def _quadapt_kwargs(measure):
    """The keyword arguments a meta-quantifier adapter forwards for ``measure``.

    Empty when ``measure`` is ``None``, so the meta-quantifier falls through to
    upstream's own default instead of this project restating it.
    """
    return {} if measure is None else {"measure": measure}


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

    #: The distance measure every meta-quantifier arm in this sweep minimises,
    #: fixed for the whole grid — comparing measures against each other is the
    #: ablation's job (:func:`run_measure_ablation`), not a dimension of this
    #: one. ``None`` leaves upstream's own default in place rather than this
    #: project restating it (:func:`_quadapt_kwargs`, ADR-0012).
    measure: Optional[str] = None

    def __post_init__(self):
        runs.reject_unknown(
            "base quantifier", self.base_quantifiers, runs.BASE_QUANTIFIERS
        )
        runs.reject_unknown("data simulator", self.data_simulators, runs.SIMULATORS)
        runs.reject_unknown(
            "method simulator", self.method_simulators, runs.METHOD_SIMULATORS
        )
        if self.measure is not None:
            runs.reject_unknown("measure", (self.measure,), runs.MEASURES)
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
            spec.measure,
        )

    return MetaEstimator(
        quantifier,
        spec.method_simulators[method_simulator],
        reference,
        candidate_seed,
        spec.measure,
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
    #: Ten, not three (#9): the re-run this spec exists to produce is the one
    #: whose output gets validated and published, and three repetitions is the
    #: number ADR-0001 voided every run drawn under.
    repetitions=10,
    #: The published sweep is seeded, so that the runs behind a figure can be
    #: reproduced from the spec that made them rather than only from the file
    #: they were written to. Any value would do; this one is the date the
    #: seeding landed.
    seed=20260910,
    #: Fixed at the library's own default (``measure=None`` forwards nothing,
    #: see :func:`_quadapt_kwargs`) for the whole grid. Whether another measure
    #: estimates better is :data:`MEASURE_ABLATION_SWEEP`'s question, asked on
    #: a grid this one does not have to share (ADR-0012).
    measure=None,
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


def run_measure_ablation(spec, measures=runs.MEASURES, n_jobs=1, progress=False):
    """Run ``spec`` once per measure and stack the runs into one frame.

    ``spec`` is a template: its own ``measure`` is overridden by each entry of
    ``measures`` in turn (:func:`dataclasses.replace`), so everything about the
    sweep but the measure is held fixed and a difference between the frames it
    produces is a difference the measure made. A caller passes a full-size
    spec for the published ablation and a small one in a test, the same
    relationship :func:`run_sweep` has to :data:`SYNTHETIC_SWEEP` and
    :data:`SMOKE_SWEEP`.

    Each measure's runs go through :func:`run_sweep` unchanged, so a method
    simulator broken under one measure is still reported as a missing run
    (:func:`_warn_about_missing_runs`) rather than as an exception. ``measure``
    is stamped on afterwards — the sweep that produces each frame never varies
    it internally, so there is nothing for ``run_cell`` to record row by row.
    """
    ablated = []
    for measure in measures:
        produced = run_sweep(
            dataclasses.replace(spec, measure=measure), n_jobs=n_jobs, progress=progress
        )
        produced = produced.copy()
        produced.insert(len(runs.ESTIMATOR_COLUMNS), "measure", measure)
        ablated.append(produced)

    columns = list(runs.columns_for(runs.MEASURE_ABLATION))
    if not ablated:
        return pd.DataFrame(columns=columns)
    return pd.concat(ablated, ignore_index=True)[columns]


#: The distance-measure ablation's grid (#9): every simulator once at three
#: merging factors and three prevalences, rather than the published grid's
#: nineteen and seven — reduced because this grid is about to run once per
#: entry in ``runs.MEASURES`` instead of once.
MEASURE_ABLATION_GRID = grid(tuple(DATA_SIMULATORS), (0.1, 0.5, 0.9), (0.2, 0.5, 0.8))

#: The published ablation's spec (#9): every arm and base quantifier that
#: reads ``measure`` (:func:`estimator_for`), on the reduced grid above.
#:
#: Excludes the no-method-simulator arm and the baseline quantifier: ``CC``
#: classifies and counts, and ``ReferenceEstimator`` matches the bag against
#: the real reference directly — neither consults a meta-quantifier, so
#: running either once per measure would only repeat the same rows under
#: every one of them, and the baseline would then have nowhere to run
#: (``_reject_a_baseline_with_nowhere_to_go``).
MEASURE_ABLATION_SWEEP = SweepSpec(
    cells=MEASURE_ABLATION_GRID,
    data_simulators=DATA_SIMULATORS,
    method_simulators={
        name: simulator
        for name, simulator in METHOD_SIMULATORS.items()
        if name != runs.NO_METHOD_SIMULATOR
    },
    base_quantifiers={
        name: quantifier
        for name, quantifier in BASE_QUANTIFIERS.items()
        if name != runs.BASELINE_QUANTIFIER
    },
    #: Smaller than the published grid's 2000: the ablation asks which measure
    #: wins, not how well any of them does in absolute terms, and that
    #: comparison holds at the smoke sweep's own reference size.
    reference_size=400,
    bag_size=100,
    repetitions=3,
    seed=20260910,
)


# --- Validating a produced sweep --------------------------------------------
#
# The two defect signatures ADR-0001 records: a caught exception whose row
# carried the *previous* method's estimate instead of its own, and a
# meta-quantifier that never consulted the base quantifier it wrapped, so
# every one of them computed the same thing. Both are now structurally absent
# from this seam's own code (``_estimate_or_missing`` never carries a stale
# value; the base quantifier is what produces every meta-quantifier estimate).
# What is checked here is the *output* of a produced sweep against both
# signatures anyway (#9) — before anything is plotted or written up, because a
# regression that reintroduced either defect should be caught there and not in
# a figure.


class DefectSignatureError(AssertionError):
    """A produced sweep matches one of ADR-0001's two recorded defects."""


#: Base quantifiers sharing threshold-selection machinery over the same
#: candidate set of ROC thresholds (``mlquantify.counting.ThresholdAdjustment``
#: — TAC, TX, T50, TMAX, MS and MS2 here). Two of them can legitimately land on
#: the same corrected estimate: MS2 falls back to MS's own threshold set
#: whenever no threshold clears its reliability filter (the "No cases satisfy
#: |TPR - FPR| > 0.25" warning), and on a small reference or bag the sparse set
#: of distinct thresholds is exactly where TAC's fixed point and TX's crossing
#: point coincide too. Observed on the smoke sweep: TMS/TMS2 and TAC/TX tie
#: three times in 51 rows, none of it the stale-estimate defect. A tie between
#: two of these is therefore a genuine tie, the same as one at 0.0 or 1.0.
THRESHOLD_POLICY_QUANTIFIERS = frozenset(
    name for name, quantifier in BASE_QUANTIFIERS.items()
    if issubclass(quantifier, ThresholdAdjustment)
)


#: How much the base quantifiers in one (method simulator, cell, repetition)
#: group must spread, in absolute prevalence, to count as real rather than as
#: the 0.2.0 collapse. Same order of magnitude as the smallest spread measured
#: on the smoke sweep under the fix (``tests/test_meta_quantifier.MIN_SPREAD``);
#: kept as its own constant because the two guard different seams.
MIN_BASE_QUANTIFIER_SPREAD = 0.01


def _group_columns(produced):
    """Columns identifying one (method simulator, cell, repetition) group.

    Everything but ``base_quantifier`` and ``estimated_prevalence`` — the two
    columns that vary *within* a group — rather than a list spelled by name, so
    both checks below read a measure-ablation frame's extra ``measure`` column
    the same way they read the published grid's.
    """
    return [
        column
        for column in produced.columns
        if column not in ("base_quantifier", "estimated_prevalence")
    ]


def stale_estimate_rows(produced):
    """Runs whose estimate exactly repeats the row stored immediately before
    them, within the same (method simulator, cell, repetition) group.

    ADR-0001's second defect exactly: a caught exception fell through to a row
    that then carried the *previous* method's number, one base quantifier
    after another inside a single cell and repetition (:func:`run_cell`'s own
    loop order). Scoped to the group for that reason: two rows either side of
    a group boundary describe different bags and different methods entirely,
    so a coincidence there would say nothing about this defect and comparing
    across it would only look for one.

    Within a group, two independent quantifiers landing on the same float by
    chance is vanishingly unlikely, except in two places genuine agreement is
    expected instead: where the bag leaves nothing to disagree about, every
    method answers exactly 0.0 or 1.0; and where both methods are
    :data:`THRESHOLD_POLICY_QUANTIFIERS`, sharing a threshold-selection policy
    over the same candidate thresholds is enough on its own to coincide (see
    there). Both are excluded as genuine ties, not the defect. Two runs that
    both have no estimate are excluded too — that is two independent missing
    runs, not one estimate inherited by the other.
    """
    ordered = produced.reset_index(drop=True)
    estimate = ordered["estimated_prevalence"]
    base_quantifier = ordered["base_quantifier"]
    by_group = ordered.groupby(_group_columns(ordered), dropna=False, sort=False)

    previous_estimate = by_group["estimated_prevalence"].shift()
    previous_base_quantifier = by_group["base_quantifier"].shift()

    # ``.notna()`` rather than relying on ``NaN != NaN``: a column of missing
    # estimates alone is ``object``-typed and holds Python ``None``, for which
    # ``eq`` disagrees with float ``NaN`` and would call two missing runs tied.
    tied = estimate.eq(previous_estimate) & estimate.notna()
    boundary_tie = tied & estimate.isin((0.0, 1.0))
    threshold_policy_tie = (
        tied
        & base_quantifier.isin(THRESHOLD_POLICY_QUANTIFIERS)
        & previous_base_quantifier.isin(THRESHOLD_POLICY_QUANTIFIERS)
    )

    return ordered[tied & ~boundary_tie & ~threshold_policy_tie]


def collapsed_method_simulator_groups(produced):
    """(method simulator, cell, repetition) groups with no real spread.

    ADR-0001's first defect exactly: the meta-quantifier returned its own
    mixture search's prevalence and never consulted the base quantifier it
    wrapped, so every base quantifier in a group computed the same thing. Under
    the fix, a group's rows differ in nothing but the base quantifier — same
    bag, same candidate score sets, because the seed is derived from the cell
    and the repetition and not from the quantifier (ADR-0011) — so a spread at
    or below :data:`MIN_BASE_QUANTIFIER_SPREAD` is the collapse, not noise.

    A group of one base quantifier is not evidence of anything: nothing else
    ran there to disagree with it, so its "spread" of zero is excluded rather
    than read as a collapse.
    """
    meta = produced[produced["method_simulator"] != runs.NO_METHOD_SIMULATOR]
    by_group = meta.groupby(_group_columns(meta), dropna=False)["estimated_prevalence"]
    spread = by_group.agg(lambda group: group.max() - group.min())
    return spread[(by_group.size() > 1) & (spread <= MIN_BASE_QUANTIFIER_SPREAD)]


def validate(produced):
    """Check a produced sweep against both of ADR-0001's defect signatures.

    Raises :class:`DefectSignatureError` naming which signature and how many
    rows or groups matched it, rather than returning a boolean: a caller with
    nothing to do about a defect but stop is better served by an exception it
    does not have to remember to check for.
    """
    stale = stale_estimate_rows(produced)
    if not stale.empty:
        raise DefectSignatureError(
            f"{len(stale)} run(s) exactly repeat the estimate of the run stored "
            "immediately before them, outside a genuine tie at 0.0 or 1.0 — "
            "ADR-0001's stale-estimate defect signature"
        )

    collapsed = collapsed_method_simulator_groups(produced)
    if not collapsed.empty:
        raise DefectSignatureError(
            f"{len(collapsed)} (method simulator, cell, repetition) group(s) "
            f"show no more than {MIN_BASE_QUANTIFIER_SPREAD} spread across "
            "their base quantifiers — ADR-0001's collapsed-estimate defect "
            "signature"
        )


def validate_and_save(produced, kind, root=runs.ROOT):
    """Validate, report missing runs, then save — the sequence #9 asks for.

    In this order and no other: a produced sweep is checked against both of
    ADR-0001's defect signatures *before* anything downstream — a plot, a
    saved file — can trust it, and how many of its runs came back with no
    estimate is printed independent of whether warnings are enabled. Both
    branches of :func:`_main` go through here rather than repeating the
    sequence once each, which is what let it drift out of step.
    """
    validate(produced)

    missing = int(produced["estimated_prevalence"].isna().sum())
    print(f"{missing} of {len(produced)} runs have no estimate")

    return runs.save(produced, kind, root=root)


def _main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ablation",
        action="store_true",
        help=(
            "run the distance-measure ablation on its reduced grid instead of "
            "the published sweep"
        ),
    )
    args = parser.parse_args()

    if args.ablation:
        produced = run_measure_ablation(
            MEASURE_ABLATION_SWEEP, n_jobs=-1, progress=True
        )
        kind = runs.MEASURE_ABLATION
    else:
        produced = run_sweep(SYNTHETIC_SWEEP, n_jobs=-1, progress=True)
        kind = runs.SYNTHETIC

    print(validate_and_save(produced, kind))


if __name__ == "__main__":
    _main()
