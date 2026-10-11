"""The real-data pipeline: datasets, classifier, protocol (ADR-0005, #10).

The synthetic experiment's reference and bags both come from data simulators
(``sweep.py``); this one's come from an actual classifier and an actual
dataset instead. What differs is entirely upstream of an estimate: once a bag
of real scores exists, the four estimator adapters in ``sweep.py`` cannot tell
it from a simulated one, so this module reuses them rather than growing a
second copy.

A *pool* is the fixed collection of real instances a dataset's bags are drawn
from — the counterpart of a data simulator, except nothing here can draw more
of it than exists. It holds two disjoint frames (ADR-0013): a **reference
frame**, three fifths of the capped dataset, scored by ten-fold
cross-validation within itself, which is the reference score set every method
matches a bag against; and a **bag frame**, the other two fifths, scored by a
Random Forest fitted on the whole reference frame and drawn from by
:func:`draw_bag`.

Both halves of that are load-bearing and they answer different objections.
Scoring out-of-fold rather than refitting keeps the reference from being the
classifier's own training scores, which are overconfident (ADR-0005). Keeping
the frames disjoint keeps a bag's own instances out of the distribution the
bag is estimated against — which ADR-0005 never addressed, and which a single
frame doing both jobs got wrong by as much as half the reference on the
smaller pools.

Bags are drawn under the artificial-prevalence protocol, class by class,
without replacement first — the same reason as ever: sampling with
replacement from the start would let every prevalence be reached from any
pool, however small, which would hide a shortfall rather than report it. A
class that falls short is bootstrap-replicated instead, but only up to
:data:`REPLICATION_CAP` times its own available count (#18): a modestly
undersized pool still speaks, each real instance standing in for itself a
handful of times, while a pool asked to speak far past what it holds still
fails explicitly. Haberman has 81 minority instances, so a bag of 100 at 90%
positive is filled by replication (90 is under the 405-instance cap) while one
of 1000 at the same prevalence is not (900 is over it), and :func:`draw_bag`
says so with ``None`` rather than padding the shortfall past the cap.

Everything above is binary-or-multiclass alike (#14): a pool, a bag and a
cell are the same shapes either way, and the estimator adapters decompose
one-vs-rest beyond two classes (``utils/meta_quantifier.py``). What differs
for a multiclass dataset is the target-prevalence grid — a shared sequence of
scalars cannot describe a class count :func:`grid` was not built for — so
:func:`multiclass_grid` and :func:`build_multiclass_spec` stand beside
:func:`grid` and :func:`build_spec` rather than replacing them.
"""

import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm

import runs
import sweep
from mlquantify import datasets as mlquantify_datasets
from utils import local_datasets
from utils.simulators import ScoreSimulator, as_prevalence, class_counts

# --- The pool ----------------------------------------------------------------


@dataclass(frozen=True)
class Pool:
    """One dataset's two disjoint frames: what bags come from, and what they
    are matched against (ADR-0013).

    ``scores`` and ``labels`` are the **bag frame** — the instances
    :func:`draw_bag` samples, scored by a classifier that was fitted on the
    other frame entirely and so has never seen them.
    ``reference_scores`` and ``reference_labels`` are the **reference frame**,
    the labelled score distribution a quantifier matches a bag against, scored
    out-of-fold within itself.

    The two share no instance. Before ADR-0013 there was one frame doing both
    jobs, which put a bag's own instances inside the distribution it was being
    estimated against — negligible for a bag of 100 drawn from 50,000, and
    half the reference for a bag of 1000 drawn from 2,000. Out-of-fold scoring
    (ADR-0005) answers a different question, overconfidence, and never
    addressed this one.
    """

    dataset: str
    scores: np.ndarray
    labels: np.ndarray
    reference_scores: np.ndarray
    reference_labels: np.ndarray

    @property
    def classes(self):
        """The class space, read off the reference frame.

        The reference's classes and not the bag frame's, because this is what
        a quantifier's estimate will have one entry per: the grid's prevalence
        vectors are sized by it, :func:`draw_bag` divides a bag size by it, and
        ``sweep.observed_prevalence`` reports a share for each of them. A
        stratified split gives both frames the same classes anyway; taking
        them from the side the estimate is shaped by means a bag frame that
        somehow lost one produces a missing run rather than a vector of the
        wrong length.
        """
        return np.unique(self.reference_labels)

    @property
    def reference(self):
        return sweep.ReferenceScoreSet(self.reference_scores, self.reference_labels)


#: How many times a short class may be bootstrap-replicated past its own
#: available count before :func:`draw_bag` gives up and returns ``None``
#: (#18). Five, so a shortfall the pool can plausibly still speak to — each
#: real instance standing in for itself a handful of times — is filled,
#: while a shortfall that would have every instance stand in for dozens of
#: repeats is refused the same way an entirely absent pool is.
REPLICATION_CAP = 5


def draw_bag(pool, target_prevalence, bag_size, random_state):
    """The indices of one bag drawn from ``pool``, or ``None`` if it cannot be.

    Drawn class by class, at the counts
    :func:`utils.simulators.class_counts` divides ``bag_size`` into. Every
    class is sampled without replacement first — the ordinary case, and the
    only one before #18. A class whose available count falls short of what it
    needs is bootstrap-replicated instead: sampled with replacement, but only
    up to :data:`REPLICATION_CAP` times its own available count. Beyond that
    cap, the draw is refused entirely and every class's work is thrown away —
    an unbounded cap would let every prevalence be reached from any pool,
    however small, which would hide a shortfall ADR-0005 records rather than
    report it.
    """
    classes = pool.classes
    counts = class_counts(bag_size, as_prevalence(target_prevalence))
    if len(counts) != len(classes):
        raise ValueError(
            f"{pool.dataset!r} has {len(classes)} classes, not {len(counts)}"
        )

    rng = np.random.default_rng(random_state)
    drawn = []
    for cls, count in zip(classes, counts):
        available = np.flatnonzero(pool.labels == cls)
        if count <= len(available):
            drawn.append(rng.choice(available, size=count, replace=False))
        elif count <= REPLICATION_CAP * len(available):
            drawn.append(rng.choice(available, size=count, replace=True))
        else:
            return None

    indices = np.concatenate(drawn)
    rng.shuffle(indices)
    return indices


def bag_replication(indices):
    """How many of a drawn bag's instances repeat one already in it.

    Zero whenever every class was filled without replacement — the ordinary
    case :func:`draw_bag` still tries first. Positive only for a bag that
    bootstrap-replicated a short class within :data:`REPLICATION_CAP`:
    sampling more instances than a class has available, with replacement,
    guarantees at least one repeat by the pigeonhole principle, and a pool's
    classes never share a row, so counting repeats across the whole bag is
    exactly counting them within whichever class was replicated.
    """
    return int(len(indices) - len(np.unique(indices)))


# --- The grid ------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One point of the real-data grid: which dataset, at what prevalence.

    The counterpart of ``sweep.Cell`` without the axes a real dataset has no
    say over — there is no simulator or merging factor here, only which pool
    a bag comes from and the prevalence it is asked for.

    ``target_prevalence`` is a float for a binary dataset and a tuple for a
    multiclass one (#14, mirroring ``runs.py``'s own scalar-or-vector
    convention) — a tuple rather than a list or array so that ``Cell`` stays
    hashable, which the grid being a *set* of cells (below, and in the tests)
    depends on.
    """

    dataset: str
    target_prevalence: Union[float, Tuple[float, ...]]


def grid(datasets, target_prevalences):
    """Every cell of the cross-product: each dataset at every prevalence.

    One ``target_prevalences`` sequence shared by every dataset — right for
    binary datasets, which all describe the same one-number share of the same
    two classes. A multiclass dataset's prevalence is a vector sized to its
    own class count, which no cross-product over one shared sequence can
    describe; :func:`multiclass_grid` builds that grid instead.
    """
    return tuple(
        Cell(dataset=dataset, target_prevalence=target_prevalence)
        for dataset in datasets
        for target_prevalence in target_prevalences
    )


#: How many prevalence vectors :func:`multiclass_grid` draws per dataset.
#: Matches binary's own :data:`TARGET_PREVALENCES` in count, so a multiclass
#: dataset's grid costs about as much to run as a binary one's.
N_MULTICLASS_PREVALENCES = 21


def multiclass_target_prevalences(n_classes, n_prevalences=N_MULTICLASS_PREVALENCES, random_state=None):
    """``n_prevalences`` prevalence vectors of ``n_classes``, uniform on the simplex.

    A Dirichlet draw with every concentration at 1 is the uniform distribution
    over the simplex — the direct generalisation of bracketing ``[0, 1]``
    evenly that :data:`TARGET_PREVALENCES` does for one class's share, without
    depending on the sampling helpers behind mlquantify's own protocols
    (``mlquantify.utils._sampling``, private and shaped for drawing bags
    rather than for naming a grid of prevalences).

    Rounded for the same reason :data:`TARGET_PREVALENCES` is: a renderer or a
    reader grouping runs by their target prevalence needs a value that repeats
    exactly, not one that differs in its sixteenth decimal from one call to
    the next. :func:`utils.simulators.as_prevalence` renormalises before a
    draw reads it, so the rounding error this leaves in a row's sum is never
    seen downstream.
    """
    drawn = np.random.default_rng(random_state).dirichlet(
        np.ones(n_classes), size=n_prevalences
    )
    return tuple(tuple(round(float(share), 4) for share in row) for row in drawn)


def _prevalence_grid_seed(seed, dataset):
    """One dataset's own seed for :func:`multiclass_target_prevalences`.

    Derived from the spec's seed and the dataset's name, the same reason
    ``sweep.seed_for`` derives a cell's seed rather than sharing one across a
    grid: without it, every multiclass pool would draw the identical grid of
    prevalences. ``sweep.hash_seed`` is the shared hashing recipe both
    functions derive a seed through, so there is one rule to keep reproducible
    across a loky worker boundary rather than two copies of it.
    """
    if seed is None:
        return None
    return sweep.hash_seed((seed, dataset))


def multiclass_grid(pools, n_prevalences=N_MULTICLASS_PREVALENCES, random_state=None):
    """Every multiclass cell: each pool at its own prevalence grid.

    The counterpart of :func:`grid` for pools whose class counts differ from
    one another: a 7-class and a 10-class pool cannot share one
    ``target_prevalences`` sequence, so each pool's grid is drawn from its own
    class count (:func:`multiclass_target_prevalences`), seeded so that two
    pools sharing a class count still draw different grids
    (:func:`_prevalence_grid_seed`).
    """
    return tuple(
        Cell(dataset=name, target_prevalence=target_prevalence)
        for name, pool in pools.items()
        for target_prevalence in multiclass_target_prevalences(
            len(pool.classes),
            n_prevalences,
            random_state=_prevalence_grid_seed(random_state, name),
        )
    )


# --- The spec --------------------------------------------------------------


@dataclass(frozen=True)
class RealDataSpec:
    """One real-data experiment, entire — the sibling of ``sweep.SweepSpec``.

    Carries pools rather than data simulators: a real dataset is a fixed
    collection of instances, not something more can be drawn from. There is no
    ``reference_size`` or ``reference_prevalence`` either, for the same
    reason — a pool's reference score set is however much of the dataset its
    classifier was cross-validated over, at whatever prevalence the dataset
    itself has.
    """

    #: The grid. A sequence of cells, usually from :func:`grid`.
    cells: Sequence[Cell]

    #: One pool per dataset name a cell may reference.
    pools: Mapping[str, Pool]

    #: Score simulators in their method role (CONTEXT.md) — the same registry
    #: shape ``sweep.SweepSpec.method_simulators`` takes, and usually the same
    #: values: the method simulator is a property of the estimator, not the
    #: data, so it means the same thing here as it does in the synthetic table
    #: (ADR-0003).
    method_simulators: Mapping[
        str, Union[None, ScoreSimulator, Tuple[ScoreSimulator, ...]]
    ]

    #: The base quantifiers, by the name their runs are stored under.
    base_quantifiers: Mapping[str, type]

    bag_size: int
    repetitions: int

    #: The root every draw in the sweep is seeded from (``sweep.seed_for``,
    #: which this spec reuses unchanged: it derives a seed from ``self.seed``
    #: and a cell's own fields, and does not care which spec class it is
    #: handed).
    seed: Optional[int] = None

    #: The distance measure every meta-quantifier arm minimises, forwarded the
    #: same way ``sweep.SweepSpec.measure`` is (ADR-0012).
    measure: Optional[str] = None

    def __post_init__(self):
        runs.reject_unknown(
            "base quantifier", self.base_quantifiers, runs.BASE_QUANTIFIERS
        )
        runs.reject_unknown(
            "method simulator", self.method_simulators, runs.METHOD_SIMULATORS
        )
        if self.measure is not None:
            runs.reject_unknown("measure", (self.measure,), runs.MEASURES)
        for cell in self.cells:
            runs.reject_unknown("cell dataset", (cell.dataset,), tuple(self.pools))
        sweep.reject_baseline_without_a_home(
            self.base_quantifiers, self.method_simulators
        )


# --- Running a cell ----------------------------------------------------------


def run_cell(cell, spec):
    """Run one cell of the real-data grid and return its runs.

    One bag per repetition, drawn from the cell's dataset at its target
    prevalence, and one run per method on it — the same shape
    ``sweep.run_cell`` has, reusing its estimator seam
    (``sweep.estimator_for``) rather than a second one. A cell whose
    prevalence the pool cannot supply draws no bag at all
    (:func:`draw_bag`), and every method's run in every one of its
    repetitions is recorded missing rather than skipped (ADR-0005): the
    absence is itself a result.
    """
    pool = spec.pools[cell.dataset]
    reference = pool.reference

    rows = []
    for repetition in range(1, spec.repetitions + 1):
        bag_indices = draw_bag(
            pool,
            cell.target_prevalence,
            spec.bag_size,
            sweep.seed_for(spec, cell, repetition, sweep.BAG_DRAW),
        )
        candidate_seed = sweep.seed_for(spec, cell, repetition, sweep.CANDIDATE_DRAW)

        if bag_indices is None:
            bag_scores = None
            true_prevalence = None
            replication = None
        else:
            bag_scores = pool.scores[bag_indices]
            # The pool's classes, not the bag's: a class the target prevalence
            # rounded down to zero instances is still a class this bag has a
            # (zero) share of, and the estimate beside it will carry a value
            # for it either way (``sweep.observed_prevalence``).
            true_prevalence = sweep.observed_prevalence(
                pool.labels[bag_indices], classes=pool.classes
            )
            replication = bag_replication(bag_indices)

        for method_simulator in spec.method_simulators:
            for base_quantifier in spec.base_quantifiers:
                estimator = sweep.estimator_for(
                    spec, base_quantifier, method_simulator, reference, candidate_seed
                )
                if estimator is None:
                    continue

                rows.append(
                    {
                        "base_quantifier": base_quantifier,
                        "method_simulator": method_simulator,
                        "dataset": cell.dataset,
                        "bag_size": spec.bag_size,
                        "target_prevalence": cell.target_prevalence,
                        "true_prevalence": true_prevalence,
                        "bag_replication": replication,
                        "estimated_prevalence": (
                            None
                            if bag_scores is None
                            else sweep.estimate_or_missing(estimator, bag_scores)
                        ),
                        "repetition": repetition,
                    }
                )

    return _frame(rows)


def _progress(iterable, total, desc, colour=None):
    """``iterable``, reporting progress as it is consumed.

    A live-updating bar (``tqdm``, unchanged) in an interactive terminal.
    Standard output redirected to a file — a background run's log, which is
    how the published sweep across all eight datasets actually runs — is not
    a terminal, and tqdm's carriage-return updates land there as a single
    unreadable line rather than a bar. There, one plain line is printed per
    item instead, each carrying an ETA, so the log stays readable with `tail`
    and says how much longer the run has left.
    """
    if sys.stdout.isatty():
        yield from tqdm(iterable, total=total, desc=desc, colour=colour)
        return

    start = time.monotonic()
    for done, item in enumerate(iterable, start=1):
        yield item
        elapsed = time.monotonic() - start
        eta = elapsed / done * (total - done)
        print(
            f"{desc}: {done}/{total} done "
            f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
            flush=True,
        )


def run_sweep(spec, n_jobs=1, progress=False, desc="real-data sweep"):
    """Run every cell of the spec's grid and return the runs as one frame.

    ``sweep.run_sweep`` in every respect but the cell runner and the table
    shape: cells are independent, so ``n_jobs`` spreads them over processes,
    and missing runs are reported the same way (``sweep._warn_about_missing_
    runs``), once per method rather than once per failure. ``desc`` labels the
    progress bar and is a parameter for the same reason it is there, so that
    ``sweep.run_bag_size_sweep``'s one pass per bag size can say which pass it
    is on.
    """
    frames = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
        delayed(run_cell)(cell, spec) for cell in spec.cells
    )
    if progress:
        frames = _progress(
            frames, total=len(spec.cells), desc=desc, colour="green"
        )

    frames = list(frames)
    produced = pd.concat(frames, ignore_index=True) if frames else _frame([])
    sweep.warn_about_missing_runs(produced)
    return produced


def run_bag_size_sweep(spec, bag_sizes=sweep.BAG_SIZES, n_jobs=1, progress=False):
    """Run ``spec`` once per bag size and stack the runs into one frame.

    ``sweep.run_bag_size_sweep`` already takes ``run`` and ``kind`` apart from
    the loop over bag sizes for exactly this: this module's own
    :func:`run_sweep` matches the calling convention it expects, so this is a
    delegation rather than a second copy of the replace/concat loop.
    """
    return sweep.run_bag_size_sweep(
        spec, run=run_sweep, bag_sizes=bag_sizes, kind=runs.REAL_DATA,
        n_jobs=n_jobs, progress=progress,
    )


def _frame(rows):
    """Rows as a frame in the results module's column order, even when empty."""
    return pd.DataFrame(rows, columns=list(runs.columns_for(runs.REAL_DATA)))


def missing_by_dataset(produced):
    """How many of ``produced``'s runs have no estimate, broken out by dataset.

    ``sweep.warn_about_missing_runs`` already reports a missing count per
    method, but a real dataset's shortfall is a property of the *dataset*
    (ADR-0005) — a pool too small to fill a bag at some prevalence is missing
    there for every method alike, no matter which one is asked. A reader
    checking that a run's missing count is explained by a dataset's size
    rather than by some method failing needs the count broken out that way,
    which the per-method report alone cannot show.

    Datasets with no missing runs are absent rather than present at zero, the
    same convention ``warn_about_missing_runs`` follows for methods.
    """
    missing = produced[produced["estimated_prevalence"].isna()]
    return missing.groupby("dataset").size()


# --- Building a pool -----------------------------------------------------
#
# The only part of this module that reaches the network or trains anything.
# Everything above takes a :class:`Pool` already built, so a test can hand it
# a fabricated one and never pay for either.

#: Roughly 50k instances (ADR-0005). Bags are 100 regardless of pool size, so
#: this bounds the ten-fold cross-validation's cost without changing what is
#: measured.
POOL_CAP = 50_000

#: The eight binary tabular datasets ADR-0005 asks for, keyed by the name
#: mlquantify's own fetchers already give them — no second spelling to drift
#: from the first.
DATASET_FETCHERS = {
    "mushroom": mlquantify_datasets.fetch_mushroom,
    "banknote_authentication": mlquantify_datasets.fetch_banknote_authentication,
    "haberman_survival": mlquantify_datasets.fetch_haberman_survival,
    "pima_diabetes": mlquantify_datasets.fetch_pima_diabetes,
    "electricity_elec2": mlquantify_datasets.fetch_electricity_elec2,
    "airlines": mlquantify_datasets.fetch_airlines,
    "miniboone": mlquantify_datasets.fetch_miniboone,
    "online_news_popularity": mlquantify_datasets.fetch_online_news_popularity,
}

#: Multiclass tabular datasets (#14): evidence that the simplex simulators
#: (MVN, Dirichlet) do work the binary uniform simulator cannot even attempt
#: (CONTEXT.md).
#:
#: The two mlquantify fetches came first, chosen for differing class counts (7
#: and 10) when this table's only question was whether the simplex simulators
#: generalise at all. The rest are CSVs under ``data/``, registered through
#: ``utils/local_datasets.py``, which answers that question across class
#: counts 3 to 11 instead of at two points — and lets the Friedman and Nemenyi
#: tests in ``stats.py``, which rank methods *over datasets*, say something
#: about the multiclass table rather than only the binary one. See
#: :data:`utils.local_datasets.MULTICLASS_DATASETS` for which CSVs were left
#: out and why.
#:
#: ``yeast`` is listed and then dropped: its smallest class is under
#: :data:`MIN_CLASS_SIZE`, so :func:`_above_class_floor` excludes it (#18). It
#: stays named here rather than deleted because its absence is a fact about the
#: dataset that the floor should keep reporting, not one to bury by unlisting
#: it.
MULTICLASS_DATASET_FETCHERS = {
    "dry_bean": mlquantify_datasets.fetch_dry_bean,
    "yeast": mlquantify_datasets.fetch_yeast,
    **local_datasets.MULTICLASS_FETCHERS,
}

#: Where fetched datasets are cached, anchored to this file for the same
#: reason ``runs.ROOT`` is: a caller launched from another directory must
#: still hit the same cache.
DATA_HOME = Path(__file__).resolve().parent / "results" / "datasets"


def _encode_labels(y):
    """``y`` as ``0 .. n_classes - 1``, in sorted order of the original labels.

    Binary or multiclass alike (#14): for two classes this is exactly
    ``_binary_labels``'s own rule — the higher-sorted original label maps to
    1 — which is the same rule ``sweep.observed_prevalence`` reads a drawn
    bag's prevalence by, so "positive" means the same thing on a real dataset
    as it does in a synthetic run. A dataset of one class only is rejected:
    there is no prevalence to estimate on a pool with nothing to distinguish.
    """
    classes = np.sort(pd.unique(np.asarray(y)))
    if len(classes) < 2:
        raise ValueError(f"expected at least two classes, got {classes!r}")
    return np.searchsorted(classes, np.asarray(y))


def _numeric_features(X):
    """``X`` as floats a ``RandomForestClassifier`` can fit.

    Categorical columns are ordinal-encoded rather than one-hot: a
    high-cardinality column (``flight`` on Airlines) would otherwise turn into
    thousands of columns for a split that only needs a stable ordering, not a
    faithful category geometry. Numeric columns keep their own missing values
    imputed at the median; a categorical's missing marker (mushroom's ``'?'``)
    is already its own string category and needs nothing done to it.
    """
    X = pd.DataFrame(X)
    numeric = X.select_dtypes(include="number")
    categorical = X.select_dtypes(exclude="number")

    parts = []
    if not numeric.empty:
        parts.append(numeric.fillna(numeric.median()).to_numpy(dtype=float))
    if not categorical.empty:
        parts.append(OrdinalEncoder().fit_transform(categorical.astype(str)))

    return np.column_stack(parts)


def _capped(X, y, cap, random_state):
    """``X`` and ``y``, subsampled to ``cap`` rows when the dataset is larger."""
    if len(y) <= cap:
        return X, y
    keep = np.random.default_rng(random_state).choice(len(y), size=cap, replace=False)
    return X[keep], y[keep]


#: What share of a pool becomes the reference frame, the rest being the bag
#: frame (ADR-0013). Three fifths: the reference has to carry a usable
#: estimate of each class's score distribution, and the bag frame has to be
#: deep enough that a bag of 100 at an extreme prevalence does not need
#: replicating past :data:`REPLICATION_CAP` on the small pools — Haberman's
#: 81 minority instances are the binding case either way.
REFERENCE_FRACTION = 0.6


def _split_frames(y, reference_fraction, random_state):
    """Indices of the reference frame and the bag frame, stratified and disjoint.

    Stratified so both frames hold every class in roughly the dataset's own
    proportions: an unstratified split can leave a rare class out of one side
    entirely, which would either silently shrink the class space a quantifier
    estimates over or make a prevalence unreachable for reasons that have
    nothing to do with the dataset's size.
    """
    reference, bag = train_test_split(
        np.arange(len(y)),
        train_size=reference_fraction,
        stratify=y,
        random_state=random_state,
    )
    return np.sort(reference), np.sort(bag)


def build_pool(
    name, X, y, *, cap=POOL_CAP, cv_folds=10, random_state=None, n_jobs=1,
    reference_fraction=REFERENCE_FRACTION,
):
    """A dataset's pool: capped, encoded, split in two, and scored (ADR-0013).

    Two frames, sharing no instance. The reference frame is scored
    **out-of-fold within itself** — cross-validated, so the distribution a
    method matches a bag against is never the classifier's own training
    scores, which ADR-0005 rejects as overconfident. The bag frame is scored
    by a single forest fitted on the whole reference frame, which has never
    seen it, so those scores are out-of-sample for the same reason without
    needing a second cross-validation.

    That split is what ADR-0005 left undone. Scoring out-of-fold stops the
    reference from being overconfident; it does nothing to stop a bag's own
    instances from sitting inside the distribution the bag is estimated
    against. With one frame doing both jobs, a bag of 1000 drawn from a
    2,000-instance pool *was* half of its own reference, and the mixture
    search a base quantifier runs is easier against a reference it helped
    build.

    Takes ``X`` and ``y`` already loaded, so a test can build a pool from a
    fabricated frame without reaching the network.

    ``n_jobs`` spreads the *folds* over cores, and deliberately not the
    forest's own trees. Folds are independent and each writes its own rows, so
    running them in parallel leaves the scores bit-identical; the forest is a
    different matter — it averages its trees' probabilities, and a parallel
    average sums them in a different order, which came out 2.2e-16 away from
    the serial one when measured. That is far too small to mean anything about
    a classifier and far too large to leave in the reference score set of an
    experiment whose quantifiers do discretised searches over it: a tie broken
    the other way is a different estimate, and ADR-0011's whole point is that
    a run's inputs are reproducible rather than nearly so. So
    ``RandomForestClassifier`` keeps its own default of one worker here
    whatever ``n_jobs`` says.

    Default one, like ``run_sweep``'s: a test builds pools from tiny frames,
    where starting workers costs more than the fit. :func:`_main` passes
    ``-1``, which is the case that needs it: this step is serial across
    datasets and runs entirely before the first cell, so on the published
    multiclass table it is otherwise one core building twenty pools, two of
    them 50,000 instances wide by several hundred features.
    """
    X = _numeric_features(X)
    y = _encode_labels(y)
    X, y = _capped(X, y, cap, random_state)

    reference, bag = _split_frames(y, reference_fraction, random_state)

    reference_scores = cross_val_predict(
        RandomForestClassifier(random_state=random_state),
        X[reference],
        y[reference],
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        method="predict_proba",
        n_jobs=n_jobs,
    )

    # One forest on the whole reference frame, rather than a second
    # cross-validation: the bag frame is already unseen by it, which is the
    # property out-of-fold scoring buys for the reference. Fitting on all of
    # the reference rather than on a fold of it also means the bag frame is
    # scored by a better-trained classifier than any single fold's, which is
    # what a method would face at prediction time.
    scored_by = RandomForestClassifier(random_state=random_state).fit(
        X[reference], y[reference]
    )
    bag_scores = scored_by.predict_proba(X[bag])

    # ``predict_proba`` orders its columns by ``classes_`` and
    # ``cross_val_predict`` by the classes present in what it was given. A
    # stratified split puts every class in both frames, so the two agree — but
    # a bag matched against a reference whose columns mean different classes
    # would be silently, plausibly wrong rather than loud, so it is checked.
    if not np.array_equal(scored_by.classes_, np.unique(y[reference])):
        raise ValueError(
            f"{name!r}: the bag frame's score columns do not match the reference's"
        )

    return Pool(
        dataset=name,
        scores=bag_scores,
        labels=y[bag],
        reference_scores=reference_scores,
        reference_labels=y[reference],
    )


#: Every fetcher this module knows how to name a pool from, binary and
#: multiclass together — one lookup for :func:`fetch_pool`, which does not
#: itself care which kind of dataset it was asked for.
_ALL_DATASET_FETCHERS = {**DATASET_FETCHERS, **MULTICLASS_DATASET_FETCHERS}


def fetch_pool(
    name, *, data_home=DATA_HOME, cap=POOL_CAP, cv_folds=10, random_state=None,
    n_jobs=1, reference_fraction=REFERENCE_FRACTION,
):
    """Fetch, cache and score one dataset by name (ADR-0005).

    The one function in this module that reaches outside the process — for
    most datasets to the network, for the multiclass CSVs under ``data/`` to
    the disk (``utils/local_datasets.py``). Everything downstream of
    :class:`Pool` neither knows nor cares which it was, or whether the dataset
    is binary or multiclass (#14), which is why a local dataset needed a
    fetcher-shaped loader rather than a branch here.
    """
    fetch = _ALL_DATASET_FETCHERS[name]
    X, y = fetch(data_home=str(data_home), return_X_y=True)
    return build_pool(
        name, X, y, cap=cap, cv_folds=cv_folds, random_state=random_state,
        n_jobs=n_jobs, reference_fraction=reference_fraction,
    )


# --- The published spec -----------------------------------------------------

#: 21 target prevalences bracketing the range (ADR-0005), the same way
#: ``sweep.TARGET_PREVALENCES`` brackets it with 0.01 and 0.99 rather than 0
#: and 1: a bag drawn at exactly 0 or 1 holds one class only, which
#: ``sweep.observed_prevalence`` cannot report a prevalence *of* — there is
#: no second class for it to be a share of. Rounded for the same reason
#: ``sweep.MERGING_FACTORS`` is: unrounded, the grid step lands on values like
#: 0.35000000000000003 that a renderer grouping by prevalence would read as a
#: category of its own.
TARGET_PREVALENCES = (
    (0.01,)
    + tuple(round(float(p), 2) for p in np.linspace(0.0, 1.0, 21)[1:-1])
    + (0.99,)
)

#: The fewest instances a pool's smallest class may hold and still enter the
#: grid a published spec builds (#18). Below this, bootstrap replication
#: (:data:`REPLICATION_CAP`) still cannot cover most of the grid, and running
#: the dataset anyway returns almost nothing but missing runs — Yeast's own
#: smallest class today.
MIN_CLASS_SIZE = 30


def _meets_class_floor(pool):
    """Whether every class in ``pool``'s **bag frame** clears :data:`MIN_CLASS_SIZE`.

    The bag frame and not the whole dataset, because the floor exists to say
    whether bags can actually be filled (#18) and bags come only from there
    (ADR-0013). That makes it a stricter test than before the split — a class
    needs :data:`MIN_CLASS_SIZE` instances in the two fifths of the pool that
    became the bag frame, not in all of it — which is the honest reading: a
    class with 30 instances spread across both frames never could fill a bag
    from the frame bags are drawn from.
    """
    return np.unique(pool.labels, return_counts=True)[1].min() >= MIN_CLASS_SIZE


def _above_class_floor(pools):
    """``pools``, with any pool under :data:`MIN_CLASS_SIZE` dropped (#18).

    Shared by :func:`build_spec` and :func:`build_multiclass_spec` so the
    floor is spelled once rather than copied into each.
    """
    return {name: pool for name, pool in pools.items() if _meets_class_floor(pool)}


def build_spec(pools, *, seed=None, measure=None):
    """The published real-data spec (ADR-0005) over these pools.

    A function of the pools rather than a module constant: building them
    reaches the network and trains a classifier per dataset, which nothing at
    import time should pay for. A caller who only wants one dataset passes a
    ``pools`` of one — the grid is built from whatever it is given, the same
    way ``sweep.grid`` is. A pool whose smallest class falls under
    :data:`MIN_CLASS_SIZE` is dropped before the grid is built rather than
    handed a cell of its own (#18).
    """
    pools = _above_class_floor(pools)
    return RealDataSpec(
        cells=grid(tuple(pools), TARGET_PREVALENCES),
        pools=pools,
        method_simulators=sweep.METHOD_SIMULATORS,
        base_quantifiers=sweep.BASE_QUANTIFIERS,
        bag_size=100,
        #: Matches the synthetic sweep's own ten (ADR-0005, #9).
        repetitions=10,
        seed=seed,
        measure=measure,
    )


def build_multiclass_spec(pools, *, seed=None, measure=None, n_prevalences=N_MULTICLASS_PREVALENCES):
    """The multiclass real-data spec (#14): :func:`build_spec`'s sibling.

    Same estimator block and the same bag size and repetitions — a method
    means the same thing whether the dataset behind it is binary or
    multiclass — differing only in how the grid is built: each pool's own
    prevalence vectors (:func:`multiclass_grid`) rather than one prevalence
    sequence shared across datasets. Subject to the same class floor
    (:data:`MIN_CLASS_SIZE`, #18) as :func:`build_spec`.
    """
    pools = _above_class_floor(pools)
    return RealDataSpec(
        cells=multiclass_grid(pools, n_prevalences, random_state=seed),
        pools=pools,
        method_simulators=sweep.METHOD_SIMULATORS,
        base_quantifiers=sweep.BASE_QUANTIFIERS,
        bag_size=100,
        repetitions=10,
        seed=seed,
        measure=measure,
    )


def _main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        nargs="+",
        metavar="NAME",
        choices=tuple(_ALL_DATASET_FETCHERS),
        help=(
            "run these datasets only, rather than the whole table. Names must "
            "all come from the table --multiclass selects: the binary one by "
            "default, the multiclass one with --multiclass"
        ),
    )
    parser.add_argument(
        "--multiclass",
        action="store_true",
        help="run the multiclass datasets (#14) instead of the eight binary ones",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        metavar="N",
        help=(
            "how many processes to spread cells over, and folds over while "
            "building pools (default -1, every core). Worth setting below -1's "
            "worth on a memory-bound machine rather than a core-bound one: "
            "every worker is handed a copy of every pool in the spec, so peak "
            "memory grows with this and not with the grid"
        ),
    )
    args = parser.parse_args()

    # Which table a run belongs to is the mode's call, not the dataset list's:
    # the grid, the spec builder and the prevalence shape all differ between
    # them (:func:`build_spec` against :func:`build_multiclass_spec`), so a
    # name from the wrong table would be run under the wrong grid rather than
    # merely mislabelled.
    table = MULTICLASS_DATASET_FETCHERS if args.multiclass else DATASET_FETCHERS
    if args.dataset:
        wrong_table = [name for name in args.dataset if name not in table]
        if wrong_table:
            raise SystemExit(
                f"{wrong_table} are not in the "
                f"{'multiclass' if args.multiclass else 'binary'} table; "
                f"{'drop' if args.multiclass else 'add'} --multiclass to run them"
            )
        names = tuple(dict.fromkeys(args.dataset))
    else:
        names = tuple(table)

    pools = {
        name: fetch_pool(name, random_state=20260911, n_jobs=args.n_jobs)
        for name in _progress(names, total=len(names), desc="datasets", colour="blue")
    }

    spec_builder = build_multiclass_spec if args.multiclass else build_spec
    produced = run_bag_size_sweep(
        spec_builder(pools, seed=20260911), n_jobs=args.n_jobs, progress=True
    )

    # Not sweep.validate_and_save: its collapse check (sweep.validate_no_
    # collapsed_groups) is calibrated against the synthetic sweep's continuous
    # simulators, and false-positives on a real classifier's weakly-separable
    # output — measured on the published Haberman run, where several honest,
    # differently-implemented base quantifiers legitimately agreed a hard bag
    # carried no signal, at a rate (2.9% of meta-quantifier groups, 0% of the
    # arm that searches every candidate) nothing like the systemic collapse
    # the check exists to catch. The stale-estimate half is unaffected by that
    # and stands unchanged here.
    sweep.validate_no_stale_estimates(produced)

    missing = int(produced["estimated_prevalence"].isna().sum())
    print(f"{missing} of {len(produced)} runs have no estimate")
    for dataset, count in missing_by_dataset(produced).items():
        print(f"  {dataset}: {count}")

    # Merged rather than written over: one real-data table holds the binary
    # and the multiclass runs together (``runs`` module docstring), but one
    # invocation produces only one of them, and with ``--dataset`` only part
    # of one. A plain save would have the multiclass run discard the published
    # binary table it was meant to join.
    print(runs.save_merged_by_dataset(produced, runs.REAL_DATA))


if __name__ == "__main__":
    _main()
