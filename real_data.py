"""The real-data pipeline: datasets, classifier, protocol (ADR-0005, #10).

The synthetic experiment's reference and bags both come from data simulators
(``sweep.py``); this one's come from an actual classifier and an actual
dataset instead. What differs is entirely upstream of an estimate: once a bag
of real scores exists, the four estimator adapters in ``sweep.py`` cannot tell
it from a simulated one, so this module reuses them rather than growing a
second copy.

A *pool* is the fixed collection of real instances a dataset's bags are drawn
from — the counterpart of a data simulator, except nothing here can draw more
of it than exists. Its scores are a Random Forest's ten-fold out-of-fold
probabilities over the whole (possibly capped) dataset, which double as the
reference score set every method matches a bag against: out-of-fold rather
than refit, so the reference is never the classifier's own training scores
(ADR-0005).

Bags are drawn under the artificial-prevalence protocol, without replacement,
class by class. That is what lets a cell fail explicitly: Haberman has 81
minority instances, so no bag at or above 82% positive can be filled from it,
and :func:`draw_bag` says so with ``None`` rather than padding the shortfall
with a repeated instance.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm

import runs
import sweep
from mlquantify import datasets as mlquantify_datasets
from utils.simulators import ScoreSimulator, as_prevalence, class_counts

# --- The pool ----------------------------------------------------------------


@dataclass(frozen=True)
class Pool:
    """The fixed collection of real instances one dataset's bags are drawn from.

    ``scores`` are the classifier's out-of-fold posterior probabilities
    (ADR-0005) and ``labels`` are the dataset's own, 0/1 and in the same row
    order. Together they are this dataset's reference score set, the same
    shape :class:`sweep.ReferenceScoreSet` carries for a synthetic cell — a
    pool differs only in that nothing can draw a fresh one; every bag comes
    from a subset of the rows already here.
    """

    dataset: str
    scores: np.ndarray
    labels: np.ndarray

    @property
    def classes(self):
        return np.unique(self.labels)

    @property
    def reference(self):
        return sweep.ReferenceScoreSet(self.scores, self.labels)


def draw_bag(pool, target_prevalence, bag_size, random_state):
    """The indices of one bag drawn from ``pool``, or ``None`` if it cannot be.

    Drawn without replacement, class by class, at the counts
    :func:`utils.simulators.class_counts` divides ``bag_size`` into. Sampling
    with replacement instead would let every prevalence be reached from any
    pool, however small — which would hide the shortfall ADR-0005 records
    rather than report it: a bag that reused the same instance forty times to
    reach a prevalence Haberman's 81 minority instances cannot support is not
    the same measurement as one that used forty distinct patients.
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
        if len(available) < count:
            return None
        drawn.append(rng.choice(available, size=count, replace=False))

    indices = np.concatenate(drawn)
    rng.shuffle(indices)
    return indices


# --- The grid ------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One point of the real-data grid: which dataset, at what prevalence.

    The counterpart of ``sweep.Cell`` without the axes a real dataset has no
    say over — there is no simulator or merging factor here, only which pool
    a bag comes from and the prevalence it is asked for.
    """

    dataset: str
    target_prevalence: float


def grid(datasets, target_prevalences):
    """Every cell of the cross-product: each dataset at every prevalence."""
    return tuple(
        Cell(dataset=dataset, target_prevalence=target_prevalence)
        for dataset in datasets
        for target_prevalence in target_prevalences
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
        else:
            bag_scores = pool.scores[bag_indices]
            true_prevalence = sweep.observed_prevalence(pool.labels[bag_indices])

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
                        "target_prevalence": cell.target_prevalence,
                        "true_prevalence": true_prevalence,
                        "estimated_prevalence": (
                            None
                            if bag_scores is None
                            else sweep.estimate_or_missing(estimator, bag_scores)
                        ),
                        "repetition": repetition,
                    }
                )

    return _frame(rows)


def run_sweep(spec, n_jobs=1, progress=False):
    """Run every cell of the spec's grid and return the runs as one frame.

    ``sweep.run_sweep`` in every respect but the cell runner and the table
    shape: cells are independent, so ``n_jobs`` spreads them over processes,
    and missing runs are reported the same way (``sweep._warn_about_missing_
    runs``), once per method rather than once per failure.
    """
    frames = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
        delayed(run_cell)(cell, spec) for cell in spec.cells
    )
    if progress:
        frames = tqdm(
            frames, total=len(spec.cells), desc="real-data sweep", colour="green"
        )

    frames = list(frames)
    produced = pd.concat(frames, ignore_index=True) if frames else _frame([])
    sweep.warn_about_missing_runs(produced)
    return produced


def _frame(rows):
    """Rows as a frame in the results module's column order, even when empty."""
    return pd.DataFrame(rows, columns=list(runs.columns_for(runs.REAL_DATA)))


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

#: Where fetched datasets are cached, anchored to this file for the same
#: reason ``runs.ROOT`` is: a caller launched from another directory must
#: still hit the same cache.
DATA_HOME = Path(__file__).resolve().parent / "results" / "datasets"


def _binary_labels(y):
    """``y`` as 0/1, the higher-sorted original label mapped to 1.

    The same rule ``sweep.observed_prevalence`` reads a drawn bag's
    prevalence by, so "positive" means the same thing on a real dataset as it
    does in a synthetic run.
    """
    classes = np.sort(pd.unique(np.asarray(y)))
    if len(classes) != 2:
        raise ValueError(
            f"expected a binary target, got {len(classes)} classes: {classes!r}"
        )
    return (np.asarray(y) == classes[1]).astype(int)


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


def build_pool(name, X, y, *, cap=POOL_CAP, cv_folds=10, random_state=None):
    """A dataset's pool: capped, encoded, and scored out-of-fold.

    Out-of-fold rather than refit, so the reference a method matches a bag
    against is never the classifier's own training scores (ADR-0005) — those
    are overconfident and would make it unrepresentative of what the method
    sees at prediction time. Takes ``X`` and ``y`` already loaded, so a test
    can build a pool from a fabricated frame without reaching the network.
    """
    X = _numeric_features(X)
    y = _binary_labels(y)
    X, y = _capped(X, y, cap, random_state)

    oof_scores = cross_val_predict(
        RandomForestClassifier(random_state=random_state),
        X,
        y,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        method="predict_proba",
    )
    return Pool(dataset=name, scores=oof_scores, labels=y)


def fetch_pool(name, *, data_home=DATA_HOME, cap=POOL_CAP, cv_folds=10, random_state=None):
    """Fetch, cache and score one dataset by name (ADR-0005).

    The one function in this module that reaches the network — everything
    downstream of :class:`Pool` neither knows nor cares where it came from.
    """
    fetch = DATASET_FETCHERS[name]
    X, y = fetch(data_home=str(data_home), return_X_y=True)
    return build_pool(name, X, y, cap=cap, cv_folds=cv_folds, random_state=random_state)


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


def build_spec(pools, *, seed=None, measure=None):
    """The published real-data spec (ADR-0005) over these pools.

    A function of the pools rather than a module constant: building them
    reaches the network and trains a classifier per dataset, which nothing at
    import time should pay for. A caller who only wants one dataset passes a
    ``pools`` of one — the grid is built from whatever it is given, the same
    way ``sweep.grid`` is.
    """
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


def _main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_FETCHERS),
        help="run one dataset only, rather than all eight",
    )
    args = parser.parse_args()

    names = (args.dataset,) if args.dataset else tuple(DATASET_FETCHERS)
    pools = {
        name: fetch_pool(name, random_state=20260911)
        for name in tqdm(names, desc="datasets", colour="blue")
    }

    produced = run_sweep(build_spec(pools, seed=20260911), n_jobs=-1, progress=True)

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
    print(runs.save(produced, runs.REAL_DATA))


if __name__ == "__main__":
    _main()
