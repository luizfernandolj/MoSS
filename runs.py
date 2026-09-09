"""How runs are stored, loaded and named. One owner for all three.

A *run* is one estimate: one bag, one method, one base quantifier, with the
true and estimated prevalence recorded together. Everything that reads or
writes runs goes through here, so the dashboard and the figure export can no
longer disagree about which methods exist. They did: the export filtered on
base quantifiers spelled ``"X"`` and ``"MS"`` while the sweep records ``"TX"``
and ``"TMS"``, and two methods dropped out of the published grid in silence.

Two tables, not one union table (ADR-0003). Synthetic and real-data runs share
an estimator block — the base quantifier and the method simulator inside the
meta-quantifier — and differ only in the data block. Merging factors and data
simulators describe how synthetic scores were made and mean nothing for a real
dataset, so they live in the synthetic table alone rather than as columns that
are null for half the rows. The method simulator is in **both**: it describes
the estimator, not the data.

Storage is Parquet under a layout callers never name. They ask for a *kind*.
The previous single CSV reached 291 MB, was split three ways to clear a
hosting file-size limit, and that split then leaked into every reader — which
is how the readers drifted apart in the first place.

Three parts of this module are deliberately ahead of their callers, and are
covered by tests alone until those land: the real-data table (ADR-0005, built
by #10), and ``SIMULATOR_NAMES`` / ``METHOD_SIMULATOR_NAMES``, which translate
the sweep's registry keys into the vocabulary below and are used when the
sweep starts writing through here (#6).

Prevalences are scalars for binary runs and vectors for multiclass ones, in
the same columns. That is not a schema change (ADR-0003), and
``absolute_error`` handles both. A run whose estimate could not be produced
carries a null estimated prevalence rather than being dropped: Haberman cannot
fill a size-100 bag at high prevalence, and that absence is data (ADR-0005).
"""

from pathlib import Path

import numpy as np
import pandas as pd

# --- Kinds -----------------------------------------------------------------

SYNTHETIC = "synthetic"
REAL_DATA = "real-data"
KINDS = (SYNTHETIC, REAL_DATA)

# --- Vocabulary ------------------------------------------------------------

#: Score simulators, named for what they draw from rather than for the module
#: they live in. See CONTEXT.md.
UNIFORM = "uniform"
MVN = "mvn"
DIRICHLET = "dirichlet"
SIMULATORS = (UNIFORM, MVN, DIRICHLET)

#: The sweep's arm that uses no meta-quantifier, and so no method simulator.
NO_METHOD_SIMULATOR = "none"
METHOD_SIMULATORS = SIMULATORS + (NO_METHOD_SIMULATOR,)

#: Every base quantifier the sweep runs. ``tests/test_runs.py`` asserts this
#: matches ``variables.QUANTIFIERS`` exactly, so adding one there without
#: adding it here fails rather than silently producing unreadable runs.
BASE_QUANTIFIERS = (
    "DyS",
    "HDy",
    "SORD",
    "SMM",
    "TAC",
    "TX",
    "T50",
    "TMAX",
    "TMS",
    "TMS2",
    "CC",
)

#: The sweep's registries still key their simulators by the historical
#: spellings, which ADR-0008 kept because they were the values written to the
#: result columns. Here is where that stops being true: these two maps are the
#: only place those keys survive, and runs on disk carry the canonical names
#: above. Both are asserted complete against the registries, so a rename there
#: cannot leave a stale key behind.
SIMULATOR_NAMES = {"MoSS": UNIFORM, "MoSS_MN": MVN, "MoSS_Dir": DIRICHLET}
METHOD_SIMULATOR_NAMES = {
    "Quadapt_MoSS": UNIFORM,
    "Quadapt_MvN": MVN,
    "Quadapt_Dir": DIRICHLET,
    "None": NO_METHOD_SIMULATOR,
}

#: How a simulator is spelled in a figure legend.
SIMULATOR_LABELS = {UNIFORM: "Uniform", MVN: "MVN", DIRICHLET: "Dirichlet"}

#: The one base quantifier that never reads a reference score set, so it takes
#: no meta-quantifier path and always appears once, as a baseline. Named here
#: because every reader that draws it needs the same answer.
BASELINE_QUANTIFIER = "CC"

# --- Schema ----------------------------------------------------------------

#: Identical in both tables, and first in both, so one label function and one
#: statistical comparison serve synthetic and real-data runs alike.
ESTIMATOR_COLUMNS = ("base_quantifier", "method_simulator")

_COLUMNS = {
    SYNTHETIC: ESTIMATOR_COLUMNS
    + (
        "reference_simulator",
        "reference_merging_factor",
        "bag_simulator",
        "bag_merging_factor",
        "target_prevalence",
        "true_prevalence",
        "estimated_prevalence",
        "repetition",
    ),
    REAL_DATA: ESTIMATOR_COLUMNS
    + (
        "dataset",
        "target_prevalence",
        "true_prevalence",
        "estimated_prevalence",
        "repetition",
    ),
}

_VOCABULARIES = {
    "base_quantifier": BASE_QUANTIFIERS,
    "method_simulator": METHOD_SIMULATORS,
    "reference_simulator": SIMULATORS,
    "bag_simulator": SIMULATORS,
}

# --- Storage ---------------------------------------------------------------

#: Anchored to this file, not to the working directory, so a dashboard and a
#: figure script launched from different places read the same runs.
ROOT = Path(__file__).resolve().parent / "results"

_FILES = {SYNTHETIC: "runs/synthetic.parquet", REAL_DATA: "runs/real-data.parquet"}


def columns_for(kind):
    """The columns a table of this kind carries, in order."""
    try:
        return _COLUMNS[kind]
    except KeyError:
        raise ValueError(f"unknown run kind {kind!r}; expected one of {KINDS}") from None


def save(runs, kind, root=ROOT):
    """Write a table of runs, rejecting anything a reader would misread.

    ``root`` exists so tests can write somewhere of their own. Production
    callers pass a kind and nothing else.
    """
    expected = columns_for(kind)
    if tuple(runs.columns) != expected:
        raise ValueError(
            f"{kind} runs need exactly the columns {expected}, "
            f"got {tuple(runs.columns)}"
        )
    for column, vocabulary in _VOCABULARIES.items():
        if column not in expected:
            continue
        unknown = sorted(set(runs[column].dropna().unique()) - set(vocabulary))
        if unknown:
            raise ValueError(
                f"unknown {column} {unknown}; expected one of {vocabulary}"
            )

    path = Path(root) / _FILES[kind]
    path.parent.mkdir(parents=True, exist_ok=True)
    runs.to_parquet(path, index=False)
    return path


def load(kind, root=ROOT, columns=None):
    """Read a table of runs by kind.

    ``columns`` prunes at the file rather than after loading, which is what
    makes a full sweep affordable to open.
    """
    columns_for(kind)  # rejects an unknown kind before touching the disk
    path = Path(root) / _FILES[kind]
    if not path.exists():
        raise FileNotFoundError(
            f"no {kind} runs under {root}. Every result file predating the "
            "0.5.1 port is void (ADR-0001), and the sweep does not yet write "
            "through this module — that lands with the sweep seam (#6), and "
            "the re-run itself with #9."
        )
    return pd.read_parquet(path, columns=columns)


def load_labelled(kind, root=ROOT, columns=None):
    """Runs of this kind with the two columns every reader derives attached.

    ``method`` names the method, ``absolute_error`` measures it. Both are
    derived rather than stored, and deriving them was the other half of what
    the dashboard and the figure export each used to do for themselves.
    """
    labelled = load(kind, root=root, columns=columns)
    labelled["method"] = method_labels(labelled)
    labelled["absolute_error"] = absolute_error(labelled)
    return labelled


# --- Naming ----------------------------------------------------------------


def method_label(base_quantifier, method_simulator):
    """Name the method a run used, as it should appear to a reader."""
    if method_simulator not in METHOD_SIMULATORS:
        raise ValueError(
            f"unknown method_simulator {method_simulator!r}; "
            f"expected one of {METHOD_SIMULATORS}"
        )
    if method_simulator == NO_METHOD_SIMULATOR:
        return str(base_quantifier)
    return f"QuaDapt-{SIMULATOR_LABELS[method_simulator]}({base_quantifier})"


def method_labels(runs):
    """The method label of every run, as a Series aligned to ``runs``.

    Resolved once per distinct method rather than once per run, so a full
    sweep costs a few dozen calls rather than millions.
    """
    pairs = list(
        zip(runs["base_quantifier"].astype(str), runs["method_simulator"].astype(str))
    )
    labels = {pair: method_label(*pair) for pair in set(pairs)}
    return pd.Series([labels[pair] for pair in pairs], index=runs.index, name="method")


# --- Derived quantities ----------------------------------------------------


def absolute_error(runs):
    """Each run's absolute error, averaged over classes when multiclass.

    Derived rather than stored, so no reader can find an error that disagrees
    with the prevalences it sits beside.
    """
    true = runs["true_prevalence"]
    estimated = runs["estimated_prevalence"]

    if _holds_vectors(true):
        errors = [
            np.nan if estimate is None else np.mean(np.abs(np.asarray(estimate) - np.asarray(actual)))
            for actual, estimate in zip(true, estimated)
        ]
        return pd.Series(errors, index=runs.index, name="absolute_error")

    return (estimated - true).abs().rename("absolute_error")


def _holds_vectors(prevalences):
    for value in prevalences:
        if value is None:
            continue
        return isinstance(value, (list, tuple, np.ndarray))
    return False
