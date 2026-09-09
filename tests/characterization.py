"""The characterization grid and its golden record of plain-quantifier runs.

The fixture is a golden record of runs that use no meta-quantifier. Meta-
quantifier runs are excluded by construction: their behaviour legitimately
changed across the 0.2.0 → 0.5.1 upgrade, and freezing it here would enshrine
the defect being fixed (ADR-0001).

It was first captured on ``mlquantify==0.2.0`` to prove the upgrade left plain
runs untouched. It did not: DyS and HDy changed, because 0.5.1 turns off the
histogram smoothing 0.2.0 applied unconditionally and stops aggregating DyS
over bin sizes by median. ADR-0006 records the decision to accept the new
defaults, and the fixture was re-captured on 0.5.1. The 0.2.0 record and the
measured deltas are in commit 58102cc and ADR-0006 respectively.

The fixture now guards the *current* library against future drift rather than
proving anything about the upgrade. If it fails, a plain run changed — treat
that as a finding, not as a fixture to refresh.

**The file is not regenerated when the sweep's schema changes.** It was written
when the sweep returned its own frame, in column names and registry spellings
that nothing else in the project uses any more (ADR-0009). Rather than
re-capture it
— which would silently accept whatever the new code does, the one thing a
golden record exists to prevent — it is translated forward on the way in, and
the two frames are compared on the quantity both schemas can express: the
absolute error of each run. The spellings that survive only here are
:data:`FIXTURE_COLUMNS` and :data:`FIXTURE_SIMULATORS`.

There is therefore no regeneration command. Re-capturing this record means
accepting that a plain run's number changed, which is a decision to argue for
in an ADR — as ADR-0006 did — not a command to reach for when a test goes red.
"""

from pathlib import Path

import pandas as pd

import runs
import sweep
from sweep import Cell, SweepSpec

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "plain_quantifier_runs.csv"

#: Fixed so the experiment's own data simulators are reproducible. The method
#: simulators inside a meta-quantifier are not seedable from here (ADR-0004),
#: which is one more reason the fixture covers plain runs only.
SEED = 20260908

#: Only the sweep's "no meta-quantifier" arm.
PLAIN_ONLY = {runs.NO_METHOD_SIMULATOR: None}

#: Deliberately small, and deliberately not a cross-product: each simulator
#: draws both the reference set and the bags, so the pairs are matched. The two
#: reference merging factors bracket separated and overlapping scores, and the
#: two prevalences bracket skewed and balanced bags. Every base quantifier runs
#: in every cell, which covers both distance measures the mixture search
#: supports: DyS runs the topsoe path and SORD the sord path.
GRID = tuple(
    Cell(
        reference_simulator=simulator,
        reference_merging_factor=reference_merging_factor,
        bag_simulator=simulator,
        bag_merging_factor=0.5,
        target_prevalence=target_prevalence,
    )
    for simulator in runs.SIMULATORS
    for reference_merging_factor in (0.2, 0.8)
    for target_prevalence in (0.1, 0.6)
)

#: The grid above, plus everything else the sweep needs to run it.
SPEC = SweepSpec(
    cells=GRID,
    data_simulators=sweep.DATA_SIMULATORS,
    method_simulators=PLAIN_ONLY,
    base_quantifiers=sweep.BASE_QUANTIFIERS,
    reference_size=2000,
    bag_size=100,
    repetitions=3,
    seed=SEED,
)

#: How the frozen record spells the columns the sweep used to return.
FIXTURE_COLUMNS = {
    "Quantifier": "base_quantifier",
    "Quadapt_Variant": "method_simulator",
    "MoSS_Train_Variant": "reference_simulator",
    "MoSS_Test_Variant": "bag_simulator",
    "m_train": "reference_merging_factor",
    "m_test": "bag_merging_factor",
    "alpha": "target_prevalence",
    "Iteration": "repetition",
    "MAE": "absolute_error",
}

#: How it spells the simulators. These were the sweep's registry keys when the
#: record was captured; ADR-0008 kept them because they were the values written
#: to the result columns, and ADR-0009 is where that stopped being true. This is
#: the last place they survive.
FIXTURE_SIMULATORS = {
    "MoSS": runs.UNIFORM,
    "MoSS_MN": runs.MVN,
    "MoSS_Dir": runs.DIRICHLET,
    "None": runs.NO_METHOD_SIMULATOR,
}

#: Everything identifying a run, in both schemas.
KEY_COLUMNS = [
    "base_quantifier",
    "method_simulator",
    "reference_simulator",
    "reference_merging_factor",
    "bag_simulator",
    "bag_merging_factor",
    "target_prevalence",
    "repetition",
]

#: The old frame stored an error and no prevalences; the new one stores
#: prevalences and derives the error. The error is therefore the only
#: measurement the two schemas share — and it was the only one the record ever
#: held, so comparing on it costs nothing that was there before.
COMPARISON_COLUMNS = KEY_COLUMNS + ["absolute_error"]


def run_grid():
    """Run the characterization grid and return its runs, ready to compare."""
    return comparable(sweep.run_sweep(SPEC))


def comparable(produced):
    """Reduce runs to what the golden record can be compared against."""
    with_error = produced.assign(absolute_error=runs.absolute_error(produced))
    return _ordered(with_error[COMPARISON_COLUMNS])


def load_fixture():
    """Read the committed golden record and translate it into today's schema."""
    frame = read_fixture().rename(columns=FIXTURE_COLUMNS)
    for column in ("reference_simulator", "bag_simulator", "method_simulator"):
        frame[column] = frame[column].map(FIXTURE_SIMULATORS)
    return _ordered(frame[COMPARISON_COLUMNS])


def read_fixture():
    """The golden record exactly as written, in the spellings it was written in.

    ``round_trip`` keeps the floats bit-identical, and NA inference is off
    because the sweep named its no-meta-quantifier arm ``"None"``, which pandas
    would otherwise read as a missing value.
    """
    return pd.read_csv(FIXTURE_PATH, float_precision="round_trip", keep_default_na=False)


def _ordered(frame):
    """Put runs in a stable order so two frames can be compared directly."""
    return frame.sort_values(KEY_COLUMNS).reset_index(drop=True)
