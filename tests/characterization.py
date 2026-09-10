"""The characterization grid and its golden record of plain-quantifier runs.

The fixture is a golden record of runs that use no meta-quantifier. Meta-
quantifier runs are excluded by construction: their behaviour legitimately
changed across the 0.2.0 → 0.5.1 upgrade, and freezing it here would enshrine
the defect being fixed (ADR-0001).

It was first captured on ``mlquantify==0.2.0`` to prove the upgrade left plain
runs untouched. It did not: DyS and HDy changed, because 0.5.1 turns off the
histogram smoothing 0.2.0 applied unconditionally and stops aggregating DyS
over bin sizes by median. ADR-0006 records the decision to accept the new
defaults. The 0.2.0 record and the measured deltas are in commit 58102cc and
ADR-0006 respectively.

The fixture now guards the *current* library against future drift rather than
proving anything about the upgrade. If it fails, a plain run changed — treat
that as a finding, not as a fixture to refresh.

**Re-captured once since, when the seeding changed under it** (ADR-0011). Every
draw is now seeded from the cell and the repetition it belongs to rather than
from one generator threaded through the cell, so this grid draws different
scores than the record held — the same quantifiers on different bags. What was
measured before accepting that is in the ADR: across four base seeds the two
schemes' per-quantifier error ranges overlap for all eleven quantifiers, and
their pooled means differ by 0.0002.

That re-capture is also why the record is in today's schema. It used to be in
the one the sweep returned before ADR-0009, translated forward on the way in
and compared on absolute error — the one quantity both schemas expressed. A
record being rewritten anyway has no reason to be rewritten in a schema nothing
else uses, so the translation is gone and the comparison is the whole run: the
prevalence each bag was drawn at and the prevalence each method estimated for
it, which is a stricter record than the error alone.

There is still no regeneration command. Re-capturing means accepting that a
plain run's number changed, which is a decision to argue for in an ADR — as
ADR-0006 and ADR-0011 did — not a command to reach for when a test goes red.
"""

from pathlib import Path

import pandas as pd

import runs
import sweep
from sweep import Cell, SweepSpec

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "plain_quantifier_runs.csv"

#: Fixed, so that the grid draws the same scores every time it is replayed.
#: Held at the value the record was first captured under, even though every
#: draw derived from it moved when ADR-0011 changed how it is derived: changing
#: the seed as well would have confused two reasons for the numbers to move.
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

#: Everything identifying a run. What is left over is what the run measured.
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


def run_grid():
    """Run the characterization grid and return its runs, ready to compare."""
    return _ordered(sweep.run_sweep(SPEC))


def load_fixture():
    """Read the committed golden record.

    ``round_trip`` keeps the floats bit-identical. NA inference is off because
    the sweep names its no-meta-quantifier arm ``"none"``: pandas leaves that
    spelling alone and would not leave ``"None"`` alone, and a record read with
    a column of NaNs where the arm should be would compare as a drift in the
    runs rather than as the misreading it is. The one column that may legibly
    hold nothing is the estimate, and this grid has none —
    ``test_every_run_in_the_grid_produced_an_estimate`` is what keeps that so.
    """
    frame = pd.read_csv(
        FIXTURE_PATH, float_precision="round_trip", keep_default_na=False
    )
    return _ordered(frame)


def _ordered(frame):
    """Put runs in a stable order, in the column order the runs module names."""
    return (
        frame[list(runs.columns_for(runs.SYNTHETIC))]
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )
