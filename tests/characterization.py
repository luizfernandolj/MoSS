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

Regenerate with::

    .venv/bin/python -m tests.characterization
"""

from pathlib import Path

import pandas as pd

from binary_experiment import run_experiment
from variables import MOSS_VARIANTS

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "plain_quantifier_runs.csv"

#: Fixed so the experiment's own data simulators are reproducible. The method
#: simulators inside a meta-quantifier are not seedable from here (ADR-0004),
#: which is one more reason the fixture covers plain runs only.
SEED = 20260908

#: Only the sweep entry's "no meta-quantifier" arm.
PLAIN_ONLY = {"None": None}

#: Deliberately small. Each simulator appears in both the data and the test
#: role; the two merging factors bracket separated and overlapping scores, and
#: the two prevalences bracket skewed and balanced bags. Every base quantifier
#: in the registry runs in every cell, which is what covers both distance
#: measures the mixture search supports: DyS runs the topsoe path and SORD the
#: sord path, the two branches a later measure ablation selects between.
SIMULATOR_PAIRS = (("MoSS", "MoSS"), ("MoSS_MN", "MoSS_MN"), ("MoSS_Dir", "MoSS_Dir"))
M_TRAINS = (0.2, 0.8)
M_TESTS = (0.5,)
ALPHAS = (0.1, 0.6)

#: Columns identifying a run, used to put both frames in the same order.
KEY_COLUMNS = [
    "MoSS_Train_Variant",
    "MoSS_Test_Variant",
    "m_train",
    "m_test",
    "alpha",
    "Iteration",
    "Quantifier",
]


def cells():
    """Yield the grid cells, as keyword arguments to the sweep entry point."""
    for train_name, test_name in SIMULATOR_PAIRS:
        for m_train in M_TRAINS:
            for m_test in M_TESTS:
                for alpha in ALPHAS:
                    yield {
                        "m_train": m_train,
                        "m_test": m_test,
                        "alpha": alpha,
                        "moss_train_variant": MOSS_VARIANTS[train_name],
                        "moss_test_variant": MOSS_VARIANTS[test_name],
                        "moss_train_variant_name": train_name,
                        "moss_test_variant_name": test_name,
                    }


def run_grid():
    """Run the characterization grid and return its runs as one frame."""
    frames = [
        run_experiment(
            **cell,
            random_state=SEED,
            strict=True,
            quadapt_variants=PLAIN_ONLY,
        )
        for cell in cells()
    ]
    return normalise(pd.concat(frames, ignore_index=True))


def normalise(runs):
    """Put runs in a stable order so two frames can be compared directly."""
    return runs.sort_values(KEY_COLUMNS).reset_index(drop=True)


def load_fixture():
    """Read the committed golden record back as it was written.

    ``round_trip`` keeps the floats bit-identical, and NA inference is off
    because the sweep names its no-meta-quantifier arm ``"None"``, which
    pandas would otherwise read as a missing value.
    """
    runs = pd.read_csv(FIXTURE_PATH, float_precision="round_trip", keep_default_na=False)
    return normalise(runs)


if __name__ == "__main__":
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    runs = run_grid()
    runs.to_csv(FIXTURE_PATH, index=False)
    print(f"wrote {len(runs)} runs to {FIXTURE_PATH}")
