"""Small run tables to test the runs module against.

Deliberately built in code rather than committed: every base quantifier and
every simulator name comes from ``runs``' own vocabulary, so a rename there
reaches these fixtures instead of leaving them frozen at the old spelling.
That is the whole point of the renderer-filter regression (#5) — a filter can
only be proven live against data that tracks the vocabulary.
"""

import itertools

import pandas as pd
import pytest

import runs
import sweep


@pytest.fixture
def results_root(tmp_path):
    """A results directory of this test's own, never the project's."""
    return tmp_path / "results"


@pytest.fixture
def smoke_spec():
    """The smallest sweep that still covers every method, as a value.

    Tests that vary one thing about the sweep say so by replacing one field of
    this, which is the property the spec exists to have.
    """
    return sweep.SMOKE_SWEEP


@pytest.fixture
def synthetic_runs():
    """One run per (base quantifier, method simulator, bag simulator)."""
    rows = []
    for i, (base, method, bag) in enumerate(
        itertools.product(
            runs.BASE_QUANTIFIERS, runs.METHOD_SIMULATORS, runs.SIMULATORS
        )
    ):
        rows.append(
            {
                "base_quantifier": base,
                "method_simulator": method,
                "reference_simulator": runs.UNIFORM,
                "reference_merging_factor": 0.5,
                "bag_simulator": bag,
                "bag_merging_factor": 0.25 + 0.25 * (i % 3),
                "target_prevalence": 0.4,
                "true_prevalence": 0.42,
                "estimated_prevalence": 0.42 + 0.001 * i,
                "repetition": 1,
            }
        )
    return pd.DataFrame(rows, columns=list(runs.columns_for(runs.SYNTHETIC)))


@pytest.fixture
def real_data_runs():
    """One run per (base quantifier, method simulator) on two datasets."""
    rows = []
    for i, (base, method, dataset) in enumerate(
        itertools.product(
            runs.BASE_QUANTIFIERS, runs.METHOD_SIMULATORS, ("haberman", "wine")
        )
    ):
        rows.append(
            {
                "base_quantifier": base,
                "method_simulator": method,
                "dataset": dataset,
                "target_prevalence": 0.4,
                "true_prevalence": 0.42,
                "estimated_prevalence": 0.42 + 0.001 * i,
                "repetition": 1,
            }
        )
    return pd.DataFrame(rows, columns=list(runs.columns_for(runs.REAL_DATA)))


@pytest.fixture
def measure_ablation_runs():
    """One run per (base quantifier, method simulator, measure)."""
    rows = []
    for i, (base, method, measure) in enumerate(
        itertools.product(runs.BASE_QUANTIFIERS, runs.METHOD_SIMULATORS, runs.MEASURES)
    ):
        rows.append(
            {
                "base_quantifier": base,
                "method_simulator": method,
                "measure": measure,
                "reference_simulator": runs.UNIFORM,
                "reference_merging_factor": 0.5,
                "bag_simulator": runs.UNIFORM,
                "bag_merging_factor": 0.5,
                "target_prevalence": 0.4,
                "true_prevalence": 0.42,
                "estimated_prevalence": 0.42 + 0.001 * i,
                "repetition": 1,
            }
        )
    return pd.DataFrame(rows, columns=list(runs.columns_for(runs.MEASURE_ABLATION)))
