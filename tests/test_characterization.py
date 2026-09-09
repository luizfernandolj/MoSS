"""Replay the golden record of plain-quantifier runs.

Captured on the pinned mlquantify (see ADR-0001 and ADR-0006). If this fails,
a run that should have been untouched changed. That is a finding, not a
fixture to refresh.
"""

import pandas as pd

import pytest

import sweep
from tests import characterization


@pytest.fixture(scope="module")
def replayed():
    return characterization.run_grid()


def test_replays_the_committed_runs_exactly(replayed):
    expected = characterization.load_fixture()

    pd.testing.assert_frame_equal(replayed, expected, check_exact=True)


def test_every_run_in_the_grid_produced_an_estimate(replayed):
    # The sweep records a failure as a missing run rather than raising, so a
    # method that quietly stopped estimating would otherwise reach the
    # comparison above as a NaN and read as a numeric drift.
    assert replayed["absolute_error"].notna().all()


def test_fixture_covers_every_base_quantifier():
    expected = characterization.load_fixture()

    assert set(expected["base_quantifier"]) == set(sweep.BASE_QUANTIFIERS)


def test_fixture_records_no_meta_quantifier_runs():
    # Asserted against the record as written, not against the grid that
    # replays it: the point is what is frozen in the file.
    assert set(characterization.read_fixture()["Quadapt_Variant"]) == {"None"}
