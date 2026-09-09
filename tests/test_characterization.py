"""Replay the golden record of plain-quantifier runs.

Captured on the pinned mlquantify (see ADR-0001 and ADR-0006). If this fails,
a run that should have been untouched changed. That is a finding, not a
fixture to refresh.
"""

import pandas as pd
import pytest

from tests import characterization
from variables import QUANTIFIERS


@pytest.fixture(scope="module")
def replayed():
    return characterization.run_grid()


def test_replays_the_committed_runs_exactly(replayed):
    expected = characterization.load_fixture()

    pd.testing.assert_frame_equal(replayed, expected, check_exact=True)


def test_fixture_covers_every_base_quantifier():
    expected = characterization.load_fixture()

    assert set(expected["Quantifier"]) == set(QUANTIFIERS)


def test_fixture_records_no_meta_quantifier_runs():
    expected = characterization.load_fixture()

    assert set(expected["Quadapt_Variant"]) == {"None"}
