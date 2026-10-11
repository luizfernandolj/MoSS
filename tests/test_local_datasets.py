"""Multiclass pools read from CSVs under ``data/`` (#14).

Fabricated CSVs in ``tmp_path`` throughout, for the same reason the rest of
this suite fabricates pools (docs/testing.md): the real CSVs are about a
gigabyte and are not committed, so a suite that needed them would not run.
:func:`utils.local_datasets.load` and :func:`~utils.local_datasets.fetcher`
both take ``csv_root``, which is what makes that substitution possible.

The one test that does read the real files —
:func:`test_every_registered_column_exists_in_its_csv` — reads five rows per
dataset for its header alone and skips when ``data/`` is absent. It is here
because a typo in a ``drop`` list is otherwise found by the server run that
was supposed to produce the published table, several hours in.
"""

import pandas as pd
import pytest

import real_data
from utils import local_datasets


#: Rows per class in the fixture CSV. Ten, not two: a pool is split 60/40
#: into a reference frame and a bag frame (ADR-0013), and the reference frame
#: is then cross-validated within itself, so each class needs enough rows to
#: survive both cuts and still stratify into folds.
ROWS_PER_CLASS = 10


@pytest.fixture
def csv_root(tmp_path):
    """A ``data/``-shaped directory holding one small CSV."""
    grades = [grade for grade in "xyz" for _ in range(ROWS_PER_CLASS)]
    frame = pd.DataFrame(
        {
            "row_id": range(len(grades)),
            "feature": [i / len(grades) for i in range(len(grades))],
            "category": [("a", "b")[i % 2] for i in range(len(grades))],
            "grade": grades,
        }
    )
    (tmp_path / "collection").mkdir()
    frame.to_csv(tmp_path / "collection" / "small.csv", index=False)
    return tmp_path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_loading_separates_the_label_column_from_the_features(csv_root):
    dataset = local_datasets.LocalDataset("collection/small.csv", label="grade")

    X, y = local_datasets.load(dataset, csv_root=csv_root)

    assert list(y) == [grade for grade in "xyz" for _ in range(ROWS_PER_CLASS)]
    assert "grade" not in X.columns


def test_loading_drops_the_columns_that_must_not_be_features(csv_root):
    # An identifier or a leak is a fact about the dataset, not about its
    # dtypes, so it is the one thing ``real_data._numeric_features`` cannot
    # work out for itself.
    dataset = local_datasets.LocalDataset(
        "collection/small.csv", label="grade", drop=("row_id",)
    )

    X, _ = local_datasets.load(dataset, csv_root=csv_root)

    assert list(X.columns) == ["feature", "category"]


def test_loading_keeps_categorical_columns(csv_root):
    # Dropped here they would silently stop being part of the dataset;
    # ``real_data._numeric_features`` ordinal-encodes them instead.
    dataset = local_datasets.LocalDataset("collection/small.csv", label="grade")

    X, _ = local_datasets.load(dataset, csv_root=csv_root)

    assert "category" in X.columns


def test_a_column_the_csv_does_not_have_is_refused_by_name(csv_root):
    dataset = local_datasets.LocalDataset(
        "collection/small.csv", label="grade", drop=("no_such_column",)
    )

    with pytest.raises(ValueError, match="no_such_column"):
        local_datasets.load(dataset, csv_root=csv_root)


def test_a_label_column_the_csv_does_not_have_is_refused_by_name(csv_root):
    dataset = local_datasets.LocalDataset("collection/small.csv", label="klass")

    with pytest.raises(ValueError, match="klass"):
        local_datasets.load(dataset, csv_root=csv_root)


# ---------------------------------------------------------------------------
# The fetcher shape — what lets ``real_data.fetch_pool`` stay unchanged
# ---------------------------------------------------------------------------


def test_a_fetcher_is_callable_the_way_mlquantify_s_own_fetchers_are(csv_root):
    dataset = local_datasets.LocalDataset("collection/small.csv", label="grade")

    X, y = local_datasets.fetcher(dataset, csv_root=csv_root)(
        data_home="ignored", return_X_y=True
    )

    assert len(X) == len(y) == 3 * ROWS_PER_CLASS


def test_a_fetcher_refuses_the_bunch_shape_it_cannot_build(csv_root):
    dataset = local_datasets.LocalDataset("collection/small.csv", label="grade")

    with pytest.raises(ValueError, match="X, y"):
        local_datasets.fetcher(dataset, csv_root=csv_root)(return_X_y=False)


def test_a_local_pool_is_built_through_the_same_fetch_pool_seam(csv_root, monkeypatch):
    # The point of the fetcher shape: ``fetch_pool`` needs no branch for a
    # dataset that came off disk rather than off the network.
    dataset = local_datasets.LocalDataset("collection/small.csv", label="grade")
    monkeypatch.setitem(
        real_data._ALL_DATASET_FETCHERS,
        "small",
        local_datasets.fetcher(dataset, csv_root=csv_root),
    )

    pool = real_data.fetch_pool("small", cv_folds=2, random_state=0)

    assert pool.dataset == "small"
    assert len(pool.classes) == 3
    # Two disjoint frames (ADR-0013), together the whole CSV.
    assert pool.scores.shape == (len(pool.labels), 3)
    assert pool.reference_scores.shape == (len(pool.reference_labels), 3)
    assert len(pool.labels) + len(pool.reference_labels) == 3 * ROWS_PER_CLASS


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_every_local_dataset_is_registered_as_a_multiclass_fetcher():
    assert set(local_datasets.MULTICLASS_FETCHERS) == set(
        local_datasets.MULTICLASS_DATASETS
    )


def test_the_local_datasets_reach_the_multiclass_table():
    assert set(local_datasets.MULTICLASS_DATASETS) <= set(
        real_data.MULTICLASS_DATASET_FETCHERS
    )


def test_no_csv_is_registered_twice_under_two_names():
    # Registered twice, one dataset would enter ``stats.py``'s Friedman and
    # Nemenyi tests as two and double-weight its own opinion of the methods.
    paths = [dataset.path for dataset in local_datasets.MULTICLASS_DATASETS.values()]
    assert len(paths) == len(set(paths))


def test_no_local_dataset_shadows_a_binary_one():
    assert set(local_datasets.MULTICLASS_DATASETS).isdisjoint(real_data.DATASET_FETCHERS)


def test_dry_bean_is_not_registered_twice():
    # ``quapy_data/dry-bean.csv`` is the same UCI dataset mlquantify fetches
    # as ``dry_bean``, which stays the one registered copy of it.
    paths = {dataset.path for dataset in local_datasets.MULTICLASS_DATASETS.values()}
    assert "quapy_data/dry-bean.csv" not in paths


@pytest.mark.skipif(
    not local_datasets.CSV_ROOT.exists(),
    reason="the CSVs under data/ are about a gigabyte and are not committed",
)
@pytest.mark.parametrize("name", sorted(local_datasets.MULTICLASS_DATASETS))
def test_every_registered_column_exists_in_its_csv(name):
    dataset = local_datasets.MULTICLASS_DATASETS[name]
    path = local_datasets.CSV_ROOT / dataset.path
    if not path.exists():
        pytest.skip(f"{dataset.path} is not on this machine")

    # Five rows, for the header alone: this guards the names in the registry,
    # not anything about the data behind them.
    columns = set(pd.read_csv(path, nrows=5, low_memory=False).columns)

    assert {dataset.label, *dataset.drop} <= columns
