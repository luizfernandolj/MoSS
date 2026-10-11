"""Multiclass pools read from CSVs on disk rather than fetched (#14).

``real_data.fetch_pool`` names a dataset through a *fetcher*: a callable taking
``data_home`` and ``return_X_y`` and giving back features and labels, which is
the signature mlquantify's own fetchers already have. Nothing downstream of
:class:`real_data.Pool` cares where those arrays came from, so a dataset that
lives as a CSV under ``data/`` needs no new seam in the pipeline — only a
loader shaped like a fetcher, which :func:`fetcher` builds and
:data:`MULTICLASS_FETCHERS` collects one of per dataset.

What a loader does beyond ``pd.read_csv`` is deliberately almost nothing.
``real_data._numeric_features`` already ordinal-encodes categorical columns and
median-imputes numeric ones, and ``real_data.POOL_CAP`` already bounds a large
dataset's cross-validation cost, so neither belongs here in a second copy. The
one thing it cannot do for itself is know which of a CSV's columns must not be
a feature, because that is a fact about the dataset and not about its dtypes:
see :attr:`LocalDataset.drop`.

Not every CSV under ``data/`` is registered. A dataset whose smallest class
falls under ``real_data.MIN_CLASS_SIZE`` would be dropped by
``real_data._above_class_floor`` before its cells were ever built (#18), so
registering it would only promise a table row that never arrives; the module
constants below say which were left out and why. ``data/schumacher/`` is almost
entirely regression targets and contributes one dataset; ``data/uci/
uci_42123_article_influence.csv`` is a regression target wearing a ``class``
column (3169 distinct values) and contributes none.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import pandas as pd

#: Where the CSVs live, anchored to this file for the same reason
#: ``runs.ROOT`` and ``real_data.DATA_HOME`` are: a caller launched from
#: another directory must still find them.
CSV_ROOT = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class LocalDataset:
    """One CSV under ``data/``, and what it takes to read it as X and y."""

    #: Relative to :data:`CSV_ROOT`.
    path: str

    #: The label column's name. Spelled per dataset rather than guessed: the
    #: collections disagree (``class`` in most, ``Class`` in ``data/ours/
    #: HAR.csv``), and a guess that happened to be right today would fail
    #: silently rather than loudly on the next CSV added.
    label: str = "class"

    #: Columns that must not become features. Only identifiers and leaks are
    #: listed — a per-row id (``case_id``, ``obj_ID``) is noise a Random Forest
    #: can nonetheless split on, free text at one value per row
    #: (``Track Name``) is the same thing in string form, and ``GPA`` on
    #: Student Performance is the continuous quantity its ``class`` column is
    #: the letter grade *of*, so leaving it in would hand the classifier the
    #: answer. Ordinary categorical columns are not listed: ``real_data._
    #: numeric_features`` encodes them, and dropping a real feature here would
    #: quietly change what the dataset is.
    drop: Tuple[str, ...] = field(default_factory=tuple)


#: The multiclass datasets registered from ``data/`` (#14), keyed by the name
#: their runs are stored under — lowercase and underscored, the convention
#: ``real_data.DATASET_FETCHERS`` already follows, rather than each
#: collection's own spelling of its filename.
#:
#: Every one clears ``real_data.MIN_CLASS_SIZE`` on its smallest class, so
#: every one actually reaches the grid. Deliberately absent:
#:
#: * Under the class floor, so ``_above_class_floor`` would drop them (#18):
#:   ``ours/Land-use`` (20), ``ours/Nursery`` (2), ``openml/fars`` (9),
#:   ``openml/dataset_313_spectrometer`` (1),
#:   ``openml/dataset_4552_BachChoralHarmony`` (1), the three
#:   ``openml/one-hundred-plants`` (15), ``openml/!dataset_372_internet_usage``
#:   (6), ``quapy_data/wine-quality`` (5), ``kaggle/zoo`` (10), ``kaggle/zoo2``
#:   (6), ``kaggle/zoo3`` (4), ``kaggle/cirrhosis`` (21),
#:   ``kaggle/predictive_maintenance`` (18).
#: * ``quapy_data/letter`` (26 classes) and ``ours/Walking`` (22): the
#:   one-vs-rest decomposition (``utils/meta_quantifier.py``) fits one base
#:   quantifier per class, so either costs roughly twice the whole rest of
#:   this table put together for one more point of simplex dimension.
#: * ``quapy_data/dry-bean``: the same UCI dataset
#:   ``real_data.MULTICLASS_DATASET_FETCHERS`` already fetches as ``dry_bean``.
#:   Registered twice it would enter ``stats.py``'s Friedman and Nemenyi tests
#:   as two datasets, double-weighting one dataset's opinion of the methods.
#: * The five ``openml/amazon-commerce-reviews`` subsets: five seeds of one
#:   dataset, which is the same double-weighting five times over.
MULTICLASS_DATASETS = {
    # 11 classes
    "healthcare": LocalDataset(
        "kaggle/healthcare.csv", drop=("case_id", "patientid")
    ),
    "music_genre": LocalDataset(
        "kaggle/music_genre.csv", drop=("Artist Name", "Track Name")
    ),
    # 10 classes
    "mfeat": LocalDataset("ours/Mfeat.csv"),
    "fashion_mnist": LocalDataset("kaggle/fashion-mnist.csv"),
    "digits": LocalDataset("quapy_data/digits.csv"),
    # 7 classes
    "covertype": LocalDataset("ours/Covertype.csv"),
    "fabert": LocalDataset("openml/fabert.csv"),
    # 6 classes
    "dermatology": LocalDataset("ours/Dermatology.csv"),
    "har": LocalDataset("ours/HAR.csv", label="Class"),
    "mosquitoes": LocalDataset(
        "ours/Mosquitoes.csv", drop=("sensor_id", "file", "time_elapsed")
    ),
    "satimage": LocalDataset("ours/Satimage.csv"),
    # 5 classes
    "phishing_url": LocalDataset("ours/PhishingURL.csv"),
    "microaggregation2": LocalDataset("openml/microaggregation2.csv"),
    "student_performance": LocalDataset(
        "kaggle/Student_performance_data.csv", drop=("StudentID", "GPA")
    ),
    # 4 classes
    "customer_segmentation": LocalDataset(
        "kaggle/customer_segmentation.csv", drop=("ID",)
    ),
    #: Not mlquantify's ``yeast``: four classes to that one's ten, the
    #: collapsed version Schumacher et al.'s ordinal benchmark ships. Named
    #: apart so a reader cannot take the two for one dataset measured twice.
    "yeast_schumacher": LocalDataset("schumacher/yeast_data.csv"),
    # 3 classes
    "mfeat_icdm21": LocalDataset("ours/Mfeat_icdm21.csv"),
    "star_classification": LocalDataset(
        "kaggle/star_classification.csv",
        drop=(
            "obj_ID",
            "run_ID",
            "rerun_ID",
            "cam_col",
            "field_ID",
            "spec_obj_ID",
            "plate",
            "MJD",
            "fiber_ID",
        ),
    ),
    "academic_success": LocalDataset("quapy_data/academic-success.csv"),
}


def load(dataset, *, csv_root=CSV_ROOT):
    """``dataset`` read off disk as ``(X, y)``.

    ``low_memory=False`` because a column of mixed dtypes read in chunks comes
    back as ``object`` with the chunk boundary deciding which rows are strings,
    which would make the features a function of the file's size rather than of
    its contents.
    """
    path = Path(csv_root) / dataset.path
    frame = pd.read_csv(path, low_memory=False)

    missing = [column for column in (dataset.label, *dataset.drop) if column not in frame]
    if missing:
        raise ValueError(f"{path} has no column {missing!r}")

    y = frame[dataset.label]
    X = frame.drop(columns=[dataset.label, *dataset.drop])
    return X, y


def fetcher(dataset, *, csv_root=CSV_ROOT):
    """``dataset`` as a callable with an mlquantify fetcher's signature.

    ``data_home`` is accepted and ignored: a CSV already on disk has nothing to
    cache. It is in the signature anyway so that ``real_data.fetch_pool`` can
    call every fetcher it knows the same way, whether the dataset behind it is
    downloaded or local.
    """

    def fetch(data_home=None, return_X_y=True):
        if not return_X_y:
            raise ValueError("local datasets are only available as (X, y)")
        return load(dataset, csv_root=csv_root)

    fetch.__name__ = f"fetch_{dataset.path}"
    return fetch


#: One fetcher per entry of :data:`MULTICLASS_DATASETS`, ready to merge into
#: ``real_data.MULTICLASS_DATASET_FETCHERS``.
MULTICLASS_FETCHERS = {
    name: fetcher(dataset) for name, dataset in MULTICLASS_DATASETS.items()
}
