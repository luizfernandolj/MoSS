# Setup

## Environment

No `pyproject.toml`, no pinned interpreter file — the tracked `.venv` this
project was developed against runs Python 3.11. Any recent CPython 3.11+
should work.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Every command in [running.md](running.md) and [testing.md](testing.md) is
invoked through `.venv/bin/python` (or `.venv/bin/streamlit`) rather than an
activated shell, so a script always runs against these pinned dependencies
regardless of what else is on `PATH`.

`requirements.txt` pins `mlquantify==0.5.1` deliberately — [ADR-0001](adr/0001-pin-mlquantify-0-5-1-and-void-earlier-results.md)
records why an earlier version's results are void, and every run since is
produced against this exact pin.

## Verify the install

```
.venv/bin/python -m pytest
```

See [testing.md](testing.md) for what the suite covers and how long it takes.

## Network access

`sweep.py` (the synthetic experiment) needs no network: its data comes from
the score simulators in `utils/simulators.py`.

`real_data.py` does, the first time it runs: it fetches each dataset through
`mlquantify.datasets`' own fetchers and caches the result under
`results/datasets/` (git-ignored — regenerated locally, not committed).
Later runs against an already-fetched dataset read the cache and need no
network. Fetching and cross-validating a classifier for all eight binary
datasets takes a while on first run; `--dataset <name>` (see
[running.md](running.md)) restricts a run to one dataset while iterating.

## The CSVs under `data/`

`real_data.py --multiclass` reads most of its datasets as CSVs under `data/`
rather than fetching them (`utils/local_datasets.py`, which lists them). That
directory is git-ignored — about a gigabyte of raw CSVs, the same reason
`results/datasets/` is not committed — so it has to be put in place out of
band, by copying it from wherever it was assembled. The fetch helpers at its
top level *are* committed: `data/download.py` fetches the QuaPy collection
under `data/quapy_data/`, and `data/openml_datasets.py`,
`data/uci_datasets.py` and `data/uci2.py` fetch the others.

Each collection subdirectory also carries a `preprocess.py` — not committed,
and not what the pipeline uses either. Those are the per-dataset cleaning
functions from the project the CSVs were assembled in, and they mix
subsampling and feature selection into the same step.
`utils/local_datasets.py` instead names only the columns that must not become
features — identifiers and target leaks — and leaves encoding, imputation and
the pool cap to `real_data.py`, which already does all three.

A dataset whose CSV is missing is a loud `FileNotFoundError` naming the path,
not a silently shorter table.
