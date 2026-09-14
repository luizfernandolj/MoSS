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
