# Running the tests

```
.venv/bin/python -m pytest
```

That is the whole suite. It runs in a few seconds and needs no data beyond
what is committed.

## What is in it

**Characterization** (`tests/test_characterization.py`). A golden record of
plain-quantifier runs, captured on `mlquantify==0.2.0` before the upgrade to
0.5.1 and replayed through the sweep entry point. Plain runs take a code path
the upgrade does not touch, so they must reproduce exactly. Meta-quantifier
runs are excluded on purpose: their behaviour changes across the upgrade, and
that change is the defect ADR-0001 records being fixed.

If a characterization test fails, a run that should have been untouched
changed. Treat it as a finding, not as a fixture to refresh. The regeneration
command is in `tests/characterization.py`, and running it on any version other
than 0.2.0 destroys the baseline.

**Meta-quantifier** (`tests/test_meta_quantifier.py`). A regression test that
different base quantifiers under one meta-quantifier produce different
estimates — the property whose absence went unnoticed across 3.75M runs — and
a smoke test that a small sweep covering every method and base quantifier
completes without raising.

## Determinism

Tests seed the sweep's own data simulators through `run_experiment`'s
`random_state`. The method simulators inside a meta-quantifier are not
reachable from here, so meta-quantifier estimates stay non-deterministic until
ADR-0004's upstream gap closes; the tests that touch them assert on properties
rather than on numbers.
