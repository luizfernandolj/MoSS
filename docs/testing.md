# Running the tests

```
.venv/bin/python -m pytest
```

That is the whole suite. It runs in about fifteen seconds and needs no data
beyond what is committed.

## What is in it

**Characterization** (`tests/test_characterization.py`). A golden record of
plain-quantifier runs, replayed through the sweep entry point. Meta-quantifier
runs are excluded on purpose: their behaviour changed across the 0.2.0 → 0.5.1
upgrade, and that change is the defect ADR-0001 records being fixed.

The fixture was first captured on 0.2.0 to prove the upgrade left plain runs
untouched. It did not — DyS and HDy changed, for the reasons ADR-0006 records —
so it was re-captured on 0.5.1 and now guards the pinned library against future
drift instead.

If a characterization test fails, a run that should have been untouched
changed. Treat it as a finding, not as a fixture to refresh. The regeneration
command is in `tests/characterization.py`.

**Meta-quantifier** (`tests/test_meta_quantifier.py`). A regression test that
different base quantifiers under one meta-quantifier produce different
estimates — the property whose absence went unnoticed across 3.75M runs — and a
smoke test that a small sweep covering every method and base quantifier
completes without raising.

## Determinism

Tests seed the sweep's own data simulators through `run_experiment`'s
`random_state`.

The simulators *inside* a meta-quantifier are a different matter. This
project's overrides draw from `np.random.default_rng(None)` and cannot be
seeded at all (ADR-0004). That is not merely inconvenient: those unseeded draws
make estimates differ between base quantifiers even when the meta-quantifier
ignores them, so the defect and the fix produce indistinguishable spreads at
the sweep seam. The regression test therefore drives mlquantify's own
`QuaDapt` — whose `MoSS` still draws from the legacy global `np.random`, and so
can be pinned with `np.random.seed` — rather than the sweep. Held fixed that
way, 0.2.0 returns one identical estimate for all ten base quantifiers and
0.5.1 returns a spread.

Move that test onto the sweep seam once ADR-0004's seeding lands.
