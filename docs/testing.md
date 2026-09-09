# Running the tests

```
.venv/bin/python -m pytest
```

That is the whole suite. It runs in about twenty seconds and needs no data
beyond what is committed.

## What is in it

**Characterization** (`tests/test_characterization.py`). A golden record of
plain-quantifier runs, replayed through the sweep seam. Meta-quantifier runs
are excluded on purpose: their behaviour changed across the 0.2.0 → 0.5.1
upgrade, and that change is the defect ADR-0001 records being fixed.

The fixture was first captured on 0.2.0 to prove the upgrade left plain runs
untouched. It did not — DyS and HDy changed, for the reasons ADR-0006 records —
so it was re-captured on 0.5.1 and now guards the pinned library against future
drift instead.

It is still in the schema the sweep returned when it was captured, and it stays
that way. `tests/characterization.py` translates it forward and compares on
absolute error, the one quantity both schemas express (ADR-0009). There is no
regeneration command, on purpose: if a characterization test fails, a run that
should have been untouched changed, and re-capturing is a decision to argue for
in an ADR rather than a command to reach for.

**Sweep** (`tests/test_sweep.py`). That the grid is a parameter and not a
module global; that the spec's vocabulary is the one `runs` stores, so a sweep
cannot spend hours producing runs no reader can name; the calling convention
each of the three estimator adapters follows; and the regression for ADR-0001's
second defect — a failing estimator records a missing run rather than the
*previous* method's number. Verified by reintroducing the defect: three tests
fail.

**Meta-quantifier** (`tests/test_meta_quantifier.py`). A regression test that
different base quantifiers under one meta-quantifier produce different
estimates — the property whose absence went unnoticed across 3.75M runs — and a
smoke test that `sweep.SMOKE_SWEEP`, which covers every method and base
quantifier, produces an estimate for all of them. Note what that smoke test has
to assert: a broken method simulator now reaches the results as missing runs
rather than as an exception, so "completes without raising" would no longer
catch it.

**Simulators** (`tests/test_simulators.py`). The contract all three score
simulators meet, asserted on all three: rows sum to one, class counts match the
requested prevalence, a scalar prevalence means the same as the vector it
stands for, and the call mlquantify's meta-quantifier makes is accepted
verbatim. These are what a per-simulator signature had no way to state — see
ADR-0008 for the off-by-one they would have caught.

**Runs module** (`tests/test_runs.py`). The schema of the two tables, that
they share an estimator block (ADR-0003), that runs survive a Parquet round
trip without a caller ever naming a file, and that the vocabulary `runs`
publishes matches the sweep's registries exactly. A run the module refuses to
save is one a reader would have had to guess about.

**Renderers** (`tests/test_renderers.py`). Every method name the figure export
filters on exists in the runs it filters. This is the regression for the
defect the runs module was built to end: the export selected base
quantifiers named `"X"` and `"MS"` where the sweep records `"TX"` and `"TMS"`,
so two of the grid's eight methods were never drawn and the figure rendered
cleanly with six lines.

Both run against a small table built in `tests/conftest.py` from `runs`' own
vocabulary — never against a full results file, which is what made the old
readers impossible to test at all. Building it from the vocabulary is the
point: a rename reaches the fixture too, so a filter left behind fails.

## Determinism

Tests seed the sweep's own data simulators through the spec's `seed`.

The simulators *inside* a meta-quantifier are a different matter. They now
accept a `random_state` and honour it, and the meta-quantifier forwards
whatever it is given — but mlquantify never gives it one, calling its `MoSS`
seam with no seed at all (ADR-0004). So the draws are still unseeded, and that
is not merely inconvenient: unseeded draws make estimates differ between base
quantifiers even when the meta-quantifier ignores them, so the defect and the
fix produce indistinguishable spreads at the sweep seam. The regression test
therefore drives mlquantify's own `QuaDapt` — whose `MoSS` still draws from the
legacy global `np.random`, and so can be pinned with `np.random.seed` — rather
than the sweep. Held fixed that way, 0.2.0 returns one identical estimate for
all ten base quantifiers and 0.5.1 returns a spread.

What is left of ADR-0004's gap is entirely upstream. Move that test onto the
sweep seam once the library passes the seed through.
