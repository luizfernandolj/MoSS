# Running the tests

```
.venv/bin/python -m pytest
```

That is the whole suite. It runs in about thirty-five seconds and needs no data
beyond what is committed. Most of that is the smoke sweep, which several tests
run: it doubled in cost when the arm that searches every candidate rejoined it
(ADR-0010), and the determinism tests below run it twice more on purpose.

## What is in it

**Characterization** (`tests/test_characterization.py`). A golden record of
plain-quantifier runs, replayed through the sweep seam. Meta-quantifier runs
are excluded on purpose: their behaviour changed across the 0.2.0 → 0.5.1
upgrade, and that change is the defect ADR-0001 records being fixed.

The fixture was first captured on 0.2.0 to prove the upgrade left plain runs
untouched. It did not — DyS and HDy changed, for the reasons ADR-0006 records —
so it was re-captured on 0.5.1 and now guards the pinned library against future
drift instead. It was re-captured once more when the seeding changed under it
(ADR-0011), which is also when it stopped being translated forward from the
pre-ADR-0009 schema: it is now in today's, and compares on both prevalences
rather than on absolute error alone.

There is no regeneration command, on purpose: if a characterization test fails,
a run that should have been untouched changed, and re-capturing is a decision
to argue for in an ADR rather than a command to reach for. Both re-captures
went that way, and the second one carried a measurement — across four base
seeds the old and new seeding overlap for all eleven quantifiers, pooled means
0.0999 and 0.1001.

**Sweep** (`tests/test_sweep.py`). That the grid is a parameter and not a
module global; that the spec's vocabulary is the one `runs` stores, so a sweep
cannot spend hours producing runs no reader can name; the calling convention
each of the four estimator adapters follows; where a seed comes from and what
it guarantees (see Determinism below); and the regression for ADR-0001's second
defect — a failing estimator records a missing run rather than the *previous*
method's number. Verified by reintroducing the defect: three tests fail.

**Meta-quantifier** (`tests/test_meta_quantifier.py`). That different base
quantifiers under one meta-quantifier produce different estimates — the
property whose absence went unnoticed across 3.75M runs — asserted twice, at
two seams that guard different things. At the sweep seam, on this project's own
simulators and registries. At the library seam, driving mlquantify's `QuaDapt`
directly, which is what keeps the pinned version honest: held fixed, 0.2.0
returns one identical estimate for all ten base quantifiers and 0.5.1 returns a
spread.

Plus a smoke test that `sweep.SMOKE_SWEEP`, which covers every method and base
quantifier, produces an estimate for all of them. Note what that smoke test has
to assert: a broken method simulator now reaches the results as missing runs
rather than as an exception, so "completes without raising" would no longer
catch it.

**Candidate search** (`tests/test_candidate_search.py`). The arm that chooses
among candidate score sets rather than committing to one method simulator
(ADR-0010): that the candidates are a *list* and not a sequence of mutations of
the estimator, that the real reference scores are among them, that the base
quantifier is handed the candidate matching the bag, and that the classes come
from the data rather than being 0 and 1. Which candidate wins is asserted in
both directions — a search that always returned the real reference would pass
one of those tests — and never by recomputing the distances the way the code
computes them: each pits a candidate whose mixture reproduces the bag exactly
against one whose mixtures are all the same distribution.

One test there guards a copy rather than a behaviour. The size and prevalence a
candidate is drawn at are literals inside mlquantify's own `aggregate`, with no
parameter to forward, so the arm restates them — and a restated upstream choice
that nothing compares against the original is the trap ADR-0007 caught. That
test drives the library's own meta-quantifier and records what it asks its
`MoSS` seam for.

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

A spec that carries a `seed` produces identical runs every time, and that is
asserted rather than assumed: `test_running_the_same_spec_twice_produces_
identical_runs` compares two whole sweeps, meta-quantifier arms included. Every
draw is seeded from the cell and the repetition it belongs to (ADR-0011), so
two further properties hold and are tested — the bags do not move when the
reference score set changes size, and `n_jobs` does not change the runs.

The seed is deliberately not a function of the base quantifier, which is what
lets the collapse regression stand at the sweep seam: ten quantifiers on one
bag with one set of candidates, so the 0.2.0 defect would show as one number
repeated ten times. Verified by imitating the defect — the spread goes from
0.27 to exactly 0.0.

What is left is the library's own `QuaDapt`, whose `MoSS` accepts a
`random_state`, documents it as unused, and draws from the legacy global
`np.random` ([coenlab/mlquantify#5][upstream]). That is reachable only by code
which does *not* replace the `MoSS` seam — in this project, exactly one test:
the library-seam half of `tests/test_meta_quantifier.py`, pinned with
`np.random.seed` because that is the only lever the library offers. It can drop
the pin when [#5][upstream] lands.

[upstream]: https://github.com/coenlab/mlquantify/issues/5
