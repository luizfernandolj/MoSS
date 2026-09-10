# Validate the re-run and add the distance-measure ablation

Issue #9 asked for four things once #6, #7 and #8 landed: the published grid
re-run at ten repetitions instead of three, its output checked against both of
ADR-0001's recorded defects before anything downstream trusts it, missing runs
reported explicitly, and a distance-measure ablation on a grid smaller than
the published one. All four land here.

## The measure is a spec field, forwarded only when set

`QuaDaptWithSimulator` and `QuaDaptOverCandidates` both take `measure` from
`QuaDapt.__init__`, and until now neither adapter passed one, so every run —
published or smoke — used the library's own default (`"topsoe"`) without that
being a decision anyone had made explicitly. `SweepSpec` gains a `measure`
field, `None` by default, forwarded to both adapters through
`_quadapt_kwargs`:

```python
def _quadapt_kwargs(measure):
    return {} if measure is None else {"measure": measure}
```

`None` means "upstream's default, unrestated" rather than this project's own
copy of `"topsoe"`. Writing the literal here would be exactly the trap
ADR-0007 caught: a restated upstream choice that nothing compares against the
original. `SYNTHETIC_SWEEP` keeps `measure=None` — the published grid does not
vary it, which is the acceptance criterion's "distance measure fixed" — and
only the ablation spec below sets it explicitly, once per entry of the new
`runs.MEASURES` vocabulary (`"topsoe"`, `"hellinger"`, `"probsymm"`, `"sord"`,
matching `utils.meta_quantifier`'s own dispatch).

## The measure ablation is a third table, not a column on the synthetic one

`runs.MEASURE_ABLATION` is `runs.SYNTHETIC`'s columns plus `measure`, its own
kind and its own Parquet file. A column on the synthetic table would be one
value repeated on every published row — ADR-0003's reasoning for keeping
merging factors off the real-data table applies here just as directly.

`sweep.run_measure_ablation(spec, measures=runs.MEASURES)` runs `spec` once
per measure via `dataclasses.replace(spec, measure=...)` and stamps the result
with the measure that produced it, reusing `run_sweep` unchanged — a method
simulator broken under one measure is still a missing run, not an exception.
`sweep.MEASURE_ABLATION_SWEEP` is the published ablation's spec: every
simulator at three merging factors and three prevalences (243 cells against
the published grid's 22,743), excluding the baseline quantifier and the
no-method-simulator arm — neither reads `measure`, so keeping them would
repeat the same rows once per entry of `runs.MEASURES` for nothing.

## Validating a produced sweep against both defect signatures

`sweep.validate(produced)` raises `DefectSignatureError` naming which of
ADR-0001's two defects a produced frame matches. Both read a
(method simulator, cell, repetition) group the same way, via the shared
`_group_columns` — every column but `base_quantifier` and
`estimated_prevalence`, so a measure-ablation frame's extra `measure` column
is one more thing a group shares rather than one more thing to enumerate:

- `stale_estimate_rows` — a run whose estimate exactly repeats the row
  immediately before it *within its own group*, in `run_cell`'s own base
  quantifier order. This is the second defect exactly: a caught exception
  falling through to a row that carries the *previous* method's number, one
  base quantifier after another inside a single cell and repetition. Scoped to
  the group on purpose, not compared across it: two rows either side of a
  group boundary describe different bags and different methods, so a
  coincidence there is not evidence of anything and comparing across it would
  only manufacture a false one.
- `collapsed_method_simulator_groups` — a group whose base quantifiers spread
  by no more than 0.01. This is the first defect exactly: every base
  quantifier under a meta-quantifier computing the same thing because the
  meta-quantifier never consulted it. A group of one base quantifier is
  excluded — nothing else ran there to disagree with it.

Both are already structurally impossible under the current code —
`_estimate_or_missing` never carries a stale value, and the base quantifier
produces every meta-quantifier estimate (ADR-0009, ADR-0010) — so `validate`
is a regression guard for the re-run's own output, not a proof about today's
code. That proof is the existing fault-injection tests
(`test_a_failing_run_never_carries_another_method_s_estimate`,
`test_base_quantifiers_in_a_cell_do_not_collapse_to_one_estimate`).

### Genuine ties

Run against the smoke sweep as written, `stale_estimate_rows` raised on real
output: TMS and TMS2 tied at 0.402273, and TAC and TX at 0.39895 and again at
0.39267 — three ties in 51 rows, none of them the defect. All three are base
quantifiers under `mlquantify.counting.ThresholdAdjustment`: MS2 falls back to
MS's own threshold set whenever nothing clears its reliability filter (the "No
cases satisfy |TPR - FPR| > 0.25" warning, which fired during exactly this
run), and TAC's fixed threshold and TX's crossing point can select the same
point from a small reference's sparse candidate set. Two base quantifiers in
that family tying is therefore excluded as a genuine tie, the same as a tie at
0.0 or 1.0 where the bag leaves nothing to disagree about. `validate` now
passes the smoke sweep clean, and does so as its own test
(`test_validate_passes_the_smoke_sweep`) rather than by construction.

## `validate_and_save` is the one sequence, not two copies of it

`_main` has two branches — the published sweep and the ablation — and both
need the same three steps in the same order: validate, report how many runs
are missing, then save. `sweep.validate_and_save(produced, kind, root)` is
that sequence, called once from each branch rather than written out twice,
which is what would have let the two drift out of step (say, one branch
saving before validating) with nothing to catch it. It is tested directly —
that a defective frame is refused before anything reaches `runs.save`, proved
by checking no file exists afterwards rather than by the raise alone; that the
missing-run count reaches stdout; and that a clean frame is saved through
`runs.save` and nothing else.

## Consequences

- The published sweep grows from 3,479,679 runs at three repetitions to
  11,598,930 at ten. It was not re-run as part of this change: the published
  re-run this ADR makes possible is a separate, long-running invocation
  (`.venv/bin/python -m sweep`), and this repository has never committed a
  results Parquet file — every number in ADR-0006, ADR-0009, ADR-0010 and
  ADR-0011 came from running the sweep locally and reporting what it measured,
  not from a file in the tree.
- `.venv/bin/python -m sweep --ablation` runs the published ablation and saves
  it under `runs.MEASURE_ABLATION`, through `validate_and_save`, the same as
  the published sweep.
- `runs.KINDS` has three members where it had two. Nothing that iterates over
  it generically existed before this change; `dashboard.py` and
  `export_grid_matplotlib.py` both name `runs.SYNTHETIC` explicitly and are
  unaffected.
- Which distance measure the ablation finds best is not decided here. That is
  a question for whoever reads the ablation's output, not for the code that
  produces it.
