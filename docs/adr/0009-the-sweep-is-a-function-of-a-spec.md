# The sweep is a function of a spec

The sweep used to open with `from variables import *` and read a 19-value
merging-factor range, seven prevalences, three registries and two sample sizes
out of the importing module's namespace. Nothing could run a smaller grid
without editing `variables.py`, so the only sweep anyone could execute was the
2.8M-run one, and the only thing a test could reach was a single grid cell
through a seven-positional-argument entry point.

`sweep.py` replaces it. A `SweepSpec` carries the whole experiment as one
value — the grid, the registries, the sample sizes, the seed — and
`sweep(spec)` returns the runs. `SYNTHETIC_SWEEP` is the published experiment
and `SMOKE_SWEEP` is the same code with a smaller argument, finishing in about
three seconds with every method still covered. `binary_experiment.py` and
`variables.py` are gone.

## The grid is a sequence of cells, not the ranges it is built from

`grid(simulators, merging_factors, target_prevalences)` builds the
cross-product, but the spec holds the cells it returns rather than the ranges.
The characterization grid is why: it runs matched simulator pairs only, at one
bag merging factor, which no cross-product of ranges describes. A spec that
held ranges would have forced the golden record onto a grid shaped for it.

## One estimator interface, one adapter per calling convention

mlquantify asks for a bag's prevalence in three different ways, and the sweep
used to select between them with two `if`s in the middle of the row-building
loop, where the branch and the row it produced were the same piece of code.
They are now three implementations of `Estimator.estimate(bag_scores)`:

- `BaselineEstimator` — `aggregate(bag)`. Reads no reference score set.
- `ReferenceEstimator` — `aggregate(bag, reference_scores, reference_labels)`.
- `MetaEstimator` — `aggregate(bag, reference_labels)`. Draws its own
  candidate score sets, so it is the one method that deliberately never sees
  the real reference scores.

`estimator_for` is the one place that knows which convention a
(base quantifier, method simulator) pair takes.

## A failing run is recorded, not skipped and not inherited

ADR-0001's second defect: a caught exception was logged and fell through to the
row-building code, where the local holding the estimate still held the
*previous* iteration's value, so the failing method was recorded under its own
name carrying another method's number. About 4.5% of the voided runs were
written that way.

A failing estimator now produces a run with a null `estimated_prevalence`.
Recorded rather than raised, because a sweep that stops at the first quantifier
to dislike a bag never finishes; recorded rather than dropped, because which
method failed on which bag is itself a result — the same reason ADR-0005 keeps
a bag Haberman cannot fill.

**The failure is reported from the finished frame, not from where it happens.**
`run_sweep` counts the missing runs and raises one `MissingRunWarning` per
method. Warning at the point of failure looks obvious and is wrong twice over:

- Cells run in loky workers. A warning raised in a worker lands on that
  worker's stderr and never enters the parent's warning machinery, so
  `n_jobs=-1` — the published path — was the one configuration in which nothing
  could be caught. Measured: `catch_warnings(record=True)` around a two-worker
  sweep captured zero of twelve.
- A warning per failure is a warning per run. On the full grid a method broken
  everywhere would emit millions of identical lines.

What that costs is the exception itself, which does not survive the worker
boundary. That is affordable only because of the estimator seam above: a run
names its cell and its method, and `estimator_for(spec, ...).estimate(bag)`
reproduces exactly that one estimate and raises. The old code printed a
traceback per failure into a joblib worker's stderr, which on a multi-hour
sweep nobody read either.

This supersedes issue #15, which asked for the row to be skipped. It was
written before the runs module could express a missing estimate.

The `strict` flag goes with it. It existed so tests could make the swallowed
error loud, which meant every test ran a path the sweep itself never takes —
and the untested path was the defective one. The characterization now asserts
that no run in the grid is missing, which is the same guarantee stated against
production behaviour.

## The baseline quantifier is recorded once

CC classifies the bag's members and counts them; it reads no reference score
set, so the sweep short-circuited it before the meta-quantifier — and then
recorded it anyway under all four method-simulator arms, four identical numbers
under three simulators that took no part in producing any of them. That is the
same mislabelling as the stale estimate, quieter. CC is now recorded once per
bag, under `none`, which is where `dashboard.py` already looked for it.

Because it now runs in exactly one arm, a spec that registers CC without the
`none` arm would produce no CC runs at all and say nothing about it — the quiet
version of the same defect. `SweepSpec` rejects that spec.

## Consequences

- The sweep writes through `runs.save`, in the schema of ADR-0003 and the
  vocabulary of CONTEXT.md. `results/results.csv` and the sweep's own column
  names are gone. Run it with `.venv/bin/python -m sweep`.
- The spec's registries are keyed by the names the runs module stores, so
  `runs.SIMULATOR_NAMES` and `runs.METHOD_SIMULATOR_NAMES` are deleted rather
  than maintained. There is no map between the sweep and the store that could
  go stale; a spec naming something `runs` cannot store raises when the spec is
  built, not after the sweep has run. This amends ADR-0008.
- The golden record in `tests/fixtures/plain_quantifier_runs.csv` is **not**
  regenerated. It is translated forward on the way in and compared on absolute
  error, and it replays bit-identical. That is not a weaker comparison than
  before: the error was the only measurement the record ever held. Its column
  names and the `MoSS_MN` / `Quadapt_Dir` spellings survive only there, in
  `tests/characterization.py`.
- `Method` — a base quantifier plus the method simulator wrapping it — is added
  to CONTEXT.md. It was already load-bearing in `runs.method_label` and in the
  estimator interface without being in the glossary.
- Merging factors are rounded to two decimals. `np.arange(0.05, 1.0, 0.05)`
  returns `0.15000000000000002` for its third step, and that value is now
  stored and grouped on by every renderer, which would have read it as a
  category of its own. No result predates the change and the golden record does
  not use the range.
- The published grid is 22,743 cells and 2,797,389 runs, down from 3.75M —
  ADR-0007 removed one method-simulator arm and the CC deduplication above
  removes three rows per bag.
- Seeding is unchanged and still wrong: every cell is seeded from the spec's
  one `seed`, so two cells sharing a simulator draw identical scores. That is
  #8, and deliberately not fixed here, so that the golden record replays
  bit-identical through the new seam and the port is the only thing being
  judged.
