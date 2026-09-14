# Experiments

The concrete grid values below are copied from the source constants for
reproducibility. **If you change a grid dimension in `sweep.py` or
`real_data.py`, update this file in the same change** — nothing here is
generated, so nothing re-syncs it automatically.

Source of truth: `sweep.SYNTHETIC_SWEEP`, `sweep.MEASURE_ABLATION_SWEEP` and
`sweep.MERGING_FACTORS`/`sweep.TARGET_PREVALENCES` in `sweep.py`;
`real_data.build_spec`, `real_data.build_multiclass_spec` and
`real_data.TARGET_PREVALENCES` in `real_data.py`. Terminology (simulator,
method, meta-quantifier, ...) is defined in [CONTEXT.md](../CONTEXT.md), not
repeated here.

## Synthetic / published sweep

Both the reference score set and the bags come from data simulators, so
score-distribution difficulty is a controlled variable. Produced by
`sweep.py` (no flags) — see [running.md](running.md).

- **Data simulators**: `uniform`, `mvn`, `dirichlet` (`utils/simulators.py`)
- **Grid**: every combination of reference simulator × bag simulator ×
  reference merging factor × bag merging factor × target prevalence
  (`sweep.grid`)
- **Merging factors**: 0.05 to 0.95 in steps of 0.05 (19 values)
- **Target prevalences**: 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 0.99 (7 values)
- **Reference size**: 2000, drawn balanced (prevalence 0.5)
- **Bag size**: 100
- **Repetitions**: 10
- **Seed**: 20260910
- **Distance measure**: unset — falls through to mlquantify's own default
  (fixed for the whole grid; which measure estimates best is the
  measure-ablation's question, not this one's —
  [ADR-0012](adr/0012-validate-the-re-run-and-add-the-measure-ablation.md))
- **Base quantifiers** (11): `DyS`, `HDy`, `SORD`, `SMM`, `TAC`, `TX`,
  `T50`, `TMAX`, `TMS`, `TMS2`, and the baseline `CC`
- **Method simulators / arms**: `uniform`, `mvn`, `dirichlet` (one
  meta-quantifier arm per simulator), `all` (a meta-quantifier choosing
  among all three simulators' candidates plus the real reference scores —
  [ADR-0010](adr/0010-restore-the-dropped-arm-as-a-candidate-search.md)),
  and `none` (no meta-quantifier — the base quantifier reads the real
  reference directly; the only arm `CC` runs under)

The published figures ([results.md](results.md)) draw a fixed subset of
this grid, not all of it: base quantifiers `TAC`, `TMAX`, `T50`, `HDy`,
`TX`, `TMS`, `DyS`, `SMM`; method simulators `uniform`, `mvn`, `dirichlet`
(the `all` arm is in the runs but left out of the figures by choice);
reference simulator `uniform` only; reference merging factors 0.25, 0.5,
0.75 (`export_grid_matplotlib.py`, `grid.py`).

## Measure-ablation

The same three data simulators on a smaller grid, run once per distance
measure to ask which measure a meta-quantifier's mixture search should
minimise — a question the published sweep above holds fixed and does not
ask. Produced by `sweep.py --ablation`.

- **Grid**: every simulator at merging factors 0.1, 0.5, 0.9 and
  prevalences 0.2, 0.5, 0.8 (`sweep.MEASURE_ABLATION_GRID`)
- **Reference size**: 400
- **Bag size**: 100
- **Repetitions**: 3
- **Seed**: 20260910
- **Measures varied**: `topsoe`, `hellinger`, `probsymm`, `sord`
  (`runs.MEASURES`)
- **Base quantifiers**: the same 11 minus `CC` (the baseline reads no
  meta-quantifier, so running it once per measure would only repeat the
  same rows)
- **Method simulators**: the same arms minus `none`, for the same reason

## Real-data, binary

The reference score set comes from a classifier's cross-validated scores
and bags are drawn from a real dataset under a protocol, rather than from a
data simulator. Produced by `real_data.py` (no flags, or `--dataset <name>`
for one dataset) — see [running.md](running.md).

- **Datasets** (8): `mushroom`, `banknote_authentication`,
  `haberman_survival`, `pima_diabetes`, `electricity_elec2`, `airlines`,
  `miniboone`, `online_news_popularity`
- **Classifier**: `RandomForestClassifier`, scored out-of-fold via 10-fold
  stratified cross-validation (never refit on the whole pool — the
  reference is never the classifier's own training scores,
  [ADR-0005](adr/0005-real-data-experiment-design.md))
- **Pool cap**: ~50,000 instances per dataset
- **Target prevalences**: 21 values bracketing [0, 1] (0.01, then 19 evenly
  spaced points from `linspace(0, 1, 21)`, then 0.99)
- **Bag size**: 100
- **Repetitions**: 10
- **Seed**: 20260911 (as invoked from `real_data.py`'s `_main`)
- **Base quantifiers / method simulators**: the same registries as the
  synthetic sweep (`sweep.BASE_QUANTIFIERS`, `sweep.METHOD_SIMULATORS`)

A cell a dataset's pool cannot supply — e.g. Haberman has 81 minority
instances and cannot fill a size-100 bag above 80% positive — is recorded
as an explicit missing run rather than skipped or padded
([ADR-0005](adr/0005-real-data-experiment-design.md)).

## Real-data, multiclass

The same pipeline as above, on datasets with more than two classes
([#14](https://github.com/luizfernandolj/MoSS/issues/14)) — evidence that
the simplex simulators (MVN, Dirichlet) do work the binary uniform
simulator cannot attempt. Produced by `real_data.py --multiclass`.

- **Datasets** (2): `dry_bean` (7 classes), `yeast` (10 classes) — chosen
  for different class counts rather than matching the binary table's
  coverage claim
- **Target prevalences**: 21 prevalence vectors per dataset, drawn uniformly
  from that dataset's class-count simplex (a Dirichlet(1, ..., 1) draw),
  seeded per dataset so two datasets sharing a class count still draw
  different grids
- **Bag size, repetitions, seed, classifier, base quantifiers, method
  simulators**: same as the binary real-data spec above
