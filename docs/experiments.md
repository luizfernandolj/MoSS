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
- **Pool cap**: ~50,000 instances per dataset
- **Frames**: each pool is split 60/40, stratified, into a *reference frame*
  and a *bag frame* that share no instance
  ([ADR-0013](adr/0013-disjoint-reference-and-bag-frames.md)). Bags are drawn
  only from the bag frame; the reference frame is the distribution every
  method matches them against.
- **Classifier**: `RandomForestClassifier`. The reference frame is scored
  out-of-fold via 10-fold stratified cross-validation *within itself*, so the
  reference is never the classifier's own training scores
  ([ADR-0005](adr/0005-real-data-experiment-design.md)). The bag frame is
  scored by one forest fitted on the whole reference frame, which has never
  seen it.
- **Target prevalences**: 21 values bracketing [0, 1] (0.01, then 19 evenly
  spaced points from `linspace(0, 1, 21)`, then 0.99)
- **Bag size**: 100, 500, 1000, 5000 (`sweep.BAG_SIZES`, swept via
  `real_data.run_bag_size_sweep`, #16) — the published figures in
  [results.md](results.md) other than the bag-size sensitivity ones are the
  bag_size=100 slice alone
- **Repetitions**: 10
- **Seed**: 20260911 (as invoked from `real_data.py`'s `_main`)
- **Base quantifiers / method simulators**: the same registries as the
  synthetic sweep (`sweep.BASE_QUANTIFIERS`, `sweep.METHOD_SIMULATORS`)

A cell a dataset's pool cannot supply is recorded as an explicit missing run
rather than skipped or padded
([ADR-0005](adr/0005-real-data-experiment-design.md)). Capacity is a property
of the **bag frame**, not of the dataset: Haberman's 306 instances become a
bag frame of 123, 33 of them minority, so every prevalence is satisfiable at
`bag_size=100` within #18's replication cap, only 10%–30% positive is at 500,
and nothing at all at 1000 or 5000. At the larger bag sizes a shortfall
consumes a whole dataset rather than one cell.

## Real-data, multiclass

The same pipeline as above, on datasets with more than two classes
([#14](https://github.com/luizfernandolj/MoSS/issues/14)) — evidence that
the simplex simulators (MVN, Dirichlet) do work the binary uniform
simulator cannot attempt. Produced by `real_data.py --multiclass`.

- **Datasets** (20 that reach the grid), by class count:
  - 11: `healthcare`, `music_genre`
  - 10: `mfeat`, `fashion_mnist`, `digits`
  - 7: `dry_bean`, `covertype`, `fabert`
  - 6: `dermatology`, `har`, `mosquitoes`, `satimage`
  - 5: `phishing_url`, `microaggregation2`, `student_performance`
  - 4: `customer_segmentation`, `yeast_schumacher`
  - 3: `mfeat_icdm21`, `star_classification`, `academic_success`

  `dry_bean` is fetched through mlquantify; all the rest bar `yeast` are
  CSVs under `data/`, registered in
  `utils/local_datasets.MULTICLASS_DATASETS` — which also records which CSVs
  under `data/` were *not* registered, and why. `yeast` is registered and
  then dropped: its smallest class is under `real_data.MIN_CLASS_SIZE`, so
  the class floor excludes it
  ([#18](https://github.com/luizfernandolj/MoSS/issues/18)), which is why
  the table has 20 datasets and the registry names 21.
- **Target prevalences**: 21 prevalence vectors per dataset, drawn uniformly
  from that dataset's class-count simplex (a Dirichlet(1, ..., 1) draw),
  seeded per dataset so two datasets sharing a class count still draw
  different grids
- **Bag size, repetitions, seed, classifier, base quantifiers, method
  simulators**: same as the binary real-data spec above

A class the target prevalence rounds down to zero instances — which a
uniform simplex draw produces often at a bag size of 100 and a high class
count (8 of the 21 vectors drawn for an 11-class dataset) — is recorded in
`true_prevalence` at zero rather than left out of the vector, so a run's
estimate and its truth are always the same length.

Two CSVs under `data/` were left out on cost rather than on the class floor:
`quapy_data/letter` (26 classes) and `ours/Walking` (22). The one-vs-rest
decomposition fits one base quantifier per class, so either costs roughly
twice the rest of this table put together for one more point of simplex
dimension.
