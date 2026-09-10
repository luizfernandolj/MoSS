# Restore the dropped meta-quantifier as a candidate search

`QuadaptNew` was dropped from the sweep during the 0.5.1 port because it held a
copy of mlquantify 0.2.0's mixture-search helper, which 0.5.1 removed
(ADR-0007). It is rewritten and back, as `QuaDaptOverCandidates` in
`utils/meta_quantifier.py` and the `all` arm of the sweep's
`METHOD_SIMULATORS`. **This supersedes ADR-0007**, whose warning stands for
every result produced before this change and for nothing after it.

## What it is

A meta-quantifier that commits to no one method simulator. It builds a list of
**candidate score sets** — the real reference scores, and every simulator
across the merging grid — measures each against the bag with the library's own
mixture search, and hands the closest to the base quantifier, which produces
the estimate. The old class did none of those four things:

- It **rebound its own `MoSS` attribute** between simulators mid-aggregation
  and stored the split reference on itself, so an estimate depended on which
  estimate had run before it. Candidates are now a value returned from
  `candidates()`; nothing about the object changes while it estimates, which
  is what makes it safe in the loky workers the published sweep runs in.
- It **re-implemented the mixture search**, dispatching on `measure` twice in
  two branches that had already begun to differ. `_distance_to` reaches
  upstream's `_histogram_best_mixture` and `_sord_best_mixture` instead, and
  dispatches once, over the same named measures upstream accepts. No copy of
  library internals remains here.
- It **returned the prevalence its own search found** and never consulted the
  quantifier it wrapped — ADR-0001's first defect, under which every base
  quantifier computed the same number. The base quantifier now produces the
  estimate, as it does in every other arm.
- It **split its reference with `train_labels == 1`** and answered
  `[1 - prevalence, prevalence]`. On a dataset labelled anything but 0 and 1
  both halves came back empty and the two numbers named no class. The classes
  are resolved from the reference labels and carried through the draws, the
  split and the answer, so the real-data experiment (ADR-0005) can use it.

## The real reference scores are one of the candidates

This is the one meta-quantifier arm handed the real reference scores.
`MetaEstimator` withholds them on purpose — replacing them is what its method
does, and the study's question is whether that helps — but here they compete
with the simulated candidates at their own game. "No simulated substitute
matched this bag better than the scores the classifier actually produced" is
therefore an answer this arm can give, which is the whole reason it is a
different method and not a fourth simulator.

**"At zero merging" is read as "not simulated at all".** The old code put the
real reference in the search by prepending `0.0` to the merging grid and
treating that first slot specially; issue #7 inherited the phrase from it. A
score set nobody simulated has no merging factor, so the candidate list carries
it as a candidate and no simulated candidate is drawn at zero. The search is
the same one either way. What is lost is that a run cannot say *which*
candidate won — it never could, and recording that would be a schema change
(ADR-0003) rather than a detail of this one.

## Consequences

- `runs.METHOD_SIMULATORS` gains `all`, labelled `QuaDapt-All(DyS)` and so on.
  The value is in the *method simulator* column because that column answers
  "what produced the candidates this estimate came from", and for this arm the
  answer is all of them. `runs.METHOD_SIMULATOR_LABELS` is a second map beside
  `SIMULATOR_LABELS`, which keeps answering the narrower question of which
  simulator drew a set of scores — `all` is not an answer to that one.
- A spec's `method_simulators` values now come in three shapes: one simulator,
  a tuple of them, or `None`. `estimator_for` is the only place that reads the
  difference, which is where the other calling-convention differences already
  live. **This amends ADR-0009**, which describes three estimator adapters:
  `CandidateEstimator` is a fourth, and the calling convention it adapts —
  `aggregate(bag, reference_scores, reference_labels)` on a meta-quantifier —
  is the one the other three had no arm for.
- The published grid grows from 2,797,389 runs to 3,479,679: one more arm on
  every bag under every base quantifier but the baseline. It costs far more
  than a fifth of the sweep, because it searches 16 candidates per estimate
  where the other meta arms search 5. Measured on the smoke sweep: 2.4 seconds
  without it, 4.8 with. Budget the published re-run (#9) accordingly — this
  arm alone is about as expensive as everything else put together.
- `QuaDaptOverCandidates` subclasses `QuaDapt` for one reason: to reach the
  mixture search rather than copy it. It is used through `aggregate` alone,
  and its `aggregate` takes the real reference scores where the library's takes
  training labels — so the inherited `fit`, `predict` and `best_mixture` do not
  work on it, and its docstring says so. Composition would mean calling
  upstream's private search helpers on a `QuaDapt` built to be thrown away,
  which is a worse dependency on internals, not a better one.
- `dashboard.py` draws the arm, because it draws whatever arms are in the runs
  and would otherwise stop where it met one it had no colour for. The
  published figures in `export_grid_matplotlib.py` do **not**: they compare the
  three method simulators, and adding a fourth line is a decision about what
  the paper claims. The arm's absence from those figures is a choice, unlike
  its absence from the runs under ADR-0007, which was a defect.
- Seeding is exactly as it is everywhere else: absent, and #8's to fix
  (ADR-0004). This arm draws its own candidates rather than going through the
  library's unseeded `MoSS` seam, so it is the one arm that *could* be seeded
  today — but seeding one arm and not the others would make the sweep's
  reproducibility depend on which method a run belongs to, which is worse than
  none of it being seeded.
