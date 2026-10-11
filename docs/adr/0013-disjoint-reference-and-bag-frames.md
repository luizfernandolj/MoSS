# Disjoint reference and bag frames

A pool is split 60/40, stratified: the reference frame is cross-validated
within itself and is the score distribution every method matches a bag
against; the bag frame is scored by a Random Forest fitted on the whole
reference frame, and is the only thing `draw_bag` samples. The two share no
instance.

[ADR-0005](0005-real-data-experiment-design.md) specified out-of-fold rather
than refit probabilities, and argued it on overconfidence: a classifier's
scores on its own training data are not what a method sees at prediction
time. That argument is sound and is unchanged here. What it never addressed
is a second and independent question — whether the instances *in a bag* may
also be in the reference that bag is estimated against. Under ADR-0005 they
were: one frame served as both the reference score set and the sampling frame
for bags, so every bag was a subset of its own reference.

That is not a rounding error at every size. A bag of 100 drawn from a
50,000-instance pool is 0.2% of its reference and the contamination is
negligible. A bag of 1000 drawn from `mfeat`'s 2,000 is **half** of it, and
the mixture search a base quantifier runs is easier against class-conditional
distributions it helped estimate — biasing error downward, by an amount that
grows with bag size and shrinks with pool size, which is to say by an amount
that varies systematically across exactly the two axes the study reports.

It does not affect every method equally, which is why it survived review for
so long. The 30 arms whose method simulator is `uniform`, `mvn` or
`dirichlet` read **only the reference's labels** — their candidates are
simulated, and the real scores are the one thing they deliberately never see
(`sweep.MetaEstimator`). They were never contaminated. The 11 plain
quantifiers (`sweep.ReferenceEstimator`) and the 10 `all` arms
(`sweep.CandidateEstimator`, which carries the real reference as a candidate,
[ADR-0010](0010-restore-the-dropped-arm-as-a-candidate-search.md)) were. Those
21 are the baselines the simulator arms are compared against, so the effect
was to flatter the comparison's *control* rather than its treatment.

## Considered options

**Leave it, and disclose.** Defensible for the `bag_size=100` slice the
published figures use, where the overlap is 0–5% for every dataset but
Haberman. Rejected because the bag-size sweep ([#16](https://github.com/luizfernandolj/MoSS/issues/16))
is a published axis, and the contamination grows along precisely that axis —
a reader comparing bag 100 to bag 5000 would be reading an artefact as a
finding.

**Train/test split with the reference taken from the training side's own
fitted scores.** The obvious textbook shape, and rejected: it reintroduces
exactly the overconfidence ADR-0005 exists to prevent. Both constraints have
to hold at once.

**Rotate cross-validation folds** — one held-out fold per repetition as the
bag frame, the other nine as the reference. Elegant, reuses machinery already
present, and uses all the data across repetitions. Rejected on coverage: a
bag frame of 10% leaves Haberman 31 instances and Pima 77, which fails the
`MIN_CLASS_SIZE` floor outright and would drop both datasets plus the smaller
multiclass pools.

**50/50 and 70/30** were weighed against the chosen 60/40. 70/30 estimates
the reference better but leaves Haberman 24 minority instances in the bag
frame, under the floor, dropping the dataset. 50/50 is also workable; 60/40
was chosen to keep the reference the larger side, since it is what every
method's estimate is anchored to.

## Consequences

- **Every real-data number produced before this is superseded**, including the
  binary table of 342,720 runs. Not only the 21 contaminated arms: bags are
  now drawn from a different frame, so every arm's bags differ.
- **Haberman survives, barely.** Its bag frame is 123 instances, 33 of them
  minority — three above `MIN_CLASS_SIZE`. Every prevalence at `bag_size=100`
  remains satisfiable within the replication cap
  ([#18](https://github.com/luizfernandolj/MoSS/issues/18)), which is the
  slice the published figures use. At 500 it covers only 10%–30% positive,
  where it previously reached 80%; at 1000 and 5000 it contributes nothing.
- **The class floor now reads the bag frame**, not the whole dataset. This is
  stricter and is the honest reading: a class with 30 instances spread across
  both frames never could fill a bag from the frame bags are drawn from. No
  currently registered dataset is newly excluded by it.
- The bag frame is scored by one forest fitted on the whole reference frame
  rather than by a second cross-validation. It is already unseen by that
  forest, which is the property out-of-fold scoring buys; and fitting on all
  of the reference rather than a fold of it means bags are scored by a
  better-trained classifier than any single fold's, which is what a method
  would face at prediction time.
- The synthetic experiment is untouched. There the reference and the bags are
  independent draws from a simulator, and so were always disjoint.
