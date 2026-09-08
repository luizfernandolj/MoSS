# Real-data experiment design

Eight binary tabular datasets, a Random Forest classifier, ten-fold out-of-fold
probabilities as the reference score set, and bags of 100 instances drawn under
an artificial-prevalence protocol at 21 target prevalences by 10 repetitions.
Out-of-fold rather than refit probabilities, because a classifier's scores on
its own training data are overconfident and would make the reference
unrepresentative of what the method sees at prediction time.

## Considered options

**Text datasets with transformer embeddings** — rejected despite available GPU
capacity. These methods operate on posterior scores, so changing the
representation varies the very score-distribution difficulty the synthetic
experiment exists to control. It would add a confound rather than a second
source of evidence. TF-IDF over IMDB is held in reserve should a reviewer ask
for text.

## Consequences

- Cells a dataset cannot supply are recorded as explicit missing runs rather
  than skipped. Haberman has 81 minority instances and so cannot fill a
  size-100 bag at high prevalence; that absence is data, and silently dropping
  it would misstate coverage.
- Pools for the largest datasets are capped at roughly 50k instances. Bags are
  100 instances regardless of pool size, so the cap bounds cross-validation cost
  without affecting what is measured.
- Rankings are reported with a Friedman test and Nemenyi post-hoc across
  datasets, which requires the per-dataset runs to stay individually addressable
  in the results table rather than pre-aggregated.
