# Two result tables, not one union table

Synthetic and real-data runs share an estimator vocabulary but not a data
vocabulary: merging factors and data simulators describe how synthetic scores
were made and mean nothing for a real dataset. Rather than one table whose
columns are null for half its rows, we write two tables that carry an identical
estimator block, so a single label function and a single statistical comparison
serve both.

## Consequences

- The method simulator appears in **both** tables. It describes the estimator,
  not the data, so "which simulator does the meta-quantifier use internally"
  remains a comparable axis on real datasets.
- Multiclass runs extend the real-data table by carrying a prevalence vector
  instead of a scalar, without a schema change.
- Both are stored as Parquet. The previous single CSV reached 291 MB and had to
  be split three ways to fit a hosting file-size limit; that split then leaked
  into every reader, and the readers drifted apart. Storage layout is now owned
  by one module and invisible to its callers.
