# Accept mlquantify 0.5.1's histogram defaults for DyS and HDy

ADR-0001 assumed that runs using no meta-quantifier would be numerically
identical across the 0.2.0 → 0.5.1 upgrade, because nothing in their path
changed. The characterization fixture captured on 0.2.0 disproved that. Nine of
the eleven base quantifiers reproduce exactly; the two histogram-matching ones
do not.

| Quantifier | Rows differing | Max ΔMAE |
| --- | --- | --- |
| DyS | 36/36 | 0.0238 |
| HDy | 28/36 | 0.0300 |
| The other nine | 0/36 | 0 |

Two upstream default changes account for all of it.

**Histogram smoothing is off by default.** 0.2.0's `getHist` applied additive
smoothing unconditionally — each bin was `(count + 1/nbins) / (n + 1)`. 0.5.1
exposes this as `laplace_smoothing` and defaults it to `False`, giving plain
`count / n`.

**DyS no longer aggregates over bin sizes by median.** 0.2.0 solved the mixture
independently for each bin size and took `np.median` of the resulting
prevalences. 0.5.1's `bin_strategy` defaults to `None`, which concatenates the
bin blocks and solves once instead.

Setting `DyS(bin_strategy="median", laplace_smoothing=True)` and
`HDy(laplace_smoothing=True)` recovers the old numbers: HDy becomes bit-exact,
and DyS agrees to within 4.4e-05 — inside the ternary search's own `tol=1e-4`,
so the residual is where the optimiser stops, not which base quantifier runs.

We nonetheless take 0.5.1's defaults as they ship, and re-capture the fixture
on 0.5.1. The alternative pins the study to defaults that a future release may
change again, and to a smoothing rule that was an undocumented implementation
detail rather than a stated part of either method.

## Consequences

- DyS and HDy mean different base quantifiers before and after this change. Results
  from the two eras are not comparable for those two methods. Everything from
  the 0.2.0 era is void under ADR-0001 anyway, so nothing further is lost.
- The characterization fixture is no longer evidence about the upgrade. It is a
  drift guard on the pinned version. The 0.2.0 record it replaced is in commit
  58102cc.
- As shipped, `DyS()` is not the DyS of Maletzke et al.: the median over bin
  sizes is part of the published method, and 0.5.1's docstring still says
  estimates "are aggregated by their median" while the default does not do
  that. This is an upstream bug, not a decision this project makes. If it is
  fixed upstream, DyS moves again and this ADR should be revisited.
- That bug is reported to `coenlab/mlquantify` rather than worked around here.
  Confirmed on one fixed binary cell: `DyS()` gives 0.341137 and
  `DyS(bin_strategy="median")` gives 0.339596, so the default demonstrably
  skips the median the docstring promises. Whichever way upstream resolves it —
  moving the default to match the docs, or the docs to match the default — this
  project repins and re-captures the fixture. The report body is drafted at
  `.scratch/upstream-dys-issue.md`.
