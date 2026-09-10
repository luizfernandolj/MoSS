# Seed every simulator draw, including inside the meta-quantifier

An unseeded simulator call inside the meta-quantifier is what disguised the
defect recorded in ADR-0001: had the draws been deterministic, every base
quantifier would have returned an identical estimate and the missing delegation
would have been obvious in an afternoon instead of invisible across 3.75M runs.
We derive a seed from the sweep cell and repetition index and thread it through
both the experiment's own simulators and the meta-quantifier's internal ones.

## Consequences

- mlquantify's `QuaDapt.MoSS` accepts a `random_state` but documents it as
  unused, so end-to-end determinism requires an upstream change. Until then the
  meta-quantifier's internal draws remain a nondeterminism source no amount of
  care in this repository can remove.
- Identical output across different base quantifiers is treated as an alarm
  about wiring, not as a coincidence. This is a test, not just a convention.

## Amended by ADR-0011

ADR-0011 implements this decision, and the first consequence turned out to be
too pessimistic. The seam mlquantify calls without a seed is one this project
already overrides, so the meta-quantifier's draws are seeded here after all;
what is left upstream is filed as
[coenlab/mlquantify#5](https://github.com/coenlab/mlquantify/issues/5) and now
reaches only code paths that use the library's own `QuaDapt` unmodified.

The second consequence is a test at the sweep seam,
`test_base_quantifiers_in_a_cell_do_not_collapse_to_one_estimate`. It could not
be written until the draws were seeded: while they came from OS entropy, the
alarm and the all-clear looked the same.
