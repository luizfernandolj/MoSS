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
