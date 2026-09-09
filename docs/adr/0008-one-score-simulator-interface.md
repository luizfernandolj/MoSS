# Reach all three score simulators through one interface

Each score simulator used to have its own signature, and each meta-quantifier
arm of the sweep had its own subclass whose only job was to reshape the
prevalence before delegating. The simulators are now three implementations of
one `ScoreSimulator` interface in `utils/simulators.py`, which reads the
prevalence and divides the sample between classes once, before handing an
implementation the counts to draw. The subclasses are replaced by
`QuaDaptWithSimulator`, which takes its method simulator as an argument.

The interface is mlquantify's `QuaDapt.MoSS(n, alpha, merging_factor, classes,
random_state)`, parameter for parameter, for the reason ADR-0002 gives.

## The rounding rule changed, and with it some counts

Reading a prevalence and turning it into per-class counts happened in five
places, and two of them disagreed. The uniform simulator floored the positive
count and gave the remainder to the negative class; the MVN and Dirichlet
simulators floored every class and gave the remainder to the last. Both are now
one rule — every class takes its floor, and the shortfall goes to the classes
with the largest fractional parts.

That is not a pure refactor. The old MVN and Dirichlet rule was at the mercy of
float error in the prevalence it was handed: `1 - 0.8` is a shade under `0.2`,
so flooring it dropped an observation and a requested prevalence of 0.8 came
back as 0.81. At the sweep's `TEST_SIZE` of 100 that is 81 positives where 80
were asked for, on two of the three data simulators, at one of the seven
prevalences in `ALPHAS`.

We take the fix rather than preserve the old counts. The affected runs are
those with a prevalence of 0.8 and an MVN or Dirichlet data simulator; no
post-0.5.1 results existed to invalidate, and the golden record in
`tests/fixtures/plain_quantifier_runs.csv` does not cover a prevalence of 0.8,
so it replays unchanged. `tests/test_simulators.py` asserts the counts directly
now, on all three implementations.

## Consequences

- A caller passes a prevalence and nothing else. There is no `n_classes`
  parameter to disagree with it, and a `classes` that disagrees raises rather
  than being silently ignored as the old overrides ignored it. Binary callers
  pass the positive class's share as a scalar and let the simulator read it
  into a vector, rather than writing `[1 - alpha, alpha]` at the call site —
  that spelling is what the reshaping used to look like.
- The registries in `variables.py` are `DATA_SIMULATORS` and
  `METHOD_SIMULATORS`, named for the two roles in CONTEXT.md rather than for
  the classes they used to hold. The result columns they feed —
  `MoSS_Train_Variant`, `MoSS_Test_Variant`, `Quadapt_Variant` — and the
  registry keys that supply their values are frozen by the golden record and by
  the result files already written, so they keep their old spelling.
- The simulators accept and honour a `random_state`, where the old overrides
  accepted one and dropped it. What remains of ADR-0004's gap is entirely
  upstream: mlquantify calls its `MoSS` seam without a seed, so the draws
  inside a meta-quantifier are still not reproducible.
- `QuadaptNew` still cannot run, for the reason ADR-0007 gives. Its list of
  candidate simulators was retargeted at the new classes, because the functions
  it named no longer exist; nothing else about it changed, and rewriting its
  `best_mixture` against 0.5.1's API remains its own change. It is therefore
  the one place a simulator is still chosen by subclass identity rather than by
  parameter, and it is quarantined.
- `QuaDaptWithSimulator` forwards `measure`, `merging_factors` and `strategy`
  to the library rather than restating their defaults, so an upstream change to
  any of them is not silently overridden here.
