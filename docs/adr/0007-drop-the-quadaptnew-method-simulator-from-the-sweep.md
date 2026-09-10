# Drop the QuadaptNew method simulator from the sweep

> **Superseded by ADR-0010.** The rewrite this ADR scoped to its own change has
> landed: this meta-quantifier is back in the sweep as the `all` arm. All below
> describes why it was absent, and still explains any result file produced
> before that — but the sweep no longer runs without it.

`QuadaptNew` copied mlquantify 0.2.0's mixture-search helper and calls
`best_mixture` on a matching quantifier instance. 0.5.1 removed that method,
replacing it with internal helpers on the meta-quantifier. Nothing in the
project can call it as written.

We remove it from `QUADAPT_VARIANTS` and leave the class in place, unreferenced,
rather than adapt copied library internals for code the follow-up refactor
rewrites anyway. Its results were void under ADR-0001 regardless.

## Consequences

- The sweep runs three method simulators plus the no-meta-quantifier arm, not
  four. **Its absence from the rebuilt results is not a finding**, and must not
  be read as one — that is the whole reason this ADR exists rather than a code
  comment alone.
- `dashboard.py` and `export_grid_matplotlib.py` carried `"QuadaptNew"`
  display-label mappings that could no longer fire. Both were rewritten behind
  the results module (#5) and the mappings are gone. The figure that used to
  compare the uniform method simulator against `QuadaptNew` now compares the
  three method simulators that remain, which is the comparison the study is
  about; the missing fourth arm is still not a finding.
- Restoring the simulator means rewriting `best_mixture` against 0.5.1's API,
  which belongs with that rewrite and not with the port.
- The registry named `QUADAPT_VARIANTS` above is now `METHOD_SIMULATORS`
  (ADR-0008). The decision this ADR records is unchanged: the class is still
  absent from it.
