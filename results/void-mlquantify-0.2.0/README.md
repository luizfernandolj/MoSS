# Void results — do not use

Every run in this directory was produced under `mlquantify==0.2.0`, whose
meta-quantifier ignored the base quantifier it was constructed with. Roughly
80% of these runs encode a base-quantifier axis with no causal effect, and
about 4.5% carry a stale estimate from a caught exception that logged and fell
through without skipping the row. The two defects cannot be disentangled from
the stored data, which is why the sweep is being re-run rather than filtered.

ADR-0001 records the decision. These files are kept only so the claim can be
checked; nothing in the rebuilt project should read them.

The figures still in `results/` were rendered from this data and are void for
the same reason.
