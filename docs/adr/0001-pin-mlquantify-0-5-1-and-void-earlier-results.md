# Pin mlquantify 0.5.1 and discard all results produced under 0.2.0

In mlquantify 0.2.0, `QuaDapt.aggregate` returns the prevalence found by its own
internal mixture search and never reads the base quantifier it was constructed
with — so every run that wrapped a different base quantifier computed the same
thing. 0.5.1 corrects this: the mixture search selects only a merging factor,
and the base quantifier produces the estimate from a reference simulated at that
factor. We pin 0.5.1 exactly and treat every result produced under 0.2.0 as void.

## Consequences

- Roughly 80% of the existing runs encode a base-quantifier axis with no causal
  effect. Mean absolute error spread across the ten base quantifiers is 0.0602
  without the meta-quantifier and 0.0003–0.0012 with it — the latter is the
  Monte Carlo noise of unseeded simulator draws, not method difference.
- Separately, about 4.5% of runs carry a stale estimate: a caught exception
  logged and fell through without skipping the row, so the failing method was
  recorded with the previous method's number. The two defects cannot be
  disentangled from the stored data, which is why the sweep is re-run rather
  than filtered.
- The published LQ 2025 QuaDapt results used the corrected behaviour and are
  unaffected by this.
- The library version is part of every result and is recorded alongside it.
