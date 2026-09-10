# Derive every seed from the cell and the repetition

A spec carried one `seed` and handed it to every cell, which opened one
generator and drew everything in the cell from it. Two consequences, both
measured on the published grid:

- **Cells shared their reference score sets.** Every cell opened its stream at
  the same position, so any two cells agreeing on reference simulator and
  reference merging factor drew a byte-identical reference. The published grid
  is 22,743 cells and 57 such pairs: 22,743 cells resting on 57 distinct
  reference score sets, and a spread across the grid that understated the
  sampling variation it was there to measure.
- **Every draw depended on the draws before it.** The bags continued the stream
  the reference had just been drawn from, so doubling `reference_size` changed
  every bag in the sweep — measured on the smoke grid, 100% of the baseline
  quantifier's estimates moved when only the reference size changed. The
  baseline reads no reference at all.

And the meta-quantifier's own draws were seeded by nothing whatsoever, which is
what ADR-0004 was written about.

`seed_for(spec, cell, repetition, draw)` replaces the single stream. It derives
a seed by hashing the spec's seed together with the cell, the repetition index
and which of the cell's three draws is being made — the reference score set,
the bag, or the meta-quantifier's candidate score sets. Two runs of one spec
derive the same seeds; two draws that should differ derive different ones; a
spec with no seed still draws from OS entropy.

blake2b rather than `hash`, because Python salts string hashing per process and
cells cross into loky workers on the published path. A per-process hash would
have seeded each worker differently and left the property holding only at
`n_jobs=1`, which is the one configuration the published sweep never uses.

## The seed is not a function of the base quantifier

Deliberately. Every base quantifier in a repetition now sees the same bag and
the same simulated candidates, so the only thing varying between their
estimates is the quantifier itself.

That is what makes ADR-0004's second consequence into a test. Under 0.2.0 the
meta-quantifier never consulted the base quantifier it wrapped, and ten
wrappers returned one number; unseeded draws hid it, because the per-draw
variation was of the same size as the between-method variation (medians 0.258
and 0.264 — indistinguishable). Seeded this way, the defect is one number
repeated ten times. `test_base_quantifiers_in_a_cell_do_not_collapse_to_one_
estimate` asserts it at the sweep seam, where it belongs; verified by imitating
the defect, which collapses the spread from 0.27 to exactly 0.0.

## The meta-quantifier is seeded here, not upstream

ADR-0004 expected end-to-end determinism to need an upstream change. It does
not, quite. mlquantify calls its `MoSS` seam with no `random_state` — and its
own `MoSS` documents the argument as unused — but `QuaDaptWithSimulator`
*overrides* that seam already, which is the whole reason the class exists. The
override now falls back to a generator the estimator holds, restarted per
`aggregate`, so the several candidate draws of one estimate differ from each
other while the estimate as a whole repeats. A `random_state` the caller passes
still wins, so the day upstream threads one through, the caller's seed governs.

The gap is filed as [coenlab/mlquantify#5][upstream]: the argument is accepted
and documented as unused, and `aggregate` and `best_mixture` never pass one, so
no caller can reach it through the public API.

[upstream]: https://github.com/coenlab/mlquantify/issues/5

The other meta-quantifier arm — `QuaDaptOverCandidates`, ADR-0010 — takes the
same seed from the same place and holds no seeding state at all: it opens one
generator inside `candidates()` and advances it across the draws, because it
builds its candidates itself rather than through a seam the library calls. That
is the better shape, and the arm above cannot have it. Both were seeded when
the two changes met, and the sweep-wide determinism test is what caught the arm
that had not been.

## Consequences

- **The golden record was re-captured.** Its runs are the same quantifiers on
  different bags, so its numbers had to move; the question was whether anything
  else moved with them. Measured across four base seeds under both schemes, the
  per-quantifier mean absolute error ranges overlap for all eleven quantifiers,
  and the pooled means differ by 0.0002 (0.0999 old, 0.1001 new). The
  per-quantifier change from the committed record, 0.005–0.035, sits inside the
  0.001–0.033 the old scheme itself moved between seeds. Re-captured rather
  than translated, so it is now in the schema everything else uses and compares
  on both prevalences rather than on absolute error alone —
  `FIXTURE_COLUMNS` and `FIXTURE_SIMULATORS` are gone, and the `MoSS_MN` and
  `Quadapt_Dir` spellings with them. This is the second re-capture; ADR-0006
  was the first.
- **The published sweep carries a seed.** It had none, so nothing published
  from it could be reproduced except from the file it was written to.
- **What is left of ADR-0004's gap is the library's own `QuaDapt`.** Anything
  reaching `mlquantify.meta.QuaDapt` without replacing its `MoSS` — which
  `tests/test_meta_quantifier.py` does deliberately, to pin the library rather
  than the sweep — is still unseeded, and still pinned with `np.random.seed`.
  That test moves onto the sweep seam when [#5][upstream] lands; the
  sweep-seam assertion above does not wait for it.
- **A cell is now hashed, so a field added to `Cell` changes every seed in
  every sweep.** That is the correct behaviour — a cell that describes the
  draw differently is a different draw — but it means the golden record will
  need re-capturing again, on the same terms as here.
