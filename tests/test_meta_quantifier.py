"""The property whose absence let ADR-0001's defect survive 3.75M runs.

Under mlquantify 0.2.0 the meta-quantifier returned the prevalence found by
its own mixture search and never consulted the base quantifier it wrapped, so
every base quantifier computed the same thing. Under 0.5.1 the mixture search
selects only a merging factor and the base quantifier produces the estimate
from a reference simulated at that factor.

The property is asserted twice, at two seams, because the two guard different
things.

At the sweep seam
-----------------
``test_base_quantifiers_in_a_cell_do_not_collapse_to_one_estimate`` is the
assertion this project actually cares about: this project's simulators, this
project's registries, one bag, one method simulator, ten base quantifiers.

It could not be written until the candidate draws were seeded (ADR-0011). While
they came from OS entropy, every base quantifier in a cell drew its own
candidates, so estimates differed *even under the defect* — measured on a fixed
cell, the per-run spread was indistinguishable between 0.2.0 and 0.5.1 (medians
0.258 and 0.264), and any assertion here passed on both versions and guarded
nothing. Now that the seed is derived from the cell and the repetition and
*not* from the base quantifier, the ten estimates come from identical inputs:
under the defect they would be one number repeated ten times, and they are ten
numbers spanning about 0.27.

That is ADR-0004's second consequence made into a test rather than a
convention — identical output across base quantifiers is an alarm about wiring,
not a coincidence.

At the library seam
-------------------
``test_base_quantifiers_produce_a_material_spread`` keeps the library pin
honest, which the sweep-seam test cannot: it drives mlquantify's own
``QuaDapt``, whose ``MoSS`` still draws from the legacy global ``np.random``
(ADR-0004) and so is pinned with ``np.random.seed``. Held fixed that way, the
two versions separate completely — 0.2.0 returns one identical estimate for all
ten base quantifiers, 0.5.1 returns a spread.

It draws a larger bag than the sweep's own bag size of 100, deliberately: at
100 the counting quantifiers quantise to 0.01, the same order as MIN_SPREAD,
and the pass/fail margin would rest on that rounding rather than on the
property.
"""

import numpy as np
import pytest

from mlquantify.meta import QuaDapt

import runs
import sweep
from sweep import DATA_SIMULATORS, run_sweep

#: Controls mlquantify's internal MoSS reference draw, so the only thing
#: varying between the estimates compared below is the base quantifier.
SEED = 0

#: On 0.2.0 the estimates are byte-identical, so the spread is exactly 0.0.
#: On 0.5.1 it ranges from 0.031 to 0.268 over the first five seeds. This sits
#: an order of magnitude clear of the smallest of those.
MIN_SPREAD = 0.01

#: CC ignores the reference score set entirely, so it takes no meta-quantifier
#: path — the sweep records it once as a baseline, and QuaDapt cannot even call
#: it.
BASE_QUANTIFIERS = {
    name: quantifier
    for name, quantifier in sweep.BASE_QUANTIFIERS.items()
    if name != runs.BASELINE_QUANTIFIER
}


@pytest.fixture(scope="module")
def fixed_cell():
    """One cell of simulated scores, large enough not to quantise estimates."""
    simulate = DATA_SIMULATORS[runs.UNIFORM]
    rng = np.random.default_rng(20260908)
    train_scores, train_labels = simulate(
        n=2000, alpha=0.5, merging_factor=0.2, random_state=rng
    )
    test_scores, _ = simulate(
        n=1000, alpha=0.3, merging_factor=0.5, random_state=rng
    )
    return test_scores, train_labels


@pytest.fixture(scope="module")
def estimates(fixed_cell):
    test_scores, train_labels = fixed_cell
    estimated = {}
    for name, quantifier in BASE_QUANTIFIERS.items():
        np.random.seed(SEED)
        estimated[name] = QuaDapt(quantifier()).aggregate(test_scores, train_labels)[1]
    return estimated


def test_base_quantifiers_are_not_all_collapsed_to_one_estimate(estimates):
    # The exact signature of the defect: ten meta-quantifiers, one number.
    assert len(set(estimates.values())) > 1


def test_base_quantifiers_produce_a_material_spread(estimates):
    values = np.array(list(estimates.values()))

    assert values.max() - values.min() > MIN_SPREAD


@pytest.fixture(scope="module")
def smoke_runs():
    """The smoke sweep's runs: every method simulator, every base quantifier."""
    return run_sweep(sweep.SMOKE_SWEEP)


def test_the_smoke_sweep_estimates_under_every_method_simulator(smoke_runs):
    # The net under this project's own simulators and registries, which the
    # library-seam test above cannot reach because it drives mlquantify's
    # meta-quantifier rather than the sweep. A method simulator that stopped
    # working would reach the results as missing runs, not as an error, so the
    # assertion is that every estimate is there.
    assert set(smoke_runs["method_simulator"]) == set(runs.METHOD_SIMULATORS)
    assert set(smoke_runs["base_quantifier"]) == set(runs.BASE_QUANTIFIERS)
    assert smoke_runs["estimated_prevalence"].notna().all()


def test_base_quantifiers_in_a_cell_do_not_collapse_to_one_estimate(smoke_runs):
    # The defect, stated where this project would suffer it. Grouped by method
    # simulator and repetition, the rows differ in nothing but the base
    # quantifier: same bag, same candidate score sets, because the seed is
    # derived from the cell and the repetition and not from the quantifier
    # (ADR-0011). Under 0.2.0 each group would hold one number repeated.
    meta = smoke_runs[smoke_runs["method_simulator"] != runs.NO_METHOD_SIMULATOR]
    spread = meta.groupby(["method_simulator", "repetition"])["estimated_prevalence"]

    assert (spread.max() - spread.min() > MIN_SPREAD).all()
