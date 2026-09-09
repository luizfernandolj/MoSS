"""The property whose absence let ADR-0001's defect survive 3.75M runs.

Under mlquantify 0.2.0 the meta-quantifier returned the prevalence found by
its own mixture search and never consulted the base quantifier it wrapped, so
every base quantifier computed the same thing. Under 0.5.1 the mixture search
selects only a merging factor and the base quantifier produces the estimate
from a reference simulated at that factor.

Why this test does not go through the sweep entry point
-------------------------------------------------------
It cannot, yet. The sweep's meta-quantifier arms use this project's own
simulator overrides, which draw from ``np.random.default_rng(None)`` and are
not seedable (ADR-0004). Those unseeded draws make estimates differ between
base quantifiers *even under the defect* — measured on a fixed cell, the
per-run spread across base quantifiers is indistinguishable between 0.2.0 and
0.5.1 (medians 0.258 and 0.264). Any assertion at the sweep seam therefore
passes on both versions and guards nothing.

Held fixed, the two versions separate completely: 0.2.0 returns one identical
estimate for all ten base quantifiers, 0.5.1 returns a spread of estimates.
mlquantify's own ``QuaDapt.MoSS`` draws from the legacy global ``np.random``,
so seeding it is possible here even though seeding the project's overrides is
not. That is why this test wraps the library's meta-quantifier directly. Move
it onto the sweep seam once ADR-0004's seeding lands.

Two limits of that workaround, both consequences of the same gap. It guards
the library pin rather than the sweep, so a regression introduced in this
project's own overrides or in QUADAPT_VARIANTS would not be caught here — the
smoke test below is the only thing standing under those. And it draws a larger
test set than the sweep's TEST_SIZE of 100, deliberately: at 100 the counting
quantifiers quantise to 0.01, the same order as MIN_SPREAD, and the pass/fail
margin would rest on that rounding rather than on the property.
"""

import numpy as np
import pytest

from mlquantify.meta import QuaDapt

from binary_experiment import run_experiment
from utils.moss import MoSS
from variables import MOSS_VARIANTS, QUADAPT_VARIANTS, QUANTIFIERS

#: Controls mlquantify's internal MoSS reference draw, so the only thing
#: varying between the estimates compared below is the base quantifier.
SEED = 0

#: On 0.2.0 the estimates are byte-identical, so the spread is exactly 0.0.
#: On 0.5.1 it ranges from 0.031 to 0.268 over the first five seeds. This sits
#: an order of magnitude clear of the smallest of those.
MIN_SPREAD = 0.01

#: CC ignores the training reference entirely, so it takes no meta-quantifier
#: path — the sweep short-circuits it, and QuaDapt cannot even call it.
BASE_QUANTIFIERS = {name: q for name, q in QUANTIFIERS.items() if name != "CC"}

CELL = {
    "m_train": 0.2,
    "m_test": 0.5,
    "alpha": 0.3,
    "moss_train_variant": MOSS_VARIANTS["MoSS"],
    "moss_test_variant": MOSS_VARIANTS["MoSS"],
    "moss_train_variant_name": "MoSS",
    "moss_test_variant_name": "MoSS",
}


@pytest.fixture(scope="module")
def fixed_cell():
    """One cell of simulated scores, large enough not to quantise estimates."""
    rng = np.random.default_rng(20260908)
    train_scores, train_labels = MoSS(
        n=2000, alpha=[0.5, 0.5], merging_factor=0.2, random_state=rng
    )
    test_scores, _ = MoSS(
        n=1000, alpha=[0.7, 0.3], merging_factor=0.5, random_state=rng
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
    # The exact signature of the defect: ten wrappers, one number.
    assert len(set(estimates.values())) > 1


def test_base_quantifiers_produce_a_material_spread(estimates):
    values = np.array(list(estimates.values()))

    assert values.max() - values.min() > MIN_SPREAD


def test_sweep_completes_for_every_method_and_base_quantifier():
    runs = run_experiment(**CELL, random_state=0, strict=True)

    assert set(runs["Quantifier"]) == set(QUANTIFIERS)
    assert set(runs["Quadapt_Variant"]) == set(QUADAPT_VARIANTS)
    assert np.isfinite(runs["MAE"]).all()
