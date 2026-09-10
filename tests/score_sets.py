"""Score sets built to be reasoned about, for the tests that need them.

A quantifier's answer on simulated scores is only ever approximately knowable,
which is no basis for asserting what a method read. These build score sets
whose answer is known before anything runs: two classes, each at one score.

Kept beside the tests rather than in ``conftest.py`` because they are values
and a class, not fixtures — the same reason ``characterization.py`` sits here.
"""

import numpy as np


def scored(n, prevalence, positive_score=0.95, negative_score=0.05, classes=(0, 1)):
    """``n`` labelled rows whose two classes score at the two values given.

    Defaults to scores so cleanly separated that the prevalence they imply is
    unmistakable. Passing the two classes the *same* score builds the opposite
    and equally useful thing: a score set every mixture of which is the same
    distribution, so it matches a bag with any spread at all no better than it
    matches any other, and a search that consults it has to reject it.
    """
    n_positive = round(n * prevalence)
    positive = np.tile([1 - positive_score, positive_score], (n_positive, 1))
    negative = np.tile([1 - negative_score, negative_score], (n - n_positive, 1))
    labels = np.concatenate(
        (np.full(n_positive, classes[1]), np.full(n - n_positive, classes[0]))
    )
    return np.vstack((positive, negative)), labels


#: The two classes scoring alike: see :func:`scored`. Spread as keyword
#: arguments into either ``scored`` or :class:`FixedSimulator`, which take the
#: same two.
UNMATCHABLE = dict(positive_score=0.2, negative_score=0.2)


class FixedSimulator:
    """A method simulator drawing the same score set whatever it is asked for.

    Not a :class:`~utils.simulators.ScoreSimulator` subclass: it overrides the
    one thing such a subclass exists to provide, so inheriting would claim a
    relationship it does not have. What a caller needs of a method simulator is
    that it is callable with the library's arguments, which this is.
    """

    def __init__(self, positive_score, negative_score):
        self.positive_score = positive_score
        self.negative_score = negative_score

    def __call__(self, n, alpha, merging_factor, classes=None, random_state=None):
        return scored(
            n,
            alpha,
            self.positive_score,
            self.negative_score,
            (0, 1) if classes is None else classes,
        )
