"""The one interface every score simulator is reached through.

A score simulator turns a sample count, a prevalence and a merging factor into
soft scores with known labels. There are three of them — uniform, MVN and
Dirichlet — and two roles they play, data simulator and method simulator
(CONTEXT.md). The role is the caller's business; the interface is the same
either way, so no caller needs to know which implementation it holds.

The signature is mlquantify's ``QuaDapt.MoSS(n, alpha, merging_factor, classes,
random_state)``, parameter for parameter, so a meta-quantifier can call any
simulator here where it would call the library's own. ADR-0002 records why we
match it rather than invent our own: upstreaming these stays a file move.

The name ``alpha`` is inherited from that seam and kept only there. It is a
prevalence, and everything past the front door calls it one — the glossary
avoids ``alpha`` because a Dirichlet concentration also answers to it, and this
module has both.

Reading the prevalence and dividing the sample between classes happen once,
here, in :meth:`ScoreSimulator.__call__`. Implementations receive the counts
already worked out and only draw. That is deliberate: the two arithmetics used
to live in five places between the simulators and the meta-quantifier
subclasses that wrapped them, and they disagreed — see ``class_counts``.
"""

from abc import ABC, abstractmethod

import numpy as np

#: Floor on the MVN simulator's per-class variance, so that a merging factor of
#: zero still produces a spread rather than a point mass at the vertex.
EPS = 0.04


def as_prevalence(requested):
    """Read whatever the caller passed as a prevalence over every class.

    A scalar is the positive class's share of a binary problem — that is how
    mlquantify's meta-quantifier asks for a balanced reference set, and how the
    original MoSS paper writes it. A sequence is already the whole vector.
    """
    prevalence = np.asarray(
        [1.0 - requested, requested] if np.ndim(requested) == 0 else requested,
        dtype=float,
    )

    if prevalence.ndim != 1 or len(prevalence) < 2:
        raise ValueError(
            f"prevalence must cover at least two classes, got {prevalence!r}"
        )
    if not np.all(np.isfinite(prevalence)) or np.any(prevalence < 0):
        raise ValueError(
            f"prevalence must be finite and non-negative, got {requested!r}"
        )
    if prevalence.sum() <= 0:
        raise ValueError(
            f"prevalence must give some class a share, got {requested!r}"
        )

    return prevalence / prevalence.sum()


def resolve_classes(classes, n_classes):
    """Settle on the labels the simulated observations carry.

    Defaults to ``0 .. n_classes - 1``, which is what the sweep's own labels
    are. A ``classes`` that disagrees with the prevalence is an error rather
    than something to reconcile: the two came from different places, and the
    old overrides silently believed the prevalence.
    """
    if classes is None:
        return np.arange(n_classes)

    classes = np.asarray(classes)
    if len(classes) != n_classes:
        raise ValueError(
            f"classes has {len(classes)} labels but the prevalence covers "
            f"{n_classes} classes"
        )
    return classes


def class_counts(n, prevalence):
    """Divide ``n`` observations between the classes in those proportions.

    Each class takes its floor, then whatever the floors left over goes to the
    classes with the largest fractional parts. Handing the remainder to a fixed
    class instead — as both older rules did, one to the last class and one to
    the negative class — leaves the result at the mercy of float error in the
    prevalence: ``1 - 0.8`` is a shade under ``0.2``, so flooring it dropped a
    whole observation and a requested 0.8 came back as 0.81.
    """
    exact = n * prevalence
    counts = np.floor(exact).astype(int)

    shortfall = n - counts.sum()
    if shortfall:
        largest_remainder = np.argsort(counts - exact, kind="stable")
        counts[largest_remainder[:shortfall]] += 1

    return counts


class ScoreSimulator(ABC):
    """A score simulator: a sample description in, scores and labels out.

    Subclasses implement :meth:`_draw`, which is handed the per-class counts, a
    seeded generator and the class labels, and returns the scores and labels
    for one draw. Everything a caller might otherwise have to do first — read
    the prevalence, resolve the labels, split the sample — has already happened.
    """

    def __call__(self, n, alpha, merging_factor, classes=None, random_state=None):
        """Draw ``n`` scored observations at the requested prevalence.

        Parameters
        ----------
        n : int
            How many observations to draw in total.
        alpha : float or array-like
            The prevalence. A float is the positive class's share of a binary
            problem; an array is the share of every class, in label order.
        merging_factor : float or array-like
            How much the per-class score distributions overlap, from 0 (fully
            separated) to 1 (fully merged). An array sets it per class.
        classes : array-like or None, default=None
            The labels to attach, in the prevalence's order. Defaults to
            ``0 .. n_classes - 1``.
        random_state : int, Generator or None, default=None
            Seeds the draw. A Generator is drawn from and advanced, so a caller
            can thread one through a whole grid cell.

        Returns
        -------
        scores : ndarray of shape (n, n_classes)
            Soft scores, each row summing to one.
        labels : ndarray of shape (n,)
            The class each row was drawn from.
        """
        prevalence = as_prevalence(alpha)
        classes = resolve_classes(classes, len(prevalence))
        counts = class_counts(n, prevalence)
        rng = np.random.default_rng(random_state)

        return self._draw(counts, merging_factor, classes, rng)

    @abstractmethod
    def _draw(self, counts, merging_factor, classes, rng):
        """Draw ``counts[c]`` observations from each class ``classes[c]``."""


class UniformSimulator(ScoreSimulator):
    """The original MoSS simulator, drawing positives as ``U**m``.

    Negatives are ``1 - U**m`` for the same merging factor, which makes the two
    distributions mirror images that meet in the middle as ``m`` approaches 1.
    The construction has no reading beyond two classes, so it stays binary
    where the other two generalise.
    """

    def _draw(self, counts, merging_factor, classes, rng):
        if len(counts) != 2:
            raise ValueError(
                f"the uniform simulator is binary; got {len(counts)} classes. "
                "Use the MVN or Dirichlet simulator instead."
            )

        n_neg, n_pos = counts
        p_score = rng.uniform(size=n_pos) ** merging_factor
        n_score = 1 - (rng.uniform(size=n_neg) ** merging_factor)

        positive_score = np.concatenate((p_score, n_score))
        scores = np.column_stack((1 - positive_score, positive_score))
        labels = np.concatenate(
            (np.full(n_pos, classes[1]), np.full(n_neg, classes[0]))
        )

        return scores, labels


class MVNSimulator(ScoreSimulator):
    """Places each class at its simplex vertex and draws around it.

    The draw is a diagonal multivariate normal centred on the vertex, whose
    variance the merging factor sets; the result is folded back onto the
    simplex by taking absolute values and renormalising. Generalises to any
    number of classes.
    """

    def _draw(self, counts, merging_factor, classes, rng):
        n_classes = len(counts)
        merging_factor = np.clip(merging_factor, 0.0, 1.0)
        centers = np.eye(n_classes)

        if np.ndim(merging_factor) == 0:
            var_per_class = np.full(n_classes, float(merging_factor))
        else:
            var_per_class = np.asarray(merging_factor, dtype=float)

        scores, labels = [], []
        for c in range(n_classes):
            cov = np.diag(np.full(n_classes, EPS + var_per_class[c]))
            drawn = rng.multivariate_normal(centers[c], cov, size=counts[c])

            drawn = np.abs(drawn)
            drawn /= drawn.sum(axis=1, keepdims=True)

            scores.append(drawn)
            labels.append(np.full(counts[c], classes[c]))

        return np.vstack(scores), np.concatenate(labels)


class DirichletSimulator(ScoreSimulator):
    """Draws on the simplex directly, from a Dirichlet per class.

    The merging factor sets the concentration: low values pull the draw towards
    the class's vertex, high values towards the uniform interior. Generalises
    to any number of classes.
    """

    def _draw(self, counts, merging_factor, classes, rng):
        n_classes = len(counts)
        # Floored at 0.1 rather than the MVN's 0.0: a Dirichlet concentration
        # has to stay positive, and the mapping below sends 0 to a vertex the
        # draw cannot come back from.
        merging_factor = np.clip(merging_factor, 0.1, 1.0)
        centers = np.eye(n_classes)

        scores, labels = [], []
        for c in range(n_classes):
            if np.ndim(merging_factor) == 0:
                overlap = float(merging_factor)
            else:
                overlap = float(merging_factor[c])

            # Half the range, offset: a merging factor of 0 concentrates on the
            # vertex, 1 spreads across the whole simplex.
            overlap = 0.5 * overlap + 0.5
            high_conc = 100**overlap

            center = centers[c]
            toward_vertex = center * (1 - overlap)

            concentration = (
                (1 - overlap) * (toward_vertex * high_conc)
                + overlap * np.ones(n_classes)
            )

            scores.append(rng.dirichlet(concentration, size=counts[c]))
            labels.append(np.full(counts[c], classes[c]))

        return np.vstack(scores), np.concatenate(labels)
