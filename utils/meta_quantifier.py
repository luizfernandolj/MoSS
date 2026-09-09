"""The meta-quantifier, told which method simulator to use.

mlquantify hard-codes the uniform simulator as a ``MoSS`` classmethod on
``QuaDapt``, so the only way to give it a different one used to be a subclass
per simulator. Those three subclasses differed from one another in nothing but
how they reshaped the prevalence before delegating — reshaping the simulators
now do themselves. What is left is one class and one argument.
"""

import numpy as np
from mlquantify.matching import DyS, SORD
from mlquantify.meta import QuaDapt
from mlquantify.utils import validate_prevalences

from utils.simulators import DirichletSimulator, MVNSimulator, UniformSimulator


class QuaDaptWithSimulator(QuaDapt):
    """``QuaDapt`` whose method simulator is a constructor argument.

    The override exists only to route the library's ``MoSS`` seam to the given
    simulator. It passes the arguments straight through, because the simulators
    take the same ones the library does (ADR-0002) — including ``classes`` and
    ``random_state``, which the old subclasses accepted and dropped on the
    floor.
    """

    def __init__(self,
                 quantifier,
                 method_simulator,
                 measure="topsoe",
                 merging_factors=np.arange(0.1, 1.0, 0.2),
                 strategy="ovr"):
        super().__init__(
            quantifier=quantifier,
            measure=measure,
            merging_factors=merging_factors,
            strategy=strategy,
        )
        self.method_simulator = method_simulator

    def MoSS(self, n, alpha, merging_factor, classes=None, random_state=None):
        return self.method_simulator(n, alpha, merging_factor, classes, random_state)


class QuadaptNew(QuaDapt):
    """Unreferenced under mlquantify 0.5.1 — deliberately not in the registry.

    ``best_mixture`` below calls the same method on a matching quantifier
    instance, a helper copied from mlquantify 0.2.0 internals that 0.5.1
    removed. Rather than re-adapt copied internals, this method simulator is
    dropped from the registry and the class left here untouched. Its results
    were void under ADR-0001 regardless, so nothing is lost — but its absence
    from the results must not be read as a finding. See ADR-0007, which scopes
    the rewrite of this ``best_mixture`` against 0.5.1's API to its own change;
    the simulator interface it draws through is the current one.
    """

    METHOD_SIMULATORS = [UniformSimulator(), MVNSimulator(), DirichletSimulator()]

    def aggregate(self, predictions, train_labels, train_scores):

        self.classes = self.classes if hasattr(self, 'classes') else np.unique(train_labels)

        self.pos_scores = train_scores[train_labels == 1][:, 1]
        self.neg_scores = train_scores[train_labels == 0][:, 1]

        distances = []
        alphas = []

        for method_simulator in self.METHOD_SIMULATORS:
            self.MoSS = method_simulator

            alpha, distance, _ = self.best_mixture(predictions)
            distances.append(distance)
            alphas.append(alpha)

        prevalence = alphas[np.argmin(distances)]
        prevalences = np.asarray([1-prevalence, prevalence])

        prevalences = validate_prevalences(self, prevalences, self.classes)

        return prevalences


    def best_mixture(self, predictions):
        predictions = predictions[:, 1]

        MF = np.atleast_1d(np.round(self.merging_factors, 2)).astype(float)
        MF = np.insert(MF, 0, 0.0)


        distances = []
        alphas = []


        if self.measure in ["hellinger", "topsoe", "probsymm"]:
            method = DyS(measure=self.measure)
        elif self.measure == "sord":
            method = SORD()

        train_alpha, train_distance = method.best_mixture(predictions, self.pos_scores, self.neg_scores)

        distances.append(train_distance)
        alphas.append(train_alpha)

        for mf in MF[1:]:
            scores, labels = self.MoSS(n=1000, alpha=0.5, merging_factor=mf)
            pos_scores = scores[labels == 1][:, 1]
            neg_scores = scores[labels == 0][:, 1]

            if self.measure in ["hellinger", "topsoe", "probsymm"]:
                method = DyS(measure=self.measure)
            elif self.measure == "sord":
                method = SORD()

            alpha, distance = method.best_mixture(predictions, pos_scores, neg_scores)

            distances.append(distance)
            alphas.append(alpha)


        best_m = MF[np.argmin(distances)]
        best_alpha = alphas[np.argmin(distances)]
        best_distance = np.min(distances)
        return best_alpha, best_distance, best_m
