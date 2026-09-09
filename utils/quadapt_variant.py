from mlquantify.meta import QuaDapt
from mlquantify.utils import (
    validate_prevalences, 
    validate_data, 
    apply_cross_validation
)
from mlquantify.base_aggregative import uses_soft_predictions, get_aggregation_requirements
from mlquantify.matching import DyS, SORD
from utils.moss import (MoSS_MN, MoSS_Dir, MoSS)
import numpy as np

# Each override matches mlquantify's own ``MoSS(n, alpha, merging_factor,
# classes, random_state)`` signature exactly (ADR-0002). The library calls it
# with ``classes`` when it builds the reference score set, so a narrower
# signature raises TypeError on contact. ``random_state`` is accepted and
# unused, as upstream documents it — see ADR-0004 for the gap that leaves.


class QuadaptMoSS(QuaDapt):

    @classmethod
    def MoSS(cls, n, alpha, merging_factor, classes=None, random_state=None):
        return MoSS(n=n, alpha=alpha, merging_factor=merging_factor)


class QuadaptMoSS_MN(QuaDapt):

    @classmethod
    def MoSS(cls, n, alpha, merging_factor, classes=None, random_state=None):

        if isinstance(alpha, (float, int)):
            alpha = [1-alpha, alpha]
        return MoSS_MN(
            n=n,
            n_classes=len(alpha),
            alpha=alpha,
            merging_factor=merging_factor
        )


class QuadaptMoSS_Dir(QuaDapt):

    @classmethod
    def MoSS(cls, n, alpha, merging_factor, classes=None, random_state=None):

        if isinstance(alpha, (float, int)):
            alpha = [1-alpha, alpha]

        return MoSS_Dir(
            n=n,
            n_classes=len(alpha),
            alpha=alpha,
            merging_factor=merging_factor
        )


class QuadaptNew(QuaDapt):
    """Unreferenced under mlquantify 0.5.1 — deliberately not in the registry.

    ``best_mixture`` below calls the same method on a matching quantifier
    instance, a helper copied from mlquantify 0.2.0 internals that 0.5.1
    removed. Rather than re-adapt copied internals for code the follow-up
    refactor rewrites anyway, this method simulator is dropped from
    QUADAPT_VARIANTS for the duration of the port and the class left here
    untouched. Its results were void under ADR-0001 regardless, so nothing is
    lost — but its absence from the results must not be read as a finding.
    See ADR-0007.
    """

    MOSS_VARIANTS = [MoSS, MoSS_MN, MoSS_Dir]
    
    def aggregate(self, predictions, train_labels, train_scores):

        self.classes = self.classes if hasattr(self, 'classes') else np.unique(train_labels)

        self.pos_scores = train_scores[train_labels == 1][:, 1]
        self.neg_scores = train_scores[train_labels == 0][:, 1]

        distances = []
        alphas = []

        for Moss_Variant in self.MOSS_VARIANTS:
            self.MoSS = Moss_Variant
            
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