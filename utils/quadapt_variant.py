from mlquantify.meta import QuaDapt
from mlquantify.utils._validation import validate_prevalences
from utils.moss import (MoSS_MN, MoSS_Dir, MoSS)
import numpy as np

class QuadaptMoSS(QuaDapt):
    
    @classmethod
    def MoSS(cls, n, alpha, merging_factor):
        return MoSS(n=n, alpha=alpha, merging_factor=merging_factor)


class QuadaptMoSS_MN(QuaDapt):
    
    @classmethod
    def MoSS(cls, n, alpha, merging_factor):
        
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
    def MoSS(cls, n, alpha, merging_factor):
        
        if isinstance(alpha, (float, int)):
            alpha = [1-alpha, alpha]
            
        return MoSS_Dir(
            n=n, 
            n_classes=len(alpha), 
            alpha=alpha, 
            merging_factor=merging_factor
        )




class QuadaptMoSS_Dir(QuaDapt):
    
    @classmethod
    def MoSS(cls, n, alpha, merging_factor):
        
        if isinstance(alpha, (float, int)):
            alpha = [1-alpha, alpha]
            
        return MoSS_Dir(
            n=n, 
            n_classes=len(alpha), 
            alpha=alpha, 
            merging_factor=merging_factor
        )


class QuadaptNew(QuaDapt):

    MOSS_VARIANTS = [MoSS, MoSS_MN, MoSS_Dir]
    
    def aggregate(self, predictions, train_y_values):

        self.classes = self.classes if hasattr(self, 'classes') else np.unique(train_y_values)

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