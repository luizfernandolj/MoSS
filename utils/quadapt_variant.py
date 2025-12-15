from mlquantify.meta import QuaDapt
from utils.moss import (MoSS_MN, MoSS_Dir, MoSS)

class QuadaptMoSS(QuaDapt):
    
    @classmethod
    def MoSS(cls, n, alpha, m):
        return MoSS(n=n, alpha=alpha, merging_factor=m)


class QuadaptMoSS_MN(QuaDapt):
    
    @classmethod
    def MoSS(cls, n, alpha, m):
        
        if isinstance(alpha, (float, int)):
            alpha = [1-alpha, alpha]
            
        return MoSS_MN(
            n=n, 
            n_classes=len(alpha), 
            alpha=alpha,
            merging_factor=m
        )
        
    
class QuadaptMoSS_Dir(QuaDapt):
    
    @classmethod
    def MoSS(cls, n, alpha, m):
        
        if isinstance(alpha, (float, int)):
            alpha = [1-alpha, alpha]
            
        return MoSS_Dir(
            n=n, 
            n_classes=len(alpha), 
            alpha=alpha, 
            merging_factor=m
        )




class QuadaptNew(QuaDapt):

    MOSS_VARIANTS = [MoSS, MoSS_MN, MoSS_Dir]
    
    def aggregate(self, predictions, train_y_values):

        self.classes = self.classes if hasattr(self, 'classes') else np.unique(train_y_values)

        distances = []

        for Moss_Variant in self.MOSS_VARIANTS:
            self.MoSS = Moss_Variant
            
            distance = self.get_best_distance(predictions)
            distances.append(distance)

        moss = self.MOSS_VARIANTS[np.argmin(distances)]
        self.MoSS = moss

        moss_scores, moss_labels = self.MoSS(1000, 0.5, m)

        prevalences = self.quantifier.aggregate(predictions,
                                                moss_scores,
                                                moss_labels)
        
        prevalences = {self.classes[i]: v for i, v in enumerate(prevalences.values())}
        return prevalences