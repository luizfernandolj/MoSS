from mlquantify.meta import QuaDapt
from moss import (MoSS_MN, MoSS_Dir, MoSS)

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