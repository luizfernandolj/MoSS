import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from utils.moss import (
    MoSS_MN, 
    MoSS_Dir, 
    MoSS
)

from utils.quadapt_variant import (
    QuadaptMoSS,
    QuadaptMoSS_MN,
    QuadaptMoSS_Dir
)
from mlquantify.adjust_counting import (
    ACC,
    X_method,
    T50,
    MAX,
    MS,
    MS2,
    CC
)
from mlquantify.mixture import (
    DyS,
    HDy,
    SORD,
    SMM    
)
from mlquantify.metrics import MAE
from mlquantify.meta import QuaDapt
from mlquantify.utils import get_prev_from_labels




MOSS_VARIANTS = {
    "MoSS_Dir": MoSS_Dir,
    "MoSS": MoSS,
    "MoSS_MN": MoSS_MN,
}
QUADAPT_VARIANTS = { # Variants of MoSS for QuaDapt Framework]
    "MoSS": QuadaptMoSS,
    "MoSS_MN": QuadaptMoSS_MN,
    "MoSS_Dir": QuadaptMoSS_Dir,
    "None": None
}
QUANTIFIERS = { # Quantifiers for QuaDapt Framework
    "DyS": DyS,
    "HDy": HDy,
    "SORD": SORD,
    "SMM": SMM,
    "ACC": ACC,
    "X_method": X_method,
    "T50": T50,
    "MAX": MAX,
    "MS": MS,
    "MS2": MS2,
    "CC": CC,
}


MERGING_FACTORS = np.arange(0.05, 1.0, 0.05) # merging factors
ALPHAS = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 0.99] # positive class proportions
#MERGING_FACTORS = [0.1, 0.5, 0.8, 0.9]
#ALPHAS = [0.25, 0.5, 0.75]

TRAIN_SIZE = 2000
TEST_SIZE = 100
N_REPETITIONS = 3