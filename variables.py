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
    QuadaptMoSS_Dir,
)
from mlquantify.counting import (
    TAC,
    TX,
    T50,
    TMAX,
    MS,
    MS2,
    CC
)
from mlquantify.matching import (
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
    "Quadapt_MoSS": QuadaptMoSS,
    "Quadapt_MvN": QuadaptMoSS_MN,
    "Quadapt_Dir": QuadaptMoSS_Dir,
    # "QuadaptNew" is absent on purpose: it calls a mixture-search helper that
    # 0.5.1 removed. Its absence from the results is not a finding — ADR-0007.
    "None": None
}
QUANTIFIERS = { # Quantifiers for QuaDapt Framework
    "DyS": DyS,
    "HDy": HDy,
    "SORD": SORD,
    "SMM": SMM,
    "TAC": TAC,
    "TX": TX,
    "T50": T50,
    "TMAX": TMAX,
    "TMS": MS,
    "TMS2": MS2,
    "CC": CC,
}


MERGING_FACTORS = np.arange(0.05, 1.0, 0.05) # merging factors
ALPHAS = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 0.99] # positive class proportions
#MERGING_FACTORS = [0.1, 0.5, 0.8, 0.9]
#ALPHAS = [0.25, 0.5, 0.75]

TRAIN_SIZE = 2000
TEST_SIZE = 100
N_REPETITIONS = 3