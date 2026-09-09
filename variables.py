import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from utils.simulators import (
    DirichletSimulator,
    MVNSimulator,
    UniformSimulator,
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




# The same three simulators appear in both registries, in their two roles
# (CONTEXT.md). The keys are not names — they are the values written to the
# ``MoSS_Train_Variant``, ``MoSS_Test_Variant`` and ``Quadapt_Variant`` columns
# of every result file and of the golden record, so they stay as they are even
# where the glossary would have them read differently.

#: Score simulators in their data role: the source of the sweep's own scores.
DATA_SIMULATORS = {
    "MoSS_Dir": DirichletSimulator(),
    "MoSS": UniformSimulator(),
    "MoSS_MN": MVNSimulator(),
}

#: Score simulators in their method role: what a meta-quantifier draws its
#: candidate reference sets with. ``None`` is the arm that uses no
#: meta-quantifier at all.
METHOD_SIMULATORS = {
    "Quadapt_MoSS": UniformSimulator(),
    "Quadapt_MvN": MVNSimulator(),
    "Quadapt_Dir": DirichletSimulator(),
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