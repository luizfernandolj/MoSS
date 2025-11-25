import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from moss import (
    MoSS_MN, 
    MoSS_Dir, 
    MoSS
)

from quadapt_variant import (
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
    MS2
)
from mlquantify.mixture import (
    DyS,
    HDy,
    SORD,
    SMM    
)
from mlquantify.metrics import MAE
from mlquantify.utils import get_prev_from_labels

MERGING_FACTORS = [0.1, 0.25, 0.5, 0.75, 1.0] # merging factors
ALPHAS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0] # positive class proportions
QuaDaptVariants = { # Variants of MoSS for QuaDapt Framework
    "MoSS_Dir": QuadaptMoSS_Dir,
    "MoSS": QuadaptMoSS,
    "MoSS_MN": QuadaptMoSS_MN,
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
}
SIZE_OF_SAMPLES = 1000
    
def run_experiment(quadapt, quadapt_variant_test_name):
    print(f"Running experiments for {quadapt_variant_test_name}...")
    
    results = pd.DataFrame(columns=[
        "Quantifier",
        "MoSS Variant Test",
        "MoSS Variant Train",
        "m_proximity",
        "MAE",
        "m_test",
        "alpha"
    ])
    
    for quadapt_variant_train_name, quadapt_train in QuaDaptVariants.items():
        print(f" Training with {quadapt_variant_train_name}...")
        for qtf_name, quantifier in QUANTIFIERS.items():
            for m_test in MERGING_FACTORS:
                for alpha in ALPHAS:
                    for _ in range(3):
                        quadapt_i = quadapt_train(
                        quantifier=quantifier(), 
                        merging_factors=MERGING_FACTORS
                    )
                    
                    test_scores, test_labels = quadapt.MoSS(
                        SIZE_OF_SAMPLES,
                        alpha=alpha,
                        m=m_test
                    )
                    prediction = quadapt_i.aggregate(test_scores, [0, 1])
                    best_m = quadapt_i._get_best_merging_factor(test_scores)
                    
                    mae = np.mean(np.abs(prediction[1] - alpha))  #
                    if mae > 1:
                        import pdb 
                        pdb.set_trace()
                        mae = 1.0
                        print(prediction, alpha)
                        
                    m_proximity = abs(best_m - m_test)
                
                    result_ = {
                        "Quantifier": qtf_name,
                        "MoSS Variant Test": quadapt_variant_test_name,
                        "MoSS Variant Train": quadapt_variant_train_name,
                        "m_proximity": m_proximity,
                        "MAE": mae,
                        "m_test": m_test,
                        "alpha": alpha
                    }
                    results = pd.concat(
                        [
                            results if not results.empty else None, 
                            pd.DataFrame([result_])
                        ], 
                        ignore_index=True)
    results.reset_index(drop=True, inplace=True)
    results.to_csv(f"results/results_{quadapt_variant_name}.csv", index=False)
    print(f"Finished experiments for {quadapt_variant_name}.\n")

if __name__ == "__main__":
    
    # Parallel(n_jobs=-1)(
    #     delayed(run_experiment)(quadapt, quadapt_variant_name)
    #     for quadapt_variant_name, quadapt in QuaDaptVariants.items()
    # )
    
    for quadapt_variant_name, quadapt in QuaDaptVariants.items():
        run_experiment(quadapt, quadapt_variant_name)
