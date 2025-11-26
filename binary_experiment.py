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

MERGING_FACTORS = np.arange(0.05, 1.0, 0.05) # merging factors
ALPHAS = [0.1, 0.2, 0.4, 0.6, 0.8, 0.99] # positive class proportions
MOSS_VARIANTS = {
    #"MoSS_Dir": MoSS_Dir,
    "MoSS": MoSS,
    "MoSS_MN": MoSS_MN,
}
QuaDaptVariants = { # Variants of MoSS for QuaDapt Framework]
    "MoSS": QuadaptMoSS,
    "MoSS_MN": QuadaptMoSS_MN,
    #"MoSS_Dir": QuadaptMoSS_Dir,
}
QUANTIFIERS = { # Quantifiers for QuaDapt Framework
    "DyS": DyS(),
    "HDy": HDy(),
    "SORD": SORD(),
    "SMM": SMM(),
    "ACC": ACC(),
    "X_method": X_method(),
    #"T50": T50(),
    "MAX": MAX(),
    "MS": MS(),
    "MS2": MS2(),
    "CC": CC(),
    "QuadaptMoSS_DyS": QuadaptMoSS(DyS()),
    "QuadaptMoSS_HDy": QuadaptMoSS(HDy()),
    "QuadaptMoSS_SORD": QuadaptMoSS(SORD()),
    "QuadaptMoSS_SMM": QuadaptMoSS(SMM()),
    "QuadaptMoSS_ACC": QuadaptMoSS(ACC()),
    "QuadaptMoSS_X_method": QuadaptMoSS(X_method()),
    #"QuadaptMoSS_T50": QuaDapt(T50()),
    "QuadaptMoSS_MAX": QuadaptMoSS(MAX()),
    "QuadaptMoSS_MS": QuadaptMoSS(MS()),
    "QuadaptMoSS_MS2": QuadaptMoSS(MS2()),
    "QuadaptMoSS_MN_DyS": QuadaptMoSS_MN(DyS()),
    "QuadaptMoSS_MN_HDy": QuadaptMoSS_MN(HDy()),
    "QuadaptMoSS_MN_SORD": QuadaptMoSS_MN(SORD()),
    "QuadaptMoSS_MN_SMM": QuadaptMoSS_MN(SMM()),
    "QuadaptMoSS_MN_ACC": QuadaptMoSS_MN(ACC()),
    "QuadaptMoSS_MN_X_method": QuadaptMoSS_MN(X_method()),
    #"QuadaptMoSS_MN_T50": QuadaptMoSS_MN(T50
    "QuadaptMoSS_MN_MAX": QuadaptMoSS_MN(MAX()),
    "QuadaptMoSS_MN_MS": QuadaptMoSS_MN(MS()),
    "QuadaptMoSS_MN_MS2": QuadaptMoSS_MN(MS2()),

    # "QuadaptMoSS_Dir_DyS": QuadaptMoSS_Dir(DyS()),
    # "QuadaptMoSS_MN_DyS": QuadaptMoSS_MN(DyS()),
    # 
    # "QuadaptMoSS_Dir_HDy": QuadaptMoSS_Dir(HDy()),
    # "QuadaptMoSS_MN_HDy": QuadaptMoSS_MN(HDy()),
    # 
    # "QuadaptMoSS_Dir_SORD": QuadaptMoSS_Dir(SORD()),
    # "QuadaptMoSS_MN_SORD": QuadaptMoSS_MN(SORD()),
    # "QuadaptMoSS_SORD": QuadaptMoSS(SORD()),
    # "QuadaptMoSS_Dir_SMM": QuadaptMoSS_Dir(SMM()),
    # "QuadaptMoSS_MN_SMM": QuadaptMoSS_MN(SMM()),
    # "QuadaptMoSS_SMM": QuadaptMoSS(SMM()),
    # "QuadaptMoSS_Dir_ACC": QuadaptMoSS_Dir(ACC()),
    # "QuadaptMoSS_MN_ACC": QuadaptMoSS_MN(ACC()),
    # "QuadaptMoSS_ACC": QuadaptMoSS(ACC()),
    # "QuadaptMoSS_Dir_X_method": QuadaptMoSS_Dir(X_method()),
    # "QuadaptMoSS_MN_X_method": QuadaptMoSS_MN(X_method()),
    # "QuadaptMoSS_X_method": QuadaptMoSS(X_method()),
    # "QuadaptMoSS_Dir_T50": QuadaptMoSS_Dir(T50()),
    # "QuadaptMoSS_MN_T50": QuadaptMoSS_MN(T50()),
    # "QuadaptMoSS_T50": QuadaptMoSS(T50()),
    # "QuadaptMoSS_Dir_MAX": QuadaptMoSS_Dir(MAX()),
    # "QuadaptMoSS_MN_MAX": QuadaptMoSS_MN(MAX()),
    # "QuadaptMoSS_MAX": QuadaptMoSS(MAX()),
    # "QuadaptMoSS_Dir_MS": QuadaptMoSS_Dir(MS()),
    # "QuadaptMoSS_MN_MS": QuadaptMoSS_MN(MS()),
    # "QuadaptMoSS_MS": QuadaptMoSS(MS()),
    # "QuadaptMoSS_Dir_MS2": QuadaptMoSS_Dir(MS2()),
    # "QuadaptMoSS_MN_MS2": QuadaptMoSS_MN(MS2()),
    # "QuadaptMoSS_MS2": QuadaptMoSS(MS2()),
}
SIZE_OF_TRAIN = 2000
SIZE_OF_TEST = 100
    
def run_experiment_1(quadapt, quadapt_variant_test_name):
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
        for qtf_name, quantifier in tqdm(QUANTIFIERS.items()):
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
                    try:
                        prediction = quadapt_i.aggregate(test_scores, [0, 1])
                    except Exception as e:
                        import pdb
                        pdb.set_trace()
                        
                    best_m = quadapt_i._get_best_merging_factor(test_scores)
                    
                    mae = np.mean(np.abs(prediction[1] - alpha))  #
                    print(mae)
                        
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
    
    
def run_experiment_2(m_train, moss_variant):
    
    train_scores, train_labels = moss_variant(
        n=SIZE_OF_TRAIN,
        merging_factor=m_train
    )
    
    results = pd.DataFrame(columns=[
        "Quantifier",
        "MoSS Variant Test",
        "MoSS Variant Train", 
        "MAE",
        "m_test",
        "m_train",
        "alpha"
    ])
    
    for moss_test_variant_name, moss_test_variant in tqdm(MOSS_VARIANTS.items()):
        for m_test in MERGING_FACTORS:
            for alpha in ALPHAS:
                for _ in range(3):
                    
                    test_scores, test_labels = moss_test_variant(
                        SIZE_OF_TEST,
                        alpha=[1-alpha, alpha],
                        merging_factor=m_test
                    )
                    
                    for qtf_name, quantifier in QUANTIFIERS.items():
                        if qtf_name.startswith("Quadapt"):
                            prediction = quantifier.aggregate(test_scores, [0, 1])
                        elif qtf_name == "CC":
                            prediction = quantifier.aggregate(test_scores)
                        else:
                            prediction = quantifier.aggregate(test_scores, train_scores, train_labels)
                        real = get_prev_from_labels(test_labels)[1]
                        
                        mae = np.mean(np.abs(prediction[1] - real))
                        
                        result_ = {
                            "Quantifier": qtf_name,
                            "MoSS Variant Test": moss_test_variant_name,
                            "MoSS Variant Train": moss_variant.__name__,
                            "MAE": mae,
                            "m_test": m_test,
                            "m_train": m_train,
                            "alpha": alpha
                        }
                        results = pd.concat(
                            [
                                results if not results.empty else None, 
                                pd.DataFrame([result_])
                            ], 
                            ignore_index=True)
                        
    results.reset_index(drop=True, inplace=True)
    return results
    

if __name__ == "__main__":
    
    # Parallel(n_jobs=-1)(
    #     delayed(run_experiment)(quadapt, quadapt_variant_name)
    #     for quadapt_variant_name, quadapt in QuaDaptVariants.items()
    # )
    
    #for quadapt_variant_name, quadapt in QuaDaptVariants.items():
        #run_experiment_1(quadapt, quadapt_variant_name)
    
    result = pd.DataFrame()
    for moss_variant_name, moss_variant in MOSS_VARIANTS.items():
        result_ = run_experiment_2(
            m_train=0.5,
            moss_variant=moss_variant
        )
        result = pd.concat(
            [
                result if not result.empty else None, result_
            ],
            ignore_index=True)
    
    result.to_csv("results2/results_MoSS.csv", index=False)