import os
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from variables import *

def run_experiment(m_train,
                   m_test,
                   alpha,
                   moss_train_variant,
                   moss_test_variant,
                   moss_train_variant_name,
                   moss_test_variant_name):

    results = []

    train_scores, train_labels = moss_train_variant(
        n=TRAIN_SIZE,
        alpha=[0.5, 0.5],
        merging_factor=m_train,
    )

    for i in range(N_REPETITIONS):
        test_scores, test_labels = moss_test_variant(
            n=TEST_SIZE,
            alpha=[1 - alpha, alpha],
            merging_factor=m_test,
        )

        for quadapt_variant_name, quadapt_variant in QUADAPT_VARIANTS.items():
            for qtf_name, quantifier in QUANTIFIERS.items():
                try:
                    if qtf_name == "CC":
                        prediction = quantifier().aggregate(test_scores)[1]
                    elif quadapt_variant_name != "None":
                        prediction = quadapt_variant(quantifier()).aggregate(test_scores, 
                                                                                [0, 1])
                        prediction = list(prediction.values())[1]
                    else:
                        prediction = quantifier().aggregate(
                            test_scores,
                            train_scores,
                            train_labels,
                        )
                        prediction = list(prediction.values())[1]
                except Exception as e:
                    print(f"Error in {qtf_name} with {quadapt_variant_name}: {e}")

                real_prev = get_prev_from_labels(test_labels)
                real_prev = list(real_prev.values())[1]
                mae = np.mean(np.abs(prediction - real_prev))

                results.append({
                    "Quantifier": qtf_name,
                    "Quadapt_Variant": quadapt_variant_name,
                    "MoSS_Test_Variant": moss_test_variant_name,
                    "MoSS_Train_Variant": moss_train_variant_name,
                    "MAE": mae,
                    "m_test": m_test,
                    "m_train": m_train,
                    "alpha": alpha,
                    "Iteration": i + 1,
                })

    return pd.DataFrame(results)


def main(results_path):

    # 1) gerar TODAS as combinações de parâmetros, incluindo os nomes/variantes do MoSS
    param_grid = []
    for moss_train_variant_name, moss_train_variant in MOSS_VARIANTS.items():
        for moss_test_variant_name, moss_test_variant in MOSS_VARIANTS.items():
            for m_train in MERGING_FACTORS:
                for m_test in MERGING_FACTORS:
                    for alpha in ALPHAS:
                        param_grid.append((
                            m_train,
                            m_test,
                            alpha,
                            moss_train_variant,
                            moss_test_variant,
                            moss_train_variant_name,
                            moss_test_variant_name,
                        ))

    # 2) rodar em paralelo com joblib + tqdm
    # n_jobs=-1 usa todos os cores disponíveis; ajuste se quiser.
    dfs = list(
        tqdm(
            Parallel(
                n_jobs=-1,
                backend="loky",          # padrão recomendado para CPU-bound + sklearn [web:46][web:49]
                return_as="generator",   # permite usar tqdm em cima do gerador [web:61][web:60]
            )(
                delayed(run_experiment)(*args)
                for args in param_grid
            ),
            total=len(param_grid),
            desc="Rodando experimentos",
            colour="blue"
        )
    )  # [web:46][web:60][web:61]

    # 3) concatenar todos os resultados de uma vez
    final_results = pd.concat(dfs, ignore_index=True)

    final_results.to_csv(results_path, index=False)


if __name__ == "__main__":
    results_path = "results/results.csv"
    main(results_path)
