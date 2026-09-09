import os
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from utils.meta_quantifier import QuaDaptWithSimulator
from variables import *

def run_experiment(m_train,
                   m_test,
                   alpha,
                   train_simulator,
                   test_simulator,
                   train_simulator_name,
                   test_simulator_name,
                   random_state=None,
                   strict=False,
                   method_simulators=None):
    """Run one grid cell of the sweep and return its runs as a frame.

    The last three arguments exist for the test suite and all default to the
    sweep's own behaviour: ``random_state=None`` draws from OS entropy as
    before, ``strict=False`` keeps the caught-and-logged error handling, and
    ``method_simulators=None`` uses the full registry. Note that seeding here
    reaches only this experiment's own data simulators — the method
    simulators inside a meta-quantifier draw independently (ADR-0004).
    """

    results = []
    rng = np.random.default_rng(random_state)
    if method_simulators is None:
        method_simulators = METHOD_SIMULATORS

    # The prevalence goes in as the positive class's share; the simulator reads
    # it into a vector itself, so neither the sweep nor anyone else spells out
    # the negative class.
    train_scores, train_labels = train_simulator(
        n=TRAIN_SIZE,
        alpha=0.5,
        merging_factor=m_train,
        random_state=rng,
    )

    for i in range(N_REPETITIONS):
        test_scores, test_labels = test_simulator(
            n=TEST_SIZE,
            alpha=alpha,
            merging_factor=m_test,
            random_state=rng,
        )

        for method_simulator_name, method_simulator in method_simulators.items():
            for qtf_name, quantifier in QUANTIFIERS.items():
                try:
                    if qtf_name == "CC":
                        prediction = quantifier().aggregate(test_scores)[1]
                    elif method_simulator_name != "None":
                        prediction = QuaDaptWithSimulator(
                            quantifier(),
                            method_simulator,
                        ).aggregate(
                            test_scores,
                            train_labels
                        )[1]
                    else:
                        prediction = quantifier().aggregate(
                            test_scores,
                            train_scores,
                            train_labels,
                        )[1]
                except Exception as e:
                    if strict:
                        raise
                    import traceback
                    print(f"Error in {qtf_name} with {method_simulator_name}: {e}")
                    print(
                        "mtr:", m_train,
                        "\nmtest:", m_test,
                        "\nalpha:", alpha,
                        "\nqtf:", qtf_name,
                        "\nmethod simulator:", method_simulator_name,
                        "\ntrain simulator:", train_simulator_name,
                        "\ntest simulator:", test_simulator_name
                    )
                    traceback.print_exc()
           

                # Still a mapping in 0.5.1, unlike aggregate's return above.
                real_prev = get_prev_from_labels(test_labels)
                real_prev = list(real_prev.values())[1]
                mae = np.mean(np.abs(prediction - real_prev))

                # The column names are frozen by the golden record and by every
                # result file already written; the registry keys they take
                # their values from are frozen with them. See variables.py.
                results.append({
                    "Quantifier": qtf_name,
                    "Quadapt_Variant": method_simulator_name,
                    "MoSS_Test_Variant": test_simulator_name,
                    "MoSS_Train_Variant": train_simulator_name,
                    "MAE": mae,
                    "m_test": m_test,
                    "m_train": m_train,
                    "alpha": alpha,
                    "Iteration": i + 1,
                })

    return pd.DataFrame(results)


def main(results_path):

    # 1) gerar TODAS as combinações de parâmetros, incluindo os simuladores de dados
    param_grid = []
    for train_simulator_name, train_simulator in DATA_SIMULATORS.items():
        for test_simulator_name, test_simulator in DATA_SIMULATORS.items():
            for m_train in MERGING_FACTORS:
                for m_test in MERGING_FACTORS:
                    for alpha in ALPHAS:
                        param_grid.append((
                            m_train,
                            m_test,
                            alpha,
                            train_simulator,
                            test_simulator,
                            train_simulator_name,
                            test_simulator_name,
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
