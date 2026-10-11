import os
import openml
import pandas as pd

# ----------------------------
# Configuration
# ----------------------------
OUTPUT_DIR = "openml_datasets3"
MAX_DATASETS = 10

MIN_CLASSES = 3
MIN_INSTANCES = 200
MAX_INSTANCES = 50_000

MIN_RUNS = 50  # relevance threshold

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Step 1: Structural filtering
# ----------------------------
datasets = openml.datasets.list_datasets(output_format="dataframe")

structural_filtered = datasets[
    (datasets["NumberOfClasses"] >= MIN_CLASSES) &
    (datasets["NumberOfInstances"].between(MIN_INSTANCES, MAX_INSTANCES))
]

print(f"After structural filtering: {len(structural_filtered)} datasets")

# ----------------------------
# Step 2: Count runs via CLASSIFICATION tasks
# ----------------------------
run_cache = {}   # task_id -> num_runs
relevant = []

for did in structural_filtered["did"]:
    try:
        # Get classification tasks for this dataset
        tasks = openml.tasks.list_tasks(
            data_id=int(did),
            task_type=openml.tasks.TaskType.SUPERVISED_CLASSIFICATION
        )

        if not tasks:
            continue

        total_runs = 0

        for task_id in tasks.keys():
            task_id = int(task_id)

            if task_id not in run_cache:
                run_cache[task_id] = len(
                    openml.runs.list_runs(task=[task_id])
                )

            total_runs += run_cache[task_id]

            # early stop
            if total_runs >= MIN_RUNS:
                break

        if total_runs >= MIN_RUNS:
            relevant.append({
                "did": did,
                "num_tasks": len(tasks),
                "num_runs": total_runs
            })

    except Exception as e:
        print(f"⚠ Dataset {did}: {e}")

# ----------------------------
# Step 3: Rank by relevance
# ----------------------------
relevant_df = pd.DataFrame(relevant)

relevant_df = relevant_df.sort_values(
    by=["num_runs", "num_tasks"],
    ascending=[False, False]
)

dataset_ids = relevant_df["did"].head(MAX_DATASETS).tolist()

print("Selected relevant datasets:")
print(relevant_df.head(10))

# ----------------------------
# Step 4: Download datasets
# ----------------------------
for did in dataset_ids:
    try:
        dataset = openml.datasets.get_dataset(did)

        X, y, *_ = dataset.get_data(
            dataset_format="dataframe",
            target=dataset.default_target_attribute
        )

        df = X.copy()
        df["target"] = y

        df.to_csv(
            f"{OUTPUT_DIR}/dataset_{did}_{dataset.name}.csv",
            index=False
        )

        print(f"✔ Downloaded {dataset.name} | shape={df.shape}")

    except Exception as e:
        print(f"✖ Failed dataset {did}: {e}")
