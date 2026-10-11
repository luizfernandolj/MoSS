import os
import openml
import pandas as pd

# ----------------------------
# Configuration
# ----------------------------
OUTPUT_DIR = "uci_multiclass_high_quality"
MAX_DATASETS = 10

MIN_CLASSES = 3
MIN_INSTANCES = 200
MAX_INSTANCES = 50_000

MIN_FEATURES = 4
MAX_FEATURES = 200

MIN_TASKS = 10        # popularity filter
MIN_RUNS = 100        # popularity filter
MAX_MISSING = 0       # clean datasets only

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# List datasets
# ----------------------------
datasets = openml.datasets.list_datasets(output_format="dataframe")

# Check available columns
print("Available columns:", datasets.columns.tolist())
print()

# ----------------------------
# High-quality filtering
# ----------------------------
filtered = datasets[
    (datasets["NumberOfClasses"] >= MIN_CLASSES) &
    (datasets["NumberOfInstances"].between(MIN_INSTANCES, MAX_INSTANCES)) &
    (datasets["NumberOfFeatures"].between(MIN_FEATURES, MAX_FEATURES)) &
    (datasets["NumberOfTasks"] >= MIN_TASKS) &
    (datasets["NumberOfRuns"] >= MIN_RUNS) &
    (datasets["NumberOfMissingValues"] <= MAX_MISSING) &
    (datasets["DefaultTargetAttribute"].notna())
]

# Rank by relevance
filtered = filtered.sort_values(
    by=["NumberOfTasks", "NumberOfRuns", "NumberOfClasses"],
    ascending=[False, False, False]
)

dataset_ids = filtered["did"].head(MAX_DATASETS).tolist()

print("Selected high-quality UCI multiclass datasets:")
print(filtered[["did", "name", "NumberOfClasses", "NumberOfTasks", "NumberOfRuns"]].head(MAX_DATASETS))

# ----------------------------
# Download and save
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

        filename = f"{OUTPUT_DIR}/uci_{did}_{dataset.name}.csv"
        df.to_csv(filename, index=False)

        print(f"✔ {dataset.name} | shape={df.shape}")

    except Exception as e:
        print(f"✖ Failed dataset {did}: {e}")
