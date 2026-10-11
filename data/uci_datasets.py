import os
import pandas as pd
from ucimlrepo import fetch_ucirepo, list_available_datasets

# ----------------------------
# Configuration
# ----------------------------
OUTPUT_DIR = "uci_datasets"
MAX_DATASETS = 30          # how many datasets to download
MAX_INSTANCES = 50_000     # avoid very large datasets
MIN_INSTANCES = 200        # avoid toy datasets
MIN_CLASSES = 3            # multiclass only

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# List and filter datasets
# ----------------------------
# Get list of all available datasets
all_datasets = list_available_datasets()

# Check if the result is valid
if all_datasets is None:
    print("Error: Could not retrieve dataset list from UCI repository")
    print("Falling back to manual list of known UCI dataset IDs")
    # Use a curated list of known multiclass classification dataset IDs
    dataset_ids = [
        53, 73, 109, 116, 162, 163, 164, 165, 166, 167, 
        168, 169, 170, 171, 172, 173, 174, 175, 176, 177,
        178, 179, 186, 187, 188, 189, 193, 235, 236, 237
    ]
    filtered = pd.DataFrame({'id': dataset_ids, 'name': [f'Dataset_{i}' for i in dataset_ids]})
elif len(all_datasets) == 0:
    print("Error: No datasets available from UCI repository")
    filtered = pd.DataFrame({'id': [], 'name': []})
else:
    print(f"Total UCI datasets available: {len(all_datasets)}")
    print("Available columns:", all_datasets.columns.tolist())
    
    # Filter for classification datasets with appropriate characteristics
    filtered = all_datasets[
        (all_datasets["task"] == "Classification") &
        (all_datasets["num_instances"] >= MIN_INSTANCES) &
        (all_datasets["num_instances"] <= MAX_INSTANCES)
    ].copy()
    
    print(f"Found {len(filtered)} classification datasets matching size criteria")

# ----------------------------
# Download and filter datasets
# ----------------------------
downloaded_count = 0

for idx, row in filtered.iterrows():
    if downloaded_count >= MAX_DATASETS:
        break
    
    dataset_id = row["id"]
    dataset_name = row["name"]
    
    try:
        # Fetch the dataset
        dataset = fetch_ucirepo(id=dataset_id)
        
        # Get features and targets
        X = dataset.data.features
        y = dataset.data.targets
        
        # Check if all features are numerical
        if not all(X.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))):
            print(f"⊘ Skipped {dataset_name}: contains non-numerical features")
            continue
        
        # Check number of classes
        if y.shape[1] > 0:
            n_classes = y.iloc[:, 0].nunique()
            if n_classes < MIN_CLASSES:
                print(f"⊘ Skipped {dataset_name}: only {n_classes} classes")
                continue
        else:
            print(f"⊘ Skipped {dataset_name}: no target variable")
            continue
        
        # Combine features and target
        df = X.copy()
        df["target"] = y.iloc[:, 0]
        
        # Save to CSV
        filename = f"{OUTPUT_DIR}/dataset_{dataset_id}_{dataset_name.replace(' ', '_')}.csv"
        df.to_csv(filename, index=False)
        
        downloaded_count += 1
        print(f"✔ Downloaded {dataset_name} | shape={df.shape} | classes={n_classes}")
        
    except Exception as e:
        print(f"✖ Failed dataset {dataset_name} (id={dataset_id}): {e}")

print(f"\nSuccessfully downloaded {downloaded_count} datasets to {OUTPUT_DIR}/")
