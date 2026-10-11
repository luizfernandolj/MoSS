import quapy as qp

qp.environ["DATA_HOME"] = "./quapy_data"

# ---------------------------
# 1. UCI Multiclass datasets
# ---------------------------
uci_multiclass = [
    "dry-bean",
    "wine-quality",
    "academic-success",
    "digits",
    "letter",
]

for name in uci_multiclass:
    print(f"Downloading UCI multiclass: {name}")
    qp.datasets.fetch_UCIMulticlassDataset(name, verbose=True)

# ---------------------------
# 2. Reviews datasets
# ---------------------------
reviews = [
    "imdb",
    "kindle",
    "hp",
]

for name in reviews:
    print(f"Downloading reviews dataset: {name}")
    qp.datasets.fetch_reviews(name)

# ---------------------------
# 3. Twitter datasets
# ---------------------------
twitter = [
    "gasp",
    "hcr",
    "omd",
    # "semeval",
    "sst",
    "wa",
    "wb",
]

for name in twitter:
    print(f"Downloading twitter dataset: {name}")
    qp.datasets.fetch_twitter(name)

# ---------------------------
# LeQua datasets (QuaPy 0.2.0)
# ---------------------------
lequa = ["T1A", "T1B", "T2A", "T2B", "T3A", "T3B"]

for name in lequa:
    dataset_name = f"LeQua-{name}"
    print(f"Downloading LeQua dataset: {dataset_name}")
    qp.datasets.fetch(dataset_name)


print("\n✅ All datasets downloaded")
