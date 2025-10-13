from datasets import load_dataset

def get_conll2003_dataset():
    print("Loading CONLL2003 dataset...")
    dataset = load_dataset("eriktks/conll2003", revision="convert/parquet")
    print("CONLL2003 dataset loaded successfully.")
    dataset_splits = {}
    for split in dataset.keys():
        dataset_splits[split] = dataset[split]
    return dataset_splits