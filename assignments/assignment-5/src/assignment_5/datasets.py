from datasets import load_dataset

def get_conll2003_dataset():
    print("Loading CONLL2003 dataset...")
    dataset = load_dataset("eriktks/conll2003", revision="convert/parquet")
    print("CONLL2003 dataset loaded successfully.")
    dataset_splits = {}
    for split in dataset.keys():
        dataset_splits[split] = dataset[split]
    return dataset_splits

def get_pandas_df(split="train"):
    dataset_splits = get_conll2003_dataset()
    data = dataset_splits[split]
    tag_names = data.features['ner_tags'].feature.names
    df = data.to_pandas()
    return df, tag_names