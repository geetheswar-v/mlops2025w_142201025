import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_conll2003_dataset():
    logger.info("Loading CONLL2003 dataset...")
    dataset = load_dataset("eriktks/conll2003", revision="convert/parquet")
    logger.info("CONLL2003 dataset loaded successfully.")
    dataset_splits = {}
    for split in dataset.keys():
        dataset_splits[split] = dataset[split]
    return dataset_splits