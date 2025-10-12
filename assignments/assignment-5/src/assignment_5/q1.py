import wandb
import datasets
import logging

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

run = wandb.init(project="Q1-weak-supervision-ner", job_type="load_dataset", name="conll2003_dataset")

def load_dataset():
    logger.info("Loading the CONLL2003 dataset...")
    dataset = datasets.load_dataset("eriktks/conll2003", revision="convert/parquet")
    logger.info("Dataset loaded successfully.")
    return dataset

def main():
    dataset = load_dataset()
    for split in dataset.keys():
        num_samples = len(dataset[split])
        logger.info(f"Number of samples in {split} split: {num_samples}")
    run.finish()
    
if __name__ == "__main__":
    main()