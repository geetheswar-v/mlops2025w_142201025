import wandb
import datasets
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

run = wandb.init(project="Q1-weak-supervision-ner", job_type="data_analysis", name="conll2003_dataset_analysis_3")

def load_dataset():
    logger.info("Loading the CONLL2003 dataset...")
    dataset = datasets.load_dataset("eriktks/conll2003", revision="convert/parquet")
    logger.info("Dataset loaded successfully.")
    return dataset

def summary(dataset):
    summary_info = {}
    
    logger.info("Checking dataset splits and their sizes")
    for split in dataset.keys():
        num_samples = len(dataset[split])
        summary_info[f"{split}_num_samples"] = num_samples
        logger.info(f"Number of samples in {split} split: {num_samples}")
        
    wandb.summary.update(summary_info)
    logger.info("Dataset summary updated in wandb run.")
    
    logger.info("Calculating entity distribution")
    ner_feature = dataset["train"].features["ner_tags"].feature
    id2label = {i: label for i, label in enumerate(ner_feature.names)}

    entity_counter = Counter()
    
    for split in dataset.keys():
        for example in dataset[split]:
            tag_ids = example["ner_tags"]
            for tag_id in tag_ids:
                label = id2label[tag_id]
                if label != 'O':
                    entity_type = label.split('-')[1]
                    entity_counter[entity_type] += 1

    entity_distribution = dict(entity_counter)
    for entity, count in entity_distribution.items():
        logger.info(f"Entity: {entity}, Count: {count}")
    
    wandb_entity_stats = {f"entity_{k}_count": v for k, v in entity_distribution.items()}
    wandb.summary.update(wandb_entity_stats)
    logger.info("Entity distribution logged to W&B summary.")

def main():
    dataset = load_dataset()
    summary(dataset)
    run.finish()
    
if __name__ == "__main__":
    main()