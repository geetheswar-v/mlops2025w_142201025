import logging
from assignment_5.datasets import get_conll2003_dataset
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Question1")

def summary(dataset, run=None):
    summary_info = {}
    logger.info("Checking dataset splits and their sizes")
    for split, data in dataset.items():
        num_samples = len(data)
        logger.info(f"Number of samples in {split} split: {num_samples}")
        summary_info[f"{split}_num_samples"] = num_samples

    if run:
        run.summary.update(summary_info)
        logger.info("Dataset summary updated in wandb run.")

    logger.info("Calculating Entity distribution of each split")
    logger.info("Calculating Entity distribution of each split")
    ner_feature = dataset["train"].features["ner_tags"].feature
    id2label = {i: label for i, label in enumerate(ner_feature.names)}

    for split, data in dataset.items():
        ner_tags = data["ner_tags"]
        all_tags = [tag for seq in ner_tags for tag in seq]
        tag_counts = Counter(all_tags)
        
        entity_distribution = {id2label[i]: count for i, count in tag_counts.items()}
        logger.info(f"Entity distribution in {split} split: {entity_distribution}")

    logger.info("Summary computation completed successfully.")
        
        
if __name__ == "__main__":
    dataset = get_conll2003_dataset()
    summary(dataset)
