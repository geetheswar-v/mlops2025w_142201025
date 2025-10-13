import logging
from collections import Counter
import wandb

from assignment_5.datasets import get_conll2003_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Question1")

NER_TAG_MAP = {
    0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG',
    5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'
}

def summary(dataset, run=None):
    logger.info("Checking dataset splits and their sizes")
    for split, data in dataset.items():
        num_samples = len(data)
        logger.info(f"Number of samples (sentences) in {split} split: {num_samples}")
        if run:
            run.summary[f"{split}_num_samples_sentences"] = num_samples

    logger.info("Calculating Entity distribution of each split")
    
    entity_to_main_type = {}
    for tag_id, tag_name in NER_TAG_MAP.items():
        if tag_name == 'O':
            main_type = 'O'
        else:
            main_type = tag_name.split('-')[-1]
        entity_to_main_type[tag_id] = main_type

    for split, data in dataset.items():
        ner_tags = data["ner_tags"]
        all_tags = [tag for seq in ner_tags for tag in seq]
        total_tokens = len(all_tags)
        
        tag_counts = Counter(all_tags)
        
        main_entity_counts = Counter()
        for tag_id, count in tag_counts.items():
            main_type = entity_to_main_type[tag_id]
            main_entity_counts[main_type] += count
        
        logger.info(f"Entity token distribution in {split} split: {dict(main_entity_counts)}")
        logger.info(f"Total tokens in {split} split: {total_tokens}")
        
        if run:
            run.summary[f"{split}_total_tokens"] = total_tokens
            for entity, count in main_entity_counts.items():
                run.summary[f"{split}_entity_dist_{entity}_count"] = count
                run.summary[f"{split}_entity_dist_{entity}_percent"] = (count / total_tokens) * 100

    logger.info("Summary computation and logging completed successfully.")
    

def main():
    dataset = get_conll2003_dataset()
    summary(dataset)

if __name__ == "__main__":
    main()
