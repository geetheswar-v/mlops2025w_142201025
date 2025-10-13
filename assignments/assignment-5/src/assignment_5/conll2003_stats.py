from collections import Counter
import wandb

from assignment_5.datasets import get_conll2003_dataset

def summary(dataset, run=None):
    print("Question1: Checking dataset splits and their sizes")
    for split, data in dataset.items():
        num_samples = len(data)
        print(f"Question1: Number of samples (sentences) in {split} split: {num_samples}")
        if run:
            run.summary[f"{split}_num_samples_sentences"] = num_samples

    print("Question1: Calculating Entity distribution of each split")
    entity_to_bid = {'O': 0, 'PER': 1, 'ORG': 3, 'LOC': 5, 'MISC': 7}
    bid_to_entity = {v: k for k, v in entity_to_bid.items() if k != 'O'}

    for split, data in dataset.items():
        ner_tags = data["ner_tags"]
        
        main_entity_counts = Counter()
        for sentence_tags in ner_tags:
            for tag_id in sentence_tags:
                if tag_id in bid_to_entity:
                    entity_name = bid_to_entity[tag_id]
                    main_entity_counts[entity_name] += 1
        
        print(f"Question1: Entity counts for {split} split: {dict(main_entity_counts)}")
        
        if run:
            for entity, count in main_entity_counts.items():
                run.summary[f"{split}_{entity}_entity_count"] = count
            run.summary[f"{split}_total_entities"] = sum(main_entity_counts.values())

    print("Question1: Summary computation and logging completed successfully.")
    

def main():
    project_name = "Q1-weak-supervision-ner"
    run = wandb.init(project=project_name, job_type="dataset-analysis", name="Test 1")
    
    dataset = get_conll2003_dataset()
    summary(dataset, run)
    
    run.finish()

if __name__ == "__main__":
    main()