import re
import wandb
import pandas as pd
from assignment_5.datasets import get_pandas_df
from snorkel.labeling import labeling_function, PandasLFApplier
from sklearn.metrics import accuracy_score

ABSTAIN = -1
ORGANIZATION = 0
MISC = 1

@labeling_function()
def lf_years(x):
    for token in x.tokens:
        if re.match(r"^(19|20)\d{2}[.,]?$", token):
            return MISC
    return ABSTAIN

@labeling_function()
def lf_org_sup(x):
    for token in x.tokens:
        if re.match(r"^(Inc|Inc\.|Corp|Corp\.|Ltd|Ltd\.)$", token, flags=re.IGNORECASE):
            return ORGANIZATION
    return ABSTAIN

def has_entity(row, entity_short_name, tag_names):
    entity_tags = {f"B-{entity_short_name}", f"I-{entity_short_name}"}
    for tag_id in row.ner_tags:
        if tag_names[tag_id] in entity_tags:
            return True
    return False

def labeling_helper(df, tag_names):
    df["y_org"] = df.apply(lambda row: ORGANIZATION if has_entity(row, "ORG", tag_names) else ABSTAIN, axis=1)
    df["y_misc"] = df.apply(lambda row: MISC if has_entity(row, "MISC", tag_names) else ABSTAIN, axis=1)
    
    # Apply Labeling Functions
    lfs = [lf_years, lf_org_sup]
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df)
    
    lf_eval_map = {
    "lf_years":    {"labels": L_train[:, 0], "ground_truth": df.y_misc},
    "lf_org_sup": {"labels": L_train[:, 1], "ground_truth": df.y_org},
    }
    
    return lf_eval_map, L_train
    

def main():
    # Train Dataset
    df, tag_names = get_pandas_df("train")
    lf_eval_map,_ = labeling_helper(df, tag_names)
    
    run = wandb.init(project="Q2-snorkel-labeling", job_type="Labeling", name="years_and_org_lfs")

    for lf_name, eval_data in lf_eval_map.items():
        lf_labels = eval_data["labels"]
        ground_truth = eval_data["ground_truth"]

        coverage = (lf_labels != ABSTAIN).sum() / len(lf_labels)

        active_indices = lf_labels != ABSTAIN
        if active_indices.sum() > 0:
            accuracy = accuracy_score(ground_truth[active_indices], lf_labels[active_indices])
        else:
            accuracy = 0.0

        print(lf_name.upper(), ":")
        print(f"Coverage: {coverage:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        run.log({
            f"{lf_name}_coverage": round(coverage, 4),
            f"{lf_name}_accuracy": round(accuracy, 4)
        })

    run.finish()

if __name__ == "__main__":
    main()