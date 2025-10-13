import re
import wandb
import pandas as pd
from assignment_5.datasets import get_pandas_df
from assignment_5.conll2003_labelling import ORGANIZATION, MISC, ABSTAIN, lf_years, lf_org_sup
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import MajorityLabelVoter
from sklearn.metrics import accuracy_score

def get_true_label(row, tag_names):
    has_org, has_misc = False, False
    for tag_id in row.ner_tags:
        tag = tag_names[tag_id]
        if "ORG" in tag:
            has_org = True
        elif "MISC" in tag:
            has_misc = True
    if has_org:
        return ORGANIZATION
    elif has_misc:
        return MISC
    else:
        return ABSTAIN
    
def labeling_helper(df, tag_names):
    df["y_true"] = df.apply(lambda row: get_true_label(row, tag_names), axis=1)
    
    # Apply labeling functions
    lfs = [lf_years, lf_org_sup]
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df)

    # Keep only rows where at least one LF voted
    mask = (L_train != ABSTAIN).any(axis=1)
    df_labeled = df[mask]
    L_labeled = L_train[mask]
    
    return mask, df_labeled, L_labeled

def main():
    df, tag_names = get_pandas_df("train")
    mask,df_labeled, L_labeled = labeling_helper(df, tag_names)

    run = wandb.init(project="Q3-snorkel-majority-voter", job_type="LabelAggregation", name="majority_label_voter_true")

    # Apply MajorityLabelVoter
    majority_model = MajorityLabelVoter(cardinality=2)
    y_pred = majority_model.predict(L=L_labeled)

    # Accuracy on non-abstained data
    true_labels = df_labeled["y_true"]
    accuracy = (y_pred == true_labels).mean()

    print("MAJORITY VOTER (MISC):")
    print(f"Accuracy: {accuracy:.4f}")

    run.log({
        "majority_voter_accuracy": round(accuracy, 4),
        "coverage": round(mask.mean(), 4)
    })
    run.finish()
    
if __name__ == "__main__":
    main()
