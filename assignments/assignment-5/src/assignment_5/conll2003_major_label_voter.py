import re
import wandb
import pandas as pd
from assignment_5.datasets import get_pandas_df
from assignment_5.conll2003_labelling import ABSTAIN, labeling_helper
from snorkel.labeling.model import MajorityLabelVoter
from sklearn.metrics import accuracy_score

def main():
    df, tag_names = get_pandas_df("train")
    _, L_train = labeling_helper(df, tag_names)

    run = wandb.init(project="Q3-snorkel-majority-voter", job_type="LabelAggregation", name="majority_label_voter")

    # Majority Label Voter for MISC
    majority_model_misc = MajorityLabelVoter(cardinality=2)
    df["majvoter_misc"] = majority_model_misc.predict(L=L_train)

    misc_active = df.y_misc != ABSTAIN
    misc_accuracy = accuracy_score(df.y_misc[misc_active], df.majvoter_misc[misc_active])

    print("MAJORITY VOTER (MISC):")
    print(f"Accuracy: {misc_accuracy:.4f}")

    # Majority Label Voter for ORG
    majority_model_org = MajorityLabelVoter(cardinality=2)
    df["majvoter_org"] = majority_model_org.predict(L=L_train)

    org_active = df.y_org != ABSTAIN
    org_accuracy = accuracy_score(df.y_org[org_active], df.majvoter_org[org_active])

    print("MAJORITY VOTER (ORG):")
    print(f"Accuracy: {org_accuracy:.4f}")

    run.log({
        "misc_majority_accuracy": round(misc_accuracy, 4),
        "org_majority_accuracy": round(org_accuracy, 4),
    })
    run.finish()
    
if __name__ == "__main__":
    main()
