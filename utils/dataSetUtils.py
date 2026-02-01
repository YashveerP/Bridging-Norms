import random
from dotenv import load_dotenv
import pandas as pd

# dataset
data = pd.read_csv('data_training_selected_clusters_comments_and_rules.csv')
subredditToNorms = (
    data[data["label"] == "violation"]
    .groupby("subreddit_id")["target_reason"]
    .apply(lambda x: x.dropna().unique().tolist())
    .to_dict()
)

def getRandomNormForSubreddit(subreddit_id):
    norms = subredditToNorms.get(subreddit_id, [])
    if not norms:
        return None  # or "irrelevant_norm"
    return random.choice(norms)

# create a csv of the data set with each nv comment assigned to a community norm
def generatePreparedDataSet():
    
    eval_rows = []
    for _, row in data.iterrows():
        norm = row["target_reason"]
        if row["label"] == "non_violation":
            norm = getRandomNormForSubreddit(row["subreddit_id"])

        eval_rows.append({
            "body": row["body"],
            "norm": norm,
            "true_label": row["label"]
        })

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv("prepared_dataset.csv", index=False)