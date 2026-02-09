import random
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
# dataset
data = pd.read_csv('datasets/data_training_selected_clusters_comments_and_rules.csv')
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

def makeNewTrainTestSplit(numTests):
    df = pd.read_csv('datasets/prepared_dataset.csv')

    train_df, test_df = train_test_split(
        df,
        test_size=numTests,         
        stratify=df["true_label"] # stratify on true label so both dfs have even violations and non violations
    )
    test_df.to_csv("datasets/tests.csv")
    train_df.to_csv("datasets/train.csv")