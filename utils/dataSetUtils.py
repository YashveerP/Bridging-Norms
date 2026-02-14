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
    i = 0
    for _, row in data.iterrows():
        norm = row["target_reason"]
        if row["label"] == "non_violation":
            norm = getRandomNormForSubreddit(row["subreddit_id"])

        eval_rows.append({
            "comment_id": i,
            "body": row["body"],
            "norm": norm,
            "true_label": row["label"]
        })
        i += 1

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv("prepared_dataset.csv", index=False)

# create a balanced trainTestSplit and save to files
def makeNewTrainTestSplit(numTests):
    df = pd.read_csv('datasets/prepared_dataset.csv')

    violations = df[df["label"] == "violation"]
    nonViolations = df[df["label"] == "non_violation"]
    testDF = pd.concat(violations.sample(50), nonViolations.sample(50))
    trainDF = df.drop(testDF.index)

    testDF.to_csv("datasets/tests.csv")
    trainDF.to_csv("datasets/train.csv")