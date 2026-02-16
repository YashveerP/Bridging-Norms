import random
import pandas as pd
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
    # Assign all comments comment id's based off of index
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
    eval_df.to_csv("datasets/prepared_dataset.csv", index=False)

# create a balanced trainTestSplit and save to files
def makeNewTrainTestSplit(numTests):
    df = pd.read_csv('datasets/prepared_dataset.csv')

    violations = df[df["true_label"] == "violation"]
    nonViolations = df[df["true_label"] == "non_violation"]
    # concat 50 random violation and nonviolation samples
    testDF = pd.concat([violations.sample(int(numTests/2)), nonViolations.sample(int(numTests/2))])
    # shuffle around all the samples
    testDF = testDF.sample(frac=1).reset_index(drop=True)
    # take out test samples and put them as training samples
    trainDF = df.drop(testDF.index)
    trainDF = trainDF.sample(frac=1).reset_index(drop=True)

    testDF.to_csv("datasets/tests.csv", index=False)
    trainDF.to_csv("datasets/train.csv", index=False)