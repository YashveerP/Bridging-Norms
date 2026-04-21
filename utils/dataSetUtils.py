import ast
import csv
import json
import random
import pandas as pd
# dataset
SEED = 21
random.seed(SEED)
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
    testDF = pd.concat([violations.sample(int(numTests/2), random_state=SEED), nonViolations.sample(int(numTests/2), random_state=SEED)])
    # shuffle around all the samples
    testDF = testDF.sample(frac=1, random_state=SEED).reset_index(drop=True)
    # take out test samples and put them as training samples
    trainDF = df.drop(testDF.index)
    trainDF = trainDF.sample(frac=1, random_state=SEED).reset_index(drop=True)

    testDF.to_csv("datasets/tests2.csv", index=False)
    trainDF.to_csv("datasets/train2.csv", index=False)


def getCommunityRules(subreddit_id):
        df = pd.read_csv(
            'datasets/removed_comments_rules.csv',
            skipinitialspace=True
        )

        # Fix column names
        df.columns = df.columns.str.strip()

        # # Fix values
        df["subreddit_id"] = df["subreddit_id"].str.strip()

        rules = df[df["subreddit_id"] == subreddit_id].iloc[0].rules
        rules = ast.literal_eval(rules)
    
        simplified = []
        for r in rules:
            name = r.get("short_name", "").strip()
            reason = r.get("violation_reason", "").strip()
            
            simplified.append(f"{name}: {reason}")
        
        return simplified

def getCommunity(subredditID):
    df = pd.read_csv('datasets/subreddits-descriptions.csv')
    return df[df["name"] == subredditID].iloc[0]

def printTopCommuntiies():
    df = pd.read_csv('datasets/data_training_selected_clusters_comments_and_rules.csv')

    print(df["subreddit_id"].value_counts().head(5))

def getCommunityComments(subredditID):
    df = pd.read_csv('datasets/data_training_selected_clusters_comments_and_rules.csv')
    communityComments = df[df["subreddit_id"] == subredditID]
    # communityComments = communityComments[communityComments.columns.difference(['subreddit_id', 'assigned_rule_cluster'])]
    return communityComments


# create a csv of the data set with each nv comment assigned to a community norm
def generatePreparedDataSet(dataFrame):
    eval_rows = []
    # Assign all comments comment id's based off of index44
    i = 0
    for _, row in dataFrame.iterrows():
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
    return eval_df
    # eval_df.to_csv(f"datasets/{fileName}.csv", index=False)

def getSubreddits():
    df = pd.read_csv('datasets/data_training_selected_clusters_comments_and_rules.csv')

    reddits = df["subreddit_id"].value_counts()
    # filter subreddits with > 150 comments
    return reddits[reddits > 150].index.tolist()
