import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# --- Load data ---
# scores_df = pd.read_csv('datasets/all_comments_contrastive_scores.csv')
# scores_df = pd.read_csv('datasets/all_clauses_contrastive_scores.csv')
scores_df = pd.read_csv('datasets/violating_clauses_contrastive_scores.csv')
accuracy_df = pd.read_csv('datasets/accuracy.csv')
meta_df = pd.read_csv('datasets/subreddits-descriptions.csv')

BASE_DIR = os.path.join("results", "compareCommunities")

# --- Normalize subreddit names ---
accuracy_df["subreddit_clean"] = accuracy_df["subreddit"].str.replace(
    r"\s*\(.*\)", "", regex=True
)

meta_df["subreddit_clean"] = meta_df["display_name_prefixed"]

# --- Build accuracy map ---
accuracy_map = dict(zip(
    accuracy_df["subreddit_clean"],
    accuracy_df["accuracy"]
))

# --- Merge into single table ---
rows = []

communities = sorted(os.listdir(BASE_DIR))

for community in communities:
    row = meta_df[meta_df["name"] == community]
    if row.empty:
        continue

    row = row.iloc[0]
    subreddit = row["display_name_prefixed"]

    acc = accuracy_map.get(subreddit)
    score_row = scores_df[scores_df["subreddit"] == subreddit]

    if acc is None or score_row.empty:
        continue

    score_row = score_row.iloc[0].to_dict()

    combined = {
        "subreddit": subreddit,
        "accuracy": acc
    }

    for k, v in score_row.items():
        if k != "subreddit":
            combined[k] = v

    rows.append(combined)

df = pd.DataFrame(rows)

if df.empty:
    raise ValueError("No data after merging")

# need to drop nonnumeric collums for coorelation
metric_df = df.drop(columns=["subreddit"])

# normalize ( how ,amy std from mean are the metrics(z-score))
metric_df = (metric_df - metric_df.mean()) / metric_df.std()

subreddits = df["subreddit"].tolist()

pairs = []

for i, j in combinations(range(len(df)), 2):
    vec_i = metric_df.iloc[i].values
    vec_j = metric_df.iloc[j].values

    corr = np.corrcoef(vec_i, vec_j)[0, 1]

    pairs.append((
        subreddits[i],
        subreddits[j],
        corr
    ))

pairs_df = pd.DataFrame(pairs, columns=["subreddit_A", "subreddit_B", "correlation"])

# communities correlation pairs
top_similar = pairs_df.sort_values("correlation", ascending=False).head(10)
top_different = pairs_df.sort_values("correlation", ascending=True).head(10)

print("Most Similiar community pairs:")
print(top_similar)


print("\nLeast Similiar community pairs:")
print(top_different)

# avg correlations
avg_corr = {}

for s in subreddits:
    vals = pairs_df[
        (pairs_df["subreddit_A"] == s) |
        (pairs_df["subreddit_B"] == s)
    ]["correlation"]

    avg_corr[s] = vals.mean()

avg_corr = pd.Series(avg_corr).sort_values(ascending=False)

print("\nMost centraly similiar communities:")
print(avg_corr.head(10))

print("\nMost outlier communities:")
print(avg_corr.tail(10))