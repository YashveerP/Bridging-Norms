import pandas as pd
import os
import matplotlib.pyplot as plt

# --- Load data ---
scores_df = pd.read_csv('datasets/contrastive.csv')
accuracy_df = pd.read_csv('datasets/accuracy.csv')
meta_df = pd.read_csv('datasets/subreddits-descriptions.csv')

BASE_DIR = os.path.join("results", "compareCommunities")

# --- Normalize subreddit names ---
accuracy_df["subreddit_clean"] = accuracy_df["subreddit"].str.replace(
    r"\s*\(.*\)", "", regex=True
)

meta_df["subreddit_clean"] = meta_df["display_name_prefixed"]

# --- Build lookup maps ---
accuracy_map = dict(zip(accuracy_df["subreddit_clean"], accuracy_df["accuracy"]))

# --- Merge all data into one table ---
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

    # add all contrastive columns
    for k, v in score_row.items():
        if k != "subreddit":
            combined[k] = v

    rows.append(combined)

# --- Final dataframe ---
df = pd.DataFrame(rows)

if df.empty:
    raise ValueError("No data after merging — check normalization")

# --- Loop through all numeric contrastive columns ---
for col in df.columns:
    if col in ["subreddit", "accuracy"]:
        continue

    if not pd.api.types.is_numeric_dtype(df[col]):
        continue

    x = df[col]
    y = df["accuracy"]

    corr = x.corr(y)

    # --- Plot ---
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y)

    # label a few outliers (optional)
    # for i in range(len(df)):
    #     if y.iloc[i] < y.mean() - 0.1:
    #         plt.text(x.iloc[i], y.iloc[i], df["subreddit"].iloc[i], fontsize=7)

    plt.xlabel(col)
    plt.ylabel("Accuracy")
    plt.title(f"{col} vs Accuracy\nPearson r = {corr:.3f}")

    plt.tight_layout()
    plt.show()

    print(f"{col}: Pearson r = {corr:.4f}")