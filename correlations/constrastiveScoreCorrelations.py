import pandas as pd
import os
import matplotlib.pyplot as plt

# --- Load data ---
scores_df = pd.read_csv('datasets/violating_clauses_contrastive_scores.csv')
scores_df = pd.read_csv('datasets/violating_clauses_contrastive_scores.csv')
scores_df = pd.read_csv('datasets/violating_clauses_contrastive_scores.csv')
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
fig, axs = plt.subplots(3, 2, figsize=(12, 12))

axs = axs.flatten()  # makes indexing way easier

plot_idx = 0

for col in df.columns:

    if col in ["subreddit", "accuracy", "n_items"]:

        continue

    if not pd.api.types.is_numeric_dtype(df[col]):

        continue

    if plot_idx >= len(axs):  # stop if we run out of subplots

        break

    x = df[col]

    y = df["accuracy"]

    corr = x.corr(y)

    ax = axs[plot_idx]

    # --- Plot ---

    ax.scatter(x, y)

    ax.set_ylim(0, 1)

    ax.set_xlim(0, 0.4)

    ax.set_xlabel(col)

    ax.set_ylabel("Accuracy")

    ax.set_title(f"{col} vs Accuracy\nPearson r = {corr:.3f}")

    print(f"{col}: Pearson r = {corr:.4f}")

    plot_idx += 1

plt.tight_layout()

plt.show()