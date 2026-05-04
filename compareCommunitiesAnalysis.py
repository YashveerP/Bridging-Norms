import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.join("results", "compareCommunities") # change if needed

LLAMA3 = "meta-llama/llama-3.3-70b-instruct:free"
GPT_OSS = "openai/gpt-oss-120b:free"
GPT_4O_MINI= "openai/gpt-4o-mini"
MODEL = GPT_OSS
safe_model = re.sub(r'[<>:"/\\|?*]', '_', MODEL)

# collect all communities
communities = sorted(os.listdir(BASE_DIR))
communities.remove("t5_2qnkr")
# initialize matrix
n = len(communities)
matrix = np.zeros((n, n))

# load data
for i, prompt_comm in enumerate(communities):
    for j, eval_comm in enumerate(communities):
        metricsPath = os.path.join(safe_model, "metrics.json")
        path = os.path.join(BASE_DIR, prompt_comm, eval_comm, metricsPath)
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                
                # change this key depending on your metrics.json
                acc = data.get("accuracy", 0)
                matrix[j][i] = acc   # rows = eval, cols = prompt
        else:
            matrix[j][i] = np.nan

# --- compute raw means (for bar chart + raw sorting) ---
raw_means = {comm: np.mean(matrix[i, :]) for i, comm in enumerate(communities)}

# --- normalize matrix ---
normalized = matrix.copy()
for i in range(len(communities)):
    if matrix[i][i] > 0:
        normalized[i, :] /= matrix[i][i]

# --- compute normalized means ---
norm_means = {comm: np.mean(normalized[i, :]) for i, comm in enumerate(communities)}

# --- load labels + comment counts ---
df = pd.read_csv('datasets/subreddits-descriptions.csv')

# assuming your comments dataset has counts
counts_df = pd.read_csv('datasets/data_training_selected_clusters_comments_and_rules.csv')
comment_counts = counts_df["subreddit_id"].value_counts().to_dict()

raw_pairs, norm_pairs, count_pairs = [], [], []

for comm in communities:
    row = df[df["name"] == comm].iloc[0]
    count = comment_counts.get(comm, 0)
    label = f"{row['display_name_prefixed']} ({comm}) | n={count}"
    
    raw_pairs.append((comm, label, raw_means[comm]))
    norm_pairs.append((comm, label, norm_means[comm]))
    count_pairs.append((comm, label, count))

# --- sortings ---
raw_pairs.sort(key=lambda x: x[2], reverse=True)
norm_pairs.sort(key=lambda x: x[2], reverse=True)
count_pairs.sort(key=lambda x: x[2], reverse=True)

# unpack
raw_comms  = [p[0] for p in raw_pairs]
raw_labels = [p[1] for p in raw_pairs]
raw_vals   = [p[2] for p in raw_pairs]

norm_comms  = [p[0] for p in norm_pairs]
norm_labels = [p[1] for p in norm_pairs]

count_comms  = [p[0] for p in count_pairs]
count_labels = [p[1] for p in count_pairs]

# --- index maps ---
idx_map = {comm: i for i, comm in enumerate(communities)}

raw_idx   = [idx_map[c] for c in raw_comms]
norm_idx  = [idx_map[c] for c in norm_comms]
count_idx = [idx_map[c] for c in count_comms]

# --- reorder matrices ---
matrix_raw_sorted   = matrix[np.ix_(raw_idx, raw_idx)]
matrix_count_sorted = matrix[np.ix_(count_idx, count_idx)]

normalized_norm_sorted  = normalized[np.ix_(norm_idx, norm_idx)]
normalized_count_sorted = normalized[np.ix_(count_idx, count_idx)]

n = len(communities)

# =========================
# BAR CHART (raw mean)
# =========================
plt.figure()
plt.barh(raw_labels[::-1], raw_vals[::-1])
plt.xlim(0.5, 1)
plt.title('Evaluated Community Accuracy')
plt.xlabel('Mean Accuracy')
plt.ylabel('Evaluated Community')
plt.tight_layout()

# =========================
# RAW HEATMAP (sorted by raw mean)
# =========================
plt.figure()
plt.imshow(matrix_raw_sorted, cmap="Blues", vmin=0.5, vmax=1)
plt.xticks(range(n), raw_labels, rotation=90)
plt.yticks(range(n), raw_labels)
plt.title("Cross-Community Accuracy (Sorted by Mean Accuracy)")
plt.xlabel("Prompt Community")
plt.ylabel("Evaluation Community")
plt.colorbar()
plt.tight_layout()

# =========================
# NORMALIZED HEATMAP (sorted by normalized mean)
# =========================
plt.figure()
plt.imshow(normalized_norm_sorted, cmap="Blues", vmin=.5, vmax=1.5)
plt.xticks(range(n), norm_labels, rotation=90)
plt.yticks(range(n), norm_labels)
plt.title("Normalized Cross-Community Accuracy (Sorted by Normalized Mean)")
plt.xlabel("Prompt Community")
plt.ylabel("Evaluation Community")
plt.colorbar()
plt.tight_layout()

# =========================
# RAW HEATMAP (sorted by comment count)
# =========================
plt.figure()
plt.imshow(matrix_count_sorted, cmap="Blues", vmin=0.5, vmax=1)
plt.xticks(range(n), count_labels, rotation=90)
plt.yticks(range(n), count_labels)
plt.title("Cross-Community Accuracy (Sorted by Comment Count)")
plt.xlabel("Prompt Community")
plt.ylabel("Evaluation Community")
plt.colorbar()
plt.tight_layout()

# =========================
# NORMALIZED HEATMAP (sorted by comment count)
# =========================
plt.figure()
plt.imshow(normalized_count_sorted, cmap="Blues", vmin=.5, vmax=1.5)
plt.xticks(range(n), count_labels, rotation=90)
plt.yticks(range(n), count_labels)
plt.title("Normalized Cross-Community Accuracy (Sorted by Comment Count)")
plt.xlabel("Prompt Community")
plt.ylabel("Evaluation Community")
plt.colorbar()
plt.tight_layout()

plt.show()