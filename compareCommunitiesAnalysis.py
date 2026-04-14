import json
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
MODEL = "openai_gpt-oss-120b_free"
def load_community_results(base_dir):
    community_labels = {}

    for subreddit in os.listdir(base_dir):
        path = os.path.join(base_dir, subreddit, f"{MODEL}/results.json")
        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # map comment_id -> 0/1
        label_map = {
            item["comment_id"]: 1 if item["pred_label"] == "violation" else 0
            for item in data
        }

        community_labels[subreddit] = label_map

    return community_labels


def get_shared_comment_ids(community_labels):
    sets = [set(labels.keys()) for labels in community_labels.values()]

    if not sets:
        return []

    shared_ids = sets[0].intersection(*sets[1:])
    return sorted(shared_ids)

def build_vectors(community_labels, shared_ids):
    community_vectors = {}

    for subreddit, labels in community_labels.items():
        vector = [labels[cid] for cid in shared_ids]
        community_vectors[subreddit] = np.array(vector)

    return community_vectors




def compute_disagreement_matrix(community_vectors):
    subs = list(community_vectors.keys())
    n = len(subs)

    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            A = community_vectors[subs[i]]
            B = community_vectors[subs[j]]

            matrix[i][j] = np.mean(A != B)

    return pd.DataFrame(matrix, index=subs, columns=subs)



def cluster_communities(community_vectors):
    subs = list(community_vectors.keys())
    X = np.array([community_vectors[s] for s in subs])

    dist = pdist(X, metric='hamming')
    Z = linkage(dist, method='average')

    plt.figure()
    dendrogram(Z, labels=subs)
    plt.title("Community Clustering by Civility Enforcement")
    plt.show()

import os

BASE_DIR = "results/compareCommunities"
# Step 1: Load all community results
community_labels = load_community_results(BASE_DIR)
print(f"Loaded {len(community_labels)} communities")

# Step 2: Find shared comment IDs
shared_ids = get_shared_comment_ids(community_labels)
print(f"Shared comments: {len(shared_ids)}")

if len(shared_ids) == 0:
    print("No shared comments across communities. Exiting.")

# Step 3: Build vectors
community_vectors = build_vectors(community_labels, shared_ids)

# Step 4: Compute disagreement matrix
df = compute_disagreement_matrix(community_vectors)

print("\n=== Disagreement Matrix ===")
print(df)

# Save to CSV
df.to_csv("disagreement_matrix.csv")
print("\nSaved to disagreement_matrix.csv")

# Step 5: Cluster communities (optional visualization)
try:
    cluster_communities(community_vectors)
except Exception as e:
    print(f"Clustering failed: {e}")