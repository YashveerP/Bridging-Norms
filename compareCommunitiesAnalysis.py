import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.join("results", "compareCommunities") # change if needed

# collect all communities
communities = sorted(os.listdir(BASE_DIR))
communities.remove("t5_2qnkr")
# initialize matrix
n = len(communities)
matrix = np.zeros((n, n))

# load data
for i, prompt_comm in enumerate(communities):
    for j, eval_comm in enumerate(communities):
        metricsPath = os.path.join("openai_gpt-oss-120b_free", "metrics.json")
        path = os.path.join(BASE_DIR, prompt_comm, eval_comm, metricsPath)
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                
                # change this key depending on your metrics.json
                acc = data.get("accuracy", 0)
                matrix[j][i] = acc   # rows = eval, cols = prompt
        else:
            matrix[j][i] = np.nan

evalMeans = {}
for i, eval_comm in enumerate(communities):
    evalMeans[eval_comm] = 0
    count = 0
    for j, prompt_comm in enumerate(communities):
        metricsPath = os.path.join("openai_gpt-oss-120b_free", "metrics.json")
        path = os.path.join(BASE_DIR, prompt_comm, eval_comm, metricsPath)
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                
                # change this key depending on your metrics.json
                acc = data.get("accuracy", 0)
                evalMeans[eval_comm] += acc
                count += 1  
    evalMeans[eval_comm] = evalMeans[eval_comm] / count

df = pd.read_csv('datasets/subreddits-descriptions.csv')

pairs = []
labels = []
for community in communities:
    row = df[df["name"] == community].iloc[0]
    
    label = f"{row['display_name_prefixed']} ({community})"
    value = evalMeans[community]
    
    pairs.append((label, value))
    labels.append(label)

# Sort by value (descending is usually best)
pairs.sort(key=lambda x: x[1], reverse=True)

labels_sorted, values_sorted = zip(*pairs)

plt.barh(labels_sorted[::-1], values_sorted[::-1])
plt.title('Evaluated Community Accuracy')
plt.xlabel('Mean Accuracy')
plt.ylabel('Evaluated Community')

# normalize by diagonal (self-accuracy)
normalized = matrix.copy()

print("min:", matrix.min())

print("max:", matrix.max())
for i in range(n):
    if matrix[i][i] > 0:
        normalized[i, :] /= matrix[i][i]
print("min:", normalized.min())
print("max:", normalized.max())

# plot
plt.figure()
plt.imshow(normalized, cmap="Blues", vmin=.5, vmax=1.5)

plt.xticks(range(n), labels, rotation=90)
plt.yticks(range(n), labels)

plt.xlabel("Prompt Community")
plt.ylabel("Evaluation Community")
plt.title("Normalized Cross-Community Accuracy")

plt.colorbar()
plt.tight_layout()

plt.figure()
plt.imshow(matrix, cmap="Blues", vmin=0.5, vmax=1)

plt.xticks(range(n), labels, rotation=90)
plt.yticks(range(n), labels)

plt.xlabel("Prompt Community")
plt.ylabel("Evaluation Community")
plt.title("Cross-Community Accuracy")

plt.colorbar()
plt.tight_layout()
plt.show()