import os
import json
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = "results\compareCommunities"  # change if needed

# collect all communities
communities = sorted(os.listdir(BASE_DIR))

# initialize matrix
n = len(communities)
matrix = np.zeros((n, n))

# load data
for i, prompt_comm in enumerate(communities):
    for j, eval_comm in enumerate(communities):
        path = os.path.join(BASE_DIR, prompt_comm, eval_comm, "openai_gpt-oss-120b_free\metrics.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                
                # change this key depending on your metrics.json
                acc = data.get("accuracy", 0)
                matrix[j][i] = acc   # rows = eval, cols = prompt
        else:
            matrix[j][i] = np.nan

# normalize by diagonal (self-accuracy)
normalized = matrix.copy()
for i in range(n):
    if matrix[i][i] > 0:
        normalized[i, :] /= matrix[i][i]

# print(matrix)
# plot
plt.figure()
plt.imshow(normalized)

plt.xticks(range(n), communities, rotation=45)
plt.yticks(range(n), communities)

plt.xlabel("Prompt Community")
plt.ylabel("Evaluation Community")
plt.title("Normalized Cross-Community Accuracy")

plt.colorbar()
plt.tight_layout()

plt.figure()
plt.imshow(matrix)

plt.xticks(range(n), communities, rotation=45)
plt.yticks(range(n), communities)

plt.xlabel("Prompt Community")
plt.ylabel("Evaluation Community")
plt.title("Cross-Community Accuracy")

plt.colorbar()
plt.tight_layout()
plt.show()