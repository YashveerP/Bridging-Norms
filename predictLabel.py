import pandas as pd
from helpers import predictViolation
import json
from tqdm import tqdm


data = pd.read_csv('data_training_selected_clusters_comments_and_rules.csv')
violated = data[data["label"] == "violation"]

# go through each violated comment and store json
results = []
row = violated.iloc[0]
# for _, row in tqdm(data.iterrows(), total=len(data)):
output = predictViolation(row["body"], row["target_reason"])
# try:
    # parsed = json.loads(output)
# except json.JSONDecodeError:
    # continue

parsed = json.loads(output)
results.append({
    "body": row["body"],
    "target_reason": row["target_reason"],
    "true_label": row["label"],
    "pred_label": parsed["label"],
    "confidence": parsed["confidence"],
})

# write to results.json
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)