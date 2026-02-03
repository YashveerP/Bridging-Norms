import pandas as pd
from utils.predictLabelUtils import predictViolation
from utils.localPredictLabelUtils import localPredictViolation, COT
import json, time, re, os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

NUM_TESTS = 50
# MODEL = "meta-llama/llama-3.3-70b-instruct:free"
MODEL="qwen2.5:7b-instruct"

safe_model = re.sub(r'[<>:"/\\|?*]', '_', MODEL)
path = f"results/{safe_model}"

# df = pd.read_csv('datasets/prepared_dataset.csv')

# train_df, test_df = train_test_split(
#     df,
#     test_size=NUM_TESTS,         # constant number of tests
#     stratify=df["true_label"] # stratify on true label so both dfs have even violations and non violations
# )
# test_df.to_csv("datasets/tests.csv")

df = pd.read_csv('datasets/tests.csv')

# go through each violated comment and store json
results = []

# store true and prediceted labels
y_true = []
y_pred = []

# for each sample
for i in tqdm(range(NUM_TESTS)):
    row = df.iloc[i]

    # sleep to avoid too many requests error
    # time.sleep(3)
    # output = predictViolation(row["body"], row["norm"], MODEL)
    output = localPredictViolation(row["body"], row["norm"])
    # output = COT(row["body"], row["norm"])
    parsed = json.loads(output)
    results.append({
        "body": row["body"],
        "norm": row["norm"],
        "true_label": row["true_label"],
        "pred_label": parsed["label"],
        "evidence": parsed["evidence"],
    })
    y_true.append(row["true_label"])
    y_pred.append(parsed["label"])


    # write to results.json at end of iteration in case later breaks
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

acc = accuracy_score(y_true, y_pred)

prec = precision_score(
    y_true, y_pred,
    pos_label="violation",
    zero_division=0
)

rec = recall_score(
    y_true, y_pred,
    pos_label="violation",
    zero_division=0
)

f1 = f1_score(
    y_true, y_pred,
    pos_label="violation",
    zero_division=0
)

cm = confusion_matrix(y_true, y_pred, labels=["non_violation", "violation"])

metrics = {
    "model": MODEL,
    "num_tests": NUM_TESTS,
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "confusion_matrix": cm.tolist()
}

with open(f"{path}/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

