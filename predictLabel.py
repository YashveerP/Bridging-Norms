import pandas as pd
from utils.predictLabelUtils import predictViolation
from utils.localPredictLabelUtils import localPredictViolation
import json
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
MODEL = "meta-llama/llama-3.3-70b-instruct:free"

df = pd.read_csv('datasets/prepared_dataset.csv')

train_df, test_df = train_test_split(
    df,
    test_size=NUM_TESTS,         # 25% test(defualt)
    stratify=df["true_label"] # stratify on true label so both dfs have even violations and non violations
)

# go through each violated comment and store json
results = []

# store true and prediceted labels
y_true = []
y_pred = []

# for each sample
for i in tqdm(range(NUM_TESTS)):
    row = test_df.iloc[i]
    # output = predictViolation(row["body"], row["norm"], MODEL)
    output = localPredictViolation(row["body"], row["norm"])
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
    with open("results/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

#Performance metrics
print("\n=== Performance Metrics ===")

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


print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1:        {f1:.3f}")
print("\nConfusion Matrix:")
print(cm)


metrics = {
    "model": MODEL,
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "confusion_matrix": cm.tolist()
}

with open("results/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

