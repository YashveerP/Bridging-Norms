import pandas as pd
from utils.predictLabelUtils import predictViolation, COT
import json, time, re, os
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

NUM_TESTS = 100
BATCH_SIZE = 20

# MODEL = "meta-llama/llama-3.3-70b-instruct:free"
PROMPT_TYPE = "ZeroShot"      # ← choose: "ZeroShot", "OneShot", "FewShot"
USE_COT = False
PROMPT_NAME = PROMPT_TYPE + ("-COT" if USE_COT else "")
RUNNER = "local"         # ← choose: "local" or "openrouter"

LOCAL_MODEL="qwen2.5:7b-instruct"
OR_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
MODEL = LOCAL_MODEL

safe_model = re.sub(r'[<>:"/\\|?*]', '_', MODEL)
path = f"results/{safe_model}/{PROMPT_NAME}"

df = pd.read_csv('datasets/tests.csv')

# go through each violated comment and store json
results = []

# store true and prediceted labels
y_true = []
y_pred = []

for start in tqdm(range(0, NUM_TESTS, BATCH_SIZE)):
    batch_df = df.iloc[start:start + BATCH_SIZE]

    # build input in the format your prompt expects
    batch_input = []
    for idx, row in batch_df.iterrows():
        batch_input.append({
            "comment_id": int(idx),   # or just use enumerate if you prefer
            "norm": row["norm"],
            "comment": row["body"]
        })

    # SINGLE model call for the whole batch
    output = predictViolation(batch_input, RUNNER, MODEL, PROMPT_TYPE, USE_COT)

    parsed_list = json.loads(output)  # this should be a LIST

    for item in parsed_list:
        comment_id = item["comment_id"]
        comment = df.loc[comment_id, "body"]

        if item["evidence"] and item["evidence"].strip() not in comment.replace("\r", ""):
            item["evidence"] = ""

    # match predictions back to rows
    for item in parsed_list:
        comment_id = item["comment_id"]
        row = df.loc[comment_id]

        results.append({
            "body": row["body"],
            "norm": row["norm"],
            "true_label": row["true_label"],
            "pred_label": item["label"],
            "evidence": item["evidence"],
        })
        y_true.append(row["true_label"])
        y_pred.append(item["label"])

    # save progress
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# fraction of correct predicitons
acc = accuracy_score(y_true, y_pred)

# out of all positive predictions, what fraction was correct?
# ability not to label as positive a sample that is negative
# tp/(tp + fp)
vioPrec = precision_score(
    y_true, y_pred,
    pos_label="violation",
    zero_division=0
)

nonVioPrec = precision_score(
    y_true, y_pred,
    pos_label="non_violation",
    zero_division=0
)

# how many of the positive labels was the model able to predict?
# ability to find all the positive samples
# tp/(tp + fn)
vioRec = recall_score(
    y_true, y_pred,
    pos_label="violation",
    zero_division=0
)

nonVioRec = recall_score(
    y_true, y_pred,
    pos_label="non_violation",
    zero_division=0
)

# harmonic mean of precision and recall
# how good is your model in general?
# 2(prec * recall)/(precision + recall)
f1 = f1_score(
    y_true, y_pred,
    pos_label="violation",
    zero_division=0
)

cm = confusion_matrix(y_true, y_pred, labels=["non_violation", "violation"])

metrics = {
    "model": MODEL,
    "prompt": PROMPT_NAME,
    "num_tests": NUM_TESTS,
    "batch_size": BATCH_SIZE,
    "accuracy": acc,
    "violation_precision": vioPrec,
    "non_violation_precision": nonVioPrec,
    "violation_recall": vioRec,
    "non_violation_recall": nonVioRec,
    "f1": f1,
    "confusion_matrix": cm.tolist()
}

with open(f"{path}/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

