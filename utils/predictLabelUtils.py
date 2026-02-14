import requests, os, json, re, ollama, time
from dotenv import load_dotenv
from utils.prompts import buildMessages
import pandas as pd
import json, re, os
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load variables from .env
load_dotenv()
# Get API key
api_key = os.getenv("OPENROUTER_API_KEY")
url = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES= 3
NUM_TESTS = 100
BATCH_SIZE = 20

def predictViolation(runner, model, promptType, useCOT, extraInfo=""):
    promptName = promptType + ("-COT" if useCOT else "") + extraInfo
    safe_model = re.sub(r'[<>:"/\\|?*]', '_', model)
    path = f"results/{promptName}"

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
        if runner == "local":
            output = localPredictViolation(batch_input, model, promptType, useCOT)
        else:
            output =  openRouterPredictViolation(batch_input, model, promptType, useCOT)

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
        os.makedirs(f"{path}/{safe_model}", exist_ok=True)
        with open(f"{path}/{safe_model}/results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


    # fraction of correct predicitons
    acc = accuracy_score(y_true, y_pred)

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

    cm = confusion_matrix(y_true, y_pred, labels=["non_violation", "violation"])

    metrics = {
        "model": model,
        "prompt": promptName,
        "num_tests": NUM_TESTS,
        "batch_size": BATCH_SIZE,
        "accuracy": acc,
        "violation_precision": vioPrec,
        "non_violation_precision": nonVioPrec,
        "violation_recall": vioRec,
        "non_violation_recall": nonVioRec,
        "confusion_matrix": cm.tolist()
    }

    # Load existing metrics
    if os.path.exists(f"{path}/metrics.json"):
        with open(f"{path}/metrics.json", "r", encoding="utf-8") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = []

    # Append new run
    all_metrics.append(metrics)

    # Save back
    with open(f"{path}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

def openRouterPredictViolation(batch: list[dict], model, promptType, useCOT):
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Request body
    data = {
        "model": model,
        "messages": buildMessages(promptType, useCOT, batch),
        "max_tokens": 5000,
        "temperature": 0.0
    }

    for attempt in range(MAX_RETRIES):
        response = requests.post(url, headers=headers, json=data)

        # ----- HANDLE 429 (rate limit) -----
        if response.status_code == 429:
            wait_time = 2 ** attempt  # exponential backoff: 1,2,4,8,16...
            print(f"429 received. Retrying in {wait_time}s (attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(wait_time)
            continue

        # ----- Other errors: fail immediately -----
        if response.status_code != 200:
            print("ERROR RESPONSE:", response.text)
            response.raise_for_status()

        # ----- Success path -----
        result = response.json()
        msg = result["choices"][0]["message"]
        content = msg.get("content", "").strip()

        parsed_list = parse_or_repair_json(content)
        return json.dumps(parsed_list, ensure_ascii=False)

    # If we exhausted retries
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries due to repeated 429 errors")

def localPredictViolation(batch: list[dict], model, promptType, useCOT):
    response = ollama.chat(
        model=model,
        messages= buildMessages(promptType, useCOT, batch),
        options={
            "temperature": 0.0,
            "max_tokens": 1000
        }
    )

    content = response["message"]["content"].strip()
    parsed_list = parse_or_repair_json(content)

    return json.dumps(parsed_list, ensure_ascii=False)




import json
import re
from typing import Any

def parse_or_repair_json(content: str) -> Any:
    """
    Robust parser for messy LLM JSON output.
    Returns either:
      - a list of dicts, or
      - a single dict
    """

    if not content or not content.strip():
        raise ValueError("Model returned empty content")

    text = content.strip()

    # --------------------------------------------------
    # STEP 1: Extract the most plausible JSON block
    # Prefer arrays, but fall back to single objects.
    # --------------------------------------------------
    array_match = re.search(r"\[.*\]", text, re.DOTALL)
    obj_match = re.search(r"\{.*\}", text, re.DOTALL)

    if array_match:
        json_str = array_match.group(0)
    elif obj_match:
        json_str = obj_match.group(0)
    else:
        raise ValueError(f"No JSON found in model output: {repr(text)}")

    # --------------------------------------------------
    # STEP 2: Normalize common LLM mistakes
    # --------------------------------------------------

    def repair(s: str) -> str:
        # 1) Fix unquoted labels:  "label": violation  ->  "label": "violation"
        s = re.sub(
            r'"label"\s*:\s*(violation|non_violation)',
            r'"label": "\1"',
            s
        )

        # 2) Convert SINGLE-quoted evidence -> valid JSON
        s = re.sub(
            r'"evidence"\s*:\s*\'(.*?)\'',
            lambda m: '"evidence": ' + json.dumps(m.group(1)),
            s,
            flags=re.DOTALL
        )

        # 3) Convert TRIPLE-quoted evidence -> valid JSON
        s = re.sub(
            r'"evidence"\s*:\s*"""(.*?)"""',
            lambda m: '"evidence": ' + json.dumps(m.group(1)),
            s,
            flags=re.DOTALL
        )

        # 4) Remove trailing commas before } or ]
        s = re.sub(r",\s*([}\]])", r"\1", s)

        # 5) Normalize weird quotes (smart quotes)
        s = s.replace("“", '"').replace("”", '"').replace("’", "'")

        # 6) Fix accidental double-braces: [{ { ... } }] -> [{ ... }]
        s = re.sub(r"\[\s*\{\s*\{", "[{", s)
        s = re.sub(r"\}\s*\}\s*\]", "}]", s)
        
        # 7) Fix extra closing bracket: [...] ] -> [...]
        s = re.sub(r"\]\s*\]\s*$", "]", s) 
        # 8) Fix pattern:  "label": "comment_id": 23  ->  "comment_id": 23
        s = re.sub(
            r'"label"\s*:\s*"comment_id"\s*:\s*(\d+)',
            r'"comment_id": \1',
            s
        )

        # 9) Remove DUPLICATE "label" keys (keep the LAST one)
        def dedupe_labels(match):
            block = match.group(0)

            labels = re.findall(r'"label"\s*:\s*"([^"]+)"', block)
            if labels:
                final_label = labels[-1]  # keep last one
                # remove all existing label fields
                block = re.sub(r'"label"\s*:\s*"[^"]+",?', "", block)
                # re-insert exactly one label at the top
                block = block.replace("{", f'{{\n  "label": "{final_label}",', 1)

            return block

        s = re.sub(r"\{[^{}]*\}", dedupe_labels, s, flags=re.DOTALL)

       # 10) If it *looks like* a sequence of JSON objects, wrap in [ ]
        stripped = s.strip()

        # Case: starts with { and contains multiple objects
        if stripped.startswith("{") and stripped.count("{") > 1:
            # Remove a leading '[' if the model partially added one (defensive)
            if stripped.startswith("["):
                stripped = stripped[1:]

            # Remove a trailing ']' if partially added
            if stripped.endswith("]"):
                stripped = stripped[:-1]

            s = "[" + stripped + "]"


        return s

    # --------------------------------------------------
    # STEP 3: Try parsing (with increasing repair)
    # --------------------------------------------------
    attempts = []

    # Try raw
    attempts.append(json_str)

    # Try once repaired
    attempts.append(repair(json_str))

    # Try double-repaired (for really bad outputs)
    attempts.append(repair(repair(json_str)))

    last_error = None
    for i, candidate in enumerate(attempts):
        try:
            parsed = json.loads(candidate)

            # If the model returned a single object, wrap in a list
            if isinstance(parsed, dict):
                parsed = [parsed]

            return parsed

        except Exception as e:
            last_error = e
            # keep trying

    # --------------------------------------------------
    # STEP 4: If we reach here, give a useful error
    # --------------------------------------------------
    raise ValueError(
        "Could not parse JSON even after robust repair.\n"
        f"Last attempted string:\n{repr(attempts[-1])}\n"
        f"Last error: {last_error}"
    )



