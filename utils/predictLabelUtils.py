import requests, os, json, re, ollama, time
from dotenv import load_dotenv
from prompts.prompts import buildMessages
import pandas as pd
import json, re, os
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# Load variables from .env
load_dotenv()
# Get API key
api_key = os.getenv("OPENROUTER_API_KEY")
url = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 5
NUM_TESTS = 100
BATCH_SIZE = 20

def predictViolation(runner, model, promptType, useCOT, extraInfo=""):
    promptName = promptType + ("-COT" if useCOT else "") + extraInfo
    safe_model = re.sub(r'[<>:"/\\|?*]', '_', model)
    path = f"results/{promptName}/{safe_model}"

    df = pd.read_csv('datasets/tests.csv')
    dict = df.set_index("comment_id").to_dict("index")

    # go through each violated comment and store json
    results = []
    rawOutput = []

    # store true and prediceted labels
    y_true = []
    y_pred = []

    # go through tests in batches and with progress bar
    for start in tqdm(range(0, NUM_TESTS, BATCH_SIZE), desc=safe_model, position=0):
        batch_df = df.iloc[start:start + BATCH_SIZE]

        # build input in the format the prompt expects
        batch_input = []
        commentIDs = []
        for idx, row in batch_df.iterrows():
            batch_input.append({
                "norm": row["norm"],
                "comment": row["body"]
            })
            commentIDs.append(row["comment_id"])
        # get output from model
        if runner == "local":
            output = localPredictViolation(batch_input, model, promptType, useCOT)
        else:
            output =  openRouterPredictViolation(batch_input, model, promptType, useCOT)
        rawOutput.append(output)
        # load the json data into a list
        parsed_list = json.loads(output)

        i = 0
        for item in parsed_list:
            comment_id = commentIDs[i]
            i += 1
            row = dict[comment_id]

            # if the item had no evidence or invalid evidence replace with empyty quotes
            if item["evidence"] and item["evidence"].strip() not in row["body"].replace("\r", ""):
                item["evidence"] = ""

            # append to results
            results.append({
                "comment_id": comment_id,
                "body": row["body"],
                "norm": row["norm"],
                "true_label": row["true_label"],
                "pred_label": item["label"],
                "evidence": item["evidence"],
            })
            y_true.append(row["true_label"])
            y_pred.append(item["label"])

        # save progress
        os.makedirs(f"{path}", exist_ok=True)
        with open(f"{path}/results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        with open(f"{path}/output.json", "w", encoding="utf-8") as f:
            json.dump(rawOutput, f, indent=2, ensure_ascii=False)


    # metrics
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
    
    with open(f"{path}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

def openRouterPredictViolation(batch: list[dict], model, promptType, useCOT):
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": buildMessages(promptType, useCOT, batch),
        "max_tokens": 5000,
        "temperature": 0.0
    }

    # give multiple attempts for model to correctly output
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, headers=headers, json=data)
        except requests.exceptions.RequestException as e:
            print(f"Request failed (attempt {attempt+1}/{MAX_RETRIES})")
            print(e)
            time.sleep(2 ** attempt)
            continue

        # 429 error
        if response.status_code == 429:
            wait_time = 2 ** attempt  # exponential backoff: 1,2,4,8,16...
            print(f"429 received. Retrying in {wait_time}s (attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(wait_time)
            continue

        # Fail on other errors
        if response.status_code != 200:
            print("ERROR RESPONSE:", response.text)
            response.raise_for_status()

        result = response.json()
        msg = result["choices"][0]["message"]
        content = msg.get("content", "").strip()

        parsed_list = parse_or_repair_json(content)
        return json.dumps(parsed_list, ensure_ascii=False)

    # If we exhausted attempts
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries due to repeated errors")

def localPredictViolation(batch, model, promptType, useCOT):

    try:
        response = ollama.chat(
        model=model,
        messages=buildMessages(promptType, useCOT, batch),
            options={
                "temperature": 0.0,
                "max_tokens": 1000
            }
        )

        content = response["message"]["content"].strip()

        if not content:
            raise ValueError("Empty model output")

        parsed_list = parse_or_repair_json(content)

        return json.dumps(parsed_list, ensure_ascii=False)

    except Exception as e:
        print(f"localPredictViolation failed")
        print("Error:", e)

    raise RuntimeError("Local model retries exhausted")




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

    
    # 1: Extract the most plausible JSON block
    # Prefer arrays, but fall back to single objects.
    array_match = re.search(r"\[.*\]", text, re.DOTALL)
    obj_match = re.search(r"\{.*\}", text, re.DOTALL)

    if array_match:
        json_str = array_match.group(0)
    elif obj_match:
        json_str = obj_match.group(0)
    else:
        raise ValueError(f"No JSON found in model output: {repr(text)}")

    # STEP 2: Normalize common LLM mistakes
    def repair(s: str) -> str:
        # 1) Fix unquoted labels
        s = re.sub(
            r'"label"\s*:\s*(violation|non_violation)',
            r'"label": "\1"',
            s
        )

        # 2) Convert SINGLE-quoted evidence
        s = re.sub(
            r'"evidence"\s*:\s*\'(.*?)\'',
            lambda m: '"evidence": ' + json.dumps(m.group(1)),
            s,
            flags=re.DOTALL
        )

        # 3) Convert trple-quoted evidence
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

        # 6) Fix accidental double-braces
        s = re.sub(r"\[\s*\{\s*\{", "[{", s)
        s = re.sub(r"\}\s*\}\s*\]", "}]", s)

        # 7) Fix extra closing bracket
        s = re.sub(r"\]\s*\]\s*$", "]", s) 

        # 9) Remove DUPLICATE "label" keys
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

    
    #3: Try parsing
    attempts = []

    # Try raw
    attempts.append(json_str)

    # Try once repaired
    attempts.append(repair(json_str))

    # Try double-repaired
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

    # 4: try to salvage
    objects = re.findall(r"\{.*?\}", attempts[-1], re.DOTALL)

    if not objects:
        raise ValueError("No JSON objects found during salvage attempt.")

    results = []

    for i, obj in enumerate(objects):
        try:
            fixed = repair(obj)
            parsed_obj = json.loads(fixed)
            results.append(parsed_obj)

        except Exception as e:
            print("JSON PARSE FAILURE")
            print(f"Object index: {i}")
            print(f"Error: {e}")
            print(f"Raw object:\n{obj}")
            raise

    #Enforce batch size
    if len(results) != BATCH_SIZE:
        raise ValueError(
            f"Expected {BATCH_SIZE} items, got {len(results)}"
        )

    return results


