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
MAX_RETRIES= 3
NUM_TESTS = 100
BATCH_SIZE = 20

def predictViolation(runner, model, promptType, useCOT, extraInfo=""):
    promptName = promptType + ("-COT" if useCOT else "") + extraInfo
    safe_model = re.sub(r'[<>:"/\\|?*]', '_', model)
    path = f"results/{promptName}/{safe_model}"

    df = pd.read_csv('datasets/train.csv')
    dict = df.set_index("comment_id").to_dict("index")

    # go through each violated comment and store json
    results = []

    # store true and prediceted labels
    y_true = []
    y_pred = []

    # go through tests in batches and with progress bar
    for start in tqdm(range(0, NUM_TESTS, BATCH_SIZE), desc=safe_model, position=0):
        batch_df = df.iloc[start:start + BATCH_SIZE]

        # build input in the format the prompt expects
        batch_input = []
        for idx, row in batch_df.iterrows():
            batch_input.append({
                "comment_id": row["comment_id"], 
                "norm": row["norm"],
                "comment": row["body"]
            })

        # get output from model
        if runner == "local":
            output = localPredictViolation(batch_input, model, promptType, useCOT)
        else:
            output =  openRouterPredictViolation(batch_input, model, promptType, useCOT)

        # load the json data into a list
        parsed_list = json.loads(output)

        for item in parsed_list:
            comment_id = item["comment_id"]
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

    Returns:
        list[dict]
    """

    if not content or not content.strip():
        raise ValueError("Model returned empty content")

    text = content.strip()

    # --------------------------------------------------
    # STEP 1: Extract likely JSON block (non-greedy)
    # --------------------------------------------------
    array_match = re.search(r"\[.*?\]", text, re.DOTALL)
    obj_match = re.search(r"\{.*?\}", text, re.DOTALL)

    if array_match:
        json_str = array_match.group(0)
    elif obj_match:
        json_str = obj_match.group(0)
    else:
        raise ValueError(f"No JSON found in model output:\n{text}")

    # --------------------------------------------------
    # STEP 2: Repair common LLM issues
    # --------------------------------------------------
    def repair(s: str) -> str:

        # 0) Fix invalid escapes like \. \, \( etc.
        s = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)

        # 1) Unquoted labels
        s = re.sub(
            r'"label"\s*:\s*(violation|non_violation)',
            r'"label": "\1"',
            s
        )

        # 2) SINGLE quoted evidence
        s = re.sub(
            r'"evidence"\s*:\s*\'(.*?)\'',
            lambda m: '"evidence": ' + json.dumps(m.group(1)),
            s,
            flags=re.DOTALL
        )

        # 3) TRIPLE quoted evidence
        s = re.sub(
            r'"evidence"\s*:\s*"""(.*?)"""',
            lambda m: '"evidence": ' + json.dumps(m.group(1)),
            s,
            flags=re.DOTALL
        )

        # 4) Remove trailing commas
        s = re.sub(r",\s*([}\]])", r"\1", s)

        # 5) Normalize smart quotes
        s = s.replace("“", '"').replace("”", '"').replace("’", "'")

        # 6) Fix accidental double braces
        s = re.sub(r"\[\s*\{\s*\{", "[{", s)
        s = re.sub(r"\}\s*\}\s*\]", "}]", s)

        # 7) Extra closing bracket
        s = re.sub(r"\]\s*\]\s*$", "]", s)

        # 8) Fix broken key pattern
        s = re.sub(
            r'"label"\s*:\s*"comment_id"\s*:\s*(\d+)',
            r'"comment_id": \1',
            s
        )

        # 9) Fix missing quote on comment_id
        s = re.sub(
            r'(?<!")comment_id"\s*:',
            r'"comment_id":',
            s
        )

        # 10) Wrap objects in [ ] if model forgot array
        stripped = s.strip()
        if stripped.startswith("{") and "},{" in stripped:
            s = "[" + stripped + "]"

        return s

    # --------------------------------------------------
    # STEP 3: Try parsing normally
    # --------------------------------------------------
    attempts = [
        json_str,
        repair(json_str),
        repair(repair(json_str))
    ]

    last_error = None

    for candidate in attempts:
        try:
            parsed = json.loads(candidate)

            if isinstance(parsed, dict):
                parsed = [parsed]

            # normalize output
            for item in parsed:
                if isinstance(item, dict):
                    item.setdefault("comment_id", None)
                    item.setdefault("label", "non_violation")
                    item.setdefault("evidence", "")

            return parsed

        except Exception as e:
            last_error = e

    # --------------------------------------------------
    # STEP 4 ⭐ SALVAGE MODE
    # Parse objects individually if whole batch fails
    # --------------------------------------------------
    objects = re.findall(r"\{.*?\}", attempts[-1], re.DOTALL)

    salvaged = []

    for obj in objects:
        try:
            fixed = repair(obj)
            parsed_obj = json.loads(fixed)

            parsed_obj.setdefault("comment_id", None)
            parsed_obj.setdefault("label", "non_violation")
            parsed_obj.setdefault("evidence", "")

            salvaged.append(parsed_obj)

        except Exception:
            continue

    if salvaged:
        print(f"⚠️ Salvaged {len(salvaged)} items from broken batch")
        return salvaged

    # --------------------------------------------------
    # STEP 5: Fail with useful error
    # --------------------------------------------------
    raise ValueError(
        "Could not parse JSON even after robust repair.\n"
        f"Last attempted string:\n{repr(attempts[-1])}\n"
        f"Last error: {last_error}"
    )




