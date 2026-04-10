import requests, os, json, re, ollama, time
from dotenv import load_dotenv
from utils.jsonParser import parse_or_repair_json
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from param import *
from defs import *

# Load variables from .env
load_dotenv()
# Get API key
api_key = os.getenv("OPENROUTER_API_KEY")
url = "https://openrouter.ai/api/v1/chat/completions"


async def predictViolation(model, prompt, directory):
    safe_model = re.sub(r'[<>:"/\\|?*]', '_', model.name)
    path = f"results/{directory}/{safe_model}"
    os.makedirs(f"{path}", exist_ok=True)

    df = pd.read_csv('datasets/tests.csv')
    dict = df.set_index("comment_id").to_dict("index")

    # go through each violated comment and store json
    results = []
    rawOutput = []
    numSucceses = 0

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
        if model.runner == "local":
            output = localPredictViolation(batch_input, model, prompt)
        else:
            async with aiohttp.ClientSession() as session:
                output = await openRouterPredictViolation(batch_input, model, prompt, session)
        rawOutput.append(output)
        # load the json data into a list

        repairedOutput = parse_or_repair_json(output, path)
        parsed_list = repairedOutput
        numSucceses += len(parsed_list)
        ##
        if len(parsed_list) != BATCH_SIZE:
            print(f"WARNING: {model.name} outputed {len(parsed_list)}/{BATCH_SIZE} results for batch #{int(start/BATCH_SIZE) + 1}")

        for item in parsed_list:
            comment_id = item["comment_id"]
            row = dict[comment_id]

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
    #print out message saying the results are invalid if not all tests were able to run

    metrics = {
        "model": model.name,
        "directory": directory,
        "num_tests": NUM_TESTS,
        "num_successes": numSucceses,
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

# def openRouterPredictViolation(batch: list[dict], model, prompt):
        
#     headers = {
#         "Authorization": f"Bearer {api_key}",
#         "Content-Type": "application/json"
#     }

#     data = {
#         "model": model.name,
#         "messages": prompt.buildMessages(prompt, batch),
#         "max_tokens": 5000,
#         "temperature": 0.0
#     }

#     # give multiple attempts for model to correctly output
#     for attempt in range(MAX_RETRIES):
#         try:
#             response = requests.post(url, headers=headers, json=data)
#         except requests.exceptions.RequestException as e:
#             print(f"Request failed (attempt {attempt+1}/{MAX_RETRIES})")
#             print(e)
#             time.sleep(2 ** attempt)
#             continue

#         # 429 error
#         if response.status_code == 429:
#             wait_time = 2 ** attempt  # exponential backoff: 1,2,4,8,16...
#             print(f"429 received. Retrying in {wait_time}s (attempt {attempt+1}/{MAX_RETRIES})")
#             time.sleep(wait_time)
#             continue

#         # Fail on other errors
#         if response.status_code != 200:
#             print("ERROR RESPONSE:", response.text)
#             response.raise_for_status()

#         result = response.json()
#         msg = result["choices"][0]["message"]
#         content = msg.get("content", "").strip()

#         return content

#     # If we exhausted attempts
    # raise RuntimeError(f"Failed after {MAX_RETRIES} retries due to repeated errors")

def localPredictViolation(batch, model, prompt):
    try:
        response = ollama.chat(
        model=model.name,
        messages= prompt.buildMessages(prompt, batch),
            options={
                "temperature": 0.0,
                "max_tokens": 1000
            }
        )

        content = response["message"]["content"].strip()

        if not content:
            raise ValueError("Empty model output")
        
        return content

    except Exception as e:
        print(f"localPredictViolation failed")
        print("Error:", e)



import aiohttp
import asyncio

async def openRouterPredictViolation(batch: list[dict], model, prompt, session):

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model.name,
        "messages": prompt.buildMessages(prompt, batch),
        "max_tokens": 5000,
        "temperature": 0.0
    }

    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(url, headers=headers, json=data) as response:

                # 429 handling
                if response.status == 429:
                    wait_time = 2 ** attempt
                    print(f"429 retry in {wait_time}s (attempt {attempt+1})")
                    await asyncio.sleep(wait_time)
                    continue

                if response.status != 200:
                    text = await response.text()
                    print("ERROR RESPONSE:", text)
                    response.raise_for_status()

                result = await response.json()
                msg = result["choices"][0]["message"]
                content = msg.get("content", "").strip()

                return content

        except aiohttp.ClientError as e:
            print(f"Request failed (attempt {attempt+1}/{MAX_RETRIES})")
            print(e)
            await asyncio.sleep(2 ** attempt)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")