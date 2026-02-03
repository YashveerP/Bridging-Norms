import requests, os, json, re
from dotenv import load_dotenv
from utils.prompts import (predictLabelMakePrompt, predictLabelSysPromptZS, predictLabelSysPromptOS, 
predictLabelSysPrompt3S, predictLabelSysPromptCOT, predictLabelMakePromptCOT1, predictLabelMakePromptCOT2, predictLabelMakePromptCOT3)

# Load variables from .env
load_dotenv()
# Get API key
api_key = os.getenv("OPENROUTER_API_KEY")
url = "https://openrouter.ai/api/v1/chat/completions"

def getModels():
    models = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"}
    ).json()

    for m in models["data"]:
        if ":free" in m["id"]:
            print(m["id"])

def predictViolation(comment, norm, model):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Request body
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": predictLabelSysPromptZS},
            {"role": "user", "content": predictLabelMakePrompt(comment, norm)}
        ],
        "max_tokens": 1000,
        "temperature": 0.0
    }

    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code != 200:
        print("ERROR RESPONSE:", response.text)
        response.raise_for_status()

    result = response.json()
    msg = result["choices"][0]["message"]

    content = msg.get("content", "").strip()

    parsed = parse_or_repair_json(content)

    # Extra safety: evidence must exist in comment
    if parsed["evidence"] and parsed["evidence"] not in comment:
        parsed["evidence"] = ""

    return json.dumps(parsed, ensure_ascii=False)


def parse_or_repair_json(content: str):
    content = content.strip()

    # --- 🔹 FIX #1: HANDLE EMPTY OUTPUT FIRST ---
    if not content:
        raise ValueError("Model returned empty content")

    # Try to extract JSON if the model added extra text
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON in model output: {repr(content)}")

    json_str = match.group(0)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # --- 🔹 FIX #2: REPAIR COMMON LLM MISTAKES ---
        json_str = re.sub(
            r'"label"\s*:\s*(violation|non_violation)',
            r'"label": "\1"',
            json_str
        )

        json_str = re.sub(
            r'"evidence"\s*:\s*"""(.*?)"""',
            lambda m: '"evidence": ' + json.dumps(m.group(1)),
            json_str,
            flags=re.DOTALL
        )

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse JSON even after repair: {repr(json_str)}")

def COT(comment, norm, model):
    messages = [
        {"role": "system", "content": predictLabelSysPromptCOT},
        {"role": "user", "content": predictLabelMakePromptCOT1(norm)},
        {"role": "user", "content": predictLabelMakePromptCOT2(comment)},
        {"role": "user", "content": predictLabelMakePromptCOT3()}
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Request body
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.0
    }

    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code != 200:
        print("ERROR RESPONSE:", response.text)
        response.raise_for_status()

    result = response.json()
    msg = result["choices"][0]["message"]

    content = msg.get("content", "").strip()

    parsed = parse_or_repair_json(content)

    # Extra safety: evidence must exist in comment
    if parsed["evidence"] and parsed["evidence"] not in comment:
        parsed["evidence"] = ""

    return json.dumps(parsed, ensure_ascii=False)
