import requests, re
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()
# Get API key
api_key = os.getenv("OPENROUTER_API_KEY")
# chose llama 3.3 instruct since out task focuses on text in then out
# model = "meta-llama/llama-3.3-70b-instruct:free"
url = "https://openrouter.ai/api/v1/chat/completions"

predictLabelSysPrompt = """
You are a Reddit moderation classifier.

Your task is to decide whether a comment violates a given norm.

Rules:
- Use ONLY the comment text
- Count as a violation if their is explicit evidence
- Output MUST be valid JSON
"""

def make_prompt(comment, norm):
    return f"""
NORM TO CHECK:
\"\"\"{norm}\"\"\"

COMMENT:
\"\"\"{comment}\"\"\"

Respond with JSON in this format:
{{
  "label": violation or non_violation,
  "evidence": EXACT substring copied verbatim from the input text
}}
"""

def predictViolation(comment, norm, model):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Request body
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": predictLabelSysPrompt},
            {"role": "user", "content": make_prompt(comment, norm)}
        ],
        "max_tokens": 1000,
        "temperature": 0.0
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    result = response.json()
    msg = result["choices"][0]["message"]

    content = msg.get("content", "").strip()

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON in model output: {text}")

    return match.group(0)


