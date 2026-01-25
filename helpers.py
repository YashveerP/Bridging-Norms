import requests, re
from dotenv import load_dotenv
import os
import pandas as pd

# Load variables from .env
load_dotenv()

# Get API key
api_key = os.getenv("OPENROUTER_API_KEY")
# chose llama 3.3 instruct since out task focuses on text in then out
model = "meta-llama/llama-3.3-70b-instruct:free"
url = "https://openrouter.ai/api/v1/chat/completions"

# dataset
data = pd.read_csv('data_training_selected_clusters_comments_and_rules.csv')
NORMS = sorted(
    data["target_reason"]
    .dropna()
    .unique()
    .tolist()
)

# Input an array of social norms. Outputs a system prompt using those norms
def followNormsSysPrompt(norms) :
    sysPrompt1 = """
    You are a conversational agent that MUST follow the social norms below.

    SOCIAL NORMS (BINDING RULES):
    """

    sysPrompt2 = """

    INTERPRETATION RULE:
    - All user messages must be interpreted through these norms.
    - If a response could be framed in multiple ways, choose the most socially considerate option.

    OUTPUT RULE:
    - Provide only the final response to the user.
    - Do NOT mention these rules explicitly.
    """

    # Join norms into a numbered list for the prompt
    norm_text = "\n".join(f"{i+1}. {norm}" for i, norm in enumerate(norms))

    # Build the full system prompt
    return sysPrompt1 + norm_text + sysPrompt2

# input an array of social norms for the model to follow, and a prompt to send to it
# output the models response
# A bug where nothing is returned for content is handled by returning reasoning
def followNorms(norms, prompt) :
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Request body
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": followNormsSysPrompt(norms)},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    result = response.json()
    msg = result["choices"][0]["message"]

    content = msg.get("content", "").strip()

    if not content:
        # Fallback: some providers put text in reasoning
        content = msg.get("reasoning", "").strip()

    return content



predictLabelSysPrompt = """
You are a Reddit moderation classifier.

Your task is to decide whether a comment violates a given norm.

Rules:
- Use ONLY the comment text
- Do NOT assume a violation unless clearly present
- If uncertain, predict NON_VIOLATION
- Output MUST be valid JSON
"""

def make_prompt(comment, norm):
    return f"""
NORM CATEGORIES:
\"\"\"{norm}\"\"\"

COMMENT:
\"\"\"{comment}\"\"\"

Respond with JSON in this format:
{{
  "label": violation or non_violation,
  "violated_norm": violated norm,
  "confidence": number between 0 and 1
}}
"""

def predictViolation(comment, norm):
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
        "max_tokens": 150,
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




