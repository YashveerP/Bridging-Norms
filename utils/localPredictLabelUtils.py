import ollama
import json
import re
from utils.predictLabelUtils import parse_or_repair_json
from utils.prompts import (predictLabelMakePrompt, predictLabelSysPromptZS, predictLabelSysPromptOS, 
predictLabelSysPrompt3S, predictLabelSysPromptCOT, predictLabelMakePromptCOT1, predictLabelMakePromptCOT2, predictLabelMakePromptCOT3)

MODEL="qwen2.5:7b-instruct"

def localPredictViolation(comment, norm):
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": predictLabelSysPromptOS},
            {"role": "user", "content": predictLabelMakePrompt(comment, norm)}
        ],
        options={
            "temperature": 0.0,
            "max_tokens": 1000
        }
    )

    content = response["message"]["content"].strip()
    parsed = parse_or_repair_json(content)

    # Extra safety: evidence must exist in comment
    if parsed["evidence"] and parsed["evidence"] not in comment:
        parsed["evidence"] = ""

    return json.dumps(parsed, ensure_ascii=False)

def COT(comment, norm):
    messages = [
        # ---------- SYSTEM PROMPT ----------
        {"role": "system", "content": predictLabelSysPromptCOT},

        # ---------- STEP 1: Interpret the norm ----------
        {"role": "user", "content": predictLabelMakePromptCOT1(norm)},

        # (Model will answer Step 1 here)

        # ---------- STEP 2: Identify relevant parts ----------
        {"role": "user", "content": predictLabelMakePromptCOT2(comment)},

        # (Model will answer Step 2 here)

        # ---------- STEP 3: Final JSON decision ----------
        {"role": "user", "content": predictLabelMakePromptCOT3("Use your prior reasoning above.")}
    ]

    response = ollama.chat(
        model=MODEL,
        messages=messages,
        options={
            "temperature": 0.0,   # low temp for final classification
            "max_tokens": 1500
        }
    )

    content = response["message"]["content"].strip()
    parsed = parse_or_repair_json(content)

    # Extra safety: evidence must exist in comment
    if parsed["evidence"] and parsed["evidence"] not in comment:
        parsed["evidence"] = ""

    return json.dumps(parsed, ensure_ascii=False)
