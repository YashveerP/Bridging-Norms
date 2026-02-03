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
            {"role": "system", "content": predictLabelSysPromptZS},
            {"role": "user", "content": predictLabelMakePrompt(comment, norm)}
        ],
        options={
            "temperature": 0.0,
            "max_tokens": 1000
        }
    )

    content = response["message"]["content"].strip()
    parsed = parse_or_repair_json(content)

    if parsed["evidence"] and parsed["evidence"] not in comment:
        parsed["evidence"] = ""

    return json.dumps(parsed, ensure_ascii=False)

def localCOT(comment, norm):
    messages = [
        {"role": "system", "content": predictLabelSysPromptCOT},
        {"role": "user", "content": predictLabelMakePromptCOT1(norm)},
        {"role": "user", "content": predictLabelMakePromptCOT2(comment)},
        {"role": "user", "content": predictLabelMakePromptCOT3()}
    ]

    response = ollama.chat(
        model=MODEL,
        messages=messages,
        options={
            "temperature": 0.0,
            "max_tokens": 1500
        }
    )

    content = response["message"]["content"].strip()
    parsed = parse_or_repair_json(content)

    if parsed["evidence"] and parsed["evidence"] not in comment:
        parsed["evidence"] = ""

    return json.dumps(parsed, ensure_ascii=False)
