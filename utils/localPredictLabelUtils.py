import ollama
import json
import re
from utils.predictLabelUtils import make_prompt, parse_or_repair_json
from utils.prompts import predictLabelSysPrompt

MODEL="qwen2.5:7b-instruct"

def localPredictViolation(comment, norm):
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": predictLabelSysPrompt},
            {"role": "user", "content": make_prompt(comment, norm)}
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
