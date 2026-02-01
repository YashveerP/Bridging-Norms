import ollama
import json
import re

model="qwen2.5:7b-instruct"


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

Respond ONLY with valid JSON in this format:
{{
  "label": "violation" or "non_violation",
  "evidence": "EXACT substring copied verbatim from the COMMENT, with newlines escaped as \\n"
}}

Rules:
- Do NOT paraphrase or summarize
- Do NOT include explanations
- The evidence string MUST appear verbatim in the COMMENT
- The "label" must be a JSON string (quoted)
"""


def parse_or_repair_json(content):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Fix unquoted labels
        content = re.sub(r'"label"\s*:\s*(violation|non_violation)',
                         r'"label": "\1"', content)
        # Fix triple-quoted evidence
        content = re.sub(r'"evidence"\s*:\s*"""(.*?)"""',
                         lambda m: '"evidence": ' + json.dumps(m.group(1)),
                         content,
                         flags=re.DOTALL)
        return json.loads(content)

def localPredictViolation(comment, norm):
    response = ollama.chat(
        model=model,
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
