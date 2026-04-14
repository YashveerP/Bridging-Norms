import json
from prompts.ZeroShot import *
from prompts.OneShot import *
from prompts.ThreeShot import *
from prompts.SixShot import *
from defs import *

def buildMessages(prompt, batch):
        return [
            {"role": "system", "content": prompt.sysPrompt},
            {"role": "user", "content": prompt.userPrompt(batch)}
        ]


def predictViolationUserPrompt(content_to_check):
    """
    Input:
        content_to_check: list of dicts:
        [
            {
                "comment_id": XX,
                "norm": "...",
                "comment": "..."
            },
            ...
        ]
    """
    return f"""
For each of the following comments in **valid JSON format**:

{json.dumps(content_to_check, indent=2, ensure_ascii=False)}

Respond with a **single valid JSON array** in exactly this format:

[
  {{
    "comment_id": XX,
    "label": "violation" or "non_violation",
    "evidence": "EXACT substring copied verbatim from the COMMENT, with newlines escaped as \\n"
  }},
  ...
]
"""


def makePromptCOT1(content_to_check):
    return f"""
For each of the following comments, reason independently:

{json.dumps(content_to_check, indent=2, ensure_ascii=False)}

In ONE sentence, restate what the norm prohibits in your own words.
"""

def makePromptCOT2(content_to_check):
    return f"""
For each of the following comments, reason independently:

{json.dumps(content_to_check, indent=2, ensure_ascii=False)}

Identify any parts of the comment that could be relevant to the norm.
Quote them verbatim. If nothing is relevant, say: "No relevant text found."
"""

def makePromptCOT3(content_to_check):
    return f"""
For each of the following comments, reason independently:

{json.dumps(content_to_check, indent=2, ensure_ascii=False)}

Based on all previous steps, respond with a **single valid JSON array** in exactly this format:

[
  {{
    "comment_id": XX,
    "label": "violation" or "non_violation",
    "evidence": "EXACT substring copied verbatim from the COMMENT, with newlines escaped as \\n"
  }},
  ...
]
"""


chainOfThought = """
You are a Reddit moderation classifier.

Your task is to decide whether a comment violates a given norm.

Rules:
- Use ONLY the comment text
- Count as a violation if there is explicit evidence
"""
