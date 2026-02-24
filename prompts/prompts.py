import json
from prompts.ZeroShot import *
from prompts.OneShot import *
from prompts.ThreeShot import *
from prompts.SixShot import *

def buildMessages(promptType, useCOT, batch):
    # Choose system prompt
    if promptType == "ZeroShot":
        sysPrompt = zeroShot
    elif promptType == "OneShot":
        sysPrompt = oneShot
    elif promptType == "ThreeShot":
        sysPrompt = ThreeShotRestrictive
    elif promptType == "SixShot":
        sysPrompt = sixShotPrescriptiveAndRestrictive
    else:
        raise ValueError(f"Unknown PROMPT: {promptType}")

    #choose whether or not to use COT
    if useCOT:
        return [
        {"role": "system", "content": sysPrompt},
        {"role": "user", "content": makePromptCOT1(batch)},
        {"role": "user", "content": makePromptCOT2(batch)},
        {"role": "user", "content": makePromptCOT3(batch)}
    ]
    else:
        return [
            {"role": "system", "content": sysPrompt},
            {"role": "user", "content": makePrompt(batch)}
        ]


def makePrompt(content_to_check):
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
