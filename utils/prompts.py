import json

def buildMessages(promptType, useCOT, batch):
    # Choose system prompt
    if promptType == "ZeroShot":
        sysPrompt = predictLabelSysPromptZS
    elif promptType == "OneShot":
        sysPrompt = predictLabelSysPrompt1S
    elif promptType == "FewShot":
        sysPrompt = predictLabelSysPrompt3S
    else:
        raise ValueError(f"Unknown PROMPT: {promptType}")

    #choose whether or not to use COT
    if useCOT:
        return [
        {"role": "system", "content": sysPrompt},
        {"role": "user", "content": predictLabelMakePromptCOT1(norm)},
        {"role": "user", "content": predictLabelMakePromptCOT2(comment)},
        {"role": "user", "content": predictLabelMakePromptCOT3()}
    ]
    else:
        return [
            {"role": "system", "content": sysPrompt},
            {"role": "user", "content": predictLabelMakePrompt(batch)}
        ]

predictLabelSysPromptZS = """
You are a Reddit moderation classifier.

Your task is to decide whether a comment violates a given norm.

Rules:
- Use ONLY the comment text
- Count as a violation if their is explicit evidence
- Output MUST be valid JSON
"""

predictLabelSysPrompt1S = """
You are a Reddit moderation classifier.

Your task is to decide whether a comment violates a given norm.

Rules:
- Use ONLY the comment text
- Count as a violation if their is explicit evidence
- Output MUST be valid JSON

Example:
comment: "$39.99 though? That’s a 20 fucking dollar game right there. It’s also almost 10 years old. You buy it, you become part of the reason why Nintendo gets to rip off everyone with their insane prices."
norm: "no hate-speech, personal attacks or harassment"

Output: {
"label": "violation", 
"evidence": "$39.99 though? That’s a 20 fucking dollar game right there. It’s also almost 10 years old. You buy it, you become part of the reason why Nintendo gets to rip off everyone with their insane prices."
}
"""

predictLabelSysPrompt3S = """
You are a Reddit moderation classifier.

Your task is to decide whether a comment violates a given norm.

Rules:
- Use ONLY the comment text
- Count as a violation if their is explicit evidence
- Output MUST be valid JSON

Example 1:
comment: "$39.99 though? That’s a 20 fucking dollar game right there. It’s also almost 10 years old. You buy it, you become part of the reason why Nintendo gets to rip off everyone with their insane prices."
norm: "no hate-speech, personal attacks or harassment"

Output: {
"label": "violation", 
"evidence": "$39.99 though? That’s a 20 fucking dollar game right there. It’s also almost 10 years old. You buy it, you become part of the reason why Nintendo gets to rip off everyone with their insane prices."
}

Example 2:
comment: "it expires after next season."
norm: "incivility, overly cruel or hostile"

Output: {
    "label": "non_violation",
    "evidence": ""
}

Example 3:
comment: "Maybe nobody wants to hang out with you because you're a big, fat freak?"
norm: "incivility, overly cruel or hostile"

Output: {
    "label": "violation",
    "evidence": "a big, fat freak"
}
"""

def predictLabelMakePrompt(content_to_check):
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


def predictLabelMakePromptCOT1(norm):
    return f"""
NORM TO CHECK:
\"\"\"{norm}\"\"\"

In ONE sentence, restate what this norm prohibits in your own words.
"""


def predictLabelMakePromptCOT2(comment):
    return f"""
COMMENT:
\"\"\"{comment}\"\"\"

Identify any parts of the comment that could be relevant to the norm.
Quote them verbatim. If nothing is relevant, say: "No relevant text found."
"""


def predictLabelMakePromptCOT3():
    return """
Based on all previous steps, provide ONLY valid JSON in exactly this format:

{
  "label": "violation" or "non_violation",
  "evidence": "EXACT substring copied verbatim from the COMMENT, with newlines escaped as \\n, or empty string if non_violation"
}
"""


predictLabelSysPromptCOT = """
You are a Reddit moderation classifier.

Your task is to decide whether a comment violates a given norm.

Rules:
- Use ONLY the comment text
- Count as a violation if their is explicit evidence
"""