def predictLabelMakePrompt(comment, norm):
    return f"""
NORM TO CHECK:
\"\"\"{norm}\"\"\"

COMMENT:
\"\"\"{comment}\"\"\"

Respond with JSON in this format:
{{
  "label": "violation" or "non_violation",
  "evidence": "EXACT substring copied verbatim from the COMMENT, with newlines escaped as \\n"
}}
"""

predictLabelSysPromptZS = """
You are a Reddit moderation classifier.

Your task is to decide whether a comment violates a given norm.

Rules:
- Use ONLY the comment text
- Count as a violation if their is explicit evidence
- Output MUST be valid JSON
"""

predictLabelSysPromptOS = """
You are a Reddit moderation classifier.

Your task is to decide whether a comment violates a given norm.

Rules:
- Use ONLY the comment text
- Count as a violation if their is explicit evidence
- Output MUST be valid JSON

Example:
comment: "Just take a Xanax and quit crying"
norm: "be respectful to others"

Output: {
"label": "violation", 
"evidence": "quit crying"
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
comment: "Just take a Xanax and quit crying"
norm: "be respectful to others"

Output: {
"label": "violation", 
"evidence": "quit crying"
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


def predictLabelMakePromptCOT3(_):
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