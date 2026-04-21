import json
from defs import *
from utils.dataSetUtils import *

def compareCommmunitiesSysPrompt(subredditID):
    community = getCommunity(subredditID)
    rules = getCommunityRules(subredditID)
    comments = getCommunityComments(subredditID).head(8)
    return f"""
You are a Reddit moderation classifier for {community["display_name_prefixed"]}.
Community Description:
{community["public_description"]}

Community Rules:
{rules}

Moderated Comments:
{comments}

Your task is to decide whether a comment violates community norms.

Rules:
- Use ONLY the comment text
- Count as a violation if there is explicit evidence
- Output MUST be valid JSON
"""
        
def compareCommunitesUserPrompt(content_to_check):
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


