import json

def buildMessages(promptType, useCOT, batch):
    # Choose system prompt
    if promptType == "ZeroShot":
        sysPrompt = predictLabelSysPromptZS
    elif promptType == "OneShot":
        sysPrompt = predictLabelSysPrompt1S
    elif promptType == "ThreeShot":
        sysPrompt = predictLabelSysPrompt3S
    elif promptType == "SixShot":
        sysPrompt = predictLabelSysPrompt6S
    else:
        raise ValueError(f"Unknown PROMPT: {promptType}")

    #choose whether or not to use COT
    if useCOT:
        return [
        {"role": "system", "content": sysPrompt},
        {"role": "user", "content": predictLabelMakePromptCOT1(batch)},
        {"role": "user", "content": predictLabelMakePromptCOT2(batch)},
        {"role": "user", "content": predictLabelMakePromptCOT3(batch)}
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
comment:  "&gt;I never understood peoples outrage with the game.\n\nMaybe because the game was unfinished? theres no 8th gym (clearly unfinished), 7th gym has no city and is a  corridor. the monotonous elevator ride that was clearly rushed. routes that take 5 seconds to cross. \n\nwild area is completely devoid of content except pop in models. \n\nstory clearly getting cut as things happen off screen.\n\nthey also LIED about not reusing models. all the old models are reused.\n\nI mean, yea its great. some people will gladly guzzle down shit. thats great. some of us expect a finished game when we pay full price. Sword and Shield is a 15 dollar game."
norm: "no hate-speech, personal attacks or harassment"

Output: {
"label": "violation", 
"evidence": "some people will gladly guzzle down shit."
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
comment: "You seem like a nice person"
norm: "personal attack",

Output: {
"label": "violation", 
"evidence": "You seem like a nice person"
}

Example 2:
comment: "&gt;I'm glad the 'white male' role is being disregarded again to represent our current society.\n\nSure buddy. He isn't a representation of current society. He represents a checklist.",
norm: "be nice"

Output: {
    "label": "violation",
    "evidence": "Sure buddy. He isn't a representation of current society. He represents a checklist."
}

Example 3:
comment: "The BBC hired Chinballs since he was down with their diversity and inclusion doctrine. That was the kicker for who the next showrunner was."
norm: "be nice"

Output: {
    "label": "violation",
    "evidence": "The BBC hired Chinballs since he was down with their diversity and inclusion doctrine."
}
"""

predictLabelSysPrompt3SOG = """
You are a Reddit moderation classifier.

Your task is to decide whether a comment violates a given norm.

Rules:
- Use ONLY the comment text
- Count as a violation if their is explicit evidence
- Output MUST be valid JSON

Example 1:
comment: "Transphobic? I think you'll find they're more afraid of me 🗿"
norm: "no uncivil behavior, witch hunting, etc"

Output: {
"label": "violation", 
"evidence": "Transphobic? I think you'll find they're more afraid of me"
}

Example 2:
comment: "Go touch grass maybe? You don’t need every new thing. It’s been two days."
norm: "no uncivil behavior, witchhunting, etc"

Output: {
    "label": "violation",
    "evidence": "Go touch grass maybe? You don’t need every new thing. It’s been two days."
}

Example 3:
comment: "Thanks for posting this, I needed it badly. Will never slap my bro's ass again."
norm: "not nice"

Output: {
    "label": "violation",
    "evidence": "Will never slap my bro's ass again."
}
"""

predictLabelSysPrompt6S = """
You are a Reddit moderation classifier.

Your task is to decide whether a comment violates a given norm.

Rules:
- Use ONLY the comment text
- Count as a violation if their is explicit evidence
- Output MUST be valid JSON

Example 1:
comment: "Transphobic? I think you'll find they're more afraid of me 🗿"
norm: "no uncivil behavior, witch hunting, etc"

Output: {
"label": "violation", 
"evidence": "Transphobic? I think you'll find they're more afraid of me"
}

Example 2:
comment: "Um are you seriously removing MY COMMENT AND NOT THEIRS. What a joke bye bye lame sub"
norm: "be respectful to others"

Output: {
    "label": "non_violation",
    "evidence": ""
}

Example 3:
comment: "Thanks for posting this, I needed it badly. Will never slap my bro's ass again."
norm: "not nice"

Output: {
    "label": "violation",
    "evidence": "Will never slap my bro's ass again."
}

Example 4:
comment: "No, the graph was shared, not this meme. Just because you do not agree means you are looking for any reason to remove it?"
norm: "be friendly"

Output: {
    "label": "non_violation",
    "evidence": ""
}

Example 5:
comment: "Go touch grass maybe? You don’t need every new thing. It’s been two days."
norm: "no uncivil behavior, witchhunting, etc"

Output: {
    "label": "violation",
    "evidence": "Go touch grass maybe? You don’t need every new thing. It’s been two days."
}

Example 6:
comment: "What the hell is with people giving us a C draft grade?\n\nDolphins haters!!! :|"
norm: "incivility, overly cruel or hostile"

Output: {
    "label": "non_violation",
    "evidence": ""
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


def predictLabelMakePromptCOT1(content_to_check):
    return f"""
For each of the following comments, reason independently:

{json.dumps(content_to_check, indent=2, ensure_ascii=False)}

In ONE sentence, restate what the norm prohibits in your own words.
"""

def predictLabelMakePromptCOT2(content_to_check):
    return f"""
For each of the following comments, reason independently:

{json.dumps(content_to_check, indent=2, ensure_ascii=False)}

Identify any parts of the comment that could be relevant to the norm.
Quote them verbatim. If nothing is relevant, say: "No relevant text found."
"""

def predictLabelMakePromptCOT3(content_to_check):
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


predictLabelSysPromptCOT = """
You are a Reddit moderation classifier.

Your task is to decide whether a comment violates a given norm.

Rules:
- Use ONLY the comment text
- Count as a violation if their is explicit evidence
"""