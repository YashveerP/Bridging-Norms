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

predictLabelSysPrompt3SSuccessfulExamples = """
You are a Reddit moderation classifier.

Your task is to decide whether a comment violates a given norm.

Rules:
- Use ONLY the comment text
- Count as a violation if their is explicit evidence
- Output MUST be valid JSON

Example 1:
comment:  "Screw your stupid Rick roll.  And it wasn’t even the right song….get bent bro lol"
norm: "no hate-speech, personal attacks or harassment"

Output: {
"label": "violation", 
"evidence": "Screw your stupid Rick roll.  And it wasn’t even the right song….get bent bro lol"
}

Example 2:
comment: "Dude save your typing, it's not worth it.  Haziko is God's most brain damaged Destiny redditor.  \n\n&amp;#x200B;\n\nEvery time he loses an argument (see: often) he goes \"I'm not gonna continue\", repeats his debunked low-tier points, and declares himself the winner.  He's literally downvote-on-sight tier."
norm: "no uncivil behavior, witchhunting, etc"

Output: {
    "label": "violation",
    "evidence": "Haziko is God's most brain damaged Destiny redditor."
}

Example 3:
comment: "The \"man\" is a lying piss-baby who can't accept that they're wrong."
norm: "no uncivil behavior, witchhunting, etc"

Output: {
    "label": "violation",
    "evidence": "The \"man\" is a lying piss-baby who can't accept that they're wrong."
}
"""

predictLabelSysPrompt3SNEW = """
You are a Reddit moderation classifier.

Your task is to decide whether a comment violates a given norm.

Rules:
- Use ONLY the comment text
- Count as a violation if their is explicit evidence
- Output MUST be valid JSON

Example 1:
comment:  "&gt;I never understood peoples outrage with the game.\n\nMaybe because the game was unfinished? theres no 8th gym (clearly unfinished), 7th gym has no city and is a  corridor. the monotonous elevator ride that was clearly rushed. routes that take 5 seconds to cross. \n\nwild area is completely devoid of content except pop in models. \n\nstory clearly getting cut as things happen off screen.\n\nthey also LIED about not reusing models. all the old models are reused.\n\nI mean, yea its great. some people will gladly guzzle down shit. thats great. some of us expect a finished game when we pay full price. Sword and Shield is a 15 dollar game."
norm: "no hate-speech, personal attacks or harassment"

Output: {
"label": "violation", 
"evidence": "some people will gladly guzzle down shit."
}

Example 2:
comment: "I hope you get 63 downvotes"
norm: "no hate-speech, personal attacks or harassment"

Output: {
    "label": "violation",
    "evidence": "I hope you get 63 downvotes"
}

Example 3:
comment: "What’s up with all the rainbow stuff on your profile avatar? 🤔 \nYa I’ll refrain from taking your advice."
norm: "treat people kindly"

Output: {
    "label": "violation",
    "evidence": "What’s up with all the rainbow stuff on your profile avatar?"
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
comment:  "&gt;I never understood peoples outrage with the game.\n\nMaybe because the game was unfinished? theres no 8th gym (clearly unfinished), 7th gym has no city and is a  corridor. the monotonous elevator ride that was clearly rushed. routes that take 5 seconds to cross. \n\nwild area is completely devoid of content except pop in models. \n\nstory clearly getting cut as things happen off screen.\n\nthey also LIED about not reusing models. all the old models are reused.\n\nI mean, yea its great. some people will gladly guzzle down shit. thats great. some of us expect a finished game when we pay full price. Sword and Shield is a 15 dollar game."
norm: "no hate-speech, personal attacks or harassment"

Output: {
"label": "violation", 
"evidence": "some people will gladly guzzle down shit."
}

Example 2:
comment: "I hope you get 63 downvotes"
norm: "no hate-speech, personal attacks or harassment"

Output: {
    "label": "violation",
    "evidence": "I hope you get 63 downvotes"
}

Example 3:
comment: "What’s up with all the rainbow stuff on your profile avatar? 🤔 \nYa I’ll refrain from taking your advice."
norm: "treat people kindly"

Output: {
    "label": "violation",
    "evidence": "What’s up with all the rainbow stuff on your profile avatar?"
}

Example 4:
comment: "Lol not you claiming blackness to speak with such authority about racism towards asians. Lol. Step back my latin looking, live behind a security fenced private mansion, but will evoke the black card for limited beneficial situation, redditor. . . Lol"
norm: "trolling"

Output: {
    "label": "non_violation",
    "evidence": ""
}

Example 5:
comment: "If he is lucky he will get to see the police throw their speakers off the roof."
norm: "post in a civilized manner"

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