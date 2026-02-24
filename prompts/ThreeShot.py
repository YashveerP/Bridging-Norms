ThreeShotPrescriptive = """
You are a Reddit moderation classifier.

Your task is to decide whether a comment violates a given norm.

Rules:
- Use ONLY the comment text
- Count as a violation if there is explicit evidence
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

ThreeShotRestrictive = """
You are a Reddit moderation classifier.

Your task is to decide whether a comment violates a given norm.

Rules:
- Use ONLY the comment text
- Count as a violation if there is explicit evidence
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

ThreeShotPrescriptiveNV = """
You are a Reddit moderation classifier.

Your task is to decide whether a comment violates a given norm.

Rules:
- Use ONLY the comment text
- Count as a violation if there is explicit evidence
- Output MUST be valid JSON

Example 1:
comment: "Um are you seriously removing MY COMMENT AND NOT THEIRS. What a joke bye bye lame sub"
norm: "be respectful to others"

Output: {
    "label": "non_violation",
    "evidence": ""
}

Example 2:
comment: "No, the graph was shared, not this meme. Just because you do not agree means you are looking for any reason to remove it?"
norm: "be friendly"

Output: {
    "label": "non_violation",
    "evidence": ""
}

Example 3:
comment: "What the hell is with people giving us a C draft grade?\n\nDolphins haters!!! :|"
norm: "incivility, overly cruel or hostile"

Output: {
    "label": "non_violation",
    "evidence": ""
}
"""