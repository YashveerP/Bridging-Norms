import json
import re
from typing import Any
def parse_or_repair_json(content: str, path) -> Any:
    """
    Robust parser for messy LLM JSON output.
    Returns either:
      - a list of dicts, or
      - a single dict
    """

    if not content or not content.strip():
        raise ValueError("Model returned empty content")

    text = content.strip()

    
    # 1: Extract the most plausible JSON block
    # Prefer arrays, but fall back to single objects.
    array_match = re.search(r"\[.*\]", text, re.DOTALL)
    obj_match = re.search(r"\{.*\}", text, re.DOTALL)

    if array_match:
        json_str = array_match.group(0)
    elif obj_match:
        json_str = obj_match.group(0)
    else:
        raise ValueError(f"No JSON found in model output: {repr(text)}")

    # STEP 2: Normalize common LLM mistakes
    def repair(s: str) -> str:
        # 1) Fix unquoted labels
        s = re.sub(
            r'"label"\s*:\s*(violation|non_violation)',
            r'"label": "\1"',
            s
        )

        # 2) Convert SINGLE-quoted evidence
        s = re.sub(
            r'"evidence"\s*:\s*\'(.*?)\'',
            lambda m: '"evidence": ' + json.dumps(m.group(1)),
            s,
            flags=re.DOTALL
        )

        # 3) Convert trple-quoted evidence
        s = re.sub(
            r'"evidence"\s*:\s*"""(.*?)"""',
            lambda m: '"evidence": ' + json.dumps(m.group(1)),
            s,
            flags=re.DOTALL
        )

        # 4) Remove trailing commas before } or ]
        s = re.sub(r",\s*([}\]])", r"\1", s)

        # 5) Normalize weird quotes (smart quotes)
        s = s.replace("“", '"').replace("”", '"').replace("’", "'")

        # 6) Fix accidental double-braces
        s = re.sub(r"\[\s*\{\s*\{", "[{", s)
        s = re.sub(r"\}\s*\}\s*\]", "}]", s)

        # 7) Fix extra closing bracket
        s = re.sub(r"\]\s*\]\s*$", "]", s) 

        # 9) Remove DUPLICATE "label" keys
        def dedupe_labels(match):
            block = match.group(0)

            labels = re.findall(r'"label"\s*:\s*"([^"]+)"', block)
            if labels:
                final_label = labels[-1]  # keep last one
                # remove all existing label fields
                block = re.sub(r'"label"\s*:\s*"[^"]+",?', "", block)
                # re-insert exactly one label at the top
                block = block.replace("{", f'{{\n  "label": "{final_label}",', 1)

            return block

        s = re.sub(r"\{[^{}]*\}", dedupe_labels, s, flags=re.DOTALL)

       # 10) If it *looks like* a sequence of JSON objects, wrap in [ ]
        stripped = s.strip()

        # Case: starts with { and contains multiple objects
        if stripped.startswith("{") and stripped.count("{") > 1:
            # Remove a leading '[' if the model partially added one (defensive)
            if stripped.startswith("["):
                stripped = stripped[1:]

            # Remove a trailing ']' if partially added
            if stripped.endswith("]"):
                stripped = stripped[:-1]

            s = "[" + stripped + "]"

        return s

    
    #3: Try parsing
    attempts = []

    # Try raw
    attempts.append(json_str)

    # Try once repaired
    attempts.append(repair(json_str))

    # Try double-repaired
    attempts.append(repair(repair(json_str)))

    last_error = None
    for i, candidate in enumerate(attempts):
        try:
            parsed = json.loads(candidate)

            # If the model returned a single object, wrap in a list
            if isinstance(parsed, dict):
                parsed = [parsed]

            return parsed

        except Exception as e:
            last_error = e
            # keep trying

    # 4: try to salvage
    objects = re.findall(r"\{.*?\}", attempts[-1], re.DOTALL)

    if not objects:
        raise ValueError("No JSON objects found during salvage attempt.")

    results = []

    for i, obj in enumerate(objects):
        try:
            fixed = repair(obj)
            parsed_obj = json.loads(fixed)
            results.append(parsed_obj)

        except Exception as e:
            print("JSON PARSE FAILURE")
            print(f"Object index: {i}")
            print(f"Error: {e}")
            print(f"Raw object:\n{obj}")
            raise

    #Enforce batch size
    if len(results) != BATCH_SIZE:
        raise ValueError(
            f"Expected {BATCH_SIZE} items, got {len(results)}"
        )

    return results