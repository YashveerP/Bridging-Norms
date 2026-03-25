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
    BATCH_SIZE= 20

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
    def repair(s: str):
        repairs_applied = []

        def apply(pattern, repl, text, desc, flags=0):
            new_text, count = re.subn(pattern, repl, text, flags=flags)
            if count > 0:
                repairs_applied.append(f"{desc} (x{count})")
            return new_text
        
        # 0) Fix unquoted comment_id
        # Fix unquoted comment_id (numbers → strings)
        # s = apply(
        #     r'"comment_id"\s*:\s*([0-9]+)',
        #     r'"comment_id": "\1"',
        #     s,
        #     "fixed unquoted comment_id"
        # )

        # # 1) Fix unquoted labels
        # s = apply(
        #     r'"label"\s*:\s*(violation|non_violation)',
        #     r'"label": "\1"',
        #     s,
        #     "fixed unquoted label"
        # )

        # # 2) SINGLE-quoted evidence
        # def single_quote_repl(m):
        #     return '"evidence": ' + json.dumps(m.group(1))

        # s = apply(
        #     r'"evidence"\s*:\s*\'(.*?)\'',
        #     single_quote_repl,
        #     s,
        #     "fixed single-quoted evidence",
        #     flags=re.DOTALL
        # )

        # # 3) TRIPLE-quoted evidence
        # def triple_quote_repl(m):
        #     return '"evidence": ' + json.dumps(m.group(1))

        # s = apply(
        #     r'"evidence"\s*:\s*"""(.*?)"""',
        #     triple_quote_repl,
        #     s,
        #     "fixed triple-quoted evidence",
        #     flags=re.DOTALL
        # )

        # # 4) trailing commas
        # s = apply(r",\s*([}\]])", r"\1", s, "removed trailing commas")

        # # 5) smart quotes (manual check)
        # before = s
        # s = s.replace("“", '"').replace("”", '"').replace("’", "'")
        # if s != before:
        #     repairs_applied.append("normalized smart quotes")

        # # 6–7) structural fixes
        # s = apply(r"\[\s*\{\s*\{", "[{", s, "fixed double opening braces")
        # s = apply(r"\}\s*\}\s*\]", "}]", s, "fixed double closing braces")
        # s = apply(r"\]\s*\]\s*$", "]", s, "fixed extra closing bracket")

        # # 9) dedupe labels
        # def dedupe_labels(match):
        #     block = match.group(0)
        #     labels = re.findall(r'"label"\s*:\s*"([^"]+)"', block)
        #     if labels:
        #         final_label = labels[-1]
        #         block = re.sub(r'"label"\s*:\s*"[^"]+",?', "", block)
        #         block = block.replace("{", f'{{\n  "label": "{final_label}",', 1)
        #         repairs_applied.append("deduplicated labels")
        #     return block

        # s = re.sub(r"\{[^{}]*\}", dedupe_labels, s, flags=re.DOTALL)

        # # 10) wrap multiple objects
        # stripped = s.strip()
        # if stripped.startswith("{") and stripped.count("{") > 1:
        #     s = "[" + stripped.strip("[]") + "]"
        #     repairs_applied.append("wrapped multiple JSON objects in array")

        return s, repairs_applied

    
    #3: Try parsing
    attempts = []

    # Attempt 0: raw
    attempts.append((json_str, ["raw input"]))

    # Attempt 1: single repair
    s1, r1 = repair(json_str)
    attempts.append((s1, r1))

    log_lines = []

    for i, (attempt_str, repairs) in enumerate(attempts):
        try:
            parsed = json.loads(attempt_str)

            # normalize to list
            if isinstance(parsed, dict):
                parsed = [parsed]

            if len(parsed) != BATCH_SIZE:
                log_lines.append(
                    f"Attempt {i}: parsed but wrong size "
                    f"({len(parsed)} != {BATCH_SIZE}) "
                    f"Attempted Repairs: {repairs}"
                )
                continue

            log_lines.append(f"Attempt {i}: SUCCESS")
            log_lines.append(f"Repairs: {repairs}")
            
            with open(f"{path}/repair_log.txt", "a") as f:
                f.write("\n".join(log_lines) + "\n")

            return parsed

        except json.JSONDecodeError as e:
            log_lines.append(f"Attempt {i}: JSON decode error: {str(e)}")

        except Exception as e:
            log_lines.append(f"Attempt {i}: unexpected error: {str(e)}")
            raise  # don't hide real bugs
    # write log
    with open(f"{path}/repair_log.txt", "a") as f:
        f.write("\n".join(log_lines) + "\n")

   # 4: try to salvage
    final_str = attempts[-1][0]  # extract string only

    objects = re.findall(r"\{.*?\}", final_str, re.DOTALL)

    if not objects:
        raise ValueError("No JSON objects found during salvage attempt.")

    results = []
    failed_indices = []

    for i, obj in enumerate(objects):
        try:
            fixed_str, repairs = repair(obj)
            parsed_obj = json.loads(fixed_str)
            results.append(parsed_obj)

        except json.JSONDecodeError as e:
            failed_indices.append(i)
            log_lines.append(f"Salvage: failed object {i} (JSON error: {str(e)})")

        except Exception as e:
            failed_indices.append(i)
            log_lines.append(f"Salvage: failed object {i} (unexpected: {str(e)})")

    if len(results) < BATCH_SIZE:
        log_lines.append(
            f"Salvage partial success: {len(results)}/{BATCH_SIZE} objects parsed"
        )


    return results