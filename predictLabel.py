import pandas as pd
from utils.predictLabelUtils import predictViolation



PROMPT_TYPE = "ZeroShot"      # ← choose: "ZeroShot", "OneShot", "FewShot"
USE_COT = False
EXTRAINFO = ""
PROMPT_NAME = PROMPT_TYPE + ("-COT" if USE_COT else "") + EXTRAINFO
RUNNER = "local"         # ← choose: "local" or "openrouter"

LOCAL_MODEL="qwen2.5:7b-instruct"
LLAMA3 = "meta-llama/llama-3.3-70b-instruct:free"
GPT_OSS = "openai/gpt-oss-120b:free"

MODEL = LOCAL_MODEL

predictViolation("local", LOCAL_MODEL, PROMPT_TYPE, USE_COT, EXTRAINFO)
# predictViolation("openrouter", LLAMA3, PROMPT_TYPE, USE_COT, EXTRAINFO)
# predictViolation("openrouter", GPT_OSS, PROMPT_TYPE, USE_COT, EXTRAINFO)


