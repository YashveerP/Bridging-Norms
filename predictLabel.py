import pandas as pd
from utils.predictLabelUtils import predictViolation
from utils.dataSetUtils import generatePreparedDataSet, makeNewTrainTestSplit
from concurrent.futures import ThreadPoolExecutor, as_completed


PROMPT_TYPE = "ThreeShot"      # ← choose: "ZeroShot", "OneShot", "ThreeShot", "SixShot"
USE_COT = False
EXTRAINFO = "-Restrictive-2.1"
PROMPT_NAME = PROMPT_TYPE + ("-COT" if USE_COT else "") + EXTRAINFO

QWEN="qwen3:14b"
LLAMA3 = "meta-llama/llama-3.3-70b-instruct:free"
GPT_OSS = "openai/gpt-oss-120b:free"


predictViolation("local", QWEN, PROMPT_TYPE, USE_COT, EXTRAINFO)
# predictViolationNOID("local", QWEN, PROMPT_TYPE, USE_COT, EXTRAINFO)

