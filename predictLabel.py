import pandas as pd
from utils.predictLabelUtils import predictViolation
from utils.dataSetUtils import generatePreparedDataSet, makeNewTrainTestSplit
from concurrent.futures import ThreadPoolExecutor, as_completed


PROMPT_TYPE = "ThreeShot"      # ← choose: "ZeroShot", "OneShot", "ThreeShot", "SixShot"
USE_COT = False
EXTRAINFO = "-Prescriptive"
PROMPT_NAME = PROMPT_TYPE + ("-COT" if USE_COT else "") + EXTRAINFO

LOCAL_MODEL="qwen2.5:7b-instruct"
LLAMA3 = "meta-llama/llama-3.3-70b-instruct:free"
GPT_OSS = "openai/gpt-oss-120b:free"

EXPERIMENTS = [
    {
        "runner": "local",
        "model": LOCAL_MODEL,
        "prompt_type": PROMPT_TYPE,
        "use_cot": USE_COT,
        "extra_info": EXTRAINFO
    },
    {
        "runner": "openrouter",
        "model": LLAMA3,
        "prompt_type": PROMPT_TYPE,
        "use_cot": USE_COT,
        "extra_info": EXTRAINFO
    },
    {
        "runner": "openrouter",
        "model": GPT_OSS,
        "prompt_type": PROMPT_TYPE,
        "use_cot": USE_COT,
        "extra_info": EXTRAINFO
    }
]

def run_experiment(exp):
    print(f"Starting: {exp['model']}")
    return predictViolation(
        exp["runner"],
        exp["model"],
        exp["prompt_type"],
        exp["use_cot"],
        exp["extra_info"]
    )

with ThreadPoolExecutor(max_workers=3) as executor:
    executor.map(run_experiment, EXPERIMENTS)

