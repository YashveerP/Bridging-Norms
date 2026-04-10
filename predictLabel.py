import asyncio

import pandas as pd
from utils.predictLabelUtils import predictViolation
from utils.dataSetUtils import generatePreparedDataSet, makeNewTrainTestSplit
from concurrent.futures import ThreadPoolExecutor, as_completed
from defs import *

from prompts.ZeroShot import *
from prompts.OneShot import *
from prompts.ThreeShot import *
from prompts.SixShot import *
from prompts.prompts import buildMessages


PROMPT = prompt(sixShotPrescriptiveAndRestrictive, buildMessages)
DIRECTORY = "sixShotPrescriptiveAndRestrictive_DEEZ"

QWEN= model("qwen3:14b", "local")
LLAMA3 = model("meta-llama/llama-3.3-70b-instruct:free", "openrouter")
GPT_OSS = model("openai/gpt-oss-120b:free", "openrouter")
SAFEGUARD_LOCAL= model("gpt-oss-safeguard:20b", "local")

async def run_experiments(jobs):
    local_jobs = []
    api_jobs = []

    # split jobs
    for model, prompt, directory in jobs:
        if model.runner == "local":
            local_jobs.append((model, prompt, directory))
        else:
            api_jobs.append((model, prompt, directory))

    #run local sequentially
    for model, prompt, directory in local_jobs:
        print(f"Running LOCAL model: {model.name}")
        await predictViolation(model, prompt, directory)

    # run API concurrently
    if api_jobs:
        print(f"Running {len(api_jobs)} API models concurrently")

        await asyncio.gather(*[
            predictViolation(model, prompt, directory)
            for model, prompt, directory in api_jobs
        ])

jobs = [
    (QWEN, PROMPT, DIRECTORY),
    (LLAMA3, PROMPT, DIRECTORY),
    (GPT_OSS, PROMPT, DIRECTORY+"2"),
]

asyncio.run(run_experiments(jobs))