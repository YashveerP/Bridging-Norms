import asyncio

import pandas as pd
from utils.predictLabelUtils import predictViolation
from utils.dataSetUtils import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from defs import *

from prompts.ZeroShot import *
from prompts.OneShot import *
from prompts.ThreeShot import *
from prompts.SixShot import *
from prompts.prompts import predictViolationUserPrompt
from prompts.compareCommunities import *

DIRECTORY = "compareCommunities"

QWEN= model("qwen3:14b", "local")
LLAMA3 = model("meta-llama/llama-3.3-70b-instruct:free", "openrouter")
GPT_OSS = model("openai/gpt-oss-120b:free", "openrouter")
SAFEGUARD_LOCAL= model("gpt-oss-safeguard:20b", "local")

# The 5 communities with most comments
COMMUNITIES = ["t5_2xhvq", "t5_2w2s8", "t5_2qhw9", "t5_2qho4", "t5_3h47q"]

predictViolationPrompt = prompt(sixShotPrescriptiveAndRestrictive, predictViolationUserPrompt, pd.read_csv('datasets/tests.csv'))

def makeCompareCommunityPrompt(community):
    return prompt(compareCommmunitiesSysPrompt(community), compareCommunitesUserPrompt, pd.read_csv('datasets/tests.csv'))

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

# holds each experiment that will be run
# Give it a Model, Prompt, and Directory to store results
jobs = [
    (GPT_OSS, makeCompareCommunityPrompt(COMMUNITIES[0]), f"{DIRECTORY}/{COMMUNITIES[0]} "),
]

asyncio.run(run_experiments(jobs))
