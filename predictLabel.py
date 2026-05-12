import asyncio
import re

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

DIRECTORY = "compareCommunities10"

QWEN= model("qwen3:14b", "local")
LLAMA3 = model("meta-llama/llama-3.3-70b-instruct:free", "openrouter")
GPT_OSS = model("openai/gpt-oss-120b:free", "openrouter")
GPT_4O_MINI= model("openai/gpt-4o-mini", "openrouter")

MODEL_TO_RUN = GPT_4O_MINI

# The communities with >= 106 comments(6 for training, 100 for testing)
COMMUNITIES = getSubreddits()
COMMUNITIES.remove("t5_2qnkr")



predictViolationPrompt = prompt(sixShotPrescriptiveAndRestrictive, predictViolationUserPrompt, pd.read_csv('datasets/tests.csv'))

communityComments = {}
sysPrompts = {}
def makeCompareCommunityPrompt(communityA, communityB):
    # cache communityA sysPrompt
    if communityA not in sysPrompts:
        sysPrompts[communityA] = compareCommmunitiesSysPrompt(communityA)
    # cache community B comments
    if communityB not in communityComments:
        communityComments[communityB] = getCommunityTests(communityB)
    return prompt(sysPrompts[communityA], compareCommunitesUserPrompt, communityComments[communityB])

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
    
]

safe_model = re.sub(r'[<>:"/\\|?*]', '_', MODEL_TO_RUN.name)
# code will be iterating by evaluated communities then the prompt communities
for i in range(0, len(COMMUNITIES)):
    A = COMMUNITIES[i]
    for j in range(0, len(COMMUNITIES)):
        B = COMMUNITIES[j]
        if (not os.path.exists(f"results/{DIRECTORY}/{B}/{A}/{safe_model}/metrics.json")):
            jobs.append((MODEL_TO_RUN, makeCompareCommunityPrompt(B, A), f"{DIRECTORY}/{B}/{A}"))
            if (len(jobs) == 8):
                asyncio.run(run_experiments(jobs))
                jobs = []
asyncio.run(run_experiments(jobs))