import pandas as pd
from utils.predictLabelUtils import predictViolation
from utils.dataSetUtils import generatePreparedDataSet, makeNewTrainTestSplit
from concurrent.futures import ThreadPoolExecutor, as_completed
from defs import *

from prompts.ZeroShot import *
from prompts.OneShot import *
from prompts.ThreeShot import *
from prompts.SixShot import *


PROMPT = prompt(zeroShot)
DIRECTORY = "ZeroShot_DEEZ"

QWEN= model("qwen3:14b", "local")
LLAMA3 = model("meta-llama/llama-3.3-70b-instruct:free", "openrouter")
GPT_OSS = model("openai/gpt-oss-120b:free", "openrouter")
SAFEGUARD_OR = model("openai/gpt-oss-safeguard-20b", "openrouter")
SAFEGUARD_LOCAL= model("gpt-oss-safeguard:20b", "local")

predictViolation(QWEN, PROMPT, DIRECTORY)
# predictViolation(SAFEGUARD_LOCAL, PROMPT, DIRECTORY)
# predictViolation(SAFEGUARD_OR, PROMPT, DIRECTORY)

