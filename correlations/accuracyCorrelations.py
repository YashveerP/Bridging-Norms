import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.join("results", "compareCommunities") # change if needed

LLAMA3 = "meta-llama/llama-3.3-70b-instruct:free"
GPT_OSS = "openai/gpt-oss-120b:free"
GPT_4O_MINI= "openai/gpt-4o-mini"
MODEL = GPT_OSS
safe_model = re.sub(r'[<>:"/\\|?*]', '_', MODEL)

# collect all communities
communities = sorted(os.listdir(BASE_DIR))
communities.remove("t5_2qnkr")
df = pd.read_csv('datasets/subreddits-descriptions.csv')

rows = []

# collect data into flat list
for prompt_comm in communities:
    for eval_comm in communities:
        metricsPath = os.path.join(safe_model, "metrics.json")
        path = os.path.join(BASE_DIR, prompt_comm, eval_comm, metricsPath)

        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                acc = data.get("accuracy", np.nan)
        else:
            acc = np.nan
        promptCommName = df[df["name"] == prompt_comm].iloc[0]["display_name_prefixed"]
        evalCommName = df[df["name"] == eval_comm].iloc[0]["display_name_prefixed"]
        rows.append({
            "prompt_community": f"{promptCommName} ({prompt_comm})",
            "eval_community": f"{evalCommName} ({eval_comm})",
            "accuracy": acc
        })

# convert to DataFrame
df = pd.DataFrame(rows)

# sort by accuracy (descending)
df_sorted = df.sort_values(by="accuracy", ascending=False)

# optional: drop NaNs if you don’t want missing entries
df_sorted = df_sorted.dropna(subset=["accuracy"])

print(df_sorted.head(20))