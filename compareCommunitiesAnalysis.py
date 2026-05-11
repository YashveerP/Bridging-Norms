import os
import json
import re
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.join("results", "compareCommunities")

MODEL = "openai/gpt-oss-120b:free"
safe_model = re.sub(r'[<>:"/\\|?*]', '_', MODEL)

# ============================================================
# LOAD COMMUNITIES
# ============================================================

communities = sorted(os.listdir(BASE_DIR))

if "t5_2qnkr" in communities:
    communities.remove("t5_2qnkr")

# ============================================================
# BUILD DATAFRAME
# ============================================================

rows = []

for prompt_comm in communities:
    for eval_comm in communities:

        path = os.path.join(
            BASE_DIR,
            prompt_comm,
            eval_comm,
            safe_model,
            "metrics.json"
        )

        if not os.path.exists(path):
            continue

        with open(path, "r") as f:
            data = json.load(f)

        acc = data.get("accuracy", np.nan)

        if acc is None or np.isnan(acc):
            continue

        rows.append({
            "prompt_community": prompt_comm,
            "evaluation_community": eval_comm,
            "accuracy": float(acc)
        })

df = pd.DataFrame(rows)

# ============================================================
# CLEANING
# ============================================================

df = df.dropna(subset=["accuracy"])

df["prompt_community"] = df["prompt_community"].astype("category")
df["evaluation_community"] = df["evaluation_community"].astype("category")

# ============================================================
# 1. BASELINE MODEL (evaluation only)
# ============================================================

model_eval = smf.ols(
    "accuracy ~ C(evaluation_community)",
    data=df
).fit()

r2_eval = model_eval.rsquared

# ============================================================
# 2. PROMPT MODEL (adds prompt)
# ============================================================

model_prompt = smf.ols(
    "accuracy ~ C(evaluation_community) + C(prompt_community)",
    data=df
).fit()

r2_prompt = model_prompt.rsquared

# ============================================================
# 3. BASELINE MODEL (prompt only)
# ============================================================

model_prompt_only = smf.ols(
    "accuracy ~ C(prompt_community)",
    data=df
).fit()

r2_prompt_only = model_prompt_only.rsquared

# ============================================================
# 4. FULL MODEL (both)
# ============================================================

model_full = smf.ols(
    "accuracy ~ C(prompt_community) + C(evaluation_community)",
    data=df
).fit()

r2_full = model_full.rsquared

# ============================================================
# SINGLE-NUMBER EFFECTS (THE IMPORTANT PART)
# ============================================================

prompt_effect = r2_full - r2_eval
eval_effect = r2_full - r2_prompt_only

print("\n==============================")
print("VARIANCE EXPLAINED SUMMARY")
print("==============================")

print(f"R² (evaluation only): {r2_eval:.4f}")
print(f"R² (prompt only):     {r2_prompt_only:.4f}")
print(f"R² (full model):      {r2_full:.4f}")

print("\n==============================")
print("KEY RESULTS (SINGLE NUMBERS)")
print("==============================")

print(f"Prompt community effect (ΔR²): {prompt_effect:.4f}")
print(f"Evaluation community effect (ΔR²): {eval_effect:.4f}")

# ============================================================
# OPTIONAL: PERCENT VERSION (for papers)
# ============================================================

print("\n==============================")
print("PERCENT VARIANCE EXPLAINED")
print("==============================")

print(f"Prompt community explains: {prompt_effect * 100:.2f}% of variance")
print(f"Evaluation community explains: {eval_effect * 100:.2f}% of variance")


