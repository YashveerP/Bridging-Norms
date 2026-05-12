import os
import json
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.join("results", "compareCommunities")

MODEL = "openai/gpt-oss-120b:free"
safe_model = re.sub(r'[<>:"/\\|?*]', '_', MODEL)

# ============================================================
# LOAD DATA
# ============================================================

communities = sorted(os.listdir(BASE_DIR))

if "t5_2qnkr" in communities:
    communities.remove("t5_2qnkr")

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
# TWO-WAY ANOVA
# ============================================================

model = smf.ols(
    "accuracy ~ C(prompt_community) + C(evaluation_community)",
    data=df
).fit()

anova_table = anova_lm(model, typ=2)

print("\n==============================")
print("TWO-WAY ANOVA")
print("==============================\n")

print(anova_table)

# ============================================================
# EFFECT SIZES (ETA SQUARED)
# ============================================================

ss_total = anova_table["sum_sq"].sum()

anova_table["eta_sq"] = anova_table["sum_sq"] / ss_total

print("\n==============================")
print("EFFECT SIZES (ETA SQUARED)")
print("==============================\n")

print(anova_table[["sum_sq", "F", "PR(>F)", "eta_sq"]])

# ============================================================
# CLEAN SUMMARY
# ============================================================

prompt_eta = anova_table.loc["C(prompt_community)", "eta_sq"]
prompt_p = anova_table.loc["C(prompt_community)", "PR(>F)"]

eval_eta = anova_table.loc["C(evaluation_community)", "eta_sq"]
eval_p = anova_table.loc["C(evaluation_community)", "PR(>F)"]

print("\n==============================")
print("SUMMARY")
print("==============================\n")

print(f"Prompt community effect:")
print(f"  η² = {prompt_eta:.4f}")
print(f"  p  = {prompt_p:.6g}")

print()

print(f"Evaluation community effect:")
print(f"  η² = {eval_eta:.4f}")
print(f"  p  = {eval_p:.6g}")