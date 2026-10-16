"""
eFootball Performance Detective — Milestone 1
Synthetic data only. These rows are for learning and are NOT real match results.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create a tiny synthetic dataset so we can practise the analysis.
# Replace these example rows with real match records later.
sample_matches = [
    {"match": 1, "fps": 20, "goals_scored": 2, "goals_conceded": 1,
     "passes_completed": 120, "passes_attempted": 145},
    {"match": 2, "fps": 20, "goals_scored": 1, "goals_conceded": 2,
     "passes_completed": 105, "passes_attempted": 138},
    {"match": 3, "fps": 20, "goals_scored": 3, "goals_conceded": 0,
     "passes_completed": 132, "passes_attempted": 150},
    {"match": 4, "fps": 30, "goals_scored": 0, "goals_conceded": 2,
     "passes_completed": 94, "passes_attempted": 130},
    {"match": 5, "fps": 30, "goals_scored": 1, "goals_conceded": 1,
     "passes_completed": 110, "passes_attempted": 140},
    {"match": 6, "fps": 30, "goals_scored": 2, "goals_conceded": 3,
     "passes_completed": 101, "passes_attempted": 137},
]

df = pd.DataFrame(sample_matches)

# 2. Calculate derived values.
df["goal_difference"] = df["goals_scored"] - df["goals_conceded"]
df["passing_accuracy_pct"] = (
    df["passes_completed"] / df["passes_attempted"] * 100
).round(1)

# 3. Label each result based on goals scored and conceded.
df["result"] = "Draw"
df.loc[df["goals_scored"] > df["goals_conceded"], "result"] = "Win"
df.loc[df["goals_scored"] < df["goals_conceded"], "result"] = "Loss"

print("\n=== SYNTHETIC MATCH DATA — NOT REAL RESULTS ===")
print(df.to_string(index=False))

print("\n=== SUMMARY BY FPS ===")
summary = df.groupby("fps").agg(
    matches=("match", "count"),
    avg_goals_scored=("goals_scored", "mean"),
    avg_goals_conceded=("goals_conceded", "mean"),
    avg_passing_accuracy_pct=("passing_accuracy_pct", "mean"),
    wins=("result", lambda values: (values == "Win").sum()),
    draws=("result", lambda values: (values == "Draw").sum()),
    losses=("result", lambda values: (values == "Loss").sum()),
).round(2)
print(summary)

# 4. Save the practice dataset so we can inspect it separately.
os.makedirs("data", exist_ok=True)
df.to_csv("data/matches_sample.csv", index=False)
print("\nSaved practice data to data/matches_sample.csv")

# 5. Plot goals by match. This is a practice visualisation.
sns.set_theme(style="whitegrid")
plt.figure(figsize=(9, 5))
sns.lineplot(data=df, x="match", y="goals_scored", marker="o", label="Goals scored")
sns.lineplot(data=df, x="match", y="goals_conceded", marker="o", label="Goals conceded")
plt.title("Practice data: goals by match (synthetic)")
plt.xlabel("Match number")
plt.ylabel("Goals")
plt.tight_layout()
plt.savefig("goals_by_match_sample.png", dpi=150)
plt.show()

print("\nReminder: this sample is only for learning. It cannot establish what happened in your actual matches.")
