
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed" / "deliveries"
    / "player_enriched_deliveries.csv"
)

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("=" * 60)
print("BOWLING METRIC AUDIT")
print("=" * 60)

# 1. Bowler runs
print("\n1. BOWLER RUNS")
print(df["bowler_runs_provisional"].sum())

# 2. Legal balls
print("\n2. LEGAL BALLS")
print(df["legal_ball_provisional"].value_counts())

print(
    "Total provisional legal balls:",
    df["legal_ball_provisional"].sum()
)

# 3. Bowler wickets
print("\n3. BOWLER WICKETS")
print(df["bowler_wicket_provisional"].sum())

# 4. Dismissal types
print("\n4. DISMISSAL TYPES")
print(
    df[df["is_wicket"] == True]
    ["dismissal_type"]
    .value_counts(dropna=False)
)

# 5. Extras
print("\n5. EXTRAS DISTRIBUTION")
print(
    df["extras_type"]
    .fillna("none")
    .value_counts()
)

# 6. Quality flags
print("\n6. QUALITY FLAGGED DELIVERIES")
print(df["has_quality_flag"].sum())

print("\nBOWLING AUDIT COMPLETED")
