
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data"
    / "interim"
    / "chase_completion_detail.csv"
)

OUTPUT = (
    ROOT
    / "data"
    / "interim"
    / "overs_completion_audit.csv"
)

df = pd.read_csv(INPUT, low_memory=False)

# Focus on matches previously marked unresolved.
unresolved = df[
    df["chase_ending_category"].isin([
        "INCOMPLETE_OR_UNRESOLVED",
        "REVISED_TARGET_REQUIRES_VALIDATION",
    ])
].copy()

# Identify matches with a full standard 20-over innings.
unresolved["second_innings_120_balls"] = (
    unresolved["second_innings_legal_balls"] == 120
)

unresolved["second_innings_under_120"] = (
    unresolved["second_innings_legal_balls"] < 120
)

print("=" * 65)
print("OVERS COMPLETION AUDIT")
print("=" * 65)

print(f"\nPreviously unresolved matches: {len(unresolved)}")

print("\nSecond innings legal-ball distribution:")
print(
    unresolved["second_innings_legal_balls"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)

print("\n120 legal-ball matches:")
print(
    unresolved["second_innings_120_balls"]
    .value_counts()
    .to_string()
)

print("\nUnder 120 legal-ball matches:")
print(
    unresolved["second_innings_under_120"]
    .value_counts()
    .to_string()
)

# Show completed standard-overs cases.
completed = unresolved[
    unresolved["second_innings_120_balls"]
].copy()

columns = [
    "match_id",
    "match_date",
    "home_team",
    "away_team",
    "match_format",
    "first_innings_innings_runs",
    "second_innings_innings_runs",
    "second_innings_legal_balls",
    "second_innings_wicket_records",
    "standard_chase_target",
    "result",
    "chase_ending_category",
]

print("\nSample 120-ball matches:")
print(
    completed[columns]
    .head(20)
    .to_string(index=False)
)

completed[columns].to_csv(
    OUTPUT,
    index=False
)

print(f"\nSaved completed-overs cases to: {OUTPUT}")
