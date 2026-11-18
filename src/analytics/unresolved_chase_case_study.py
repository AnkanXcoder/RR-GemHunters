
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data"
    / "interim"
    / "chase_completion_detail.csv"
)

OUTPUT_DIR = ROOT / "data" / "interim"

df = pd.read_csv(INPUT, low_memory=False)

unresolved = df[
    df["chase_ending_category"].isin([
        "INCOMPLETE_OR_UNRESOLVED",
        "REVISED_TARGET_REQUIRES_VALIDATION",
        "NO_RESULT_OR_ABANDONED",
    ])
].copy()

columns = [
    "match_id",
    "match_date",
    "home_team",
    "away_team",
    "match_format",
    "season",
    "first_innings_innings_runs",
    "second_innings_innings_runs",
    "first_innings_legal_balls",
    "second_innings_legal_balls",
    "second_innings_wicket_records",
    "standard_chase_target",
    "revised_target",
    "result",
    "chase_ending_category",
    "manual_review_reason",
]

unresolved = unresolved[columns]

output = OUTPUT_DIR / "unresolved_chase_case_study.csv"

unresolved.to_csv(output, index=False)

print("=" * 60)
print("UNRESOLVED CHASE CASE STUDY")
print("=" * 60)

print(f"\nTotal matches selected: {len(unresolved)}")

print("\nClassification distribution:")
print(
    unresolved["chase_ending_category"]
    .value_counts()
    .to_string()
)

print("\nManual review reason distribution:")
print(
    unresolved["manual_review_reason"]
    .value_counts()
    .head(20)
    .to_string()
)

print("\nSample records:")
print(
    unresolved.head(15).to_string(index=False)
)

print(f"\nSaved to: {output}")
