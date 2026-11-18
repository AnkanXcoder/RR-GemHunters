
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
    / "overs_completion_crosscheck.csv"
)

df = pd.read_csv(INPUT, low_memory=False)

# Select matches that were previously unresolved
# or had revised targets requiring validation.
df = df[
    df["chase_ending_category"].isin([
        "INCOMPLETE_OR_UNRESOLVED",
        "REVISED_TARGET_REQUIRES_VALIDATION",
    ])
].copy()

# Identify standard 120-ball innings.
df["has_120_legal_balls"] = (
    df["second_innings_legal_balls"] == 120
)

df["has_revised_target"] = (
    df["revised_target"].notna()
)

df["has_dls_result"] = (
    df["result"].fillna("")
    .astype(str)
    .str.contains(
        r"\bdls\b|duckworth|lewis",
        case=False,
        regex=True
    )
)

df["is_all_out"] = (
    df["second_innings_wicket_records"] >= 10
)

df["target_not_reached"] = (
    df["second_innings_innings_runs"]
    < df["standard_chase_target"]
)

# Focus on 120-ball matches.
completed = df[
    df["has_120_legal_balls"]
].copy()

# Print independent evidence counts.
checks = {
    "120_legal_ball_matches": len(completed),

    "with_revised_target":
        completed["has_revised_target"].sum(),

    "with_DLS_result":
        completed["has_dls_result"].sum(),

    "all_out":
        completed["is_all_out"].sum(),

    "below_standard_target":
        completed["target_not_reached"].sum(),

    "without_revised_target_and_DLS":
        (
            ~completed["has_revised_target"]
            & ~completed["has_dls_result"]
        ).sum(),
}

print("=" * 65)
print("OVERS COMPLETION CROSS-CHECK")
print("=" * 65)

print("\nIndependent evidence counts:")

for key, value in checks.items():
    print(f"{key}: {value}")

# Candidate defended innings.
candidates = completed[
    ~completed["has_revised_target"]
    & ~completed["has_dls_result"]
    & ~completed["is_all_out"]
    & completed["target_not_reached"]
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
    "revised_target",
    "result",
]

print("\nPotential completed-overs defended matches:")
print(f"Total candidates: {len(candidates)}")

print(
    candidates[columns]
    .head(20)
    .to_string(index=False)
)

# Save the full candidate list.
candidates[columns].to_csv(
    OUTPUT,
    index=False
)

print(f"\nSaved to: {OUTPUT}")
