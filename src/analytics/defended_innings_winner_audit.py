
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_FILE = (
    ROOT / "data" / "interim"
    / "overs_completion_crosscheck.csv"
)

DELIVERY_FILE = (
    ROOT / "data" / "processed" / "deliveries"
    / "regulation_deliveries.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "interim"
    / "defended_innings_winner_audit.csv"
)

candidates = pd.read_csv(
    CANDIDATE_FILE,
    low_memory=False
)

deliveries = pd.read_csv(
    DELIVERY_FILE,
    low_memory=False
)

print("=" * 65)
print("DEFENDED INNINGS WINNER AUDIT")
print("=" * 65)

# ------------------------------------------------------------
# 1. Extract second-innings team information
# ------------------------------------------------------------

second_innings = deliveries[
    deliveries["innings"] == 2
].copy()

team_metrics = (
    second_innings
    .groupby("match_id")
    .agg(
        batting_team_id=("batting_team_id", "first"),
        bowling_team_id=("bowling_team_id", "first"),
        batting_team_count=("batting_team_id", "nunique"),
        bowling_team_count=("bowling_team_id", "nunique"),
        winner_team_id=("win_team_id", "first"),
        winner_team_count=("win_team_id", "nunique"),
    )
    .reset_index()
)

# ------------------------------------------------------------
# 2. Join with the 217 candidate matches
# ------------------------------------------------------------

audit = candidates.merge(
    team_metrics,
    on="match_id",
    how="left",
    validate="one_to_one"
)

# ------------------------------------------------------------
# 3. Validate team consistency
# ------------------------------------------------------------

audit["team_data_consistent"] = (
    audit["batting_team_count"].eq(1)
    & audit["bowling_team_count"].eq(1)
    & audit["winner_team_count"].le(1)
)

audit["winner_is_second_innings_bowling_team"] = (
    audit["winner_team_id"].notna()
    & audit["bowling_team_id"].notna()
    & (
        audit["winner_team_id"]
        == audit["bowling_team_id"]
    )
)

audit["winner_is_second_innings_batting_team"] = (
    audit["winner_team_id"].notna()
    & audit["batting_team_id"].notna()
    & (
        audit["winner_team_id"]
        == audit["batting_team_id"]
    )
)

# ------------------------------------------------------------
# 4. Assign validation status
# ------------------------------------------------------------

audit["winner_validation_status"] = "REQUIRES_REVIEW"

audit.loc[
    audit["team_data_consistent"]
    & audit["winner_is_second_innings_bowling_team"],
    "winner_validation_status"
] = "DEFENDING_TEAM_WON"

audit.loc[
    audit["team_data_consistent"]
    & audit["winner_is_second_innings_batting_team"],
    "winner_validation_status"
] = "CHASING_TEAM_WON"

audit.loc[
    ~audit["team_data_consistent"],
    "winner_validation_status"
] = "INCONSISTENT_TEAM_DATA"

# ------------------------------------------------------------
# 5. Summary
# ------------------------------------------------------------

print(f"\nCandidate matches: {len(audit)}")

print("\nWinner validation distribution:")
print(
    audit["winner_validation_status"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nTeam consistency:")
print(
    audit["team_data_consistent"]
    .value_counts(dropna=False)
    .to_string()
)

# ------------------------------------------------------------
# 6. Show sample records
# ------------------------------------------------------------

columns = [
    "match_id",
    "home_team",
    "away_team",
    "first_innings_innings_runs",
    "second_innings_innings_runs",
    "result",
    "batting_team_id",
    "bowling_team_id",
    "winner_team_id",
    "winner_validation_status",
]

print("\nSample validation records:")
print(
    audit[columns]
    .head(15)
    .to_string(index=False)
)

# ------------------------------------------------------------
# 7. Save
# ------------------------------------------------------------

audit.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nSaved to: {OUTPUT_FILE}")
print("\nWinner audit completed.")
