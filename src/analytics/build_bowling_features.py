
from pathlib import Path
import pandas as pd
import numpy as np


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed" / "deliveries"
    / "player_enriched_deliveries.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "features"
OUTPUT_FILE = OUTPUT_DIR / "bowling_features.csv"


# ==================================================
# 2. LOAD DATASET
# ==================================================

print("=" * 65)
print("RR GEMHUNTERS - BOWLING FEATURE ENGINEERING")
print("=" * 65)

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("\nDelivery records:", len(df))


# ==================================================
# 3. VALIDATE REQUIRED COLUMNS
# ==================================================

required_columns = [
    "match_id",
    "innings",
    "bowler_id",
    "mapped_bowler_name",
    "comp_id",
    "comp_name",
    "match_format",
    "season",
    "bowler_runs_provisional",
    "legal_ball_provisional",
    "bowler_wicket_provisional",
    "has_quality_flag",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("Required column validation: PASS")


# ==================================================
# 4. PREPARE BOWLING METRICS
# ==================================================

df["bowler_runs"] = pd.to_numeric(
    df["bowler_runs_provisional"],
    errors="coerce"
).fillna(0)

df["legal_balls"] = (
    df["legal_ball_provisional"]
    .fillna(False)
    .astype(bool)
    .astype("int64")
)

df["bowler_wickets"] = pd.to_numeric(
    df["bowler_wicket_provisional"],
    errors="coerce"
).fillna(0)


# ==================================================
# 5. DEFINE GROUPING
# ==================================================

group_columns = [
    "bowler_id",
    "mapped_bowler_name",
    "comp_id",
    "comp_name",
    "match_format",
    "season",
]


# ==================================================
# 6. AGGREGATE BOWLING PERFORMANCE
# ==================================================

print("\nCalculating bowling statistics...")

bowling = (
    df.groupby(
        group_columns,
        dropna=False
    )
    .agg(
        runs_conceded=("bowler_runs", "sum"),
        legal_balls=("legal_balls", "sum"),
        wickets=("bowler_wickets", "sum"),
        matches=("match_id", "nunique"),
        quality_flagged_deliveries=(
            "has_quality_flag",
            "sum"
        ),
    )
    .reset_index()
)


# ==================================================
# 7. CALCULATE BOWLING INNINGS
# ==================================================

bowling_innings = (
    df[
        [
            "bowler_id",
            "comp_id",
            "comp_name",
            "match_format",
            "season",
            "match_id",
            "innings",
        ]
    ]
    .drop_duplicates()
    .groupby(
        [
            "bowler_id",
            "comp_id",
            "comp_name",
            "match_format",
            "season",
        ],
        dropna=False
    )
    .size()
    .reset_index(name="innings")
)

bowling = bowling.merge(
    bowling_innings,
    on=[
        "bowler_id",
        "comp_id",
        "comp_name",
        "match_format",
        "season",
    ],
    how="left",
    validate="one_to_one"
)


# ==================================================
# 8. CALCULATE DERIVED METRICS
# ==================================================

# Overs

bowling["overs"] = (
    bowling["legal_balls"] / 6
)


# Economy rate

bowling["economy_rate"] = np.where(
    bowling["legal_balls"] > 0,
    bowling["runs_conceded"] /
    (bowling["legal_balls"] / 6),
    np.nan
)


# Bowling average

bowling["bowling_average"] = np.where(
    bowling["wickets"] > 0,
    bowling["runs_conceded"] /
    bowling["wickets"],
    np.nan
)


# Bowling strike rate

bowling["bowling_strike_rate"] = np.where(
    bowling["wickets"] > 0,
    bowling["legal_balls"] /
    bowling["wickets"],
    np.nan
)


# ==================================================
# 9. VALIDATE FEATURE TABLE
# ==================================================

print("\nRunning validation...")

duplicate_count = bowling.duplicated(
    subset=group_columns
).sum()

if duplicate_count > 0:
    raise ValueError(
        "Duplicate bowling feature records found."
    )


source_runs = int(df["bowler_runs"].sum())
feature_runs = int(bowling["runs_conceded"].sum())

source_balls = int(df["legal_balls"].sum())
feature_balls = int(bowling["legal_balls"].sum())

source_wickets = int(df["bowler_wickets"].sum())
feature_wickets = int(bowling["wickets"].sum())

if source_runs != feature_runs:
    raise ValueError("Bowler runs reconciliation failed.")

if source_balls != feature_balls:
    raise ValueError("Legal balls reconciliation failed.")

if source_wickets != feature_wickets:
    raise ValueError("Bowler wickets reconciliation failed.")


# ==================================================
# 10. SORT AND SAVE
# ==================================================

bowling = bowling.sort_values(
    ["wickets", "economy_rate"],
    ascending=[False, True]
).reset_index(drop=True)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

bowling.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==================================================
# 11. FINAL REPORT
# ==================================================

print("\n" + "=" * 65)
print("BOWLING FEATURE ENGINEERING REPORT")
print("=" * 65)

print("Bowler competition-season records:", len(bowling))
print("Unique bowlers:", bowling["bowler_id"].nunique())

print("\nBOWLING TOTALS")

print("Runs conceded:", feature_runs)
print("Legal balls:", feature_balls)
print("Bowler wickets:", feature_wickets)
print("Bowling innings:", bowling["innings"].sum())

print("\nTOP 10 BOWLERS BY WICKETS")

print(
    bowling[
        [
            "mapped_bowler_name",
            "runs_conceded",
            "legal_balls",
            "wickets",
            "economy_rate",
            "bowling_average",
            "bowling_strike_rate",
        ]
    ].head(10).to_string(index=False)
)

print("\nVALIDATION RESULTS")

print("Duplicate feature records:", duplicate_count)
print("Bowler runs reconciliation: PASS")
print("Legal balls reconciliation: PASS")
print("Bowler wickets reconciliation: PASS")

print("\nSaved:", OUTPUT_FILE)
print("Final shape:", bowling.shape)

print("\nBOWLING FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
