
from pathlib import Path
import pandas as pd
import numpy as np


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "deliveries"
    / "player_enriched_deliveries.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "features"
)

OUTPUT_FILE = OUTPUT_DIR / "batting_features.csv"


# ==================================================
# 2. LOAD DATASET
# ==================================================

print("=" * 65)
print("RR GEMHUNTERS - BATTING FEATURE ENGINEERING")
print("=" * 65)

print("\nLoading delivery dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Total delivery records:", len(df))
print("Total columns:", len(df.columns))


# ==================================================
# 3. VALIDATE REQUIRED COLUMNS
# ==================================================

required_columns = [
    "match_id",
    "innings",
    "striker_id",
    "mapped_batter_name",
    "comp_id",
    "comp_name",
    "match_format",
    "season",
    "runs",
    "is_four",
    "is_six",
    "is_wicket",
    "dismissed_player_id",
    "extras_type",
    "has_quality_flag",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}"
    )

print("\nRequired column validation: PASS")


# ==================================================
# 4. PREPARE BATTING METRICS
# ==================================================

# Standard provisional convention:
# Exclude wides from balls faced.
# Include no-balls provisionally.

extras = df["extras_type"].fillna("none").str.strip().str.lower()

df["balls_faced"] = (
    extras != "wide"
).astype("int64")


# Batter runs
df["batter_runs"] = pd.to_numeric(
    df["runs"],
    errors="coerce"
).fillna(0)


# Fours and sixes
df["fours"] = (
    df["is_four"]
    .fillna(False)
    .astype(bool)
    .astype("int64")
)

df["sixes"] = (
    df["is_six"]
    .fillna(False)
    .astype(bool)
    .astype("int64")
)


# Actual batter dismissals:
# Credit the player identified by dismissed_player_id.
# Do not automatically credit the striker.

df["batter_dismissal"] = (
    df["is_wicket"].fillna(False).astype(bool)
    & df["dismissed_player_id"].notna()
    & (
        df["dismissed_player_id"]
        == df["striker_id"]
    )
).astype("int64")


# ==================================================
# 5. DEFINE PLAYER-COMPETITION-SEASON GROUPING
# ==================================================

group_columns = [
    "striker_id",
    "mapped_batter_name",
    "comp_id",
    "comp_name",
    "match_format",
    "season",
]


# ==================================================
# 6. AGGREGATE BATTING PERFORMANCE
# ==================================================

print("\nCalculating batting statistics...")

batting = (
    df.groupby(
        group_columns,
        dropna=False
    )
    .agg(
        runs=("batter_runs", "sum"),
        balls_faced=("balls_faced", "sum"),
        fours=("fours", "sum"),
        sixes=("sixes", "sum"),
        dismissals=("batter_dismissal", "sum"),
        matches=("match_id", "nunique"),
        quality_flagged_deliveries=(
            "has_quality_flag",
            "sum"
        ),
    )
    .reset_index()
)


# ==================================================
# 7. CALCULATE ACTUAL BATTING INNINGS
# ==================================================

print("Calculating actual batting innings...")

# A batting innings is one unique combination of:
# player + match + innings + competition + season.

batting_innings = (
    df[
        [
            "striker_id",
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
            "striker_id",
            "comp_id",
            "comp_name",
            "match_format",
            "season",
        ],
        dropna=False,
    )
    .size()
    .reset_index(name="innings")
)


# Merge actual innings count into batting table

batting = batting.merge(
    batting_innings,
    on=[
        "striker_id",
        "comp_id",
        "comp_name",
        "match_format",
        "season",
    ],
    how="left",
    validate="one_to_one",
)


# ==================================================
# 8. CALCULATE DERIVED METRICS
# ==================================================

print("Calculating derived batting metrics...")

# Strike rate = Runs / Balls faced * 100

batting["strike_rate"] = np.where(
    batting["balls_faced"] > 0,
    (
        batting["runs"]
        / batting["balls_faced"]
    ) * 100,
    np.nan,
)


# Batting average = Runs / Dismissals

batting["batting_average"] = np.where(
    batting["dismissals"] > 0,
    batting["runs"] / batting["dismissals"],
    np.nan,
)


# Boundary runs

batting["boundary_runs"] = (
    batting["fours"] * 4
    + batting["sixes"] * 6
)


# Boundary percentage

batting["boundary_percentage"] = np.where(
    batting["runs"] > 0,
    (
        batting["boundary_runs"]
        / batting["runs"]
    ) * 100,
    np.nan,
)


# ==================================================
# 9. ADD DATA QUALITY INDICATORS
# ==================================================

# Keep quality information visible.
# These flags are not automatically removed from totals.

batting["quality_flagged_percentage"] = np.where(
    batting["balls_faced"] > 0,
    (
        batting["quality_flagged_deliveries"]
        / batting["balls_faced"]
    ) * 100,
    np.nan,
)


# ==================================================
# 10. VALIDATE FEATURE TABLE
# ==================================================

print("\nRunning validation checks...")

# No duplicate player-competition-season records

duplicate_count = batting.duplicated(
    subset=group_columns
).sum()

if duplicate_count != 0:
    raise ValueError(
        f"Duplicate batting records found: {duplicate_count}"
    )


# Total runs must match delivery-level total

source_runs = int(df["batter_runs"].sum())
feature_runs = int(batting["runs"].sum())

if source_runs != feature_runs:
    raise ValueError(
        "Batter runs mismatch between source and feature table."
    )


# Total balls faced must match

source_balls = int(df["balls_faced"].sum())
feature_balls = int(batting["balls_faced"].sum())

if source_balls != feature_balls:
    raise ValueError(
        "Balls faced mismatch between source and feature table."
    )


# Total dismissals must match

source_dismissals = int(
    df["batter_dismissal"].sum()
)

feature_dismissals = int(
    batting["dismissals"].sum()
)

if source_dismissals != feature_dismissals:
    raise ValueError(
        "Dismissals mismatch between source and feature table."
    )


# ==================================================
# 11. SORT RESULTS
# ==================================================

batting = batting.sort_values(
    by=["runs", "strike_rate"],
    ascending=[False, False],
).reset_index(drop=True)


# ==================================================
# 12. SAVE FEATURE DATASET
# ==================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

batting.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==================================================
# 13. FINAL REPORT
# ==================================================

print("\n" + "=" * 65)
print("BATTING FEATURE ENGINEERING REPORT")
print("=" * 65)

print("\nDATASET SUMMARY")

print("Delivery records:", len(df))
print("Player competition-season records:", len(batting))
print("Unique players:", batting["striker_id"].nunique())
print("Unique competitions:", batting["comp_id"].nunique())

print("\nBATTING TOTALS")

print("Total batter runs:", feature_runs)
print("Total balls faced:", feature_balls)
print("Total batter dismissals:", feature_dismissals)
print("Total batting innings:", batting["innings"].sum())

print("\nTOP 10 BATTERS BY RUNS")

print(
    batting[
        [
            "mapped_batter_name",
            "runs",
            "balls_faced",
            "innings",
            "dismissals",
            "strike_rate",
            "batting_average",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ==================================================
# 14. VALIDATION SUMMARY
# ==================================================

print("\nVALIDATION RESULTS")

print("Required columns: PASS")
print("Duplicate feature records:", duplicate_count)
print("Runs reconciliation: PASS")
print("Balls reconciliation: PASS")
print("Dismissals reconciliation: PASS")

print("\nOUTPUT FILE")

print(OUTPUT_FILE)
print("Final shape:", batting.shape)

print("\nBATTING FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
