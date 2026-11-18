
from pathlib import Path
import pandas as pd
import numpy as np

# =====================================================
# 1. PATHS
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data/processed/features/"
    "pressure_bowler_match_features.csv"
)

OUTPUT_DIR = ROOT / "data/processed/features"

OUTPUT_FILE = (
    OUTPUT_DIR / "pressure_bowler_consistency_features.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. LOAD DATA
# =====================================================

print("\nLoading pressure bowler match features...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Input shape:", df.shape)


# =====================================================
# 3. REQUIRED COLUMNS
# =====================================================

required_columns = [
    "bowler_id",
    "bowler_name",
    "match_id",
    "pressure_deliveries",
    "legal_balls",
    "runs_conceded",
    "bowler_wickets",
    "dot_balls",
    "fours_conceded",
    "sixes_conceded",
    "boundaries_conceded",
    "quality_flagged_deliveries",
    "economy_rate",
    "bowling_strike_rate",
    "dot_ball_percentage",
    "boundary_percentage",
    "quality_flag_percentage",
    "match_format",
    "season",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")


# =====================================================
# 4. DATA QUALITY CHECKS
# =====================================================

if df.duplicated(["bowler_id", "match_id"]).any():
    raise ValueError("Duplicate bowler-match records found.")

print("Duplicate bowler-match records: 0")


# =====================================================
# 5. BOWLER CONSISTENCY AGGREGATION
# =====================================================

print("\nCalculating bowler consistency features...")


def calculate_consistency(group):

    valid_economy = group["economy_rate"].dropna()

    valid_dot = group["dot_ball_percentage"].dropna()

    valid_boundary = group["boundary_percentage"].dropna()

    valid_sr = group["bowling_strike_rate"].dropna()

    return pd.Series({

        # Exposure
        "pressure_matches": group["match_id"].nunique(),

        "total_pressure_deliveries":
            group["pressure_deliveries"].sum(),

        "total_legal_balls":
            group["legal_balls"].sum(),

        # Overall bowling performance
        "total_runs_conceded":
            group["runs_conceded"].sum(),

        "total_wickets":
            group["bowler_wickets"].sum(),

        "total_dot_balls":
            group["dot_balls"].sum(),

        "total_boundaries_conceded":
            group["boundaries_conceded"].sum(),

        # Economy consistency
        "average_match_economy":
            valid_economy.mean(),

        "median_match_economy":
            valid_economy.median(),

        "std_match_economy":
            valid_economy.std(),

        # Wicket consistency
        "matches_with_wicket":
            (group["bowler_wickets"] > 0).sum(),

        "wicket_match_percentage":
            (group["bowler_wickets"] > 0).mean() * 100,

        "average_wickets_per_match":
            group["bowler_wickets"].mean(),

        # Dot-ball consistency
        "average_dot_ball_percentage":
            valid_dot.mean(),

        "median_dot_ball_percentage":
            valid_dot.median(),

        "std_dot_ball_percentage":
            valid_dot.std(),

        # Boundary prevention
        "average_boundary_percentage":
            valid_boundary.mean(),

        # Bowling strike rate
        "average_bowling_strike_rate":
            valid_sr.mean(),

        # Quality information
        "quality_flagged_deliveries":
            group["quality_flagged_deliveries"].sum(),

        "quality_flag_percentage":
            (
                group["quality_flagged_deliveries"].sum()
                / group["pressure_deliveries"].sum()
                * 100
            ),

        # Valid metric samples
        "valid_economy_matches":
            len(valid_economy),

        "valid_dot_percentage_matches":
            len(valid_dot),

    })


bowler_features = (
    df.groupby(
        ["bowler_id", "bowler_name"],
        dropna=False
    )
    .apply(calculate_consistency)
    .reset_index()
)


# =====================================================
# 6. OVERALL ECONOMY AND DOT BALL RATE
# =====================================================

bowler_features["overall_pressure_economy"] = np.where(
    bowler_features["total_legal_balls"] > 0,
    bowler_features["total_runs_conceded"] * 6
    / bowler_features["total_legal_balls"],
    np.nan
)

bowler_features["overall_dot_ball_percentage"] = np.where(
    bowler_features["total_legal_balls"] > 0,
    bowler_features["total_dot_balls"] * 100
    / bowler_features["total_legal_balls"],
    np.nan
)

bowler_features["overall_boundary_percentage"] = np.where(
    bowler_features["total_legal_balls"] > 0,
    bowler_features["total_boundaries_conceded"] * 100
    / bowler_features["total_legal_balls"],
    np.nan
)


# =====================================================
# 7. VALIDATION
# =====================================================

print("\n========== BOWLER CONSISTENCY VALIDATION ==========")

print("Input match-level records:", len(df))

print("Output bowler records:", len(bowler_features))

print("Unique input bowlers:", df["bowler_id"].nunique())

print("Unique output bowlers:", bowler_features["bowler_id"].nunique())

duplicate_bowlers = bowler_features.duplicated(
    ["bowler_id"]
).sum()

print("Duplicate bowler records:", duplicate_bowlers)

delivery_check = (
    bowler_features["total_pressure_deliveries"].sum()
    == df["pressure_deliveries"].sum()
)

runs_check = np.isclose(
    bowler_features["total_runs_conceded"].sum(),
    df["runs_conceded"].sum()
)

wickets_check = (
    bowler_features["total_wickets"].sum()
    == df["bowler_wickets"].sum()
)

balls_check = (
    bowler_features["total_legal_balls"].sum()
    == df["legal_balls"].sum()
)

print("Delivery reconciliation:", delivery_check)
print("Runs reconciliation:", runs_check)
print("Wickets reconciliation:", wickets_check)
print("Legal balls reconciliation:", balls_check)

assert duplicate_bowlers == 0
assert delivery_check
assert runs_check
assert wickets_check
assert balls_check


# =====================================================
# 8. BOWLER REPORT
# =====================================================

print("\n========== BOWLER CONSISTENCY REPORT ==========")

report_columns = [
    "bowler_name",
    "pressure_matches",
    "total_wickets",
    "total_legal_balls",
    "overall_pressure_economy",
    "average_match_economy",
    "std_match_economy",
    "wicket_match_percentage",
    "overall_dot_ball_percentage",
]

print(
    bowler_features.sort_values(
        ["total_wickets", "overall_pressure_economy"],
        ascending=[False, True],
        na_position="last"
    )[report_columns]
    .head(20)
    .to_string(index=False)
)


# =====================================================
# 9. SAVE
# =====================================================

bowler_features.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nOutput shape:", bowler_features.shape)

print("\nSaved output:")
print(OUTPUT_FILE)

print("\nPRESSURE BOWLER CONSISTENCY GENERATION COMPLETE")
