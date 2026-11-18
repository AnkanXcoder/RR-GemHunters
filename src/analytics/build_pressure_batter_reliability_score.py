
from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = ROOT / "data" / "processed" / "features"

INPUT_FILE = (
    FEATURE_DIR / "pressure_batter_consistency_features.csv"
)

ELIGIBILITY_FILE = (
    FEATURE_DIR / "pressure_batter_eligibility.csv"
)

OUTPUT_FILE = (
    FEATURE_DIR / "pressure_batter_reliability_scores.csv"
)

FEATURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading batting consistency features...")

batting = pd.read_csv(INPUT_FILE)
eligibility = pd.read_csv(ELIGIBILITY_FILE)

print("Batting consistency data:", batting.shape)
print("Batting eligibility data:", eligibility.shape)


# ============================================================
# 3. VALIDATE REQUIRED COLUMNS
# ============================================================

required_batting_columns = [
    "striker_id",
    "striker_name",
    "pressure_matches",
    "total_pressure_deliveries",
    "total_pressure_runs",
    "total_balls_faced",
    "total_striker_dismissals",
    "median_match_strike_rate",
    "std_match_strike_rate",
    "matches_with_boundary_percentage",
    "overall_pressure_strike_rate",
]

required_eligibility_columns = [
    "striker_id",
    "primary_eligible",
]

missing_batting = [
    col for col in required_batting_columns
    if col not in batting.columns
]

missing_eligibility = [
    col for col in required_eligibility_columns
    if col not in eligibility.columns
]

if missing_batting:
    raise ValueError(
        f"Missing batting columns: {missing_batting}"
    )

if missing_eligibility:
    raise ValueError(
        f"Missing eligibility columns: {missing_eligibility}"
    )


# ============================================================
# 4. DUPLICATE CHECKS
# ============================================================

if batting["striker_id"].duplicated().any():
    raise ValueError(
        "Duplicate striker IDs in batting consistency data."
    )

if eligibility["striker_id"].duplicated().any():
    raise ValueError(
        "Duplicate striker IDs in eligibility data."
    )


# ============================================================
# 5. MERGE ELIGIBILITY
# ============================================================

df = batting.merge(
    eligibility[
        ["striker_id", "primary_eligible"]
    ],
    on="striker_id",
    how="left",
    validate="one_to_one"
)

if df["primary_eligible"].isna().any():
    raise ValueError(
        "Some batters do not have an eligibility classification."
    )

print("\nTotal batters:", len(df))


# ============================================================
# 6. FILTER PRIMARY-ELIGIBLE BATTERS
# ============================================================

eligible = df[
    df["primary_eligible"].eq(True)
].copy()

print("Primary-eligible batters:", len(eligible))

if eligible.empty:
    raise ValueError(
        "No primary-eligible batters found."
    )


# ============================================================
# 7. NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "pressure_matches",
    "total_pressure_deliveries",
    "total_pressure_runs",
    "total_balls_faced",
    "total_striker_dismissals",
    "median_match_strike_rate",
    "std_match_strike_rate",
    "matches_with_boundary_percentage",
    "overall_pressure_strike_rate",
]

for col in numeric_columns:

    eligible[col] = pd.to_numeric(
        eligible[col],
        errors="coerce"
    )


# ============================================================
# 8. DERIVE DISMISSAL RATE
# ============================================================

print("\nCalculating dismissal resistance...")

if (
    eligible["total_balls_faced"].isna().any()
    or eligible["total_striker_dismissals"].isna().any()
):
    raise ValueError(
        "Missing balls-faced or dismissal values."
    )

if (eligible["total_balls_faced"] <= 0).any():
    raise ValueError(
        "Primary-eligible batter has zero balls faced."
    )

if (eligible["total_striker_dismissals"] < 0).any():
    raise ValueError("Negative dismissal count detected.")

eligible["dismissal_rate_per_100_balls"] = (
    eligible["total_striker_dismissals"]
    / eligible["total_balls_faced"]
    * 100
)


# ============================================================
# 9. HANDLE MISSING CONSISTENCY VALUES
# ============================================================

# A missing standard deviation can occur when there is
# insufficient valid match-level strike-rate information.
# Use the eligible-group median and preserve an audit flag.

eligible["consistency_imputed"] = (
    eligible["std_match_strike_rate"].isna()
)

std_median = eligible[
    "std_match_strike_rate"
].median()

if pd.isna(std_median):
    raise ValueError(
        "No valid match strike-rate standard deviations found."
    )

eligible["std_match_strike_rate"] = (
    eligible["std_match_strike_rate"].fillna(std_median)
)


# ============================================================
# 10. VALIDATE SCORING INPUTS
# ============================================================

scoring_inputs = [
    "overall_pressure_strike_rate",
    "median_match_strike_rate",
    "std_match_strike_rate",
    "matches_with_boundary_percentage",
    "dismissal_rate_per_100_balls",
    "pressure_matches",
    "total_pressure_deliveries",
]

print("\nMissing scoring metrics:")

print(
    eligible[scoring_inputs].isna().sum()
)

if eligible[scoring_inputs].isna().any().any():
    raise ValueError(
        "Missing scoring metrics remain. "
        "Inspect the source data before scoring."
    )

if (
    ~eligible["matches_with_boundary_percentage"]
    .between(0, 100)
).any():
    raise ValueError(
        "Boundary-match percentage outside 0-100."
    )

if (eligible["std_match_strike_rate"] < 0).any():
    raise ValueError(
        "Negative strike-rate standard deviation."
    )

if (eligible["overall_pressure_strike_rate"] < 0).any():
    raise ValueError(
        "Negative overall strike rate."
    )


# ============================================================
# 11. PERCENTILE NORMALIZATION
# ============================================================

def percentile_score(series, higher_is_better=True):

    values = series if higher_is_better else -series

    return (
        values.rank(method="average", pct=True) * 100
    )


# Pressure scoring ability:
# Higher overall strike rate is better.

eligible["scoring_ability_score"] = percentile_score(
    eligible["overall_pressure_strike_rate"],
    higher_is_better=True
)


# Match consistency:
# Higher median match SR is better.
# Lower variation in match SR is better.

eligible["median_strike_rate_score"] = percentile_score(
    eligible["median_match_strike_rate"],
    higher_is_better=True
)

eligible["strike_rate_stability_score"] = percentile_score(
    eligible["std_match_strike_rate"],
    higher_is_better=False
)

eligible["consistency_score"] = (
    eligible["median_strike_rate_score"] * 0.5
    + eligible["strike_rate_stability_score"] * 0.5
)


# Boundary involvement:
# Higher percentage of matches with a boundary is better.

eligible["boundary_score"] = percentile_score(
    eligible["matches_with_boundary_percentage"],
    higher_is_better=True
)


# Dismissal resistance:
# Lower dismissal rate is better.

eligible["dismissal_resistance_score"] = percentile_score(
    eligible["dismissal_rate_per_100_balls"],
    higher_is_better=False
)


# Evidence strength:
# Combine pressure matches and pressure deliveries.

eligible["match_evidence_score"] = percentile_score(
    eligible["pressure_matches"],
    higher_is_better=True
)

eligible["delivery_evidence_score"] = percentile_score(
    eligible["total_pressure_deliveries"],
    higher_is_better=True
)

eligible["evidence_score"] = (
    eligible["match_evidence_score"] * 0.5
    + eligible["delivery_evidence_score"] * 0.5
)


# ============================================================
# 12. WEIGHTED RELIABILITY SCORE
# ============================================================

WEIGHTS = {
    "scoring_ability_score": 0.30,
    "consistency_score": 0.25,
    "boundary_score": 0.15,
    "dismissal_resistance_score": 0.15,
    "evidence_score": 0.15,
}

if abs(sum(WEIGHTS.values()) - 1.0) > 1e-9:
    raise ValueError(
        "Scoring weights must sum to 1."
    )

eligible["batting_reliability_score"] = sum(
    eligible[column] * weight
    for column, weight in WEIGHTS.items()
)

eligible["batting_reliability_score"] = (
    eligible["batting_reliability_score"].clip(0, 100)
)


# ============================================================
# 13. QUALITY AND SAMPLE REVIEW
# ============================================================

eligible["sample_review_required"] = (
    eligible["consistency_imputed"]
)

eligible["sample_review_reason"] = np.where(
    eligible["consistency_imputed"],
    "MISSING_MATCH_VARIATION_IMPUTED",
    "NO_AUTOMATIC_SAMPLE_REVIEW_FLAG"
)


# ============================================================
# 14. RANK PRIMARY-ELIGIBLE BATTERS
# ============================================================

eligible = eligible.sort_values(
    by=[
        "batting_reliability_score",
        "total_pressure_deliveries",
    ],
    ascending=[
        False,
        False,
    ]
).reset_index(drop=True)

eligible["provisional_rank"] = (
    eligible["batting_reliability_score"]
    .rank(method="min", ascending=False)
    .astype(int)
)


# ============================================================
# 15. VALIDATION
# ============================================================

print("\n========== BATTING SCORING VALIDATION ==========")

print("Eligible batters scored:", len(eligible))

print(
    "Score range:",
    round(eligible["batting_reliability_score"].min(), 2),
    "to",
    round(eligible["batting_reliability_score"].max(), 2)
)

print(
    "Missing reliability scores:",
    eligible["batting_reliability_score"].isna().sum()
)

print(
    "Scores within 0-100:",
    eligible["batting_reliability_score"]
    .between(0, 100).all()
)

print(
    "Duplicate striker IDs:",
    eligible["striker_id"].duplicated().sum()
)

print(
    "Consistency values imputed:",
    eligible["consistency_imputed"].sum()
)

print(
    "Sample review required:",
    eligible["sample_review_required"].sum()
)

assert eligible["batting_reliability_score"].notna().all()

assert eligible[
    "batting_reliability_score"
].between(0, 100).all()

assert eligible["striker_id"].duplicated().sum() == 0

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


# ============================================================
# 16. SCORE DISTRIBUTION
# ============================================================

print("\n========== BATTING SCORE DISTRIBUTION ==========")

print(
    eligible["batting_reliability_score"]
    .describe()
    .round(2)
)

component_columns = [
    "scoring_ability_score",
    "consistency_score",
    "boundary_score",
    "dismissal_resistance_score",
    "evidence_score",
]

print("\nComponent score averages:")

print(
    eligible[component_columns]
    .mean()
    .round(2)
)


# ============================================================
# 17. TOP 15 PREVIEW
# ============================================================

print("\n========== TOP 15 BATTERS PREVIEW ==========")

preview_columns = [
    "provisional_rank",
    "striker_name",
    "batting_reliability_score",
    "pressure_matches",
    "total_pressure_deliveries",
    "overall_pressure_strike_rate",
    "dismissal_rate_per_100_balls",
    "sample_review_required",
]

print(
    eligible[preview_columns]
    .head(15)
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 18. SAVE ALL BATTERS
# ============================================================

score_columns = [
    "striker_id",
    "provisional_rank",
    "batting_reliability_score",
    "scoring_ability_score",
    "consistency_score",
    "boundary_score",
    "dismissal_resistance_score",
    "evidence_score",
    "dismissal_rate_per_100_balls",
    "consistency_imputed",
    "sample_review_required",
    "sample_review_reason",
]

final_scores = eligible[
    score_columns
].copy()

final_df = df.merge(
    final_scores,
    on="striker_id",
    how="left",
    validate="one_to_one"
)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:", OUTPUT_FILE)

print("\nPRESSURE BATTING RELIABILITY SCORING COMPLETE")
