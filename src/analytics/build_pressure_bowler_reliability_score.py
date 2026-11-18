
from pathlib import Path
import numpy as np
import pandas as pd

# =====================================================
# 1. PATHS
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data/processed/features/"
    "pressure_bowler_eligibility.csv"
)

OUTPUT_DIR = ROOT / "data/processed/features"

OUTPUT_FILE = (
    OUTPUT_DIR / "pressure_bowler_reliability_scores.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. LOAD DATA
# =====================================================

print("\nLoading bowler eligibility data...")

df = pd.read_csv(INPUT_FILE)

print("Input shape:", df.shape)


# =====================================================
# 3. REQUIRED COLUMNS
# =====================================================

required_columns = [
    "bowler_id",
    "bowler_name",
    "primary_eligible",
    "pressure_matches",
    "total_pressure_deliveries",
    "average_wickets_per_match",
    "overall_pressure_economy",
    "overall_dot_ball_percentage",
    "std_match_economy",
    "quality_flagged_deliveries",
    "quality_flag_percentage",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

if df["bowler_id"].duplicated().any():
    raise ValueError("Duplicate bowler IDs found.")


# =====================================================
# 4. FILTER PRIMARY ELIGIBLE BOWLERS
# =====================================================

eligible = df[df["primary_eligible"] == True].copy()

print("Total bowlers:", len(df))
print("Primary eligible bowlers:", len(eligible))

if eligible.empty:
    raise ValueError("No primary-eligible bowlers found.")


# =====================================================
# 5. VALIDATE SCORING INPUTS
# =====================================================

metrics = [
    "average_wickets_per_match",
    "overall_pressure_economy",
    "overall_dot_ball_percentage",
    "std_match_economy",
    "pressure_matches",
    "total_pressure_deliveries",
]

for metric in metrics:
    eligible[metric] = pd.to_numeric(
        eligible[metric],
        errors="coerce"
    )

missing_metrics = eligible[metrics].isna().sum()

print("\nMissing scoring metrics:")
print(missing_metrics)

if missing_metrics.any():
    raise ValueError(
        "Missing scoring metrics detected. "
        "Review the source data before scoring."
    )

if (eligible["overall_pressure_economy"] < 0).any():
    raise ValueError("Negative economy values detected.")

if (
    ~eligible["overall_dot_ball_percentage"].between(0, 100)
).any():
    raise ValueError("Dot-ball percentage outside 0-100.")

if (eligible["std_match_economy"] < 0).any():
    raise ValueError("Negative economy standard deviation detected.")


# =====================================================
# 6. PERCENTILE NORMALIZATION
# =====================================================

def percentile_score(series, higher_is_better=True):

    values = series if higher_is_better else -series

    return values.rank(
        method="average",
        pct=True
    ) * 100


# Wicket-taking ability: higher is better
eligible["wicket_score"] = percentile_score(
    eligible["average_wickets_per_match"],
    higher_is_better=True
)

# Economy control: lower economy is better
eligible["economy_score"] = percentile_score(
    eligible["overall_pressure_economy"],
    higher_is_better=False
)

# Dot-ball pressure: higher is better
eligible["dot_ball_score"] = percentile_score(
    eligible["overall_dot_ball_percentage"],
    higher_is_better=True
)

# Match consistency: lower economy variation is better
eligible["consistency_score"] = percentile_score(
    eligible["std_match_economy"],
    higher_is_better=False
)

# Evidence: combine match and delivery sample strength
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


# =====================================================
# 7. WEIGHTED RELIABILITY SCORE
# =====================================================

WEIGHTS = {
    "wicket_score": 0.30,
    "economy_score": 0.25,
    "dot_ball_score": 0.20,
    "consistency_score": 0.15,
    "evidence_score": 0.10,
}

eligible["reliability_score"] = (
    eligible["wicket_score"] * WEIGHTS["wicket_score"]
    + eligible["economy_score"] * WEIGHTS["economy_score"]
    + eligible["dot_ball_score"] * WEIGHTS["dot_ball_score"]
    + eligible["consistency_score"] * WEIGHTS["consistency_score"]
    + eligible["evidence_score"] * WEIGHTS["evidence_score"]
)

# Keep full precision in the dataset.
eligible["reliability_score"] = (
    eligible["reliability_score"].clip(0, 100)
)


# =====================================================
# 8. QUALITY REVIEW FLAGS
# =====================================================

eligible["quality_review_required"] = (
    eligible["quality_flag_percentage"] > 5
)

eligible["quality_review_status"] = np.where(
    eligible["quality_review_required"],
    "REVIEW_REQUIRED",
    "WITHIN_REVIEW_THRESHOLD"
)


# =====================================================
# 9. RANKING WITHIN PRIMARY ELIGIBILITY GROUP
# =====================================================

eligible = eligible.sort_values(
    by=["reliability_score", "total_pressure_deliveries"],
    ascending=[False, False]
).reset_index(drop=True)

eligible["provisional_rank"] = (
    eligible["reliability_score"]
    .rank(method="min", ascending=False)
    .astype(int)
)


# =====================================================
# 10. VALIDATION
# =====================================================

print("\n========== SCORING VALIDATION ==========")

print("Eligible bowlers scored:", len(eligible))

print(
    "Score range:",
    round(eligible["reliability_score"].min(), 2),
    "to",
    round(eligible["reliability_score"].max(), 2)
)

print(
    "Missing reliability scores:",
    eligible["reliability_score"].isna().sum()
)

print(
    "Scores within 0-100:",
    eligible["reliability_score"].between(0, 100).all()
)

print(
    "Duplicate bowler IDs:",
    eligible["bowler_id"].duplicated().sum()
)

print(
    "Quality review required:",
    eligible["quality_review_required"].sum()
)

assert eligible["reliability_score"].notna().all()
assert eligible["reliability_score"].between(0, 100).all()
assert eligible["bowler_id"].duplicated().sum() == 0

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


# =====================================================
# 11. SCORE DISTRIBUTION
# =====================================================

print("\n========== SCORE DISTRIBUTION ==========")

print(
    eligible["reliability_score"].describe().round(2)
)

print("\nComponent score averages:")

component_columns = [
    "wicket_score",
    "economy_score",
    "dot_ball_score",
    "consistency_score",
    "evidence_score",
]

print(
    eligible[component_columns]
    .mean()
    .round(2)
)


# =====================================================
# 12. SAVE RESULTS
# =====================================================

# Include all 518 bowlers in the output.
# Non-primary-eligible bowlers have no reliability score.

score_columns = [
    "bowler_id",
    "provisional_rank",
    "reliability_score",
    "wicket_score",
    "economy_score",
    "dot_ball_score",
    "consistency_score",
    "evidence_score",
    "quality_review_required",
    "quality_review_status",
]

scored_columns = eligible[
    ["bowler_id"] + score_columns[1:]
]

final_df = df.merge(
    scored_columns,
    on="bowler_id",
    how="left",
    validate="one_to_one"
)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:", OUTPUT_FILE)

print("\nPRESSURE BOWLING RELIABILITY SCORING COMPLETE")
