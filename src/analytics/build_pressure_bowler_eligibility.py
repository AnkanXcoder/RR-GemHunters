
from pathlib import Path
import pandas as pd

# =====================================================
# 1. PATHS
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data/processed/features/"
    "pressure_bowler_consistency_features.csv"
)

OUTPUT_DIR = ROOT / "data/processed/features"

DETAIL_FILE = (
    OUTPUT_DIR / "pressure_bowler_eligibility.csv"
)

SUMMARY_FILE = (
    ROOT / "data/interim/"
    "pressure_bowler_eligibility_summary.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. LOAD DATA
# =====================================================

print("\nLoading bowler consistency features...")

df = pd.read_csv(INPUT_FILE)

print("Input shape:", df.shape)


# =====================================================
# 3. REQUIRED COLUMNS
# =====================================================

required_columns = [
    "bowler_id",
    "bowler_name",
    "pressure_matches",
    "total_pressure_deliveries",
    "total_legal_balls",
    "total_wickets",
    "quality_flagged_deliveries",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

if df["bowler_id"].duplicated().any():
    raise ValueError("Duplicate bowler IDs found.")


# =====================================================
# 4. ELIGIBILITY THRESHOLD SENSITIVITY
# =====================================================

print("\n========== THRESHOLD SENSITIVITY ==========")

delivery_thresholds = [5, 10, 15, 20, 30, 50]
match_thresholds = [1, 3, 5, 8, 10]

sensitivity_rows = []

for deliveries in delivery_thresholds:

    row = {"minimum_deliveries": deliveries}

    for matches in match_thresholds:

        eligible_count = (
            (df["total_pressure_deliveries"] >= deliveries)
            & (df["pressure_matches"] >= matches)
        ).sum()

        row[f"matches_{matches}"] = int(eligible_count)

    sensitivity_rows.append(row)

sensitivity = pd.DataFrame(sensitivity_rows)

print(sensitivity.to_string(index=False))

sensitivity.to_csv(
    ROOT / "data/interim/"
    "pressure_bowler_eligibility_threshold_sensitivity.csv",
    index=False
)


# =====================================================
# 5. DEFINE ELIGIBILITY GROUPS
# =====================================================

# Primary: at least 20 pressure deliveries and 5 matches.
# Relaxed: at least 10 pressure deliveries and 3 matches.
# Strict: at least 30 pressure deliveries and 8 matches.

df["primary_eligible"] = (
    (df["total_pressure_deliveries"] >= 20)
    & (df["pressure_matches"] >= 5)
)

df["relaxed_eligible"] = (
    (df["total_pressure_deliveries"] >= 10)
    & (df["pressure_matches"] >= 3)
)

df["strict_eligible"] = (
    (df["total_pressure_deliveries"] >= 30)
    & (df["pressure_matches"] >= 8)
)


# =====================================================
# 6. EVIDENCE CLASSIFICATION
# =====================================================

df["evidence_category"] = "INSUFFICIENT_EVIDENCE"

df.loc[
    df["relaxed_eligible"],
    "evidence_category"
] = "RELAXED_EVIDENCE_ONLY"

df.loc[
    df["primary_eligible"],
    "evidence_category"
] = "PRIMARY_EVIDENCE"

df.loc[
    df["strict_eligible"],
    "evidence_category"
] = "STRICT_EVIDENCE"


# =====================================================
# 7. EVIDENCE REASONS
# =====================================================

def classify_reason(row):

    if row["strict_eligible"]:
        return "MEETS_STRICT_THRESHOLD"

    if row["primary_eligible"]:
        return "MEETS_PRIMARY_THRESHOLD"

    if row["relaxed_eligible"]:
        return "MEETS_RELAXED_THRESHOLD_ONLY"

    if row["total_pressure_deliveries"] < 10:
        return "INSUFFICIENT_PRESSURE_DELIVERIES"

    if row["pressure_matches"] < 3:
        return "INSUFFICIENT_PRESSURE_MATCHES"

    return "INSUFFICIENT_COMBINED_EVIDENCE"


df["eligibility_reason"] = df.apply(
    classify_reason,
    axis=1
)


# =====================================================
# 8. SAMPLE QUALITY REVIEW
# =====================================================

df["quality_flag_percentage"] = (
    df["quality_flagged_deliveries"]
    / df["total_pressure_deliveries"].replace(0, pd.NA)
    * 100
)

df["quality_review_required"] = (
    df["quality_flag_percentage"] > 5
)


# =====================================================
# 9. VALIDATION
# =====================================================

print("\n========== ELIGIBILITY VALIDATION ==========")

print("Total bowlers:", len(df))

print("Primary eligible:", df["primary_eligible"].sum())

print("Relaxed eligible:", df["relaxed_eligible"].sum())

print("Strict eligible:", df["strict_eligible"].sum())

print("\nEvidence distribution:")
print(df["evidence_category"].value_counts())

print("\nEligibility reason distribution:")
print(df["eligibility_reason"].value_counts())

print(
    "\nStrict subset of primary:",
    (df["strict_eligible"] <= df["primary_eligible"]).all()
)

print(
    "Primary subset of relaxed:",
    (df["primary_eligible"] <= df["relaxed_eligible"]).all()
)

print(
    "Unique bowler IDs:",
    df["bowler_id"].nunique()
)

print(
    "Duplicate bowler IDs:",
    df["bowler_id"].duplicated().sum()
)

assert (df["strict_eligible"] <= df["primary_eligible"]).all()
assert (df["primary_eligible"] <= df["relaxed_eligible"]).all()
assert df["bowler_id"].duplicated().sum() == 0


# =====================================================
# 10. SUMMARY
# =====================================================

summary = (
    df.groupby("evidence_category")
    .agg(
        bowlers=("bowler_id", "nunique"),
        pressure_matches=("pressure_matches", "mean"),
        pressure_deliveries=("total_pressure_deliveries", "mean"),
        legal_balls=("total_legal_balls", "mean"),
        wickets=("total_wickets", "mean"),
    )
    .reset_index()
)

print("\n========== EVIDENCE SUMMARY ==========")
print(summary.to_string(index=False))

summary.to_csv(SUMMARY_FILE, index=False)


# =====================================================
# 11. SAVE DETAIL
# =====================================================

df.to_csv(
    DETAIL_FILE,
    index=False
)

print("\nSaved detail:")
print(DETAIL_FILE)

print("\nSaved summary:")
print(SUMMARY_FILE)

print("\nBOWLER ELIGIBILITY AUDIT COMPLETE")
