
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data/processed/features/pressure_batting_features.csv"
)

OUTPUT_DIR = ROOT / "data/processed/features"

OUTPUT_FILE = (
    OUTPUT_DIR / "pressure_batter_eligibility.csv"
)

df = pd.read_csv(INPUT_FILE, low_memory=False)

# --------------------------------------------------
# 1. DEFINE CANDIDATE ELIGIBILITY GROUPS
# --------------------------------------------------

df["primary_eligible"] = (
    (df["pressure_deliveries"] >= 20)
    & (df["matches"] >= 5)
)

df["relaxed_eligible"] = (
    (df["pressure_deliveries"] >= 10)
    & (df["matches"] >= 3)
)

df["strict_eligible"] = (
    (df["pressure_deliveries"] >= 30)
    & (df["matches"] >= 8)
)


# --------------------------------------------------
# 2. ASSIGN EVIDENCE CATEGORY
# --------------------------------------------------

def classify_evidence(row):

    if row["primary_eligible"]:
        return "PRIMARY_EVIDENCE"

    if row["relaxed_eligible"]:
        return "RELAXED_EVIDENCE_ONLY"

    return "INSUFFICIENT_EVIDENCE"


df["evidence_category"] = df.apply(
    classify_evidence,
    axis=1,
)


# --------------------------------------------------
# 3. ADD REASONS
# --------------------------------------------------

def eligibility_reason(row):

    reasons = []

    if row["pressure_deliveries"] < 20:
        reasons.append("LESS_THAN_20_PRESSURE_DELIVERIES")

    if row["matches"] < 5:
        reasons.append("LESS_THAN_5_MATCHES")

    if not reasons:
        return "MEETS_PRIMARY_THRESHOLD"

    return ";".join(reasons)


df["eligibility_reason"] = df.apply(
    eligibility_reason,
    axis=1,
)


# --------------------------------------------------
# 4. VALIDATION
# --------------------------------------------------

print("\n========== PRESSURE ELIGIBILITY REPORT ==========")

print("Total batters:", len(df))

print("\nPrimary eligible:", df["primary_eligible"].sum())
print("Relaxed eligible:", df["relaxed_eligible"].sum())
print("Strict eligible:", df["strict_eligible"].sum())

print("\nEvidence category distribution:")
print(
    df["evidence_category"]
    .value_counts()
    .to_string()
)

print("\nEligibility reason distribution:")
print(
    df["eligibility_reason"]
    .value_counts()
    .to_string()
)

print("\nPrimary eligible batters:")

columns = [
    "striker_id",
    "striker_name",
    "pressure_deliveries",
    "matches",
    "pressure_runs",
    "striker_dismissals",
    "strike_rate",
    "batting_average",
    "evidence_category",
]

print(
    df[df["primary_eligible"]][columns]
    .sort_values(
        ["pressure_deliveries", "matches"],
        ascending=False,
    )
    .head(20)
    .to_string(index=False)
)


# --------------------------------------------------
# 5. SAVE
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)

print("\nPRESSURE ELIGIBILITY CLASSIFICATION COMPLETE")
