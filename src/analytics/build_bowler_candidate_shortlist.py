
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = ROOT / "data" / "processed" / "features"
INTERIM_DIR = ROOT / "data" / "interim"

INPUT_FILE = FEATURE_DIR / "bowler_evidence_matrix.csv"

OUTPUT_FILE = (
    FEATURE_DIR / "bowler_candidate_shortlist.csv"
)

SUMMARY_FILE = (
    INTERIM_DIR / "bowler_candidate_shortlist_summary.csv"
)

INTERIM_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading bowler evidence matrix...")

df = pd.read_csv(INPUT_FILE)

print("Input shape:", df.shape)


# ============================================================
# 3. VALIDATE COLUMNS
# ============================================================

required_columns = [
    "bowler_id",
    "bowler_name",
    "reliability_score",
    "pressure_matches",
    "total_pressure_deliveries",
    "format_evidence_category",
    "top10_scenario_count",
    "rank_spread",
    "top10_stable",
    "format_quality_review_required",
    "BASELINE_rank",
    "WICKET_FOCUSED_rank",
    "ECONOMY_FOCUSED_rank",
    "CONSISTENCY_FOCUSED_rank",
    "EVIDENCE_FOCUSED_rank",
    "BALANCED_rank",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

if df["bowler_id"].duplicated().any():
    raise ValueError("Duplicate bowler IDs detected.")


# ============================================================
# 4. NUMERIC VALIDATION
# ============================================================

numeric_columns = [
    "reliability_score",
    "pressure_matches",
    "total_pressure_deliveries",
    "top10_scenario_count",
    "rank_spread",
    "BASELINE_rank",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

if df[numeric_columns].isna().any().any():
    raise ValueError("Missing numeric candidate metrics found.")


# ============================================================
# 5. IDENTIFY CANDIDATE GROUPS
# ============================================================

print("\nClassifying candidate groups...")

# Stable top-10 candidates:
# Top 10 in at least five of six scenarios.

df["stable_top10_candidate"] = (
    df["top10_scenario_count"] >= 5
)

# Strong contenders:
# Top 10 in at least three scenarios.

df["strong_contender"] = (
    df["top10_scenario_count"] >= 3
)

# Baseline backup pool:
# Top 30 by baseline score.

df["baseline_top30"] = (
    df["BASELINE_rank"] <= 30
)

# Candidate pool is the union of these groups.

df["shortlist_candidate"] = (
    df["stable_top10_candidate"]
    | df["strong_contender"]
    | df["baseline_top30"]
)


# ============================================================
# 6. ASSIGN CANDIDATE CATEGORY
# ============================================================

def classify_candidate(row):

    if row["stable_top10_candidate"]:
        return "CORE_STABLE_CANDIDATE"

    if row["strong_contender"]:
        return "STRONG_CONTENDER"

    if row["baseline_top30"]:
        return "BACKUP_CANDIDATE"

    return "NOT_SHORTLISTED"


df["candidate_category"] = df.apply(
    classify_candidate,
    axis=1
)


# ============================================================
# 7. FORMAT EVIDENCE REVIEW
# ============================================================

def format_recommendation(row):

    category = row["format_evidence_category"]

    if category == "BOTH_FORMATS_SUPPORTED":
        return "EVIDENCE_IN_BOTH_FORMATS"

    if category == "IPL_EVIDENCE_ONLY":
        return "IPL_FOCUSED_EVIDENCE"

    if category == "T20I_EVIDENCE_ONLY":
        return "T20I_FOCUSED_EVIDENCE"

    return "LIMITED_FORMAT_EVIDENCE"


df["format_evidence_status"] = df.apply(
    format_recommendation,
    axis=1
)


# ============================================================
# 8. RANK STABILITY REVIEW
# ============================================================

def rank_stability(row):

    count = row["top10_scenario_count"]

    if count == 6:
        return "VERY_HIGH"

    if count >= 5:
        return "HIGH"

    if count >= 3:
        return "MODERATE"

    return "LOW"


df["rank_stability_category"] = df.apply(
    rank_stability,
    axis=1
)


# ============================================================
# 9. MANUAL REVIEW FLAGS
# ============================================================

df["manual_review_required"] = (
    df["format_quality_review_required"].fillna(False)
    | (df["rank_spread"] > 20)
    | (df["format_evidence_category"] == "LIMITED_FORMAT_EVIDENCE")
)

df["review_reason"] = np.select(
    [
        df["format_quality_review_required"].fillna(False),
        df["rank_spread"] > 20,
        df["format_evidence_category"] == "LIMITED_FORMAT_EVIDENCE",
    ],
    [
        "DELIVERY_QUALITY_REVIEW",
        "HIGH_RANK_VARIATION",
        "LIMITED_FORMAT_EVIDENCE",
    ],
    default="NO_AUTOMATIC_REVIEW_FLAG"
)


# ============================================================
# 10. CREATE SHORTLIST
# ============================================================

shortlist = df[
    df["shortlist_candidate"]
].copy()

shortlist = shortlist.sort_values(
    by=[
        "top10_scenario_count",
        "reliability_score",
        "BASELINE_rank",
    ],
    ascending=[
        False,
        False,
        True,
    ]
).reset_index(drop=True)

shortlist["candidate_review_order"] = (
    range(1, len(shortlist) + 1)
)


# ============================================================
# 11. VALIDATION
# ============================================================

print("\n========== SHORTLIST VALIDATION ==========")

print("Total eligible bowlers:", len(df))

print("Shortlisted candidates:", len(shortlist))

print(
    "Stable top-10 candidates:",
    df["stable_top10_candidate"].sum()
)

print(
    "Strong contenders:",
    df["strong_contender"].sum()
)

print(
    "Baseline top-30 candidates:",
    df["baseline_top30"].sum()
)

print(
    "Manual review required:",
    shortlist["manual_review_required"].sum()
)

print(
    "Duplicate shortlisted IDs:",
    shortlist["bowler_id"].duplicated().sum()
)

assert shortlist["bowler_id"].nunique() == len(shortlist)

assert (
    shortlist["shortlist_candidate"].all()
)

assert (
    shortlist["reliability_score"].between(0, 100).all()
)


# ============================================================
# 12. CANDIDATE CATEGORY SUMMARY
# ============================================================

print("\n========== CANDIDATE CATEGORY SUMMARY ==========")

category_summary = (
    shortlist.groupby("candidate_category")
    .agg(
        bowlers=("bowler_id", "nunique"),
        average_score=("reliability_score", "mean"),
        average_pressure_matches=("pressure_matches", "mean"),
        average_pressure_deliveries=(
            "total_pressure_deliveries",
            "mean"
        ),
        average_top10_scenario_count=(
            "top10_scenario_count",
            "mean"
        ),
    )
    .reset_index()
)

print(
    category_summary.round(2).to_string(index=False)
)


# ============================================================
# 13. FORMAT EVIDENCE SUMMARY
# ============================================================

print("\n========== FORMAT EVIDENCE SUMMARY ==========")

format_summary = (
    shortlist.groupby("format_evidence_category")
    .agg(
        candidates=("bowler_id", "nunique"),
        average_score=("reliability_score", "mean"),
        stable_candidates=("stable_top10_candidate", "sum"),
    )
    .reset_index()
)

print(
    format_summary.round(2).to_string(index=False)
)


# ============================================================
# 14. TOP CANDIDATE PREVIEW
# ============================================================

print("\n========== CANDIDATE PREVIEW ==========")

preview_columns = [
    "candidate_review_order",
    "bowler_name",
    "reliability_score",
    "BASELINE_rank",
    "top10_scenario_count",
    "rank_spread",
    "format_evidence_category",
    "candidate_category",
    "manual_review_required",
]

print(
    shortlist[preview_columns]
    .head(20)
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 15. SAVE OUTPUTS
# ============================================================

shortlist.to_csv(
    OUTPUT_FILE,
    index=False
)

category_summary.to_csv(
    SUMMARY_FILE,
    index=False
)

print("\nSaved candidate shortlist:")
print(OUTPUT_FILE)

print("\nSaved candidate summary:")
print(SUMMARY_FILE)

print("\nBOWLER CANDIDATE SHORTLIST AUDIT COMPLETE")
