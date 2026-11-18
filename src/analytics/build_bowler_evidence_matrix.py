
from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = ROOT / "data" / "processed" / "features"
INTERIM_DIR = ROOT / "data" / "interim"

SCORE_FILE = (
    FEATURE_DIR / "pressure_bowler_reliability_scores.csv"
)

FORMAT_FILE = (
    FEATURE_DIR / "bowler_format_comparison.csv"
)

SENSITIVITY_FILE = (
    INTERIM_DIR / "pressure_bowler_weight_sensitivity_detail.csv"
)

OUTPUT_FILE = (
    FEATURE_DIR / "bowler_evidence_matrix.csv"
)

SUMMARY_FILE = (
    INTERIM_DIR / "bowler_evidence_matrix_summary.csv"
)

INTERIM_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD INPUT FILES
# ============================================================

print("\nLoading bowler analysis files...")

scores = pd.read_csv(SCORE_FILE)
formats = pd.read_csv(FORMAT_FILE)
sensitivity = pd.read_csv(SENSITIVITY_FILE)

print("Reliability scores:", scores.shape)
print("Format comparison:", formats.shape)
print("Sensitivity analysis:", sensitivity.shape)


# ============================================================
# 3. VALIDATE REQUIRED COLUMNS
# ============================================================

score_required = [
    "bowler_id",
    "bowler_name",
    "primary_eligible",
    "reliability_score",
    "quality_flag_percentage",
]

format_required = [
    "bowler_id",
    "bowler_name",
    "comparison_format",
    "pressure_matches",
    "pressure_deliveries",
    "legal_balls",
    "runs_conceded",
    "wickets",
    "economy_rate",
    "dot_ball_percentage",
    "bowling_strike_rate",
    "boundary_percentage",
    "quality_flag_percentage",
]

rank_columns = [
    "BASELINE_rank",
    "WICKET_FOCUSED_rank",
    "ECONOMY_FOCUSED_rank",
    "CONSISTENCY_FOCUSED_rank",
    "EVIDENCE_FOCUSED_rank",
    "BALANCED_rank",
]

sensitivity_required = [
    "bowler_id",
    *rank_columns,
]


def validate_columns(data, required, filename):

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {filename}: {missing}"
        )


validate_columns(
    scores,
    score_required,
    "reliability scores"
)

validate_columns(
    formats,
    format_required,
    "format comparison"
)

validate_columns(
    sensitivity,
    sensitivity_required,
    "sensitivity analysis"
)


# ============================================================
# 4. FILTER PRIMARY-ELIGIBLE BOWLERS
# ============================================================

print("\nFiltering primary-eligible bowlers...")

scores = scores[
    scores["primary_eligible"].eq(True)
].copy()

if scores["bowler_id"].duplicated().any():
    raise ValueError(
        "Duplicate bowler IDs found in reliability scores."
    )

print("Primary-eligible bowlers:", len(scores))


# ============================================================
# 5. FORMAT DATA PIVOT
# ============================================================

print("\nPreparing format-wise features...")

format_metrics = [
    "pressure_matches",
    "pressure_deliveries",
    "legal_balls",
    "runs_conceded",
    "wickets",
    "economy_rate",
    "dot_ball_percentage",
    "bowling_strike_rate",
    "boundary_percentage",
    "quality_flag_percentage",
]

formats = formats.copy()

formats["comparison_format"] = (
    formats["comparison_format"]
    .astype(str)
    .str.strip()
    .str.upper()
)

formats = formats[
    formats["comparison_format"].isin(["IPL", "T20I"])
].copy()

if formats.duplicated(
    ["bowler_id", "comparison_format"]
).any():
    raise ValueError(
        "Duplicate bowler-format records detected."
    )

format_wide = formats.pivot(
    index="bowler_id",
    columns="comparison_format",
    values=format_metrics
)

format_wide.columns = [
    f"{metric}_{format_name}"
    for metric, format_name in format_wide.columns
]

format_wide = format_wide.reset_index()


# ============================================================
# 6. MERGE FORMAT DATA WITH RELIABILITY SCORES
# ============================================================

print("\nMerging format and reliability data...")

df = scores.merge(
    format_wide,
    on="bowler_id",
    how="left",
    validate="one_to_one"
)

print("After format merge:", df.shape)


# ============================================================
# 7. MERGE SENSITIVITY RANKS
# ============================================================

print("\nMerging sensitivity ranks...")

if sensitivity["bowler_id"].duplicated().any():
    raise ValueError(
        "Duplicate bowler IDs found in sensitivity data."
    )

df = df.merge(
    sensitivity[
        ["bowler_id"] + rank_columns
    ],
    on="bowler_id",
    how="left",
    validate="one_to_one"
)

if df[rank_columns].isna().any().any():
    raise ValueError(
        "Missing sensitivity ranks after merge."
    )

print("After sensitivity merge:", df.shape)


# ============================================================
# 8. FORMAT EVIDENCE CLASSIFICATION
# ============================================================

print("\nClassifying format evidence...")

for col in [
    "pressure_matches_IPL",
    "pressure_matches_T20I",
]:
    if col not in df.columns:
        df[col] = np.nan

df["has_IPL_evidence"] = (
    df["pressure_matches_IPL"].notna()
)

df["has_T20I_evidence"] = (
    df["pressure_matches_T20I"].notna()
)

# Three pressure matches is the provisional
# minimum for format-specific evidence.

df["IPL_evidence_sufficient"] = (
    df["pressure_matches_IPL"].fillna(0) >= 3
)

df["T20I_evidence_sufficient"] = (
    df["pressure_matches_T20I"].fillna(0) >= 3
)


def classify_format_evidence(row):

    ipl = row["IPL_evidence_sufficient"]
    t20i = row["T20I_evidence_sufficient"]

    if ipl and t20i:
        return "BOTH_FORMATS_SUPPORTED"

    elif ipl:
        return "IPL_EVIDENCE_ONLY"

    elif t20i:
        return "T20I_EVIDENCE_ONLY"

    return "LIMITED_FORMAT_EVIDENCE"


df["format_evidence_category"] = df.apply(
    classify_format_evidence,
    axis=1
)


# ============================================================
# 9. RANK STABILITY ANALYSIS
# ============================================================

print("\nCalculating rank stability...")

df["best_scenario_rank"] = (
    df[rank_columns].min(axis=1)
)

df["worst_scenario_rank"] = (
    df[rank_columns].max(axis=1)
)

df["rank_spread"] = (
    df["worst_scenario_rank"]
    - df["best_scenario_rank"]
)

df["top10_scenario_count"] = (
    df[rank_columns]
    .le(10)
    .sum(axis=1)
)

df["top10_stable"] = (
    df["top10_scenario_count"] >= 5
)


# ============================================================
# 10. DATA QUALITY REVIEW
# ============================================================

print("\nChecking quality flags...")

# The reliability score file may contain duplicate
# quality-review columns after earlier merges.
# We safely combine all available versions.

quality_review_columns = [
    col
    for col in [
        "quality_review_required",
        "quality_review_required_x",
        "quality_review_required_y",
    ]
    if col in df.columns
]

if quality_review_columns:

    base_quality_review = (
        df[quality_review_columns]
        .fillna(False)
        .astype(bool)
        .any(axis=1)
    )

else:

    base_quality_review = pd.Series(
        False,
        index=df.index
    )


ipl_quality = df.get(
    "quality_flag_percentage_IPL",
    pd.Series(0, index=df.index)
)

t20i_quality = df.get(
    "quality_flag_percentage_T20I",
    pd.Series(0, index=df.index)
)

overall_quality = df.get(
    "quality_flag_percentage",
    pd.Series(0, index=df.index)
)

ipl_quality = pd.to_numeric(
    ipl_quality,
    errors="coerce"
).fillna(0)

t20i_quality = pd.to_numeric(
    t20i_quality,
    errors="coerce"
).fillna(0)

overall_quality = pd.to_numeric(
    overall_quality,
    errors="coerce"
).fillna(0)

df["format_quality_review_required"] = (
    ipl_quality.gt(5)
    | t20i_quality.gt(5)
    | overall_quality.gt(5)
    | base_quality_review
)


# ============================================================
# 11. EVIDENCE MATRIX VALIDATION
# ============================================================

print("\n========== EVIDENCE MATRIX VALIDATION ==========")

print("Matrix rows:", len(df))

print(
    "Unique bowlers:",
    df["bowler_id"].nunique()
)

print(
    "Duplicate bowler IDs:",
    df["bowler_id"].duplicated().sum()
)

print(
    "Missing baseline scores:",
    df["reliability_score"].isna().sum()
)

print(
    "Missing sensitivity ranks:",
    df[rank_columns].isna().sum().sum()
)

print(
    "Bowlers with IPL evidence:",
    df["has_IPL_evidence"].sum()
)

print(
    "Bowlers with T20I evidence:",
    df["has_T20I_evidence"].sum()
)

print(
    "Bowlers with both-format evidence:",
    df["format_evidence_category"]
    .eq("BOTH_FORMATS_SUPPORTED")
    .sum()
)

print(
    "Quality review required:",
    df["format_quality_review_required"].sum()
)


# ============================================================
# 12. ASSERTIONS
# ============================================================

assert len(df) == len(scores)

assert (
    df["bowler_id"].nunique()
    == len(df)
)

assert (
    df["reliability_score"].notna().all()
)

assert (
    df[rank_columns].notna().all().all()
)

assert (
    df["rank_spread"].ge(0).all()
)

assert (
    df["top10_scenario_count"].between(0, 6).all()
)


# ============================================================
# 13. FORMAT EVIDENCE SUMMARY
# ============================================================

print("\n========== FORMAT EVIDENCE SUMMARY ==========")

summary = (
    df.groupby("format_evidence_category")
    .agg(
        bowlers=("bowler_id", "nunique"),
        average_pressure_matches=(
            "pressure_matches",
            "mean"
        ),
        average_reliability_score=(
            "reliability_score",
            "mean"
        ),
        average_rank_spread=(
            "rank_spread",
            "mean"
        ),
        stable_top10_bowlers=(
            "top10_stable",
            "sum"
        ),
    )
    .reset_index()
)

print(
    summary.round(2).to_string(index=False)
)


# ============================================================
# 14. RANK STABILITY SUMMARY
# ============================================================

print("\n========== RANK STABILITY SUMMARY ==========")

print(
    "Top 10 in all six scenarios:",
    (df["top10_scenario_count"] == 6).sum()
)

print(
    "Top 10 in at least five scenarios:",
    df["top10_stable"].sum()
)

print(
    "Average rank spread:",
    round(df["rank_spread"].mean(), 2)
)

print(
    "Median rank spread:",
    round(df["rank_spread"].median(), 2)
)


# ============================================================
# 15. TOP CANDIDATE EVIDENCE PREVIEW
# ============================================================

print("\n========== TOP CANDIDATE EVIDENCE PREVIEW ==========")

preview_columns = [
    "bowler_name",
    "reliability_score",
    "pressure_matches_IPL",
    "pressure_matches_T20I",
    "format_evidence_category",
    "top10_scenario_count",
    "rank_spread",
    "format_quality_review_required",
]

preview_columns = [
    col for col in preview_columns
    if col in df.columns
]

preview = df.sort_values(
    by="reliability_score",
    ascending=False
).head(15)

print(
    preview[preview_columns]
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 16. SAVE OUTPUTS
# ============================================================

print("\nSaving evidence matrix...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)

print("\nSaved evidence matrix:")
print(OUTPUT_FILE)

print("\nSaved evidence summary:")
print(SUMMARY_FILE)

print("\nBOWLER EVIDENCE MATRIX COMPLETE")
