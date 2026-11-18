
from pathlib import Path
import numpy as np
import pandas as pd

# =====================================================
# 1. PATHS
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

MATCH_FILE = (
    ROOT / "data/processed/features/"
    "pressure_bowler_match_features.csv"
)

ELIGIBILITY_FILE = (
    ROOT / "data/processed/features/"
    "pressure_bowler_eligibility.csv"
)

OUTPUT_DIR = ROOT / "data/processed/features"

FORMAT_FILE = (
    OUTPUT_DIR / "bowler_format_comparison.csv"
)

CROSS_FORMAT_FILE = (
    OUTPUT_DIR / "bowler_cross_format_comparison.csv"
)

SUMMARY_FILE = (
    ROOT / "data/interim/"
    "bowler_format_comparison_summary.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. LOAD DATA
# =====================================================

print("\nLoading pressure bowling data...")

matches = pd.read_csv(MATCH_FILE)
eligibility = pd.read_csv(ELIGIBILITY_FILE)

print("Bowler-match data:", matches.shape)
print("Eligibility data:", eligibility.shape)


# =====================================================
# 3. VALIDATE COLUMNS
# =====================================================

required_match_columns = [
    "bowler_id",
    "bowler_name",
    "match_id",
    "comp_name",
    "match_format",
    "pressure_deliveries",
    "legal_balls",
    "runs_conceded",
    "bowler_wickets",
    "dot_balls",
    "boundaries_conceded",
    "quality_flagged_deliveries",
]

required_eligibility_columns = [
    "bowler_id",
    "primary_eligible",
]

missing_match = [
    c for c in required_match_columns
    if c not in matches.columns
]

missing_eligibility = [
    c for c in required_eligibility_columns
    if c not in eligibility.columns
]

if missing_match:
    raise ValueError(f"Missing match columns: {missing_match}")

if missing_eligibility:
    raise ValueError(
        f"Missing eligibility columns: {missing_eligibility}"
    )


# =====================================================
# 4. SELECT PRIMARY-ELIGIBLE BOWLERS
# =====================================================

eligible_ids = eligibility.loc[
    eligibility["primary_eligible"] == True,
    "bowler_id"
].dropna().unique()

matches = matches[
    matches["bowler_id"].isin(eligible_ids)
].copy()

print("\nPrimary eligible bowlers:", len(eligible_ids))
print("Eligible bowler-match records:", len(matches))


# =====================================================
# 5. CLASSIFY IPL AND T20I
# =====================================================

matches["competition_name_clean"] = (
    matches["comp_name"]
    .astype("string")
    .str.strip()
)

matches["match_format_clean"] = (
    matches["match_format"]
    .astype("string")
    .str.strip()
    .str.upper()
)

is_ipl = (
    matches["competition_name_clean"]
    .str.contains(r"\bIPL\b", case=False, na=False)
)

is_t20i = matches["match_format_clean"].eq("T20I")

matches["comparison_format"] = np.select(
    [is_ipl, is_t20i],
    ["IPL", "T20I"],
    default="OTHER_T20"
)

print("\n========== FORMAT CLASSIFICATION ==========")

print(
    matches["comparison_format"]
    .value_counts(dropna=False)
)

comparison = matches[
    matches["comparison_format"].isin(["IPL", "T20I"])
].copy()

print("\nRecords selected for comparison:", len(comparison))
print(
    "Other T20 records excluded:",
    (matches["comparison_format"] == "OTHER_T20").sum()
)


# =====================================================
# 6. NUMERIC VALIDATION
# =====================================================

numeric_columns = [
    "pressure_deliveries",
    "legal_balls",
    "runs_conceded",
    "bowler_wickets",
    "dot_balls",
    "boundaries_conceded",
    "quality_flagged_deliveries",
]

for col in numeric_columns:
    comparison[col] = pd.to_numeric(
        comparison[col],
        errors="coerce"
    )

if comparison[numeric_columns].isna().any().any():
    raise ValueError("Missing numeric match-level metrics found.")

if (comparison["legal_balls"] < 0).any():
    raise ValueError("Negative legal balls found.")

if (comparison["runs_conceded"] < 0).any():
    raise ValueError("Negative runs conceded found.")

if (comparison["bowler_wickets"] < 0).any():
    raise ValueError("Negative wickets found.")


# =====================================================
# 7. AGGREGATE BOWLER PERFORMANCE BY FORMAT
# =====================================================

group_columns = [
    "bowler_id",
    "bowler_name",
    "comparison_format",
]

format_features = (
    comparison
    .groupby(group_columns)
    .agg(
        pressure_matches=("match_id", "nunique"),
        pressure_deliveries=("pressure_deliveries", "sum"),
        legal_balls=("legal_balls", "sum"),
        runs_conceded=("runs_conceded", "sum"),
        wickets=("bowler_wickets", "sum"),
        dot_balls=("dot_balls", "sum"),
        boundaries_conceded=("boundaries_conceded", "sum"),
        quality_flagged_deliveries=(
            "quality_flagged_deliveries", "sum"
        ),
    )
    .reset_index()
)


# =====================================================
# 8. CALCULATE FORMAT METRICS
# =====================================================

format_features["economy_rate"] = np.where(
    format_features["legal_balls"] > 0,
    (
        format_features["runs_conceded"] * 6
        / format_features["legal_balls"]
    ),
    np.nan
)

format_features["dot_ball_percentage"] = np.where(
    format_features["legal_balls"] > 0,
    (
        format_features["dot_balls"] * 100
        / format_features["legal_balls"]
    ),
    np.nan
)

format_features["bowling_strike_rate"] = np.where(
    format_features["wickets"] > 0,
    (
        format_features["legal_balls"]
        / format_features["wickets"]
    ),
    np.nan
)

format_features["boundary_percentage"] = np.where(
    format_features["legal_balls"] > 0,
    (
        format_features["boundaries_conceded"] * 100
        / format_features["legal_balls"]
    ),
    np.nan
)

format_features["quality_flag_percentage"] = np.where(
    format_features["pressure_deliveries"] > 0,
    (
        format_features["quality_flagged_deliveries"] * 100
        / format_features["pressure_deliveries"]
    ),
    np.nan
)


# =====================================================
# 9. CREATE CROSS-FORMAT COMPARISON
# =====================================================

metric_columns = [
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

wide = format_features.pivot(
    index=["bowler_id", "bowler_name"],
    columns="comparison_format",
    values=metric_columns
)

wide.columns = [
    f"{metric}_{format_name}"
    for metric, format_name in wide.columns
]

wide = wide.reset_index()

# Keep bowlers with evidence in both formats.
required_cross_format_columns = [
    "pressure_matches_IPL",
    "pressure_matches_T20I",
]

for col in required_cross_format_columns:
    if col not in wide.columns:
        wide[col] = np.nan

wide["has_both_formats"] = (
    wide["pressure_matches_IPL"].notna()
    & wide["pressure_matches_T20I"].notna()
)

cross_format = wide[
    wide["has_both_formats"]
].copy()


# =====================================================
# 10. CALCULATE FORMAT DIFFERENCES
# =====================================================

difference_metrics = [
    "economy_rate",
    "dot_ball_percentage",
    "bowling_strike_rate",
    "boundary_percentage",
]

for metric in difference_metrics:

    ipl_col = f"{metric}_IPL"
    t20i_col = f"{metric}_T20I"

    cross_format[f"{metric}_difference_IPL_minus_T20I"] = (
        cross_format[ipl_col] - cross_format[t20i_col]
    )


# =====================================================
# 11. VALIDATION
# =====================================================

print("\n========== FORMAT COMPARISON VALIDATION ==========")

print("IPL bowler records:",
      (format_features["comparison_format"] == "IPL").sum())

print("T20I bowler records:",
      (format_features["comparison_format"] == "T20I").sum())

print("Unique bowlers in format dataset:",
      format_features["bowler_id"].nunique())

print("Bowlers with both formats:",
      len(cross_format))

print("Duplicate bowler-format records:",
      format_features.duplicated(
          ["bowler_id", "comparison_format"]
      ).sum())

assert not format_features.duplicated(
    ["bowler_id", "comparison_format"]
).any()

assert (
    cross_format["has_both_formats"].all()
)

# Reconcile format-level delivery totals against selected match data.
assert (
    format_features["pressure_deliveries"].sum()
    == comparison["pressure_deliveries"].sum()
)

assert (
    format_features["legal_balls"].sum()
    == comparison["legal_balls"].sum()
)

assert (
    format_features["runs_conceded"].sum()
    == comparison["runs_conceded"].sum()
)

assert (
    format_features["wickets"].sum()
    == comparison["bowler_wickets"].sum()
)


# =====================================================
# 12. FORMAT SUMMARY
# =====================================================

summary = (
    format_features
    .groupby("comparison_format")
    .agg(
        bowlers=("bowler_id", "nunique"),
        pressure_matches=("pressure_matches", "sum"),
        pressure_deliveries=("pressure_deliveries", "sum"),
        legal_balls=("legal_balls", "sum"),
        runs_conceded=("runs_conceded", "sum"),
        wickets=("wickets", "sum"),
        dot_balls=("dot_balls", "sum"),
        boundaries=("boundaries_conceded", "sum"),
    )
    .reset_index()
)

summary["economy_rate"] = (
    summary["runs_conceded"] * 6
    / summary["legal_balls"].replace(0, np.nan)
)

summary["dot_ball_percentage"] = (
    summary["dot_balls"] * 100
    / summary["legal_balls"].replace(0, np.nan)
)

summary["bowling_strike_rate"] = (
    summary["legal_balls"]
    / summary["wickets"].replace(0, np.nan)
)

summary["boundary_percentage"] = (
    summary["boundaries"] * 100
    / summary["legal_balls"].replace(0, np.nan)
)


# =====================================================
# 13. SAVE OUTPUTS
# =====================================================

format_features.to_csv(
    FORMAT_FILE,
    index=False
)

cross_format.to_csv(
    CROSS_FORMAT_FILE,
    index=False
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# =====================================================
# 14. PRINT RESULTS
# =====================================================

print("\n========== FORMAT-WISE SUMMARY ==========")

print(summary.round(2).to_string(index=False))

print("\n========== CROSS-FORMAT SAMPLE ==========")

print(
    cross_format[
        [
            "bowler_name",
            "pressure_matches_IPL",
            "pressure_matches_T20I",
            "economy_rate_IPL",
            "economy_rate_T20I",
            "dot_ball_percentage_IPL",
            "dot_ball_percentage_T20I",
        ]
    ]
    .head(15)
    .round(2)
    .to_string(index=False)
)

print("\n========== FINAL VALIDATION ==========")

print("Format records:", len(format_features))
print("Cross-format bowlers:", len(cross_format))
print("Duplicate records:", format_features.duplicated(
    ["bowler_id", "comparison_format"]
).sum())

print("\nSaved format comparison:", FORMAT_FILE)
print("Saved cross-format comparison:", CROSS_FORMAT_FILE)
print("Saved format summary:", SUMMARY_FILE)

print("\nBOWLER FORMAT COMPARISON COMPLETE")
