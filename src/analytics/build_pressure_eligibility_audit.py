"""
Build a match-level eligibility audit for the primary RR GemHunters
second-innings pressure model.

Inputs
------
data/interim/chase_completion_detail_v3.csv
data/interim/defended_innings_winner_audit.csv

Outputs
-------
data/interim/pressure_model_eligibility_detail.csv
data/interim/pressure_model_eligibility_summary.csv
data/interim/pressure_model_eligible_matches.csv
data/interim/pressure_model_manual_review.csv

Run from the repository root:
    python src/analytics/build_pressure_eligibility_audit.py
"""

from pathlib import Path
import sys

import pandas as pd


# ---------------------------------------------------------------------
# 1. PATHS AND CONFIGURATION
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DETAIL_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "chase_completion_detail_v3.csv"
)

WINNER_AUDIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "defended_innings_winner_audit.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"

DETAIL_OUTPUT_PATH = OUTPUT_DIR / "pressure_model_eligibility_detail.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "pressure_model_eligibility_summary.csv"
ELIGIBLE_OUTPUT_PATH = OUTPUT_DIR / "pressure_model_eligible_matches.csv"
REVIEW_OUTPUT_PATH = OUTPUT_DIR / "pressure_model_manual_review.csv"

EXPECTED_MATCH_COUNT = 837
EXPECTED_DEFENDED_AUDIT_COUNT = 217

PRIMARY_ENDING_CATEGORIES = {
    "STANDARD_TARGET_REACHED",
    "OVERS_COMPLETED_DEFENDED",
    "ALL_OUT_BEFORE_STANDARD_TARGET",
}

REVIEW_CATEGORIES = {
    "REVISED_TARGET_REQUIRES_VALIDATION",
    "NO_RESULT_OR_DRAW",
    "OTHER_OR_REQUIRES_REVIEW",
    "SUPER_OVER_OR_ELIMINATOR",
    "SUPER_OVER_RESULT_SCORE_CONFLICT_REVIEW",
    "REGULATION_SCORES_LEVEL_REVIEW",
}


# ---------------------------------------------------------------------
# 2. HELPERS
# ---------------------------------------------------------------------

def stop(message: str) -> None:
    """Print a clear error and stop execution."""
    print(f"\nERROR: {message}")
    sys.exit(1)


def parse_bool(value):
    """
    Convert common CSV boolean representations to pandas nullable boolean.

    Missing or unrecognized values remain pd.NA rather than being silently
    interpreted as False.
    """
    if pd.isna(value):
        return pd.NA

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False

    value = str(value).strip().lower()

    if value in {"true", "1", "yes", "y"}:
        return True

    if value in {"false", "0", "no", "n"}:
        return False

    return pd.NA


def add_boolean_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Parse the requested columns as pandas nullable BooleanDtype."""
    for column in columns:
        if column not in df.columns:
            stop(f"Required boolean column is missing: {column}")

        df[column] = df[column].map(parse_bool).astype("boolean")

    return df


def true_mask(series: pd.Series) -> pd.Series:
    """Treat only explicit True as True. Missing values are not True."""
    return series.eq(True).fillna(False)


def false_mask(series: pd.Series) -> pd.Series:
    """Treat only explicit False as False. Missing values are not False."""
    return series.eq(False).fillna(False)


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        stop(f"{label} is missing required columns: {missing}")


# ---------------------------------------------------------------------
# 3. LOAD INPUT FILES
# ---------------------------------------------------------------------

print_section("PRESSURE MODEL ELIGIBILITY AUDIT")

if not DETAIL_PATH.exists():
    stop(f"Input file not found: {DETAIL_PATH}")

if not WINNER_AUDIT_PATH.exists():
    stop(f"Input file not found: {WINNER_AUDIT_PATH}")

detail = pd.read_csv(DETAIL_PATH, low_memory=False)
winner_audit = pd.read_csv(WINNER_AUDIT_PATH, low_memory=False)

print("\nINPUT DATA")
print("-" * 72)
print("V3 match detail:", detail.shape)
print("Defended winner audit:", winner_audit.shape)


# ---------------------------------------------------------------------
# 4. REQUIRED COLUMN CHECKS
# ---------------------------------------------------------------------

detail_required = [
    "match_id",
    "match_date",
    "home_team",
    "away_team",
    "match_format",
    "season",
    "chase_ending_category_v3",
    "standard_chase_target",
    "revised_target",
    "result",
    "target_reached_v3",
    "all_out_v3",
    "has_revised_target_v3",
    "revised_target_evidence_v3",
    "dls_or_duckworth_lewis_text_v3",
    "super_over_outcome_v3",
    "tie_score_conflict_v3",
    "no_result_v3",
    "no_second_innings_v3",
    "winner_validation_passed_v3",
    "validated_defended_v3",
    "manual_review_reason_v3",
    "review_status_v3",
]

winner_required = [
    "match_id",
    "batting_team_id",
    "bowling_team_id",
    "batting_team_count",
    "bowling_team_count",
    "winner_team_id",
    "winner_team_count",
    "team_data_consistent",
    "winner_is_second_innings_bowling_team",
    "winner_is_second_innings_batting_team",
    "winner_validation_status",
]

require_columns(detail, detail_required, "V3 match detail")
require_columns(winner_audit, winner_required, "Defended winner audit")


# ---------------------------------------------------------------------
# 5. BASIC INTEGRITY
# ---------------------------------------------------------------------

print_section("BASIC INTEGRITY")

detail["match_id"] = pd.to_numeric(detail["match_id"], errors="coerce")
winner_audit["match_id"] = pd.to_numeric(
    winner_audit["match_id"], errors="coerce"
)

if detail["match_id"].isna().any():
    stop("V3 detail contains missing or invalid match_id values.")

if winner_audit["match_id"].isna().any():
    stop("Winner audit contains missing or invalid match_id values.")

if detail["match_id"].duplicated().any():
    stop("V3 detail contains duplicate match_id values.")

if winner_audit["match_id"].duplicated().any():
    stop("Winner audit contains duplicate match_id values.")

print("V3 detail unique matches:", detail["match_id"].nunique())
print("Winner audit unique matches:", winner_audit["match_id"].nunique())

if len(detail) != EXPECTED_MATCH_COUNT:
    stop(
        f"Expected {EXPECTED_MATCH_COUNT} V3 matches, "
        f"found {len(detail)}. Check the input scope."
    )

if len(winner_audit) != EXPECTED_DEFENDED_AUDIT_COUNT:
    stop(
        f"Expected {EXPECTED_DEFENDED_AUDIT_COUNT} defended-winner audit "
        f"records, found {len(winner_audit)}."
    )


# ---------------------------------------------------------------------
# 6. PARSE BOOLEAN FIELDS
# ---------------------------------------------------------------------

detail_boolean_columns = [
    "target_reached_v3",
    "all_out_v3",
    "has_revised_target_v3",
    "revised_target_evidence_v3",
    "dls_or_duckworth_lewis_text_v3",
    "super_over_outcome_v3",
    "tie_score_conflict_v3",
    "no_result_v3",
    "no_second_innings_v3",
    "winner_validation_passed_v3",
    "validated_defended_v3",
]

winner_boolean_columns = [
    "team_data_consistent",
    "winner_is_second_innings_bowling_team",
    "winner_is_second_innings_batting_team",
]

detail = add_boolean_columns(detail, detail_boolean_columns)
winner_audit = add_boolean_columns(winner_audit, winner_boolean_columns)


# ---------------------------------------------------------------------
# 7. PREPARE WINNER AUDIT AND MERGE
# ---------------------------------------------------------------------

print_section("DEFENDED WINNER AUDIT MERGE")

# Keep only the winner-audit fields needed for eligibility verification.
# Rename overlapping fields before merging so no ambiguous suffixes occur.
winner_columns = [
    "match_id",
    "batting_team_id",
    "bowling_team_id",
    "batting_team_count",
    "bowling_team_count",
    "winner_team_id",
    "winner_team_count",
    "team_data_consistent",
    "winner_is_second_innings_bowling_team",
    "winner_is_second_innings_batting_team",
    "winner_validation_status",
]

winner_for_merge = winner_audit[winner_columns].copy()

winner_for_merge = winner_for_merge.rename(
    columns={
        "team_data_consistent": "team_data_consistent_audit",
        "winner_is_second_innings_bowling_team":
            "winner_is_second_innings_bowling_team_audit",
        "winner_is_second_innings_batting_team":
            "winner_is_second_innings_batting_team_audit",
        "winner_validation_status": "winner_validation_status_audit",
    }
)

detail = detail.merge(
    winner_for_merge,
    on="match_id",
    how="left",
    validate="one_to_one",
    indicator="_winner_merge_status",
)

print("Merge status:")
print(detail["_winner_merge_status"].value_counts().to_string())

matched_winner_rows = int(
    detail["_winner_merge_status"].eq("both").sum()
)

unmatched_winner_rows = int(
    detail["_winner_merge_status"].eq("left_only").sum()
)

if matched_winner_rows != EXPECTED_DEFENDED_AUDIT_COUNT:
    stop(
        "Winner-audit merge count mismatch: "
        f"expected {EXPECTED_DEFENDED_AUDIT_COUNT}, "
        f"found {matched_winner_rows}."
    )

if unmatched_winner_rows + matched_winner_rows != len(detail):
    stop("Winner-audit merge counts do not reconcile.")

detail["has_winner_audit_record"] = (
    detail["_winner_merge_status"].eq("both")
)

detail = detail.drop(columns=["_winner_merge_status"])


# ---------------------------------------------------------------------
# 8. CHECK CRITICAL BOOLEAN INPUT COMPLETENESS
# ---------------------------------------------------------------------

print_section("BOOLEAN INPUT COMPLETENESS")

# These flags apply to every match and must be populated.
# Winner-validation flags are checked separately only for defended matches,
# because non-defended matches may legitimately have no winner-audit record.
critical_boolean_columns = [
    "target_reached_v3",
    "all_out_v3",
    "has_revised_target_v3",
    "revised_target_evidence_v3",
    "dls_or_duckworth_lewis_text_v3",
    "super_over_outcome_v3",
    "tie_score_conflict_v3",
    "no_result_v3",
    "no_second_innings_v3",
]

detail["missing_eligibility_flags"] = detail[
    critical_boolean_columns
].isna().any(axis=1)

for column in critical_boolean_columns:
    print(f"{column}: missing={int(detail[column].isna().sum())}")

print(
    "Rows with at least one missing eligibility flag:",
    int(detail["missing_eligibility_flags"].sum()),
)


# ---------------------------------------------------------------------
# 9. BUILD EXCLUSION FLAGS
# ---------------------------------------------------------------------

print_section("ELIGIBILITY RULES")

category = (
    detail["chase_ending_category_v3"]
    .fillna("MISSING_CATEGORY")
    .astype(str)
    .str.strip()
    .str.upper()
)

detail["is_standard_target_reached"] = (
    category.eq("STANDARD_TARGET_REACHED")
)

detail["is_all_out_before_target"] = (
    category.eq("ALL_OUT_BEFORE_STANDARD_TARGET")
)

detail["is_overs_completed_defended"] = (
    category.eq("OVERS_COMPLETED_DEFENDED")
)

detail["is_revised_target_or_dls"] = (
    true_mask(detail["has_revised_target_v3"])
    | true_mask(detail["revised_target_evidence_v3"])
    | true_mask(detail["dls_or_duckworth_lewis_text_v3"])
    | detail["revised_target"].notna()
)

detail["is_super_over_or_eliminator"] = (
    true_mask(detail["super_over_outcome_v3"])
    | category.eq("SUPER_OVER_OR_ELIMINATOR")
    | category.eq("SUPER_OVER_RESULT_SCORE_CONFLICT_REVIEW")
)

detail["is_tie_score_conflict"] = (
    true_mask(detail["tie_score_conflict_v3"])
    | category.eq("SUPER_OVER_RESULT_SCORE_CONFLICT_REVIEW")
)

detail["is_no_result_or_draw"] = (
    true_mask(detail["no_result_v3"])
    | category.eq("NO_RESULT_OR_DRAW")
)

detail["is_missing_second_innings"] = (
    true_mask(detail["no_second_innings_v3"])
    | category.eq("NO_SECOND_INNINGS_DATA")
)

detail["is_unresolved_ending"] = (
    category.isin(
        {
            "OTHER_OR_REQUIRES_REVIEW",
            "REGULATION_SCORES_LEVEL_REVIEW",
        }
    )
)

detail["is_manual_review_category"] = category.isin(REVIEW_CATEGORIES)


# ---------------------------------------------------------------------
# 10. VALIDATE DEFENDED MATCHES
# ---------------------------------------------------------------------

print_section("DEFENDED MATCH VALIDATION")

detail["defended_winner_audit_passed"] = (
    true_mask(detail["winner_validation_passed_v3"])
    & true_mask(detail["validated_defended_v3"])
    & true_mask(detail["team_data_consistent_audit"])
    & true_mask(detail["winner_is_second_innings_bowling_team_audit"])
    & detail["has_winner_audit_record"]
)

# A defended record must have all winner-validation flags explicitly present.
detail["defended_validation_flags_complete"] = True
defended_rows = detail["is_overs_completed_defended"]
defended_validation_columns = [
    "winner_validation_passed_v3",
    "validated_defended_v3",
    "team_data_consistent_audit",
    "winner_is_second_innings_bowling_team_audit",
]
detail.loc[
    defended_rows, "defended_validation_flags_complete"
] = detail.loc[
    defended_rows, defended_validation_columns
].notna().all(axis=1)

detail["defended_validation_failed"] = (
    detail["is_overs_completed_defended"]
    & (
        ~detail["defended_winner_audit_passed"]
        | ~detail["defended_validation_flags_complete"]
    )
)

print(
    "V3 validated defended records:",
    int(true_mask(detail["validated_defended_v3"]).sum()),
)
print(
    "Defended category records:",
    int(detail["is_overs_completed_defended"].sum()),
)
print(
    "Defended winner audit passed:",
    int(detail["defended_winner_audit_passed"].sum()),
)
print(
    "Defended validation failures:",
    int(detail["defended_validation_failed"].sum()),
)


# ---------------------------------------------------------------------
# 11. DETERMINE PRIMARY MODEL ELIGIBILITY
# ---------------------------------------------------------------------

# Standard target reached and all-out endings can be included only when
# the match is not affected by revised-target/DLS evidence, Super Overs,
# no-result status, missing innings, or unresolved source flags.
standard_ending = (
    detail["is_standard_target_reached"]
    | detail["is_all_out_before_target"]
)

standard_ending_is_clean = (
    standard_ending
    & ~detail["is_revised_target_or_dls"]
    & ~detail["is_super_over_or_eliminator"]
    & ~detail["is_tie_score_conflict"]
    & ~detail["is_no_result_or_draw"]
    & ~detail["is_missing_second_innings"]
    & ~detail["missing_eligibility_flags"]
)

# Defended matches need the separate winner/team validation to pass.
defended_ending_is_clean = (
    detail["is_overs_completed_defended"]
    & detail["defended_winner_audit_passed"]
    & ~detail["is_revised_target_or_dls"]
    & ~detail["is_super_over_or_eliminator"]
    & ~detail["is_tie_score_conflict"]
    & ~detail["is_no_result_or_draw"]
    & ~detail["is_missing_second_innings"]
    & ~detail["missing_eligibility_flags"]
)

detail["eligible_for_primary_pressure_model"] = (
    standard_ending_is_clean | defended_ending_is_clean
)


# ---------------------------------------------------------------------
# 12. ASSIGN ELIGIBILITY REASONS
# ---------------------------------------------------------------------

def eligibility_reason(row) -> str:
    if row["missing_eligibility_flags"]:
        return "EXCLUDED_MISSING_ELIGIBILITY_FLAGS"

    if row["is_revised_target_or_dls"]:
        return "EXCLUDED_REVISED_TARGET_OR_DLS"

    if row["is_super_over_or_eliminator"]:
        return "EXCLUDED_SUPER_OVER_OR_ELIMINATOR"

    if row["is_tie_score_conflict"]:
        return "EXCLUDED_TIE_SCORE_CONFLICT"

    if row["is_no_result_or_draw"]:
        return "EXCLUDED_NO_RESULT_OR_DRAW"

    if row["is_missing_second_innings"]:
        return "EXCLUDED_NO_SECOND_INNINGS"

    if row["is_unresolved_ending"]:
        return "EXCLUDED_UNRESOLVED_CHASE_ENDING"

    if row["is_overs_completed_defended"]:
        if row["defended_winner_audit_passed"]:
            return "ELIGIBLE_VALIDATED_OVERS_COMPLETED_DEFENCE"
        return "EXCLUDED_DEFENDED_WINNER_VALIDATION_FAILED"

    if row["is_standard_target_reached"]:
        return "ELIGIBLE_STANDARD_TARGET_REACHED"

    if row["is_all_out_before_target"]:
        return "ELIGIBLE_ALL_OUT_BEFORE_STANDARD_TARGET"

    return "EXCLUDED_UNSUPPORTED_OR_UNCLASSIFIED_ENDING"


detail["pressure_eligibility_reason"] = detail.apply(
    eligibility_reason,
    axis=1,
)

detail["pressure_eligibility_status"] = (
    detail["eligible_for_primary_pressure_model"]
    .map({True: "ELIGIBLE", False: "EXCLUDED"})
)


# ---------------------------------------------------------------------
# 13. BUILD SUMMARY TABLES
# ---------------------------------------------------------------------

print_section("ELIGIBILITY SUMMARY")

total_matches = len(detail)
eligible_matches = int(
    detail["eligible_for_primary_pressure_model"].sum()
)
excluded_matches = total_matches - eligible_matches

summary = (
    detail.groupby(
        [
            "pressure_eligibility_status",
            "pressure_eligibility_reason",
            "chase_ending_category_v3",
        ],
        dropna=False,
    )
    .agg(
        match_count=("match_id", "nunique"),
        t20_matches=(
            "match_format",
            lambda s: int(s.astype(str).str.upper().eq("T20").sum()),
        ),
        t20i_matches=(
            "match_format",
            lambda s: int(s.astype(str).str.upper().eq("T20I").sum()),
        ),
    )
    .reset_index()
    .sort_values(
        ["pressure_eligibility_status", "match_count"],
        ascending=[True, False],
    )
)

print("\nEligibility status:")
print(
    detail["pressure_eligibility_status"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nEligibility reasons:")
print(
    detail["pressure_eligibility_reason"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nEnding categories:")
print(
    detail["chase_ending_category_v3"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nTOTAL MATCHES:", total_matches)
print("ELIGIBLE MATCHES:", eligible_matches)
print("EXCLUDED MATCHES:", excluded_matches)


# ---------------------------------------------------------------------
# 14. CREATE ELIGIBLE AND MANUAL-REVIEW DATASETS
# ---------------------------------------------------------------------

eligible = detail[
    detail["eligible_for_primary_pressure_model"]
].copy()

manual_review = detail[
    detail["is_manual_review_category"]
    | detail["missing_eligibility_flags"]
    | detail["defended_validation_failed"]
].copy()

# Keep the eligible-match file compact and convenient for the next stage.
eligible_columns = [
    "match_id",
    "match_date",
    "home_team",
    "away_team",
    "match_format",
    "season",
    "first_innings_innings_runs",
    "second_innings_innings_runs",
    "first_innings_legal_balls",
    "second_innings_legal_balls",
    "standard_chase_target",
    "chase_ending_category_v3",
    "pressure_eligibility_reason",
]

eligible_columns = [
    column for column in eligible_columns if column in eligible.columns
]

eligible_matches_output = eligible[eligible_columns].copy()


# ---------------------------------------------------------------------
# 15. VALIDATION CHECKS
# ---------------------------------------------------------------------

print_section("FINAL VALIDATION")

checks = {}

checks["total_match_count_837"] = len(detail) == EXPECTED_MATCH_COUNT

checks["unique_match_ids"] = (
    detail["match_id"].nunique() == len(detail)
)

checks["winner_audit_count_217"] = (
    matched_winner_rows == EXPECTED_DEFENDED_AUDIT_COUNT
)

checks["eligible_excluded_reconcile"] = (
    eligible_matches + excluded_matches == total_matches
)

checks["eligible_file_reconciles"] = (
    len(eligible_matches_output) == eligible_matches
)

checks["manual_review_subset"] = (
    set(manual_review["match_id"]).issubset(set(detail["match_id"]))
)

checks["all_defended_records_validated"] = (
    not detail["defended_validation_failed"].any()
)

checks["no_dls_or_revised_target_eligible"] = (
    not detail.loc[
        detail["eligible_for_primary_pressure_model"],
        "is_revised_target_or_dls",
    ].any()
)

checks["no_super_over_eligible"] = (
    not detail.loc[
        detail["eligible_for_primary_pressure_model"],
        "is_super_over_or_eliminator",
    ].any()
)

checks["no_unresolved_ending_eligible"] = (
    not detail.loc[
        detail["eligible_for_primary_pressure_model"],
        "is_unresolved_ending",
    ].any()
)

checks["no_missing_flags_eligible"] = (
    not detail.loc[
        detail["eligible_for_primary_pressure_model"],
        "missing_eligibility_flags",
    ].any()
)

for name, passed in checks.items():
    print(f"{name}: {'PASS' if passed else 'FAIL'}")

failed_checks = [name for name, passed in checks.items() if not passed]

if failed_checks:
    stop(f"Validation failed: {failed_checks}")


# ---------------------------------------------------------------------
# 16. SAVE OUTPUTS
# ---------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

detail.to_csv(DETAIL_OUTPUT_PATH, index=False)
summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)
eligible_matches_output.to_csv(ELIGIBLE_OUTPUT_PATH, index=False)
manual_review.to_csv(REVIEW_OUTPUT_PATH, index=False)

print_section("OUTPUT FILES")

print("Detailed eligibility audit:")
print(DETAIL_OUTPUT_PATH)

print("\nEligibility summary:")
print(SUMMARY_OUTPUT_PATH)

print("\nEligible matches for pressure modelling:")
print(ELIGIBLE_OUTPUT_PATH)

print("\nManual review records:")
print(REVIEW_OUTPUT_PATH)

print_section("AUDIT COMPLETED SUCCESSFULLY")
print(f"Total matches: {total_matches}")
print(f"Eligible matches: {eligible_matches}")
print(f"Excluded matches: {excluded_matches}")
print(f"Manual review records: {len(manual_review)}")
print(f"Validation checks passed: {sum(checks.values())}/{len(checks)}")
