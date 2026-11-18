
"""
RR GemHunters
Chase Completion Audit V2

Classifies all 837 regulation-innings matches using the actual
33-column chase_completion_detail.csv schema.

Inputs:
    data/interim/chase_completion_detail.csv
    data/interim/defended_innings_winner_audit.csv

Outputs:
    data/interim/chase_completion_detail_v2.csv
    data/interim/chase_completion_summary_v2.csv
    data/interim/chase_completion_manual_review_v2.csv

Important:
    - Super Overs are excluded from this primary audit.
    - Revised-target matches are not classified using standard targets.
    - Defended matches require independent winner validation.
    - Unresolved matches remain unresolved.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DETAIL_INPUT = (
    PROJECT_ROOT
    / "data/interim/chase_completion_detail.csv"
)

WINNER_INPUT = (
    PROJECT_ROOT
    / "data/interim/defended_innings_winner_audit.csv"
)

DETAIL_OUTPUT = (
    PROJECT_ROOT
    / "data/interim/chase_completion_detail_v2.csv"
)

SUMMARY_OUTPUT = (
    PROJECT_ROOT
    / "data/interim/chase_completion_summary_v2.csv"
)

REVIEW_OUTPUT = (
    PROJECT_ROOT
    / "data/interim/chase_completion_manual_review_v2.csv"
)

EXPECTED_MATCHES = 837
EXPECTED_WINNER_AUDIT_MATCHES = 217

STANDARD_T20_LEGAL_BALL_LIMIT = 120


# ============================================================
# 2. REQUIRED INPUT COLUMNS
# ============================================================

DETAIL_REQUIRED = [
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
    "first_innings_delivery_records",
    "second_innings_delivery_records",
    "second_innings_wicket_records",
    "second_innings_bowler_wickets",
    "second_innings_unknown_legal_ball_records",
    "second_innings_unknown_wicket_records",
    "standard_chase_target",
    "revised_target",
    "revised_target_difference",
    "revised_target_comparison",
    "target_validation_status",
    "standard_target_score_comparison",
    "revised_target_score_comparison",
    "result",
    "win_team_id",
    "result_mentions_dls",
    "result_mentions_no_result",
    "result_mentions_tie",
    "result_mentions_draw",
    "chase_ending_category",
    "result_score_consistency",
    "manual_review_reason",
    "review_status",
]

WINNER_REQUIRED = [
    "match_id",
    "winner_validation_status",
    "team_data_consistent",
    "winner_is_second_innings_bowling_team",
    "winner_is_second_innings_batting_team",
]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def to_bool(value):
    """Safely parse boolean values from CSV files."""

    if pd.isna(value):
        return False

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        return value == 1

    return str(value).strip().lower() in {
        "true", "1", "yes", "y"
    }


def bool_column(df, column):
    return df[column].apply(to_bool).astype(bool)


def numeric_column(df, column):
    return pd.to_numeric(df[column], errors="coerce")


def require_columns(df, required, file_name):

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{file_name} is missing columns:\n{missing}"
        )


# ============================================================
# 4. LOAD DATA
# ============================================================

def load_data():

    print_section("LOADING INPUT DATA")

    if not DETAIL_INPUT.exists():
        raise FileNotFoundError(
            f"Missing file: {DETAIL_INPUT}"
        )

    if not WINNER_INPUT.exists():
        raise FileNotFoundError(
            f"Missing file: {WINNER_INPUT}"
        )

    detail = pd.read_csv(DETAIL_INPUT)
    winner = pd.read_csv(WINNER_INPUT)

    print(f"Detail records: {len(detail):,}")
    print(f"Winner audit records: {len(winner):,}")

    return detail, winner


# ============================================================
# 5. VALIDATE INPUTS
# ============================================================

def validate_inputs(detail, winner):

    print_section("INPUT VALIDATION")

    require_columns(
        detail,
        DETAIL_REQUIRED,
        "chase_completion_detail.csv",
    )

    require_columns(
        winner,
        WINNER_REQUIRED,
        "defended_innings_winner_audit.csv",
    )

    detail = detail.copy()
    winner = winner.copy()

    detail["match_id"] = pd.to_numeric(
        detail["match_id"],
        errors="raise",
    ).astype("int64")

    winner["match_id"] = pd.to_numeric(
        winner["match_id"],
        errors="raise",
    ).astype("int64")

    if detail["match_id"].duplicated().any():
        raise ValueError(
            "Duplicate match IDs found in detail file."
        )

    if winner["match_id"].duplicated().any():
        raise ValueError(
            "Duplicate match IDs found in winner audit."
        )

    detail_ids = set(detail["match_id"])
    winner_ids = set(winner["match_id"])

    missing_ids = detail_ids - winner_ids
    extra_ids = winner_ids - detail_ids

    print(f"Detail matches: {len(detail):,}")
    print(f"Winner audit matches: {len(winner):,}")
    print(f"Matches missing from winner audit: {len(missing_ids):,}")
    print(f"Extra matches in winner audit: {len(extra_ids):,}")

    if len(detail) != EXPECTED_MATCHES:
        raise ValueError(
            f"Expected {EXPECTED_MATCHES} matches, "
            f"found {len(detail)}."
        )

    if len(winner) != EXPECTED_WINNER_AUDIT_MATCHES:
        raise ValueError(
            f"Expected {EXPECTED_WINNER_AUDIT_MATCHES} "
            f"winner audit records, found {len(winner)}."
        )

    if extra_ids:
        raise ValueError(
            "Winner audit contains match IDs not present "
            "in the detail file."
        )

    print(
        "Input validation passed. Winner audit coverage is "
        "limited to the 217 independently validated candidates."
    )

    return detail, winner


# ============================================================
# 6. MERGE WINNER AUDIT
# ============================================================

def merge_winner_audit(detail, winner):

    print_section("MERGING WINNER VALIDATION")

    audit_columns = [
        "match_id",
        "winner_validation_status",
        "team_data_consistent",
        "winner_is_second_innings_bowling_team",
        "winner_is_second_innings_batting_team",
    ]

    df = detail.merge(
        winner[audit_columns],
        on="match_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    matched = df["_merge"].eq("both").sum()
    unmatched = df["_merge"].eq("left_only").sum()

    df.drop(columns="_merge", inplace=True)

    print(f"Matched winner audit records: {matched:,}")
    print(f"Unmatched detail records: {unmatched:,}")

    if matched != EXPECTED_WINNER_AUDIT_MATCHES:
        raise ValueError(
            "Winner audit merge count does not equal 217."
        )

    return df


# ============================================================
# 7. BUILD CLASSIFICATION CONDITIONS
# ============================================================

def build_conditions(df):

    # Result flags
    no_result = (
        bool_column(df, "result_mentions_no_result")
        | bool_column(df, "result_mentions_draw")
    )

    dls_result = bool_column(
        df,
        "result_mentions_dls",
    )

    tie_result = bool_column(
        df,
        "result_mentions_tie",
    )

    # Innings availability
    second_delivery_records = numeric_column(
        df,
        "second_innings_delivery_records",
    )

    no_second_innings = (
        second_delivery_records.fillna(0).eq(0)
    )

    # Target information
    standard_target = numeric_column(
        df,
        "standard_chase_target",
    )

    revised_target = numeric_column(
        df,
        "revised_target",
    )

    has_standard_target = standard_target.notna()
    has_revised_target = revised_target.notna()

    first_score = numeric_column(
        df,
        "first_innings_innings_runs",
    )

    second_score = numeric_column(
        df,
        "second_innings_innings_runs",
    )

    # Use the actual comparison terminology from the source CSV.
    comparison = (
        df["standard_target_score_comparison"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # A chase reaches the standard target when the second innings
    # score is at least the standard chase target.
    target_reached = (
        has_standard_target
        & second_score.notna()
        & second_score.ge(standard_target)
    )

    # Also use the existing source comparison as corroborating evidence.
    target_reached = (
        target_reached
        | comparison.eq("SCORE_AT_OR_ABOVE_STANDARD_TARGET")
    )

    # Regulation scores level.
    tied_score = (
        first_score.notna()
        & second_score.notna()
        & first_score.eq(second_score)
    )

    tied_score = tied_score | tie_result

    # All out before reaching the target.
    second_wickets = numeric_column(
        df,
        "second_innings_wicket_records",
    )

    all_out = (
        second_wickets.ge(10)
        & ~target_reached
    )

    # Winner validation
    winner_status = (
        df["winner_validation_status"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    winner_validation_passed = (
        winner_status.eq("DEFENDING_TEAM_WON")
        & bool_column(df, "team_data_consistent")
        & bool_column(
            df,
            "winner_is_second_innings_bowling_team",
        )
    )

    # Completed 20 overs, using the audited legal-ball count.
    second_legal_balls = numeric_column(
        df,
        "second_innings_legal_balls",
    )

    completed_overs = second_legal_balls.eq(
        STANDARD_T20_LEGAL_BALL_LIMIT
    )

    # Strict defended-match classification.
    validated_defended = (
        winner_validation_passed
        & completed_overs
        & ~has_revised_target
        & ~dls_result
        & ~no_result
        & ~no_second_innings
        & ~all_out
        & ~target_reached
        & ~tied_score
    )

    return {
        "no_result": no_result,
        "no_second_innings": no_second_innings,
        "has_standard_target": has_standard_target,
        "has_revised_target": has_revised_target,
        "dls_result": dls_result,
        "target_reached": target_reached,
        "tied_score": tied_score,
        "all_out": all_out,
        "winner_validation_passed": winner_validation_passed,
        "completed_overs": completed_overs,
        "validated_defended": validated_defended,
    }


# ============================================================
# 8. CLASSIFY MATCHES
# ============================================================

def classify_matches(df):

    print_section("CLASSIFYING MATCHES")

    df = df.copy()

    conditions = build_conditions(df)

    ordered_conditions = [
        conditions["no_result"],
        conditions["no_second_innings"],
        conditions["has_revised_target"],
        conditions["target_reached"],
        conditions["tied_score"],
        conditions["all_out"],
        conditions["validated_defended"],
    ]

    category_names = [
        "NO_RESULT_OR_DRAW",
        "NO_SECOND_INNINGS_DATA",
        "REVISED_TARGET_REQUIRES_VALIDATION",
        "STANDARD_TARGET_REACHED",
        "LEVEL_WITH_FIRST_INNINGS_SCORE",
        "ALL_OUT_BEFORE_STANDARD_TARGET",
        "OVERS_COMPLETED_DEFENDED",
    ]

    condition_arrays = [
        condition.fillna(False).to_numpy(dtype=bool)
        for condition in ordered_conditions
    ]

    categories = category_names + [
        "OTHER_OR_REQUIRES_REVIEW"
    ]

    df["chase_ending_category_v2"] = pd.Categorical(
        np.select(
            condition_arrays,
            category_names,
            default="OTHER_OR_REQUIRES_REVIEW",
        ),
        categories=categories,
    )

    # Retain important audit flags.
    df["winner_validation_passed"] = (
        conditions["winner_validation_passed"]
    )

    df["validated_defended_match"] = (
        conditions["validated_defended"]
    )

    df["standard_target_reached_flag"] = (
        conditions["target_reached"]
    )

    df["all_out_before_target_flag"] = (
        conditions["all_out"]
    )

    df["revised_target_review_flag"] = (
        conditions["has_revised_target"]
    )

    df["dls_result_flag"] = (
        conditions["dls_result"]
    )

    df["no_result_flag"] = (
        conditions["no_result"]
    )

    df["second_innings_completed_120_balls"] = (
        conditions["completed_overs"]
    )

    return df


# ============================================================
# 9. MANUAL REVIEW REASONS
# ============================================================

def manual_review_reason(row):

    reasons = []

    category = str(row["chase_ending_category_v2"])

    if category == "NO_RESULT_OR_DRAW":
        reasons.append("NO_RESULT_OR_DRAW")

    if category == "NO_SECOND_INNINGS_DATA":
        reasons.append("SECOND_INNINGS_MISSING")

    if category == "REVISED_TARGET_REQUIRES_VALIDATION":
        reasons.append("REVISED_TARGET_REQUIRES_VALIDATION")

    if to_bool(row.get("result_mentions_dls")):
        reasons.append("RESULT_MENTIONS_DLS")

    if to_bool(row.get("result_mentions_tie")):
        reasons.append("RESULT_MENTIONS_TIE")

    if category == "LEVEL_WITH_FIRST_INNINGS_SCORE":
        reasons.append("REGULATION_SCORES_LEVEL")

    if category == "OTHER_OR_REQUIRES_REVIEW":
        reasons.append("CHASE_ENDING_UNRESOLVED")

    if (
        category == "OVERS_COMPLETED_DEFENDED"
        and not to_bool(row.get("team_data_consistent"))
    ):
        reasons.append("TEAM_DATA_INCONSISTENT")

    return " | ".join(dict.fromkeys(reasons))


def generate_manual_review(df):

    print_section("GENERATING MANUAL REVIEW")

    df = df.copy()

    df["manual_review_reason_v2"] = df.apply(
        manual_review_reason,
        axis=1,
    )

    df["review_status_v2"] = np.where(
        df["manual_review_reason_v2"].eq(""),
        "NO_AUTOMATIC_REVIEW_FLAG",
        "REVIEW_REQUIRED",
    )

    review = df[
        df["review_status_v2"].eq("REVIEW_REQUIRED")
    ].copy()

    print(f"Manual review records: {len(review):,}")

    return df, review


# ============================================================
# 10. GENERATE SUMMARY
# ============================================================

def generate_summary(df):

    print_section("CHASE ENDING DISTRIBUTION")

    summary = (
        df["chase_ending_category_v2"]
        .value_counts(dropna=False)
        .rename_axis("chase_ending_category")
        .reset_index(name="match_count")
    )

    summary["percentage"] = (
        summary["match_count"] / len(df) * 100
    ).round(2)

    summary = summary.sort_values(
        "match_count",
        ascending=False,
    ).reset_index(drop=True)

    for _, row in summary.iterrows():

        print(
            f"{row['chase_ending_category']:<43}"
            f"{row['match_count']:>5} "
            f"({row['percentage']:>6.2f}%)"
        )

    print("-" * 65)
    print(f"Total matches: {summary['match_count'].sum():,}")

    return summary


# ============================================================
# 11. FINAL VALIDATION
# ============================================================

def final_validation(df, summary, review):

    print_section("FINAL VALIDATION")

    checks = []

    # 1. Total matches
    checks.append((
        "Total match count equals 837",
        len(df) == EXPECTED_MATCHES,
    ))

    # 2. Unique match IDs
    checks.append((
        "No duplicate match IDs",
        df["match_id"].is_unique,
    ))

    # 3. No missing categories
    checks.append((
        "Every match has a classification",
        df["chase_ending_category_v2"].notna().all(),
    ))

    # 4. Summary reconciliation
    checks.append((
        "Summary total reconciles to detail",
        summary["match_count"].sum() == len(df),
    ))

    # 5. Defended matches
    defended = df[
        df["chase_ending_category_v2"].astype(str)
        == "OVERS_COMPLETED_DEFENDED"
    ]

    checks.append((
        "217 defended matches identified",
        len(defended) == EXPECTED_WINNER_AUDIT_MATCHES,
    ))

    # 6. Winner validation
    checks.append((
        "All defended matches passed winner validation",
        defended["winner_validation_passed"]
        .fillna(False)
        .astype(bool)
        .all(),
    ))

    # 7. Revised targets are not classified as standard chases
    revised_rows = df[
        df["revised_target_review_flag"].fillna(False).astype(bool)
    ]

    checks.append((
        "Revised-target matches excluded from standard classification",
        (
            ~revised_rows["chase_ending_category_v2"]
            .astype(str)
            .eq("STANDARD_TARGET_REACHED")
        ).all(),
    ))

    # 8. No revised-target match classified as defended
    checks.append((
        "No revised-target match classified as defended",
        (
            ~defended["revised_target_review_flag"]
            .fillna(False)
            .astype(bool)
        ).all(),
    ))

    # 9. Review subset
    checks.append((
        "Manual review records belong to detail dataset",
        set(review["match_id"]).issubset(set(df["match_id"])),
    ))

    passed = 0

    for description, result in checks:

        status = "PASS" if result else "FAIL"

        print(f"[{status}] {description}")

        if result:
            passed += 1

    print(f"\nValidation: {passed}/{len(checks)} checks passed")

    if passed != len(checks):

        raise AssertionError(
            "Final validation failed. "
            "Do not proceed until the failed checks are investigated."
        )


# ============================================================
# 12. SAVE OUTPUTS
# ============================================================

def save_outputs(df, summary, review):

    print_section("SAVING OUTPUT FILES")

    DETAIL_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        DETAIL_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    review.to_csv(
        REVIEW_OUTPUT,
        index=False,
    )

    print(f"Detail: {DETAIL_OUTPUT}")
    print(f"Summary: {SUMMARY_OUTPUT}")
    print(f"Manual review: {REVIEW_OUTPUT}")


# ============================================================
# 13. MAIN
# ============================================================

def main():

    print_section("RR GEMHUNTERS: CHASE COMPLETION AUDIT V2")

    detail, winner = load_data()

    detail, winner = validate_inputs(
        detail,
        winner,
    )

    df = merge_winner_audit(
        detail,
        winner,
    )

    df = classify_matches(df)

    df, review = generate_manual_review(df)

    summary = generate_summary(df)

    final_validation(
        df,
        summary,
        review,
    )

    save_outputs(
        df,
        summary,
        review,
    )

    print_section("AUDIT COMPLETED SUCCESSFULLY")

    defended_count = (
        df["chase_ending_category_v2"]
        .astype(str)
        .eq("OVERS_COMPLETED_DEFENDED")
        .sum()
    )

    print(f"Total matches: {len(df):,}")
    print(f"Validated defended matches: {defended_count:,}")
    print(f"Manual review records: {len(review):,}")

    print("\nNext step:")
    print("Inspect the V2 summary and manual review CSV files.")


if __name__ == "__main__":
    main()
