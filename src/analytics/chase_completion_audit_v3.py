"""
RR GemHunters
Chase Completion Audit V3

Corrected classification of 837 regulation-innings matches.

Inputs:
    data/interim/chase_completion_detail.csv
    data/interim/defended_innings_winner_audit.csv

Outputs:
    data/interim/chase_completion_detail_v3.csv
    data/interim/chase_completion_summary_v3.csv
    data/interim/chase_completion_manual_review_v3.csv

Methodological rules:
    - Super Overs and one-over eliminators are separated from regulation outcomes.
    - Tie-result / score conflicts are explicitly flagged.
    - DLS, D/L and Duckworth-Lewis result descriptions are treated as revised-target
      evidence, even if the revised_target column is empty.
    - Defended matches require the existing independent winner validation,
      120 legal balls, no revised-target evidence and no conflicting outcome.
    - Unresolved matches remain unresolved.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DETAIL_INPUT = PROJECT_ROOT / "data/interim/chase_completion_detail.csv"
WINNER_INPUT = PROJECT_ROOT / "data/interim/defended_innings_winner_audit.csv"

DETAIL_OUTPUT = PROJECT_ROOT / "data/interim/chase_completion_detail_v3.csv"
SUMMARY_OUTPUT = PROJECT_ROOT / "data/interim/chase_completion_summary_v3.csv"
REVIEW_OUTPUT = PROJECT_ROOT / "data/interim/chase_completion_manual_review_v3.csv"

EXPECTED_MATCHES = 837
EXPECTED_WINNER_AUDIT_MATCHES = 217
STANDARD_T20_LEGAL_BALL_LIMIT = 120


DETAIL_REQUIRED = [
    "match_id", "match_date", "home_team", "away_team", "match_format",
    "season", "first_innings_innings_runs", "second_innings_innings_runs",
    "first_innings_legal_balls", "second_innings_legal_balls",
    "first_innings_delivery_records", "second_innings_delivery_records",
    "second_innings_wicket_records", "second_innings_bowler_wickets",
    "second_innings_unknown_legal_ball_records",
    "second_innings_unknown_wicket_records", "standard_chase_target",
    "revised_target", "revised_target_difference", "revised_target_comparison",
    "target_validation_status", "standard_target_score_comparison",
    "revised_target_score_comparison", "result", "win_team_id",
    "result_mentions_dls", "result_mentions_no_result", "result_mentions_tie",
    "result_mentions_draw", "chase_ending_category", "result_score_consistency",
    "manual_review_reason", "review_status",
]

WINNER_REQUIRED = [
    "match_id", "winner_validation_status", "team_data_consistent",
    "winner_is_second_innings_bowling_team",
    "winner_is_second_innings_batting_team",
]


# ============================================================
# 2. HELPERS
# ============================================================

def print_section(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def to_bool(value):
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return value == 1
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def bool_column(df, column):
    return df[column].apply(to_bool).astype(bool)


def numeric_column(df, column):
    return pd.to_numeric(df[column], errors="coerce")


def require_columns(df, required, file_name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{file_name} is missing required columns:\n{missing}")


def normalize_text(series):
    return series.fillna("").astype(str).str.strip().str.lower()


# ============================================================
# 3. LOAD AND VALIDATE INPUTS
# ============================================================

def load_data():
    print_section("LOADING INPUT DATA")

    if not DETAIL_INPUT.exists():
        raise FileNotFoundError(f"Missing file: {DETAIL_INPUT}")
    if not WINNER_INPUT.exists():
        raise FileNotFoundError(f"Missing file: {WINNER_INPUT}")

    detail = pd.read_csv(DETAIL_INPUT, low_memory=False)
    winner = pd.read_csv(WINNER_INPUT, low_memory=False)

    print(f"Detail records: {len(detail):,}")
    print(f"Winner audit records: {len(winner):,}")
    return detail, winner


def validate_inputs(detail, winner):
    print_section("INPUT VALIDATION")

    require_columns(detail, DETAIL_REQUIRED, "chase_completion_detail.csv")
    require_columns(winner, WINNER_REQUIRED, "defended_innings_winner_audit.csv")

    detail = detail.copy()
    winner = winner.copy()

    detail["match_id"] = pd.to_numeric(detail["match_id"], errors="raise").astype("int64")
    winner["match_id"] = pd.to_numeric(winner["match_id"], errors="raise").astype("int64")

    if detail["match_id"].duplicated().any():
        raise ValueError("Duplicate match IDs in chase detail.")
    if winner["match_id"].duplicated().any():
        raise ValueError("Duplicate match IDs in winner audit.")
    if len(detail) != EXPECTED_MATCHES:
        raise ValueError(f"Expected {EXPECTED_MATCHES} detail matches, found {len(detail)}.")
    if len(winner) != EXPECTED_WINNER_AUDIT_MATCHES:
        raise ValueError(
            f"Expected {EXPECTED_WINNER_AUDIT_MATCHES} winner audit rows, "
            f"found {len(winner)}."
        )

    extra_ids = set(winner["match_id"]) - set(detail["match_id"])
    if extra_ids:
        raise ValueError(f"Winner audit contains unknown match IDs: {sorted(extra_ids)[:10]}")

    print(f"Detail matches: {len(detail):,}")
    print(f"Winner audit matches: {len(winner):,}")
    print(f"Matches without winner audit: {len(set(detail.match_id) - set(winner.match_id)):,}")
    print("Input validation passed.")
    return detail, winner


# ============================================================
# 4. MERGE INDEPENDENT WINNER VALIDATION
# ============================================================

def merge_winner_audit(detail, winner):
    print_section("MERGING WINNER VALIDATION")

    cols = [
        "match_id", "winner_validation_status", "team_data_consistent",
        "winner_is_second_innings_bowling_team",
        "winner_is_second_innings_batting_team",
    ]
    merged = detail.merge(
        winner[cols],
        on="match_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    matched = merged["_merge"].eq("both").sum()
    merged = merged.drop(columns=["_merge"])

    print(f"Matched winner audit records: {matched:,}")
    print(f"Unmatched detail records: {len(merged) - matched:,}")

    if matched != EXPECTED_WINNER_AUDIT_MATCHES:
        raise ValueError("Winner audit merge count does not equal 217.")
    return merged


# ============================================================
# 5. BUILD EVIDENCE FLAGS
# ============================================================

def build_conditions(df):
    result = normalize_text(df["result"])

    # Broader result-text detection than the original boolean source column.
    dls_pattern = re.compile(
        r"\b(dls|d\s*/\s*l|d\s*-\s*l|duckworth[\s-]*lewis)\b",
        flags=re.IGNORECASE,
    )
    super_over_pattern = re.compile(
        r"\b(super[\s-]*over|one[\s-]*over[\s-]*eliminator|eliminator)\b",
        flags=re.IGNORECASE,
    )

    dls_text = result.str.contains(dls_pattern, na=False)
    dls_flag = dls_text | bool_column(df, "result_mentions_dls")

    super_over = result.str.contains(super_over_pattern, na=False)
    no_result = (
        bool_column(df, "result_mentions_no_result")
        | bool_column(df, "result_mentions_draw")
        | result.str.contains(r"\bno\s*result\b|\babandoned\b|\bdraw\b", regex=True)
    )

    tie_text = bool_column(df, "result_mentions_tie") | result.str.contains(
        r"\bmatch\s*tied\b|\btied\s+with\b|\bmatch\s+tie\b", regex=True
    )

    first_score = numeric_column(df, "first_innings_innings_runs")
    second_score = numeric_column(df, "second_innings_innings_runs")
    standard_target = numeric_column(df, "standard_chase_target")
    revised_target = numeric_column(df, "revised_target")
    second_deliveries = numeric_column(df, "second_innings_delivery_records")
    second_legal_balls = numeric_column(df, "second_innings_legal_balls")
    second_wickets = numeric_column(df, "second_innings_wicket_records")

    has_second_innings = second_deliveries.fillna(0).gt(0)
    no_second_innings = ~has_second_innings
    has_revised_target = revised_target.notna()
    revised_target_evidence = has_revised_target | dls_flag

    score_level = (
        first_score.notna()
        & second_score.notna()
        & first_score.eq(second_score)
    )

    # A tie in result text with unequal reconstructed regulation scores is a
    # data/result conflict. Do not silently call it a regulation tie.
    tie_score_conflict = tie_text & ~score_level
    super_over_outcome = super_over
    regulation_level_review = score_level & ~super_over_outcome

    target_reached = (
        standard_target.notna()
        & second_score.notna()
        & second_score.ge(standard_target)
        & ~revised_target_evidence
    )

    # Retain the independently audited comparison as corroboration.
    source_comparison = (
        df["standard_target_score_comparison"]
        .fillna("").astype(str).str.strip().str.upper()
    )
    target_reached = target_reached | (
        source_comparison.eq("SCORE_AT_OR_ABOVE_STANDARD_TARGET")
        & ~revised_target_evidence
    )

    all_out = second_wickets.ge(10) & ~target_reached

    winner_status = df["winner_validation_status"].fillna("").astype(str).str.strip()
    winner_validation_passed = (
        winner_status.eq("DEFENDING_TEAM_WON")
        & bool_column(df, "team_data_consistent")
        & bool_column(df, "winner_is_second_innings_bowling_team")
    )

    completed_overs = second_legal_balls.eq(STANDARD_T20_LEGAL_BALL_LIMIT)

    validated_defended = (
        winner_validation_passed
        & completed_overs
        & ~revised_target_evidence
        & ~no_result
        & ~no_second_innings
        & ~all_out
        & ~target_reached
        & ~super_over_outcome
        & ~tie_score_conflict
        & ~regulation_level_review
    )

    return {
        "result_text_dls": dls_text,
        "dls_flag": dls_flag,
        "super_over_outcome": super_over_outcome,
        "tie_text": tie_text,
        "tie_score_conflict": tie_score_conflict,
        "regulation_level_review": regulation_level_review,
        "score_level": score_level,
        "no_result": no_result,
        "no_second_innings": no_second_innings,
        "has_revised_target": has_revised_target,
        "revised_target_evidence": revised_target_evidence,
        "target_reached": target_reached,
        "all_out": all_out,
        "winner_validation_passed": winner_validation_passed,
        "completed_overs": completed_overs,
        "validated_defended": validated_defended,
    }


# ============================================================
# 6. CLASSIFY MATCHES
# ============================================================

def classify_matches(df):
    print_section("CLASSIFYING MATCHES")
    df = df.copy()
    c = build_conditions(df)

    ordered = [
        c["no_result"],
        c["no_second_innings"],
        c["tie_score_conflict"],
        c["super_over_outcome"],
        c["revised_target_evidence"],
        c["target_reached"],
        c["regulation_level_review"],
        c["all_out"],
        c["validated_defended"],
    ]

    names = [
        "NO_RESULT_OR_DRAW",
        "NO_SECOND_INNINGS_DATA",
        "SUPER_OVER_RESULT_SCORE_CONFLICT_REVIEW",
        "SUPER_OVER_OR_ELIMINATOR",
        "REVISED_TARGET_REQUIRES_VALIDATION",
        "STANDARD_TARGET_REACHED",
        "REGULATION_SCORES_LEVEL_REVIEW",
        "ALL_OUT_BEFORE_STANDARD_TARGET",
        "OVERS_COMPLETED_DEFENDED",
    ]

    arrays = [x.fillna(False).to_numpy(dtype=bool) for x in ordered]
    categories = names + ["OTHER_OR_REQUIRES_REVIEW"]

    df["chase_ending_category_v3"] = pd.Categorical(
        np.select(arrays, names, default="OTHER_OR_REQUIRES_REVIEW"),
        categories=categories,
    )

    # Retain explicit flags for auditability.
    for key, value in c.items():
        df[f"{key}_v3"] = value.fillna(False).astype(bool)

    df["dls_or_duckworth_lewis_text_v3"] = c["result_text_dls"].astype(bool)
    return df


# ============================================================
# 7. MANUAL REVIEW
# ============================================================

def manual_review_reason(row):
    category = str(row["chase_ending_category_v3"])
    reasons = []

    if category == "NO_RESULT_OR_DRAW":
        reasons.append("NO_RESULT_OR_DRAW")
    elif category == "NO_SECOND_INNINGS_DATA":
        reasons.append("SECOND_INNINGS_MISSING")
    elif category == "SUPER_OVER_OR_ELIMINATOR":
        reasons.append("SUPER_OVER_OR_ONE_OVER_ELIMINATOR_EXCLUDED_FROM_PRIMARY_MODEL")
    elif category == "SUPER_OVER_RESULT_SCORE_CONFLICT_REVIEW":
        reasons.append("TIE_RESULT_CONFLICTS_WITH_RECONSTRUCTED_REGULATION_SCORE")
    elif category == "REVISED_TARGET_REQUIRES_VALIDATION":
        reasons.append("REVISED_TARGET_OR_DLS_RESULT_REQUIRES_VALIDATION")
    elif category == "REGULATION_SCORES_LEVEL_REVIEW":
        reasons.append("REGULATION_SCORES_LEVEL_RESULT_REQUIRES_REVIEW")
    elif category == "OTHER_OR_REQUIRES_REVIEW":
        reasons.append("CHASE_ENDING_UNRESOLVED")

    if to_bool(row.get("dls_flag_v3")) and "DLS_OR_DUCKWORTH_LEWIS_EVIDENCE" not in reasons:
        reasons.append("DLS_OR_DUCKWORTH_LEWIS_EVIDENCE")

    if category == "OVERS_COMPLETED_DEFENDED" and not to_bool(
        row.get("team_data_consistent")
    ):
        reasons.append("TEAM_DATA_INCONSISTENT")

    return " | ".join(dict.fromkeys(reasons))


def generate_manual_review(df):
    print_section("GENERATING MANUAL REVIEW")
    df = df.copy()
    df["manual_review_reason_v3"] = df.apply(manual_review_reason, axis=1)
    df["review_status_v3"] = np.where(
        df["manual_review_reason_v3"].eq(""),
        "NO_AUTOMATIC_REVIEW_FLAG",
        "REVIEW_REQUIRED",
    )
    review = df[df["review_status_v3"].eq("REVIEW_REQUIRED")].copy()
    print(f"Manual review records: {len(review):,}")
    return df, review


# ============================================================
# 8. SUMMARY AND VALIDATION
# ============================================================

def generate_summary(df):
    print_section("CHASE ENDING DISTRIBUTION")
    summary = (
        df["chase_ending_category_v3"]
        .value_counts(dropna=False)
        .rename_axis("chase_ending_category")
        .reset_index(name="match_count")
    )
    summary["percentage"] = (summary["match_count"] / len(df) * 100).round(2)
    summary = summary.sort_values("match_count", ascending=False).reset_index(drop=True)

    for _, row in summary.iterrows():
        print(
            f"{row['chase_ending_category']:<48}"
            f"{row['match_count']:>5} ({row['percentage']:>6.2f}%)"
        )
    print("-" * 68)
    print(f"Total matches: {summary['match_count'].sum():,}")
    return summary


def final_validation(df, summary, review):
    print_section("FINAL VALIDATION")
    defended = df[df["chase_ending_category_v3"].astype(str).eq("OVERS_COMPLETED_DEFENDED")]

    checks = [
        ("Total match count equals 837", len(df) == EXPECTED_MATCHES),
        ("No duplicate match IDs", df["match_id"].is_unique),
        ("Every match has a classification", df["chase_ending_category_v3"].notna().all()),
        ("Summary total reconciles to detail", summary["match_count"].sum() == len(df)),
        ("Exactly 217 defended matches retained", len(defended) == EXPECTED_WINNER_AUDIT_MATCHES),
        (
            "All defended matches passed winner validation",
            defended["winner_validation_passed_v3"].all(),
        ),
        (
            "No DLS/revised-target evidence classified as standard target reached",
            not (
                df["revised_target_evidence_v3"]
                & df["chase_ending_category_v3"].astype(str).eq("STANDARD_TARGET_REACHED")
            ).any(),
        ),
        (
            "Super Over outcomes are not classified as standard target reached",
            not (
                df["super_over_outcome_v3"]
                & df["chase_ending_category_v3"].astype(str).eq("STANDARD_TARGET_REACHED")
            ).any(),
        ),
        (
            "Tie-result score conflicts are explicitly flagged",
            not (
                df["tie_score_conflict_v3"]
                & ~df["chase_ending_category_v3"].astype(str).eq(
                    "SUPER_OVER_RESULT_SCORE_CONFLICT_REVIEW"
                )
            ).any(),
        ),
        ("Review subset belongs to detail dataset", set(review.match_id).issubset(set(df.match_id))),
    ]

    passed = 0
    for description, result in checks:
        print(f"[{'PASS' if result else 'FAIL'}] {description}")
        passed += int(bool(result))

    print(f"\nValidation: {passed}/{len(checks)} checks passed")
    if passed != len(checks):
        raise AssertionError("Final validation failed. Do not proceed.")
    return True


# ============================================================
# 9. SAVE OUTPUTS
# ============================================================

def save_outputs(df, summary, review):
    print_section("SAVING OUTPUT FILES")
    DETAIL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DETAIL_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    review.to_csv(REVIEW_OUTPUT, index=False)

    print(f"Detail: {DETAIL_OUTPUT}")
    print(f"Summary: {SUMMARY_OUTPUT}")
    print(f"Manual review: {REVIEW_OUTPUT}")


# ============================================================
# 10. MAIN
# ============================================================

def main():
    print_section("RR GEMHUNTERS: CHASE COMPLETION AUDIT V3")

    detail, winner = load_data()
    detail, winner = validate_inputs(detail, winner)
    df = merge_winner_audit(detail, winner)

    df = classify_matches(df)
    df, review = generate_manual_review(df)
    summary = generate_summary(df)

    final_validation(df, summary, review)
    save_outputs(df, summary, review)

    print_section("AUDIT COMPLETED SUCCESSFULLY")
    print(f"Total matches: {len(df):,}")
    print(
        "Validated defended matches: "
        f"{df['chase_ending_category_v3'].astype(str).eq('OVERS_COMPLETED_DEFENDED').sum():,}"
    )
    print(f"Manual review records: {len(review):,}")
    print("\nNext step: inspect the V3 summary and manual review CSV files.")


if __name__ == "__main__":
    main()
