"""
Build pre-delivery pressure-state features for eligible second innings.

Inputs:
    data/processed/deliveries/regulation_deliveries.csv
    data/interim/pressure_model_eligible_matches.csv

Outputs:
    data/processed/features/pressure_state_deliveries.csv
    data/interim/pressure_state_feature_audit.csv

Important:
- Features describe the state BEFORE the current delivery.
- Team runs include runs + extras1_runs + extras_2_runs.
- legal_ball_provisional is a provisional legal-ball policy.
- All wicket records count as wickets lost for innings-state purposes.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DELIVERY_PATH = ROOT / "data/processed/deliveries/regulation_deliveries.csv"
ELIGIBLE_PATH = ROOT / "data/interim/pressure_model_eligible_matches.csv"

OUTPUT_DIR = ROOT / "data/processed/features"
AUDIT_DIR = ROOT / "data/interim"

OUTPUT_PATH = OUTPUT_DIR / "pressure_state_deliveries.csv"
AUDIT_PATH = AUDIT_DIR / "pressure_state_feature_audit.csv"

STANDARD_LEGAL_BALL_LIMIT = 120
EXPECTED_ELIGIBLE_MATCHES = 770


def require_columns(df, columns, label):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def main():
    print("=" * 72)
    print("BUILD PRE-DELIVERY PRESSURE-STATE FEATURES")
    print("=" * 72)

    if not DELIVERY_PATH.exists():
        raise FileNotFoundError(f"Delivery input not found: {DELIVERY_PATH}")
    if not ELIGIBLE_PATH.exists():
        raise FileNotFoundError(f"Eligible-match input not found: {ELIGIBLE_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    deliveries = pd.read_csv(DELIVERY_PATH, low_memory=False)
    eligible = pd.read_csv(ELIGIBLE_PATH, low_memory=False)

    delivery_required = [
        "match_id", "innings", "innings_ball_num", "runs",
        "extras1_runs", "extras_2_runs", "legal_ball_provisional",
        "is_wicket", "has_quality_flag", "striker_id", "striker_name",
        "non_striker_id", "non_striker_name", "bowler_id", "bowler_name",
        "comp_name", "match_format", "season", "match_date",
        "batting_team_id", "bowling_team_id"
    ]
    eligible_required = [
        "match_id", "standard_chase_target",
        "chase_ending_category_v3", "pressure_eligibility_reason"
    ]

    require_columns(deliveries, delivery_required, "Regulation deliveries")
    require_columns(eligible, eligible_required, "Eligible matches")

    # Match IDs must be unique in the eligibility file.
    if eligible["match_id"].duplicated().any():
        raise ValueError("Eligible match file contains duplicate match_id values.")

    eligible_ids = set(eligible["match_id"].dropna().astype("int64"))
    eligible_second = deliveries[
        deliveries["match_id"].isin(eligible_ids)
        & deliveries["innings"].eq(2)
    ].copy()

    found_ids = set(eligible_second["match_id"].dropna().astype("int64"))
    missing_ids = sorted(eligible_ids - found_ids)

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} eligible matches have no second-innings "
            f"delivery records. Examples: {missing_ids[:10]}"
        )

    if len(eligible_ids) != EXPECTED_ELIGIBLE_MATCHES:
        raise ValueError(
            f"Expected {EXPECTED_ELIGIBLE_MATCHES} eligible matches, "
            f"found {len(eligible_ids)}."
        )

    # Attach only the eligibility fields required for this feature dataset.
    eligibility_fields = eligible[
        [
            "match_id", "standard_chase_target",
            "chase_ending_category_v3", "pressure_eligibility_reason"
        ]
    ].copy()

    eligible_second = eligible_second.merge(
        eligibility_fields,
        on="match_id",
        how="left",
        validate="many_to_one"
    )

    if eligible_second["standard_chase_target"].isna().any():
        raise ValueError("Some eligible second-innings deliveries have no target.")

    # Convert calculation inputs to numeric values and reject missing values.
    numeric_columns = [
        "runs", "extras1_runs", "extras_2_runs",
        "legal_ball_provisional", "is_wicket", "has_quality_flag"
    ]
    for column in numeric_columns:
        eligible_second[column] = pd.to_numeric(
            eligible_second[column], errors="coerce"
        )

    for column in ["runs", "extras1_runs", "extras_2_runs"]:
        if eligible_second[column].isna().any():
            raise ValueError(f"Missing values found in required run column: {column}")

    for column in ["legal_ball_provisional", "is_wicket", "has_quality_flag"]:
        if eligible_second[column].isna().any():
            raise ValueError(f"Missing values found in required state column: {column}")

    # Sort by the source delivery sequence within each match.
    eligible_second = eligible_second.sort_values(
        ["match_id", "innings_ball_num"],
        kind="mergesort"
    ).reset_index(drop=True)

    duplicate_delivery_keys = eligible_second.duplicated(
        ["match_id", "innings", "innings_ball_num"]
    ).sum()
    if duplicate_delivery_keys:
        raise ValueError(
            f"Found {duplicate_delivery_keys} duplicate delivery sequence keys."
        )

    # Delivery outcome components.
    eligible_second["team_runs_this_delivery"] = (
        eligible_second["runs"]
        + eligible_second["extras1_runs"]
        + eligible_second["extras_2_runs"]
    )

    eligible_second["legal_ball_this_delivery"] = (
        eligible_second["legal_ball_provisional"].astype("int8")
    )
    eligible_second["wicket_this_delivery"] = (
        eligible_second["is_wicket"].astype("int8")
    )
    eligible_second["quality_flag_this_delivery"] = (
        eligible_second["has_quality_flag"].astype("int8")
    )

    grouped = eligible_second.groupby("match_id", sort=False)

    # Cumulative sums shifted by one row ensure the current delivery is excluded.
    eligible_second["score_before_ball"] = (
        grouped["team_runs_this_delivery"].cumsum()
        - eligible_second["team_runs_this_delivery"]
    )

    eligible_second["legal_balls_bowled_before"] = (
        grouped["legal_ball_this_delivery"].cumsum()
        - eligible_second["legal_ball_this_delivery"]
    )

    eligible_second["wickets_lost_before"] = (
        grouped["wicket_this_delivery"].cumsum()
        - eligible_second["wicket_this_delivery"]
    )

    eligible_second["prior_quality_flag_count"] = (
        grouped["quality_flag_this_delivery"].cumsum()
        - eligible_second["quality_flag_this_delivery"]
    )

    # Chase state before the current delivery.
    eligible_second["runs_required"] = (
        eligible_second["standard_chase_target"]
        - eligible_second["score_before_ball"]
    )

    eligible_second["balls_remaining"] = (
        STANDARD_LEGAL_BALL_LIMIT
        - eligible_second["legal_balls_bowled_before"]
    )

    eligible_second["wickets_remaining"] = (
        10 - eligible_second["wickets_lost_before"]
    )

    eligible_second["current_run_rate"] = np.where(
        eligible_second["legal_balls_bowled_before"] > 0,
        eligible_second["score_before_ball"] * 6
        / eligible_second["legal_balls_bowled_before"],
        np.nan
    )

    eligible_second["required_run_rate"] = np.where(
        eligible_second["balls_remaining"] > 0,
        eligible_second["runs_required"] * 6
        / eligible_second["balls_remaining"],
        np.nan
    )

    eligible_second["pressure_state_has_prior_quality_flag"] = (
        eligible_second["prior_quality_flag_count"] > 0
    )

    # Useful descriptive state labels. These are transparent rule-based buckets,
    # not a trained pressure score.
    eligible_second["runs_per_ball_required"] = np.where(
        eligible_second["balls_remaining"] > 0,
        eligible_second["runs_required"]
        / eligible_second["balls_remaining"],
        np.nan
    )

    eligible_second["required_rate_band"] = pd.cut(
        eligible_second["required_run_rate"],
        bins=[-np.inf, 6, 8, 10, 12, np.inf],
        labels=[
            "LOW_0_TO_6",
            "MODERATE_6_TO_8",
            "HIGH_8_TO_10",
            "VERY_HIGH_10_TO_12",
            "EXTREME_ABOVE_12"
        ],
        right=False
    ).astype("string")

    eligible_second["balls_remaining_band"] = pd.cut(
        eligible_second["balls_remaining"],
        bins=[-1, 6, 12, 30, 60, 120],
        labels=[
            "FINAL_1_TO_6",
            "FINAL_7_TO_12",
            "LATE_13_TO_30",
            "MIDDLE_31_TO_60",
            "EARLY_61_TO_120"
        ],
        right=True
    ).astype("string")

    eligible_second["wickets_remaining_band"] = pd.cut(
        eligible_second["wickets_remaining"],
        bins=[-1, 2, 5, 8, 10],
        labels=[
            "CRITICAL_0_TO_2",
            "LOW_3_TO_5",
            "MODERATE_6_TO_8",
            "HIGH_9_TO_10"
        ],
        right=True
    ).astype("string")

    # Keep a clear, useful set of columns while retaining delivery/player identity.
    output_columns = [
        "match_id", "match_date", "comp_name", "match_format", "season",
        "chase_ending_category_v3", "pressure_eligibility_reason",
        "innings", "innings_ball_num", "overid", "over_ball_num",
        "batting_team_id", "bowling_team_id",
        "striker_id", "striker_name", "non_striker_id", "non_striker_name",
        "bowler_id", "bowler_name",
        "standard_chase_target",
        "score_before_ball", "runs_required",
        "legal_balls_bowled_before", "balls_remaining",
        "wickets_lost_before", "wickets_remaining",
        "current_run_rate", "required_run_rate",
        "required_rate_band", "balls_remaining_band",
        "wickets_remaining_band",
        "runs", "extras1_runs", "extras_2_runs",
        "team_runs_this_delivery", "legal_ball_provisional",
        "is_wicket", "has_quality_flag",
        "prior_quality_flag_count",
        "pressure_state_has_prior_quality_flag"
    ]

    output = eligible_second[output_columns].copy()

    # Validation checks.
    checks = []

    def record_check(name, passed, detail):
        checks.append({
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "detail": str(detail)
        })

    record_check(
        "eligible_match_coverage",
        output["match_id"].nunique() == EXPECTED_ELIGIBLE_MATCHES,
        f"{output['match_id'].nunique()} unique matches"
    )
    record_check(
        "second_innings_only",
        output["innings"].eq(2).all(),
        f"Innings values: {sorted(output['innings'].unique().tolist())}"
    )
    record_check(
        "no_duplicate_delivery_keys",
        output.duplicated(["match_id", "innings", "innings_ball_num"]).sum() == 0,
        f"Duplicates: {output.duplicated(['match_id', 'innings', 'innings_ball_num']).sum()}"
    )
    record_check(
        "first_delivery_starts_at_zero",
        output.groupby("match_id")["score_before_ball"].first().eq(0).all(),
        "Every match starts with score_before_ball = 0"
    )
    record_check(
        "first_delivery_has_zero_balls_and_wickets",
        (
            output.groupby("match_id")["legal_balls_bowled_before"].first().eq(0).all()
            and output.groupby("match_id")["wickets_lost_before"].first().eq(0).all()
        ),
        "Every match starts with 0 legal balls and 0 wickets lost"
    )
    record_check(
        "no_negative_balls_remaining",
        output["balls_remaining"].ge(0).all(),
        f"Minimum balls remaining: {output['balls_remaining'].min()}"
    )
    record_check(
        "no_negative_wickets_remaining",
        output["wickets_remaining"].ge(0).all(),
        f"Minimum wickets remaining: {output['wickets_remaining'].min()}"
    )
    record_check(
        "target_attached",
        output["standard_chase_target"].notna().all(),
        f"Missing targets: {output['standard_chase_target'].isna().sum()}"
    )
    record_check(
        "pre_delivery_score_monotonic",
        output.groupby("match_id")["score_before_ball"].apply(
            lambda values: values.is_monotonic_increasing
        ).all(),
        "Pre-delivery score does not decrease within a match"
    )
    record_check(
        "pre_delivery_legal_balls_monotonic",
        output.groupby("match_id")["legal_balls_bowled_before"].apply(
            lambda values: values.is_monotonic_increasing
        ).all(),
        "Pre-delivery legal-ball count does not decrease within a match"
    )

    audit = pd.DataFrame(checks)

    print("\nFEATURE DATASET")
    print("Rows:", len(output))
    print("Unique matches:", output["match_id"].nunique())
    print("Unique batters:", output["striker_id"].nunique())
    print("Unique bowlers:", output["bowler_id"].nunique())
    print("Columns:", len(output.columns))
    print("Quality-flagged current deliveries:", int(output["has_quality_flag"].sum()))
    print(
        "Rows with a prior quality flag:",
        int(output["pressure_state_has_prior_quality_flag"].sum())
    )

    print("\nVALIDATION CHECKS")
    print(audit.to_string(index=False))

    if not audit["status"].eq("PASS").all():
        raise AssertionError("One or more pressure-state validation checks failed.")

    output.to_csv(OUTPUT_PATH, index=False)
    audit.to_csv(AUDIT_PATH, index=False)

    print("\nOUTPUT FILES")
    print(OUTPUT_PATH)
    print(AUDIT_PATH)
    print("\nSUCCESS: Pressure-state features created and all checks passed.")


if __name__ == "__main__":
    main()
