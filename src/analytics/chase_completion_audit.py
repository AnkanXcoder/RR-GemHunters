
from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "deliveries"
    / "regulation_deliveries.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"

DETAIL_FILE = OUTPUT_DIR / "chase_completion_detail.csv"
SUMMARY_FILE = OUTPUT_DIR / "chase_completion_summary.csv"
REVIEW_FILE = OUTPUT_DIR / "chase_completion_manual_review.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    required_columns = [
        "match_id",
        "innings",
        "runs",
        "extras1_runs",
        "extras_2_runs",
        "legal_ball_provisional",
        "is_wicket",
        "bowler_wicket_provisional",
        "revised_target",
        "first_inn_revised_overs",
        "second_inn_revised_overs",
        "result",
        "win_team_id",
        "match_date",
        "home_team",
        "away_team",
        "match_format",
        "season",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df


# ============================================================
# 3. DATA PREPARATION
# ============================================================

def prepare_data(df):

    df = df.copy()

    numeric_columns = [
        "match_id",
        "innings",
        "runs",
        "extras1_runs",
        "extras_2_runs",
        "revised_target",
        "first_inn_revised_overs",
        "second_inn_revised_overs",
        "win_team_id",
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    run_columns = [
        "runs",
        "extras1_runs",
        "extras_2_runs",
    ]

    missing_run_values = df[run_columns].isna().sum()

    print("\nMissing run-component values before filling:")
    print(missing_run_values)

    # Preserve source-quality information separately.
    df["legal_ball_provisional"] = (
        df["legal_ball_provisional"]
        .astype("boolean")
    )

    df["is_wicket"] = (
        df["is_wicket"]
        .astype("boolean")
    )

    df["bowler_wicket_provisional"] = (
        df["bowler_wicket_provisional"]
        .astype("boolean")
    )

    # Fill missing run components for arithmetic.
    df[run_columns] = df[run_columns].fillna(0)

    # Match result normalization.
    df["result_clean"] = (
        df["result"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    return df


# ============================================================
# 4. BUILD INNINGS METRICS
# ============================================================

def build_innings_metrics(df):

    df = df.copy()

    # Reconstruct total runs using the reconciled
    # delivery-level run components.
    df["delivery_total_runs"] = (
        df["runs"]
        + df["extras1_runs"]
        + df["extras_2_runs"]
    )

    regulation = df[
        df["innings"].isin([1, 2])
    ].copy()

    innings = (
        regulation
        .groupby(
            ["match_id", "innings"],
            dropna=False
        )
        .agg(
            innings_runs=(
                "delivery_total_runs",
                "sum"
            ),

            delivery_records=(
                "delivery_total_runs",
                "size"
            ),

            legal_balls=(
                "legal_ball_provisional",
                lambda x: x.eq(True).sum()
            ),

            unknown_legal_ball_records=(
                "legal_ball_provisional",
                lambda x: x.isna().sum()
            ),

            wicket_records=(
                "is_wicket",
                lambda x: x.eq(True).sum()
            ),

            unknown_wicket_records=(
                "is_wicket",
                lambda x: x.isna().sum()
            ),

            bowler_wickets=(
                "bowler_wicket_provisional",
                lambda x: x.eq(True).sum()
            ),

        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Pivot innings metrics into match-level columns
    # --------------------------------------------------------

    metrics = [
        "innings_runs",
        "delivery_records",
        "legal_balls",
        "unknown_legal_ball_records",
        "wicket_records",
        "unknown_wicket_records",
        "bowler_wickets",
    ]

    combined = None

    for metric in metrics:

        pivoted = innings.pivot(
            index="match_id",
            columns="innings",
            values=metric
        )

        pivoted = pivoted.rename(
            columns={
                1: f"first_innings_{metric}",
                2: f"second_innings_{metric}",
            }
        )

        pivoted = pivoted.reset_index()

        if combined is None:

            combined = pivoted

        else:

            combined = combined.merge(
                pivoted,
                on="match_id",
                how="outer",
                validate="one_to_one"
            )

    return combined


# ============================================================
# 5. BUILD MATCH-LEVEL METADATA
# ============================================================

def build_match_metadata(df):

    metadata_columns = [
        "match_id",
        "match_date",
        "home_team",
        "away_team",
        "win_team_id",
        "result",
        "result_clean",
        "revised_target",
        "first_inn_revised_overs",
        "second_inn_revised_overs",
        "match_format",
        "season",
    ]

    metadata = df[metadata_columns].copy()

    # Verify that match-level fields are consistent
    # across delivery records.
    check_columns = [
        col for col in metadata_columns
        if col != "match_id"
    ]

    inconsistencies = (
        metadata
        .groupby("match_id")[check_columns]
        .nunique(dropna=False)
    )

    inconsistent_matches = inconsistencies[
        inconsistencies.gt(1).any(axis=1)
    ]

    if len(inconsistent_matches) > 0:

        raise ValueError(
            "Match-level metadata is inconsistent across "
            f"{len(inconsistent_matches)} matches. "
            "Inspect the source before continuing."
        )

    metadata = metadata.drop_duplicates(
        subset=["match_id"]
    )

    return metadata


# ============================================================
# 6. RESULT TEXT CLASSIFICATION
# ============================================================

def classify_match_results(audit):

    result = audit["result_clean"]

    audit["result_mentions_dls"] = (
        result.str.contains(
            r"\bdls\b|duckworth|lewis",
            regex=True,
            na=False
        )
    )

    audit["result_mentions_no_result"] = (
        result.str.contains(
            r"no result|abandon",
            regex=True,
            na=False
        )
    )

    audit["result_mentions_tie"] = (
        result.str.contains(
            r"\btie\b|\btied\b|super over",
            regex=True,
            na=False
        )
    )

    audit["result_mentions_draw"] = (
        result.str.contains(
            r"\bdraw\b",
            regex=True,
            na=False
        )
    )

    return audit


# ============================================================
# 7. TARGET ANALYSIS
# ============================================================

def build_target_analysis(audit):

    audit = audit.copy()

    # Standard chase target:
    # First innings score + 1.
    audit["standard_chase_target"] = (
        audit["first_innings_innings_runs"] + 1
    )

    # Target availability.
    audit["has_recorded_revised_target"] = (
        audit["revised_target"].notna()
    )

    audit["has_standard_target"] = (
        audit["standard_chase_target"].notna()
    )

    # Difference is diagnostic only.
    audit["revised_target_difference"] = (
        audit["revised_target"]
        - audit["standard_chase_target"]
    )

    audit["revised_target_comparison"] = "NOT_RECORDED"

    audit.loc[
        audit["has_recorded_revised_target"]
        & audit["revised_target_difference"].eq(0),
        "revised_target_comparison"
    ] = "SAME_AS_STANDARD_TARGET"

    audit.loc[
        audit["has_recorded_revised_target"]
        & audit["revised_target_difference"].ne(0),
        "revised_target_comparison"
    ] = "DIFFERS_FROM_STANDARD_TARGET"

    # Do not automatically trust revised targets.
    audit["target_validation_status"] = np.where(
        audit["has_recorded_revised_target"],
        "UNVERIFIED_REVISED_TARGET",
        "STANDARD_TARGET_PROVISIONAL"
    )

    # Diagnostic score comparisons.
    audit["standard_target_score_comparison"] = "UNAVAILABLE"

    has_standard_comparison = (
        audit["standard_chase_target"].notna()
        & audit["second_innings_innings_runs"].notna()
    )

    audit.loc[
        has_standard_comparison
        & (
            audit["second_innings_innings_runs"]
            >= audit["standard_chase_target"]
        ),
        "standard_target_score_comparison"
    ] = "SCORE_AT_OR_ABOVE_STANDARD_TARGET"

    audit.loc[
        has_standard_comparison
        & (
            audit["second_innings_innings_runs"]
            == audit["standard_chase_target"] - 1
        ),
        "standard_target_score_comparison"
    ] = "LEVEL_WITH_FIRST_INNINGS_SCORE"

    audit.loc[
        has_standard_comparison
        & (
            audit["second_innings_innings_runs"]
            < audit["standard_chase_target"] - 1
        ),
        "standard_target_score_comparison"
    ] = "BELOW_STANDARD_TARGET"

    # Revised-target score comparison is descriptive only.
    audit["revised_target_score_comparison"] = "NOT_AVAILABLE"

    has_revised_comparison = (
        audit["revised_target"].notna()
        & audit["second_innings_innings_runs"].notna()
    )

    audit.loc[
        has_revised_comparison
        & (
            audit["second_innings_innings_runs"]
            >= audit["revised_target"]
        ),
        "revised_target_score_comparison"
    ] = "SCORE_AT_OR_ABOVE_RECORDED_TARGET"

    audit.loc[
        has_revised_comparison
        & (
            audit["second_innings_innings_runs"]
            < audit["revised_target"]
        ),
        "revised_target_score_comparison"
    ] = "BELOW_RECORDED_TARGET"

    return audit


# ============================================================
# 8. CHASE ENDING CLASSIFICATION
# ============================================================

def classify_chase_endings(audit):

    audit = audit.copy()

    # --------------------------------------------------------
    # Determine available information
    # --------------------------------------------------------

    no_second_innings = (
        audit["second_innings_innings_runs"].isna()
    )

    no_result = (
        audit["result_mentions_no_result"]
        | audit["result_mentions_draw"]
    )

    revised_target_exists = (
        audit["has_recorded_revised_target"]
    )

    second_innings_all_out = (
        audit["second_innings_wicket_records"]
        .ge(10)
        .fillna(False)
    )

    second_innings_missing_wicket_data = (
        audit["second_innings_unknown_wicket_records"]
        .gt(0)
        .fillna(False)
    )

    second_innings_unknown_legal_data = (
        audit["second_innings_unknown_legal_ball_records"]
        .gt(0)
        .fillna(False)
    )

    # --------------------------------------------------------
    # Standard-target comparison
    # --------------------------------------------------------

    standard_target_reached = (
        audit["second_innings_innings_runs"]
        >= audit["standard_chase_target"]
    ).fillna(False)

    standard_target_tied = (
        audit["second_innings_innings_runs"]
        == audit["standard_chase_target"] - 1
    ).fillna(False)

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    conditions = [
        # Priority 1: No result or abandoned.
        no_result,

        # Priority 2: Missing second innings.
        no_second_innings,

        # Priority 3: Revised target not independently validated.
        revised_target_exists,

        # Priority 4: Standard target reached.
        standard_target_reached,

        # Priority 5: Match tied on regulation score.
        standard_target_tied,

        # Priority 6: All out before reaching target.
        second_innings_all_out,

        # Priority 7: Remaining cases.
        second_innings_unknown_legal_data,
    ]

    choices = [
        "NO_RESULT_OR_ABANDONED",
        "NO_SECOND_INNINGS_DATA",
        "REVISED_TARGET_REQUIRES_VALIDATION",
        "STANDARD_TARGET_REACHED",
        "LEVEL_WITH_FIRST_INNINGS_SCORE",
        "ALL_OUT_BEFORE_STANDARD_TARGET",
        "UNKNOWN_LEGAL_BALL_DATA",
    ]

    audit["chase_ending_category"] = np.select(
        [
            condition.fillna(False).to_numpy(dtype=bool)
            for condition in conditions
        ],
        choices,
        default="INCOMPLETE_OR_UNRESOLVED",
    )

    # --------------------------------------------------------
    # Separate result text and score consistency
    # --------------------------------------------------------

    audit["result_score_consistency"] = "NOT_ASSESSED"

    standard_matches = (
        ~revised_target_exists
        & ~no_result
        & ~no_second_innings
    )

    audit.loc[
        standard_matches
        & standard_target_reached,
        "result_score_consistency"
    ] = "SCORE_REACHED_STANDARD_TARGET"

    audit.loc[
        standard_matches
        & standard_target_tied,
        "result_score_consistency"
    ] = "REGULATION_SCORE_LEVEL"

    audit.loc[
        standard_matches
        & ~standard_target_reached
        & ~standard_target_tied
        & second_innings_all_out,
        "result_score_consistency"
    ] = "ALL_OUT_BELOW_STANDARD_TARGET"

    audit.loc[
        revised_target_exists,
        "result_score_consistency"
    ] = "REVISED_TARGET_NOT_INDEPENDENTLY_VALIDATED"

    return audit


# ============================================================
# 9. MANUAL REVIEW FLAGS
# ============================================================

def build_review_flags(audit):

    review_reasons = []

    for _, row in audit.iterrows():

        reasons = []

        if row["result_mentions_dls"]:
            reasons.append("RESULT_MENTIONS_DLS")

        if row["result_mentions_no_result"]:
            reasons.append("NO_RESULT_OR_ABANDONED")

        if row["result_mentions_tie"]:
            reasons.append("TIE_OR_SUPER_OVER")

        if row["has_recorded_revised_target"]:
            reasons.append("REVISED_TARGET_REQUIRES_VALIDATION")

        if pd.isna(row["second_innings_innings_runs"]):
            reasons.append("SECOND_INNINGS_MISSING")

        if (
            pd.notna(row["second_innings_unknown_legal_ball_records"])
            and row["second_innings_unknown_legal_ball_records"] > 0
        ):
            reasons.append("UNKNOWN_LEGAL_BALL_RECORDS")

        if (
            pd.notna(row["second_innings_unknown_wicket_records"])
            and row["second_innings_unknown_wicket_records"] > 0
        ):
            reasons.append("UNKNOWN_WICKET_RECORDS")

        if (
            row["chase_ending_category"]
            == "INCOMPLETE_OR_UNRESOLVED"
        ):
            reasons.append("CHASE_ENDING_UNRESOLVED")

        if (
            row["chase_ending_category"]
            == "UNKNOWN_LEGAL_BALL_DATA"
        ):
            reasons.append("LEGAL_BALL_DATA_REQUIRES_REVIEW")

        review_reasons.append(
            " | ".join(reasons)
        )

    audit["manual_review_reason"] = review_reasons

    audit["review_status"] = np.where(
        audit["manual_review_reason"].eq(""),
        "NO_AUTOMATIC_REVIEW_FLAG",
        "REVIEW_REQUIRED"
    )

    return audit


# ============================================================
# 10. BUILD SUMMARY
# ============================================================

def build_summary(audit):

    rows = [

        ("total_matches", len(audit)),

        (
            "matches_with_first_innings",
            audit["first_innings_innings_runs"].notna().sum()
        ),

        (
            "matches_with_second_innings",
            audit["second_innings_innings_runs"].notna().sum()
        ),

        (
            "matches_without_second_innings",
            audit["second_innings_innings_runs"].isna().sum()
        ),

        (
            "matches_with_revised_target",
            audit["has_recorded_revised_target"].sum()
        ),

        (
            "matches_with_dls_in_result",
            audit["result_mentions_dls"].sum()
        ),

        (
            "matches_with_no_result",
            audit["result_mentions_no_result"].sum()
        ),

        (
            "matches_with_tie_or_super_over_text",
            audit["result_mentions_tie"].sum()
        ),

        (
            "standard_target_reached",
            (
                audit["chase_ending_category"]
                == "STANDARD_TARGET_REACHED"
            ).sum()
        ),

        (
            "level_with_first_innings_score",
            (
                audit["chase_ending_category"]
                == "LEVEL_WITH_FIRST_INNINGS_SCORE"
            ).sum()
        ),

        (
            "all_out_before_standard_target",
            (
                audit["chase_ending_category"]
                == "ALL_OUT_BEFORE_STANDARD_TARGET"
            ).sum()
        ),

        (
            "revised_target_requires_validation",
            (
                audit["chase_ending_category"]
                == "REVISED_TARGET_REQUIRES_VALIDATION"
            ).sum()
        ),

        (
            "incomplete_or_unresolved",
            (
                audit["chase_ending_category"]
                == "INCOMPLETE_OR_UNRESOLVED"
            ).sum()
        ),

        (
            "matches_requiring_manual_review",
            (
                audit["review_status"] == "REVIEW_REQUIRED"
            ).sum()
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=["metric", "value"]
    )


# ============================================================
# 11. MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("CHASE COMPLETION AUDIT")
    print("=" * 65)

    # Load and prepare.
    df = load_data()

    print(f"\nDelivery rows loaded: {len(df):,}")
    print(f"Unique matches: {df['match_id'].nunique():,}")

    df = prepare_data(df)

    # Build innings-level metrics.
    innings_metrics = build_innings_metrics(df)

    print(
        f"Matches with innings metrics: "
        f"{len(innings_metrics):,}"
    )

    # Build match-level metadata.
    metadata = build_match_metadata(df)

    # Merge match metadata and innings metrics.
    audit = metadata.merge(
        innings_metrics,
        on="match_id",
        how="left",
        validate="one_to_one"
    )

    # Classify results.
    audit = classify_match_results(audit)

    # Analyze targets.
    audit = build_target_analysis(audit)

    # Classify chase endings.
    audit = classify_chase_endings(audit)

    # Build manual review flags.
    audit = build_review_flags(audit)

    # Build summary.
    summary = build_summary(audit)

    review = audit[
        audit["review_status"] == "REVIEW_REQUIRED"
    ].copy()

    # --------------------------------------------------------
    # OUTPUT COLUMNS
    # --------------------------------------------------------

    output_columns = [

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

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    audit[output_columns].to_csv(
        DETAIL_FILE,
        index=False
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    review[output_columns].to_csv(
        REVIEW_FILE,
        index=False
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n--- CHASE COMPLETION SUMMARY ---")

    print(
        summary.to_string(index=False)
    )

    print("\n--- CHASE ENDING DISTRIBUTION ---")

    print(
        audit["chase_ending_category"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n--- TARGET VALIDATION STATUS ---")

    print(
        audit["target_validation_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n--- RESULT CLASSIFICATION ---")

    print(
        audit["result_mentions_dls"]
        .value_counts()
        .rename(
            {
                True: "DLS mentioned",
                False: "DLS not mentioned"
            }
        )
        .to_string()
    )

    print("\n--- MANUAL REVIEW ---")

    print(
        f"Records requiring review: {len(review):,}"
    )

    print("\n--- OUTPUT FILES ---")

    print(DETAIL_FILE)
    print(SUMMARY_FILE)
    print(REVIEW_FILE)

    print("\nChase completion audit completed successfully.")


if __name__ == "__main__":
    main()
