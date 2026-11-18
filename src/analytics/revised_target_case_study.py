
from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MATCH_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "official"
    / "matches.csv"
)

DELIVERY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "deliveries"
    / "regulation_deliveries.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"

DETAIL_FILE = OUTPUT_DIR / "revised_target_case_study.csv"
SUMMARY_FILE = OUTPUT_DIR / "revised_target_case_summary.csv"
REVIEW_FILE = OUTPUT_DIR / "revised_target_manual_review.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    matches = pd.read_csv(MATCH_FILE, low_memory=False)
    deliveries = pd.read_csv(DELIVERY_FILE, low_memory=False)

    required_match_columns = [
        "match_id",
        "match_date",
        "home_team",
        "away_team",
        "revised_target",
        "first_inn_revised_overs",
        "second_inn_revised_overs",
        "result",
    ]

    required_delivery_columns = [
        "match_id",
        "innings",
        "runs",
        "extras1_runs",
        "extras_2_runs",
        "legal_ball_provisional",
    ]

    for col in required_match_columns:
        if col not in matches.columns:
            raise ValueError(f"Missing match column: {col}")

    for col in required_delivery_columns:
        if col not in deliveries.columns:
            raise ValueError(f"Missing delivery column: {col}")

    return matches, deliveries


# ============================================================
# 3. PREPARE DATA
# ============================================================

def prepare_data(matches, deliveries):

    matches["match_id"] = pd.to_numeric(
        matches["match_id"], errors="coerce"
    )

    deliveries["match_id"] = pd.to_numeric(
        deliveries["match_id"], errors="coerce"
    )

    numeric_match_columns = [
        "revised_target",
        "first_inn_revised_overs",
        "second_inn_revised_overs",
    ]

    for col in numeric_match_columns:
        matches[col] = pd.to_numeric(
            matches[col], errors="coerce"
        )

    numeric_delivery_columns = [
        "runs",
        "extras1_runs",
        "extras_2_runs",
    ]

    for col in numeric_delivery_columns:
        deliveries[col] = pd.to_numeric(
            deliveries[col], errors="coerce"
        ).fillna(0)

    deliveries["innings"] = pd.to_numeric(
        deliveries["innings"], errors="coerce"
    )

    # Preserve nullable boolean values as missing where applicable.
    deliveries["legal_ball_provisional"] = (
        deliveries["legal_ball_provisional"]
        .astype("boolean")
    )

    return matches, deliveries


# ============================================================
# 4. RECONSTRUCT INNINGS METRICS
# ============================================================

def reconstruct_innings(deliveries):

    deliveries = deliveries.copy()

    deliveries["delivery_total_runs"] = (
        deliveries["runs"]
        + deliveries["extras1_runs"]
        + deliveries["extras_2_runs"]
    )

    innings = (
        deliveries[deliveries["innings"].isin([1, 2])]
        .groupby(["match_id", "innings"])
        .agg(
            innings_runs=("delivery_total_runs", "sum"),
            delivery_records=("delivery_total_runs", "size"),
            legal_balls=(
                "legal_ball_provisional",
                lambda x: x.eq(True).sum()
            ),
            unknown_legal_ball_records=(
                "legal_ball_provisional",
                lambda x: x.isna().sum()
            ),
        )
        .reset_index()
    )

    score_wide = innings.pivot(
        index="match_id",
        columns="innings",
        values="innings_runs",
    )

    score_wide = score_wide.rename(
        columns={
            1: "first_innings_runs",
            2: "second_innings_runs",
        }
    ).reset_index()

    balls_wide = innings.pivot(
        index="match_id",
        columns="innings",
        values="legal_balls",
    )

    balls_wide = balls_wide.rename(
        columns={
            1: "first_innings_legal_balls",
            2: "second_innings_legal_balls",
        }
    ).reset_index()

    delivery_wide = innings.pivot(
        index="match_id",
        columns="innings",
        values="delivery_records",
    )

    delivery_wide = delivery_wide.rename(
        columns={
            1: "first_innings_deliveries",
            2: "second_innings_deliveries",
        }
    ).reset_index()

    unknown_wide = innings.pivot(
        index="match_id",
        columns="innings",
        values="unknown_legal_ball_records",
    )

    unknown_wide = unknown_wide.rename(
        columns={
            1: "first_innings_unknown_legal_records",
            2: "second_innings_unknown_legal_records",
        }
    ).reset_index()

    result = score_wide.merge(
        balls_wide, on="match_id", how="left"
    )

    result = result.merge(
        delivery_wide, on="match_id", how="left"
    )

    result = result.merge(
        unknown_wide, on="match_id", how="left"
    )

    return result


# ============================================================
# 5. BUILD CASE STUDY
# ============================================================

def build_case_study(matches, innings_metrics):

    df = matches.merge(
        innings_metrics,
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    # Focus only on matches with recorded revised targets.
    df = df[df["revised_target"].notna()].copy()

    # Ordinary target for comparison.
    df["standard_chase_target"] = (
        df["first_innings_runs"] + 1
    )

    df["target_difference"] = (
        df["revised_target"]
        - df["standard_chase_target"]
    )

    df["target_matches_standard"] = (
        df["target_difference"] == 0
    )

    # Revised overs compared with actual legal balls.
    df["first_innings_revised_balls"] = (
        df["first_inn_revised_overs"] * 6
    )

    df["second_innings_revised_balls"] = (
        df["second_inn_revised_overs"] * 6
    )

    df["first_overs_difference"] = (
        df["first_innings_legal_balls"]
        - df["first_innings_revised_balls"]
    )

    df["second_overs_difference"] = (
        df["second_innings_legal_balls"]
        - df["second_innings_revised_balls"]
    )

    df["first_innings_overs_status"] = "NOT_RECORDED"
    df["second_innings_overs_status"] = "NOT_RECORDED"

    df.loc[
        df["first_inn_revised_overs"].notna()
        & df["first_overs_difference"].abs().le(1),
        "first_innings_overs_status"
    ] = "CLOSE_TO_REVISED_OVERS"

    df.loc[
        df["first_inn_revised_overs"].notna()
        & df["first_overs_difference"].abs().gt(1),
        "first_innings_overs_status"
    ] = "DIFFERS_FROM_REVISED_OVERS"

    df.loc[
        df["second_inn_revised_overs"].notna()
        & df["second_overs_difference"].abs().le(1),
        "second_innings_overs_status"
    ] = "CLOSE_TO_REVISED_OVERS"

    df.loc[
        df["second_inn_revised_overs"].notna()
        & df["second_overs_difference"].abs().gt(1),
        "second_innings_overs_status"
    ] = "DIFFERS_FROM_REVISED_OVERS"

    # Match result text is evidence for investigation,
    # not an automatic DLS classification.
    df["result_clean"] = (
        df["result"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["result_mentions_dls"] = (
        df["result_clean"].str.contains(
            r"\bdls\b|duckworth|lewis",
            regex=True,
            na=False,
        )
    )

    df["result_mentions_tie"] = (
        df["result_clean"].str.contains(
            r"\btie\b|super over",
            regex=True,
            na=False,
        )
    )

    df["result_mentions_abandoned"] = (
        df["result_clean"].str.contains(
            r"abandon|no result|without a result",
            regex=True,
            na=False,
        )
    )

    # --------------------------------------------------------
    # Manual review classification
    # --------------------------------------------------------

    reasons = []

    for _, row in df.iterrows():

        row_reasons = []

        if pd.notna(row["target_difference"]) and row["target_difference"] != 0:
            row_reasons.append(
                "Revised target differs from first innings score + 1"
            )

        if pd.isna(row["first_innings_runs"]):
            row_reasons.append(
                "First innings score unavailable"
            )

        if pd.isna(row["second_innings_runs"]):
            row_reasons.append(
                "Second innings score unavailable"
            )

        if pd.notna(row["first_inn_revised_overs"]):
            if row["first_innings_overs_status"] == "DIFFERS_FROM_REVISED_OVERS":
                row_reasons.append(
                    "First innings legal balls differ from revised overs"
                )

        if pd.notna(row["second_inn_revised_overs"]):
            if row["second_innings_overs_status"] == "DIFFERS_FROM_REVISED_OVERS":
                row_reasons.append(
                    "Second innings legal balls differ from revised overs"
                )

        if row["result_mentions_dls"]:
            row_reasons.append(
                "Result mentions DLS or Duckworth-Lewis"
            )

        if row["result_mentions_abandoned"]:
            row_reasons.append(
                "Result mentions abandoned or no result"
            )

        if row["result_mentions_tie"]:
            row_reasons.append(
                "Result mentions tie or Super Over"
            )

        if (
            row["first_innings_unknown_legal_records"] > 0
            or row["second_innings_unknown_legal_records"] > 0
        ):
            row_reasons.append(
                "Unknown legal-ball classification exists"
            )

        reasons.append(" | ".join(row_reasons))

    df["manual_review_reason"] = reasons

    df["case_study_status"] = np.where(
        df["manual_review_reason"].eq(""),
        "NO_AUTOMATIC_REVIEW_FLAG",
        "MANUAL_REVIEW_REQUIRED",
    )

    return df


# ============================================================
# 6. BUILD SUMMARY
# ============================================================

def build_summary(df):

    summary = [
        ("revised_target_matches", len(df)),
        (
            "targets_matching_standard",
            df["target_matches_standard"].sum(),
        ),
        (
            "targets_differing_from_standard",
            (~df["target_matches_standard"]).sum(),
        ),
        (
            "matches_mentioning_dls",
            df["result_mentions_dls"].sum(),
        ),
        (
            "matches_mentioning_tie_or_super_over",
            df["result_mentions_tie"].sum(),
        ),
        (
            "matches_mentioning_abandoned_or_no_result",
            df["result_mentions_abandoned"].sum(),
        ),
        (
            "matches_with_both_innings_scores",
            (
                df["first_innings_runs"].notna()
                & df["second_innings_runs"].notna()
            ).sum(),
        ),
        (
            "matches_requiring_manual_review",
            (
                df["case_study_status"]
                == "MANUAL_REVIEW_REQUIRED"
            ).sum(),
        ),
    ]

    return pd.DataFrame(
        summary,
        columns=["metric", "value"]
    )


# ============================================================
# 7. MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("REVISED TARGET CASE STUDY")
    print("=" * 65)

    matches, deliveries = load_data()

    print(f"\nMatches loaded: {len(matches):,}")
    print(f"Regulation deliveries: {len(deliveries):,}")

    matches, deliveries = prepare_data(
        matches,
        deliveries
    )

    innings_metrics = reconstruct_innings(deliveries)

    print(
        f"Matches with reconstructed innings metrics: "
        f"{len(innings_metrics):,}"
    )

    df = build_case_study(
        matches,
        innings_metrics
    )

    summary = build_summary(df)

    review = df[
        df["case_study_status"] == "MANUAL_REVIEW_REQUIRED"
    ].copy()

    output_columns = [
        "match_id",
        "match_date",
        "home_team",
        "away_team",
        "first_innings_runs",
        "second_innings_runs",
        "first_innings_legal_balls",
        "second_innings_legal_balls",
        "first_innings_deliveries",
        "second_innings_deliveries",
        "standard_chase_target",
        "revised_target",
        "target_difference",
        "first_inn_revised_overs",
        "second_inn_revised_overs",
        "first_overs_difference",
        "second_overs_difference",
        "first_innings_overs_status",
        "second_innings_overs_status",
        "result",
        "result_mentions_dls",
        "result_mentions_tie",
        "result_mentions_abandoned",
        "manual_review_reason",
        "case_study_status",
    ]

    df[output_columns].to_csv(
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

    print("\n--- CASE STUDY SUMMARY ---")
    print(summary.to_string(index=False))

    print("\n--- MATCH-LEVEL DETAILS ---")

    print(
        df[
            [
                "match_id",
                "first_innings_runs",
                "second_innings_runs",
                "revised_target",
                "target_difference",
                "first_innings_overs_status",
                "second_innings_overs_status",
                "result",
            ]
        ].to_string(index=False)
    )

    print("\n--- MANUAL REVIEW ---")
    print(f"Review records: {len(review):,}")

    print("\n--- OUTPUT FILES ---")
    print(DETAIL_FILE)
    print(SUMMARY_FILE)
    print(REVIEW_FILE)

    print("\nRevised target case study completed.")


if __name__ == "__main__":
    main()
