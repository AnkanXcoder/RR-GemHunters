
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

DETAIL_FILE = OUTPUT_DIR / "target_semantics_detail.csv"
SUMMARY_FILE = OUTPUT_DIR / "target_semantics_summary.csv"
REVIEW_FILE = OUTPUT_DIR / "target_semantics_manual_review.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    if not MATCH_FILE.exists():
        raise FileNotFoundError(MATCH_FILE)

    if not DELIVERY_FILE.exists():
        raise FileNotFoundError(DELIVERY_FILE)

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
    ]

    for col in required_match_columns:
        if col not in matches.columns:
            raise ValueError(f"Missing match column: {col}")

    for col in required_delivery_columns:
        if col not in deliveries.columns:
            raise ValueError(f"Missing delivery column: {col}")

    return matches, deliveries


# ============================================================
# 3. NORMALIZE DATA TYPES
# ============================================================

def prepare_data(matches, deliveries):

    matches["match_id"] = pd.to_numeric(
        matches["match_id"], errors="coerce"
    )

    deliveries["match_id"] = pd.to_numeric(
        deliveries["match_id"], errors="coerce"
    )

    matches["revised_target"] = pd.to_numeric(
        matches["revised_target"], errors="coerce"
    )

    for col in [
        "first_inn_revised_overs",
        "second_inn_revised_overs",
    ]:
        matches[col] = pd.to_numeric(
            matches[col], errors="coerce"
        )

    for col in ["runs", "extras1_runs", "extras_2_runs"]:
        deliveries[col] = pd.to_numeric(
            deliveries[col], errors="coerce"
        ).fillna(0)

    deliveries["innings"] = pd.to_numeric(
        deliveries["innings"], errors="coerce"
    )

    return matches, deliveries


# ============================================================
# 4. RECONSTRUCT INNINGS SCORES
# ============================================================

def reconstruct_innings_scores(deliveries):

    deliveries["delivery_total_runs"] = (
        deliveries["runs"]
        + deliveries["extras1_runs"]
        + deliveries["extras_2_runs"]
    )

    innings_scores = (
        deliveries[
            deliveries["innings"].isin([1, 2])
        ]
        .groupby(["match_id", "innings"])
        .agg(
            innings_runs=("delivery_total_runs", "sum"),
            delivery_records=("delivery_total_runs", "size"),
        )
        .reset_index()
    )

    score_table = innings_scores.pivot(
        index="match_id",
        columns="innings",
        values="innings_runs",
    )

    score_table = score_table.rename(
        columns={
            1: "first_innings_runs",
            2: "second_innings_runs",
        }
    ).reset_index()

    count_table = innings_scores.pivot(
        index="match_id",
        columns="innings",
        values="delivery_records",
    )

    count_table = count_table.rename(
        columns={
            1: "first_innings_deliveries",
            2: "second_innings_deliveries",
        }
    ).reset_index()

    result = score_table.merge(
        count_table,
        on="match_id",
        how="left",
    )

    return result


# ============================================================
# 5. BUILD TARGET AUDIT
# ============================================================

def build_target_audit(matches, innings_scores):

    df = matches.merge(
        innings_scores,
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    # Standard target for an ordinary second innings.
    df["standard_chase_target"] = (
        df["first_innings_runs"] + 1
    )

    # Difference between recorded revised target and
    # reconstructed standard target.
    df["target_difference"] = (
        df["revised_target"]
        - df["standard_chase_target"]
    )

    df["has_revised_target"] = (
        df["revised_target"].notna()
    )

    df["has_first_revised_overs"] = (
        df["first_inn_revised_overs"].notna()
    )

    df["has_second_revised_overs"] = (
        df["second_inn_revised_overs"].notna()
    )

    df["has_any_revised_overs"] = (
        df["has_first_revised_overs"]
        | df["has_second_revised_overs"]
    )

    df["has_both_innings_scores"] = (
        df["first_innings_runs"].notna()
        & df["second_innings_runs"].notna()
    )

    df["target_comparison"] = "NO_RECORDED_TARGET"

    df.loc[
        df["has_revised_target"]
        & df["target_difference"].eq(0),
        "target_comparison"
    ] = "MATCHES_STANDARD_TARGET"

    df.loc[
        df["has_revised_target"]
        & df["target_difference"].ne(0),
        "target_comparison"
    ] = "DIFFERS_FROM_STANDARD_TARGET"

    df["target_status"] = "REVIEW"

    df.loc[
        ~df["has_revised_target"],
        "target_status"
    ] = "NO_REVISED_TARGET_RECORDED"

    df.loc[
        df["has_revised_target"]
        & df["target_difference"].eq(0),
        "target_status"
    ] = "TARGET_MATCHES_STANDARD"

    df.loc[
        df["has_revised_target"]
        & df["target_difference"].ne(0),
        "target_status"
    ] = "TARGET_REQUIRES_REVIEW"

    # Flag unusual target records.
    df["manual_review_reason"] = ""

    df.loc[
        df["has_revised_target"]
        & df["target_difference"].ne(0),
        "manual_review_reason"
    ] = "Recorded target differs from first innings score + 1"

    df.loc[
        df["has_revised_target"]
        & ~df["has_both_innings_scores"],
        "manual_review_reason"
    ] = "Revised target exists but both innings scores are unavailable"

    df.loc[
        df["has_any_revised_overs"],
        "manual_review_reason"
    ] = (
        df["manual_review_reason"]
        + " | Revised overs recorded"
    )

    return df


# ============================================================
# 6. SUMMARY
# ============================================================

def build_summary(df):

    summary_rows = []

    def add_metric(metric, value):
        summary_rows.append({
            "metric": metric,
            "value": value,
        })

    add_metric("total_matches", len(df))

    add_metric(
        "matches_with_first_innings_score",
        df["first_innings_runs"].notna().sum()
    )

    add_metric(
        "matches_with_second_innings_score",
        df["second_innings_runs"].notna().sum()
    )

    add_metric(
        "matches_with_revised_target",
        df["has_revised_target"].sum()
    )

    add_metric(
        "matches_with_first_revised_overs",
        df["has_first_revised_overs"].sum()
    )

    add_metric(
        "matches_with_second_revised_overs",
        df["has_second_revised_overs"].sum()
    )

    add_metric(
        "matches_with_both_revised_overs",
        (
            df["has_first_revised_overs"]
            & df["has_second_revised_overs"]
        ).sum()
    )

    add_metric(
        "target_matches_standard",
        (df["target_comparison"] == "MATCHES_STANDARD_TARGET").sum()
    )

    add_metric(
        "target_differs_from_standard",
        (df["target_comparison"] == "DIFFERS_FROM_STANDARD_TARGET").sum()
    )

    add_metric(
        "matches_requiring_manual_review",
        (df["manual_review_reason"] != "").sum()
    )

    return pd.DataFrame(summary_rows)


# ============================================================
# 7. MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("TARGET SEMANTICS AUDIT")
    print("=" * 65)

    matches, deliveries = load_data()

    print(f"\nRaw matches loaded: {len(matches):,}")
    print(f"Regulation deliveries loaded: {len(deliveries):,}")

    matches, deliveries = prepare_data(
        matches,
        deliveries
    )

    innings_scores = reconstruct_innings_scores(deliveries)

    print(
        f"\nMatches with reconstructed innings scores: "
        f"{len(innings_scores):,}"
    )

    audit = build_target_audit(
        matches,
        innings_scores
    )

    summary = build_summary(audit)

    review = audit[
        audit["manual_review_reason"] != ""
    ].copy()

    # Select useful columns for inspection.
    detail_columns = [
        "match_id",
        "match_date",
        "home_team",
        "away_team",
        "first_innings_runs",
        "second_innings_runs",
        "standard_chase_target",
        "revised_target",
        "target_difference",
        "first_inn_revised_overs",
        "second_inn_revised_overs",
        "has_both_innings_scores",
        "target_comparison",
        "target_status",
        "result",
        "manual_review_reason",
    ]

    detail = audit[detail_columns].copy()

    detail.to_csv(DETAIL_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)
    review[detail_columns].to_csv(REVIEW_FILE, index=False)

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n--- TARGET SUMMARY ---")
    print(summary.to_string(index=False))

    print("\n--- TARGET COMPARISON ---")
    print(
        audit["target_comparison"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n--- REVISED TARGET MATCHES ---")

    revised = audit[audit["has_revised_target"]]

    if revised.empty:
        print("No revised targets found.")
    else:
        print(
            revised[
                [
                    "match_id",
                    "first_innings_runs",
                    "standard_chase_target",
                    "revised_target",
                    "target_difference",
                    "first_inn_revised_overs",
                    "second_inn_revised_overs",
                ]
            ].to_string(index=False)
        )

    print("\n--- MANUAL REVIEW RECORDS ---")
    print(f"Records requiring review: {len(review):,}")

    print("\n--- OUTPUT FILES ---")
    print(DETAIL_FILE)
    print(SUMMARY_FILE)
    print(REVIEW_FILE)

    print("\nTarget semantics audit completed.")


if __name__ == "__main__":
    main()
