
from pathlib import Path
import pandas as pd

# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    ROOT / "data" / "raw" / "official"
    / "ball_by_ball.csv"
)

CLEAN_FILE = (
    ROOT / "data" / "processed" / "deliveries"
    / "regulation_deliveries.csv"
)

OUTPUT_DIR = ROOT / "data" / "interim"

DETAIL_FILE = OUTPUT_DIR / "extras_category_full_detail.csv"
SUMMARY_FILE = OUTPUT_DIR / "extras_category_full_summary.csv"
REVIEW_FILE = OUTPUT_DIR / "extras_category_manual_review.csv"

KEYS = ["match_id", "innings", "overid", "over_ball_num"]

FIELDS = [
    "extras_type",
    "runs",
    "extras1_runs",
    "extras_2_runs",
    "is_wicket",
    "dismissal_type",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_extras(series):
    return (
        series.fillna("none")
        .astype(str)
        .str.strip()
        .str.lower()
    )


def numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)


def print_section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print_section("FULL EXTRAS CATEGORY AUDIT")

    # --------------------------------------------------------
    # 1. Load datasets
    # --------------------------------------------------------

    raw = pd.read_csv(
        RAW_FILE,
        low_memory=False
    )

    clean = pd.read_csv(
        CLEAN_FILE,
        low_memory=False
    )

    print("Raw dataset:", raw.shape)
    print("Clean dataset:", clean.shape)

    # --------------------------------------------------------
    # 2. Validate columns
    # --------------------------------------------------------

    required = KEYS + FIELDS

    missing_raw = [
        col for col in required
        if col not in raw.columns
    ]

    missing_clean = [
        col for col in required
        if col not in clean.columns
    ]

    if missing_raw or missing_clean:
        raise ValueError(
            f"Missing raw columns: {missing_raw}\n"
            f"Missing clean columns: {missing_clean}"
        )

    # --------------------------------------------------------
    # 3. Normalize categories
    # --------------------------------------------------------

    raw["extras_type_normalized"] = normalize_extras(
        raw["extras_type"]
    )

    clean["extras_type_normalized"] = normalize_extras(
        clean["extras_type"]
    )

    # --------------------------------------------------------
    # 4. Convert numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "runs",
        "extras1_runs",
        "extras_2_runs",
    ]

    for col in numeric_columns:
        raw[col] = numeric(raw[col])
        clean[col] = numeric(clean[col])

    # --------------------------------------------------------
    # 5. Filter unusual categories
    # --------------------------------------------------------

    unusual_categories = ["overstep", "other"]

    unusual = clean[
        clean["extras_type_normalized"].isin(
            unusual_categories
        )
    ].copy()

    print_section("CATEGORY COUNTS")

    print(
        unusual["extras_type_normalized"]
        .value_counts()
        .to_string()
    )

    print("\nTotal unusual records:", len(unusual))

    # --------------------------------------------------------
    # 6. Match original records
    # --------------------------------------------------------

    raw_subset = raw[
        KEYS + FIELDS + ["extras_type_normalized"]
    ].copy()

    merged = unusual.merge(
        raw_subset,
        on=KEYS,
        how="left",
        suffixes=("_clean", "_raw"),
        validate="one_to_one",
        indicator=True,
    )

    # --------------------------------------------------------
    # 7. Compare original and clean values
    # --------------------------------------------------------

    merged["extras_type_matches"] = (
        merged["extras_type_normalized_clean"]
        == merged["extras_type_normalized_raw"]
    )

    for col in numeric_columns:

        merged[f"{col}_matches"] = (
            merged[f"{col}_clean"]
            == merged[f"{col}_raw"]
        )

    merged["is_wicket_matches"] = (
        merged["is_wicket_clean"].fillna("NA").astype(str)
        == merged["is_wicket_raw"].fillna("NA").astype(str)
    )

    merged["dismissal_type_matches"] = (
        merged["dismissal_type_clean"].fillna("NA").astype(str)
        == merged["dismissal_type_raw"].fillna("NA").astype(str)
    )

    # --------------------------------------------------------
    # 8. Provisional legal-ball classification
    # --------------------------------------------------------

    # Existing provisional policy:
    # wide, overstep and other are excluded from legal balls.

    merged["provisional_legal_ball"] = (
        ~merged["extras_type_normalized_clean"].isin(
            ["wide", "overstep", "other"]
        )
    )

    # --------------------------------------------------------
    # 9. Run relationships
    # --------------------------------------------------------

    merged["runs_without_extras2"] = (
        merged["runs_clean"]
        + merged["extras1_runs_clean"]
    )

    merged["runs_with_extras2"] = (
        merged["runs_clean"]
        + merged["extras1_runs_clean"]
        + merged["extras_2_runs_clean"]
    )

    # --------------------------------------------------------
    # 10. Category-level summary
    # --------------------------------------------------------

    print_section("RUN DISTRIBUTION")

    summary = (
        merged.groupby(
            "extras_type_normalized_clean",
            dropna=False
        )
        .agg(
            records=("match_id", "size"),
            batter_runs=("runs_clean", "sum"),
            extras1_runs=("extras1_runs_clean", "sum"),
            extras2_runs=("extras_2_runs_clean", "sum"),
            total_runs_without_extras2=(
                "runs_without_extras2", "sum"
            ),
            total_runs_with_extras2=(
                "runs_with_extras2", "sum"
            ),
            wicket_records=("is_wicket_clean", "sum"),
            legal_ball_records=(
                "provisional_legal_ball", "sum"
            ),
        )
        .reset_index()
    )

    print(summary.to_string(index=False))

    # --------------------------------------------------------
    # 11. Source comparison report
    # --------------------------------------------------------

    print_section("SOURCE COMPARISON")

    comparison_columns = [
        "extras_type_matches",
        "runs_matches",
        "extras1_runs_matches",
        "extras_2_runs_matches",
        "is_wicket_matches",
        "dismissal_type_matches",
    ]

    for col in comparison_columns:

        print(
            f"{col}:",
            int(merged[col].sum()),
            "/",
            len(merged)
        )

    print(
        "\nOriginal records matched:",
        int((merged["_merge"] == "both").sum())
    )

    print(
        "Original records unmatched:",
        int((merged["_merge"] != "both").sum())
    )

    # --------------------------------------------------------
    # 12. Identify records requiring manual review
    # --------------------------------------------------------

    merged["manual_review_required"] = (
        (merged["_merge"] != "both")
        | (~merged["extras_type_matches"])
        | (~merged["runs_matches"])
        | (~merged["extras1_runs_matches"])
        | (~merged["extras_2_runs_matches"])
        | (~merged["is_wicket_matches"])
        | (~merged["dismissal_type_matches"])
        | (merged["extras_2_runs_clean"] != 0)
        | (merged["extras_type_normalized_clean"] == "other")
    )

    review = merged[
        merged["manual_review_required"]
    ].copy()

    print_section("MANUAL REVIEW")

    print("Records requiring review:", len(review))

    print(
        "Records with nonzero extras_2_runs:",
        int((merged["extras_2_runs_clean"] != 0).sum())
    )

    print(
        "Other category records:",
        int(
            (
                merged["extras_type_normalized_clean"]
                == "other"
            ).sum()
        )
    )

    # --------------------------------------------------------
    # 13. Save outputs
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    merged.to_csv(
        DETAIL_FILE,
        index=False
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    review.to_csv(
        REVIEW_FILE,
        index=False
    )

    print_section("AUDIT OUTPUTS")

    print("Detailed report:", DETAIL_FILE)
    print("Summary report:", SUMMARY_FILE)
    print("Manual review report:", REVIEW_FILE)

    print_section("AUDIT COMPLETED")


if __name__ == "__main__":
    main()
