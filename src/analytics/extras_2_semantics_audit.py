
from pathlib import Path
import pandas as pd

# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    ROOT / "data" / "raw" / "official" / "ball_by_ball.csv"
)

CLEAN_FILE = (
    ROOT / "data" / "processed" / "deliveries"
    / "regulation_deliveries.csv"
)

OUTPUT_DIR = ROOT / "data" / "interim"

DETAIL_OUTPUT = OUTPUT_DIR / "extras_2_semantics_detail.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "extras_2_semantics_summary.csv"

KEYS = ["match_id", "innings", "overid", "over_ball_num"]

SOURCE_COLUMNS = [
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

def to_numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)


def normalize_extras(series):
    return (
        series.fillna("none")
        .astype(str)
        .str.strip()
        .str.lower()
    )


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print_section("EXTRAS_2_RUNS SEMANTICS AUDIT")

    # --------------------------------------------------------
    # 1. Load datasets
    # --------------------------------------------------------

    raw = pd.read_csv(RAW_FILE, low_memory=False)
    clean = pd.read_csv(CLEAN_FILE, low_memory=False)

    print("Raw dataset shape:", raw.shape)
    print("Clean dataset shape:", clean.shape)

    # --------------------------------------------------------
    # 2. Validate required columns
    # --------------------------------------------------------

    required_raw = KEYS + SOURCE_COLUMNS
    required_clean = KEYS + SOURCE_COLUMNS

    missing_raw = [
        col for col in required_raw if col not in raw.columns
    ]

    missing_clean = [
        col for col in required_clean if col not in clean.columns
    ]

    if missing_raw or missing_clean:
        raise ValueError(
            f"Missing raw columns: {missing_raw}\n"
            f"Missing clean columns: {missing_clean}"
        )

    # --------------------------------------------------------
    # 3. Convert numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "runs",
        "extras1_runs",
        "extras_2_runs",
    ]

    for col in numeric_columns:
        raw[col] = to_numeric(raw[col])
        clean[col] = to_numeric(clean[col])

    # --------------------------------------------------------
    # 4. Filter records with nonzero extras_2_runs
    # --------------------------------------------------------

    unusual = clean[
        clean["extras_2_runs"] != 0
    ].copy()

    print_section("NONZERO EXTRAS_2 RECORDS")

    print("Records investigated:", len(unusual))
    print(
        "Total extras_2_runs:",
        unusual["extras_2_runs"].sum()
    )

    # --------------------------------------------------------
    # 5. Match against original dataset
    # --------------------------------------------------------

    raw_subset = raw[
        KEYS + SOURCE_COLUMNS
    ].copy()

    merged = unusual.merge(
        raw_subset,
        on=KEYS,
        how="left",
        suffixes=("_clean", "_raw"),
        validate="one_to_one",
        indicator=True,
    )

    print_section("ORIGINAL DATA VALIDATION")

    print(
        "Original records matched:",
        int((merged["_merge"] == "both").sum())
    )

    print(
        "Original records not matched:",
        int((merged["_merge"] != "both").sum())
    )

    # --------------------------------------------------------
    # 6. Compare source and clean values
    # --------------------------------------------------------

    merged["extras_type_raw_normalized"] = normalize_extras(
        merged["extras_type_raw"]
    )

    merged["extras_type_clean_normalized"] = normalize_extras(
        merged["extras_type_clean"]
    )

    merged["extras_type_matches"] = (
        merged["extras_type_raw_normalized"]
        == merged["extras_type_clean_normalized"]
    )

    for col in numeric_columns:
        merged[f"{col}_matches"] = (
            merged[f"{col}_raw"] == merged[f"{col}_clean"]
        )

    merged["is_wicket_matches"] = (
        merged["is_wicket_raw"].fillna("NA").astype(str)
        == merged["is_wicket_clean"].fillna("NA").astype(str)
    )

    merged["dismissal_type_matches"] = (
        merged["dismissal_type_raw"].fillna("NA").astype(str)
        == merged["dismissal_type_clean"].fillna("NA").astype(str)
    )

    # --------------------------------------------------------
    # 7. Examine run relationships
    # --------------------------------------------------------

    merged["combined_runs_without_extras2"] = (
        merged["runs_clean"]
        + merged["extras1_runs_clean"]
    )

    merged["combined_runs_with_extras2"] = (
        merged["runs_clean"]
        + merged["extras1_runs_clean"]
        + merged["extras_2_runs_clean"]
    )

    # --------------------------------------------------------
    # 8. Detailed validation summary
    # --------------------------------------------------------

    print_section("FIELD COMPARISON")

    comparison_columns = [
        "extras_type",
        "runs",
        "extras1_runs",
        "extras_2_runs",
        "is_wicket",
        "dismissal_type",
    ]

    for col in comparison_columns:

        match_col = (
            "extras_type_matches"
            if col == "extras_type"
            else f"{col}_matches"
        )

        if match_col in merged.columns:

            print(
                f"{col}:",
                int(merged[match_col].sum()),
                "/",
                len(merged),
                "matched"
            )

    # --------------------------------------------------------
    # 9. Group by extras type
    # --------------------------------------------------------

    print_section("EXTRAS TYPE ANALYSIS")

    extras_summary = (
        merged.groupby(
            "extras_type_clean",
            dropna=False
        )
        .agg(
            records=("extras_2_runs_clean", "size"),
            total_extras2=("extras_2_runs_clean", "sum"),
            total_batter_runs=("runs_clean", "sum"),
            total_extras1=("extras1_runs_clean", "sum"),
            min_extras2=("extras_2_runs_clean", "min"),
            max_extras2=("extras_2_runs_clean", "max"),
        )
        .reset_index()
    )

    print(extras_summary.to_string(index=False))

    # --------------------------------------------------------
    # 10. Inspect individual records
    # --------------------------------------------------------

    print_section("DETAILED RECORD INSPECTION")

    detail_columns = [
        "match_id",
        "innings",
        "overid",
        "over_ball_num",
        "extras_type_clean",
        "runs_clean",
        "extras1_runs_clean",
        "extras_2_runs_clean",
        "combined_runs_without_extras2",
        "combined_runs_with_extras2",
        "is_wicket_clean",
        "dismissal_type_clean",
    ]

    print(
        merged[detail_columns].to_string(index=False)
    )

    # --------------------------------------------------------
    # 11. Build summary report
    # --------------------------------------------------------

    summary = pd.DataFrame({
        "metric": [
            "records_investigated",
            "total_extras_2_runs",
            "original_records_matched",
            "original_records_unmatched",
            "extras_type_matches",
            "runs_matches",
            "extras1_runs_matches",
            "extras_2_runs_matches",
            "is_wicket_matches",
            "dismissal_type_matches",
            "records_with_other_type",
            "records_with_overstep_type",
        ],

        "value": [
            len(merged),
            merged["extras_2_runs_clean"].sum(),
            (merged["_merge"] == "both").sum(),
            (merged["_merge"] != "both").sum(),
            merged["extras_type_matches"].sum(),
            merged["runs_matches"].sum(),
            merged["extras1_runs_matches"].sum(),
            merged["extras_2_runs_matches"].sum(),
            merged["is_wicket_matches"].sum(),
            merged["dismissal_type_matches"].sum(),
            (merged["extras_type_clean"] == "other").sum(),
            (merged["extras_type_clean"] == "overstep").sum(),
        ]
    })

    # --------------------------------------------------------
    # 12. Save audit outputs
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    merged.to_csv(DETAIL_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    print_section("AUDIT OUTPUTS")

    print("Detailed report:", DETAIL_OUTPUT)
    print("Summary report:", SUMMARY_OUTPUT)

    print_section("AUDIT COMPLETED")


if __name__ == "__main__":
    main()
