
from pathlib import Path
import pandas as pd

# ============================================================
# 1. PATH CONFIGURATION
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

DETAIL_FILE = OUTPUT_DIR / "retireout_detail.csv"
SUMMARY_FILE = OUTPUT_DIR / "retireout_summary.csv"
REVIEW_FILE = OUTPUT_DIR / "retireout_manual_review.csv"

KEYS = [
    "match_id",
    "innings",
    "overid",
    "over_ball_num",
]

COMPARE_COLUMNS = [
    "dismissal_type",
    "is_wicket",
    "dismissed_player_id",
    "striker_id",
    "non_striker_id",
    "bowler_id",
    "runs",
    "extras1_runs",
    "extras_2_runs",
    "extras_type",
]

ID_COLUMNS = [
    "dismissed_player_id",
    "striker_id",
    "non_striker_id",
    "bowler_id",
]

NUMERIC_COLUMNS = [
    "runs",
    "extras1_runs",
    "extras_2_runs",
]


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def normalize_id(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    ).astype("Int64")


def normalize_boolean(series):

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "1.0"])
    )


def print_section(title):

    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


# ============================================================
# 3. MAIN AUDIT
# ============================================================

def main():

    print_section("RETIRE OUT RECORD AUDIT")

    # --------------------------------------------------------
    # STEP 1: LOAD DATASETS
    # --------------------------------------------------------

    raw = pd.read_csv(
        RAW_FILE,
        low_memory=False
    )

    clean = pd.read_csv(
        CLEAN_FILE,
        low_memory=False
    )

    print("Raw dataset shape:", raw.shape)
    print("Clean dataset shape:", clean.shape)

    # --------------------------------------------------------
    # STEP 2: VALIDATE REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = KEYS + COMPARE_COLUMNS

    missing_raw = [
        col for col in required_columns
        if col not in raw.columns
    ]

    missing_clean = [
        col for col in required_columns
        if col not in clean.columns
    ]

    if missing_raw or missing_clean:

        raise ValueError(
            f"Missing raw columns: {missing_raw}\n"
            f"Missing clean columns: {missing_clean}"
        )

    # --------------------------------------------------------
    # STEP 3: NORMALIZE DISMISSAL TYPES
    # --------------------------------------------------------

    raw["dismissal_normalized"] = (
        raw["dismissal_type"].apply(normalize_text)
    )

    clean["dismissal_normalized"] = (
        clean["dismissal_type"].apply(normalize_text)
    )

    # --------------------------------------------------------
    # STEP 4: FILTER RETIRE OUT RECORDS
    # --------------------------------------------------------

    raw_retire = raw[
        raw["dismissal_normalized"] == "retireout"
    ].copy()

    clean_retire = clean[
        clean["dismissal_normalized"] == "retireout"
    ].copy()

    print_section("RETIRE OUT COUNTS")

    print(
        "Original Retire Out records:",
        len(raw_retire)
    )

    print(
        "Cleaned Retire Out records:",
        len(clean_retire)
    )

    # --------------------------------------------------------
    # STEP 5: MATCH ORIGINAL AND CLEANED RECORDS
    # --------------------------------------------------------

    raw_subset = raw_retire[
        KEYS + COMPARE_COLUMNS
    ].copy()

    merged = clean_retire.merge(
        raw_subset,
        on=KEYS,
        how="left",
        suffixes=("_clean", "_raw"),
        validate="one_to_one",
        indicator=True
    )

    print_section("SOURCE MATCHING")

    matched_count = int(
        (merged["_merge"] == "both").sum()
    )

    unmatched_count = int(
        (merged["_merge"] != "both").sum()
    )

    print("Original records matched:", matched_count)
    print("Original records unmatched:", unmatched_count)

    # --------------------------------------------------------
    # STEP 6: NORMALIZE PLAYER IDS
    # --------------------------------------------------------

    for col in ID_COLUMNS:

        merged[f"{col}_clean_normalized"] = normalize_id(
            merged[f"{col}_clean"]
        )

        merged[f"{col}_raw_normalized"] = normalize_id(
            merged[f"{col}_raw"]
        )

        merged[f"{col}_matches"] = (
            merged[f"{col}_clean_normalized"]
            == merged[f"{col}_raw_normalized"]
        )

    # --------------------------------------------------------
    # STEP 7: COMPARE DISMISSAL TYPES
    # --------------------------------------------------------

    # The merged column is named dismissal_normalized
    # because only the cleaned dataset contains that
    # pre-existing normalized column.

    merged["dismissal_type_matches"] = (
        merged["dismissal_normalized"]
        == merged["dismissal_type_raw"].apply(normalize_text)
    )

    # --------------------------------------------------------
    # STEP 8: COMPARE NUMERIC VALUES
    # --------------------------------------------------------

    for col in NUMERIC_COLUMNS:

        clean_values = pd.to_numeric(
            merged[f"{col}_clean"],
            errors="coerce"
        )

        raw_values = pd.to_numeric(
            merged[f"{col}_raw"],
            errors="coerce"
        )

        merged[f"{col}_matches"] = (
            clean_values.eq(raw_values)
            | (
                clean_values.isna()
                & raw_values.isna()
            )
        )

    # --------------------------------------------------------
    # STEP 9: COMPARE WICKET INDICATORS
    # --------------------------------------------------------

    merged["is_wicket_matches"] = (
        normalize_boolean(merged["is_wicket_clean"])
        == normalize_boolean(merged["is_wicket_raw"])
    )

    # --------------------------------------------------------
    # STEP 10: COMPARE EXTRAS TYPES
    # --------------------------------------------------------

    merged["extras_type_matches"] = (
        merged["extras_type_clean"].apply(normalize_text)
        == merged["extras_type_raw"].apply(normalize_text)
    )

    # --------------------------------------------------------
    # STEP 11: IDENTIFY MISSING PLAYER IDS
    # --------------------------------------------------------

    merged["missing_dismissed_player_clean"] = (
        merged["dismissed_player_id_clean_normalized"].isna()
    )

    merged["missing_dismissed_player_raw"] = (
        merged["dismissed_player_id_raw_normalized"].isna()
    )

    merged["missing_striker"] = (
        merged["striker_id_clean_normalized"].isna()
    )

    merged["missing_non_striker"] = (
        merged["non_striker_id_clean_normalized"].isna()
    )

    merged["missing_bowler"] = (
        merged["bowler_id_clean_normalized"].isna()
    )

    # --------------------------------------------------------
    # STEP 12: IDENTIFY DISMISSED PLAYER POSITION
    # --------------------------------------------------------

    merged["dismissed_is_striker"] = (
        merged["dismissed_player_id_clean_normalized"].notna()
        & (
            merged["dismissed_player_id_clean_normalized"]
            == merged["striker_id_clean_normalized"]
        )
    )

    merged["dismissed_is_non_striker"] = (
        merged["dismissed_player_id_clean_normalized"].notna()
        & (
            merged["dismissed_player_id_clean_normalized"]
            == merged["non_striker_id_clean_normalized"]
        )
    )

    merged["dismissed_player_position"] = "unmatched"

    merged.loc[
        merged["dismissed_is_striker"],
        "dismissed_player_position"
    ] = "striker"

    merged.loc[
        merged["dismissed_is_non_striker"],
        "dismissed_player_position"
    ] = "non_striker"

    # --------------------------------------------------------
    # STEP 13: IDENTIFY MISSING ID ORIGIN
    # --------------------------------------------------------

    def identify_missing_origin(row):

        clean_missing = row["missing_dismissed_player_clean"]
        raw_missing = row["missing_dismissed_player_raw"]

        if raw_missing and clean_missing:
            return "missing_in_original_and_cleaned"

        if not raw_missing and clean_missing:
            return "missing_introduced_during_cleaning"

        if raw_missing and not clean_missing:
            return "cleaned_value_present_but_source_missing"

        return "dismissed_player_id_available"

    merged["missing_id_origin"] = merged.apply(
        identify_missing_origin,
        axis=1
    )

    # --------------------------------------------------------
    # STEP 14: MANUAL REVIEW FLAGS
    # --------------------------------------------------------

    merged["manual_review_required"] = (
        (merged["_merge"] != "both")
        | merged["missing_dismissed_player_clean"]
        | merged["missing_striker"]
        | merged["missing_non_striker"]
        | merged["missing_bowler"]
        | (
            merged["dismissed_player_position"]
            == "unmatched"
        )
        | (
            merged["missing_id_origin"]
            == "missing_introduced_during_cleaning"
        )
    )

    # --------------------------------------------------------
    # STEP 15: DETAILED RECORD INSPECTION
    # --------------------------------------------------------

    print_section("DETAILED RETIRE OUT RECORDS")

    detail_columns = [
        "match_id",
        "innings",
        "overid",
        "over_ball_num",
        "dismissed_player_id_clean",
        "dismissed_player_id_raw",
        "striker_id_clean",
        "non_striker_id_clean",
        "bowler_id_clean",
        "dismissed_player_position",
        "missing_id_origin",
        "dismissal_type_matches",
        "is_wicket_matches",
        "dismissed_player_id_matches",
        "manual_review_required",
    ]

    print(
        merged[detail_columns].to_string(index=False)
    )

    # --------------------------------------------------------
    # STEP 16: FIELD COMPARISON SUMMARY
    # --------------------------------------------------------

    print_section("FIELD COMPARISON")

    comparison_columns = [
        "dismissal_type_matches",
        "is_wicket_matches",
        "dismissed_player_id_matches",
        "striker_id_matches",
        "non_striker_id_matches",
        "bowler_id_matches",
        "runs_matches",
        "extras1_runs_matches",
        "extras_2_runs_matches",
        "extras_type_matches",
    ]

    for col in comparison_columns:

        print(
            f"{col}:",
            int(merged[col].sum()),
            "/",
            len(merged)
        )

    # --------------------------------------------------------
    # STEP 17: MISSING ID SUMMARY
    # --------------------------------------------------------

    print_section("MISSING ID INVESTIGATION")

    print(
        "Missing dismissed player in original:",
        int(merged["missing_dismissed_player_raw"].sum())
    )

    print(
        "Missing dismissed player in cleaned data:",
        int(merged["missing_dismissed_player_clean"].sum())
    )

    print("\nMissing ID origin:")

    print(
        merged["missing_id_origin"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # STEP 18: POSITION SUMMARY
    # --------------------------------------------------------

    print_section("DISMISSED PLAYER POSITION")

    print(
        merged["dismissed_player_position"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # STEP 19: OVERALL SUMMARY
    # --------------------------------------------------------

    print_section("OVERALL AUDIT SUMMARY")

    summary = pd.DataFrame({
        "metric": [
            "original_retireout_records",
            "clean_retireout_records",
            "original_records_matched",
            "original_records_unmatched",
            "missing_dismissed_player_original",
            "missing_dismissed_player_cleaned",
            "missing_introduced_during_cleaning",
            "striker_dismissals",
            "non_striker_dismissals",
            "unmatched_dismissed_players",
            "manual_review_records",
            "bowler_credited_wickets",
        ],

        "value": [
            len(raw_retire),
            len(clean_retire),
            matched_count,
            unmatched_count,
            merged["missing_dismissed_player_raw"].sum(),
            merged["missing_dismissed_player_clean"].sum(),
            (
                merged["missing_id_origin"]
                == "missing_introduced_during_cleaning"
            ).sum(),
            merged["dismissed_is_striker"].sum(),
            merged["dismissed_is_non_striker"].sum(),
            (
                merged["dismissed_player_position"]
                == "unmatched"
            ).sum(),
            merged["manual_review_required"].sum(),
            0,
        ]
    })

    print(summary.to_string(index=False))

    # --------------------------------------------------------
    # STEP 20: SAVE OUTPUT FILES
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

    merged[
        merged["manual_review_required"]
    ].to_csv(
        REVIEW_FILE,
        index=False
    )

    print_section("OUTPUT FILES")

    print("Detailed report:", DETAIL_FILE)
    print("Summary report:", SUMMARY_FILE)
    print("Manual review report:", REVIEW_FILE)

    print_section("AUDIT COMPLETED")


if __name__ == "__main__":
    main()
