
from pathlib import Path
import pandas as pd

# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed" / "deliveries"
    / "regulation_deliveries.csv"
)

OUTPUT_DIR = ROOT / "data" / "interim"

DETAIL_FILE = OUTPUT_DIR / "dismissal_type_detail.csv"
SUMMARY_FILE = OUTPUT_DIR / "dismissal_type_summary.csv"
REVIEW_FILE = OUTPUT_DIR / "dismissal_type_manual_review.csv"

# Dismissals not credited to the bowler under standard
# cricket scoring conventions.
NON_BOWLER_DISMISSALS = {
    "runout",
    "retireout",
    "obstructingthefield",
    "obstractingthefield",
}

# ============================================================
# HELPER FUNCTIONS
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
# MAIN AUDIT
# ============================================================

def main():

    print_section("DISMISSAL TYPE INTEGRITY AUDIT")

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print("Input dataset shape:", df.shape)

    # --------------------------------------------------------
    # 2. Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "match_id",
        "innings",
        "overid",
        "over_ball_num",
        "is_wicket",
        "dismissal_type",
        "dismissed_player_id",
        "striker_id",
        "non_striker_id",
        "bowler_id",
        "extras_type",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # 3. Normalize wicket indicator and dismissal type
    # --------------------------------------------------------

    df["is_wicket_normalized"] = normalize_boolean(
        df["is_wicket"]
    )

    df["dismissal_normalized"] = (
        df["dismissal_type"]
        .apply(normalize_text)
    )

    # --------------------------------------------------------
    # 4. Identify wicket records
    # --------------------------------------------------------

    wickets = df[
        df["is_wicket_normalized"]
    ].copy()

    print_section("WICKET RECORD COUNTS")

    print("Total deliveries:", len(df))
    print("Total wicket records:", len(wickets))

    print(
        "Records with dismissal type missing:",
        int(wickets["dismissal_normalized"].eq("").sum())
    )

    # --------------------------------------------------------
    # 5. Identify unique dismissal types
    # --------------------------------------------------------

    print_section("DISMISSAL TYPE DISTRIBUTION")

    distribution = (
        wickets["dismissal_normalized"]
        .value_counts(dropna=False)
        .rename_axis("dismissal_type")
        .reset_index(name="records")
    )

    print(distribution.to_string(index=False))

    # --------------------------------------------------------
    # 6. Classify bowler wicket credit
    # --------------------------------------------------------

    def classify_dismissal(value):

        if value == "":
            return "unknown"

        if value in NON_BOWLER_DISMISSALS:
            return "not_credited_to_bowler"

        return "credited_to_bowler"

    wickets["bowler_credit_status"] = (
        wickets["dismissal_normalized"]
        .apply(classify_dismissal)
    )

    # --------------------------------------------------------
    # 7. Validate dismissed player IDs
    # --------------------------------------------------------

    player_id_columns = [
        "dismissed_player_id",
        "striker_id",
        "non_striker_id",
        "bowler_id",
    ]

    for col in player_id_columns:
        wickets[col] = pd.to_numeric(
            wickets[col],
            errors="coerce"
        ).astype("Int64")

    wickets["missing_dismissed_player"] = (
        wickets["dismissed_player_id"].isna()
    )

    wickets["missing_striker"] = (
        wickets["striker_id"].isna()
    )

    wickets["missing_non_striker"] = (
        wickets["non_striker_id"].isna()
    )

    wickets["missing_bowler"] = (
        wickets["bowler_id"].isna()
    )

    # --------------------------------------------------------
    # 8. Check dismissed player position
    # --------------------------------------------------------

    wickets["dismissed_is_striker"] = (
        wickets["dismissed_player_id"].notna()
        & (
            wickets["dismissed_player_id"]
            == wickets["striker_id"]
        )
    )

    wickets["dismissed_is_non_striker"] = (
        wickets["dismissed_player_id"].notna()
        & (
            wickets["dismissed_player_id"]
            == wickets["non_striker_id"]
        )
    )

    wickets["dismissed_player_position"] = "unmatched"

    wickets.loc[
        wickets["dismissed_is_striker"],
        "dismissed_player_position"
    ] = "striker"

    wickets.loc[
        wickets["dismissed_is_non_striker"],
        "dismissed_player_position"
    ] = "non_striker"

    # --------------------------------------------------------
    # 9. Flag unusual or incomplete records
    # --------------------------------------------------------

    wickets["manual_review_required"] = (
        wickets["dismissal_normalized"].eq("")
        | (wickets["dismissal_normalized"] == "unknown")
        | wickets["missing_dismissed_player"]
        | wickets["missing_striker"]
        | wickets["missing_non_striker"]
        | wickets["missing_bowler"]
        | (
            wickets["dismissed_player_position"]
            == "unmatched"
        )
    )

    # --------------------------------------------------------
    # 10. Category-wise summary
    # --------------------------------------------------------

    print_section("BOWLER CREDIT CLASSIFICATION")

    credit_summary = (
        wickets["bowler_credit_status"]
        .value_counts()
        .rename_axis("bowler_credit_status")
        .reset_index(name="records")
    )

    print(credit_summary.to_string(index=False))

    print_section("DISMISSAL TYPE SUMMARY")

    summary = (
        wickets.groupby(
            ["dismissal_normalized", "bowler_credit_status"],
            dropna=False
        )
        .agg(
            total_records=("match_id", "size"),
            unique_matches=("match_id", "nunique"),
            missing_dismissed_player=(
                "missing_dismissed_player", "sum"
            ),
            striker_dismissals=(
                "dismissed_is_striker", "sum"
            ),
            non_striker_dismissals=(
                "dismissed_is_non_striker", "sum"
            ),
            unmatched_dismissals=(
                "dismissed_player_position",
                lambda s: (s == "unmatched").sum()
            ),
            manual_review_records=(
                "manual_review_required", "sum"
            ),
        )
        .reset_index()
    )

    print(summary.to_string(index=False))

    # --------------------------------------------------------
    # 11. Reconciliation checks
    # --------------------------------------------------------

    print_section("WICKET RECONCILIATION")

    total_wickets = len(wickets)

    bowler_credited = int(
        (
            wickets["bowler_credit_status"]
            == "credited_to_bowler"
        ).sum()
    )

    non_bowler = int(
        (
            wickets["bowler_credit_status"]
            == "not_credited_to_bowler"
        ).sum()
    )

    unknown = int(
        (
            wickets["bowler_credit_status"]
            == "unknown"
        ).sum()
    )

    print("Total wicket records:", total_wickets)
    print("Bowler-credited:", bowler_credited)
    print("Non-bowler dismissals:", non_bowler)
    print("Unknown classifications:", unknown)

    print(
        "Reconciliation:",
        bowler_credited + non_bowler + unknown,
        "==",
        total_wickets
    )

    # --------------------------------------------------------
    # 12. Build manual review dataset
    # --------------------------------------------------------

    review = wickets[
        wickets["manual_review_required"]
    ].copy()

    print_section("MANUAL REVIEW")

    print("Records requiring review:", len(review))

    # --------------------------------------------------------
    # 13. Save audit outputs
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    wickets.to_csv(
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

    print_section("OUTPUT FILES")

    print("Detailed report:", DETAIL_FILE)
    print("Summary report:", SUMMARY_FILE)
    print("Manual review report:", REVIEW_FILE)

    print_section("AUDIT COMPLETED")


if __name__ == "__main__":
    main()
