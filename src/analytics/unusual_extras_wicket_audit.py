
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

DETAIL_FILE = OUTPUT_DIR / "unusual_extras_wicket_detail.csv"
SUMMARY_FILE = OUTPUT_DIR / "unusual_extras_wicket_summary.csv"

KEYS = ["match_id", "innings", "overid", "over_ball_num"]

UNUSUAL_CATEGORIES = ["other", "overstep"]

# Dismissals that are not credited to the bowler.
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


def to_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    )


def print_section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print_section("UNUSUAL EXTRAS WICKET AUDIT")

    # --------------------------------------------------------
    # 1. Load regulation deliveries
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
        "extras_type",
        "is_wicket",
        "dismissal_type",
        "dismissed_player_id",
        "striker_id",
        "bowler_id",
        "runs",
        "extras1_runs",
        "extras_2_runs",
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
    # 3. Normalize unusual extras categories
    # --------------------------------------------------------

    df["extras_type_normalized"] = (
        df["extras_type"]
        .fillna("none")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # 4. Filter unusual deliveries containing wickets
    # --------------------------------------------------------

    unusual = df[
        df["extras_type_normalized"].isin(
            UNUSUAL_CATEGORIES
        )
    ].copy()

    # Handle boolean values safely.
    unusual["is_wicket_normalized"] = (
        unusual["is_wicket"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "1.0"])
    )

    wickets = unusual[
        unusual["is_wicket_normalized"]
    ].copy()

    print_section("WICKET RECORD COUNTS")

    print(
        "Unusual deliveries:",
        len(unusual)
    )

    print(
        "Total wicket records:",
        len(wickets)
    )

    print("\nWickets by extras category:")

    print(
        wickets["extras_type_normalized"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # 5. Normalize dismissal types
    # --------------------------------------------------------

    wickets["dismissal_normalized"] = (
        wickets["dismissal_type"]
        .apply(normalize_text)
    )

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
    # 7. Check dismissed batter information
    # --------------------------------------------------------

    wickets["dismissed_player_missing"] = (
        wickets["dismissed_player_id"].isna()
    )

    wickets["dismissed_player_matches_striker"] = (
        wickets["dismissed_player_id"]
        .notna()
        & wickets["striker_id"].notna()
        & (
            wickets["dismissed_player_id"].astype("string")
            == wickets["striker_id"].astype("string")
        )
    )

    # A missing dismissed player ID or a dismissed batter
    # different from the striker requires manual inspection.
    wickets["dismissed_player_review"] = (
        wickets["dismissed_player_missing"]
        | (
            wickets["dismissed_player_id"].notna()
            & ~wickets["dismissed_player_matches_striker"]
        )
    )

    # --------------------------------------------------------
    # 8. Identify unresolved dismissal types
    # --------------------------------------------------------

    wickets["unknown_dismissal"] = (
        wickets["bowler_credit_status"] == "unknown"
    )

    wickets["manual_review_required"] = (
        wickets["unknown_dismissal"]
        | wickets["dismissed_player_review"]
    )

    # --------------------------------------------------------
    # 9. Detailed wicket report
    # --------------------------------------------------------

    print_section("DETAILED WICKET INSPECTION")

    detail_columns = [
        "match_id",
        "innings",
        "overid",
        "over_ball_num",
        "extras_type_normalized",
        "runs",
        "extras1_runs",
        "extras_2_runs",
        "striker_id",
        "bowler_id",
        "dismissed_player_id",
        "dismissal_type",
        "bowler_credit_status",
        "dismissed_player_missing",
        "dismissed_player_matches_striker",
        "manual_review_required",
    ]

    print(
        wickets[detail_columns].to_string(index=False)
    )

    # --------------------------------------------------------
    # 10. Summary by extras category
    # --------------------------------------------------------

    print_section("CATEGORY-LEVEL WICKET SUMMARY")

    summary = (
        wickets.groupby(
            "extras_type_normalized",
            dropna=False
        )
        .agg(
            total_wickets=("match_id", "size"),
            bowler_credited=(
                "bowler_credit_status",
                lambda s: (s == "credited_to_bowler").sum()
            ),
            not_bowler_credited=(
                "bowler_credit_status",
                lambda s: (
                    s == "not_credited_to_bowler"
                ).sum()
            ),
            unknown_dismissals=(
                "unknown_dismissal", "sum"
            ),
            missing_dismissed_player=(
                "dismissed_player_missing", "sum"
            ),
            manual_review_records=(
                "manual_review_required", "sum"
            ),
        )
        .reset_index()
    )

    print(
        summary.to_string(index=False)
    )

    # --------------------------------------------------------
    # 11. Overall audit statistics
    # --------------------------------------------------------

    print_section("OVERALL AUDIT RESULTS")

    print(
        "Total unusual wickets:",
        len(wickets)
    )

    print(
        "Bowler-credited wickets:",
        int(
            (
                wickets["bowler_credit_status"]
                == "credited_to_bowler"
            ).sum()
        )
    )

    print(
        "Non-bowler dismissals:",
        int(
            (
                wickets["bowler_credit_status"]
                == "not_credited_to_bowler"
            ).sum()
        )
    )

    print(
        "Unknown dismissal types:",
        int(wickets["unknown_dismissal"].sum())
    )

    print(
        "Missing dismissed player IDs:",
        int(wickets["dismissed_player_missing"].sum())
    )

    print(
        "Records requiring manual review:",
        int(wickets["manual_review_required"].sum())
    )

    # --------------------------------------------------------
    # 12. Save outputs
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

    print_section("OUTPUT FILES")

    print("Detailed report:", DETAIL_FILE)
    print("Summary report:", SUMMARY_FILE)

    print_section("AUDIT COMPLETED")


if __name__ == "__main__":
    main()
