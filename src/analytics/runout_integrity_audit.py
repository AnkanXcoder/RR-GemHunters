
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

DETAIL_FILE = OUTPUT_DIR / "runout_integrity_detail.csv"
SUMMARY_FILE = OUTPUT_DIR / "runout_integrity_summary.csv"
REVIEW_FILE = OUTPUT_DIR / "runout_manual_review.csv"

KEYS = ["match_id", "innings", "overid", "over_ball_num"]


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


def print_section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print_section("RUN OUT INTEGRITY AUDIT")

    # --------------------------------------------------------
    # 1. Load data
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
            f"Missing columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # 3. Normalize dismissal types
    # --------------------------------------------------------

    df["dismissal_normalized"] = (
        df["dismissal_type"]
        .apply(normalize_text)
    )

    # --------------------------------------------------------
    # 4. Identify all Run Out records
    # --------------------------------------------------------

    runouts = df[
        df["dismissal_normalized"] == "runout"
    ].copy()

    print_section("RUN OUT COUNTS")

    print("Total Run Out records:", len(runouts))

    print(
        "Unique matches:",
        runouts["match_id"].nunique()
    )

    print(
        "Unique dismissed players:",
        runouts["dismissed_player_id"].nunique()
    )

    # --------------------------------------------------------
    # 5. Check player ID availability
    # --------------------------------------------------------

    runouts["missing_dismissed_player"] = (
        runouts["dismissed_player_id"].isna()
    )

    runouts["missing_striker"] = (
        runouts["striker_id"].isna()
    )

    runouts["missing_non_striker"] = (
        runouts["non_striker_id"].isna()
    )

    runouts["missing_bowler"] = (
        runouts["bowler_id"].isna()
    )

    
    # --------------------------------------------------------
    # 6. Compare dismissed player with batting positions
    # --------------------------------------------------------

    # Normalize IDs to nullable integers before comparing.
    # This handles values such as 5034.0 and 5034 consistently.

    player_id_columns = [
    "dismissed_player_id",
    "striker_id",
    "non_striker_id",
    ]

    for col in player_id_columns:
        runouts[col] = pd.to_numeric(
        runouts[col],
        errors="coerce"
    ).astype("Int64")

    runouts["dismissed_is_striker"] = (
    runouts["dismissed_player_id"].notna()
    & (
        runouts["dismissed_player_id"]
        == runouts["striker_id"]
    )
    )

    runouts["dismissed_is_non_striker"] = (
    runouts["dismissed_player_id"].notna()
    & (
        runouts["dismissed_player_id"]
        == runouts["non_striker_id"]
    )
    )

    runouts["dismissed_player_position"] = "unmatched"

    runouts.loc[
    runouts["dismissed_is_striker"],
    "dismissed_player_position"
    ] = "striker"

    runouts.loc[
    runouts["dismissed_is_non_striker"],
    "dismissed_player_position"
    ] = "non_striker"

    # --------------------------------------------------------
    # 7. Check wicket indicator
    # --------------------------------------------------------

    runouts["is_wicket_normalized"] = (
        runouts["is_wicket"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "1.0"])
    )

    # --------------------------------------------------------
    # 8. Bowler credit classification
    # --------------------------------------------------------

    # Run Out is not credited to the bowler.

    runouts["bowler_wicket_credit"] = False

    # --------------------------------------------------------
    # 9. Identify manual review records
    # --------------------------------------------------------

    runouts["manual_review_required"] = (
        runouts["missing_dismissed_player"]
        | runouts["missing_striker"]
        | runouts["missing_non_striker"]
        | runouts["missing_bowler"]
        | ~runouts["is_wicket_normalized"]
        | (
            runouts["dismissed_player_position"]
            == "unmatched"
        )
    )

    review = runouts[
        runouts["manual_review_required"]
    ].copy()

    # --------------------------------------------------------
    # 10. Print player position analysis
    # --------------------------------------------------------

    print_section("DISMISSED PLAYER POSITION")

    print(
        runouts["dismissed_player_position"]
        .value_counts(dropna=False)
        .to_string()
    )

    print_section("MISSING PLAYER IDS")

    print(
        "Missing dismissed player:",
        int(runouts["missing_dismissed_player"].sum())
    )

    print(
        "Missing striker:",
        int(runouts["missing_striker"].sum())
    )

    print(
        "Missing non-striker:",
        int(runouts["missing_non_striker"].sum())
    )

    print(
        "Missing bowler:",
        int(runouts["missing_bowler"].sum())
    )

    # --------------------------------------------------------
    # 11. Category-wise summary
    # --------------------------------------------------------

    print_section("RUN OUT SUMMARY")

    runouts["extras_type_normalized"] = (
        runouts["extras_type"]
        .fillna("none")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    summary = (
        runouts.groupby(
            "extras_type_normalized",
            dropna=False
        )
        .agg(
            runout_records=("match_id", "size"),
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
    # 12. Overall audit results
    # --------------------------------------------------------

    print_section("OVERALL AUDIT RESULTS")

    print("Total Run Outs:", len(runouts))

    print(
        "Valid wicket indicators:",
        int(runouts["is_wicket_normalized"].sum())
    )

    print(
        "Bowler-credited wickets:",
        int(runouts["bowler_wicket_credit"].sum())
    )

    print(
        "Manual review records:",
        len(review)
    )

    # --------------------------------------------------------
    # 13. Save reports
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    runouts.to_csv(
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
