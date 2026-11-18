
from pathlib import Path
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = ROOT / "data" / "processed" / "features"

BATTER_FILE = FEATURE_DIR / "batter_candidate_shortlist.csv"
BOWLER_FILE = FEATURE_DIR / "bowler_candidate_shortlist_status_audited.csv"

BATTER_OUTPUT = FEATURE_DIR / "final_batter_selection_review.csv"
BOWLER_OUTPUT = FEATURE_DIR / "final_bowler_selection_review.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def select_existing_columns(df, preferred_columns):
    """
    Select only columns that are available in the input dataset.
    """
    return [
        col for col in preferred_columns
        if col in df.columns
    ]


def prepare_review(
    df,
    preferred_columns,
    id_column,
    sort_column
):
    """
    Prepare a candidate review sheet.

    - Select relevant columns.
    - Add reviewer input columns.
    - Sort candidates by the specified ranking column.
    """

    if id_column not in df.columns:
        raise ValueError(
            f"Required ID column '{id_column}' is missing."
        )

    if sort_column not in df.columns:
        raise ValueError(
            f"Required sorting column '{sort_column}' is missing."
        )

    columns = select_existing_columns(
        df,
        preferred_columns
    )

    review = df[columns].copy()

    # Reviewer input columns
    review["final_selection"] = ""
    review["reviewer_decision"] = ""
    review["reviewer_notes"] = ""

    # Ensure ranking column is numeric
    review[sort_column] = pd.to_numeric(
        review[sort_column],
        errors="coerce"
    )

    # Sort candidates by review priority
    review = (
        review
        .sort_values(
            by=sort_column,
            ascending=True,
            kind="stable",
            na_position="last"
        )
        .reset_index(drop=True)
    )

    return review


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading candidate datasets...")

if not BATTER_FILE.exists():
    raise FileNotFoundError(
        f"Batting shortlist not found: {BATTER_FILE}"
    )

if not BOWLER_FILE.exists():
    raise FileNotFoundError(
        f"Bowling shortlist not found: {BOWLER_FILE}"
    )

batters = pd.read_csv(BATTER_FILE)
bowlers = pd.read_csv(BOWLER_FILE)

print("Batters:", len(batters))
print("Bowlers:", len(bowlers))


# ============================================================
# BATTER REVIEW SHEET
# ============================================================

print("\nPreparing batter review sheet...")

batter_columns = [
    "striker_id",
    "striker_name",
    "batting_reliability_score",
    "provisional_rank",
    "pressure_matches",
    "total_pressure_deliveries",
    "total_pressure_runs",
    "total_balls_faced",
    "total_striker_dismissals",
    "overall_pressure_strike_rate",
    "average_match_strike_rate",
    "median_match_strike_rate",
    "std_match_strike_rate",
    "top10_scenario_count",
    "top10_scenario_percentage",
    "candidate_category",
    "ipl_2027_status",
    "ipl_2027_team",
    "selection_review_status",
]

batter_review = prepare_review(
    df=batters,
    preferred_columns=batter_columns,
    id_column="striker_id",
    sort_column="provisional_rank"
)

# Save batting review sheet
batter_review.to_csv(
    BATTER_OUTPUT,
    index=False
)

print("Batting review sheet created.")


# ============================================================
# BOWLER REVIEW SHEET
# ============================================================

print("\nPreparing bowler review sheet...")

bowler_columns = [
    "bowler_id",
    "bowler_name",
    "reliability_score",
    "provisional_rank",
    "candidate_review_order",
    "pressure_matches",
    "total_pressure_deliveries",
    "total_legal_balls",
    "total_runs_conceded",
    "total_wickets",
    "overall_pressure_economy",
    "overall_dot_ball_percentage",
    "overall_boundary_percentage",
    "average_match_economy",
    "median_match_economy",
    "std_match_economy",
    "top10_scenario_count",
    "rank_spread",
    "format_evidence_category",
    "IPL_evidence_sufficient",
    "T20I_evidence_sufficient",
    "candidate_category",
    "ipl_2027_status",
    "ipl_2027_team",
    "selection_review_status",
]

bowler_review = prepare_review(
    df=bowlers,
    preferred_columns=bowler_columns,
    id_column="bowler_id",
    sort_column="candidate_review_order"
)

# Save bowling review sheet
bowler_review.to_csv(
    BOWLER_OUTPUT,
    index=False
)

print("Bowling review sheet created.")


# ============================================================
# VALIDATION
# ============================================================

print("\n========== REVIEW SHEET VALIDATION ==========")

print("\nBatting review shape:", batter_review.shape)
print("Bowling review shape:", bowler_review.shape)

print(
    "\nBatting duplicate IDs:",
    batter_review["striker_id"].duplicated().sum()
)

print(
    "Bowling duplicate IDs:",
    bowler_review["bowler_id"].duplicated().sum()
)

# Check unique player IDs
assert batter_review["striker_id"].is_unique, (
    "Duplicate batter IDs found."
)

assert bowler_review["bowler_id"].is_unique, (
    "Duplicate bowler IDs found."
)

# Check sorting
assert batter_review["provisional_rank"].is_monotonic_increasing, (
    "Batters are not sorted by provisional rank."
)

assert bowler_review["candidate_review_order"].is_monotonic_increasing, (
    "Bowlers are not sorted by candidate review order."
)

# Check reviewer columns
review_columns = [
    "final_selection",
    "reviewer_decision",
    "reviewer_notes"
]

for column in review_columns:
    assert column in batter_review.columns
    assert column in bowler_review.columns

print("\nBatting review preview:")
print(
    batter_review[
        ["provisional_rank", "striker_name", "batting_reliability_score"]
    ].head(10).to_string(index=False)
)

print("\nBowling review preview:")
print(
    bowler_review[
        [
            "candidate_review_order",
            "bowler_name",
            "reliability_score",
            "candidate_category"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\nSaved batting review:")
print(BATTER_OUTPUT)

print("\nSaved bowling review:")
print(BOWLER_OUTPUT)

print("\nFINAL SELECTION REVIEW SHEETS CREATED SUCCESSFULLY.")
