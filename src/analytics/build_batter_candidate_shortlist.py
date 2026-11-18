
from pathlib import Path
import pandas as pd

# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = ROOT / "data" / "processed" / "features"
INTERIM_DIR = ROOT / "data" / "interim"

SCORE_FILE = FEATURE_DIR / "pressure_batter_reliability_scores.csv"
SENSITIVITY_FILE = INTERIM_DIR / "pressure_batter_weight_sensitivity_detail.csv"
ELIGIBILITY_FILE = FEATURE_DIR / "pressure_batter_eligibility.csv"
STATUS_FILE = INTERIM_DIR / "ipl_2027_player_status.csv"

HISTORICAL_OUTPUT = FEATURE_DIR / "batter_candidate_shortlist_all.csv"
FINAL_OUTPUT = FEATURE_DIR / "batter_candidate_shortlist.csv"
SUMMARY_OUTPUT = INTERIM_DIR / "batter_candidate_shortlist_summary.csv"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading batting analysis and IPL status data...")

scores = pd.read_csv(SCORE_FILE)
sensitivity = pd.read_csv(SENSITIVITY_FILE)
eligibility = pd.read_csv(ELIGIBILITY_FILE)
status = pd.read_csv(STATUS_FILE)

print("Score records:", len(scores))
print("Sensitivity records:", len(sensitivity))
print("Eligibility records:", len(eligibility))
print("IPL status records:", len(status))


# ============================================================
# 3. VALIDATE REQUIRED COLUMNS
# ============================================================

score_columns = [
    "striker_id",
    "striker_name",
    "batting_reliability_score",
    "primary_eligible",
    "pressure_matches",
    "total_pressure_deliveries",
    "overall_pressure_strike_rate",
]

scenarios = [
    "BASELINE",
    "SCORING_FOCUSED",
    "CONSISTENCY_FOCUSED",
    "BOUNDARY_FOCUSED",
    "DISMISSAL_FOCUSED",
    "EVIDENCE_FOCUSED",
    "BALANCED",
]

rank_columns = [f"{s}_rank" for s in scenarios]

for col in score_columns:
    if col not in scores.columns:
        raise ValueError(f"Missing score column: {col}")

for col in ["striker_id", "primary_eligible"]:
    if col not in eligibility.columns:
        raise ValueError(f"Missing eligibility column: {col}")

for col in ["striker_id", *rank_columns]:
    if col not in sensitivity.columns:
        raise ValueError(f"Missing sensitivity column: {col}")

for col in ["striker_name", "ipl_2027_status", "ipl_2027_team"]:
    if col not in status.columns:
        raise ValueError(f"Missing status column: {col}")


# ============================================================
# 4. CHECK DUPLICATES
# ============================================================

for name, df in [
    ("scores", scores),
    ("eligibility", eligibility),
    ("sensitivity", sensitivity),
]:
    if df["striker_id"].duplicated().any():
        raise ValueError(f"Duplicate striker IDs in {name}.")

if status["striker_name"].duplicated().any():
    raise ValueError("Duplicate player names in status file.")


# ============================================================
# 5. NORMALIZE PLAYER NAMES
# ============================================================

def normalize_name(series):
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


scores["player_key"] = normalize_name(scores["striker_name"])
status["player_key"] = normalize_name(status["striker_name"])

if status["player_key"].duplicated().any():
    raise ValueError("Duplicate normalized names in status file.")


# ============================================================
# 6. FILTER PRIMARY-ELIGIBLE BATTERS
# ============================================================

scores["primary_eligible"] = (
    scores["primary_eligible"]
    .astype(str)
    .str.lower()
    .isin(["true", "1", "yes"])
)

primary_scores = scores[
    scores["primary_eligible"]
].copy()

eligibility["primary_eligible"] = (
    eligibility["primary_eligible"]
    .astype(str)
    .str.lower()
    .isin(["true", "1", "yes"])
)

primary_eligibility = eligibility[
    ["striker_id", "primary_eligible"]
].rename(
    columns={"primary_eligible": "eligibility_verified"}
)

primary_scores = primary_scores.merge(
    primary_eligibility,
    on="striker_id",
    how="left",
    validate="one_to_one"
)

if primary_scores["eligibility_verified"].isna().any():
    raise ValueError("Missing eligibility verification.")

if not primary_scores["eligibility_verified"].all():
    raise ValueError("Primary eligibility mismatch.")


# ============================================================
# 7. SENSITIVITY ANALYSIS
# ============================================================

if set(primary_scores["striker_id"]) != set(sensitivity["striker_id"]):
    raise ValueError("Score and sensitivity player IDs do not match.")

sensitivity["top10_scenario_count"] = sum(
    sensitivity[col].le(10).astype(int)
    for col in rank_columns
)

sensitivity["top10_scenario_percentage"] = (
    sensitivity["top10_scenario_count"] / len(scenarios) * 100
)


# ============================================================
# 8. IDENTIFY CANDIDATES
# ============================================================

baseline_top20 = set(
    sensitivity.loc[
        sensitivity["BASELINE_rank"] <= 20,
        "striker_id"
    ]
)

top10_any_scenario = set()

for col in rank_columns:
    top10_any_scenario.update(
        sensitivity.loc[
            sensitivity[col] <= 10,
            "striker_id"
        ]
    )

candidate_ids = baseline_top20 | top10_any_scenario

candidates = primary_scores[
    primary_scores["striker_id"].isin(candidate_ids)
].copy()

candidates = candidates.merge(
    sensitivity[
        [
            "striker_id",
            "top10_scenario_count",
            "top10_scenario_percentage",
            *rank_columns,
        ]
    ],
    on="striker_id",
    how="left",
    validate="one_to_one"
)


# ============================================================
# 9. CANDIDATE CATEGORY
# ============================================================

def classify_candidate(count):

    if count >= 5:
        return "CORE_STABLE_CANDIDATE"

    elif count >= 3:
        return "STRONG_CONTENDER"

    return "BACKUP_CANDIDATE"


candidates["candidate_category"] = (
    candidates["top10_scenario_count"]
    .apply(classify_candidate)
)


# ============================================================
# 10. JOIN IPL 2027 STATUS
# ============================================================

candidates = candidates.merge(
    status[
        [
            "player_key",
            "ipl_2027_status",
            "ipl_2027_team",
        ]
    ],
    on="player_key",
    how="left",
    validate="many_to_one",
    indicator=True
)

missing_status = candidates[
    candidates["_merge"] != "both"
]

if not missing_status.empty:
    print("\nPlayers missing IPL status:")
    print(missing_status["striker_name"].to_list())

    raise ValueError(
        "Some shortlist players have no status record. "
        "Update the status CSV before proceeding."
    )

candidates.drop(columns=["_merge", "player_key"], inplace=True)


# ============================================================
# 11. DEFINE SELECTION STATUS
# ============================================================

valid_statuses = {
    "RETAINED",
    "RELEASED",
    "RETIRED",
    "UNVERIFIED",
}

unknown_statuses = (
    set(candidates["ipl_2027_status"].dropna().unique())
    - valid_statuses
)

if unknown_statuses:
    raise ValueError(f"Unexpected player statuses: {unknown_statuses}")

# Historical records are always preserved.
candidates["historical_analysis_included"] = True

# Retired players are excluded from active playing consideration.
candidates["retired_from_ipl"] = (
    candidates["ipl_2027_status"] == "RETIRED"
)

# Retained and released players are not retired.
# Unverified players remain for manual review.
candidates["not_retired"] = ~candidates["retired_from_ipl"]

candidates["availability_verified"] = (
    candidates["ipl_2027_status"].isin(["RETAINED", "RELEASED"])
)

candidates["selection_review_status"] = (
    candidates["ipl_2027_status"].map({
        "RETAINED": "RETAINED_PLAYER",
        "RELEASED": "RELEASED_AUCTION_REVIEW",
        "RETIRED": "EXCLUDED_RETIRED",
        "UNVERIFIED": "MANUAL_STATUS_VERIFICATION_REQUIRED",
    })
)


# ============================================================
# 12. CREATE TWO OUTPUT DATASETS
# ============================================================

# Historical dataset includes everyone.
historical_candidates = candidates.copy()

# Final review shortlist excludes retired players only.
# Unverified players remain, but are clearly flagged.
final_candidates = candidates[
    candidates["not_retired"]
].copy()

final_candidates = final_candidates.sort_values(
    by=[
        "top10_scenario_count",
        "batting_reliability_score",
        "total_pressure_deliveries",
    ],
    ascending=[False, False, False]
).reset_index(drop=True)

final_candidates["shortlist_position"] = (
    range(1, len(final_candidates) + 1)
)


# ============================================================
# 13. VALIDATION
# ============================================================

print("\n========== FINAL VALIDATION ==========")

print("Historical candidates:", len(historical_candidates))
print("Non-retired review candidates:", len(final_candidates))

print(
    "Retired candidates excluded:",
    int(candidates["retired_from_ipl"].sum())
)

print(
    "Unverified candidates retained for review:",
    int((final_candidates["ipl_2027_status"] == "UNVERIFIED").sum())
)

print(
    "Duplicate candidate IDs:",
    final_candidates["striker_id"].duplicated().sum()
)

print(
    "Missing player statuses:",
    final_candidates["ipl_2027_status"].isna().sum()
)

assert final_candidates["striker_id"].duplicated().sum() == 0
assert final_candidates["ipl_2027_status"].notna().all()
assert not final_candidates["retired_from_ipl"].any()


# ============================================================
# 14. STATUS SUMMARY
# ============================================================

print("\n========== IPL 2027 STATUS SUMMARY ==========")

status_summary = (
    historical_candidates.groupby("ipl_2027_status")
    .agg(
        candidate_count=("striker_id", "count"),
        average_score=("batting_reliability_score", "mean"),
        average_pressure_matches=("pressure_matches", "mean"),
        average_top10_scenarios=("top10_scenario_count", "mean"),
    )
    .reset_index()
)

print(status_summary.round(2).to_string(index=False))


# ============================================================
# 15. CATEGORY SUMMARY
# ============================================================

print("\n========== CANDIDATE CATEGORY SUMMARY ==========")

category_summary = (
    final_candidates.groupby("candidate_category")
    .agg(
        candidate_count=("striker_id", "count"),
        average_score=("batting_reliability_score", "mean"),
        average_pressure_matches=("pressure_matches", "mean"),
        average_pressure_deliveries=(
            "total_pressure_deliveries", "mean"
        ),
        average_top10_scenarios=("top10_scenario_count", "mean"),
    )
    .reset_index()
)

print(category_summary.round(2).to_string(index=False))


# ============================================================
# 16. CANDIDATE PREVIEW
# ============================================================

preview_columns = [
    "shortlist_position",
    "striker_name",
    "batting_reliability_score",
    "pressure_matches",
    "total_pressure_deliveries",
    "overall_pressure_strike_rate",
    "top10_scenario_count",
    "candidate_category",
    "ipl_2027_status",
    "ipl_2027_team",
    "selection_review_status",
]

print("\n========== UPDATED BATTING SHORTLIST ==========")

print(
    final_candidates[preview_columns]
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 17. SAVE OUTPUTS
# ============================================================

historical_candidates.to_csv(
    HISTORICAL_OUTPUT,
    index=False
)

final_candidates.to_csv(
    FINAL_OUTPUT,
    index=False
)

status_summary.to_csv(
    INTERIM_DIR / "batter_ipl_status_summary.csv",
    index=False
)

category_summary.to_csv(
    SUMMARY_OUTPUT,
    index=False
)

print("\nSaved historical candidates:", HISTORICAL_OUTPUT)
print("Saved non-retired review shortlist:", FINAL_OUTPUT)
print("Saved IPL status summary.")
print("Saved category summary.")

print("\nUPDATED BATTING SHORTLIST COMPLETE")
