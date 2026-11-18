
from pathlib import Path
import pandas as pd

# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = ROOT / "data" / "processed" / "features"
INTERIM_DIR = ROOT / "data" / "interim"

SHORTLIST_FILE = FEATURE_DIR / "bowler_candidate_shortlist.csv"
STATUS_FILE = INTERIM_DIR / "ipl_2027_bowler_status.csv"

HISTORICAL_OUTPUT = FEATURE_DIR / "bowler_candidate_shortlist_all.csv"
REVIEW_OUTPUT = FEATURE_DIR / "bowler_candidate_shortlist_status_audited.csv"
SUMMARY_OUTPUT = INTERIM_DIR / "bowler_ipl_status_summary.csv"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading bowling shortlist and IPL 2027 status...")

shortlist = pd.read_csv(SHORTLIST_FILE)
status = pd.read_csv(STATUS_FILE)

print("Bowling shortlist records:", len(shortlist))
print("IPL status records:", len(status))


# ============================================================
# 3. REQUIRED COLUMN VALIDATION
# ============================================================

required_shortlist_columns = [
    "bowler_id",
    "bowler_name",
    "reliability_score",
    "pressure_matches",
    "total_pressure_deliveries",
    "candidate_category",
    "candidate_review_order",
]

required_status_columns = [
    "bowler_id",
    "bowler_name",
    "ipl_2027_status",
    "ipl_2027_team",
]

for col in required_shortlist_columns:
    if col not in shortlist.columns:
        raise ValueError(f"Missing shortlist column: {col}")

for col in required_status_columns:
    if col not in status.columns:
        raise ValueError(f"Missing status column: {col}")


# ============================================================
# 4. CHECK ID INTEGRITY
# ============================================================

shortlist["bowler_id"] = pd.to_numeric(
    shortlist["bowler_id"], errors="raise"
).astype("int64")

status["bowler_id"] = pd.to_numeric(
    status["bowler_id"], errors="raise"
).astype("int64")

if shortlist["bowler_id"].duplicated().any():
    raise ValueError("Duplicate bowler IDs in shortlist.")

if status["bowler_id"].duplicated().any():
    raise ValueError("Duplicate bowler IDs in status file.")


# ============================================================
# 5. CHECK PLAYER NAME CONSISTENCY
# ============================================================

name_check = shortlist[
    ["bowler_id", "bowler_name"]
].merge(
    status[["bowler_id", "bowler_name"]],
    on="bowler_id",
    how="left",
    suffixes=("_shortlist", "_status"),
    validate="one_to_one",
    indicator=True,
)

missing_ids = name_check[name_check["_merge"] != "both"]

if not missing_ids.empty:
    raise ValueError(
        "Some shortlist bowler IDs are missing from status file:\n"
        + missing_ids.to_string(index=False)
    )

name_check["name_matches"] = (
    name_check["bowler_name_shortlist"].str.strip().str.lower()
    ==
    name_check["bowler_name_status"].str.strip().str.lower()
)

if not name_check["name_matches"].all():
    raise ValueError(
        "Bowler ID/name consistency check failed:\n"
        + name_check.loc[
            ~name_check["name_matches"]
        ].to_string(index=False)
    )


# ============================================================
# 6. MERGE STATUS USING BOWLER ID
# ============================================================

candidates = shortlist.merge(
    status[
        ["bowler_id", "ipl_2027_status", "ipl_2027_team"]
    ],
    on="bowler_id",
    how="left",
    validate="one_to_one",
    indicator=True,
)

if not candidates["_merge"].eq("both").all():
    raise ValueError("Some bowlers have missing IPL status.")

candidates = candidates.drop(columns=["_merge"])


# ============================================================
# 7. VALIDATE STATUS VALUES
# ============================================================

valid_statuses = {
    "RETAINED",
    "RELEASED",
    "RETIRED",
    "UNVERIFIED",
}

candidates["ipl_2027_status"] = (
    candidates["ipl_2027_status"]
    .astype("string")
    .str.strip()
    .str.upper()
)

unknown_statuses = (
    set(candidates["ipl_2027_status"].dropna().unique())
    - valid_statuses
)

if unknown_statuses:
    raise ValueError(
        f"Unexpected IPL status values: {unknown_statuses}"
    )

if candidates["ipl_2027_status"].isna().any():
    raise ValueError("Missing IPL 2027 status values found.")


# ============================================================
# 8. ADD SELECTION REVIEW FLAGS
# ============================================================

candidates["historical_analysis_included"] = True

candidates["retired_from_ipl"] = (
    candidates["ipl_2027_status"] == "RETIRED"
)

# A known status does not automatically mean auction availability.
candidates["status_verified"] = (
    candidates["ipl_2027_status"].isin(
        ["RETAINED", "RELEASED"]
    )
)

# Retained players cannot be acquired by RR through the auction.
# Released players still require confirmation of auction registration.
candidates["rr_auction_acquisition_possible"] = (
    candidates["ipl_2027_status"] == "RELEASED"
)

candidates["auction_registration_verified"] = False

candidates["selection_review_status"] = (
    candidates["ipl_2027_status"].map({
        "RETAINED": "RETAINED_NOT_AVAILABLE_TO_RR",
        "RELEASED": "RELEASED_AUCTION_REGISTRATION_CHECK_REQUIRED",
        "RETIRED": "EXCLUDED_RETIRED",
        "UNVERIFIED": "MANUAL_STATUS_VERIFICATION_REQUIRED",
    })
)


# ============================================================
# 9. CREATE HISTORICAL AND REVIEW DATASETS
# ============================================================

# Preserve every candidate for historical reliability analysis.
historical_candidates = candidates.copy()

# Exclude only players marked RETIRED from the playing review.
review_candidates = candidates[
    ~candidates["retired_from_ipl"]
].copy()

review_candidates = review_candidates.sort_values(
    "candidate_review_order"
)


# ============================================================
# 10. VALIDATION
# ============================================================

print("\n========== BOWLING STATUS VALIDATION ==========")

print("Historical candidates:", len(historical_candidates))
print("Non-retired review candidates:", len(review_candidates))
print(
    "Retired players excluded:",
    int(candidates["retired_from_ipl"].sum()),
)
print(
    "Retained players:",
    int((candidates["ipl_2027_status"] == "RETAINED").sum()),
)
print(
    "Released players:",
    int((candidates["ipl_2027_status"] == "RELEASED").sum()),
)
print(
    "Unverified players:",
    int((candidates["ipl_2027_status"] == "UNVERIFIED").sum()),
)
print(
    "Status verified:",
    int(candidates["status_verified"].sum()),
)
print(
    "Released, registration still to verify:",
    int(candidates["rr_auction_acquisition_possible"].sum()),
)

assert len(historical_candidates) == len(shortlist)
assert review_candidates["bowler_id"].is_unique
assert review_candidates["ipl_2027_status"].notna().all()
assert not review_candidates["retired_from_ipl"].any()
assert candidates["bowler_id"].is_unique


# ============================================================
# 11. STATUS SUMMARY
# ============================================================

print("\n========== IPL 2027 BOWLER STATUS SUMMARY ==========")

status_summary = (
    historical_candidates
    .groupby("ipl_2027_status")
    .agg(
        candidate_count=("bowler_id", "count"),
        average_reliability_score=("reliability_score", "mean"),
        average_pressure_matches=("pressure_matches", "mean"),
        average_pressure_deliveries=(
            "total_pressure_deliveries", "mean"
        ),
    )
    .reset_index()
)

print(status_summary.round(2).to_string(index=False))


# ============================================================
# 12. DISPLAY REVIEW SHORTLIST
# ============================================================

preview_columns = [
    "bowler_id",
    "bowler_name",
    "candidate_review_order",
    "reliability_score",
    "pressure_matches",
    "total_pressure_deliveries",
    "candidate_category",
    "ipl_2027_status",
    "ipl_2027_team",
    "selection_review_status",
]

print("\n========== UPDATED BOWLING CANDIDATES ==========")

print(
    review_candidates[preview_columns]
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 13. SAVE OUTPUTS
# ============================================================

historical_candidates.to_csv(HISTORICAL_OUTPUT, index=False)
review_candidates.to_csv(REVIEW_OUTPUT, index=False)
status_summary.to_csv(SUMMARY_OUTPUT, index=False)

print("\nSaved historical bowling candidates:")
print(HISTORICAL_OUTPUT)

print("\nSaved status-audited bowling candidates:")
print(REVIEW_OUTPUT)

print("\nSaved status summary:")
print(SUMMARY_OUTPUT)

print("\nBOWLING STATUS INTEGRATION COMPLETE")
