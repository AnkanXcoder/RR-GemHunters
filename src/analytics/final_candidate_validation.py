
from pathlib import Path
import pandas as pd

# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = ROOT / "data" / "processed" / "features"
INTERIM_DIR = ROOT / "data" / "interim"

BATTER_FILE = FEATURE_DIR / "batter_candidate_shortlist.csv"
BOWLER_FILE = FEATURE_DIR / "bowler_candidate_shortlist_status_audited.csv"

OUTPUT_FILE = INTERIM_DIR / "final_candidate_validation.csv"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading final batting and bowling candidates...")

batters = pd.read_csv(BATTER_FILE)
bowlers = pd.read_csv(BOWLER_FILE)

print("Batting candidates:", len(batters))
print("Bowling candidates:", len(bowlers))


# ============================================================
# 3. REQUIRED COLUMNS
# ============================================================

batter_required = [
    "striker_id",
    "striker_name",
    "batting_reliability_score",
    "pressure_matches",
    "total_pressure_deliveries",
    "ipl_2027_status",
    "selection_review_status",
]

bowler_required = [
    "bowler_id",
    "bowler_name",
    "reliability_score",
    "pressure_matches",
    "total_pressure_deliveries",
    "ipl_2027_status",
    "selection_review_status",
]

for col in batter_required:
    if col not in batters.columns:
        raise ValueError(f"Missing batter column: {col}")

for col in bowler_required:
    if col not in bowlers.columns:
        raise ValueError(f"Missing bowler column: {col}")


# ============================================================
# 4. VALIDATION FUNCTION
# ============================================================

def validate_candidates(df, id_col, name_col, score_col, role):

    checks = []

    checks.append({
        "role": role,
        "check": "Unique player IDs",
        "result": not df[id_col].duplicated().any(),
        "details": f"Duplicates: {df[id_col].duplicated().sum()}",
    })

    checks.append({
        "role": role,
        "check": "Player names present",
        "result": df[name_col].notna().all(),
        "details": f"Missing: {df[name_col].isna().sum()}",
    })

    checks.append({
        "role": role,
        "check": "Reliability scores present",
        "result": df[score_col].notna().all(),
        "details": f"Missing: {df[score_col].isna().sum()}",
    })

    checks.append({
        "role": role,
        "check": "Reliability scores within 0-100",
        "result": df[score_col].between(0, 100).all(),
        "details": (
            f"Min: {df[score_col].min():.2f}, "
            f"Max: {df[score_col].max():.2f}"
        ),
    })

    checks.append({
        "role": role,
        "check": "IPL statuses present",
        "result": df["ipl_2027_status"].notna().all(),
        "details": f"Missing: {df['ipl_2027_status'].isna().sum()}",
    })

    checks.append({
        "role": role,
        "check": "No retired players in review shortlist",
        "result": not df["ipl_2027_status"].eq("RETIRED").any(),
        "details": (
            f"Retired: "
            f"{df['ipl_2027_status'].eq('RETIRED').sum()}"
        ),
    })

    checks.append({
        "role": role,
        "check": "Pressure sample counts valid",
        "result": (
            df["pressure_matches"].ge(0).all()
            and df["total_pressure_deliveries"].ge(0).all()
        ),
        "details": "No negative sample counts expected",
    })

    return checks


# ============================================================
# 5. RUN VALIDATION
# ============================================================

checks = []

checks.extend(
    validate_candidates(
        batters,
        "striker_id",
        "striker_name",
        "batting_reliability_score",
        "BATTER",
    )
)

checks.extend(
    validate_candidates(
        bowlers,
        "bowler_id",
        "bowler_name",
        "reliability_score",
        "BOWLER",
    )
)

validation = pd.DataFrame(checks)

validation["status"] = validation["result"].map({
    True: "PASS",
    False: "FAIL",
})


# ============================================================
# 6. PRINT VALIDATION RESULTS
# ============================================================

print("\n========== FINAL VALIDATION CHECKS ==========")

print(
    validation[
        ["role", "check", "status", "details"]
    ].to_string(index=False)
)

passed = int(validation["result"].sum())
failed = int((~validation["result"]).sum())

print("\nPassed:", passed)
print("Failed:", failed)


# ============================================================
# 7. IPL STATUS SUMMARY
# ============================================================

print("\n========== BATTING IPL STATUS ==========")

print(
    batters["ipl_2027_status"]
    .value_counts(dropna=False)
    .to_string()
)

print("\n========== BOWLING IPL STATUS ==========")

print(
    bowlers["ipl_2027_status"]
    .value_counts(dropna=False)
    .to_string()
)


# ============================================================
# 8. CANDIDATE CATEGORY SUMMARY
# ============================================================

print("\n========== BATTING CATEGORIES ==========")

print(
    batters["candidate_category"]
    .value_counts(dropna=False)
    .to_string()
)

print("\n========== BOWLING CATEGORIES ==========")

print(
    bowlers["candidate_category"]
    .value_counts(dropna=False)
    .to_string()
)


# ============================================================
# 9. UNVERIFIED PLAYER REVIEW
# ============================================================

print("\n========== UNVERIFIED BATTERS ==========")

print(
    batters.loc[
        batters["ipl_2027_status"] == "UNVERIFIED",
        "striker_name"
    ].to_list()
)

print("\n========== UNVERIFIED BOWLERS ==========")

print(
    bowlers.loc[
        bowlers["ipl_2027_status"] == "UNVERIFIED",
        "bowler_name"
    ].to_list()
)


# ============================================================
# 10. SAVE AUDIT
# ============================================================

validation.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved validation report:", OUTPUT_FILE)

if failed:
    raise SystemExit(
        "\nFINAL CANDIDATE VALIDATION FAILED. "
        "Review failed checks before proceeding."
    )

print("\nFINAL CANDIDATE VALIDATION PASSED")
