
from pathlib import Path
import pandas as pd
import numpy as np

# =====================================================
# 1. PATHS
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

PRESSURE_FILE = (
    ROOT / "data/processed/features/"
    "pressure_state_deliveries.csv"
)

DELIVERY_FILE = (
    ROOT / "data/processed/deliveries/"
    "regulation_deliveries.csv"
)

OUTPUT_DIR = ROOT / "data/processed/features"

OUTPUT_FILE = (
    OUTPUT_DIR / "pressure_bowler_match_features.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. LOAD DATA
# =====================================================

print("\nLoading datasets...")

pressure = pd.read_csv(PRESSURE_FILE, low_memory=False)
deliveries = pd.read_csv(DELIVERY_FILE, low_memory=False)

print("All match-state deliveries:", pressure.shape)
print("Regulation deliveries:", deliveries.shape)


# =====================================================
# 3. REQUIRED COLUMN VALIDATION
# =====================================================

JOIN_KEYS = [
    "match_id",
    "innings",
    "innings_ball_num",
]

PRESSURE_COLUMNS = [
    "required_run_rate",
    "balls_remaining",
    "wickets_remaining",
]

pressure_required = JOIN_KEYS + PRESSURE_COLUMNS + [
    "match_date",
    "comp_name",
    "match_format",
    "season",
    "bowler_id",
    "bowler_name",
    "team_runs_this_delivery",
    "is_wicket",
    "has_quality_flag",
]

delivery_required = JOIN_KEYS + [
    "bowler_id",
    "bowler_name",
    "bowler_runs_provisional",
    "bowler_wicket_provisional",
    "legal_ball_provisional",
    "is_four",
    "is_six",
    "is_wicket",
    "has_quality_flag",
    "non_bowler_wicket",
]

for col in pressure_required:
    if col not in pressure.columns:
        raise ValueError(f"Missing pressure column: {col}")

for col in delivery_required:
    if col not in deliveries.columns:
        raise ValueError(f"Missing delivery column: {col}")

print("\nRequired columns validation: PASS")


# =====================================================
# 4. DUPLICATE KEY AUDIT
# =====================================================

print("\nChecking duplicate join keys...")

pressure_duplicates = pressure.duplicated(JOIN_KEYS).sum()
delivery_duplicates = deliveries.duplicated(JOIN_KEYS).sum()

print("Pressure duplicate keys:", pressure_duplicates)
print("Delivery duplicate keys:", delivery_duplicates)

if pressure_duplicates > 0 or delivery_duplicates > 0:
    raise ValueError("Duplicate delivery keys detected.")


# =====================================================
# 5. APPLY COMBINED PRESSURE FILTER
# =====================================================

print("\nApplying combined pressure definition...")

for col in PRESSURE_COLUMNS:
    pressure[col] = pd.to_numeric(
        pressure[col],
        errors="coerce"
    )

pressure["is_combined_pressure"] = (
    (pressure["required_run_rate"] >= 10)
    & (pressure["balls_remaining"] <= 30)
    & (pressure["wickets_remaining"] <= 5)
)

print("\n========== PRESSURE FILTER AUDIT ==========")

print("All second-innings deliveries:", len(pressure))

print(
    "Combined-pressure deliveries:",
    pressure["is_combined_pressure"].sum()
)

print(
    "Non-pressure deliveries:",
    (~pressure["is_combined_pressure"]).sum()
)

print(
    "Missing pressure-state values:",
    pressure[PRESSURE_COLUMNS].isna().any(axis=1).sum()
)

# Retain only combined-pressure deliveries.
pressure = pressure[
    pressure["is_combined_pressure"]
].copy()

if pressure.empty:
    raise ValueError("No combined-pressure deliveries found.")

# Every retained row must satisfy all three conditions.
assert pressure["required_run_rate"].ge(10).all()
assert pressure["balls_remaining"].le(30).all()
assert pressure["wickets_remaining"].le(5).all()

print("\nCombined pressure filter validation: PASS")


# =====================================================
# 6. SELECT REGULATION DELIVERY COLUMNS
# =====================================================

source_columns = JOIN_KEYS + [
    "bowler_id",
    "bowler_name",
    "bowler_runs_provisional",
    "bowler_wicket_provisional",
    "legal_ball_provisional",
    "is_four",
    "is_six",
    "is_wicket",
    "has_quality_flag",
    "non_bowler_wicket",
]

source = deliveries[source_columns].copy()


# =====================================================
# 7. MERGE PRESSURE DELIVERIES
# =====================================================

print("\nJoining combined-pressure deliveries...")

df = pressure.merge(
    source,
    on=JOIN_KEYS,
    how="left",
    suffixes=("_pressure", "_source"),
    validate="one_to_one",
    indicator=True,
)

print("\nJOIN STATUS:")
print(df["_merge"].value_counts())

matched = (df["_merge"] == "both").sum()
unmatched = (df["_merge"] == "left_only").sum()

print("\nMatched deliveries:", matched)
print("Unmatched deliveries:", unmatched)

if unmatched > 0:
    raise ValueError("Unmatched pressure deliveries found.")


# =====================================================
# 8. JOIN CONSISTENCY CHECKS
# =====================================================

bowler_mismatch = (
    df["bowler_id_pressure"] != df["bowler_id_source"]
).sum()

bowler_name_mismatch = (
    df["bowler_name_pressure"].fillna("")
    != df["bowler_name_source"].fillna("")
).sum()

wicket_mismatch = (
    df["is_wicket_pressure"].fillna(False).astype(bool)
    != df["is_wicket_source"].fillna(False).astype(bool)
).sum()

quality_mismatch = (
    df["has_quality_flag_pressure"].fillna(False).astype(bool)
    != df["has_quality_flag_source"].fillna(False).astype(bool)
).sum()

legal_ball_mismatch = (
    df["legal_ball_provisional_pressure"].fillna(False).astype(bool)
    != df["legal_ball_provisional_source"].fillna(False).astype(bool)
).sum()

print("\n========== JOIN CONSISTENCY CHECKS ==========")

print("Bowler ID mismatches:", bowler_mismatch)
print("Bowler name mismatches:", bowler_name_mismatch)
print("Wicket mismatches:", wicket_mismatch)
print("Quality flag mismatches:", quality_mismatch)
print("Legal ball mismatches:", legal_ball_mismatch)

if any([
    bowler_mismatch,
    bowler_name_mismatch,
    wicket_mismatch,
    quality_mismatch,
    legal_ball_mismatch,
]):
    raise ValueError("Join consistency validation failed.")

print("\nDelivery join validation: PASS")


# =====================================================
# 9. RESTORE METRIC COLUMNS
# =====================================================

df["bowler_id"] = df["bowler_id_pressure"]
df["bowler_name"] = df["bowler_name_pressure"]

df["bowler_runs"] = pd.to_numeric(
    df["bowler_runs_provisional"],
    errors="coerce"
)

df["bowler_wicket"] = (
    df["bowler_wicket_provisional"]
    .fillna(False)
    .astype(bool)
)

df["legal_ball"] = (
    df["legal_ball_provisional_source"]
    .fillna(False)
    .astype(bool)
)

df["team_runs_this_delivery"] = pd.to_numeric(
    df["team_runs_this_delivery"],
    errors="coerce"
)

df["quality_flag"] = (
    df["has_quality_flag_pressure"]
    .fillna(False)
    .astype(bool)
)

df["is_four"] = df["is_four"].fillna(False).astype(bool)
df["is_six"] = df["is_six"].fillna(False).astype(bool)

df["non_bowler_wicket"] = (
    df["non_bowler_wicket"]
    .fillna(False)
    .astype(bool)
)

missing_inputs = df[
    ["bowler_runs", "team_runs_this_delivery"]
].isna().sum()

print("\nMissing metric inputs:")
print(missing_inputs)

if missing_inputs.sum() > 0:
    raise ValueError("Missing numerical bowling inputs found.")

if df["bowler_id"].isna().any():
    raise ValueError("Missing bowler IDs in pressure deliveries.")


# =====================================================
# 10. BOWLING PERFORMANCE INDICATORS
# =====================================================

# Legal delivery with zero total runs scored by batting team.
df["is_dot_ball"] = (
    df["legal_ball"]
    & (df["team_runs_this_delivery"] == 0)
)

# Batter hits a four or six during pressure.
df["boundary_conceded"] = (
    df["is_four"] | df["is_six"]
)

df["quality_flag_count"] = (
    df["quality_flag"].astype(int)
)


# =====================================================
# 11. BOWLER-MATCH AGGREGATION
# =====================================================

print("\nAggregating combined-pressure bowler-match features...")

GROUP_COLUMNS = [
    "bowler_id",
    "bowler_name",
    "match_id",
    "match_date",
    "comp_name",
    "match_format",
    "season",
]

match_features = (
    df.groupby(
        GROUP_COLUMNS,
        dropna=False
    )
    .agg(
        pressure_deliveries=("innings_ball_num", "size"),
        legal_balls=("legal_ball", "sum"),
        runs_conceded=("bowler_runs", "sum"),
        bowler_wickets=("bowler_wicket", "sum"),
        dot_balls=("is_dot_ball", "sum"),
        fours_conceded=("is_four", "sum"),
        sixes_conceded=("is_six", "sum"),
        boundaries_conceded=("boundary_conceded", "sum"),
        quality_flagged_deliveries=("quality_flag_count", "sum"),
        non_bowler_dismissals=("non_bowler_wicket", "sum"),
    )
    .reset_index()
)


# =====================================================
# 12. CALCULATE BOWLING METRICS
# =====================================================

match_features["economy_rate"] = np.where(
    match_features["legal_balls"] > 0,
    match_features["runs_conceded"] * 6
    / match_features["legal_balls"],
    np.nan,
)

match_features["bowling_strike_rate"] = np.where(
    match_features["bowler_wickets"] > 0,
    match_features["legal_balls"]
    / match_features["bowler_wickets"],
    np.nan,
)

match_features["dot_ball_percentage"] = np.where(
    match_features["legal_balls"] > 0,
    match_features["dot_balls"] * 100
    / match_features["legal_balls"],
    np.nan,
)

match_features["boundary_percentage"] = np.where(
    match_features["legal_balls"] > 0,
    match_features["boundaries_conceded"] * 100
    / match_features["legal_balls"],
    np.nan,
)

match_features["quality_flag_percentage"] = (
    match_features["quality_flagged_deliveries"]
    * 100
    / match_features["pressure_deliveries"]
)


# =====================================================
# 13. VALIDATION
# =====================================================

print("\n========== MATCH-LEVEL BOWLING VALIDATION ==========")

print("Combined-pressure deliveries:", len(df))
print("Bowler-match records:", len(match_features))

print(
    "Unique bowlers:",
    match_features["bowler_id"].nunique()
)

print(
    "Unique matches:",
    match_features["match_id"].nunique()
)

duplicate_count = match_features.duplicated(
    ["bowler_id", "match_id"]
).sum()

delivery_check = (
    match_features["pressure_deliveries"].sum() == len(df)
)

runs_check = np.isclose(
    match_features["runs_conceded"].sum(),
    df["bowler_runs"].sum()
)

wickets_check = (
    match_features["bowler_wickets"].sum()
    == df["bowler_wicket"].sum()
)

legal_balls_check = (
    match_features["legal_balls"].sum()
    == df["legal_ball"].sum()
)

print("\nDuplicate bowler-match records:", duplicate_count)
print("Delivery reconciliation:", delivery_check)
print("Runs reconciliation:", runs_check)
print("Wickets reconciliation:", wickets_check)
print("Legal balls reconciliation:", legal_balls_check)

assert duplicate_count == 0
assert delivery_check
assert runs_check
assert wickets_check
assert legal_balls_check

print("\nAll match-level validations: PASS")


# =====================================================
# 14. PERFORMANCE REPORT
# =====================================================

print("\n========== TOP PRESSURE BOWLER-MATCH PERFORMANCES ==========")

report_columns = [
    "bowler_name",
    "match_id",
    "pressure_deliveries",
    "legal_balls",
    "runs_conceded",
    "bowler_wickets",
    "economy_rate",
    "dot_ball_percentage",
    "boundary_percentage",
]

report = match_features.sort_values(
    ["bowler_wickets", "economy_rate"],
    ascending=[False, True],
    na_position="last"
)

print(
    report[report_columns]
    .head(15)
    .to_string(index=False)
)


# =====================================================
# 15. SAVE OUTPUT
# =====================================================

match_features.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nOutput shape:", match_features.shape)

print("\nSaved output:")
print(OUTPUT_FILE)

print("\nCOMBINED-PRESSURE BOWLER MATCH FEATURES COMPLETE")
