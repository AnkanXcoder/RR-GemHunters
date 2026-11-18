
from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PRESSURE_FILE = (
    ROOT / "data/processed/features/pressure_state_deliveries.csv"
)

DELIVERY_FILE = (
    ROOT / "data/processed/deliveries/regulation_deliveries.csv"
)

OUTPUT_DIR = ROOT / "data/processed/features"

OUTPUT_FILE = OUTPUT_DIR / "pressure_batting_features.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading datasets...")

if not PRESSURE_FILE.exists():
    raise FileNotFoundError(f"Pressure file not found: {PRESSURE_FILE}")

if not DELIVERY_FILE.exists():
    raise FileNotFoundError(f"Delivery file not found: {DELIVERY_FILE}")

pressure = pd.read_csv(PRESSURE_FILE, low_memory=False)
deliveries = pd.read_csv(DELIVERY_FILE, low_memory=False)

print("Pressure dataset:", pressure.shape)
print("Regulation dataset:", deliveries.shape)


# ============================================================
# 3. FILTER COMBINED-PRESSURE STATES
# ============================================================

combined_pressure = pressure[
    (pressure["required_run_rate"] >= 10)
    & (pressure["balls_remaining"] <= 30)
    & (pressure["wickets_remaining"] <= 5)
].copy()

print("\nCombined-pressure deliveries:", len(combined_pressure))

if combined_pressure.empty:
    raise ValueError("No combined-pressure deliveries found.")


# ============================================================
# 4. VALIDATE REQUIRED COLUMNS
# ============================================================

keys = ["match_id", "innings", "innings_ball_num"]

pressure_required = keys + [
    "striker_id",
    "striker_name",
    "runs",
    "is_wicket",
    "has_quality_flag",
]

delivery_required = keys + [
    "striker_id",
    "dismissed_player_id",
    "is_wicket",
    "runs",
    "extras_type",
    "has_quality_flag",
]

missing_pressure = [
    col for col in pressure_required
    if col not in combined_pressure.columns
]

missing_delivery = [
    col for col in delivery_required
    if col not in deliveries.columns
]

if missing_pressure:
    raise ValueError(
        f"Missing pressure columns: {missing_pressure}"
    )

if missing_delivery:
    raise ValueError(
        f"Missing delivery columns: {missing_delivery}"
    )


# ============================================================
# 5. VALIDATE DELIVERY KEYS
# ============================================================

pressure_duplicates = combined_pressure.duplicated(keys).sum()
delivery_duplicates = deliveries.duplicated(keys).sum()

print("\nDuplicate pressure keys:", pressure_duplicates)
print("Duplicate regulation keys:", delivery_duplicates)

if pressure_duplicates > 0:
    raise ValueError("Duplicate keys found in pressure dataset.")

if delivery_duplicates > 0:
    raise ValueError("Duplicate keys found in regulation dataset.")


# ============================================================
# 6. SELECT SOURCE DISMISSAL INFORMATION
# ============================================================

source = deliveries[
    delivery_required
].copy()


# ============================================================
# 7. JOIN PRESSURE AND DELIVERY DATA
# ============================================================

merged = combined_pressure.merge(
    source,
    on=keys,
    how="left",
    suffixes=("_pressure", "_source"),
    indicator=True,
    validate="one_to_one",
)

join_counts = merged["_merge"].value_counts()

print("\nJOIN STATUS:")
print(join_counts.to_string())

if not (merged["_merge"] == "both").all():
    raise ValueError("Some pressure deliveries did not match source data.")

# Check that important values agree between both datasets.

checks = {
    "striker_id": (
        merged["striker_id_pressure"]
        == merged["striker_id_source"]
    ),
    "runs": (
        merged["runs_pressure"]
        == merged["runs_source"]
    ),
    "is_wicket": (
        merged["is_wicket_pressure"]
        == merged["is_wicket_source"]
    ),
    "quality_flag": (
        merged["has_quality_flag_pressure"]
        == merged["has_quality_flag_source"]
    ),
}

print("\nJOIN CONSISTENCY CHECKS:")

for name, result in checks.items():
    mismatch_count = int((~result).sum())
    print(f"{name} mismatches: {mismatch_count}")

    if mismatch_count > 0:
        raise ValueError(
            f"Mismatch detected in {name}."
        )

print("\nDelivery join validation: PASS")


# ============================================================
# 8. CREATE BATTING FEATURES
# ============================================================

# Restore convenient column names after the merge.

merged["striker_id"] = merged["striker_id_pressure"]
merged["striker_name"] = merged["striker_name"]

merged["batter_runs"] = merged["runs_pressure"]

# Provisional balls-faced definition:
# Exclude wides, retain other non-wide deliveries.
merged["ball_faced"] = (
    merged["extras_type"]
    .fillna("")
    .str.strip()
    .str.lower()
    .ne("wide")
    .astype(int)
)

# Boundary events based on batter runs, not team runs.

merged["fours"] = (
    merged["batter_runs"] == 4
).astype(int)

merged["sixes"] = (
    merged["batter_runs"] == 6
).astype(int)

# Individual dismissal:
# dismissed_player_id must match the striker_id.

merged["striker_dismissal"] = (
    merged["dismissed_player_id"].notna()
    & (
        merged["dismissed_player_id"]
        == merged["striker_id_source"]
    )
).astype(int)

# Preserve delivery-quality flags.

merged["quality_flag"] = (
    merged["has_quality_flag_pressure"]
    .fillna(False)
    .astype(bool)
)


# ============================================================
# 9. AGGREGATE BY BATTER
# ============================================================

group_columns = [
    "striker_id",
    "striker_name",
]

features = (
    merged.groupby(
        group_columns,
        dropna=False,
    )
    .agg(
        pressure_deliveries=("match_id", "size"),
        matches=("match_id", "nunique"),
        pressure_runs=("batter_runs", "sum"),
        balls_faced=("ball_faced", "sum"),
        fours=("fours", "sum"),
        sixes=("sixes", "sum"),
        striker_dismissals=("striker_dismissal", "sum"),
        quality_flagged_deliveries=("quality_flag", "sum"),
    )
    .reset_index()
)


# ============================================================
# 10. CALCULATE BATTING METRICS
# ============================================================

features["strike_rate"] = (
    features["pressure_runs"]
    / features["balls_faced"].replace(0, pd.NA)
) * 100

features["batting_average"] = (
    features["pressure_runs"]
    / features["striker_dismissals"].replace(0, pd.NA)
)

features["boundary_runs"] = (
    features["fours"] * 4
    + features["sixes"] * 6
)

features["boundary_percentage"] = (
    features["boundary_runs"]
    / features["pressure_runs"].replace(0, pd.NA)
) * 100

features["quality_flag_percentage"] = (
    features["quality_flagged_deliveries"]
    / features["pressure_deliveries"]
) * 100


# ============================================================
# 11. ADD SAMPLE-SIZE INDICATORS
# ============================================================

features["sample_size_category"] = pd.cut(
    features["pressure_deliveries"],
    bins=[-1, 5, 15, 30, float("inf")],
    labels=[
        "VERY_SMALL",
        "SMALL",
        "MODERATE",
        "LARGE",
    ],
)

# This is a descriptive sample-size category,
# not a final player eligibility decision.


# ============================================================
# 12. SORT FEATURES
# ============================================================

features = features.sort_values(
    by=[
        "pressure_runs",
        "pressure_deliveries",
    ],
    ascending=[
        False,
        False,
    ],
).reset_index(drop=True)


# ============================================================
# 13. VALIDATION
# ============================================================

print("\n========== VALIDATION REPORT ==========")

print(
    "Input combined-pressure deliveries:",
    len(combined_pressure),
)

print(
    "Joined deliveries:",
    len(merged),
)

print(
    "Aggregated pressure deliveries:",
    features["pressure_deliveries"].sum(),
)

print(
    "Unique batters:",
    features["striker_id"].nunique(),
)

print(
    "Duplicate batter records:",
    features["striker_id"].duplicated().sum(),
)

print(
    "Pressure runs reconciliation:",
    int(features["pressure_runs"].sum())
    == int(merged["batter_runs"].sum()),
)

print(
    "Balls-faced reconciliation:",
    int(features["balls_faced"].sum())
    == int(merged["ball_faced"].sum()),
)

print(
    "Dismissal reconciliation:",
    int(features["striker_dismissals"].sum())
    == int(merged["striker_dismissal"].sum()),
)

if features["striker_id"].duplicated().any():
    raise ValueError("Duplicate batter feature records found.")

if features["pressure_deliveries"].sum() != len(merged):
    raise ValueError("Pressure delivery aggregation mismatch.")


# ============================================================
# 14. SAVE OUTPUT
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

features.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# 15. DISPLAY RESULTS
# ============================================================

print("\n========== PRESSURE BATTING FEATURE REPORT ==========")

summary_columns = [
    "striker_id",
    "striker_name",
    "pressure_deliveries",
    "matches",
    "pressure_runs",
    "balls_faced",
    "fours",
    "sixes",
    "striker_dismissals",
    "strike_rate",
    "batting_average",
    "boundary_percentage",
    "quality_flag_percentage",
    "sample_size_category",
]

print("\nTOP 15 BATTERS BY PRESSURE RUNS:")

print(
    features[summary_columns]
    .head(15)
    .to_string(index=False)
)

print("\nTOTAL PRESSURE RUNS:", features["pressure_runs"].sum())
print("TOTAL BALLS FACED:", features["balls_faced"].sum())
print("TOTAL FOURS:", features["fours"].sum())
print("TOTAL SIXES:", features["sixes"].sum())
print("TOTAL STRIKER DISMISSALS:", features["striker_dismissals"].sum())

print(
    "QUALITY-FLAGGED DELIVERIES:",
    features["quality_flagged_deliveries"].sum(),
)

print("\nOUTPUT FILE:")
print(OUTPUT_FILE)

print("\nPRESSURE BATTING FEATURE BUILD COMPLETE")
