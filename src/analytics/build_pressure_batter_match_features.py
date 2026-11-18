
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

ELIGIBILITY_FILE = (
    ROOT / "data/processed/features/pressure_batter_eligibility.csv"
)

OUTPUT_DIR = ROOT / "data/processed/features"

OUTPUT_FILE = (
    OUTPUT_DIR / "pressure_batter_match_features.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading datasets...")

for file_path in [
    PRESSURE_FILE,
    DELIVERY_FILE,
    ELIGIBILITY_FILE,
]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

pressure = pd.read_csv(
    PRESSURE_FILE,
    low_memory=False
)

deliveries = pd.read_csv(
    DELIVERY_FILE,
    low_memory=False
)

eligibility = pd.read_csv(
    ELIGIBILITY_FILE,
    low_memory=False
)

print("Pressure data:", pressure.shape)
print("Regulation deliveries:", deliveries.shape)
print("Eligibility data:", eligibility.shape)


# ============================================================
# 3. FILTER COMBINED-PRESSURE DELIVERIES
# ============================================================

pressure = pressure[
    (pressure["required_run_rate"] >= 10)
    & (pressure["balls_remaining"] <= 30)
    & (pressure["wickets_remaining"] <= 5)
].copy()

print(
    "\nCombined-pressure deliveries:",
    len(pressure)
)

if pressure.empty:
    raise ValueError(
        "No combined-pressure deliveries found."
    )


# ============================================================
# 4. VALIDATE REQUIRED COLUMNS
# ============================================================

keys = [
    "match_id",
    "innings",
    "innings_ball_num",
]

pressure_required = keys + [
    "striker_id",
    "striker_name",
    "runs",
    "is_wicket",
    "has_quality_flag",
    "match_date",
    "comp_name",
    "match_format",
    "season",
]

delivery_required = keys + [
    "striker_id",
    "dismissed_player_id",
    "runs",
    "is_wicket",
    "extras_type",
    "has_quality_flag",
]

eligibility_required = [
    "striker_id",
    "primary_eligible",
    "relaxed_eligible",
    "strict_eligible",
    "evidence_category",
]

for name, dataframe, required_columns in [
    ("Pressure", pressure, pressure_required),
    ("Deliveries", deliveries, delivery_required),
    ("Eligibility", eligibility, eligibility_required),
]:
    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {name} dataset: {missing}"
        )


# ============================================================
# 5. VALIDATE DELIVERY KEYS
# ============================================================

pressure_duplicates = pressure.duplicated(keys).sum()

delivery_duplicates = deliveries.duplicated(keys).sum()

print("\nDuplicate pressure keys:", pressure_duplicates)
print("Duplicate regulation keys:", delivery_duplicates)

if pressure_duplicates > 0:
    raise ValueError(
        "Duplicate delivery keys in pressure dataset."
    )

if delivery_duplicates > 0:
    raise ValueError(
        "Duplicate delivery keys in regulation dataset."
    )


# ============================================================
# 6. PREPARE SOURCE DELIVERY DATA
# ============================================================

source_columns = delivery_required

source = deliveries[source_columns].copy()


# ============================================================
# 7. JOIN SOURCE DISMISSAL INFORMATION
# ============================================================

df = pressure.merge(
    source,
    on=keys,
    how="left",
    suffixes=("_pressure", "_source"),
    indicator=True,
    validate="one_to_one",
)

print("\nJOIN STATUS:")
print(df["_merge"].value_counts().to_string())

if not (df["_merge"] == "both").all():
    raise ValueError(
        "Unmatched pressure delivery records found."
    )


# ============================================================
# 8. JOIN CONSISTENCY VALIDATION
# ============================================================

checks = {
    "striker_id": (
        df["striker_id_pressure"]
        == df["striker_id_source"]
    ),
    "runs": (
        df["runs_pressure"]
        == df["runs_source"]
    ),
    "is_wicket": (
        df["is_wicket_pressure"]
        == df["is_wicket_source"]
    ),
    "quality_flag": (
        df["has_quality_flag_pressure"]
        == df["has_quality_flag_source"]
    ),
}

print("\nJOIN CONSISTENCY CHECKS:")

for column_name, check_result in checks.items():

    mismatch_count = int((~check_result).sum())

    print(
        f"{column_name} mismatches:",
        mismatch_count
    )

    if mismatch_count > 0:
        raise ValueError(
            f"{column_name} mismatch detected."
        )

print("\nDelivery join validation: PASS")


# ============================================================
# 9. RESTORE ORIGINAL COLUMN NAMES
# ============================================================

# Both datasets contained striker_id.
# Pandas therefore added suffixes during the merge.
# Restore the expected column names explicitly.

df["striker_id"] = df["striker_id_pressure"]

df["striker_name"] = df["striker_name"]

df["batter_runs"] = df["runs_pressure"]

df["dismissed_player_id"] = df["dismissed_player_id"]


# ============================================================
# 10. CREATE MATCH-LEVEL BATTING FEATURES
# ============================================================

# Provisional balls-faced definition:
# Exclude wides and retain other non-wide deliveries.

df["balls_faced"] = (
    df["extras_type"]
    .fillna("")
    .str.strip()
    .str.lower()
    .ne("wide")
    .astype(int)
)

# Boundary events are based on batter runs.

df["fours"] = (
    df["batter_runs"] == 4
).astype(int)

df["sixes"] = (
    df["batter_runs"] == 6
).astype(int)


# Individual batter dismissal:
# dismissed_player_id must match striker_id.

df["striker_dismissal"] = (
    df["dismissed_player_id"].notna()
    & (
        df["dismissed_player_id"]
        == df["striker_id"]
    )
).astype(int)


# Preserve quality flags.

df["quality_flag"] = (
    df["has_quality_flag_pressure"]
    .fillna(False)
    .astype(bool)
)


# ============================================================
# 11. AGGREGATE BY BATTER AND MATCH
# ============================================================

group_columns = [
    "striker_id",
    "striker_name",
    "match_id",
    "match_date",
    "comp_name",
    "match_format",
    "season",
]

match_features = (
    df.groupby(
        group_columns,
        dropna=False,
    )
    .agg(
        pressure_deliveries=(
            "match_id",
            "size"
        ),

        pressure_runs=(
            "batter_runs",
            "sum"
        ),

        balls_faced=(
            "balls_faced",
            "sum"
        ),

        fours=(
            "fours",
            "sum"
        ),

        sixes=(
            "sixes",
            "sum"
        ),

        striker_dismissals=(
            "striker_dismissal",
            "sum"
        ),

        quality_flagged_deliveries=(
            "quality_flag",
            "sum"
        ),

        combined_pressure_states=(
            "match_id",
            "size"
        ),
    )
    .reset_index()
)

print("\nBatter-match aggregation completed.")


# ============================================================
# 12. CALCULATE MATCH-LEVEL PERFORMANCE METRICS
# ============================================================

match_features["strike_rate"] = (
    match_features["pressure_runs"]
    / match_features["balls_faced"].replace(
        0,
        pd.NA
    )
) * 100


match_features["boundary_runs"] = (
    match_features["fours"] * 4
    + match_features["sixes"] * 6
)


match_features["boundary_percentage"] = (
    match_features["boundary_runs"]
    / match_features["pressure_runs"].replace(
        0,
        pd.NA
    )
) * 100


match_features["quality_flag_percentage"] = (
    match_features["quality_flagged_deliveries"]
    / match_features["pressure_deliveries"]
) * 100


# ============================================================
# 13. VALIDATE ELIGIBILITY DATA
# ============================================================

if eligibility["striker_id"].duplicated().any():
    raise ValueError(
        "Duplicate striker IDs in eligibility dataset."
    )


# ============================================================
# 14. ATTACH ELIGIBILITY CLASSIFICATION
# ============================================================

eligibility_columns = [
    "striker_id",
    "primary_eligible",
    "relaxed_eligible",
    "strict_eligible",
    "evidence_category",
]

match_features = match_features.merge(
    eligibility[eligibility_columns],
    on="striker_id",
    how="left",
    validate="many_to_one",
)

if match_features["evidence_category"].isna().any():
    raise ValueError(
        "Some batters lack eligibility classification."
    )


# ============================================================
# 15. MATCH-LEVEL VALIDATION
# ============================================================

print(
    "\n========== MATCH-LEVEL VALIDATION =========="
)

print(
    "Combined-pressure deliveries:",
    len(df)
)

print(
    "Match-level records:",
    len(match_features)
)

print(
    "Unique batters:",
    match_features["striker_id"].nunique()
)

print(
    "Unique matches:",
    match_features["match_id"].nunique()
)


# Check unique batter-match records.

duplicate_player_match = match_features.duplicated(
    ["striker_id", "match_id"]
).sum()

print(
    "Duplicate batter-match records:",
    duplicate_player_match
)

if duplicate_player_match > 0:
    raise ValueError(
        "Duplicate batter-match records found."
    )


# Check delivery reconciliation.

delivery_check = (
    match_features["pressure_deliveries"].sum()
    == len(df)
)

print(
    "Pressure delivery reconciliation:",
    delivery_check
)

if not delivery_check:
    raise ValueError(
        "Pressure delivery aggregation mismatch."
    )


# Check batter-run reconciliation.

run_check = (
    match_features["pressure_runs"].sum()
    == df["batter_runs"].sum()
)

print(
    "Pressure run reconciliation:",
    run_check
)

if not run_check:
    raise ValueError(
        "Pressure run aggregation mismatch."
    )


# Check balls-faced reconciliation.

balls_check = (
    match_features["balls_faced"].sum()
    == df["balls_faced"].sum()
)

print(
    "Balls-faced reconciliation:",
    balls_check
)

if not balls_check:
    raise ValueError(
        "Balls-faced aggregation mismatch."
    )


# Check dismissal reconciliation.

dismissal_check = (
    match_features["striker_dismissals"].sum()
    == df["striker_dismissal"].sum()
)

print(
    "Dismissal reconciliation:",
    dismissal_check
)

if not dismissal_check:
    raise ValueError(
        "Dismissal aggregation mismatch."
    )


# ============================================================
# 16. SAVE OUTPUT
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

match_features.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 17. DISPLAY MATCH-LEVEL RESULTS
# ============================================================

print(
    "\n========== MATCH-LEVEL FEATURE REPORT =========="
)

print("\nTop 15 batter-match performances:")

display_columns = [
    "striker_name",
    "match_id",
    "pressure_deliveries",
    "pressure_runs",
    "balls_faced",
    "striker_dismissals",
    "strike_rate",
    "evidence_category",
]

print(
    match_features.sort_values(
        ["pressure_runs", "strike_rate"],
        ascending=[False, False],
    )[display_columns]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 18. ELIGIBILITY DISTRIBUTION
# ============================================================

print("\nEligibility distribution:")

eligibility_distribution = (
    match_features.groupby(
        "evidence_category"
    )["striker_id"]
    .nunique()
)

print(
    eligibility_distribution.to_string()
)


# ============================================================
# 19. FINAL REPORT
# ============================================================

print("\nOutput shape:", match_features.shape)

print("\nSaved output:")
print(OUTPUT_FILE)

print(
    "\nMATCH-LEVEL PRESSURE FEATURES COMPLETE"
)
