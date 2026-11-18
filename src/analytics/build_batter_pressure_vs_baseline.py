
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DELIVERY_FILE = (
    ROOT / "data" / "processed" / "deliveries"
    / "regulation_deliveries.csv"
)

PRESSURE_FILE = (
    ROOT / "data" / "processed" / "features"
    / "pressure_state_deliveries.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed" / "features"
    / "batter_pressure_vs_baseline.csv"
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading pressure and regulation deliveries...")

pressure = pd.read_csv(PRESSURE_FILE, low_memory=False)
deliveries = pd.read_csv(DELIVERY_FILE, low_memory=False)

print("Pressure-state shape:", pressure.shape)
print("Regulation delivery shape:", deliveries.shape)


# ============================================================
# 3. REQUIRED COLUMNS
# ============================================================

keys = ["match_id", "innings", "innings_ball_num"]

pressure_required = keys + [
    "striker_id",
    "striker_name",
    "required_run_rate",
    "balls_remaining",
    "wickets_remaining",
]

delivery_required = keys + [
    "striker_id",
    "striker_name",
    "runs",
    "extras_type",
    "is_wicket",
    "dismissed_player_id",
    "is_four",
    "is_six",
    "has_quality_flag",
]

for col in pressure_required:
    if col not in pressure.columns:
        raise ValueError(f"Missing pressure column: {col}")

for col in delivery_required:
    if col not in deliveries.columns:
        raise ValueError(f"Missing delivery column: {col}")


# ============================================================
# 4. CHECK JOIN KEYS
# ============================================================

if pressure.duplicated(keys).any():
    raise ValueError("Duplicate pressure delivery keys.")

if deliveries.duplicated(keys).any():
    raise ValueError("Duplicate regulation delivery keys.")

print("\nDelivery key validation: PASS")


# ============================================================
# 5. DEFINE PRESSURE SITUATION
# ============================================================

pressure["is_pressure_delivery"] = (
    pressure["required_run_rate"].ge(10)
    & pressure["balls_remaining"].le(30)
    & pressure["wickets_remaining"].le(5)
)

print(
    "\nPressure deliveries:",
    int(pressure["is_pressure_delivery"].sum())
)

print(
    "Normal deliveries:",
    int((~pressure["is_pressure_delivery"]).sum())
)


# ============================================================
# 6. SELECT PRESSURE-ELIGIBLE MATCH DELIVERIES
# ============================================================

# All rows in pressure_state_deliveries belong to eligible
# second innings. The comparison baseline uses the same matches.

comparison = pressure[
    keys + [
        "striker_id",
        "striker_name",
        "is_pressure_delivery",
        "has_quality_flag",
    ]
].merge(
    deliveries[
        keys + [
            "striker_id",
            "runs",
            "extras_type",
            "is_wicket",
            "dismissed_player_id",
            "is_four",
            "is_six",
            "has_quality_flag",
        ]
    ],
    on=keys,
    how="left",
    suffixes=("_state", "_delivery"),
    validate="one_to_one",
    indicator=True
)


# ============================================================
# 7. JOIN VALIDATION
# ============================================================

print("\n========== JOIN VALIDATION ==========")

print(comparison["_merge"].value_counts().to_string())

if not comparison["_merge"].eq("both").all():
    raise ValueError("Some pressure-state deliveries failed to join.")

if not (
    comparison["striker_id_state"]
    == comparison["striker_id_delivery"]
).all():
    raise ValueError("Striker ID mismatch.")

if not (
    comparison["has_quality_flag_state"]
    == comparison["has_quality_flag_delivery"]
).all():
    raise ValueError("Quality flag mismatch.")

comparison = comparison.drop(
    columns=["_merge", "striker_id_delivery",
             "has_quality_flag_delivery"]
)

comparison = comparison.rename(
    columns={
        "striker_id_state": "striker_id",
        "has_quality_flag_state": "has_quality_flag",
    }
)


# ============================================================
# 8. CREATE BATTING METRICS
# ============================================================

comparison["runs"] = pd.to_numeric(
    comparison["runs"], errors="coerce"
).fillna(0)

comparison["extras_type"] = (
    comparison["extras_type"]
    .astype("string")
    .str.strip()
    .str.lower()
)

# Provisional balls-faced policy:
# Exclude wides, include other non-wide delivery records.
comparison["ball_faced"] = (
    comparison["extras_type"].ne("wide")
    & comparison["extras_type"].notna()
)

# Individual striker dismissal, not every team wicket.
comparison["striker_dismissal"] = (
    comparison["dismissed_player_id"].notna()
    & (
        comparison["dismissed_player_id"]
        == comparison["striker_id"]
    )
)

comparison["four"] = comparison["is_four"].fillna(False).astype(bool)
comparison["six"] = comparison["is_six"].fillna(False).astype(bool)

comparison["boundary"] = (
    comparison["four"] | comparison["six"]
)

comparison["boundary_runs"] = (
    comparison["four"].astype(int) * 4
    + comparison["six"].astype(int) * 6
)

comparison["quality_flag"] = (
    comparison["has_quality_flag"].fillna(False).astype(bool)
)


# ============================================================
# 9. AGGREGATE PRESSURE AND NORMAL PERFORMANCE
# ============================================================

def calculate_metrics(df, label):

    result = (
        df.groupby(
            ["striker_id", "striker_name"],
            dropna=False
        )
        .agg(
            deliveries=("runs", "size"),
            runs=("runs", "sum"),
            balls_faced=("ball_faced", "sum"),
            dismissals=("striker_dismissal", "sum"),
            fours=("four", "sum"),
            sixes=("six", "sum"),
            boundary_runs=("boundary_runs", "sum"),
            boundary_deliveries=("boundary", "sum"),
            quality_flagged_deliveries=("quality_flag", "sum"),
        )
        .reset_index()
    )

    result[f"{label}_strike_rate"] = np.where(
        result["balls_faced"] > 0,
        result["runs"] / result["balls_faced"] * 100,
        np.nan
    )

    result[f"{label}_batting_average"] = np.where(
        result["dismissals"] > 0,
        result["runs"] / result["dismissals"],
        np.nan
    )

    result[f"{label}_dismissals_per_100_balls"] = np.where(
        result["balls_faced"] > 0,
        result["dismissals"] / result["balls_faced"] * 100,
        np.nan
    )

    result[f"{label}_boundary_percentage"] = np.where(
        result["balls_faced"] > 0,
        result["boundary_deliveries"] / result["balls_faced"] * 100,
        np.nan
    )

    result[f"{label}_quality_flag_percentage"] = (
        result["quality_flagged_deliveries"]
        / result["deliveries"] * 100
    )

    rename_columns = {
        "deliveries": f"{label}_deliveries",
        "runs": f"{label}_runs",
        "balls_faced": f"{label}_balls_faced",
        "dismissals": f"{label}_dismissals",
        "fours": f"{label}_fours",
        "sixes": f"{label}_sixes",
        "boundary_runs": f"{label}_boundary_runs",
        "boundary_deliveries": f"{label}_boundary_deliveries",
        "quality_flagged_deliveries": (
            f"{label}_quality_flagged_deliveries"
        ),
    }

    return result.rename(columns=rename_columns)


pressure_data = comparison[
    comparison["is_pressure_delivery"]
].copy()

normal_data = comparison[
    ~comparison["is_pressure_delivery"]
].copy()

pressure_metrics = calculate_metrics(
    pressure_data, "pressure"
)

normal_metrics = calculate_metrics(
    normal_data, "normal"
)


# ============================================================
# 10. MERGE PLAYER PERFORMANCE
# ============================================================

final = pressure_metrics.merge(
    normal_metrics,
    on=["striker_id", "striker_name"],
    how="outer",
    validate="one_to_one"
)

count_columns = [
    col for col in final.columns
    if col.endswith((
        "_deliveries",
        "_runs",
        "_balls_faced",
        "_dismissals",
        "_fours",
        "_sixes",
        "_boundary_runs",
        "_boundary_deliveries",
        "_quality_flagged_deliveries",
    ))
]

final[count_columns] = final[count_columns].fillna(0)


# ============================================================
# 11. CALCULATE PRESSURE PERFORMANCE DIFFERENCE
# ============================================================

final["strike_rate_change"] = (
    final["pressure_strike_rate"]
    - final["normal_strike_rate"]
)

final["batting_average_change"] = (
    final["pressure_batting_average"]
    - final["normal_batting_average"]
)

final["dismissal_rate_change"] = (
    final["pressure_dismissals_per_100_balls"]
    - final["normal_dismissals_per_100_balls"]
)

final["boundary_percentage_change"] = (
    final["pressure_boundary_percentage"]
    - final["normal_boundary_percentage"]
)


# ============================================================
# 12. ADD EVIDENCE FLAGS
# ============================================================

final["has_pressure_sample"] = (
    final["pressure_deliveries"] > 0
)

final["has_normal_sample"] = (
    final["normal_deliveries"] > 0
)

final["comparison_valid"] = (
    final["has_pressure_sample"]
    & final["has_normal_sample"]
    & final["pressure_balls_faced"].gt(0)
    & final["normal_balls_faced"].gt(0)
)

final["comparison_quality_review"] = (
    final["pressure_quality_flagged_deliveries"]
    + final["normal_quality_flagged_deliveries"]
) > 0


# ============================================================
# 13. SORT AND SAVE
# ============================================================

final = final.sort_values(
    by=["pressure_strike_rate", "pressure_deliveries"],
    ascending=[False, False]
).reset_index(drop=True)

final.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# 14. RECONCILIATION
# ============================================================

print("\n========== RECONCILIATION ==========")

print("Pressure rows:", len(pressure_data))
print("Normal rows:", len(normal_data))
print("Combined rows:", len(comparison))

print(
    "Pressure delivery reconciliation:",
    len(pressure_data) == int(
        comparison["is_pressure_delivery"].sum()
    )
)

print(
    "Total delivery reconciliation:",
    len(pressure_data) + len(normal_data) == len(comparison)
)

print("Unique batters:", final["striker_id"].nunique())
print("Duplicate batter IDs:", final["striker_id"].duplicated().sum())

print(
    "Valid comparisons:",
    int(final["comparison_valid"].sum())
)

print(
    "Comparisons requiring quality review:",
    int(final["comparison_quality_review"].sum())
)

assert len(pressure_data) + len(normal_data) == len(comparison)
assert final["striker_id"].is_unique


# ============================================================
# 15. PREVIEW
# ============================================================

preview = [
    "striker_name",
    "pressure_deliveries",
    "normal_deliveries",
    "pressure_runs",
    "normal_runs",
    "pressure_strike_rate",
    "normal_strike_rate",
    "strike_rate_change",
    "pressure_dismissals_per_100_balls",
    "normal_dismissals_per_100_balls",
    "dismissal_rate_change",
    "comparison_valid",
]

print("\n========== BATTER PRESSURE VS BASELINE ==========")

print(
    final[preview]
    .head(20)
    .round(2)
    .to_string(index=False)
)

print("\nSaved:", OUTPUT_FILE)
print("\nBATTER PRESSURE VS BASELINE ANALYSIS COMPLETE")
