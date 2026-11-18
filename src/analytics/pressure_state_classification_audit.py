
from pathlib import Path

import pandas as pd

INPUT = Path(
    "data/processed/features/pressure_state_deliveries.csv"
)

OUTPUT_DIR = Path("data/interim")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT, low_memory=False)

# 1. Validate required columns
required = [
    "match_id",
    "striker_id",
    "required_run_rate",
    "balls_remaining",
    "wickets_remaining",
    "runs_required",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# 2. Preserve continuous values and create audit bands
df["rrr_audit_band"] = pd.cut(
    df["required_run_rate"],
    bins=[-float("inf"), 6, 8, 10, 15, 20, 30, float("inf")],
    labels=[
        "LOW_0_TO_6",
        "MODERATE_6_TO_8",
        "HIGH_8_TO_10",
        "VERY_HIGH_10_TO_15",
        "EXTREME_15_TO_20",
        "EXTREME_20_TO_30",
        "EXTREME_ABOVE_30",
    ],
    right=False,
)

df["time_audit_band"] = pd.cut(
    df["balls_remaining"],
    bins=[0, 6, 12, 30, 60, 120],
    labels=[
        "FINAL_1_TO_6",
        "FINAL_7_TO_12",
        "LATE_13_TO_30",
        "MIDDLE_31_TO_60",
        "EARLY_61_TO_120",
    ],
    right=True,
)

df["wicket_audit_band"] = pd.cut(
    df["wickets_remaining"],
    bins=[0, 2, 5, 8, 10],
    labels=[
        "CRITICAL_1_TO_2",
        "LOW_3_TO_5",
        "MODERATE_6_TO_8",
        "HIGH_9_TO_10",
    ],
    right=True,
)

# 3. Individual dimension summaries
summaries = {
    "rrr_audit_band": "required_run_rate",
    "time_audit_band": "balls_remaining",
    "wicket_audit_band": "wickets_remaining",
}

for band, metric in summaries.items():
    print(f"\n{band.upper()} DISTRIBUTION")
    print(
        df.groupby(band, observed=False)
        .agg(
            delivery_states=("match_id", "size"),
            unique_matches=("match_id", "nunique"),
            unique_batters=("striker_id", "nunique"),
            average_metric=(metric, "mean"),
        )
        .round(2)
        .to_string()
    )

# 4. Combined pressure dimensions
combined = (
    df.groupby(
        ["rrr_audit_band", "time_audit_band", "wicket_audit_band"],
        observed=False,
    )
    .agg(
        delivery_states=("match_id", "size"),
        unique_matches=("match_id", "nunique"),
        unique_batters=("striker_id", "nunique"),
    )
    .reset_index()
)

combined = combined.sort_values(
    "delivery_states", ascending=False
)

combined.to_csv(
    OUTPUT_DIR / "pressure_state_combination_audit.csv",
    index=False,
)

# 5. Basic integrity checks
checks = {
    "all_rows_retained": len(df) == 88035,
    "no_missing_rrr": df["required_run_rate"].notna().all(),
    "no_missing_time_band": df["time_audit_band"].notna().all(),
    "no_missing_wicket_band": df["wicket_audit_band"].notna().all(),
    "no_negative_runs_required": (df["runs_required"] >= 0).all(),
    "no_duplicate_delivery_states": not df.duplicated(
        ["match_id", "innings", "innings_ball_num"]
    ).any(),
}

print("\nTOP 20 COMBINED PRESSURE STATES")
print(combined.head(20).to_string(index=False))

print("\nVALIDATION")
for name, passed in checks.items():
    print(f"{name}: {'PASS' if passed else 'FAIL'}")

print(f"\nTOTAL ROWS: {len(df):,}")
print(f"COMBINATIONS: {len(combined):,}")

if not all(checks.values()):
    raise AssertionError("One or more validation checks failed.")

print("\nPRESSURE CLASSIFICATION AUDIT COMPLETED.")
