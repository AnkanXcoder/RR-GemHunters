
from pathlib import Path

import pandas as pd

INPUT = Path(
    "data/processed/features/pressure_state_deliveries.csv"
)

OUTPUT_DIR = Path("data/interim")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT, low_memory=False)

required = [
    "match_id",
    "striker_id",
    "striker_name",
    "runs",
    "is_wicket",
    "required_run_rate",
    "balls_remaining",
    "wickets_remaining",
    "has_quality_flag",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# Classify pressure exposure for this audit only.
df["scoring_pressure"] = df["required_run_rate"] >= 10

df["time_pressure"] = df["balls_remaining"] <= 30

df["resource_pressure"] = df["wickets_remaining"] <= 5

df["combined_pressure"] = (
    df["scoring_pressure"]
    & df["time_pressure"]
    & df["resource_pressure"]
)

# Aggregate at batter level.
summary = (
    df.groupby(["striker_id", "striker_name"])
    .agg(
        delivery_states=("match_id", "size"),
        matches=("match_id", "nunique"),
        runs=("runs", "sum"),
        wickets_lost_while_batting=("is_wicket", "sum"),
        scoring_pressure_states=("scoring_pressure", "sum"),
        time_pressure_states=("time_pressure", "sum"),
        resource_pressure_states=("resource_pressure", "sum"),
        combined_pressure_states=("combined_pressure", "sum"),
        quality_flagged_states=("has_quality_flag", "sum"),
    )
    .reset_index()
)

summary["scoring_pressure_share"] = (
    summary["scoring_pressure_states"]
    / summary["delivery_states"]
)

summary["combined_pressure_share"] = (
    summary["combined_pressure_states"]
    / summary["delivery_states"]
)

summary = summary.sort_values(
    ["combined_pressure_states", "matches"],
    ascending=False,
)

output = OUTPUT_DIR / "batter_pressure_exposure_audit.csv"
summary.to_csv(output, index=False)

print("=" * 72)
print("BATTER PRESSURE EXPOSURE AUDIT")
print("=" * 72)

print("\nSUMMARY")
print("Unique batters:", summary["striker_id"].nunique())
print("Total delivery states:", summary["delivery_states"].sum())
print("Total matches represented:", df["match_id"].nunique())

print("\nPRESSURE EXPOSURE TOTALS")
print("Scoring pressure:", df["scoring_pressure"].sum())
print("Time pressure:", df["time_pressure"].sum())
print("Resource pressure:", df["resource_pressure"].sum())
print("Combined pressure:", df["combined_pressure"].sum())

print("\nTOP 20 BATTERS BY COMBINED PRESSURE EXPOSURE")
print(
    summary[
        [
            "striker_name",
            "matches",
            "delivery_states",
            "combined_pressure_states",
            "combined_pressure_share",
            "quality_flagged_states",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

checks = {
    "all_delivery_states_accounted": (
        summary["delivery_states"].sum() == len(df)
    ),
    "unique_batter_keys": (
        not summary["striker_id"].duplicated().any()
    ),
    "no_negative_exposure": (
        (summary["combined_pressure_states"] >= 0).all()
    ),
    "combined_pressure_within_all_states": (
        (
            summary["combined_pressure_states"]
            <= summary["delivery_states"]
        ).all()
    ),
}

print("\nVALIDATION")
for name, passed in checks.items():
    print(f"{name}: {'PASS' if passed else 'FAIL'}")

if not all(checks.values()):
    raise AssertionError("Validation failed.")

print("\nOUTPUT:")
print(output)
print("\nBATTER PRESSURE EXPOSURE AUDIT COMPLETED.")
