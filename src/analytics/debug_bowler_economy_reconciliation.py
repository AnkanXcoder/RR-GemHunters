
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = ROOT / "data/processed/features/pressure_state_deliveries.csv"
REGULATION_FILE = ROOT / "data/processed/deliveries/regulation_deliveries.csv"
SCORE_FILE = ROOT / "data/processed/features/pressure_bowler_reliability_scores.csv"

KEYS = ["match_id", "innings", "innings_ball_num"]

print("Loading datasets...")

pressure = pd.read_csv(FEATURE_FILE, low_memory=False)
regulation = pd.read_csv(REGULATION_FILE, low_memory=False)
scores = pd.read_csv(SCORE_FILE, low_memory=False)

# Keep only regulation columns needed for the audit.
reg_cols = KEYS + [
    "bowler_id",
    "bowler_name",
    "bowler_runs_provisional",
    "legal_ball_provisional",
    "extras_type",
]

# Avoid duplicate bowler columns by renaming before joining.
regulation_subset = regulation[reg_cols].rename(
    columns={
        "bowler_id": "audit_bowler_id",
        "bowler_name": "audit_bowler_name",
        "legal_ball_provisional": "audit_legal_ball",
    }
)

data = pressure.merge(
    regulation_subset,
    on=KEYS,
    how="left",
    validate="one_to_one",
    indicator=True,
)

print("\nJOIN STATUS:")
print(data["_merge"].value_counts())

data = data.loc[data["_merge"] == "both"].copy()

# Combined pressure definition.
pressure_mask = (
    (data["required_run_rate"] >= 10)
    & (data["balls_remaining"] <= 30)
    & (data["wickets_remaining"] <= 5)
)

pressure_data = data.loc[pressure_mask].copy()

print("\nPRESSURE DATA:")
print("Rows:", len(pressure_data))
print("Unique bowlers:", pressure_data["audit_bowler_id"].nunique())

pressure_data["audit_runs"] = (
    pd.to_numeric(
        pressure_data["bowler_runs_provisional"],
        errors="coerce"
    ).fillna(0)
)

pressure_data["audit_legal_ball"] = (
    pd.to_numeric(
        pressure_data["audit_legal_ball"],
        errors="coerce"
    ).fillna(0)
)

summary = (
    pressure_data.groupby("audit_bowler_id")
    .agg(
        bowler_name=("audit_bowler_name", "first"),
        deliveries=("match_id", "size"),
        legal_balls=("audit_legal_ball", "sum"),
        bowler_runs=("audit_runs", "sum"),
    )
    .reset_index()
)

summary["audit_economy"] = (
    summary["bowler_runs"] * 6
    / summary["legal_balls"].replace(0, pd.NA)
)

comparison = summary.merge(
    scores[
        [
            "bowler_id",
            "overall_pressure_economy",
            "total_pressure_deliveries",
            "total_legal_balls",
            "total_runs_conceded",
        ]
    ],
    left_on="audit_bowler_id",
    right_on="bowler_id",
    how="inner",
    validate="one_to_one",
)

comparison["delivery_difference"] = (
    comparison["deliveries"]
    - comparison["total_pressure_deliveries"]
)

comparison["legal_ball_difference"] = (
    comparison["legal_balls"]
    - comparison["total_legal_balls"]
)

comparison["run_difference"] = (
    comparison["bowler_runs"]
    - comparison["total_runs_conceded"]
)

comparison["economy_difference"] = (
    comparison["audit_economy"]
    - comparison["overall_pressure_economy"]
)

print("\n========== ECONOMY RECONCILIATION ==========")

columns = [
    "bowler_name",
    "deliveries",
    "total_pressure_deliveries",
    "delivery_difference",
    "legal_balls",
    "total_legal_balls",
    "legal_ball_difference",
    "bowler_runs",
    "total_runs_conceded",
    "run_difference",
    "audit_economy",
    "overall_pressure_economy",
    "economy_difference",
]

print(
    comparison[columns]
    .assign(abs_difference=lambda x: x["economy_difference"].abs())
    .sort_values("abs_difference", ascending=False)
    .drop(columns="abs_difference")
    .head(20)
    .to_string(index=False)
)

print("\n========== SUMMARY ==========")
print("Compared bowlers:", len(comparison))
print(
    "Exact delivery matches:",
    (comparison["delivery_difference"] == 0).sum()
)
print(
    "Exact legal-ball matches:",
    (comparison["legal_ball_difference"] == 0).sum()
)
print(
    "Exact run matches:",
    (comparison["run_difference"] == 0).sum()
)

output = ROOT / "data/interim/bowler_economy_reconciliation.csv"
comparison.to_csv(output, index=False)

print("\nSaved:", output)
print("\nAudit completed.")
