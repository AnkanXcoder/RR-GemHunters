
import pandas as pd
from pathlib import Path

# =====================================================
# 1. PATHS
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

PRESSURE_FILE = (
    ROOT / "data/processed/features/pressure_state_deliveries.csv"
)
REGULATION_FILE = (
    ROOT / "data/processed/deliveries/regulation_deliveries.csv"
)
SHORTLIST_FILE = (
    ROOT / "data/processed/features/bowler_candidate_shortlist_status_audited.csv"
)

OUTPUT_DIR = ROOT / "data/interim"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR / "shortlisted_bowler_pressure_economy_sensitivity.csv"
)

KEYS = ["match_id", "innings", "innings_ball_num"]

# =====================================================
# 2. LOAD DATA
# =====================================================

print("\nLoading datasets...")

pressure = pd.read_csv(PRESSURE_FILE, low_memory=False)
regulation = pd.read_csv(REGULATION_FILE, low_memory=False)
shortlist = pd.read_csv(SHORTLIST_FILE, low_memory=False)

# =====================================================
# 3. VALIDATE DELIVERY KEYS
# =====================================================

assert not pressure.duplicated(KEYS).any(), "Duplicate pressure keys"
assert not regulation.duplicated(KEYS).any(), "Duplicate regulation keys"

# =====================================================
# 4. JOIN EXTRAS INFORMATION
# =====================================================

extra_columns = KEYS + [
    "extras_type",
    "bowler_runs_provisional",
]

data = pressure.merge(
    regulation[extra_columns],
    on=KEYS,
    how="left",
    validate="one_to_one",
    indicator=True,
)

print("\nJOIN STATUS:")
print(data["_merge"].value_counts().to_string())

if not data["_merge"].eq("both").all():
    raise ValueError("Some pressure deliveries did not match regulation data.")

data = data.drop(columns="_merge")

# =====================================================
# 5. FILTER COMBINED PRESSURE DELIVERIES
# =====================================================

pressure_mask = (
    data["required_run_rate"].ge(10)
    & data["balls_remaining"].le(30)
    & data["wickets_remaining"].le(5)
)

data = data.loc[pressure_mask].copy()

other_mask = (
    data["extras_type"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("other")
)

# =====================================================
# 6. BUILD RUN-ATTRIBUTION SCENARIOS
# =====================================================

data["run_A"] = data["bowler_runs_provisional"]

data["run_B"] = (
    data["run_A"]
    + data["runs"].where(other_mask, 0)
)

data["run_C"] = (
    data["run_B"]
    + data["extras1_runs"].where(other_mask, 0)
)

data["run_D"] = (
    data["run_C"]
    + data["extras_2_runs"].where(other_mask, 0)
)

# =====================================================
# 7. AGGREGATE BY BOWLER
# =====================================================

summary = (
    data.groupby("bowler_id")
    .agg(
        bowler_name=("bowler_name", "first"),
        pressure_deliveries=("match_id", "size"),
        legal_balls=("legal_ball_provisional", "sum"),
        other_deliveries=("extras_type", lambda x: (
            x.astype(str).str.strip().str.lower().eq("other").sum()
        )),
        runs_A=("run_A", "sum"),
        runs_B=("run_B", "sum"),
        runs_C=("run_C", "sum"),
        runs_D=("run_D", "sum"),
    )
    .reset_index()
)

# =====================================================
# 8. JOIN SHORTLIST AND RELIABILITY SCORES
# =====================================================

candidate_columns = [
    "bowler_id",
    "reliability_score",
    "provisional_rank",
    "overall_pressure_economy",
]

candidates = (
    shortlist[candidate_columns]
    .drop_duplicates("bowler_id")
)

summary = summary.merge(
    candidates,
    on="bowler_id",
    how="inner",
    validate="one_to_one",
)

# =====================================================
# 9. CALCULATE ECONOMY RATES
# =====================================================

for scenario in ["A", "B", "C", "D"]:
    summary[f"economy_{scenario}"] = (
        summary[f"runs_{scenario}"] * 6
        / summary["legal_balls"].replace(0, pd.NA)
    )

# =====================================================
# 10. PRINT RESULTS
# =====================================================

print("\n========== PRESSURE ECONOMY SENSITIVITY ==========")

print("Shortlisted bowlers with pressure data:", len(summary))
print(
    "Shortlisted bowlers with other pressure deliveries:",
    int(summary["other_deliveries"].gt(0).sum()),
)

columns = [
    "bowler_name",
    "provisional_rank",
    "reliability_score",
    "other_deliveries",
    "legal_balls",
    "overall_pressure_economy",
    "economy_A",
    "economy_B",
    "economy_C",
    "economy_D",
]

print(
    summary.sort_values("provisional_rank")[columns]
    .to_string(index=False)
)

# =====================================================
# 11. SAVE
# =====================================================

summary.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)
print("\nAudit completed.")
