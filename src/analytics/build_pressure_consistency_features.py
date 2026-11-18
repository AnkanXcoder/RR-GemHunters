
from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================
# PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data/processed/features/"
    "pressure_batter_match_features.csv"
)

OUTPUT_DIR = ROOT / "data/processed/features"
OUTPUT_FILE = OUTPUT_DIR / "pressure_batter_consistency_features.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

print("\nLoading match-level pressure features...")

df = pd.read_csv(INPUT_FILE)

print("Input shape:", df.shape)

required_columns = [
    "striker_id",
    "striker_name",
    "match_id",
    "pressure_deliveries",
    "pressure_runs",
    "balls_faced",
    "striker_dismissals",
    "strike_rate",
    "evidence_category"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

if df.duplicated(["striker_id", "match_id"]).any():
    raise ValueError("Duplicate batter-match records found.")

# Keep records with a valid strike rate for SR-based metrics.
df["strike_rate"] = pd.to_numeric(df["strike_rate"], errors="coerce")


# ==========================================
# PLAYER-LEVEL CONSISTENCY FEATURES
# ==========================================

print("\nCalculating player-level consistency...")

def calculate_consistency(group):

    valid_sr = group["strike_rate"].dropna()

    return pd.Series({

        "pressure_matches": group["match_id"].nunique(),

        "total_pressure_deliveries": group["pressure_deliveries"].sum(),

        "total_pressure_runs": group["pressure_runs"].sum(),

        "total_balls_faced": group["balls_faced"].sum(),

        "total_striker_dismissals": group["striker_dismissals"].sum(),

        "average_match_strike_rate": valid_sr.mean(),

        "median_match_strike_rate": valid_sr.median(),

        "std_match_strike_rate": valid_sr.std(),

        "matches_sr_above_150": (valid_sr > 150).sum(),

        "matches_sr_above_200": (valid_sr > 200).sum(),

        "percentage_matches_sr_above_150":
            (valid_sr > 150).mean() * 100 if len(valid_sr) else np.nan,

        "percentage_matches_sr_above_200":
            (valid_sr > 200).mean() * 100 if len(valid_sr) else np.nan,

        "matches_with_boundary":
            (group["fours"] + group["sixes"] > 0).sum(),

        "matches_with_boundary_percentage":
            (group["fours"] + group["sixes"] > 0).mean() * 100,

        "average_pressure_runs_per_match":
            group["pressure_runs"].mean(),

        "valid_strike_rate_matches": len(valid_sr)
    })


player_features = (
    df.groupby(
        ["striker_id", "striker_name", "evidence_category"],
        dropna=False
    )
    .apply(calculate_consistency)
    .reset_index()
)

# ==========================================
# ADD OVERALL PRESSURE STRIKE RATE
# ==========================================

player_features["overall_pressure_strike_rate"] = np.where(
    player_features["total_balls_faced"] > 0,
    player_features["total_pressure_runs"]
    / player_features["total_balls_faced"] * 100,
    np.nan
)


# ==========================================
# VALIDATION
# ==========================================

print("\n========== CONSISTENCY VALIDATION ==========")

print("Input match-level records:", len(df))
print("Output player-level records:", len(player_features))
print("Unique input batters:", df["striker_id"].nunique())
print("Unique output batters:", player_features["striker_id"].nunique())

print(
    "Duplicate player records:",
    player_features.duplicated(["striker_id"]).sum()
)

print(
    "Pressure delivery reconciliation:",
    player_features["total_pressure_deliveries"].sum()
    == df["pressure_deliveries"].sum()
)

print(
    "Pressure run reconciliation:",
    player_features["total_pressure_runs"].sum()
    == df["pressure_runs"].sum()
)

print(
    "Balls-faced reconciliation:",
    player_features["total_balls_faced"].sum()
    == df["balls_faced"].sum()
)

assert player_features["striker_id"].nunique() == df["striker_id"].nunique()
assert player_features.duplicated(["striker_id"]).sum() == 0

# ==========================================
# REPORT
# ==========================================

print("\n========== PLAYER CONSISTENCY REPORT ==========")

print(
    player_features[
        [
            "striker_name",
            "pressure_matches",
            "total_pressure_runs",
            "average_match_strike_rate",
            "median_match_strike_rate",
            "std_match_strike_rate",
            "percentage_matches_sr_above_150",
            "percentage_matches_sr_above_200",
            "evidence_category"
        ]
    ].head(15).to_string(index=False)
)

print("\nEvidence distribution:")
print(player_features["evidence_category"].value_counts())

# ==========================================
# SAVE
# ==========================================

player_features.to_csv(OUTPUT_FILE, index=False)

print("\nOutput shape:", player_features.shape)
print("Saved:", OUTPUT_FILE)
print("\nPLAYER CONSISTENCY FEATURE GENERATION COMPLETE")
