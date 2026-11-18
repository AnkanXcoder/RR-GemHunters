
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data/processed/features/pressure_batting_features.csv"
)

OUTPUT_DIR = ROOT / "data/interim"

OUTPUT_FILE = OUTPUT_DIR / "pressure_batting_sample_audit.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("\n========== PRESSURE BATTING SAMPLE AUDIT ==========")

print("\nTotal batters:", len(df))

print("\nPressure delivery distribution:")
print(df["pressure_deliveries"].describe().round(2))

print("\nMatches distribution:")
print(df["matches"].describe().round(2))

print("\nSample-size categories:")
print(
    df["sample_size_category"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nBatters by match-count groups:")

df["match_sample_group"] = pd.cut(
    df["matches"],
    bins=[0, 3, 5, 10, 20, float("inf")],
    labels=[
        "1_TO_3_MATCHES",
        "4_TO_5_MATCHES",
        "6_TO_10_MATCHES",
        "11_TO_20_MATCHES",
        "ABOVE_20_MATCHES",
    ],
)

print(
    df["match_sample_group"]
    .value_counts(sort=False)
    .to_string()
)

print("\nTop batters with smallest samples:")

columns = [
    "striker_name",
    "pressure_deliveries",
    "matches",
    "pressure_runs",
    "striker_dismissals",
    "strike_rate",
    "batting_average",
]

print(
    df.sort_values(
        ["pressure_deliveries", "matches"],
        ascending=True,
    )[columns]
    .head(15)
    .to_string(index=False)
)

print("\nTop batters with largest samples:")

print(
    df.sort_values(
        ["pressure_deliveries", "matches"],
        ascending=False,
    )[columns]
    .head(15)
    .to_string(index=False)
)

print("\nMissing values:")
print(df.isna().sum().to_string())

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)
print("\nSAMPLE AUDIT COMPLETE")
