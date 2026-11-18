
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "deliveries"
    / "player_enriched_deliveries.csv"
)

OUTPUT_DIR = ROOT / "data" / "interim"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("=" * 60)
print("PERFORMANCE INPUT AUDIT")
print("=" * 60)

print("Total deliveries:", len(df))
print("Total columns:", len(df.columns))


# --------------------------------------------------
# 3. Audit important batting fields
# --------------------------------------------------

batting_fields = [
    "match_id",
    "innings",
    "striker_id",
    "mapped_batter_name",
    "runs",
    "is_four",
    "is_six",
    "dismissed_player_id",
    "is_wicket",
]

print("\nBATTING FIELD MISSING VALUES")

print(
    df[batting_fields]
    .isna()
    .sum()
    .to_string()
)


# --------------------------------------------------
# 4. Audit important bowling fields
# --------------------------------------------------

bowling_fields = [
    "bowler_id",
    "mapped_bowler_name",
    "bowler_runs_provisional",
    "bowler_wicket_provisional",
    "legal_ball_provisional",
    "non_bowler_wicket",
    "unresolved_wicket",
]

print("\nBOWLING FIELD MISSING VALUES")

print(
    df[bowling_fields]
    .isna()
    .sum()
    .to_string()
)


# --------------------------------------------------
# 5. Quality flag distribution
# --------------------------------------------------

quality_fields = [
    "flag_unresolved_extras",
    "flag_extras2_nonzero",
    "flag_missing_extras_with_runs",
    "flag_missing_dismissed_player",
    "flag_missing_bowler",
    "has_quality_flag",
]

print("\nQUALITY FLAG COUNTS")

for column in quality_fields:
    print(f"{column}: {df[column].sum()}")


# --------------------------------------------------
# 6. Delivery policy distribution
# --------------------------------------------------

print("\nLEGAL BALL POLICY DISTRIBUTION")

print(
    df["legal_ball_provisional"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nBOWLER RUNS DISTRIBUTION")

print(
    df["bowler_runs_provisional"]
    .describe()
    .to_string()
)


# --------------------------------------------------
# 7. Save audit
# --------------------------------------------------

missing_report = pd.DataFrame({
    "column": df.columns,
    "missing_count": df.isna().sum().values,
    "missing_percentage": (
        df.isna().mean().values * 100
    )
})

missing_report.to_csv(
    OUTPUT_DIR / "performance_input_missing_report.csv",
    index=False
)

print("\nSaved:")
print(OUTPUT_DIR / "performance_input_missing_report.csv")

print("\nAUDIT COMPLETED")
