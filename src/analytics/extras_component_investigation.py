
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed" / "deliveries"
    / "regulation_deliveries.csv"
)

OUTPUT_DIR = ROOT / "data" / "interim"

OUTPUT_FILE = (
    OUTPUT_DIR / "extras_component_investigation.csv"
)

df = pd.read_csv(INPUT_FILE, low_memory=False)

# Find every record where extras_2_runs is nonzero.
investigation = df[
    pd.to_numeric(
        df["extras_2_runs"],
        errors="coerce"
    ).fillna(0) != 0
].copy()

# Include all useful delivery-level information.
columns = [
    "match_id",
    "innings",
    "overid",
    "over_ball_num",
    "innings_ball_num",
    "striker_name",
    "bowler_name",
    "runs",
    "extras_type",
    "extras1_runs",
    "extras_2_runs",
    "bowler_runs_provisional",
    "legal_ball_provisional",
    "flag_unresolved_extras",
    "has_quality_flag",
    "dismissal_type",
    "delivery_ist_time",
    "comp_name",
    "match_format",
    "season",
]

investigation = investigation[columns].copy()

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

investigation.to_csv(
    OUTPUT_FILE,
    index=False
)

print("=" * 65)
print("EXTRAS COMPONENT INVESTIGATION")
print("=" * 65)

print("Total records investigated:", len(investigation))

print("\nExtras type distribution:")
print(
    investigation["extras_type"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nExtras2 total by extras type:")
print(
    investigation.groupby(
        "extras_type",
        dropna=False
    )["extras_2_runs"]
    .agg(["count", "sum"])
    .to_string()
)

print("\nDetailed records:")
print(
    investigation.to_string(index=False)
)

print("\nSaved investigation:")
print(OUTPUT_FILE)
