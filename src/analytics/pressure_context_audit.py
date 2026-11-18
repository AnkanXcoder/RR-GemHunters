
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed" / "deliveries"
    / "player_enriched_deliveries.csv"
)

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("=" * 65)
print("PRESSURE CONTEXT DATA AUDIT")
print("=" * 65)

# Match and innings information
print("\n1. MATCH FORMATS")
print(df.groupby("match_format")["match_id"].nunique())

print("\n2. INNINGS DISTRIBUTION")
print(df["innings"].value_counts().sort_index())

# Target and revised overs information
target_fields = [
    "revised_target",
    "first_inn_revised_overs",
    "second_inn_revised_overs",
]

print("\n3. TARGET AND REVISED OVERS MISSING VALUES")
print(df[target_fields].isna().sum())

print("\n4. REVISED TARGET RECORDS")
print(df[df["revised_target"].notna()][
    ["match_id", "innings", "revised_target"]
].drop_duplicates().head(20).to_string(index=False))

# Match date and competition
print("\n5. MATCH DATE MISSING VALUES")
print(df["match_date"].isna().sum())

print("\n6. UNIQUE MATCHES")
print(df["match_id"].nunique())

print("\n7. INNINGS WITH REVISED TARGET")
print(
    df[df["revised_target"].notna()]
    .groupby(["match_id", "innings"])
    .size()
    .groupby(level=1)
    .size()
)

print("\nPRESSURE CONTEXT AUDIT COMPLETED")
