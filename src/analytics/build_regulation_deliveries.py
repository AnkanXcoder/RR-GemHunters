
from pathlib import Path
import pandas as pd


# ==================================================
# 1. PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed" / "deliveries"
    / "player_enriched_deliveries.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "deliveries"

REGULATION_OUTPUT = (
    OUTPUT_DIR / "regulation_deliveries.csv"
)

SUPER_OVER_OUTPUT = (
    OUTPUT_DIR / "super_over_deliveries.csv"
)


# ==================================================
# 2. LOAD DATA
# ==================================================

print("=" * 65)
print("REGULATION AND SUPER OVER SEPARATION")
print("=" * 65)

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("\nOriginal delivery records:", len(df))


# ==================================================
# 3. SPLIT INNINGS
# ==================================================

regulation = df[df["innings"].isin([1, 2])].copy()

super_overs = df[df["innings"].isin([3, 4])].copy()

unexpected = df[~df["innings"].isin([1, 2, 3, 4])].copy()


# ==================================================
# 4. VALIDATION
# ==================================================

print("\nREGULATION INNINGS")
print("Delivery records:", len(regulation))
print("Unique matches:", regulation["match_id"].nunique())

print("\nSUPER OVERS")
print("Delivery records:", len(super_overs))
print("Unique matches:", super_overs["match_id"].nunique())

print("\nUNEXPECTED INNINGS RECORDS:", len(unexpected))

if len(unexpected) > 0:
    raise ValueError(
        "Unexpected innings found. Investigate before saving."
    )


# ==================================================
# 5. CHECK MATCH COVERAGE
# ==================================================

regulation_matches = set(regulation["match_id"].unique())

super_over_matches = set(super_overs["match_id"].unique())

overlap = regulation_matches.intersection(
    super_over_matches
)

print("\nMatches with both regulation and Super Over:",
      len(overlap))


# ==================================================
# 6. SAVE DATASETS
# ==================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

regulation.to_csv(
    REGULATION_OUTPUT,
    index=False
)

super_overs.to_csv(
    SUPER_OVER_OUTPUT,
    index=False
)


# ==================================================
# 7. FINAL CHECK
# ==================================================

reconstructed_total = (
    len(regulation) + len(super_overs)
)

if reconstructed_total != len(df):
    raise ValueError(
        "Delivery records were lost during separation."
    )

print("\n" + "=" * 65)
print("FINAL VALIDATION")
print("=" * 65)

print("Original records:", len(df))
print("Regulation records:", len(regulation))
print("Super Over records:", len(super_overs))
print("Reconstructed records:", reconstructed_total)

print("\nSaved files:")
print(REGULATION_OUTPUT)
print(SUPER_OVER_OUTPUT)

print("\nREGULATION SEPARATION COMPLETED SUCCESSFULLY")
