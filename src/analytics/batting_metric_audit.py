
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed" / "deliveries"
    / "player_enriched_deliveries.csv"
)

OUTPUT_DIR = ROOT / "data" / "interim"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("=" * 60)
print("BATTING METRIC DEFINITION AUDIT")
print("=" * 60)

# 1. Batter runs
print("\n1. RUN DISTRIBUTION")
print(df["runs"].value_counts(dropna=False).sort_index())

# 2. Extras classification
print("\n2. EXTRAS TYPE DISTRIBUTION")
print(df["extras_type"].fillna("none").value_counts())

# 3. Batter balls faced candidates
# Standard convention: exclude wides, include no-balls.
df["batter_ball_faced_candidate"] = (
    df["extras_type"].fillna("none").str.lower() != "wide"
)

print("\n3. BATTER BALLS FACED CANDIDATE")
print(df["batter_ball_faced_candidate"].value_counts())

# 4. Dismissal distribution
wickets = df[df["is_wicket"] == True].copy()

print("\n4. DISMISSAL TYPE DISTRIBUTION")
print(wickets["dismissal_type"].fillna("missing").value_counts())

# 5. Identify dismissed batter vs striker
wickets["dismissed_striker"] = (
    wickets["dismissed_player_id"] == wickets["striker_id"]
)

print("\n5. DISMISSED PLAYER COMPARISON")
print(wickets["dismissed_striker"].value_counts(dropna=False))

print("\n6. WICKET RECORDS WITH MISSING DISMISSED PLAYER")
print(
    wickets[wickets["dismissed_player_id"].isna()][
        [
            "match_id",
            "innings",
            "striker_id",
            "dismissal_type",
            "bowler_id",
        ]
    ].to_string(index=False)
)

# 7. Save dismissal audit
dismissal_report = (
    wickets.groupby("dismissal_type", dropna=False)
    .agg(
        wicket_records=("is_wicket", "size"),
        dismissed_striker_count=("dismissed_striker", "sum"),
    )
    .reset_index()
)

dismissal_report.to_csv(
    OUTPUT_DIR / "batting_dismissal_audit.csv",
    index=False,
)

print("\nSaved:")
print(OUTPUT_DIR / "batting_dismissal_audit.csv")

print("\nAUDIT COMPLETED")
