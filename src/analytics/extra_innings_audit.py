
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DELIVERY_FILE = (
    ROOT / "data" / "processed" / "deliveries"
    / "player_enriched_deliveries.csv"
)

MATCH_FILE = (
    ROOT / "data" / "raw" / "official"
    / "matches.csv"
)

df = pd.read_csv(DELIVERY_FILE, low_memory=False)
matches = pd.read_csv(MATCH_FILE, low_memory=False)

print("=" * 65)
print("EXTRA INNINGS AUDIT")
print("=" * 65)

# Identify innings beyond the standard first and second innings
extra = df[df["innings"] > 2].copy()

print("\nExtra innings delivery records:", len(extra))

if len(extra) == 0:
    print("No extra innings found.")
else:
    summary = (
        extra.groupby(
            ["match_id", "innings"],
            as_index=False
        )
        .agg(
            delivery_records=("innings", "size"),
            batting_team_id=("batting_team_id", "first"),
            bowling_team_id=("bowling_team_id", "first"),
            batting_team=("mapped_batter_name", "first"),
            first_date=("match_date", "first"),
            competition=("comp_name", "first"),
            format=("match_format", "first"),
        )
    )

    summary = summary.merge(
        matches[
            [
                "match_id",
                "home_team",
                "away_team",
                "result",
                "win_team_id",
            ]
        ],
        on="match_id",
        how="left",
        validate="many_to_one",
    )

    print("\nEXTRA INNINGS DETAILS")
    print(summary.to_string(index=False))

    output = (
        ROOT / "data" / "interim"
        / "extra_innings_audit.csv"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)

    print("\nSaved:", output)

print("\nAUDIT COMPLETED")
