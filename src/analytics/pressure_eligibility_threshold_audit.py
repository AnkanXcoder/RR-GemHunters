
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data/processed/features/pressure_batting_features.csv"
)

OUTPUT_DIR = ROOT / "data/interim"

OUTPUT_FILE = (
    OUTPUT_DIR / "pressure_eligibility_threshold_sensitivity.csv"
)

df = pd.read_csv(INPUT_FILE, low_memory=False)

# Candidate thresholds for sensitivity analysis.
delivery_thresholds = [5, 10, 15, 20, 30, 50]
match_thresholds = [1, 3, 5, 8, 10]

results = []

total_batters = len(df)

for min_deliveries in delivery_thresholds:
    for min_matches in match_thresholds:

        eligible = df[
            (df["pressure_deliveries"] >= min_deliveries)
            & (df["matches"] >= min_matches)
        ]

        count = len(eligible)

        results.append({
            "minimum_pressure_deliveries": min_deliveries,
            "minimum_matches": min_matches,
            "eligible_batters": count,
            "excluded_batters": total_batters - count,
            "eligible_percentage": round(
                count / total_batters * 100, 2
            ),
        })

audit = pd.DataFrame(results)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

audit.to_csv(OUTPUT_FILE, index=False)

print("\n========== ELIGIBILITY THRESHOLD AUDIT ==========")

print("Total batters:", total_batters)

print("\nEligible batters by threshold combination:")

print(
    audit.pivot(
        index="minimum_pressure_deliveries",
        columns="minimum_matches",
        values="eligible_batters",
    ).to_string()
)

print("\nEligibility percentage:")

print(
    audit.pivot(
        index="minimum_pressure_deliveries",
        columns="minimum_matches",
        values="eligible_percentage",
    ).to_string()
)

print("\nDetailed threshold results:")

print(audit.to_string(index=False))

print("\nSaved:", OUTPUT_FILE)

print("\nTHRESHOLD SENSITIVITY AUDIT COMPLETE")
