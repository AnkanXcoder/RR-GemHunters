
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

FILES = {
    "pressure_states": (
        ROOT / "data/processed/features/"
        "pressure_state_deliveries.csv"
    ),
    "regulation_deliveries": (
        ROOT / "data/processed/deliveries/"
        "regulation_deliveries.csv"
    ),
}

print("\n========== PRESSURE BOWLING INPUT AUDIT ==========")

for name, path in FILES.items():

    print(f"\n\n{'=' * 55}")
    print(f"DATASET: {name}")
    print(f"PATH: {path}")

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, nrows=5)

    print("\nSample shape:", df.shape)
    print("\nColumns:")
    for col in df.columns:
        print(f" - {col}")

    print("\nSample records:")
    print(df.head(3).to_string(index=False))

    print("\nPotential bowling-related columns:")

    keywords = [
        "bowler",
        "run",
        "extra",
        "wicket",
        "dismiss",
        "legal",
        "ball",
        "quality",
        "match",
        "innings",
    ]

    for col in df.columns:
        if any(word in col.lower() for word in keywords):
            print(f" - {col}")

print("\n========== AUDIT COMPLETE ==========")
