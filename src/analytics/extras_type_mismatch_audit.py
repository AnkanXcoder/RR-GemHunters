
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    ROOT / "data" / "raw" / "official"
    / "ball_by_ball.csv"
)

CLEAN_FILE = (
    ROOT / "data" / "processed" / "deliveries"
    / "regulation_deliveries.csv"
)

OUTPUT_DIR = ROOT / "data" / "interim"
OUTPUT_FILE = OUTPUT_DIR / "extras_type_mismatches.csv"

KEYS = [
    "match_id",
    "innings",
    "overid",
    "over_ball_num",
]

def normalize_extras(series):
    return (
        series.fillna("none")
        .astype(str)
        .str.strip()
        .str.lower()
    )


def main():

    raw = pd.read_csv(RAW_FILE, low_memory=False)
    clean = pd.read_csv(CLEAN_FILE, low_memory=False)

    # Select only the 35 records under investigation.
    clean["extras_2_runs"] = pd.to_numeric(
        clean["extras_2_runs"],
        errors="coerce"
    ).fillna(0)

    unusual = clean[
        clean["extras_2_runs"] != 0
    ].copy()

    raw_subset = raw[
        KEYS + [
            "extras_type",
            "runs",
            "extras1_runs",
            "extras_2_runs",
        ]
    ].copy()

    merged = unusual.merge(
        raw_subset,
        on=KEYS,
        how="left",
        suffixes=("_clean", "_raw"),
        validate="one_to_one",
        indicator=True
    )

    # Normalize capitalization and whitespace.
    merged["raw_type_normalized"] = normalize_extras(
        merged["extras_type_raw"]
    )

    merged["clean_type_normalized"] = normalize_extras(
        merged["extras_type_clean"]
    )

    merged["extras_type_matches"] = (
        merged["raw_type_normalized"]
        == merged["clean_type_normalized"]
    )

    print("=" * 65)
    print("CORRECTED EXTRAS TYPE MISMATCH AUDIT")
    print("=" * 65)

    print("\nRecords investigated:", len(merged))

    print(
        "Original records matched:",
        int((merged["_merge"] == "both").sum())
    )

    print(
        "Extras types matching after normalization:",
        int(merged["extras_type_matches"].sum())
    )

    print(
        "Actual extras type mismatches:",
        int((~merged["extras_type_matches"]).sum())
    )

    print("\nTransformation summary:")

    print(
        merged.groupby(
            [
                "extras_type_raw",
                "extras_type_clean"
            ],
            dropna=False
        ).size().to_string()
    )

    print("\nDetailed comparison:")

    columns = [
        "match_id",
        "innings",
        "overid",
        "over_ball_num",
        "extras_type_raw",
        "extras_type_clean",
        "raw_type_normalized",
        "clean_type_normalized",
        "extras_2_runs_raw",
        "extras_2_runs_clean",
        "extras_type_matches",
    ]

    print(
        merged[columns].to_string(index=False)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    merged.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
