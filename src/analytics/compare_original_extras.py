
from pathlib import Path
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

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

OUTPUT_FILE = (
    OUTPUT_DIR / "original_vs_cleaned_extras.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("ORIGINAL VS CLEANED EXTRAS COMPARISON")
    print("=" * 70)

    raw = pd.read_csv(RAW_FILE, low_memory=False)
    clean = pd.read_csv(CLEAN_FILE, low_memory=False)

    print(f"\nOriginal source records: {len(raw):,}")
    print(f"Cleaned regulation records: {len(clean):,}")

    return raw, clean


# ============================================================
# 3. DEFINE MATCHING KEYS
# ============================================================

def prepare_keys(raw, clean):

    keys = [
        "match_id",
        "innings",
        "overid",
        "over_ball_num",
    ]

    for name, df in [
        ("Original", raw),
        ("Cleaned", clean),
    ]:

        missing = [
            col for col in keys
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} dataset missing keys: {missing}"
            )

        if df[keys].isna().any().any():
            raise ValueError(
                f"{name} dataset contains missing matching keys."
            )

        duplicates = df.duplicated(keys).sum()

        print(
            f"{name} duplicate matching keys: {duplicates:,}"
        )

        if duplicates:
            raise ValueError(
                f"Duplicate matching keys in {name} dataset."
            )

    return keys


# ============================================================
# 4. SELECT UNUSUAL CLEANED RECORDS
# ============================================================

def select_unusual_records(clean):

    data = clean.copy()

    data["extras_2_runs"] = pd.to_numeric(
        data["extras_2_runs"],
        errors="coerce"
    ).fillna(0)

    unusual = data[
        data["extras_2_runs"] != 0
    ].copy()

    print(
        "\nCleaned records with nonzero extras_2_runs:",
        f"{len(unusual):,}"
    )

    return unusual


# ============================================================
# 5. COMPARE ORIGINAL AND CLEANED RECORDS
# ============================================================

def compare_records(raw, unusual, keys):

    compare_columns = [
        "runs",
        "extras_type",
        "extras1_runs",
        "extras_2_runs",
        "is_wicket",
        "dismissal_type",
    ]

    raw_columns = keys + compare_columns

    clean_columns = keys + compare_columns + [
        "bowler_runs_provisional",
        "legal_ball_provisional",
        "flag_unresolved_extras",
        "has_quality_flag",
    ]

    raw_subset = raw[raw_columns].copy()

    clean_subset = unusual[clean_columns].copy()

    merged = clean_subset.merge(
        raw_subset,
        on=keys,
        how="left",
        suffixes=("_clean", "_raw"),
        indicator=True,
        validate="one_to_one",
    )

    print("\n" + "=" * 70)
    print("MATCHING RESULTS")
    print("=" * 70)

    print(
        "Matched original records:",
        int((merged["_merge"] == "both").sum())
    )

    print(
        "Records missing from original source:",
        int((merged["_merge"] == "left_only").sum())
    )

    # Compare original and cleaned source values.

    comparison_columns = [
        "runs",
        "extras_type",
        "extras1_runs",
        "extras_2_runs",
        "is_wicket",
        "dismissal_type",
    ]

    for col in comparison_columns:

        clean_col = f"{col}_clean"
        raw_col = f"{col}_raw"

        if col in [
            "runs",
            "extras1_runs",
            "extras_2_runs",
            "is_wicket",
        ]:

            left = pd.to_numeric(
                merged[clean_col],
                errors="coerce"
            )

            right = pd.to_numeric(
                merged[raw_col],
                errors="coerce"
            )

            merged[f"{col}_matches"] = (
                left.eq(right)
                | (left.isna() & right.isna())
            )

        else:

            left = (
                merged[clean_col]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            right = (
                merged[raw_col]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            merged[f"{col}_matches"] = left.eq(right)

    match_columns = [
        f"{col}_matches"
        for col in comparison_columns
    ]

    merged["all_source_fields_match"] = (
        merged[match_columns].all(axis=1)
    )

    print(
        "Records with all source fields matching:",
        int(merged["all_source_fields_match"].sum())
    )

    print(
        "Records with at least one difference:",
        int((~merged["all_source_fields_match"]).sum())
    )

    return merged


# ============================================================
# 6. DISPLAY COMPARISON
# ============================================================

def print_comparison(merged):

    print("\n" + "=" * 70)
    print("DETAILED COMPARISON")
    print("=" * 70)

    display_columns = [
        "match_id",
        "innings",
        "overid",
        "over_ball_num",
        "extras_type_clean",
        "extras1_runs_clean",
        "extras_2_runs_clean",
        "extras1_runs_raw",
        "extras_2_runs_raw",
        "extras_2_runs_matches",
        "all_source_fields_match",
        "_merge",
    ]

    print(
        merged[display_columns]
        .to_string(index=False)
    )

    print("\nSOURCE FIELD MATCH SUMMARY:")

    for col in [
        "runs",
        "extras_type",
        "extras1_runs",
        "extras_2_runs",
        "is_wicket",
        "dismissal_type",
    ]:

        print(
            f"{col}: "
            f"{int(merged[f'{col}_matches'].sum())}"
            f"/{len(merged)} matching"
        )


# ============================================================
# 7. SAVE RESULTS
# ============================================================

def save_results(merged):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    merged.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved comparison:")
    print(OUTPUT_FILE)


# ============================================================
# 8. MAIN
# ============================================================

def main():

    raw, clean = load_data()

    keys = prepare_keys(raw, clean)

    unusual = select_unusual_records(clean)

    merged = compare_records(
        raw,
        unusual,
        keys
    )

    print_comparison(merged)

    save_results(merged)

    print(
        "\nORIGINAL VS CLEANED COMPARISON COMPLETED"
    )


if __name__ == "__main__":
    main()
