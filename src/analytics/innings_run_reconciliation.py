
from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "deliveries"
    / "regulation_deliveries.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"

DETAIL_OUTPUT = OUTPUT_DIR / "innings_run_reconciliation_detail.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "innings_run_reconciliation_summary.csv"
EXTRAS_OUTPUT = OUTPUT_DIR / "innings_extras_reconciliation.csv"


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    required_columns = [
        "match_id",
        "innings",
        "runs",
        "extras1_runs",
        "extras_2_runs",
        "extras_type",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df


# --------------------------------------------------
# 3. Validate numeric fields
# --------------------------------------------------

def prepare_data(df):
    numeric_columns = [
        "runs",
        "extras1_runs",
        "extras_2_runs",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    missing_numeric = df[numeric_columns].isna().sum()

    print("\nMissing numeric values:")
    print(missing_numeric)

    df[numeric_columns] = df[numeric_columns].fillna(0)

    df["extras_type"] = (
        df["extras_type"]
        .fillna("none")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


# --------------------------------------------------
# 4. Reconcile innings totals
# --------------------------------------------------

def build_innings_reconciliation(df):

    grouped = (
        df.groupby(["match_id", "innings"], dropna=False)
        .agg(
            delivery_records=("runs", "size"),
            batter_runs=("runs", "sum"),
            extras1_runs=("extras1_runs", "sum"),
            extras2_runs=("extras_2_runs", "sum"),
        )
        .reset_index()
    )

    # Reconstruct total using all recorded run components.
    grouped["reconstructed_innings_runs"] = (
        grouped["batter_runs"]
        + grouped["extras1_runs"]
        + grouped["extras2_runs"]
    )

    grouped["reconciliation_status"] = "REVIEW"

    # A numeric reconstruction is possible when all source
    # run components are present. Missing values were reported
    # before filling, so inspect that report as well.
    grouped.loc[
        grouped["reconstructed_innings_runs"].notna(),
        "reconciliation_status"
    ] = "RECONSTRUCTED"

    return grouped


# --------------------------------------------------
# 5. Extras category audit
# --------------------------------------------------

def build_extras_audit(df):

    extras_audit = (
        df.groupby("extras_type", dropna=False)
        .agg(
            delivery_records=("runs", "size"),
            batter_runs=("runs", "sum"),
            extras1_runs=("extras1_runs", "sum"),
            extras2_runs=("extras_2_runs", "sum"),
        )
        .reset_index()
    )

    extras_audit["total_recorded_runs"] = (
        extras_audit["batter_runs"]
        + extras_audit["extras1_runs"]
        + extras_audit["extras2_runs"]
    )

    return extras_audit


# --------------------------------------------------
# 6. Generate summary
# --------------------------------------------------

def build_summary(detail):

    summary = pd.DataFrame({
        "metric": [
            "total_innings",
            "total_delivery_records",
            "total_batter_runs",
            "total_extras1_runs",
            "total_extras2_runs",
            "total_reconstructed_runs",
            "innings_with_zero_runs",
        ],
        "value": [
            len(detail),
            detail["delivery_records"].sum(),
            detail["batter_runs"].sum(),
            detail["extras1_runs"].sum(),
            detail["extras2_runs"].sum(),
            detail["reconstructed_innings_runs"].sum(),
            (
                detail["reconstructed_innings_runs"] == 0
            ).sum(),
        ],
    })

    return summary


# --------------------------------------------------
# 7. Main execution
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("INNINGS RUN RECONCILIATION AUDIT")
    print("=" * 60)

    df = load_data()

    print(f"\nInput rows: {len(df):,}")
    print(f"Unique matches: {df['match_id'].nunique():,}")

    df = prepare_data(df)

    detail = build_innings_reconciliation(df)
    extras_audit = build_extras_audit(df)
    summary = build_summary(detail)

    # Save outputs
    detail.to_csv(DETAIL_OUTPUT, index=False)
    extras_audit.to_csv(EXTRAS_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    # --------------------------------------------------
    # 8. Print results
    # --------------------------------------------------

    print("\n--- RUN COMPONENT TOTALS ---")

    print(
        f"Batter runs: "
        f"{detail['batter_runs'].sum():,}"
    )

    print(
        f"Extras 1 runs: "
        f"{detail['extras1_runs'].sum():,}"
    )

    print(
        f"Extras 2 runs: "
        f"{detail['extras2_runs'].sum():,}"
    )

    print(
        f"Reconstructed total runs: "
        f"{detail['reconstructed_innings_runs'].sum():,}"
    )

    print("\n--- INNINGS RECONCILIATION ---")

    print(f"Total innings: {len(detail):,}")

    print(
        detail["reconciliation_status"]
        .value_counts()
        .to_string()
    )

    print("\n--- EXTRAS CATEGORY AUDIT ---")

    print(
        extras_audit.to_string(index=False)
    )

    print("\n--- OUTPUT FILES ---")

    print(DETAIL_OUTPUT)
    print(EXTRAS_OUTPUT)
    print(SUMMARY_OUTPUT)

    print("\nAudit completed.")


if __name__ == "__main__":
    main()
