
from pathlib import Path
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed" / "deliveries"
    / "regulation_deliveries.csv"
)

OUTPUT_DIR = ROOT / "data" / "interim"

OUTPUT_FILE = OUTPUT_DIR / "regulation_innings_audit.csv"
ISSUES_FILE = OUTPUT_DIR / "regulation_innings_issues.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    print("=" * 65)
    print("REGULATION INNINGS AUDIT")
    print("=" * 65)

    print(f"\nTotal delivery records: {len(df):,}")

    return df


# ============================================================
# 3. VALIDATE REQUIRED COLUMNS
# ============================================================

def validate_columns(df):

    required_columns = [
        "match_id",
        "innings",
        "runs",
        "extras1_runs",
        "extras_2_runs",
        "is_wicket",
        "legal_ball_provisional",
        "bowler_wicket_provisional",
        "flag_unresolved_extras",
        "has_quality_flag",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("Required column validation: PASS")


# ============================================================
# 4. DATA TYPE NORMALIZATION
# ============================================================

def normalize_data(df):

    data = df.copy()

    numeric_columns = [
        "innings",
        "runs",
        "extras1_runs",
        "extras_2_runs",
        "is_wicket",
    ]

    for col in numeric_columns:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    boolean_columns = [
        "legal_ball_provisional",
        "bowler_wicket_provisional",
        "flag_unresolved_extras",
        "has_quality_flag",
    ]

    for col in boolean_columns:

        data[col] = (
            data[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            })
            .fillna(False)
        )

    return data


# ============================================================
# 5. VALIDATE REGULATION INNINGS
# ============================================================

def validate_innings(df):

    print("\n" + "=" * 65)
    print("INNINGS VALIDATION")
    print("=" * 65)

    print(
        "Innings found:",
        sorted(df["innings"].dropna().unique())
    )

    unexpected = df[
        ~df["innings"].isin([1, 2])
    ]

    print(
        "Unexpected innings records:",
        f"{len(unexpected):,}"
    )

    if len(unexpected) > 0:
        raise ValueError(
            "Unexpected innings found in regulation dataset."
        )

    print("Regulation innings validation: PASS")


# ============================================================
# 6. CALCULATE INNINGS METRICS
# ============================================================

def calculate_innings_metrics(df):

    print("\n" + "=" * 65)
    print("CALCULATING INNINGS METRICS")
    print("=" * 65)

    data = df.copy()

    # Missing run components are treated as zero for
    # aggregation, but remain available for quality review.

    data["runs"] = data["runs"].fillna(0)
    data["extras1_runs"] = data["extras1_runs"].fillna(0)
    data["extras_2_runs"] = data["extras_2_runs"].fillna(0)

    # Total team runs = batter runs + both extras components.

    data["total_delivery_runs"] = (
        data["runs"]
        + data["extras1_runs"]
        + data["extras_2_runs"]
    )

    # Aggregate by match and innings.

    audit = (
        data.groupby(
            ["match_id", "innings"],
            dropna=False
        )
        .agg(
            delivery_records=("match_id", "size"),

            total_runs=("total_delivery_runs", "sum"),

            batter_runs=("runs", "sum"),

            extras1_runs=("extras1_runs", "sum"),

            extras2_runs=("extras_2_runs", "sum"),

            legal_balls=("legal_ball_provisional", "sum"),

            wicket_records=("is_wicket", "sum"),

            bowler_wickets=(
                "bowler_wicket_provisional",
                "sum"
            ),

            unresolved_extra_records=(
                "flag_unresolved_extras",
                "sum"
            ),

            quality_flagged_deliveries=(
                "has_quality_flag",
                "sum"
            ),

        )
        .reset_index()
    )

    # Convert counts to integers.

    count_columns = [
        "delivery_records",
        "total_runs",
        "batter_runs",
        "extras1_runs",
        "extras2_runs",
        "legal_balls",
        "wicket_records",
        "bowler_wickets",
        "unresolved_extra_records",
        "quality_flagged_deliveries",
    ]

    for col in count_columns:
        audit[col] = audit[col].fillna(0).astype(int)

    # Calculate overs and run rate.

    audit["overs"] = (
        audit["legal_balls"] // 6
        + (audit["legal_balls"] % 6) / 10
    )

    audit["run_rate"] = (
        audit["total_runs"]
        / (audit["legal_balls"] / 6).replace(0, pd.NA)
    )

    # Identify suspicious innings.

    audit["over_120_legal_balls"] = (
        audit["legal_balls"] > 120
    )

    audit["zero_legal_balls"] = (
        audit["legal_balls"] == 0
    )

    audit["unresolved_extras_present"] = (
        audit["unresolved_extra_records"] > 0
    )

    audit["quality_issue_present"] = (
        audit["quality_flagged_deliveries"] > 0
    )

    return audit


# ============================================================
# 7. IDENTIFY INNINGS WITH ISSUES
# ============================================================

def identify_issues(audit):

    issues = audit[
        (
            audit["over_120_legal_balls"]
        )
        | (
            audit["zero_legal_balls"]
        )
        | (
            audit["unresolved_extras_present"]
        )
        | (
            audit["quality_issue_present"]
        )
    ].copy()

    issues["issue_reason"] = ""

    conditions = [
        (
            audit["over_120_legal_balls"],
            "More than 120 legal balls"
        ),
        (
            audit["zero_legal_balls"],
            "Zero legal balls"
        ),
        (
            audit["unresolved_extras_present"],
            "Unresolved extras"
        ),
        (
            audit["quality_issue_present"],
            "Quality flagged deliveries"
        ),
    ]

    for condition, reason in conditions:

        issues.loc[
            condition.loc[issues.index],
            "issue_reason"
        ] += reason + "; "

    return issues


# ============================================================
# 8. SAVE RESULTS
# ============================================================

def save_results(audit, issues):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    audit.to_csv(
        OUTPUT_FILE,
        index=False
    )

    issues.to_csv(
        ISSUES_FILE,
        index=False
    )

    print("\nFiles saved:")
    print(OUTPUT_FILE)
    print(ISSUES_FILE)


# ============================================================
# 9. PRINT SUMMARY
# ============================================================

def print_summary(audit, issues):

    print("\n" + "=" * 65)
    print("FINAL INNINGS AUDIT SUMMARY")
    print("=" * 65)

    print(
        "Total innings:",
        f"{len(audit):,}"
    )

    print(
        "Total deliveries:",
        f"{audit['delivery_records'].sum():,}"
    )

    print(
        "Total innings runs:",
        f"{audit['total_runs'].sum():,}"
    )

    print(
        "Total batter runs:",
        f"{audit['batter_runs'].sum():,}"
    )

    print(
        "Total legal balls:",
        f"{audit['legal_balls'].sum():,}"
    )

    print(
        "Total wicket records:",
        f"{audit['wicket_records'].sum():,}"
    )

    print(
        "Total bowler wickets:",
        f"{audit['bowler_wickets'].sum():,}"
    )

    print(
        "Innings exceeding 120 legal balls:",
        int(audit["over_120_legal_balls"].sum())
    )

    print(
        "Innings with zero legal balls:",
        int(audit["zero_legal_balls"].sum())
    )

    print(
        "Innings with unresolved extras:",
        int(audit["unresolved_extras_present"].sum())
    )

    print(
        "Innings with quality flags:",
        int(audit["quality_issue_present"].sum())
    )

    print(
        "Total innings requiring review:",
        f"{len(issues):,}"
    )

    print("\nFIRST 10 INNINGS:")

    display_columns = [
        "match_id",
        "innings",
        "total_runs",
        "legal_balls",
        "overs",
        "wicket_records",
        "bowler_wickets",
    ]

    print(
        audit[display_columns]
        .head(10)
        .to_string(index=False)
    )


# ============================================================
# 10. MAIN
# ============================================================

def main():

    df = load_data()

    validate_columns(df)

    df = normalize_data(df)

    validate_innings(df)

    audit = calculate_innings_metrics(df)

    issues = identify_issues(audit)

    save_results(audit, issues)

    print_summary(audit, issues)

    print("\nREGULATION INNINGS AUDIT COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
