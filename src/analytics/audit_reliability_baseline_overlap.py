
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

FEATURES = ROOT / "data" / "processed" / "features"


def audit_batters():
    scores = pd.read_csv(
        FEATURES / "pressure_batter_reliability_scores.csv"
    )

    baseline = pd.read_csv(
        FEATURES / "batter_pressure_vs_baseline.csv"
    )

    merged = scores.merge(
        baseline,
        on="striker_id",
        how="left",
        suffixes=("_score", "_baseline"),
        validate="one_to_one",
    )

    eligible = merged[merged["primary_eligible"]].copy()

    eligible["valid_baseline"] = (
        eligible["comparison_valid"].fillna(False)
    )

    eligible["strong_baseline"] = (
        eligible["valid_baseline"]
        & (eligible["pressure_deliveries"] >= 50)
        & (eligible["normal_deliveries"] >= 30)
    )

    eligible["moderate_baseline"] = (
        eligible["valid_baseline"]
        & (eligible["pressure_deliveries"] >= 20)
        & (eligible["normal_deliveries"] >= 20)
    )

    print("\n========== BATTER OVERLAP ==========")
    print("Primary eligible:", len(eligible))
    print("Valid baseline:", eligible["valid_baseline"].sum())
    print("Strong baseline:", eligible["strong_baseline"].sum())
    print("Moderate baseline:", eligible["moderate_baseline"].sum())

    print("\nBaseline evidence distribution:")
    print(
        eligible["pressure_deliveries"].describe().to_string()
    )

    print("\nEligible batters without valid baseline:")
    print(
        eligible.loc[
            ~eligible["valid_baseline"],
            ["striker_name_score", "batting_reliability_score"]
        ].to_string(index=False)
    )

    return eligible


def audit_bowlers():
    scores = pd.read_csv(
        FEATURES / "pressure_bowler_reliability_scores.csv"
    )

    baseline = pd.read_csv(
        FEATURES / "bowler_pressure_vs_baseline.csv"
    )

    merged = scores.merge(
        baseline,
        on="bowler_id",
        how="left",
        suffixes=("_score", "_baseline"),
        validate="one_to_one",
    )

    eligible = merged[merged["primary_eligible"]].copy()

    eligible["valid_baseline"] = (
        eligible["comparison_valid"].fillna(False)
    )

    eligible["strong_baseline"] = (
        eligible["valid_baseline"]
        & (eligible["pressure_deliveries"] >= 50)
        & (eligible["pressure_matches_score"] >= 8)
        & (eligible["normal_deliveries"] >= 30)
    )

    eligible["moderate_baseline"] = (
        eligible["valid_baseline"]
        & (eligible["pressure_deliveries"] >= 20)
        & (eligible["pressure_matches_score"] >= 5)
        & (eligible["normal_deliveries"] >= 20)
    )

    print("\n========== BOWLER OVERLAP ==========")
    print("Primary eligible:", len(eligible))
    print("Valid baseline:", eligible["valid_baseline"].sum())
    print("Strong baseline:", eligible["strong_baseline"].sum())
    print("Moderate baseline:", eligible["moderate_baseline"].sum())

    print("\nBaseline evidence distribution:")
    print(
        eligible["pressure_deliveries"].describe().to_string()
    )

    print("\nEligible bowlers without valid baseline:")
    print(
        eligible.loc[
            ~eligible["valid_baseline"],
            ["bowler_name_score", "reliability_score"]
        ].to_string(index=False)
    )

    return eligible


def main():
    batter = audit_batters()
    bowler = audit_bowlers()

    batter_output = (
        FEATURES / "batter_reliability_baseline_audit.csv"
    )

    bowler_output = (
        FEATURES / "bowler_reliability_baseline_audit.csv"
    )

    batter.to_csv(batter_output, index=False)
    bowler.to_csv(bowler_output, index=False)

    print("\n========== AUDIT FILES SAVED ==========")
    print(batter_output)
    print(bowler_output)


if __name__ == "__main__":
    main()
