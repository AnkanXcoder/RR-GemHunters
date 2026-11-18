
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

REGULATION_PATH = (
    ROOT / "data" / "processed" / "deliveries"
    / "regulation_deliveries.csv"
)

PRESSURE_PATH = (
    ROOT / "data" / "processed" / "features"
    / "pressure_state_deliveries.csv"
)

OUTPUT_PATH = (
    ROOT / "data" / "processed" / "features"
    / "bowler_pressure_vs_baseline.csv"
)

KEYS = ["match_id", "innings", "innings_ball_num"]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("\nLoading pressure and regulation deliveries...")

    regulation = pd.read_csv(REGULATION_PATH, low_memory=False)
    pressure = pd.read_csv(PRESSURE_PATH, low_memory=False)

    print("Regulation shape:", regulation.shape)
    print("Pressure-state shape:", pressure.shape)

    return regulation, pressure


# ============================================================
# VALIDATE AND MERGE
# ============================================================

def prepare_data(regulation, pressure):

    required_regulation = KEYS + [
        "bowler_id",
        "bowler_name",
        "bowler_runs_provisional",
        "bowler_wicket_provisional",
        "legal_ball_provisional",
        "is_four",
        "is_six",
        "has_quality_flag",
        "flag_unresolved_extras",
    ]

    required_pressure = KEYS + [
        "bowler_id",
        "bowler_name",
        "required_run_rate",
        "balls_remaining",
        "wickets_remaining",
        "team_runs_this_delivery",
    ]

    for col in required_regulation:
        if col not in regulation.columns:
            raise ValueError(f"Missing regulation column: {col}")

    for col in required_pressure:
        if col not in pressure.columns:
            raise ValueError(f"Missing pressure column: {col}")

    if regulation.duplicated(KEYS).any():
        raise ValueError("Duplicate delivery keys in regulation data")

    if pressure.duplicated(KEYS).any():
        raise ValueError("Duplicate delivery keys in pressure data")

    pressure = pressure.copy()

    pressure["is_pressure_delivery"] = (
        pressure["required_run_rate"].ge(10)
        & pressure["balls_remaining"].le(30)
        & pressure["wickets_remaining"].le(5)
    )

    merged = pressure.merge(
        regulation[required_regulation],
        on=KEYS,
        how="left",
        suffixes=("_state", "_delivery"),
        indicator=True,
        validate="one_to_one",
    )

    print("\n========== JOIN VALIDATION ==========")
    print(merged["_merge"].value_counts().to_string())

    if not merged["_merge"].eq("both").all():
        raise ValueError("Delivery join validation failed")

    # Bowler identity must agree between both datasets.
    if not merged["bowler_id_state"].eq(
        merged["bowler_id_delivery"]
    ).all():
        raise ValueError("Bowler ID mismatch after merge")

    merged["bowler_id"] = merged["bowler_id_delivery"]
    merged["bowler_name"] = merged["bowler_name_delivery"]
    # Use the regulation dataset as the source of bowling metrics.
    merged["legal_ball_provisional"] = (
    merged["legal_ball_provisional_delivery"]
    )

    merged["has_quality_flag"] = (
    merged["has_quality_flag_delivery"]
    )

    merged.drop(
        columns=[
            "_merge",
            "bowler_id_state",
            "bowler_id_delivery",
            "bowler_name_state",
            "bowler_name_delivery",
        ],
        inplace=True,
    )

    return merged


# ============================================================
# NORMALIZE BOOLEAN FIELDS
# ============================================================

def normalize_boolean(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
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


# ============================================================
# CALCULATE BOWLING METRICS
# ============================================================

def calculate_metrics(data, label):

    grouped = data.groupby(
        ["bowler_id", "bowler_name"],
        dropna=False,
    )

    result = grouped.agg(
        deliveries=("bowler_runs_provisional", "size"),
        runs_conceded=("bowler_runs_provisional", "sum"),
        wickets=("bowler_wicket_provisional", "sum"),
        legal_balls=("legal_ball_provisional", "sum"),
        fours_conceded=("is_four", "sum"),
        sixes_conceded=("is_six", "sum"),
        quality_flagged_deliveries=("has_quality_flag", "sum"),
        unresolved_extras_deliveries=(
            "flag_unresolved_extras", "sum"
        ),
    ).reset_index()

    # Dot balls: legal deliveries where total team runs = 0.
    dot_data = data.copy()
    dot_data["is_dot_ball"] = (
        dot_data["legal_ball_provisional"]
        & dot_data["team_runs_this_delivery"].eq(0)
    )

    dots = (
        dot_data.groupby("bowler_id")["is_dot_ball"]
        .sum()
        .rename("dot_balls")
    )

    matches = (
        data.groupby("bowler_id")["match_id"]
        .nunique()
        .rename("matches")
    )

    result = result.merge(
        dots, on="bowler_id", how="left"
    ).merge(
        matches, on="bowler_id", how="left"
    )

    result["economy_rate"] = np.where(
        result["legal_balls"] > 0,
        result["runs_conceded"] * 6
        / result["legal_balls"],
        np.nan,
    )

    result["bowling_strike_rate"] = np.where(
        result["wickets"] > 0,
        result["legal_balls"] / result["wickets"],
        np.nan,
    )

    result["wickets_per_match"] = np.where(
        result["matches"] > 0,
        result["wickets"] / result["matches"],
        np.nan,
    )

    result["dot_ball_percentage"] = np.where(
        result["legal_balls"] > 0,
        result["dot_balls"] * 100
        / result["legal_balls"],
        np.nan,
    )

    result["boundary_conceded_percentage"] = np.where(
        result["deliveries"] > 0,
        (
            result["fours_conceded"]
            + result["sixes_conceded"]
        ) * 100 / result["deliveries"],
        np.nan,
    )

    result["quality_flag_percentage"] = (
        result["quality_flagged_deliveries"]
        * 100 / result["deliveries"]
    )

    result["unresolved_extras_percentage"] = (
        result["unresolved_extras_deliveries"]
        * 100 / result["deliveries"]
    )

    rename = {
        col: f"{label}_{col}"
        for col in result.columns
        if col not in ["bowler_id", "bowler_name"]
    }

    return result.rename(columns=rename)


# ============================================================
# BUILD PRESSURE VS NORMAL COMPARISON
# ============================================================

def build_comparison(merged):

    pressure = merged[
        merged["is_pressure_delivery"]
    ].copy()

    normal = merged[
        ~merged["is_pressure_delivery"]
    ].copy()

    print("\nPressure deliveries:", len(pressure))
    print("Normal deliveries:", len(normal))

    pressure_metrics = calculate_metrics(
        pressure, "pressure"
    )

    normal_metrics = calculate_metrics(
        normal, "normal"
    )

    comparison = pressure_metrics.merge(
        normal_metrics,
        on=["bowler_id", "bowler_name"],
        how="outer",
        validate="one_to_one",
    )

    count_columns = [
        "pressure_deliveries",
        "pressure_runs_conceded",
        "pressure_wickets",
        "pressure_legal_balls",
        "pressure_fours_conceded",
        "pressure_sixes_conceded",
        "pressure_quality_flagged_deliveries",
        "pressure_unresolved_extras_deliveries",
        "pressure_dot_balls",
        "pressure_matches",
        "normal_deliveries",
        "normal_runs_conceded",
        "normal_wickets",
        "normal_legal_balls",
        "normal_fours_conceded",
        "normal_sixes_conceded",
        "normal_quality_flagged_deliveries",
        "normal_unresolved_extras_deliveries",
        "normal_dot_balls",
        "normal_matches",
    ]

    for col in count_columns:
        if col in comparison.columns:
            comparison[col] = comparison[col].fillna(0)

    comparison["has_pressure_sample"] = (
        comparison["pressure_deliveries"] > 0
    )

    comparison["has_normal_sample"] = (
        comparison["normal_deliveries"] > 0
    )

    comparison["comparison_valid"] = (
        comparison["has_pressure_sample"]
        & comparison["has_normal_sample"]
        & (comparison["pressure_legal_balls"] > 0)
        & (comparison["normal_legal_balls"] > 0)
    )

    # Difference metrics. Negative economy change is favourable.
    comparison["economy_rate_change"] = (
        comparison["pressure_economy_rate"]
        - comparison["normal_economy_rate"]
    )

    comparison["wickets_per_match_change"] = (
        comparison["pressure_wickets_per_match"]
        - comparison["normal_wickets_per_match"]
    )

    comparison["dot_ball_percentage_change"] = (
        comparison["pressure_dot_ball_percentage"]
        - comparison["normal_dot_ball_percentage"]
    )

    comparison["bowling_strike_rate_change"] = (
        comparison["pressure_bowling_strike_rate"]
        - comparison["normal_bowling_strike_rate"]
    )

    comparison["boundary_conceded_percentage_change"] = (
        comparison["pressure_boundary_conceded_percentage"]
        - comparison["normal_boundary_conceded_percentage"]
    )

    # Flag comparison records if either side contains quality issues.
    comparison["comparison_quality_review"] = (
        (
            comparison["pressure_quality_flagged_deliveries"] > 0
        )
        | (
            comparison["normal_quality_flagged_deliveries"] > 0
        )
        | (
            comparison["pressure_unresolved_extras_deliveries"] > 0
        )
        | (
            comparison["normal_unresolved_extras_deliveries"] > 0
        )
    )

    return comparison


# ============================================================
# MAIN
# ============================================================

def main():

    regulation, pressure = load_data()

    merged = prepare_data(regulation, pressure)

    print("\n========== PRESSURE RECONCILIATION ==========")

    pressure_count = merged["is_pressure_delivery"].sum()
    normal_count = (~merged["is_pressure_delivery"]).sum()

    print("Pressure rows:", pressure_count)
    print("Normal rows:", normal_count)
    print("Combined rows:", len(merged))
    print(
        "Reconciliation:",
        pressure_count + normal_count == len(merged),
    )

    comparison = build_comparison(merged)

    print("\n========== BOWLER COMPARISON VALIDATION ==========")

    print("Unique bowlers:", comparison["bowler_id"].nunique())
    print("Duplicate bowler IDs:", comparison["bowler_id"].duplicated().sum())
    print(
        "Valid comparisons:",
        comparison["comparison_valid"].sum(),
    )
    print(
        "Quality review cases:",
        comparison["comparison_quality_review"].sum(),
    )

    if comparison["bowler_id"].duplicated().any():
        raise ValueError("Duplicate bowler IDs in output")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison = comparison.sort_values(
        ["pressure_wickets_per_match", "pressure_dot_ball_percentage"],
        ascending=[False, False],
        na_position="last",
    )

    comparison.to_csv(OUTPUT_PATH, index=False)

    preview_columns = [
        "bowler_name",
        "pressure_deliveries",
        "normal_deliveries",
        "pressure_economy_rate",
        "normal_economy_rate",
        "economy_rate_change",
        "pressure_wickets_per_match",
        "pressure_dot_ball_percentage",
        "comparison_valid",
        "comparison_quality_review",
    ]

    print("\n========== BOWLER PRESSURE VS BASELINE ==========")
    print(
        comparison[preview_columns]
        .head(20)
        .to_string(index=False)
    )

    print("\nSaved:", OUTPUT_PATH)
    print("\nBOWLER PRESSURE VS BASELINE ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
