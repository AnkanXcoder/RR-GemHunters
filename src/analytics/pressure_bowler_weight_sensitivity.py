
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data/processed/features/"
    "pressure_bowler_reliability_scores.csv"
)

OUTPUT_DIR = ROOT / "data/interim"

DETAIL_FILE = (
    OUTPUT_DIR / "pressure_bowler_weight_sensitivity_detail.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "pressure_bowler_weight_sensitivity_summary.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nLoading reliability scores...")
df = pd.read_csv(INPUT_FILE)

# Keep only primary-eligible bowlers with complete component scores.
score_columns = [
    "wicket_score",
    "economy_score",
    "dot_ball_score",
    "consistency_score",
    "evidence_score",
]

required = [
    "bowler_id",
    "bowler_name",
    "primary_eligible",
    *score_columns,
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df[df["primary_eligible"] == True].copy()

if df["bowler_id"].duplicated().any():
    raise ValueError("Duplicate bowler IDs detected.")

if df[score_columns].isna().any().any():
    raise ValueError("Missing component scores detected.")

print("Primary eligible bowlers:", len(df))


# =====================================================
# ALTERNATIVE WEIGHT SCENARIOS
# =====================================================

scenarios = {
    "BASELINE": {
        "wicket_score": 0.30,
        "economy_score": 0.25,
        "dot_ball_score": 0.20,
        "consistency_score": 0.15,
        "evidence_score": 0.10,
    },

    "WICKET_FOCUSED": {
        "wicket_score": 0.35,
        "economy_score": 0.20,
        "dot_ball_score": 0.20,
        "consistency_score": 0.15,
        "evidence_score": 0.10,
    },

    "ECONOMY_FOCUSED": {
        "wicket_score": 0.25,
        "economy_score": 0.30,
        "dot_ball_score": 0.20,
        "consistency_score": 0.15,
        "evidence_score": 0.10,
    },

    "CONSISTENCY_FOCUSED": {
        "wicket_score": 0.25,
        "economy_score": 0.20,
        "dot_ball_score": 0.20,
        "consistency_score": 0.25,
        "evidence_score": 0.10,
    },

    "EVIDENCE_FOCUSED": {
        "wicket_score": 0.25,
        "economy_score": 0.20,
        "dot_ball_score": 0.20,
        "consistency_score": 0.15,
        "evidence_score": 0.20,
    },

    "BALANCED": {
        "wicket_score": 0.25,
        "economy_score": 0.25,
        "dot_ball_score": 0.20,
        "consistency_score": 0.15,
        "evidence_score": 0.15,
    },
}

for name, weights in scenarios.items():
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError(f"Weights do not sum to 1: {name}")


# =====================================================
# CALCULATE SCENARIO SCORES AND RANKS
# =====================================================

results = df[
    ["bowler_id", "bowler_name"]
].copy()

for scenario_name, weights in scenarios.items():

    score = sum(
        df[column] * weight
        for column, weight in weights.items()
    )

    results[f"{scenario_name}_score"] = score

    results[f"{scenario_name}_rank"] = (
        score.rank(method="min", ascending=False)
        .astype(int)
    )


# =====================================================
# TOP 10 OVERLAP AND RANK STABILITY
# =====================================================

baseline_top10 = set(
    results.nsmallest(10, "BASELINE_rank")["bowler_id"]
)

summary_rows = []

baseline_rank = results["BASELINE_rank"]

for scenario_name in scenarios:

    scenario_rank = results[f"{scenario_name}_rank"]

    scenario_top10 = set(
        results.nsmallest(
            10,
            f"{scenario_name}_rank"
        )["bowler_id"]
    )

    overlap = len(baseline_top10 & scenario_top10)

    rank_correlation = baseline_rank.corr(
        scenario_rank,
        method="spearman"
    )

    mean_rank_change = (
        scenario_rank - baseline_rank
    ).abs().mean()

    summary_rows.append({
        "scenario": scenario_name,
        "top10_overlap_with_baseline": overlap,
        "top10_overlap_percentage": overlap * 10,
        "spearman_rank_correlation": rank_correlation,
        "mean_absolute_rank_change": mean_rank_change,
    })

summary = pd.DataFrame(summary_rows)

# Scenario-specific results for every eligible bowler.
results.to_csv(DETAIL_FILE, index=False)

summary.to_csv(SUMMARY_FILE, index=False)


# =====================================================
# OUTPUT
# =====================================================

print("\n========== WEIGHT SENSITIVITY SUMMARY ==========")

print(
    summary.round(3).to_string(index=False)
)

print("\n========== TOP 10 OVERLAP ==========")

for scenario_name in scenarios:

    top10 = results.nsmallest(
        10,
        f"{scenario_name}_rank"
    )

    print(f"\n{scenario_name}")

    print(
        top10[
            [
                "bowler_name",
                f"{scenario_name}_score",
                f"{scenario_name}_rank",
            ]
        ].round(2).to_string(index=False)
    )


# =====================================================
# VALIDATION
# =====================================================

print("\n========== VALIDATION ==========")

print("Scenario count:", len(scenarios))
print("Bowler count:", len(results))
print("Duplicate bowler IDs:", results["bowler_id"].duplicated().sum())

print(
    "All scenario scores present:",
    results[
        [f"{name}_score" for name in scenarios]
    ].notna().all().all()
)

print(
    "All ranks present:",
    results[
        [f"{name}_rank" for name in scenarios]
    ].notna().all().all()
)

assert len(results) == len(df)
assert results["bowler_id"].duplicated().sum() == 0

assert results[
    [f"{name}_score" for name in scenarios]
].notna().all().all()

assert results[
    [f"{name}_rank" for name in scenarios]
].notna().all().all()

print("\nWEIGHT SENSITIVITY ANALYSIS COMPLETE")
print("Detail:", DETAIL_FILE)
print("Summary:", SUMMARY_FILE)
