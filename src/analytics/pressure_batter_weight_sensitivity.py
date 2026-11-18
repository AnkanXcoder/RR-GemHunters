
from pathlib import Path
import pandas as pd

# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = ROOT / "data" / "processed" / "features"
INTERIM_DIR = ROOT / "data" / "interim"

INPUT_FILE = (
    FEATURE_DIR / "pressure_batter_reliability_scores.csv"
)

DETAIL_FILE = (
    INTERIM_DIR / "pressure_batter_weight_sensitivity_detail.csv"
)

SUMMARY_FILE = (
    INTERIM_DIR / "pressure_batter_weight_sensitivity_summary.csv"
)

INTERIM_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading batting reliability scores...")

df = pd.read_csv(INPUT_FILE)

print("Input shape:", df.shape)

df = df[
    df["primary_eligible"].eq(True)
].copy()

print("Primary-eligible batters:", len(df))


# ============================================================
# 3. REQUIRED COMPONENTS
# ============================================================

components = [
    "scoring_ability_score",
    "consistency_score",
    "boundary_score",
    "dismissal_resistance_score",
    "evidence_score",
]

required_columns = [
    "striker_id",
    "striker_name",
    "batting_reliability_score",
    *components,
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

if df["striker_id"].duplicated().any():
    raise ValueError("Duplicate striker IDs detected.")

if df[components].isna().any().any():
    raise ValueError("Missing component scores detected.")


# ============================================================
# 4. DEFINE WEIGHT SCENARIOS
# ============================================================

scenarios = {

    "BASELINE": {
        "scoring_ability_score": 0.30,
        "consistency_score": 0.25,
        "boundary_score": 0.15,
        "dismissal_resistance_score": 0.15,
        "evidence_score": 0.15,
    },

    "SCORING_FOCUSED": {
        "scoring_ability_score": 0.40,
        "consistency_score": 0.20,
        "boundary_score": 0.15,
        "dismissal_resistance_score": 0.10,
        "evidence_score": 0.15,
    },

    "CONSISTENCY_FOCUSED": {
        "scoring_ability_score": 0.25,
        "consistency_score": 0.35,
        "boundary_score": 0.10,
        "dismissal_resistance_score": 0.15,
        "evidence_score": 0.15,
    },

    "BOUNDARY_FOCUSED": {
        "scoring_ability_score": 0.25,
        "consistency_score": 0.20,
        "boundary_score": 0.25,
        "dismissal_resistance_score": 0.15,
        "evidence_score": 0.15,
    },

    "DISMISSAL_FOCUSED": {
        "scoring_ability_score": 0.25,
        "consistency_score": 0.20,
        "boundary_score": 0.15,
        "dismissal_resistance_score": 0.25,
        "evidence_score": 0.15,
    },

    "EVIDENCE_FOCUSED": {
        "scoring_ability_score": 0.25,
        "consistency_score": 0.20,
        "boundary_score": 0.15,
        "dismissal_resistance_score": 0.10,
        "evidence_score": 0.30,
    },

    "BALANCED": {
        "scoring_ability_score": 0.25,
        "consistency_score": 0.25,
        "boundary_score": 0.15,
        "dismissal_resistance_score": 0.15,
        "evidence_score": 0.20,
    },
}

for name, weights in scenarios.items():

    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError(
            f"Invalid weights for scenario: {name}"
        )


# ============================================================
# 5. CALCULATE SCENARIO SCORES AND RANKS
# ============================================================

results = df[
    ["striker_id", "striker_name"]
].copy()

for scenario_name, weights in scenarios.items():

    scenario_score = sum(
        df[component] * weight
        for component, weight in weights.items()
    )

    results[f"{scenario_name}_score"] = scenario_score

    results[f"{scenario_name}_rank"] = (
        scenario_score
        .rank(method="min", ascending=False)
        .astype(int)
    )


# ============================================================
# 6. TOP 10 OVERLAP
# ============================================================

baseline_top10 = set(
    results.nsmallest(
        10, "BASELINE_rank"
    )["striker_id"]
)

summary_rows = []

baseline_rank = results["BASELINE_rank"]

for scenario_name in scenarios:

    scenario_rank = results[
        f"{scenario_name}_rank"
    ]

    scenario_top10 = set(
        results.nsmallest(
            10,
            f"{scenario_name}_rank"
        )["striker_id"]
    )

    overlap = len(
        baseline_top10 & scenario_top10
    )

    spearman = baseline_rank.corr(
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
        "spearman_rank_correlation": spearman,
        "mean_absolute_rank_change": mean_rank_change,
    })

summary = pd.DataFrame(summary_rows)


# ============================================================
# 7. TOP 10 IN EACH SCENARIO
# ============================================================

print("\n========== TOP 10 OVERLAP SUMMARY ==========")

print(
    summary.round(3).to_string(index=False)
)

for scenario_name in scenarios:

    print(f"\n========== {scenario_name} ==========")

    top10 = results.nsmallest(
        10,
        f"{scenario_name}_rank"
    )

    print(
        top10[
            [
                "striker_name",
                f"{scenario_name}_score",
                f"{scenario_name}_rank",
            ]
        ].round(2).to_string(index=False)
    )


# ============================================================
# 8. VALIDATION
# ============================================================

print("\n========== VALIDATION ==========")

print("Scenario count:", len(scenarios))
print("Eligible batters:", len(results))

print(
    "Duplicate striker IDs:",
    results["striker_id"].duplicated().sum()
)

score_columns = [
    f"{name}_score" for name in scenarios
]

rank_columns = [
    f"{name}_rank" for name in scenarios
]

print(
    "All scenario scores present:",
    results[score_columns].notna().all().all()
)

print(
    "All scenario ranks present:",
    results[rank_columns].notna().all().all()
)

assert results["striker_id"].duplicated().sum() == 0
assert results[score_columns].notna().all().all()
assert results[rank_columns].notna().all().all()


# ============================================================
# 9. SAVE OUTPUTS
# ============================================================

results.to_csv(
    DETAIL_FILE,
    index=False
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)

print("\nSaved detail:", DETAIL_FILE)
print("Saved summary:", SUMMARY_FILE)

print("\nBATTING WEIGHT SENSITIVITY ANALYSIS COMPLETE")
