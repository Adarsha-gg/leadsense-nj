from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from leadsense_nj.baseline import fit_tabular_logistic
from leadsense_nj.explainability import top_feature_drivers
from leadsense_nj.metrics import ModelVsHistoricalMetrics, compute_model_vs_historical_metrics
from leadsense_nj.optimization import (
    OptimizationSummary,
    optimize_replacement_plan,
    optimize_replacement_plan_ilp,
)
from leadsense_nj.policy_brief import generate_policy_brief
from leadsense_nj.preprocessing import impute_missing_values
from leadsense_nj.target import with_elevated_risk_label
from leadsense_nj.uncertainty import train_bootstrap_ensemble


@dataclass(frozen=True)
class DemoSnapshot:
    scored_df: pd.DataFrame
    selected_df: pd.DataFrame
    optimization_summary: OptimizationSummary
    policy_briefs: dict[str, str]
    comparison_metrics: ModelVsHistoricalMetrics


def build_demo_snapshot(
    df: pd.DataFrame,
    *,
    budget: float = 35000.0,
    fairness_tolerance: float = 0.05,
    min_county_coverage: int = 0,
    optimizer_method: str = "ilp",
    baseline_epochs: int = 700,
    baseline_learning_rate: float = 0.1,
    ensemble_models: int = 12,
    ensemble_epochs: int = 350,
    ensemble_learning_rate: float = 0.1,
) -> DemoSnapshot:
    cleaned = impute_missing_values(df)
    labeled = with_elevated_risk_label(cleaned)
    model, _ = fit_tabular_logistic(labeled, epochs=baseline_epochs, learning_rate=baseline_learning_rate)
    ensemble = train_bootstrap_ensemble(
        labeled,
        n_models=ensemble_models,
        epochs=ensemble_epochs,
        learning_rate=ensemble_learning_rate,
        seed=19,
    )
    mean, std = ensemble.predict_mean_std(labeled)

    scored = labeled.copy()
    scored["risk_score"] = mean
    scored["risk_uncertainty"] = std
    scored["replacement_cost"] = 10000 + (scored["pct_housing_pre_1950"] * 8000) + (scored["lead_90p_ppb"] * 200)
    if "minority_share" in scored.columns:
        scored["minority_share"] = pd.to_numeric(scored["minority_share"], errors="coerce").fillna(
            scored["poverty_rate"]
        )
    else:
        scored["minority_share"] = scored["poverty_rate"]
    scored["minority_share"] = scored["minority_share"].clip(0.0, 1.0)

    if optimizer_method == "ilp":
        selected, summary = optimize_replacement_plan_ilp(
            scored,
            budget=budget,
            fairness_tolerance=fairness_tolerance,
            min_county_coverage=min_county_coverage,
        )
    else:
        selected, summary = optimize_replacement_plan(
            scored,
            budget=budget,
            fairness_tolerance=fairness_tolerance,
            min_county_coverage=min_county_coverage,
        )

    policy_briefs: dict[str, str] = {}
    if not selected.empty:
        selected_with_rank = selected.sort_values("priority_rank")
        for _, row in selected_with_rank.iterrows():
            geoid = str(row["geoid"])
            source_row = scored.loc[scored["geoid"] == geoid].iloc[0]
            drivers = top_feature_drivers(model, source_row, top_k=3)
            brief = generate_policy_brief(
                geoid=geoid,
                county=str(row["county"]),
                municipality=str(row["municipality"]),
                risk_score=float(row["risk_score"]),
                uncertainty_std=float(row["risk_uncertainty"]),
                top_drivers=drivers,
                replacement_rank=int(row["priority_rank"]),
                replacement_cost=float(row["replacement_cost"]),
            )
            policy_briefs[geoid] = brief

    comparison_metrics = compute_model_vs_historical_metrics(scored, model_threshold=0.5, ece_bins=5)

    return DemoSnapshot(
        scored_df=scored.sort_values("risk_score", ascending=False).reset_index(drop=True),
        selected_df=selected.reset_index(drop=True),
        optimization_summary=summary,
        policy_briefs=policy_briefs,
        comparison_metrics=comparison_metrics,
    )
