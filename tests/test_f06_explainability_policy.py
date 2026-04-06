from __future__ import annotations

import pandas as pd

from leadsense_nj.baseline import fit_tabular_logistic
from leadsense_nj.explainability import compute_linear_contributions, top_feature_drivers
from leadsense_nj.policy_brief import generate_policy_brief
from leadsense_nj.target import with_elevated_risk_label
from leadsense_nj.uncertainty import train_bootstrap_ensemble


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "geoid": ["340010001001", "340030010002", "340050023001", "340070045003", "340090112001", "340110008002"],
            "county": ["Atlantic", "Bergen", "Burlington", "Camden", "Cape May", "Cumberland"],
            "municipality": ["Atlantic City", "Hackensack", "Mount Laurel", "Camden", "Middle Township", "Bridgeton"],
            "median_income": [42000, 81500, 97000, 38500, 71000, 45000],
            "poverty_rate": [0.27, 0.11, 0.07, 0.31, 0.14, 0.24],
            "pct_housing_pre_1950": [0.48, 0.23, 0.17, 0.53, 0.29, 0.41],
            "lead_90p_ppb": [11.2, 5.1, 3.4, 13.8, 6.0, 9.7],
            "ph_mean": [6.7, 7.3, 7.6, 6.5, 7.1, 6.8],
            "alkalinity_mg_l": [28, 56, 62, 24, 48, 27],
            "distance_to_tri_km": [2.4, 5.2, 9.4, 1.6, 7.8, 3.9],
            "winter_freeze_thaw_days": [42, 37, 33, 44, 36, 41],
            "pws_action_level_exceedance_5y": [0, 0, 0, 1, 0, 0],
            "pws_any_sample_gt15_3y": [1, 0, 0, 1, 0, 0],
            "median_housing_year": [1946, 1968, 1985, 1938, 1958, 1942],
        }
    )


def test_compute_linear_contributions_has_expected_shape() -> None:
    df = with_elevated_risk_label(_sample_df())
    model, _ = fit_tabular_logistic(df, epochs=400, learning_rate=0.1)
    contrib = compute_linear_contributions(model, df)
    assert contrib.shape[0] == len(df)
    assert contrib.shape[1] == len(model.feature_columns)


def test_top_feature_drivers_returns_ranked_features() -> None:
    df = with_elevated_risk_label(_sample_df())
    model, _ = fit_tabular_logistic(df, epochs=400, learning_rate=0.1)
    drivers = top_feature_drivers(model, df.iloc[0], top_k=3)
    assert len(drivers) == 3
    assert all(isinstance(item[0], str) for item in drivers)


def test_generate_policy_brief_contains_required_sections() -> None:
    df = with_elevated_risk_label(_sample_df())
    model, _ = fit_tabular_logistic(df, epochs=400, learning_rate=0.1)
    ensemble = train_bootstrap_ensemble(df, n_models=8, epochs=200, learning_rate=0.1)
    mean, std = ensemble.predict_mean_std(df)
    drivers = top_feature_drivers(model, df.iloc[0], top_k=3)
    brief = generate_policy_brief(
        geoid=df.iloc[0]["geoid"],
        county=df.iloc[0]["county"],
        municipality=df.iloc[0]["municipality"],
        risk_score=float(mean[0]),
        uncertainty_std=float(std[0]),
        top_drivers=drivers,
        replacement_rank=1,
        replacement_cost=12500,
    )

    assert "Policy Brief" in brief
    assert "Immediate action" in brief
    assert "Long-term action" in brief
    assert len(brief) > 250
