from __future__ import annotations

import pandas as pd

from leadsense_nj.demo import build_demo_snapshot


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


def test_build_demo_snapshot_returns_expected_components() -> None:
    snapshot = build_demo_snapshot(_sample_df(), budget=35000)
    assert len(snapshot.scored_df) == 6
    assert "risk_score" in snapshot.scored_df.columns
    assert "risk_uncertainty" in snapshot.scored_df.columns
    assert snapshot.optimization_summary.total_cost <= snapshot.optimization_summary.budget
    assert isinstance(snapshot.policy_briefs, dict)
    assert 0.0 <= snapshot.comparison_metrics.historical.accuracy <= 1.0
    assert 0.0 <= snapshot.comparison_metrics.model.accuracy <= 1.0


def test_build_demo_snapshot_handles_missing_numeric_inputs() -> None:
    df = _sample_df()
    df.loc[0, "lead_90p_ppb"] = None
    df.loc[1, "distance_to_tri_km"] = None
    snapshot = build_demo_snapshot(df, budget=35000)
    assert snapshot.scored_df["risk_score"].isna().sum() == 0
    assert snapshot.scored_df["replacement_cost"].isna().sum() == 0
