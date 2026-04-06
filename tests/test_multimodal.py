from __future__ import annotations

import pandas as pd

from leadsense_nj.multimodal import build_fusion_feature_table, build_temporal_features, train_fusion_model
from leadsense_nj.target import with_elevated_risk_label


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "geoid": ["1", "2", "3", "4", "5", "6"],
            "median_income": [42000, 81500, 97000, 38500, 71000, 45000],
            "pct_housing_pre_1950": [0.48, 0.23, 0.17, 0.53, 0.29, 0.41],
            "poverty_rate": [0.27, 0.11, 0.07, 0.31, 0.14, 0.24],
            "children_under_6_rate": [0.09, 0.06, 0.05, 0.11, 0.07, 0.10],
            "lead_90p_ppb": [11.2, 5.1, 3.4, 13.8, 6.0, 9.7],
            "ph_mean": [6.7, 7.3, 7.6, 6.5, 7.1, 6.8],
            "hardness_mg_l": [104, 126, 140, 98, 132, 112],
            "chlorine_residual_mg_l": [1.2, 1.8, 2.1, 1.0, 1.7, 1.3],
            "distance_to_tri_km": [2.4, 5.2, 9.4, 1.6, 7.8, 3.9],
            "winter_freeze_thaw_days": [42, 37, 33, 44, 36, 41],
            "alkalinity_mg_l": [28, 56, 62, 24, 48, 27],
            "pws_action_level_exceedance_5y": [0, 0, 0, 1, 0, 0],
            "pws_any_sample_gt15_3y": [1, 0, 0, 1, 0, 0],
            "median_housing_year": [1946, 1968, 1985, 1938, 1958, 1942],
            "q1_lead_ppb": [12.8, 5.6, 3.9, 14.7, 6.4, 10.1],
            "q2_lead_ppb": [11.9, 5.1, 3.5, 13.9, 6.1, 9.6],
            "q3_lead_ppb": [10.7, 4.8, 3.4, 13.1, 5.8, 9.2],
            "q4_lead_ppb": [9.8, 4.3, 3.2, 12.8, 5.6, 8.7],
            "q5_lead_ppb": [11.3, 5.0, 3.1, 14.2, 5.9, 9.5],
            "q6_lead_ppb": [12.1, 4.6, 3.6, 13.6, 5.7, 9.1],
            "q7_lead_ppb": [10.9, 4.9, 3.3, 13.0, 6.0, 8.9],
            "q8_lead_ppb": [11.5, 5.2, 3.4, 13.4, 5.8, 9.3],
        }
    )


def test_temporal_features_created() -> None:
    out = build_temporal_features(_sample_df())
    assert {"temporal_mean", "temporal_std", "temporal_trend", "temporal_max"}.issubset(out.columns)
    assert len(out) == 6


def test_fusion_feature_table_not_empty() -> None:
    fused = build_fusion_feature_table(_sample_df())
    assert len(fused) == 6
    assert fused.shape[1] >= 12


def test_fusion_feature_table_allows_modality_ablation() -> None:
    df = _sample_df()
    tabular_only = build_fusion_feature_table(df, include_temporal=False, include_vision=False)
    tabular_temporal = build_fusion_feature_table(df, include_temporal=True, include_vision=False)
    assert len(tabular_only.columns) < len(tabular_temporal.columns)


def test_train_fusion_model_predicts() -> None:
    df = with_elevated_risk_label(_sample_df())
    model = train_fusion_model(df)
    fused = build_fusion_feature_table(df)
    proba = model.predict_proba(fused)
    assert len(proba) == len(df)
    assert float(proba.min()) >= 0.0
    assert float(proba.max()) <= 1.0


def test_fusion_uses_satellite_columns_when_available() -> None:
    df = with_elevated_risk_label(_sample_df())
    df["s2_cloud_cover_mean"] = [10, 12, 20, 30, 8, 16]
    df["s2_vegetation_pct_mean"] = [35, 31, 40, 25, 45, 29]
    df["s2_water_pct_mean"] = [4, 2, 3, 5, 1, 2]
    df["s2_nodata_pct_mean"] = [1, 1, 2, 3, 1, 2]
    df["s2_item_count"] = [2, 2, 2, 2, 2, 2]
    df["s2_days_since_latest"] = [15, 18, 21, 25, 11, 19]
    fused = build_fusion_feature_table(df)
    sat_cols = [c for c in fused.columns if c.startswith("vision_sat_")]
    assert len(sat_cols) >= 4
