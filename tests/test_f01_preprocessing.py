from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from leadsense_nj.preprocessing import build_feature_table, impute_missing_values
from leadsense_nj.schemas import validate_feature_table


def test_build_feature_table_loads_and_imputes() -> None:
    data_path = Path("data/processed/block_group_features_sample.csv")
    df = build_feature_table(data_path)

    assert len(df) == 6
    assert df["lead_90p_ppb"].isna().sum() == 0
    assert df["hardness_mg_l"].isna().sum() == 0
    assert df["geoid"].str.len().eq(12).all()


def test_validate_feature_table_rejects_bad_geoid() -> None:
    df = pd.DataFrame(
        {
            "geoid": ["340010001001", "invalid_geoid"],
            "county": ["Atlantic", "Bergen"],
            "municipality": ["Atlantic City", "Hackensack"],
            "median_income": [42000, 81500],
            "pct_housing_pre_1950": [0.48, 0.23],
            "poverty_rate": [0.27, 0.11],
            "children_under_6_rate": [0.09, 0.06],
            "lead_90p_ppb": [11.2, 5.1],
            "ph_mean": [6.7, 7.3],
            "hardness_mg_l": [104, 126],
            "chlorine_residual_mg_l": [1.2, 1.8],
            "distance_to_tri_km": [2.4, 5.2],
            "winter_freeze_thaw_days": [42, 37],
        }
    )

    with pytest.raises(ValueError, match="Invalid GEOID"):
        validate_feature_table(df)


def test_validate_feature_table_rejects_out_of_range_values() -> None:
    df = pd.DataFrame(
        {
            "geoid": ["340010001001"],
            "county": ["Atlantic"],
            "municipality": ["Atlantic City"],
            "median_income": [42000],
            "pct_housing_pre_1950": [1.4],
            "poverty_rate": [0.27],
            "children_under_6_rate": [0.09],
            "lead_90p_ppb": [11.2],
            "ph_mean": [6.7],
            "hardness_mg_l": [104],
            "chlorine_residual_mg_l": [1.2],
            "distance_to_tri_km": [2.4],
            "winter_freeze_thaw_days": [42],
        }
    )

    with pytest.raises(ValueError, match="Out-of-range"):
        validate_feature_table(df)


def test_impute_missing_values_fills_numeric_nan() -> None:
    df = pd.DataFrame(
        {
            "geoid": ["340010001001", "340030010002", "340050023001"],
            "county": ["Atlantic", "Bergen", "Burlington"],
            "municipality": ["Atlantic City", "Hackensack", "Mount Laurel"],
            "median_income": [42000, 81500, None],
            "pct_housing_pre_1950": [0.48, 0.23, 0.17],
            "poverty_rate": [0.27, 0.11, 0.07],
            "children_under_6_rate": [0.09, 0.06, 0.05],
            "lead_90p_ppb": [11.2, 5.1, 3.4],
            "ph_mean": [6.7, 7.3, 7.6],
            "hardness_mg_l": [104, 126, 140],
            "chlorine_residual_mg_l": [1.2, 1.8, 2.1],
            "distance_to_tri_km": [2.4, 5.2, 9.4],
            "winter_freeze_thaw_days": [42, 37, 33],
        }
    )
    out = impute_missing_values(df)
    assert out["median_income"].isna().sum() == 0
