from __future__ import annotations

import pandas as pd
import pytest

from leadsense_nj.target import construct_elevated_risk_label, with_elevated_risk_label


def test_construct_elevated_risk_label_covers_all_positive_paths() -> None:
    df = pd.DataFrame(
        {
            "pws_action_level_exceedance_5y": [1, 0, 0, 0],
            "pws_any_sample_gt15_3y": [0, 1, 0, 0],
            "median_housing_year": [1975, 1985, 1940, 1982],
            "ph_mean": [7.3, 7.2, 6.8, 7.4],
            "alkalinity_mg_l": [45, 50, 35, 60],
        }
    )

    labels = construct_elevated_risk_label(df)
    assert labels.tolist() == [1, 1, 1, 0]


def test_construct_elevated_risk_label_handles_alkalinity_branch() -> None:
    df = pd.DataFrame(
        {
            "pws_action_level_exceedance_5y": [False],
            "pws_any_sample_gt15_3y": [False],
            "median_housing_year": [1945],
            "ph_mean": [7.2],
            "alkalinity_mg_l": [25],
        }
    )
    labels = construct_elevated_risk_label(df)
    assert labels.tolist() == [1]


def test_construct_elevated_risk_label_rejects_missing_columns() -> None:
    df = pd.DataFrame(
        {
            "pws_action_level_exceedance_5y": [1],
            "median_housing_year": [1940],
            "ph_mean": [6.8],
            "alkalinity_mg_l": [20],
        }
    )
    with pytest.raises(ValueError, match="Missing required columns"):
        construct_elevated_risk_label(df)


def test_with_elevated_risk_label_adds_column() -> None:
    df = pd.DataFrame(
        {
            "pws_action_level_exceedance_5y": ["true", "false"],
            "pws_any_sample_gt15_3y": ["false", "false"],
            "median_housing_year": [1965, 1988],
            "ph_mean": [7.2, 7.6],
            "alkalinity_mg_l": [55, 70],
        }
    )
    out = with_elevated_risk_label(df, label_column="elevated_lead_risk")
    assert "elevated_lead_risk" in out.columns
    assert out["elevated_lead_risk"].tolist() == [1, 0]
