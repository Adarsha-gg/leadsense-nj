from __future__ import annotations

import pandas as pd


def _coerce_bool(series: pd.Series, column: str) -> pd.Series:
    values = series.map(
        {
            True: True,
            False: False,
            1: True,
            0: False,
            "1": True,
            "0": False,
            "true": True,
            "false": False,
            "True": True,
            "False": False,
            "yes": True,
            "no": False,
            "Y": True,
            "N": False,
        }
    )
    if values.isna().any():
        bad_values = series[values.isna()].unique().tolist()
        raise ValueError(f"Column '{column}' contains invalid boolean values: {bad_values}")
    return values.astype(bool)


def construct_elevated_risk_label(df: pd.DataFrame) -> pd.Series:
    required = [
        "pws_action_level_exceedance_5y",
        "pws_any_sample_gt15_3y",
        "median_housing_year",
        "ph_mean",
        "alkalinity_mg_l",
    ]
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for label construction: {missing}")

    action_exceedance = _coerce_bool(df["pws_action_level_exceedance_5y"], "pws_action_level_exceedance_5y")
    sample_exceedance = _coerce_bool(df["pws_any_sample_gt15_3y"], "pws_any_sample_gt15_3y")
    housing_year = pd.to_numeric(df["median_housing_year"], errors="raise")
    ph_mean = pd.to_numeric(df["ph_mean"], errors="raise")
    alkalinity = pd.to_numeric(df["alkalinity_mg_l"], errors="raise")

    old_housing = housing_year < 1950
    leaching_profile = (ph_mean < 7.0) | (alkalinity < 30.0)

    positive = action_exceedance | sample_exceedance | (old_housing & leaching_profile)
    return positive.astype(int)


def with_elevated_risk_label(df: pd.DataFrame, label_column: str = "risk_label") -> pd.DataFrame:
    out = df.copy()
    out[label_column] = construct_elevated_risk_label(out)
    return out
