from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    required_columns: tuple[str, ...] = (
        "geoid",
        "county",
        "municipality",
        "median_income",
        "pct_housing_pre_1950",
        "poverty_rate",
        "children_under_6_rate",
        "lead_90p_ppb",
        "ph_mean",
        "hardness_mg_l",
        "chlorine_residual_mg_l",
        "distance_to_tri_km",
        "winter_freeze_thaw_days",
    )
    numeric_columns: tuple[str, ...] = (
        "median_income",
        "pct_housing_pre_1950",
        "poverty_rate",
        "children_under_6_rate",
        "lead_90p_ppb",
        "ph_mean",
        "hardness_mg_l",
        "chlorine_residual_mg_l",
        "distance_to_tri_km",
        "winter_freeze_thaw_days",
    )
    default_feature_table_path: Path = (
        Path(__file__).resolve().parents[1] / "data" / "processed" / "block_group_features_sample.csv"
    )
