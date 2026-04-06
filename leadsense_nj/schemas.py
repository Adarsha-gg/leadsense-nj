from __future__ import annotations

import re

import pandas as pd

from leadsense_nj.config import DataConfig

_GEOID_RE = re.compile(r"^\d{12}$")


def _validate_geoid(df: pd.DataFrame) -> None:
    geoid = df["geoid"].astype(str)
    bad_mask = ~geoid.str.match(_GEOID_RE)
    if bad_mask.any():
        bad_values = geoid[bad_mask].unique().tolist()
        raise ValueError(f"Invalid GEOID values: {bad_values}")

    if geoid.duplicated().any():
        dupes = geoid[geoid.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate GEOID values: {dupes}")


def _validate_numeric_ranges(df: pd.DataFrame) -> None:
    checks = [
        ("median_income", df["median_income"] >= 0),
        ("pct_housing_pre_1950", (df["pct_housing_pre_1950"] >= 0) & (df["pct_housing_pre_1950"] <= 1)),
        ("poverty_rate", (df["poverty_rate"] >= 0) & (df["poverty_rate"] <= 1)),
        ("children_under_6_rate", (df["children_under_6_rate"] >= 0) & (df["children_under_6_rate"] <= 1)),
        ("lead_90p_ppb", df["lead_90p_ppb"] >= 0),
        ("ph_mean", (df["ph_mean"] >= 0) & (df["ph_mean"] <= 14)),
        ("hardness_mg_l", df["hardness_mg_l"] >= 0),
        ("chlorine_residual_mg_l", df["chlorine_residual_mg_l"] >= 0),
        ("distance_to_tri_km", df["distance_to_tri_km"] >= 0),
        ("winter_freeze_thaw_days", df["winter_freeze_thaw_days"] >= 0),
    ]

    failures: list[str] = []
    for column, mask in checks:
        if not bool(mask.all()):
            failures.append(column)

    if failures:
        raise ValueError(f"Out-of-range values detected in columns: {failures}")


def validate_feature_table(df: pd.DataFrame, config: DataConfig | None = None) -> None:
    cfg = config or DataConfig()

    missing_cols = sorted(set(cfg.required_columns).difference(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Enforce numeric conversion before range checks.
    for col in cfg.numeric_columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except Exception as exc:  # pragma: no cover - pandas exception types vary by version
            raise ValueError(f"Column '{col}' contains non-numeric values") from exc

    _validate_geoid(df)
    _validate_numeric_ranges(df)
