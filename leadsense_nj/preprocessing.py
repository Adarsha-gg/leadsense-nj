from __future__ import annotations

from pathlib import Path

import pandas as pd

from leadsense_nj.config import DataConfig
from leadsense_nj.schemas import validate_feature_table


def load_feature_table(path: str | Path) -> pd.DataFrame:
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Feature table not found: {in_path}")
    return pd.read_csv(in_path, dtype={"geoid": str})


def impute_missing_values(df: pd.DataFrame, config: DataConfig | None = None) -> pd.DataFrame:
    cfg = config or DataConfig()
    out = df.copy()
    for col in cfg.numeric_columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            median = out[col].median(skipna=True)
            out[col] = out[col].fillna(median)
    return out


def build_feature_table(path: str | Path | None = None, config: DataConfig | None = None) -> pd.DataFrame:
    cfg = config or DataConfig()
    in_path = Path(path) if path is not None else cfg.default_feature_table_path
    raw_df = load_feature_table(in_path)
    clean_df = impute_missing_values(raw_df, cfg)
    validate_feature_table(clean_df, cfg)
    return clean_df.sort_values("geoid").reset_index(drop=True)
