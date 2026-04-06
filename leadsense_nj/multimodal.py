from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TEMPORAL_COLUMNS: tuple[str, ...] = (
    "q1_lead_ppb",
    "q2_lead_ppb",
    "q3_lead_ppb",
    "q4_lead_ppb",
    "q5_lead_ppb",
    "q6_lead_ppb",
    "q7_lead_ppb",
    "q8_lead_ppb",
)

TABULAR_COLUMNS: tuple[str, ...] = (
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
    "alkalinity_mg_l",
)

VISION_PROXY_COLUMNS: tuple[str, ...] = (
    "pct_housing_pre_1950",
    "poverty_rate",
    "median_income",
    "distance_to_tri_km",
)


def _to_numeric_frame(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    out = df.loc[:, cols].copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
        out[col] = out[col].fillna(out[col].median(skipna=True))
    return out


def build_temporal_features(df: pd.DataFrame, temporal_cols: tuple[str, ...] = TEMPORAL_COLUMNS) -> pd.DataFrame:
    available = [col for col in temporal_cols if col in df.columns]
    if not available:
        return pd.DataFrame(index=df.index, data={"temporal_mean": np.zeros(len(df)), "temporal_std": np.zeros(len(df))})

    tmp = _to_numeric_frame(df, tuple(available))
    features = pd.DataFrame(index=df.index)
    features["temporal_mean"] = tmp.mean(axis=1)
    features["temporal_std"] = tmp.std(axis=1).fillna(0.0)
    features["temporal_trend"] = tmp.iloc[:, -1] - tmp.iloc[:, 0]
    features["temporal_max"] = tmp.max(axis=1)
    return features


def build_vision_proxy_features(df: pd.DataFrame, cols: tuple[str, ...] = VISION_PROXY_COLUMNS) -> pd.DataFrame:
    x = _to_numeric_frame(df, cols)
    features = pd.DataFrame(index=df.index)
    features["vision_old_infra_density"] = x["pct_housing_pre_1950"] * 1.5 + x["poverty_rate"]
    features["vision_affluence_inverse"] = 1.0 / (1.0 + x["median_income"] / 100000.0)
    features["vision_tri_proximity"] = 1.0 / (1.0 + x["distance_to_tri_km"])
    return features


def build_fusion_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    tabular = _to_numeric_frame(df, TABULAR_COLUMNS)
    temporal = build_temporal_features(df)
    vision = build_vision_proxy_features(df)
    fused = pd.concat([tabular, temporal, vision], axis=1)
    return fused


@dataclass(frozen=True)
class FusionRiskModel:
    feature_columns: tuple[str, ...]
    estimator: Pipeline

    def predict_proba(self, fused_df: pd.DataFrame) -> np.ndarray:
        x = fused_df.loc[:, self.feature_columns]
        return self.estimator.predict_proba(x)[:, 1]

    def predict(self, fused_df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(fused_df) >= threshold).astype(int)


def train_fusion_model(df: pd.DataFrame, label_col: str = "risk_label") -> FusionRiskModel:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")
    y = pd.to_numeric(df[label_col], errors="raise").astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("Need at least two classes to train fusion model.")

    fused = build_fusion_feature_table(df)
    feature_columns = tuple(fused.columns.tolist())
    estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )
    estimator.fit(fused, y)
    return FusionRiskModel(feature_columns=feature_columns, estimator=estimator)
