from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_BASELINE_FEATURES: tuple[str, ...] = (
    "median_income",
    "poverty_rate",
    "pct_housing_pre_1950",
    "lead_90p_ppb",
    "ph_mean",
    "alkalinity_mg_l",
    "distance_to_tri_km",
    "winter_freeze_thaw_days",
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Clip to avoid overflow in exp on large magnitudes.
    clipped = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class TabularBaselineModel:
    feature_columns: tuple[str, ...]
    means: np.ndarray
    stds: np.ndarray
    weights: np.ndarray
    bias: float

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df.loc[:, self.feature_columns].astype(float).to_numpy()
        return (x - self.means) / self.stds

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x = self._transform(df)
        logits = x @ self.weights + self.bias
        return _sigmoid(logits)

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(df) >= threshold).astype(int)


def _binary_log_loss(y_true: np.ndarray, y_pred_proba: np.ndarray, l2: float, weights: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(y_pred_proba, eps, 1.0 - eps)
    ce = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)).mean()
    return float(ce + (l2 * (weights**2).sum() / 2.0))


def fit_tabular_logistic(
    df: pd.DataFrame,
    label_column: str = "risk_label",
    feature_columns: tuple[str, ...] = DEFAULT_BASELINE_FEATURES,
    learning_rate: float = 0.1,
    epochs: int = 1200,
    l2: float = 0.001,
) -> tuple[TabularBaselineModel, list[float]]:
    missing = sorted(set(feature_columns + (label_column,)).difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for baseline model: {missing}")

    x_raw = df.loc[:, feature_columns].astype(float).to_numpy()
    y = df[label_column].astype(int).to_numpy()
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Label column must be binary (0/1).")
    if len(np.unique(y)) < 2:
        raise ValueError("Label column needs at least one positive and one negative sample.")

    means = x_raw.mean(axis=0)
    stds = x_raw.std(axis=0)
    stds[stds == 0.0] = 1.0
    x = (x_raw - means) / stds

    n_samples, n_features = x.shape
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0

    losses: list[float] = []
    for _ in range(epochs):
        logits = x @ weights + bias
        probs = _sigmoid(logits)

        dw = (x.T @ (probs - y)) / n_samples + l2 * weights
        db = float((probs - y).mean())

        weights -= learning_rate * dw
        bias -= learning_rate * db
        losses.append(_binary_log_loss(y, probs, l2=l2, weights=weights))

    model = TabularBaselineModel(
        feature_columns=feature_columns,
        means=means,
        stds=stds,
        weights=weights,
        bias=bias,
    )
    return model, losses
