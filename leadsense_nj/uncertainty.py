from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from leadsense_nj.baseline import DEFAULT_BASELINE_FEATURES, TabularBaselineModel, fit_tabular_logistic


@dataclass(frozen=True)
class BootstrappedRiskEnsemble:
    models: tuple[TabularBaselineModel, ...]

    def predict_distribution(self, df: pd.DataFrame) -> np.ndarray:
        preds = [model.predict_proba(df) for model in self.models]
        return np.vstack(preds)

    def predict_mean_std(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        dist = self.predict_distribution(df)
        return dist.mean(axis=0), dist.std(axis=0)

    def predict_interval(self, df: pd.DataFrame, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
        mean, std = self.predict_mean_std(df)
        lower = np.clip(mean - z * std, 0.0, 1.0)
        upper = np.clip(mean + z * std, 0.0, 1.0)
        return lower, upper


def train_bootstrap_ensemble(
    df: pd.DataFrame,
    n_models: int = 40,
    label_column: str = "risk_label",
    feature_columns: tuple[str, ...] = DEFAULT_BASELINE_FEATURES,
    epochs: int = 600,
    learning_rate: float = 0.1,
    l2: float = 0.001,
    seed: int = 7,
) -> BootstrappedRiskEnsemble:
    if n_models < 2:
        raise ValueError("n_models must be >= 2")

    rng = np.random.default_rng(seed)
    n = len(df)
    labels = df[label_column].astype(int).to_numpy()
    if len(np.unique(labels)) < 2:
        raise ValueError("Bootstrap ensemble requires at least one positive and one negative label.")

    models: list[TabularBaselineModel] = []
    for _ in range(n_models):
        # Resample until both classes exist in the bootstrap sample.
        for _attempt in range(25):
            idx = rng.integers(0, n, size=n)
            sampled = df.iloc[idx].reset_index(drop=True)
            sampled_labels = sampled[label_column].astype(int).to_numpy()
            if len(np.unique(sampled_labels)) >= 2:
                break
        else:
            sampled = df

        model, _ = fit_tabular_logistic(
            sampled,
            label_column=label_column,
            feature_columns=feature_columns,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        models.append(model)

    return BootstrappedRiskEnsemble(models=tuple(models))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("y_true and y_prob must have same length.")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")

    y_true = y_true.astype(int)
    y_prob = np.clip(y_prob.astype(float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # Include the right edge only on the final bin.
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)
