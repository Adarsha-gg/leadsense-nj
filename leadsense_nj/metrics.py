from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from leadsense_nj.uncertainty import expected_calibration_error


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    positive_rate: float
    tp: int
    fp: int
    tn: int
    fn: int


@dataclass(frozen=True)
class ModelVsHistoricalMetrics:
    historical: BinaryClassificationMetrics
    model: BinaryClassificationMetrics
    model_ece: float
    model_brier: float
    model_threshold: float
    accuracy_delta_model_minus_historical: float


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> BinaryClassificationMetrics:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = max(len(y_true), 1)
    accuracy = _safe_div(tp + tn, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    positive_rate = float((y_pred == 1).mean()) if len(y_pred) > 0 else 0.0

    return BinaryClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        positive_rate=positive_rate,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def historical_signal_prediction(df: pd.DataFrame) -> np.ndarray:
    required = ["pws_action_level_exceedance_5y", "pws_any_sample_gt15_3y", "lead_90p_ppb"]
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for historical prediction: {missing}")

    action = pd.to_numeric(df["pws_action_level_exceedance_5y"], errors="coerce").fillna(0) > 0
    sample = pd.to_numeric(df["pws_any_sample_gt15_3y"], errors="coerce").fillna(0) > 0
    lead_90p = pd.to_numeric(df["lead_90p_ppb"], errors="coerce").fillna(0.0) > 15.0
    return (action | sample | lead_90p).astype(int).to_numpy()


def compute_model_vs_historical_metrics(
    scored_df: pd.DataFrame,
    *,
    label_col: str = "risk_label",
    model_score_col: str = "risk_score",
    model_threshold: float = 0.5,
    ece_bins: int = 10,
) -> ModelVsHistoricalMetrics:
    required = [label_col, model_score_col]
    missing = sorted(set(required).difference(scored_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for metrics: {missing}")

    y_true = pd.to_numeric(scored_df[label_col], errors="raise").astype(int).to_numpy()
    y_prob = pd.to_numeric(scored_df[model_score_col], errors="coerce").fillna(0.0).to_numpy()
    y_prob = np.clip(y_prob.astype(float), 0.0, 1.0)
    y_model = (y_prob >= model_threshold).astype(int)
    y_hist = historical_signal_prediction(scored_df)

    hist_metrics = compute_binary_metrics(y_true, y_hist)
    model_metrics = compute_binary_metrics(y_true, y_model)
    model_ece = expected_calibration_error(y_true, y_prob, n_bins=ece_bins)
    model_brier = float(np.mean((y_prob - y_true) ** 2))

    return ModelVsHistoricalMetrics(
        historical=hist_metrics,
        model=model_metrics,
        model_ece=model_ece,
        model_brier=model_brier,
        model_threshold=model_threshold,
        accuracy_delta_model_minus_historical=float(model_metrics.accuracy - hist_metrics.accuracy),
    )
