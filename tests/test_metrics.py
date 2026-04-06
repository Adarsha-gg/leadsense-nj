from __future__ import annotations

import numpy as np
import pandas as pd

from leadsense_nj.metrics import (
    compute_binary_metrics,
    compute_model_vs_historical_metrics,
    compute_probabilistic_metrics,
)


def test_compute_binary_metrics_ranges() -> None:
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 1, 0, 0])
    metrics = compute_binary_metrics(y_true, y_pred)

    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert 0.0 <= metrics.specificity <= 1.0
    assert metrics.tp + metrics.fp + metrics.tn + metrics.fn == len(y_true)


def test_compute_model_vs_historical_metrics_outputs_comparison() -> None:
    df = pd.DataFrame(
        {
            "risk_label": [1, 0, 1, 0, 1, 0],
            "risk_score": [0.92, 0.31, 0.77, 0.40, 0.68, 0.22],
            "pws_action_level_exceedance_5y": [1, 0, 0, 0, 0, 0],
            "pws_any_sample_gt15_3y": [1, 0, 1, 0, 0, 0],
            "lead_90p_ppb": [12.5, 5.0, 9.3, 4.8, 8.1, 3.2],
        }
    )
    result = compute_model_vs_historical_metrics(df, model_threshold=0.5, ece_bins=5)

    assert 0.0 <= result.historical.accuracy <= 1.0
    assert 0.0 <= result.model.accuracy <= 1.0
    assert 0.0 <= result.model_ece <= 1.0
    assert 0.0 <= result.model_brier <= 1.0
    assert 0.0 <= result.model_auroc <= 1.0
    assert 0.0 <= result.model_auprc <= 1.0


def test_compute_probabilistic_metrics_outputs_ranking_metrics() -> None:
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.92, 0.21, 0.73, 0.44, 0.61, 0.18])
    metrics = compute_probabilistic_metrics(y_true, y_prob, threshold=0.5, ece_bins=5)

    assert 0.0 <= metrics.auroc <= 1.0
    assert 0.0 <= metrics.auprc <= 1.0
    assert 0.0 <= metrics.brier <= 1.0
    assert 0.0 <= metrics.ece <= 1.0
