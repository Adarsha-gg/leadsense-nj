from __future__ import annotations

import numpy as np
import pandas as pd

from leadsense_nj.target import with_elevated_risk_label
from leadsense_nj.uncertainty import expected_calibration_error, train_bootstrap_ensemble


def _sample_training_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "median_income": [42000, 81500, 97000, 38500, 71000, 45000, 102000, 36000, 89000, 52000],
            "poverty_rate": [0.27, 0.11, 0.07, 0.31, 0.14, 0.24, 0.05, 0.34, 0.09, 0.20],
            "pct_housing_pre_1950": [0.48, 0.23, 0.17, 0.53, 0.29, 0.41, 0.09, 0.58, 0.18, 0.37],
            "lead_90p_ppb": [11.2, 5.1, 3.4, 13.8, 6.0, 9.7, 2.8, 15.7, 4.1, 8.8],
            "ph_mean": [6.7, 7.3, 7.6, 6.5, 7.1, 6.8, 7.8, 6.4, 7.4, 6.9],
            "alkalinity_mg_l": [28, 56, 62, 24, 48, 27, 66, 22, 58, 30],
            "distance_to_tri_km": [2.4, 5.2, 9.4, 1.6, 7.8, 3.9, 10.5, 1.2, 8.9, 4.3],
            "winter_freeze_thaw_days": [42, 37, 33, 44, 36, 41, 32, 45, 34, 39],
            "pws_action_level_exceedance_5y": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            "pws_any_sample_gt15_3y": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            "median_housing_year": [1946, 1968, 1985, 1938, 1958, 1942, 1991, 1935, 1979, 1948],
        }
    )


def test_bootstrap_ensemble_outputs_mean_and_std() -> None:
    df = with_elevated_risk_label(_sample_training_df())
    ensemble = train_bootstrap_ensemble(df, n_models=12, epochs=300, learning_rate=0.1)
    mean, std = ensemble.predict_mean_std(df)

    assert mean.shape[0] == len(df)
    assert std.shape[0] == len(df)
    assert np.all(mean >= 0.0) and np.all(mean <= 1.0)
    assert np.all(std >= 0.0)
    assert float(std.mean()) > 0.0


def test_bootstrap_prediction_intervals_are_bounded() -> None:
    df = with_elevated_risk_label(_sample_training_df())
    ensemble = train_bootstrap_ensemble(df, n_models=8, epochs=250, learning_rate=0.1)
    lower, upper = ensemble.predict_interval(df)

    assert np.all(lower >= 0.0) and np.all(lower <= 1.0)
    assert np.all(upper >= 0.0) and np.all(upper <= 1.0)
    assert np.all(lower <= upper)


def test_expected_calibration_error_range() -> None:
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.85, 0.72, 0.91, 0.3, 0.66, 0.4])
    ece = expected_calibration_error(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0
