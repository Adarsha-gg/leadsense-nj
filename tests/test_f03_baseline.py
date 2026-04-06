from __future__ import annotations

import numpy as np
import pandas as pd

from leadsense_nj.baseline import fit_tabular_logistic
from leadsense_nj.target import with_elevated_risk_label


def _sample_training_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "median_income": [42000, 81500, 97000, 38500, 71000, 45000, 102000, 36000],
            "poverty_rate": [0.27, 0.11, 0.07, 0.31, 0.14, 0.24, 0.05, 0.34],
            "pct_housing_pre_1950": [0.48, 0.23, 0.17, 0.53, 0.29, 0.41, 0.09, 0.58],
            "lead_90p_ppb": [11.2, 5.1, 3.4, 13.8, 6.0, 9.7, 2.8, 15.7],
            "ph_mean": [6.7, 7.3, 7.6, 6.5, 7.1, 6.8, 7.8, 6.4],
            "alkalinity_mg_l": [28, 56, 62, 24, 48, 27, 66, 22],
            "distance_to_tri_km": [2.4, 5.2, 9.4, 1.6, 7.8, 3.9, 10.5, 1.2],
            "winter_freeze_thaw_days": [42, 37, 33, 44, 36, 41, 32, 45],
            "pws_action_level_exceedance_5y": [0, 0, 0, 1, 0, 0, 0, 1],
            "pws_any_sample_gt15_3y": [1, 0, 0, 1, 0, 0, 0, 1],
            "median_housing_year": [1946, 1968, 1985, 1938, 1958, 1942, 1991, 1935],
        }
    )


def test_fit_tabular_logistic_outputs_model_and_loss_curve() -> None:
    df = with_elevated_risk_label(_sample_training_df())
    model, losses = fit_tabular_logistic(df, epochs=400, learning_rate=0.12)

    assert len(losses) == 400
    assert losses[-1] < losses[0]
    assert model.weights.shape[0] == len(model.feature_columns)


def test_predict_proba_bounds_and_shape() -> None:
    df = with_elevated_risk_label(_sample_training_df())
    model, _ = fit_tabular_logistic(df, epochs=250, learning_rate=0.12)
    proba = model.predict_proba(df)
    assert proba.shape[0] == len(df)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)


def test_training_accuracy_is_reasonable_on_small_dataset() -> None:
    df = with_elevated_risk_label(_sample_training_df())
    model, _ = fit_tabular_logistic(df, epochs=1200, learning_rate=0.1)
    preds = model.predict(df)
    accuracy = float((preds == df["risk_label"].to_numpy()).mean())
    assert accuracy >= 0.75
