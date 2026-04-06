from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leadsense_nj.config import DataConfig
from leadsense_nj.baseline import fit_tabular_logistic
from leadsense_nj.preprocessing import build_feature_table
from leadsense_nj.target import with_elevated_risk_label


def run_feature_01_checks() -> None:
    df = build_feature_table()
    if df.empty:
        raise RuntimeError("F01 failed: feature table is empty.")

    print("F01 checks passed.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Source: {DataConfig().default_feature_table_path}")


def run_feature_02_checks() -> None:
    df = build_feature_table()
    labeled = with_elevated_risk_label(df)
    if "risk_label" not in labeled.columns:
        raise RuntimeError("F02 failed: risk_label column missing.")
    if not labeled["risk_label"].isin([0, 1]).all():
        raise RuntimeError("F02 failed: risk_label contains values other than 0/1.")
    prevalence = float(labeled["risk_label"].mean())
    if prevalence <= 0.0 or prevalence >= 1.0:
        raise RuntimeError("F02 failed: label prevalence collapsed to all-zero or all-one.")

    print("F02 checks passed.")
    print(f"Positive label prevalence: {prevalence:.3f}")


def run_feature_03_checks() -> None:
    df = with_elevated_risk_label(build_feature_table())
    model, losses = fit_tabular_logistic(df, epochs=600, learning_rate=0.1)
    if losses[-1] >= losses[0]:
        raise RuntimeError("F03 failed: training loss did not decrease.")

    proba = model.predict_proba(df)
    if not ((proba >= 0.0) & (proba <= 1.0)).all():
        raise RuntimeError("F03 failed: model produced out-of-bound probabilities.")

    preds = model.predict(df)
    accuracy = float((preds == df["risk_label"]).mean())
    if accuracy < 0.65:
        raise RuntimeError(f"F03 failed: training accuracy too low ({accuracy:.3f}).")

    print("F03 checks passed.")
    print(f"Training accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    run_feature_01_checks()
    run_feature_02_checks()
    run_feature_03_checks()
