from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leadsense_nj.config import DataConfig
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


if __name__ == "__main__":
    run_feature_01_checks()
    run_feature_02_checks()
