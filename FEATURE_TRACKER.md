# Feature Tracker

## F01 - Data Foundation

- Status: `completed`
- Scope:
  - Project scaffold for `leadsense_nj`
  - Feature table loader
  - Required-column and GEOID validation
  - Numeric range validation
  - Missing-value imputation
  - Automated feature check script
- Exit criteria:
  - `python -m pytest -q` passes
  - `python scripts/run_feature_checks.py` passes
- Verification:
  - `python -m pytest -q` -> `4 passed`
  - `python scripts/run_feature_checks.py` -> `F01 checks passed`

## F02 - Target Construction

- Status: `completed`
- Scope:
  - Deterministic label logic from Spec Section 5.4
  - Branch coverage tests for all positive paths and negative control
  - Integrated `risk_label` feature check in script
- Exit criteria:
  - `python -m pytest -q` passes
  - `python scripts/run_feature_checks.py` passes
- Verification:
  - `python -m pytest -q` -> `8 passed`
  - `python scripts/run_feature_checks.py` -> `F02 checks passed`

## F03 - Baseline Predictor

- Status: `completed`
- Scope:
  - In-repo logistic baseline model (NumPy) with feature standardization
  - Probability prediction and thresholded classification
  - Loss tracking and guardrails for binary labels
  - Feature-check integration for training loss and accuracy
- Exit criteria:
  - `python -m pytest -q` passes
  - `python scripts/run_feature_checks.py` passes
- Verification:
  - `python -m pytest -q` -> `11 passed`
  - `python scripts/run_feature_checks.py` -> `F03 checks passed`

## F04 - Uncertainty Layer

- Status: `completed`
- Scope:
  - Bootstrapped uncertainty ensemble over baseline predictor
  - Mean/std risk output and confidence interval generation
  - Expected Calibration Error (ECE) metric
  - Feature-check integration for uncertainty and calibration bounds
- Exit criteria:
  - `python -m pytest -q` passes
  - `python scripts/run_feature_checks.py` passes
- Verification:
  - `python -m pytest -q` -> `14 passed`
  - `python scripts/run_feature_checks.py` -> `F04 checks passed`

## F05 - Fairness Optimizer

- Status: `pending`

## F06 - Explainability + Policy Brief

- Status: `pending`

## F07 - Demo App

- Status: `pending`
