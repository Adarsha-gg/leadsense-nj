# LeadSense NJ

Incremental implementation of the LeadSense NJ specification with strict
feature gating. Each feature must pass tests before the next one starts.

## Current Status

See `FEATURE_TRACKER.md` for feature-by-feature progress.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Tests

```bash
python -m pytest -q
```

## Feature Gate Check

```bash
python scripts/run_feature_checks.py
```

## Demo App

```bash
streamlit run app/streamlit_app.py
```
