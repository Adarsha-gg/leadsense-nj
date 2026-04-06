# LeadSense NJ

Incremental implementation of the LeadSense NJ specification with strict
feature gating. Each feature must pass tests before the next one starts.

## Current Status

See `FEATURE_TRACKER.md` for feature-by-feature progress.
See `RESEARCH_EXECUTION_PLAN.md` for research milestones and evidence workflow.

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

## Research Benchmark

Runs spatial cross-validation comparing:
- Historical signal baseline
- Multimodal fusion model
- Graph-enhanced model

```bash
python scripts/run_research_benchmark.py
```

Outputs:
- `artifacts/research/benchmark_results.json`
- `artifacts/research/benchmark_results.md`

## Demo App

```bash
python app.py
```

Or directly:

```bash
streamlit run app/streamlit_app.py
```
