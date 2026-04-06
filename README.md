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
- Logistic tabular baseline
- Fusion tabular-only
- Fusion tabular+temporal
- Multimodal fusion model
- Graph-enhanced model

```bash
python scripts/run_research_benchmark.py --dataset data/processed/nj_research_features.csv --n-splits 5 --max-rows 2500
```

Outputs:
- `artifacts/research/benchmark_results.json`
- `artifacts/research/benchmark_results.md`

## Expanded Research Dataset (F14)

Builds an NJ-wide benchmark dataset from ACS cache:

```bash
python scripts/build_research_dataset.py --acs-cache data/cache/acs_nj_block_groups_2022.csv --out data/processed/nj_research_features.csv --seed 42
```

## Infrastructure Graph Edges (F11)

Builds a proxy infrastructure edge list (county-constrained KNN), used as a hook
until NJ DEP/NJGIN network edges are wired in.

```bash
python scripts/build_infrastructure_edges.py --feature-table data/processed/block_group_features_sample.csv --out data/processed/graph_edges_sample.csv
```

## Sentinel-2 Tile Features (F12)

Fetches Sentinel-2 STAC item metadata per block-group point and caches derived
vision features for modeling.

```bash
python scripts/fetch_sentinel_features.py --feature-table data/processed/block_group_features_sample.csv --cache-dir data/cache --start-date 2024-04-01 --end-date 2024-10-31 --items-per-block 1 --max-cloud-cover 60
```

Outputs:
- `data/cache/sentinel_features_sample.csv`
- `data/cache/sentinel_features_metadata.json`

## Real Data Ingestion (F10)

Fetches real NJ data from:
- Census ACS block-group API
- EPA SDWIS efservice tables (`LCR_SAMPLE`, `LCR_SAMPLE_RESULT`, lead `VIOLATION`)

```bash
python scripts/fetch_real_data.py --cache-dir data/cache --acs-year 2022 --max-violation-rows 10000
```

Outputs:
- `data/cache/acs_nj_block_groups_2022.csv`
- `data/cache/epa_nj_lcr_samples.csv`
- `data/cache/epa_nj_lcr_sample_results.csv`
- `data/cache/epa_nj_lead_violations.csv`
- `data/cache/epa_nj_pws_lead_signals.csv`
- `data/cache/ingestion_metadata.json`

## Demo App (JS Frontend + Python Backend)

```bash
python app.py
```

Then open: `http://127.0.0.1:8000`

Notes:
- Dashboard now uses expanded NJ research dataset when available (`data/processed/nj_research_features.csv`).
- Performance cards show cross-validated metrics from benchmark artifacts.
- AI Copilot tab supports:
  - area-level Q&A and policy brief generation
  - AI Portfolio Designer: converts a natural-language policy goal into objective weights/constraints and generates a replacement portfolio with baseline deltas
  Both use LLM responses when `OPENAI_API_KEY` is set; otherwise deterministic fallback is used.

Enable AI Copilot:

```bash
set OPENAI_API_KEY=your_key_here
set OPENAI_MODEL=gpt-4.1-mini
python app.py
```

## Streamlit App (Legacy)

```bash
python app_streamlit.py
```
