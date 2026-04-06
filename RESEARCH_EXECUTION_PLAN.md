# LeadSense NJ Research Execution Plan

This plan converts the technical spec into an evidence-first workflow where every claim is tied to an experiment artifact.

## Principles

1. No metric in slides/paper without a reproducible script and artifact.
2. Every model variant is evaluated with the same spatial CV protocol.
3. We report uncertainty/calibration and fairness, not just raw accuracy.

## Implemented Research Stack

- Historical-rule baseline (`historical_signal_prediction`)
- Multimodal fusion model (tabular + temporal + vision-proxy features)
- Graph-enhanced model (KNN topology + iterative mean message passing)
- Uncertainty and calibration metrics (ECE, Brier)
- Fairness-aware optimizer (greedy + ILP)
- Research benchmark runner and artifact writer

Run:

```bash
python scripts/run_research_benchmark.py
```

Artifacts:

- `artifacts/research/benchmark_results.json`
- `artifacts/research/benchmark_results.md`

## Next Phase (Spec-Complete Research)

### Phase A: Real Data Ingestion

- Add NJ ACS pull scripts for block-group demographics.
- Add EPA/SDWIS ingestion for lead and copper violations/samples.
- Add NJ DEP/NJGIN ingestion hooks for service-area and infrastructure geometry.
- Add cached ingestion artifacts and schema checks.

### Phase B: Full Modalities

- Satellite imagery ingestion and tile-level features from Sentinel-2.
- Replace vision-proxy features with real image encoder features.
- Build quarterly temporal matrices from real historical records.

### Phase C: Advanced Graph Model

- Replace KNN spatial proxy with infrastructure-informed graph edges.
- Add real graph ablations:
  - no-graph
  - KNN graph
  - infrastructure graph

### Phase D: Explainability and Briefing

- Add SHAP-based explainability artifacts for the selected production model.
- Add cached per-block policy brief generation with LLM + offline fallback.

### Phase E: Publication-Grade Evaluation

- Expand spatial CV to 5 folds on full NJ dataset.
- Add calibration plots, fairness tradeoff curves, and optimizer sensitivity.
- Add reproducible result tables ready for poster/paper.
