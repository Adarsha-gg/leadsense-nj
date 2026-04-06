from __future__ import annotations

from pathlib import Path

from leadsense_nj.preprocessing import build_feature_table
from leadsense_nj.research import run_model_research_benchmark, spatial_kfold_splits
from leadsense_nj.target import with_elevated_risk_label


def test_spatial_kfold_splits_non_empty() -> None:
    df = with_elevated_risk_label(build_feature_table(Path("data/processed/block_group_features_sample.csv")))
    splits = spatial_kfold_splits(df, n_splits=3, random_state=42)
    assert len(splits) >= 2
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0
        assert len(test_idx) > 0


def test_run_model_research_benchmark_has_expected_keys() -> None:
    df = build_feature_table(Path("data/processed/block_group_features_sample.csv"))
    report = run_model_research_benchmark(df, n_splits=3, threshold=0.5, random_state=42)
    assert report["n_folds"] >= 2
    assert "historical" in report
    assert "baseline_tabular" in report
    assert "tabular" in report
    assert "tabular_temporal" in report
    assert "fusion" in report
    assert "graph_knn" in report
    assert "graph_infrastructure" in report
    assert "graph" in report
    assert "ablation_accuracy_table" in report
    assert "fold_results" in report
