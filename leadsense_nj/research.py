from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold

from leadsense_nj.baseline import fit_tabular_logistic
from leadsense_nj.graph_model import train_graph_enhanced_model
from leadsense_nj.metrics import BinaryClassificationMetrics, compute_probabilistic_metrics, historical_signal_prediction
from leadsense_nj.multimodal import build_fusion_feature_table, train_fusion_model
from leadsense_nj.target import with_elevated_risk_label


@dataclass(frozen=True)
class FoldResult:
    fold_id: int
    historical: BinaryClassificationMetrics
    baseline_tabular: BinaryClassificationMetrics
    tabular: BinaryClassificationMetrics
    tabular_temporal: BinaryClassificationMetrics
    fusion: BinaryClassificationMetrics
    graph: BinaryClassificationMetrics


def _resolve_spatial_clusters(df: pd.DataFrame, n_splits: int, random_state: int = 42) -> np.ndarray:
    if {"lat", "lon"}.issubset(df.columns):
        coords = df[["lat", "lon"]].copy()
        coords["lat"] = pd.to_numeric(coords["lat"], errors="coerce")
        coords["lon"] = pd.to_numeric(coords["lon"], errors="coerce")
        coords = coords.fillna(coords.median(numeric_only=True))
        n_clusters = min(n_splits, len(df))
        if n_clusters <= 1:
            return np.zeros(len(df), dtype=int)
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        return km.fit_predict(coords.to_numpy())

    # Fallback when coordinates are unavailable.
    y = pd.to_numeric(df["risk_label"], errors="raise").astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=min(n_splits, len(df)), shuffle=True, random_state=random_state)
    folds = np.zeros(len(df), dtype=int)
    for fold_id, (_, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[test_idx] = fold_id
    return folds


def spatial_kfold_splits(df: pd.DataFrame, n_splits: int = 3, random_state: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(df) < 3:
        raise ValueError("Need at least 3 rows for spatial cross-validation.")
    clusters = _resolve_spatial_clusters(df, n_splits=n_splits, random_state=random_state)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in sorted(np.unique(clusters).tolist()):
        test_idx = np.where(clusters == fold)[0]
        train_idx = np.where(clusters != fold)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits


def _mean_std(rows: list[float]) -> dict[str, float]:
    arr = np.asarray(rows, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def run_model_research_benchmark(
    df: pd.DataFrame,
    *,
    n_splits: int = 3,
    threshold: float = 0.5,
    random_state: int = 42,
) -> dict:
    labeled = with_elevated_risk_label(df)
    splits = spatial_kfold_splits(labeled, n_splits=n_splits, random_state=random_state)
    if not splits:
        raise RuntimeError("Unable to build valid CV splits.")

    fold_results: list[FoldResult] = []
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        train_df = labeled.iloc[train_idx].reset_index(drop=True)
        test_df = labeled.iloc[test_idx].reset_index(drop=True)
        y_true = pd.to_numeric(test_df["risk_label"], errors="raise").astype(int).to_numpy()

        y_hist_prob = historical_signal_prediction(test_df).astype(float)
        hist_metrics = compute_probabilistic_metrics(y_true, y_hist_prob, threshold=threshold, ece_bins=5)

        baseline_model, _ = fit_tabular_logistic(train_df, label_column="risk_label", epochs=700, learning_rate=0.1)
        y_baseline_prob = baseline_model.predict_proba(test_df)
        baseline_metrics = compute_probabilistic_metrics(y_true, y_baseline_prob, threshold=threshold, ece_bins=5)

        tabular_model = train_fusion_model(
            train_df,
            label_col="risk_label",
            include_tabular=True,
            include_temporal=False,
            include_vision=False,
        )
        tabular_test = build_fusion_feature_table(
            test_df,
            include_tabular=True,
            include_temporal=False,
            include_vision=False,
        )
        y_tabular_prob = tabular_model.predict_proba(tabular_test)
        tabular_metrics = compute_probabilistic_metrics(y_true, y_tabular_prob, threshold=threshold, ece_bins=5)

        temporal_model = train_fusion_model(
            train_df,
            label_col="risk_label",
            include_tabular=True,
            include_temporal=True,
            include_vision=False,
        )
        temporal_test = build_fusion_feature_table(
            test_df,
            include_tabular=True,
            include_temporal=True,
            include_vision=False,
        )
        y_temporal_prob = temporal_model.predict_proba(temporal_test)
        temporal_metrics = compute_probabilistic_metrics(y_true, y_temporal_prob, threshold=threshold, ece_bins=5)

        fusion_model = train_fusion_model(
            train_df,
            label_col="risk_label",
            include_tabular=True,
            include_temporal=True,
            include_vision=True,
        )
        fused_test = build_fusion_feature_table(test_df)
        y_fusion_prob = fusion_model.predict_proba(fused_test)
        fusion_metrics = compute_probabilistic_metrics(y_true, y_fusion_prob, threshold=threshold, ece_bins=5)

        graph_model = train_graph_enhanced_model(train_df, label_col="risk_label", knn_k=2, num_layers=2)
        y_graph_prob = graph_model.predict_proba(test_df)
        graph_metrics = compute_probabilistic_metrics(y_true, y_graph_prob, threshold=threshold, ece_bins=5)

        fold_results.append(
            FoldResult(
                fold_id=fold_id,
                historical=hist_metrics,
                baseline_tabular=baseline_metrics,
                tabular=tabular_metrics,
                tabular_temporal=temporal_metrics,
                fusion=fusion_metrics,
                graph=graph_metrics,
            )
        )

    def pack_metric(name: str, accessor) -> dict[str, float]:
        vals = [float(accessor(fr)) for fr in fold_results]
        out = _mean_std(vals)
        out["metric"] = name
        return out

    summary = {
        "n_folds": len(fold_results),
        "historical": {
            "accuracy": pack_metric("accuracy", lambda fr: fr.historical.accuracy),
            "precision": pack_metric("precision", lambda fr: fr.historical.precision),
            "recall": pack_metric("recall", lambda fr: fr.historical.recall),
            "specificity": pack_metric("specificity", lambda fr: fr.historical.specificity),
            "f1": pack_metric("f1", lambda fr: fr.historical.f1),
            "auroc": pack_metric("auroc", lambda fr: fr.historical.auroc or 0.5),
            "auprc": pack_metric("auprc", lambda fr: fr.historical.auprc or 0.0),
            "ece": pack_metric("ece", lambda fr: fr.historical.ece or 0.0),
        },
        "baseline_tabular": {
            "accuracy": pack_metric("accuracy", lambda fr: fr.baseline_tabular.accuracy),
            "precision": pack_metric("precision", lambda fr: fr.baseline_tabular.precision),
            "recall": pack_metric("recall", lambda fr: fr.baseline_tabular.recall),
            "specificity": pack_metric("specificity", lambda fr: fr.baseline_tabular.specificity),
            "f1": pack_metric("f1", lambda fr: fr.baseline_tabular.f1),
            "auroc": pack_metric("auroc", lambda fr: fr.baseline_tabular.auroc or 0.5),
            "auprc": pack_metric("auprc", lambda fr: fr.baseline_tabular.auprc or 0.0),
            "ece": pack_metric("ece", lambda fr: fr.baseline_tabular.ece or 0.0),
        },
        "tabular": {
            "accuracy": pack_metric("accuracy", lambda fr: fr.tabular.accuracy),
            "precision": pack_metric("precision", lambda fr: fr.tabular.precision),
            "recall": pack_metric("recall", lambda fr: fr.tabular.recall),
            "specificity": pack_metric("specificity", lambda fr: fr.tabular.specificity),
            "f1": pack_metric("f1", lambda fr: fr.tabular.f1),
            "auroc": pack_metric("auroc", lambda fr: fr.tabular.auroc or 0.5),
            "auprc": pack_metric("auprc", lambda fr: fr.tabular.auprc or 0.0),
            "ece": pack_metric("ece", lambda fr: fr.tabular.ece or 0.0),
        },
        "tabular_temporal": {
            "accuracy": pack_metric("accuracy", lambda fr: fr.tabular_temporal.accuracy),
            "precision": pack_metric("precision", lambda fr: fr.tabular_temporal.precision),
            "recall": pack_metric("recall", lambda fr: fr.tabular_temporal.recall),
            "specificity": pack_metric("specificity", lambda fr: fr.tabular_temporal.specificity),
            "f1": pack_metric("f1", lambda fr: fr.tabular_temporal.f1),
            "auroc": pack_metric("auroc", lambda fr: fr.tabular_temporal.auroc or 0.5),
            "auprc": pack_metric("auprc", lambda fr: fr.tabular_temporal.auprc or 0.0),
            "ece": pack_metric("ece", lambda fr: fr.tabular_temporal.ece or 0.0),
        },
        "fusion": {
            "accuracy": pack_metric("accuracy", lambda fr: fr.fusion.accuracy),
            "precision": pack_metric("precision", lambda fr: fr.fusion.precision),
            "recall": pack_metric("recall", lambda fr: fr.fusion.recall),
            "specificity": pack_metric("specificity", lambda fr: fr.fusion.specificity),
            "f1": pack_metric("f1", lambda fr: fr.fusion.f1),
            "auroc": pack_metric("auroc", lambda fr: fr.fusion.auroc or 0.5),
            "auprc": pack_metric("auprc", lambda fr: fr.fusion.auprc or 0.0),
            "ece": pack_metric("ece", lambda fr: fr.fusion.ece or 0.0),
        },
        "graph": {
            "accuracy": pack_metric("accuracy", lambda fr: fr.graph.accuracy),
            "precision": pack_metric("precision", lambda fr: fr.graph.precision),
            "recall": pack_metric("recall", lambda fr: fr.graph.recall),
            "specificity": pack_metric("specificity", lambda fr: fr.graph.specificity),
            "f1": pack_metric("f1", lambda fr: fr.graph.f1),
            "auroc": pack_metric("auroc", lambda fr: fr.graph.auroc or 0.5),
            "auprc": pack_metric("auprc", lambda fr: fr.graph.auprc or 0.0),
            "ece": pack_metric("ece", lambda fr: fr.graph.ece or 0.0),
        },
        "improvement_graph_over_historical_accuracy": float(
            np.mean([fr.graph.accuracy - fr.historical.accuracy for fr in fold_results])
        ),
        "improvement_graph_over_fusion_accuracy": float(
            np.mean([fr.graph.accuracy - fr.fusion.accuracy for fr in fold_results])
        ),
        "ablation_order": [
            "historical",
            "baseline_tabular",
            "tabular",
            "tabular_temporal",
            "fusion",
            "graph",
        ],
        "ablation_accuracy_table": [
            {
                "model": "historical",
                "accuracy_mean": pack_metric("accuracy", lambda fr: fr.historical.accuracy)["mean"],
                "accuracy_std": pack_metric("accuracy", lambda fr: fr.historical.accuracy)["std"],
                "auroc_mean": pack_metric("auroc", lambda fr: fr.historical.auroc or 0.5)["mean"],
                "auprc_mean": pack_metric("auprc", lambda fr: fr.historical.auprc or 0.0)["mean"],
            },
            {
                "model": "baseline_tabular",
                "accuracy_mean": pack_metric("accuracy", lambda fr: fr.baseline_tabular.accuracy)["mean"],
                "accuracy_std": pack_metric("accuracy", lambda fr: fr.baseline_tabular.accuracy)["std"],
                "auroc_mean": pack_metric("auroc", lambda fr: fr.baseline_tabular.auroc or 0.5)["mean"],
                "auprc_mean": pack_metric("auprc", lambda fr: fr.baseline_tabular.auprc or 0.0)["mean"],
            },
            {
                "model": "tabular",
                "accuracy_mean": pack_metric("accuracy", lambda fr: fr.tabular.accuracy)["mean"],
                "accuracy_std": pack_metric("accuracy", lambda fr: fr.tabular.accuracy)["std"],
                "auroc_mean": pack_metric("auroc", lambda fr: fr.tabular.auroc or 0.5)["mean"],
                "auprc_mean": pack_metric("auprc", lambda fr: fr.tabular.auprc or 0.0)["mean"],
            },
            {
                "model": "tabular_temporal",
                "accuracy_mean": pack_metric("accuracy", lambda fr: fr.tabular_temporal.accuracy)["mean"],
                "accuracy_std": pack_metric("accuracy", lambda fr: fr.tabular_temporal.accuracy)["std"],
                "auroc_mean": pack_metric("auroc", lambda fr: fr.tabular_temporal.auroc or 0.5)["mean"],
                "auprc_mean": pack_metric("auprc", lambda fr: fr.tabular_temporal.auprc or 0.0)["mean"],
            },
            {
                "model": "fusion",
                "accuracy_mean": pack_metric("accuracy", lambda fr: fr.fusion.accuracy)["mean"],
                "accuracy_std": pack_metric("accuracy", lambda fr: fr.fusion.accuracy)["std"],
                "auroc_mean": pack_metric("auroc", lambda fr: fr.fusion.auroc or 0.5)["mean"],
                "auprc_mean": pack_metric("auprc", lambda fr: fr.fusion.auprc or 0.0)["mean"],
            },
            {
                "model": "graph",
                "accuracy_mean": pack_metric("accuracy", lambda fr: fr.graph.accuracy)["mean"],
                "accuracy_std": pack_metric("accuracy", lambda fr: fr.graph.accuracy)["std"],
                "auroc_mean": pack_metric("auroc", lambda fr: fr.graph.auroc or 0.5)["mean"],
                "auprc_mean": pack_metric("auprc", lambda fr: fr.graph.auprc or 0.0)["mean"],
            },
        ],
        "fold_results": [
            {
                "fold_id": fr.fold_id,
                "historical": fr.historical.__dict__,
                "baseline_tabular": fr.baseline_tabular.__dict__,
                "tabular": fr.tabular.__dict__,
                "tabular_temporal": fr.tabular_temporal.__dict__,
                "fusion": fr.fusion.__dict__,
                "graph": fr.graph.__dict__,
            }
            for fr in fold_results
        ],
    }
    return summary
