from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold

from leadsense_nj.graph_model import train_graph_enhanced_model
from leadsense_nj.metrics import BinaryClassificationMetrics, compute_binary_metrics, historical_signal_prediction
from leadsense_nj.multimodal import build_fusion_feature_table, train_fusion_model
from leadsense_nj.target import with_elevated_risk_label


@dataclass(frozen=True)
class FoldResult:
    fold_id: int
    historical: BinaryClassificationMetrics
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

        y_hist = historical_signal_prediction(test_df)
        hist_metrics = compute_binary_metrics(y_true, y_hist)

        fusion_model = train_fusion_model(train_df, label_col="risk_label")
        fused_test = build_fusion_feature_table(test_df)
        y_fusion = (fusion_model.predict_proba(fused_test) >= threshold).astype(int)
        fusion_metrics = compute_binary_metrics(y_true, y_fusion)

        graph_model = train_graph_enhanced_model(train_df, label_col="risk_label", knn_k=2, num_layers=2)
        y_graph = graph_model.predict(test_df, threshold=threshold)
        graph_metrics = compute_binary_metrics(y_true, y_graph)

        fold_results.append(
            FoldResult(
                fold_id=fold_id,
                historical=hist_metrics,
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
            "f1": pack_metric("f1", lambda fr: fr.historical.f1),
        },
        "fusion": {
            "accuracy": pack_metric("accuracy", lambda fr: fr.fusion.accuracy),
            "precision": pack_metric("precision", lambda fr: fr.fusion.precision),
            "recall": pack_metric("recall", lambda fr: fr.fusion.recall),
            "f1": pack_metric("f1", lambda fr: fr.fusion.f1),
        },
        "graph": {
            "accuracy": pack_metric("accuracy", lambda fr: fr.graph.accuracy),
            "precision": pack_metric("precision", lambda fr: fr.graph.precision),
            "recall": pack_metric("recall", lambda fr: fr.graph.recall),
            "f1": pack_metric("f1", lambda fr: fr.graph.f1),
        },
        "improvement_graph_over_historical_accuracy": float(
            np.mean([fr.graph.accuracy - fr.historical.accuracy for fr in fold_results])
        ),
        "improvement_graph_over_fusion_accuracy": float(
            np.mean([fr.graph.accuracy - fr.fusion.accuracy for fr in fold_results])
        ),
        "fold_results": [
            {
                "fold_id": fr.fold_id,
                "historical": fr.historical.__dict__,
                "fusion": fr.fusion.__dict__,
                "graph": fr.graph.__dict__,
            }
            for fr in fold_results
        ],
    }
    return summary
