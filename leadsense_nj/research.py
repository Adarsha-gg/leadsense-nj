from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold

from leadsense_nj.baseline import fit_tabular_logistic
from leadsense_nj.graph_model import train_graph_enhanced_model
from leadsense_nj.infrastructure import build_county_proxy_edge_list
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
    graph_knn: BinaryClassificationMetrics
    graph_infrastructure: BinaryClassificationMetrics


def _subsample_for_benchmark(
    df: pd.DataFrame,
    *,
    label_col: str,
    max_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)
    if max_rows < 10:
        raise ValueError("max_rows must be >= 10 when provided.")

    work = df.copy()
    y = pd.to_numeric(work[label_col], errors="raise").astype(int)
    rng = np.random.default_rng(random_state)
    selected_idx: list[int] = []
    for cls in sorted(y.unique().tolist()):
        cls_idx = np.where(y.to_numpy() == cls)[0]
        n_take = max(1, int(round(max_rows * (len(cls_idx) / len(work)))))
        n_take = min(n_take, len(cls_idx))
        take = rng.choice(cls_idx, size=n_take, replace=False)
        selected_idx.extend(take.tolist())
    if len(selected_idx) < max_rows:
        remaining = sorted(set(range(len(work))).difference(selected_idx))
        add_n = min(max_rows - len(selected_idx), len(remaining))
        selected_idx.extend(rng.choice(np.array(remaining), size=add_n, replace=False).tolist())
    sampled = work.iloc[sorted(set(selected_idx))].copy()
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=random_state)
    return sampled.reset_index(drop=True)


def _validate_split_integrity(
    df: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    id_col: str = "geoid",
) -> dict[str, int]:
    if id_col not in df.columns:
        return {"fold_overlap_count": 0, "rows_covered": 0}

    overlap_count = 0
    covered = set()
    ids = df[id_col].astype(str).to_numpy()
    for train_idx, test_idx in splits:
        train_ids = set(ids[train_idx].tolist())
        test_ids = set(ids[test_idx].tolist())
        overlap_count += len(train_ids.intersection(test_ids))
        covered.update(test_ids)
    return {"fold_overlap_count": int(overlap_count), "rows_covered": int(len(covered))}


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
    max_rows: int | None = None,
) -> dict:
    labeled = with_elevated_risk_label(df) if "risk_label" not in df.columns else df.copy()
    labeled = _subsample_for_benchmark(
        labeled,
        label_col="risk_label",
        max_rows=max_rows,
        random_state=random_state,
    )
    splits = spatial_kfold_splits(labeled, n_splits=n_splits, random_state=random_state)
    if not splits:
        raise RuntimeError("Unable to build valid CV splits.")
    split_integrity = _validate_split_integrity(labeled, splits, id_col="geoid")

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

        graph_knn_model = train_graph_enhanced_model(
            train_df,
            label_col="risk_label",
            knn_k=2,
            num_layers=2,
            graph_mode="knn",
        )
        y_graph_knn_prob = graph_knn_model.predict_proba(test_df)
        graph_knn_metrics = compute_probabilistic_metrics(y_true, y_graph_knn_prob, threshold=threshold, ece_bins=5)

        infra_edges = build_county_proxy_edge_list(train_df)
        graph_infra_model = train_graph_enhanced_model(
            train_df,
            label_col="risk_label",
            knn_k=2,
            num_layers=2,
            graph_mode="infrastructure",
            infrastructure_edges=infra_edges,
        )
        y_graph_infra_prob = graph_infra_model.predict_proba(test_df)
        graph_infra_metrics = compute_probabilistic_metrics(y_true, y_graph_infra_prob, threshold=threshold, ece_bins=5)

        fold_results.append(
            FoldResult(
                fold_id=fold_id,
                historical=hist_metrics,
                baseline_tabular=baseline_metrics,
                tabular=tabular_metrics,
                tabular_temporal=temporal_metrics,
                fusion=fusion_metrics,
                graph_knn=graph_knn_metrics,
                graph_infrastructure=graph_infra_metrics,
            )
        )

    def pack_metric(name: str, accessor) -> dict[str, float]:
        vals = [float(accessor(fr)) for fr in fold_results]
        out = _mean_std(vals)
        out["metric"] = name
        return out

    summary = {
        "n_folds": len(fold_results),
        "n_rows": int(len(labeled)),
        "max_rows": int(max_rows) if max_rows is not None else None,
        "split_integrity": split_integrity,
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
        "graph_knn": {
            "accuracy": pack_metric("accuracy", lambda fr: fr.graph_knn.accuracy),
            "precision": pack_metric("precision", lambda fr: fr.graph_knn.precision),
            "recall": pack_metric("recall", lambda fr: fr.graph_knn.recall),
            "specificity": pack_metric("specificity", lambda fr: fr.graph_knn.specificity),
            "f1": pack_metric("f1", lambda fr: fr.graph_knn.f1),
            "auroc": pack_metric("auroc", lambda fr: fr.graph_knn.auroc or 0.5),
            "auprc": pack_metric("auprc", lambda fr: fr.graph_knn.auprc or 0.0),
            "ece": pack_metric("ece", lambda fr: fr.graph_knn.ece or 0.0),
        },
        "graph_infrastructure": {
            "accuracy": pack_metric("accuracy", lambda fr: fr.graph_infrastructure.accuracy),
            "precision": pack_metric("precision", lambda fr: fr.graph_infrastructure.precision),
            "recall": pack_metric("recall", lambda fr: fr.graph_infrastructure.recall),
            "specificity": pack_metric("specificity", lambda fr: fr.graph_infrastructure.specificity),
            "f1": pack_metric("f1", lambda fr: fr.graph_infrastructure.f1),
            "auroc": pack_metric("auroc", lambda fr: fr.graph_infrastructure.auroc or 0.5),
            "auprc": pack_metric("auprc", lambda fr: fr.graph_infrastructure.auprc or 0.0),
            "ece": pack_metric("ece", lambda fr: fr.graph_infrastructure.ece or 0.0),
        },
        # Keep stable alias used by existing checks/UI.
        "graph": {
            "accuracy": pack_metric("accuracy", lambda fr: fr.graph_infrastructure.accuracy),
            "precision": pack_metric("precision", lambda fr: fr.graph_infrastructure.precision),
            "recall": pack_metric("recall", lambda fr: fr.graph_infrastructure.recall),
            "specificity": pack_metric("specificity", lambda fr: fr.graph_infrastructure.specificity),
            "f1": pack_metric("f1", lambda fr: fr.graph_infrastructure.f1),
            "auroc": pack_metric("auroc", lambda fr: fr.graph_infrastructure.auroc or 0.5),
            "auprc": pack_metric("auprc", lambda fr: fr.graph_infrastructure.auprc or 0.0),
            "ece": pack_metric("ece", lambda fr: fr.graph_infrastructure.ece or 0.0),
        },
        "improvement_graph_over_historical_accuracy": float(
            np.mean([fr.graph_infrastructure.accuracy - fr.historical.accuracy for fr in fold_results])
        ),
        "improvement_graph_over_fusion_accuracy": float(
            np.mean([fr.graph_infrastructure.accuracy - fr.fusion.accuracy for fr in fold_results])
        ),
        "improvement_graph_infrastructure_over_graph_knn_accuracy": float(
            np.mean([fr.graph_infrastructure.accuracy - fr.graph_knn.accuracy for fr in fold_results])
        ),
        "ablation_order": [
            "historical",
            "baseline_tabular",
            "tabular",
            "tabular_temporal",
            "fusion",
            "graph_knn",
            "graph_infrastructure",
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
                "model": "graph_knn",
                "accuracy_mean": pack_metric("accuracy", lambda fr: fr.graph_knn.accuracy)["mean"],
                "accuracy_std": pack_metric("accuracy", lambda fr: fr.graph_knn.accuracy)["std"],
                "auroc_mean": pack_metric("auroc", lambda fr: fr.graph_knn.auroc or 0.5)["mean"],
                "auprc_mean": pack_metric("auprc", lambda fr: fr.graph_knn.auprc or 0.0)["mean"],
            },
            {
                "model": "graph_infrastructure",
                "accuracy_mean": pack_metric("accuracy", lambda fr: fr.graph_infrastructure.accuracy)["mean"],
                "accuracy_std": pack_metric("accuracy", lambda fr: fr.graph_infrastructure.accuracy)["std"],
                "auroc_mean": pack_metric("auroc", lambda fr: fr.graph_infrastructure.auroc or 0.5)["mean"],
                "auprc_mean": pack_metric("auprc", lambda fr: fr.graph_infrastructure.auprc or 0.0)["mean"],
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
                "graph_knn": fr.graph_knn.__dict__,
                "graph_infrastructure": fr.graph_infrastructure.__dict__,
                "graph": fr.graph_infrastructure.__dict__,
            }
            for fr in fold_results
        ],
    }
    return summary
