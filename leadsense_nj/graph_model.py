from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from leadsense_nj.multimodal import FusionRiskModel, build_fusion_feature_table, train_fusion_model


def _resolve_coordinates(df: pd.DataFrame) -> np.ndarray:
    if {"lat", "lon"}.issubset(df.columns):
        coords = df[["lat", "lon"]].copy()
        coords["lat"] = pd.to_numeric(coords["lat"], errors="coerce")
        coords["lon"] = pd.to_numeric(coords["lon"], errors="coerce")
        coords = coords.fillna(coords.median(numeric_only=True))
        return coords.to_numpy()

    # Fallback deterministic pseudo-coordinates from row index.
    idx = np.arange(len(df), dtype=float)
    return np.column_stack([idx, np.zeros_like(idx)])


def build_knn_adjacency(df: pd.DataFrame, k: int = 3) -> np.ndarray:
    n = len(df)
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.ones((1, 1), dtype=float)

    k_eff = max(1, min(k, n - 1))
    coords = _resolve_coordinates(df)
    knn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    knn.fit(coords)
    neigh_idx = knn.kneighbors(coords, n_neighbors=k_eff + 1, return_distance=False)

    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        adj[i, i] = 1.0
        for j in neigh_idx[i]:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    return adj


def graph_mean_aggregate(features: np.ndarray, adjacency: np.ndarray, num_layers: int = 2) -> np.ndarray:
    if features.shape[0] != adjacency.shape[0]:
        raise ValueError("features and adjacency must have same number of nodes")

    x = features.copy().astype(float)
    deg = adjacency.sum(axis=1, keepdims=True)
    deg[deg == 0.0] = 1.0
    for _ in range(max(1, num_layers)):
        x = adjacency @ x / deg
    return x


@dataclass(frozen=True)
class GraphEnhancedRiskModel:
    fusion_model: FusionRiskModel
    estimator: Pipeline
    feature_columns: tuple[str, ...]
    knn_k: int
    num_layers: int

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        fused = build_fusion_feature_table(df)
        x = fused.loc[:, self.fusion_model.feature_columns].to_numpy()
        adj = build_knn_adjacency(df, k=self.knn_k)
        agg = graph_mean_aggregate(x, adj, num_layers=self.num_layers)
        combined = np.concatenate([x, agg], axis=1)
        combined_df = pd.DataFrame(combined, columns=self.feature_columns, index=df.index)
        return self.estimator.predict_proba(combined_df)[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(df) >= threshold).astype(int)


def train_graph_enhanced_model(
    df: pd.DataFrame,
    label_col: str = "risk_label",
    knn_k: int = 3,
    num_layers: int = 2,
) -> GraphEnhancedRiskModel:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")
    y = pd.to_numeric(df[label_col], errors="raise").astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("Need at least two classes to train graph model.")

    fusion_model = train_fusion_model(df, label_col=label_col)
    fused = build_fusion_feature_table(df)
    x = fused.loc[:, fusion_model.feature_columns].to_numpy()

    adjacency = build_knn_adjacency(df, k=knn_k)
    x_agg = graph_mean_aggregate(x, adjacency, num_layers=num_layers)
    combined = np.concatenate([x, x_agg], axis=1)
    feature_columns = tuple([f"base_{i}" for i in range(x.shape[1])] + [f"graph_{i}" for i in range(x_agg.shape[1])])
    combined_df = pd.DataFrame(combined, columns=feature_columns, index=df.index)

    estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )
    estimator.fit(combined_df, y)
    return GraphEnhancedRiskModel(
        fusion_model=fusion_model,
        estimator=estimator,
        feature_columns=feature_columns,
        knn_k=knn_k,
        num_layers=num_layers,
    )
