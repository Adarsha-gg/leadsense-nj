from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


EDGE_SOURCE_COL = "source_geoid"
EDGE_TARGET_COL = "target_geoid"


def validate_edge_list(
    edges_df: pd.DataFrame,
    *,
    source_col: str = EDGE_SOURCE_COL,
    target_col: str = EDGE_TARGET_COL,
) -> None:
    missing = sorted({source_col, target_col}.difference(edges_df.columns))
    if missing:
        raise ValueError(f"Edge list missing required columns: {missing}")
    if edges_df.empty:
        raise ValueError("Edge list is empty.")

    source = edges_df[source_col].astype(str)
    target = edges_df[target_col].astype(str)
    if source.str.len().eq(0).any() or target.str.len().eq(0).any():
        raise ValueError("Edge list contains blank node ids.")


def load_edge_list(
    path: str | Path,
    *,
    source_col: str = EDGE_SOURCE_COL,
    target_col: str = EDGE_TARGET_COL,
) -> pd.DataFrame:
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Edge list file not found: {in_path}")
    df = pd.read_csv(in_path, dtype={source_col: str, target_col: str})
    validate_edge_list(df, source_col=source_col, target_col=target_col)
    return df


def build_adjacency_from_edge_list(
    node_ids: pd.Series | list[str] | np.ndarray,
    edges_df: pd.DataFrame,
    *,
    source_col: str = EDGE_SOURCE_COL,
    target_col: str = EDGE_TARGET_COL,
    add_self_loops: bool = True,
) -> np.ndarray:
    nodes = pd.Series(node_ids, dtype=str)
    n = len(nodes)
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if edges_df.empty:
        adj = np.zeros((n, n), dtype=float)
        if add_self_loops:
            np.fill_diagonal(adj, 1.0)
        return adj

    validate_edge_list(edges_df, source_col=source_col, target_col=target_col)
    idx = {node: i for i, node in enumerate(nodes.tolist())}
    adj = np.zeros((n, n), dtype=float)

    for _, row in edges_df.iterrows():
        a = str(row[source_col])
        b = str(row[target_col])
        if a not in idx or b not in idx:
            continue
        i = idx[a]
        j = idx[b]
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    if add_self_loops:
        np.fill_diagonal(adj, 1.0)
    return adj


def build_county_proxy_edge_list(
    df: pd.DataFrame,
    *,
    geoid_col: str = "geoid",
    county_col: str = "county",
    k_within_county: int = 2,
) -> pd.DataFrame:
    required = {geoid_col}
    if not required.issubset(df.columns):
        missing = sorted(required.difference(df.columns))
        raise ValueError(f"Missing required columns for proxy edge build: {missing}")

    work = df.copy()
    work[geoid_col] = work[geoid_col].astype(str)
    if county_col not in work.columns:
        work[county_col] = "ALL"

    if {"lat", "lon"}.issubset(work.columns):
        coords = work[["lat", "lon"]].copy()
        coords["lat"] = pd.to_numeric(coords["lat"], errors="coerce")
        coords["lon"] = pd.to_numeric(coords["lon"], errors="coerce")
        coords = coords.fillna(coords.median(numeric_only=True))
    else:
        idx = np.arange(len(work), dtype=float)
        coords = pd.DataFrame({"lat": idx, "lon": np.zeros_like(idx)})

    work = work.reset_index(drop=True)
    edges: set[tuple[str, str]] = set()

    for county, idxs in work.groupby(county_col).groups.items():
        county_indices = np.array(list(idxs), dtype=int)
        if len(county_indices) <= 1:
            continue
        k_eff = max(1, min(k_within_county, len(county_indices) - 1))
        county_coords = coords.iloc[county_indices].to_numpy()
        knn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
        knn.fit(county_coords)
        neigh = knn.kneighbors(county_coords, n_neighbors=k_eff + 1, return_distance=False)
        for local_i, nbrs in enumerate(neigh):
            src_global_i = county_indices[local_i]
            src = str(work.at[src_global_i, geoid_col])
            for local_j in nbrs:
                dst_global_i = county_indices[int(local_j)]
                dst = str(work.at[dst_global_i, geoid_col])
                if src == dst:
                    continue
                a, b = sorted((src, dst))
                edges.add((a, b))

    # If counties are too fragmented for the sample, add a sparse global KNN fallback.
    if not edges and len(work) > 1:
        k_eff = max(1, min(2, len(work) - 1))
        global_knn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
        global_knn.fit(coords.to_numpy())
        neigh = global_knn.kneighbors(coords.to_numpy(), n_neighbors=k_eff + 1, return_distance=False)
        for i, nbrs in enumerate(neigh):
            src = str(work.at[i, geoid_col])
            for j in nbrs:
                dst = str(work.at[int(j), geoid_col])
                if src == dst:
                    continue
                a, b = sorted((src, dst))
                edges.add((a, b))

    out = pd.DataFrame(sorted(edges), columns=[EDGE_SOURCE_COL, EDGE_TARGET_COL])
    if out.empty:
        if len(work) > 1:
            out = pd.DataFrame(
                [{EDGE_SOURCE_COL: str(work.at[0, geoid_col]), EDGE_TARGET_COL: str(work.at[1, geoid_col])}]
            )
        else:
            out = pd.DataFrame(columns=[EDGE_SOURCE_COL, EDGE_TARGET_COL])
    return out
