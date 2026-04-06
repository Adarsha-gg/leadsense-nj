from __future__ import annotations

import numpy as np
import pandas as pd

from leadsense_nj.graph_model import (
    build_infrastructure_adjacency,
    build_knn_adjacency,
    graph_mean_aggregate,
    train_graph_enhanced_model,
)
from leadsense_nj.infrastructure import build_county_proxy_edge_list
from leadsense_nj.target import with_elevated_risk_label


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "geoid": ["1", "2", "3", "4", "5", "6"],
            "lat": [39.1, 39.3, 39.5, 40.1, 40.3, 40.5],
            "lon": [-74.1, -74.3, -74.5, -75.1, -75.3, -75.5],
            "median_income": [42000, 81500, 97000, 38500, 71000, 45000],
            "pct_housing_pre_1950": [0.48, 0.23, 0.17, 0.53, 0.29, 0.41],
            "poverty_rate": [0.27, 0.11, 0.07, 0.31, 0.14, 0.24],
            "children_under_6_rate": [0.09, 0.06, 0.05, 0.11, 0.07, 0.10],
            "lead_90p_ppb": [11.2, 5.1, 3.4, 13.8, 6.0, 9.7],
            "ph_mean": [6.7, 7.3, 7.6, 6.5, 7.1, 6.8],
            "hardness_mg_l": [104, 126, 140, 98, 132, 112],
            "chlorine_residual_mg_l": [1.2, 1.8, 2.1, 1.0, 1.7, 1.3],
            "distance_to_tri_km": [2.4, 5.2, 9.4, 1.6, 7.8, 3.9],
            "winter_freeze_thaw_days": [42, 37, 33, 44, 36, 41],
            "alkalinity_mg_l": [28, 56, 62, 24, 48, 27],
            "pws_action_level_exceedance_5y": [0, 0, 0, 1, 0, 0],
            "pws_any_sample_gt15_3y": [1, 0, 0, 1, 0, 0],
            "median_housing_year": [1946, 1968, 1985, 1938, 1958, 1942],
            "q1_lead_ppb": [12.8, 5.6, 3.9, 14.7, 6.4, 10.1],
            "q2_lead_ppb": [11.9, 5.1, 3.5, 13.9, 6.1, 9.6],
            "q3_lead_ppb": [10.7, 4.8, 3.4, 13.1, 5.8, 9.2],
            "q4_lead_ppb": [9.8, 4.3, 3.2, 12.8, 5.6, 8.7],
            "q5_lead_ppb": [11.3, 5.0, 3.1, 14.2, 5.9, 9.5],
            "q6_lead_ppb": [12.1, 4.6, 3.6, 13.6, 5.7, 9.1],
            "q7_lead_ppb": [10.9, 4.9, 3.3, 13.0, 6.0, 8.9],
            "q8_lead_ppb": [11.5, 5.2, 3.4, 13.4, 5.8, 9.3],
        }
    )


def test_build_knn_adjacency_square() -> None:
    adj = build_knn_adjacency(_sample_df(), k=2)
    assert adj.shape == (6, 6)
    assert np.allclose(adj, adj.T)
    assert np.all(np.diag(adj) == 1.0)


def test_graph_mean_aggregate_shape() -> None:
    x = np.random.RandomState(0).rand(6, 5)
    adj = np.eye(6)
    out = graph_mean_aggregate(x, adj, num_layers=2)
    assert out.shape == x.shape


def test_train_graph_model_predicts() -> None:
    df = with_elevated_risk_label(_sample_df())
    model = train_graph_enhanced_model(df, knn_k=2, num_layers=2)
    proba = model.predict_proba(df)
    assert len(proba) == len(df)
    assert float(proba.min()) >= 0.0
    assert float(proba.max()) <= 1.0


def test_train_graph_model_infrastructure_mode_predicts() -> None:
    df = with_elevated_risk_label(_sample_df())
    edges = build_county_proxy_edge_list(df)
    adj = build_infrastructure_adjacency(df, edges_df=edges)
    assert adj.shape == (len(df), len(df))
    model = train_graph_enhanced_model(
        df,
        knn_k=2,
        num_layers=2,
        graph_mode="infrastructure",
        infrastructure_edges=edges,
    )
    proba = model.predict_proba(df)
    assert len(proba) == len(df)
    assert float(proba.min()) >= 0.0
    assert float(proba.max()) <= 1.0
