from __future__ import annotations

import numpy as np
import pandas as pd

from leadsense_nj.infrastructure import (
    build_adjacency_from_edge_list,
    build_county_proxy_edge_list,
    validate_edge_list,
)


def test_build_adjacency_from_edge_list_symmetric() -> None:
    nodes = ["a", "b", "c", "d"]
    edges = pd.DataFrame({"source_geoid": ["a", "b"], "target_geoid": ["b", "c"]})
    adj = build_adjacency_from_edge_list(nodes, edges)
    assert adj.shape == (4, 4)
    assert np.allclose(adj, adj.T)
    assert np.all(np.diag(adj) == 1.0)
    assert adj[0, 1] == 1.0
    assert adj[1, 2] == 1.0


def test_build_county_proxy_edge_list_has_required_columns() -> None:
    df = pd.DataFrame(
        {
            "geoid": ["1", "2", "3", "4"],
            "county": ["A", "A", "B", "B"],
            "lat": [1.0, 1.1, 2.0, 2.1],
            "lon": [1.0, 1.1, 2.0, 2.1],
        }
    )
    edges = build_county_proxy_edge_list(df, k_within_county=1)
    validate_edge_list(edges)
    assert len(edges) >= 2
