from __future__ import annotations

import pandas as pd

from leadsense_nj.ai_patterns import build_ai_patterns


def test_build_ai_patterns_returns_expected_sections() -> None:
    rows = []
    for i in range(60):
        rows.append(
            {
                "geoid": f"3400{i:06d}",
                "county": "Essex" if i % 2 == 0 else "Camden",
                "municipality": f"Area {i}",
                "risk_score": 0.2 + (i % 12) * 0.05,
                "risk_uncertainty": 0.01 + (i % 7) * 0.01,
                "replacement_cost": 12000 + i * 300,
                "poverty_rate": 0.05 + (i % 10) * 0.03,
                "minority_share": 0.15 + (i % 8) * 0.07,
                "pct_housing_pre_1950": 0.2 + (i % 9) * 0.05,
                "lead_90p_ppb": 4 + (i % 11) * 1.8,
                "lat": 39.1 + (i % 10) * 0.12,
                "lon": -75.2 + (i % 10) * 0.11,
                "lead_trend": [4.0 + 0.2 * j + (i % 3) * 0.1 for j in range(8)],
            }
        )
    result = build_ai_patterns(pd.DataFrame(rows), max_clusters=7, max_outliers=12, max_watchlist=10)

    assert "summary" in result
    assert result["summary"]["cluster_count"] >= 2
    assert len(result["hotspot_clusters"]) > 0
    assert len(result["outliers"]) > 0
    assert len(result["rising_watchlist"]) > 0
