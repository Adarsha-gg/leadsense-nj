from __future__ import annotations

import pandas as pd

from leadsense_nj.research_data import build_nj_research_feature_table


def test_build_nj_research_feature_table_outputs_required_columns() -> None:
    acs = pd.DataFrame(
        {
            "geoid": ["340010001001", "340030010002", "340050023001", "340070045003"],
            "median_income": [50000, 82000, 73000, 42000],
            "pct_housing_pre_1950": [0.30, 0.15, 0.22, 0.41],
            "poverty_rate": [0.12, 0.08, 0.10, 0.21],
            "children_under_6_rate": [0.07, 0.05, 0.06, 0.09],
            "minority_share": [0.40, 0.25, 0.34, 0.55],
            "median_housing_year": [1958, 1972, 1966, 1948],
        }
    )
    out = build_nj_research_feature_table(acs, seed=7)
    assert len(out) == 4
    assert "risk_label" in out.columns
    assert {"lat", "lon", "q8_lead_ppb", "pws_action_level_exceedance_5y"}.issubset(out.columns)
