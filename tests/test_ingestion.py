from __future__ import annotations

import pandas as pd
import pytest

from leadsense_nj.ingestion import (
    build_epa_pws_lead_signals,
    fetch_census_acs_block_groups,
    fetch_epa_efservice_table,
    validate_acs_block_group_frame,
)


class _FakeResponse:
    def __init__(self, *, json_payload=None, text_payload: str = "") -> None:
        self._json_payload = json_payload
        self.text = text_payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._json_payload


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.last_url = ""
        self.last_params = None

    def get(self, url, params=None, timeout=None):  # noqa: ANN001
        self.last_url = str(url)
        self.last_params = params
        return self._response


def test_fetch_census_acs_block_groups_parses_api_payload() -> None:
    payload = [
        [
            "B19013_001E",
            "B25034_001E",
            "B25034_010E",
            "B25034_011E",
            "B25035_001E",
            "B17001_001E",
            "B17001_002E",
            "B01001_001E",
            "B01001_003E",
            "B01001_027E",
            "B02001_001E",
            "B02001_002E",
            "state",
            "county",
            "tract",
            "block group",
        ],
        ["55000", "100", "20", "30", "1955", "200", "40", "300", "10", "8", "300", "180", "34", "001", "000100", "1"],
        ["62000", "120", "12", "18", "1968", "240", "30", "350", "12", "9", "350", "240", "34", "003", "000200", "2"],
    ]
    session = _FakeSession(_FakeResponse(json_payload=payload))
    out = fetch_census_acs_block_groups(year=2022, session=session)

    assert len(out) == 2
    assert out["geoid"].tolist() == ["340010001001", "340030002002"]
    assert out["pct_housing_pre_1950"].between(0.0, 1.0).all()
    assert out["poverty_rate"].between(0.0, 1.0).all()


def test_validate_acs_block_group_frame_rejects_bad_geoid() -> None:
    df = pd.DataFrame(
        {
            "geoid": ["bad_geoid"],
            "median_income": [50000],
            "pct_housing_pre_1950": [0.3],
            "poverty_rate": [0.1],
            "children_under_6_rate": [0.08],
            "minority_share": [0.25],
            "median_housing_year": [1960],
        }
    )
    with pytest.raises(ValueError, match="invalid GEOID"):
        validate_acs_block_group_frame(df)


def test_fetch_epa_efservice_table_parses_csv() -> None:
    csv_payload = "pwsid,sample_id,sample_measure,unit_of_measure\nNJ1,S1,0.012,mg/L\n"
    session = _FakeSession(_FakeResponse(text_payload=csv_payload))
    out = fetch_epa_efservice_table("LCR_SAMPLE_RESULT", session=session)
    assert len(out) == 1
    assert out.loc[0, "pwsid"] == "NJ1"


def test_build_epa_pws_lead_signals_computes_flags() -> None:
    samples = pd.DataFrame(
        {
            "pwsid": ["NJ1", "NJ1", "NJ2"],
            "sample_id": ["S1", "S2", "S3"],
            "sampling_end_date": ["2024-01-01", "2021-01-01", "2024-02-01"],
        }
    )
    sample_results = pd.DataFrame(
        {
            "pwsid": ["NJ1", "NJ1", "NJ2"],
            "sample_id": ["S1", "S2", "S3"],
            "contaminant_code": ["PB90", "PB90", "PB90"],
            "sample_measure": [0.020, 0.010, 0.005],  # mg/L
            "unit_of_measure": ["mg/L", "mg/L", "mg/L"],
        }
    )
    violations = pd.DataFrame(
        {
            "pwsid": ["NJ1", "NJ2"],
            "contaminant_code": [1030, 1030],
            "compl_per_end_date": ["2024-06-01", "2018-01-01"],
        }
    )

    out = build_epa_pws_lead_signals(
        samples,
        sample_results,
        violations=violations,
        reference_date=pd.Timestamp("2025-01-01"),
    )

    row_nj1 = out.loc[out["pwsid"] == "NJ1"].iloc[0]
    row_nj2 = out.loc[out["pwsid"] == "NJ2"].iloc[0]
    assert bool(row_nj1["pws_action_level_exceedance_5y"])
    assert bool(row_nj1["pws_any_sample_gt15_3y"])
    assert not bool(row_nj2["pws_action_level_exceedance_5y"])
