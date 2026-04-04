from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ScenarioProfile:
    name: str
    traffic_index: float
    weather_index: float
    school_zone_index: float
    equity_index: float
    budget_usd: float
    top_k: int
    notes: str


SCENARIOS: dict[str, ScenarioProfile] = {
    "North Jersey Commuter Corridors": ScenarioProfile(
        name="North Jersey Commuter Corridors",
        traffic_index=0.85,
        weather_index=0.55,
        school_zone_index=0.40,
        equity_index=0.60,
        budget_usd=180000.0,
        top_k=14,
        notes="High traffic pressure and dense commuting patterns.",
    ),
    "Central Jersey School & Suburban Mix": ScenarioProfile(
        name="Central Jersey School & Suburban Mix",
        traffic_index=0.65,
        weather_index=0.50,
        school_zone_index=0.70,
        equity_index=0.50,
        budget_usd=120000.0,
        top_k=10,
        notes="School-zone exposure emphasized for safety prioritization.",
    ),
    "South Jersey Freight & Port Feeder": ScenarioProfile(
        name="South Jersey Freight & Port Feeder",
        traffic_index=0.80,
        weather_index=0.45,
        school_zone_index=0.30,
        equity_index=0.55,
        budget_usd=160000.0,
        top_k=12,
        notes="Heavy truck movement and freight bottlenecks prioritized.",
    ),
    "Shoreline Flood-Exposed Roads": ScenarioProfile(
        name="Shoreline Flood-Exposed Roads",
        traffic_index=0.55,
        weather_index=0.88,
        school_zone_index=0.35,
        equity_index=0.58,
        budget_usd=140000.0,
        top_k=11,
        notes="Standing water and weather vulnerability weighted heavily.",
    ),
}


COUNTIES = [
    "Atlantic",
    "Bergen",
    "Burlington",
    "Camden",
    "Cape May",
    "Cumberland",
    "Essex",
    "Gloucester",
    "Hudson",
    "Hunterdon",
    "Mercer",
    "Middlesex",
    "Monmouth",
    "Morris",
    "Ocean",
    "Passaic",
    "Salem",
    "Somerset",
    "Sussex",
    "Union",
    "Warren",
]


COUNTY_MULTIPLIER = {
    "Bergen": 1.10,
    "Essex": 1.14,
    "Hudson": 1.17,
    "Middlesex": 1.12,
    "Monmouth": 1.08,
    "Ocean": 1.07,
    "Passaic": 1.11,
    "Camden": 1.09,
}


def get_scenario(name: str) -> ScenarioProfile:
    return SCENARIOS.get(name, next(iter(SCENARIOS.values())))


def county_risk_multiplier(county: str) -> float:
    return float(COUNTY_MULTIPLIER.get(county, 1.0))


def apply_county_risk_adjustment(detections_df: pd.DataFrame, county: str) -> pd.DataFrame:
    if detections_df.empty:
        return detections_df
    mult = county_risk_multiplier(county)
    out = detections_df.copy()
    out["risk_score"] = (out["risk_score"] * mult).clip(upper=100.0).round(2)
    return out

