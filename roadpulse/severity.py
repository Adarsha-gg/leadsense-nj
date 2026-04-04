from __future__ import annotations


BASE_HAZARD_MULTIPLIER = {
    "pothole": 1.0,
    "alligator_crack": 0.95,
    "longitudinal_crack": 0.8,
    "transverse_crack": 0.85,
    "standing_water": 1.1,
    "faded_lane_marking": 0.7,
    "road_debris": 1.05,
}


def compute_severity(area_ratio: float, confidence: float, hazard_class: str) -> float:
    """Return severity from 0-100."""
    multiplier = BASE_HAZARD_MULTIPLIER.get(hazard_class, 0.8)
    geom = min(1.0, area_ratio * 12.0)
    score = (0.55 * geom + 0.45 * confidence) * multiplier
    return round(max(0.0, min(100.0, score * 100.0)), 2)


def compute_risk_score(
    severity: float,
    traffic_index: float = 0.5,
    weather_index: float = 0.5,
    school_zone_index: float = 0.0,
    equity_index: float = 0.5,
    severity_weight: float = 0.2,
    traffic_weight: float = 0.3,
    weather_weight: float = 0.2,
    school_zone_weight: float = 0.15,
    equity_weight: float = 0.15,
) -> float:
    severity_norm = max(0.0, min(1.0, severity / 100.0))
    score = (
        severity_weight * severity_norm
        + traffic_weight * max(0.0, min(1.0, traffic_index))
        + weather_weight * max(0.0, min(1.0, weather_index))
        + school_zone_weight * max(0.0, min(1.0, school_zone_index))
        + equity_weight * max(0.0, min(1.0, equity_index))
    )
    return round(max(0.0, min(100.0, score * 100.0)), 2)

