from roadpulse.severity import compute_risk_score, compute_severity


def test_severity_increases_with_area():
    small = compute_severity(area_ratio=0.005, confidence=0.8, hazard_class="pothole")
    large = compute_severity(area_ratio=0.03, confidence=0.8, hazard_class="pothole")
    assert large > small


def test_risk_with_context_is_bounded():
    risk = compute_risk_score(
        severity=80,
        traffic_index=0.9,
        weather_index=0.9,
        school_zone_index=1.0,
        equity_index=0.8,
    )
    assert 0 <= risk <= 100

