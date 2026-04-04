import pandas as pd

from roadpulse.prioritization import PrioritizationConfig, build_priority_queue


def test_priority_queue_respects_budget():
    df = pd.DataFrame(
        [
            {"hazard_class": "pothole", "risk_score": 90, "severity": 80, "timestamp_s": 2, "lat": 40.1, "lon": -74.5},
            {"hazard_class": "pothole", "risk_score": 70, "severity": 60, "timestamp_s": 5, "lat": 40.1, "lon": -74.5},
            {"hazard_class": "alligator_crack", "risk_score": 85, "severity": 75, "timestamp_s": 22, "lat": 40.2, "lon": -74.6},
        ]
    )
    out = build_priority_queue(df, PrioritizationConfig(top_k=10, budget_usd=5000, equity_boost_enabled=True))
    assert not out.empty
    assert out["estimated_repair_cost_usd"].sum() <= 5000
