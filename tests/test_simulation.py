import pandas as pd

from roadpulse.simulation import simulate_repair_impact


def test_simulation_reduces_risk_for_selected_segments():
    detections = pd.DataFrame(
        [
            {"segment_id": "s1", "risk_score": 80, "timestamp_s": 1},
            {"segment_id": "s1", "risk_score": 70, "timestamp_s": 2},
            {"segment_id": "s2", "risk_score": 60, "timestamp_s": 25},
        ]
    )
    priority = pd.DataFrame([{"segment_id": "s1"}])
    out = simulate_repair_impact(detections, priority, repaired_segments=1)
    assert out["post_repair_avg_risk"] < out["baseline_avg_risk"]

