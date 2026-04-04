import pandas as pd

from roadpulse.analytics import class_risk_heatmap, confidence_buckets, risk_band_breakdown


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"hazard_class": "pothole", "risk_score": 25, "confidence": 0.42},
            {"hazard_class": "pothole", "risk_score": 65, "confidence": 0.77},
            {"hazard_class": "standing_water", "risk_score": 85, "confidence": 0.88},
        ]
    )


def test_risk_band_breakdown_not_empty():
    out = risk_band_breakdown(_sample_df())
    assert not out.empty


def test_confidence_buckets_not_empty():
    out = confidence_buckets(_sample_df())
    assert not out.empty


def test_class_risk_heatmap_not_empty():
    out = class_risk_heatmap(_sample_df())
    assert not out.empty

