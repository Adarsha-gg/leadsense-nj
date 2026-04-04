import pandas as pd

from roadpulse.nj_context import apply_county_risk_adjustment, county_risk_multiplier


def test_county_multiplier_default():
    assert county_risk_multiplier("UnknownCounty") == 1.0


def test_county_adjustment_changes_risk():
    df = pd.DataFrame([{"risk_score": 50.0}, {"risk_score": 70.0}])
    out = apply_county_risk_adjustment(df, "Hudson")
    assert float(out["risk_score"].mean()) > float(df["risk_score"].mean())

