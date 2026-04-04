from __future__ import annotations

import pandas as pd


def summarize_detections(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "detection_count": 0,
            "avg_severity": 0.0,
            "avg_risk": 0.0,
            "high_risk_count": 0,
        }
    return {
        "detection_count": float(len(df)),
        "avg_severity": float(df["severity"].mean()),
        "avg_risk": float(df["risk_score"].mean()),
        "high_risk_count": float((df["risk_score"] >= 70).sum()),
    }


def hazard_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["hazard_class", "count"])
    out = df.groupby("hazard_class", as_index=False).size()
    return out.rename(columns={"size": "count"}).sort_values("count", ascending=False)


def risk_timeline(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["bucket_min", "avg_risk"])
    tmp = df.copy()
    tmp["bucket_min"] = (tmp["timestamp_s"] // 60).astype(int)
    out = tmp.groupby("bucket_min", as_index=False)["risk_score"].mean()
    return out.rename(columns={"risk_score": "avg_risk"})

