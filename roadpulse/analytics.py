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


def risk_band_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["risk_band", "count"])
    tmp = df.copy()
    tmp["risk_band"] = pd.cut(
        tmp["risk_score"],
        bins=[-0.1, 30, 60, 80, 100.0],
        labels=["Low", "Moderate", "High", "Critical"],
    )
    out = tmp.groupby("risk_band", as_index=False, observed=False).size()
    return out.rename(columns={"size": "count"})


def confidence_buckets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["confidence_bucket", "count"])
    tmp = df.copy()
    tmp["confidence_bucket"] = pd.cut(
        tmp["confidence"],
        bins=[0.0, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        include_lowest=True,
    )
    out = tmp.groupby("confidence_bucket", as_index=False, observed=False).size()
    return out.rename(columns={"size": "count"})


def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["segment_id", "issue_count", "avg_risk", "max_severity"])
    tmp = df.copy()
    if "segment_id" not in tmp.columns:
        tmp["segment_id"] = (tmp["timestamp_s"] // 20).astype(int).map(lambda x: f"seg_time_{x:03d}")
    out = (
        tmp.groupby("segment_id", as_index=False)
        .agg(issue_count=("hazard_class", "count"), avg_risk=("risk_score", "mean"), max_severity=("severity", "max"))
        .sort_values(["avg_risk", "max_severity"], ascending=False)
    )
    return out.round(2)


def class_risk_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["hazard_class", "risk_band", "count"])
    tmp = df.copy()
    tmp["risk_band"] = pd.cut(
        tmp["risk_score"],
        bins=[-0.1, 30, 60, 80, 100.0],
        labels=["Low", "Moderate", "High", "Critical"],
    )
    out = tmp.groupby(["hazard_class", "risk_band"], as_index=False, observed=False).size()
    return out.rename(columns={"size": "count"})
