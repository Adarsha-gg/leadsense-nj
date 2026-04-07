from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def _to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if out.isna().all():
        return pd.Series(np.full(len(out), default), index=out.index, dtype=float)
    return out.fillna(float(out.median())).astype(float)


def _trend_slope(values: Any) -> float:
    if not isinstance(values, list):
        return 0.0
    clean = [float(v) for v in values if pd.notna(v)]
    if len(clean) < 3:
        return 0.0
    x = np.arange(len(clean), dtype=float)
    y = np.asarray(clean, dtype=float)
    try:
        slope = np.polyfit(x, y, 1)[0]
    except Exception:
        slope = 0.0
    return float(slope)


def build_ai_patterns(
    scored_df: pd.DataFrame,
    *,
    max_clusters: int = 8,
    max_outliers: int = 15,
    max_watchlist: int = 15,
) -> dict[str, Any]:
    if scored_df.empty:
        return {
            "summary": {
                "method": "kmeans + isolation_forest + trend_regression",
                "cluster_count": 0,
                "outlier_count": 0,
                "watchlist_count": 0,
                "avg_scope_risk": 0.0,
            },
            "hotspot_clusters": [],
            "outliers": [],
            "rising_watchlist": [],
        }

    work = scored_df.copy().reset_index(drop=True)
    required_defaults = {
        "risk_score": 0.0,
        "risk_uncertainty": 0.0,
        "replacement_cost": 0.0,
        "poverty_rate": 0.0,
        "minority_share": 0.0,
        "pct_housing_pre_1950": 0.0,
        "lead_90p_ppb": 0.0,
        "lat": 40.0,
        "lon": -74.5,
    }
    for col, default in required_defaults.items():
        if col not in work.columns:
            work[col] = default
        work[col] = _to_numeric(work[col], default=default)

    feature_cols = [
        "risk_score",
        "risk_uncertainty",
        "poverty_rate",
        "minority_share",
        "pct_housing_pre_1950",
        "lead_90p_ppb",
        "lat",
        "lon",
    ]
    feat = work[feature_cols].astype(float)
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feat)

    n_rows = len(work)
    n_clusters = max(2, min(int(np.sqrt(max(n_rows, 4) / 180)) + 3, max_clusters, n_rows))
    if n_rows < 2:
        work["cluster_id"] = 0
    else:
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        work["cluster_id"] = model.fit_predict(feat_scaled)

    cluster_rows: list[dict[str, Any]] = []
    for cluster_id, group in work.groupby("cluster_id", dropna=False):
        avg_risk = float(group["risk_score"].mean())
        avg_uncertainty = float(group["risk_uncertainty"].mean())
        hotspot_score = float(avg_risk * (1.0 + avg_uncertainty))
        cluster_rows.append(
            {
                "cluster_id": int(cluster_id),
                "size": int(len(group)),
                "avg_risk": avg_risk,
                "avg_uncertainty": avg_uncertainty,
                "avg_cost": float(group["replacement_cost"].mean()),
                "dominant_county": str(group["county"].astype(str).mode().iloc[0]) if "county" in group.columns else "Unknown",
                "hotspot_score": hotspot_score,
            }
        )
    cluster_rows = sorted(cluster_rows, key=lambda x: x["hotspot_score"], reverse=True)

    if n_rows >= 20:
        contamination = min(0.12, max(float(max_outliers / max(n_rows, 1)), 0.02))
        iforest = IsolationForest(
            n_estimators=220,
            contamination=contamination,
            random_state=42,
        )
        iforest.fit(feat_scaled)
        anomaly_score = -iforest.decision_function(feat_scaled)
    else:
        risk_z = (work["risk_score"] - work["risk_score"].mean()) / max(work["risk_score"].std(ddof=0), 1e-6)
        unc_z = (work["risk_uncertainty"] - work["risk_uncertainty"].mean()) / max(
            work["risk_uncertainty"].std(ddof=0), 1e-6
        )
        anomaly_score = (risk_z.abs() + unc_z.abs()).to_numpy()

    work["anomaly_score"] = pd.Series(anomaly_score, index=work.index).fillna(0.0)
    outliers_df = work.sort_values(["anomaly_score", "risk_score"], ascending=False).head(max_outliers)
    outliers = [
        {
            "geoid": str(row.get("geoid")),
            "county": str(row.get("county", "Unknown")),
            "municipality": str(row.get("municipality", row.get("geoid"))),
            "risk_score": float(row.get("risk_score", 0.0)),
            "risk_uncertainty": float(row.get("risk_uncertainty", 0.0)),
            "anomaly_score": float(row.get("anomaly_score", 0.0)),
            "replacement_cost": float(row.get("replacement_cost", 0.0)),
        }
        for _, row in outliers_df.iterrows()
    ]

    work["trend_slope_ppb_per_q"] = work.get("lead_trend", pd.Series([[]] * len(work))).apply(_trend_slope)
    work["trend_alert_score"] = (
        work["trend_slope_ppb_per_q"].clip(lower=0.0) * (0.65 + work["risk_score"]) + work["risk_uncertainty"] * 0.25
    )
    watch_df = (
        work[work["trend_slope_ppb_per_q"] > 0.0]
        .sort_values(["trend_alert_score", "risk_score"], ascending=False)
        .head(max_watchlist)
    )
    watchlist = [
        {
            "geoid": str(row.get("geoid")),
            "county": str(row.get("county", "Unknown")),
            "municipality": str(row.get("municipality", row.get("geoid"))),
            "risk_score": float(row.get("risk_score", 0.0)),
            "risk_uncertainty": float(row.get("risk_uncertainty", 0.0)),
            "trend_slope_ppb_per_q": float(row.get("trend_slope_ppb_per_q", 0.0)),
            "trend_alert_score": float(row.get("trend_alert_score", 0.0)),
        }
        for _, row in watch_df.iterrows()
    ]

    return {
        "summary": {
            "method": "kmeans + isolation_forest + trend_regression",
            "cluster_count": int(len(cluster_rows)),
            "outlier_count": int(len(outliers)),
            "watchlist_count": int(len(watchlist)),
            "avg_scope_risk": float(work["risk_score"].mean()),
        },
        "hotspot_clusters": cluster_rows,
        "outliers": outliers,
        "rising_watchlist": watchlist,
    }
