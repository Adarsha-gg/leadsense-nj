from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass
class PrioritizationConfig:
    top_k: int = 10
    budget_usd: float = 100000.0
    equity_boost_enabled: bool = True


def _segment_id(row: pd.Series) -> str:
    if pd.notna(row.get("lat")) and pd.notna(row.get("lon")):
        return f"seg_{round(float(row['lat']), 3)}_{round(float(row['lon']), 3)}"
    # Fallback segment from time window.
    window = int(float(row["timestamp_s"]) // 20)
    return f"seg_time_{window:03d}"


def build_priority_queue(df: pd.DataFrame, cfg: PrioritizationConfig) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "segment_id",
                "avg_risk",
                "max_severity",
                "issue_count",
                "estimated_repair_cost_usd",
                "expected_risk_reduction_pct",
            ]
        )
    work = df.copy()
    work["segment_id"] = work.apply(_segment_id, axis=1)

    grouped = (
        work.groupby("segment_id", as_index=False)
        .agg(
            avg_risk=("risk_score", "mean"),
            max_severity=("severity", "max"),
            issue_count=("hazard_class", "count"),
            lat=("lat", "mean"),
            lon=("lon", "mean"),
        )
    )
    grouped["lat"] = pd.to_numeric(grouped["lat"], errors="coerce").fillna(0.0)
    grouped["lon"] = pd.to_numeric(grouped["lon"], errors="coerce").fillna(0.0)

    grouped["estimated_repair_cost_usd"] = (
        600 * grouped["issue_count"] + 35 * grouped["max_severity"]
    ).round(2)
    grouped["expected_risk_reduction_pct"] = (
        (0.4 * grouped["avg_risk"] + 0.6 * grouped["max_severity"]) / 100.0 * 65.0
    ).round(2)

    if cfg.equity_boost_enabled:
        grouped["equity_boost"] = grouped["lat"].apply(lambda x: 1.05 if abs(x) > 0 else 1.0)
        grouped["priority_score"] = grouped["avg_risk"] * grouped["equity_boost"]
    else:
        grouped["priority_score"] = grouped["avg_risk"]

    grouped = grouped.sort_values(
        by=["priority_score", "max_severity", "issue_count"], ascending=False
    ).reset_index(drop=True)

    chosen_rows = []
    spent = 0.0
    for _, row in grouped.iterrows():
        projected = spent + float(row["estimated_repair_cost_usd"])
        if projected > cfg.budget_usd:
            continue
        spent = projected
        chosen_rows.append(row)
        if len(chosen_rows) >= cfg.top_k:
            break

    out = pd.DataFrame(chosen_rows)
    if out.empty:
        return out
    return out[
        [
            "segment_id",
            "avg_risk",
            "max_severity",
            "issue_count",
            "estimated_repair_cost_usd",
            "expected_risk_reduction_pct",
            "lat",
            "lon",
        ]
    ].round(2)
