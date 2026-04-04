from __future__ import annotations

import pandas as pd


def simulate_repair_impact(
    detections_df: pd.DataFrame, priority_df: pd.DataFrame, repaired_segments: int = 5
) -> dict[str, float]:
    if detections_df.empty:
        return {
            "baseline_avg_risk": 0.0,
            "post_repair_avg_risk": 0.0,
            "risk_drop_pct": 0.0,
        }

    baseline = float(detections_df["risk_score"].mean())
    if priority_df.empty:
        return {
            "baseline_avg_risk": baseline,
            "post_repair_avg_risk": baseline,
            "risk_drop_pct": 0.0,
        }

    target_segments = set(priority_df.head(repaired_segments)["segment_id"].tolist())
    work = detections_df.copy()
    work["risk_score"] = pd.to_numeric(work["risk_score"], errors="coerce").fillna(0.0).astype(float)
    if "segment_id" not in work.columns:
        # This fallback keeps simulation functional if queue was built externally.
        work["segment_id"] = (work["timestamp_s"] // 20).astype(int).map(lambda x: f"seg_time_{x:03d}")

    work.loc[work["segment_id"].isin(target_segments), "risk_score"] *= 0.45
    post = float(work["risk_score"].mean())
    drop = 0.0 if baseline == 0 else ((baseline - post) / baseline) * 100.0
    return {
        "baseline_avg_risk": round(baseline, 2),
        "post_repair_avg_risk": round(post, 2),
        "risk_drop_pct": round(max(0.0, drop), 2),
    }
