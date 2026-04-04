from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def export_csv(df: pd.DataFrame, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def export_markdown_report(
    metrics: dict[str, Any], detections_df: pd.DataFrame, priority_df: pd.DataFrame, path: str | Path
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RoadPulse AI Report",
        "",
        "## Key Metrics",
        f"- Total detections: {int(metrics.get('detection_count', 0))}",
        f"- Average severity: {metrics.get('avg_severity', 0):.2f}",
        f"- Average risk: {metrics.get('avg_risk', 0):.2f}",
        f"- High-risk events (>=70): {int(metrics.get('high_risk_count', 0))}",
        "",
        "## Top Priority Segments",
    ]
    if priority_df.empty:
        lines.append("- No priority segments generated.")
    else:
        for _, row in priority_df.head(10).iterrows():
            lines.append(
                f"- {row['segment_id']}: avg_risk={row['avg_risk']:.2f}, "
                f"issues={int(row['issue_count'])}, cost=${row['estimated_repair_cost_usd']:.0f}"
            )
    lines.append("")
    lines.append("## Notes")
    lines.append("- This report is generated from uploaded video and optional GPS metadata.")
    lines.append("- Use this as decision support, not as final engineering judgment.")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path

