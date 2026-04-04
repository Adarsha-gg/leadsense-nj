from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


HAZARD_CLASSES = [
    "pothole",
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
    "standing_water",
    "faded_lane_marking",
    "road_debris",
]


@dataclass
class Detection:
    hazard_class: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    frame_idx: int
    timestamp_s: float
    severity: float
    risk_score: float
    lat: float | None = None
    lon: float | None = None
    source: str = "mock"

    @property
    def bbox_area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PriorityItem:
    segment_id: str
    avg_risk: float
    max_severity: float
    issue_count: int
    estimated_repair_cost_usd: float
    expected_risk_reduction_pct: float
    lat: float | None = None
    lon: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

