from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    confidence_threshold: float = 0.35
    sample_every_n_frames: int = 5
    dedupe_pixel_radius: int = 80
    dedupe_time_window_s: float = 2.0
    max_frames: int = 400


@dataclass
class RiskWeights:
    traffic_weight: float = 0.30
    weather_weight: float = 0.20
    school_zone_weight: float = 0.15
    equity_weight: float = 0.15
    severity_weight: float = 0.20

