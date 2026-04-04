from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd

from .config import PipelineConfig, RiskWeights
from .gps import interpolate_lat_lon
from .schemas import Detection
from .severity import compute_risk_score, compute_severity


@dataclass
class PipelineResult:
    detections_df: pd.DataFrame
    preview_frames: list


def _draw_detection(frame, det: Detection) -> None:
    color = (0, 165, 255) if det.risk_score < 70 else (0, 0, 255)
    cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)
    label = f"{det.hazard_class} {det.risk_score:.0f}"
    cv2.putText(
        frame,
        label,
        (det.x1, max(20, det.y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def _is_duplicate(candidate: Detection, recent: list[Detection], px_radius: int, time_window_s: float) -> bool:
    cx = (candidate.x1 + candidate.x2) / 2
    cy = (candidate.y1 + candidate.y2) / 2
    for prev in recent:
        if candidate.hazard_class != prev.hazard_class:
            continue
        if abs(candidate.timestamp_s - prev.timestamp_s) > time_window_s:
            continue
        px = (prev.x1 + prev.x2) / 2
        py = (prev.y1 + prev.y2) / 2
        if ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 <= px_radius:
            return True
    return False


def run_pipeline(
    video_path: str | Path,
    detector,
    cfg: PipelineConfig,
    risk_weights: RiskWeights,
    gps_df: pd.DataFrame | None = None,
    weather_index: float = 0.5,
    traffic_index: float = 0.6,
    school_zone_index: float = 0.3,
    equity_index: float = 0.5,
) -> PipelineResult:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:  # NaN guard
        fps = 30.0

    detections: list[Detection] = []
    preview_frames = []
    frame_idx = 0
    processed = 0
    recent: list[Detection] = []

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % cfg.sample_every_n_frames != 0:
            frame_idx += 1
            continue
        timestamp_s = frame_idx / fps
        raw = detector.detect(frame, frame_idx, timestamp_s)
        frame_h, frame_w = frame.shape[:2]
        annotated = frame.copy()

        for item in raw:
            hazard_class = str(item["hazard_class"]).lower().strip()
            confidence = float(item["confidence"])
            if confidence < cfg.confidence_threshold:
                continue
            x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
            area_ratio = max(0.0, ((x2 - x1) * (y2 - y1)) / float(frame_h * frame_w))
            severity = compute_severity(area_ratio, confidence, hazard_class)
            risk_score = compute_risk_score(
                severity=severity,
                traffic_index=traffic_index,
                weather_index=weather_index,
                school_zone_index=school_zone_index,
                equity_index=equity_index,
                severity_weight=risk_weights.severity_weight,
                traffic_weight=risk_weights.traffic_weight,
                weather_weight=risk_weights.weather_weight,
                school_zone_weight=risk_weights.school_zone_weight,
                equity_weight=risk_weights.equity_weight,
            )
            lat, lon = (None, None)
            if gps_df is not None and not gps_df.empty:
                lat, lon = interpolate_lat_lon(gps_df, timestamp_s)
            det = Detection(
                hazard_class=hazard_class,
                confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                frame_idx=frame_idx,
                timestamp_s=round(timestamp_s, 2),
                severity=severity,
                risk_score=risk_score,
                lat=lat,
                lon=lon,
                source=getattr(detector, "name", "unknown"),
            )
            if _is_duplicate(det, recent, cfg.dedupe_pixel_radius, cfg.dedupe_time_window_s):
                continue
            detections.append(det)
            recent.append(det)
            if len(recent) > 120:
                recent = recent[-120:]
            _draw_detection(annotated, det)

        if len(preview_frames) < 8:
            preview_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        frame_idx += 1
        processed += 1
        if processed >= cfg.max_frames:
            break

    cap.release()
    df = pd.DataFrame([d.to_dict() for d in detections])
    return PipelineResult(detections_df=df, preview_frames=preview_frames)

