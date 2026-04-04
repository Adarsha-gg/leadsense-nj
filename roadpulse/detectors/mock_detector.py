from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .base import BaseDetector


class MockRoadDamageDetector(BaseDetector):
    """Heuristic detector used for local demo/testing without custom weights."""

    name = "mock"

    def __init__(self, min_area: int = 250, max_candidates: int = 25) -> None:
        self.min_area = min_area
        self.max_candidates = max_candidates

    def detect(self, frame: np.ndarray, frame_idx: int, timestamp_s: float) -> list[dict[str, Any]]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 140)

        dark = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            5,
        )
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[dict[str, Any]] = []
        candidates = sorted(contours, key=cv2.contourArea, reverse=True)[: self.max_candidates]
        for cnt in candidates:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            x2, y2 = x + bw, y + bh
            ar = bw / max(1.0, float(bh))

            region_edges = edges[y:y2, x:x2]
            edge_density = float(np.mean(region_edges > 0)) if region_edges.size else 0.0
            confidence = max(0.25, min(0.95, 0.35 + edge_density + area / float(h * w)))

            hazard_class = self._label_from_shape(ar, area / float(h * w))
            detections.append(
                {
                    "hazard_class": hazard_class,
                    "confidence": confidence,
                    "bbox": (int(x), int(y), int(x2), int(y2)),
                }
            )

        # Lane marking fade proxy from low edge content on road bottom-half.
        road_half = gray[h // 2 :, :]
        low_texture_score = float(np.mean(cv2.Canny(road_half, 80, 160) > 0))
        if low_texture_score < 0.02:
            detections.append(
                {
                    "hazard_class": "faded_lane_marking",
                    "confidence": 0.45,
                    "bbox": (int(w * 0.35), int(h * 0.55), int(w * 0.65), int(h * 0.95)),
                }
            )

        return detections

    @staticmethod
    def _label_from_shape(aspect_ratio: float, area_ratio: float) -> str:
        if area_ratio > 0.08:
            return "standing_water"
        if aspect_ratio > 4.0:
            return "longitudinal_crack"
        if 2.0 <= aspect_ratio <= 4.0:
            return "transverse_crack"
        if 0.7 <= aspect_ratio <= 1.6 and area_ratio > 0.01:
            return "pothole"
        return "alligator_crack"

