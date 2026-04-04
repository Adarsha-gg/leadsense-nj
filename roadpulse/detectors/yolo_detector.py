from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseDetector


class YoloRoadDamageDetector(BaseDetector):
    name = "yolo"

    def __init__(self, model_path: str, conf_threshold: float = 0.25) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Install it with `pip install ultralytics`."
            ) from exc
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray, frame_idx: int, timestamp_s: float) -> list[dict[str, Any]]:
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        raw: list[dict[str, Any]] = []
        if not results:
            return raw
        result = results[0]
        boxes = result.boxes
        names = result.names
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cls_idx = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            hazard_class = str(names.get(cls_idx, "road_damage")).lower().replace(" ", "_")
            raw.append({"hazard_class": hazard_class, "confidence": conf, "bbox": (x1, y1, x2, y2)})
        return raw

