from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseDetector(ABC):
    name = "base"

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_idx: int, timestamp_s: float) -> list[dict[str, Any]]:
        """Return list of raw detections with class/conf/bbox."""

