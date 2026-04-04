from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def create_synthetic_video(video_path: Path, width: int = 960, height: int = 540, frames: int = 320) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 20.0, (width, height))
    for i in range(frames):
        frame = np.full((height, width, 3), 120, dtype=np.uint8)
        cv2.rectangle(frame, (0, int(height * 0.48)), (width, height), (88, 88, 88), -1)

        lane_color = 230 - (i % 60)
        for x in range(80, width, 140):
            cv2.rectangle(
                frame,
                (x, int(height * 0.7)),
                (x + 56, int(height * 0.72)),
                (lane_color, lane_color, lane_color),
                -1,
            )

        cx = 220 + (i * 3) % 450
        cy = int(height * 0.78)
        cv2.circle(frame, (cx, cy), 24, (35, 35, 35), -1)
        cv2.line(frame, (150, int(height * 0.63) + (i % 20)), (840, int(height * 0.63) + (i % 20)), (30, 30, 30), 3)

        if i % 55 < 28:
            cv2.ellipse(
                frame,
                (700, int(height * 0.8)),
                (90, 30),
                0,
                0,
                360,
                (70, 70, 95),
                -1,
            )
        writer.write(frame)
    writer.release()


def create_gps_csv(csv_path: Path, points: int = 400) -> None:
    base_lat = 40.7357
    base_lon = -74.1724
    timestamps = np.linspace(0, points / 2, points)
    lat = base_lat + np.linspace(0, 0.085, points)
    lon = base_lon + np.linspace(0, 0.11, points)
    df = pd.DataFrame({"timestamp": timestamps.round(2), "lat": lat.round(6), "lon": lon.round(6)})
    df.to_csv(csv_path, index=False)

