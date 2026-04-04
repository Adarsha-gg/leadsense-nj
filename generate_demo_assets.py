from __future__ import annotations

import argparse
from pathlib import Path

from roadpulse.demo_data import create_gps_csv, create_synthetic_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic demo video and GPS assets.")
    parser.add_argument("--outdir", default="data/samples")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    video_path = outdir / "synthetic_nj_drive.mp4"
    gps_path = outdir / "synthetic_nj_drive_gps.csv"

    create_synthetic_video(video_path)
    create_gps_csv(gps_path)
    print(f"Created video: {video_path}")
    print(f"Created gps:   {gps_path}")


if __name__ == "__main__":
    main()
