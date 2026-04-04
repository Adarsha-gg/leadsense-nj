from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO model for RoadPulse AI.")
    parser.add_argument("--data", required=True, help="Path to dataset YAML")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default="runs/roadpulse")
    parser.add_argument("--name", default="road_damage")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise SystemExit("Install ultralytics first: pip install ultralytics") from exc

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()

