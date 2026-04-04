# RoadPulse AI (NJBDA Edition)

RoadPulse AI is a computer-vision application for road hazard detection and repair prioritization.
This NJBDA edition adds New Jersey scenario presets, county context adjustment, synthetic demo asset generation, and judge-friendly KPI outputs.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Demo Assets

Generate synthetic NJ demo data:

```bash
python generate_demo_assets.py
```

This creates:
- `data/samples/synthetic_nj_drive.mp4`
- `data/samples/synthetic_nj_drive_gps.csv`

The app can also generate demo assets directly from the sidebar.

## Input Data

- Road video (`.mp4`, `.avi`, `.mov`, `.mkv`)
- Optional GPS CSV with columns: `timestamp`, `lat`, `lon`
- Optional templates in:
  - `data/templates/nj_gps_template.csv`
  - `data/templates/nj_context_profiles.csv`
  - `data/templates/nj_county_priority_weights.csv`

## Feature Set

45 implemented features are tracked in `FEATURE_TRACKER.md`.

## Optional Model Upgrade

To use trained YOLO weights:

```bash
pip install ultralytics
python train_yolo.py --data path/to/data.yaml --model yolov8n.pt --epochs 50
```

Then switch backend to `yolo` and provide model path in the app.

## Validation

```bash
python -m pytest -q
```

