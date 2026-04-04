# RoadPulse AI

RoadPulse AI is a computer-vision application for road hazard detection and repair prioritization.
It processes road videos, detects damage, scores severity/risk, and outputs a budget-constrained repair queue.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Data Inputs

- Road video (`.mp4`, `.avi`, `.mov`, `.mkv`)
- Optional GPS CSV with columns: `timestamp`, `lat`, `lon`

## 30 Implemented Features

1. Upload road video for analysis
2. Analyze camera snapshot frame
3. Optional GPS CSV geotagging
4. Detector backend switch (Mock / YOLO)
5. Confidence threshold control
6. Frame sampling control
7. Duplicate detection suppression
8. Max-frame processing budget
9. Bounding-box hazard overlays
10. Hazard-class taxonomy
11. Severity scoring
12. Composite risk scoring
13. Traffic context weighting
14. Weather context weighting
15. School-zone context weighting
16. Equity context weighting
17. Priority queue generation
18. Budget-constrained prioritization
19. Top-K segment selection
20. What-if repair simulation
21. Detection summary metrics
22. Hazard distribution chart
23. Risk timeline chart
24. Interactive map view
25. Class filter
26. Minimum risk filter
27. CSV export (detections)
28. CSV export (priority queue)
29. Markdown report export
30. Session run history

## Optional Model Upgrade

To use a trained YOLO model:

```bash
pip install ultralytics
```

Then select `yolo` backend in the UI and provide a `.pt` model path.

