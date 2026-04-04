from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from roadpulse.analytics import (
    class_risk_heatmap,
    confidence_buckets,
    hazard_distribution,
    risk_band_breakdown,
    risk_timeline,
    segment_summary,
    summarize_detections,
)
from roadpulse.benchmark import compute_kpis, recommendation_cards
from roadpulse.config import PipelineConfig, RiskWeights
from roadpulse.demo_data import create_gps_csv, create_synthetic_video
from roadpulse.detectors.mock_detector import MockRoadDamageDetector
from roadpulse.detectors.yolo_detector import YoloRoadDamageDetector
from roadpulse.exporter import export_csv, export_markdown_report
from roadpulse.gps import normalize_gps_dataframe
from roadpulse.nj_context import COUNTIES, SCENARIOS, apply_county_risk_adjustment, get_scenario
from roadpulse.pipeline import run_pipeline
from roadpulse.prioritization import PrioritizationConfig, build_priority_queue
from roadpulse.simulation import simulate_repair_impact


st.set_page_config(page_title="RoadPulse AI - NJBDA Edition", layout="wide")

FEATURES = [
    "Upload road video for analysis",
    "Analyze camera snapshot frame",
    "Optional GPS CSV geotagging",
    "Detector backend switch (Mock / YOLO)",
    "Confidence threshold control",
    "Frame sampling control",
    "Duplicate detection suppression",
    "Max-frame processing budget",
    "Bounding-box hazard overlays",
    "Hazard-class taxonomy",
    "Severity scoring",
    "Composite risk scoring",
    "Traffic context weighting",
    "Weather context weighting",
    "School-zone context weighting",
    "Equity context weighting",
    "Priority queue generation",
    "Budget-constrained prioritization",
    "Top-K segment selection",
    "What-if repair simulation",
    "Detection summary metrics",
    "Hazard distribution chart",
    "Risk timeline chart",
    "Interactive map view",
    "Class filter",
    "Minimum risk filter",
    "CSV export (detections)",
    "CSV export (priority queue)",
    "Markdown report export",
    "Session run history",
    "NJ scenario presets",
    "County risk adjustment",
    "Synthetic demo video generation",
    "Synthetic GPS generation",
    "Scenario notes display",
    "Risk band breakdown chart",
    "Confidence diagnostics chart",
    "Segment leaderboard table",
    "Class-by-risk heatmap",
    "Operational KPI estimator",
    "CO2 reduction proxy",
    "Intervention recommendation cards",
    "NJBDA talking points panel",
    "Assumptions/model transparency panel",
    "Template data quick reference",
]


def _write_uploaded_to_temp(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return Path(tmp.name)


def _run_camera_snapshot(detector, threshold: float) -> pd.DataFrame:
    snap = st.camera_input("Quick Snapshot Analysis")
    if snap is None:
        return pd.DataFrame()
    file_bytes = snap.getvalue()
    frame = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        st.warning("Unable to decode camera image.")
        return pd.DataFrame()
    raw = detector.detect(frame, frame_idx=0, timestamp_s=0.0)
    rows = []
    for item in raw:
        if item["confidence"] < threshold:
            continue
        x1, y1, x2, y2 = item["bbox"]
        rows.append(
            {
                "hazard_class": item["hazard_class"],
                "confidence": float(item["confidence"]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "frame_idx": 0,
                "timestamp_s": 0.0,
                "severity": 45.0,
                "risk_score": 50.0,
                "lat": None,
                "lon": None,
                "source": getattr(detector, "name", "unknown"),
            }
        )
    return pd.DataFrame(rows)


def _render_map(priority_df: pd.DataFrame) -> None:
    st.subheader("Repair Priority Map")
    map_df = priority_df.copy().dropna(subset=["lat", "lon"])
    if map_df.empty:
        st.info("No geotagged segments available. Upload GPS CSV or use generated demo assets.")
        return
    center = [map_df["lat"].mean(), map_df["lon"].mean()]
    fmap = folium.Map(location=center, zoom_start=10, control_scale=True)
    for _, row in map_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=max(4, min(12, int(row["avg_risk"] / 10))),
            color="crimson" if row["avg_risk"] >= 70 else "orange",
            fill=True,
            fill_opacity=0.75,
            tooltip=f"{row['segment_id']} | risk {row['avg_risk']:.1f}",
        ).add_to(fmap)
    st_folium(fmap, width=950, height=430)


def _download_buttons(detections_df: pd.DataFrame, priority_df: pd.DataFrame, metrics: dict) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    detections_path = export_csv(detections_df, Path("artifacts") / f"detections_{ts}.csv")
    priority_path = export_csv(priority_df, Path("artifacts") / f"priority_{ts}.csv")
    report_path = export_markdown_report(metrics, detections_df, priority_df, Path("reports") / f"report_{ts}.md")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.download_button(
            "Download Detections CSV",
            data=detections_path.read_bytes(),
            file_name=detections_path.name,
            mime="text/csv",
        )
    with col_b:
        st.download_button(
            "Download Priority CSV",
            data=priority_path.read_bytes(),
            file_name=priority_path.name,
            mime="text/csv",
        )
    with col_c:
        st.download_button(
            "Download Markdown Report",
            data=report_path.read_text(encoding="utf-8"),
            file_name=report_path.name,
            mime="text/markdown",
        )


def _ensure_segment_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "segment_id" not in out.columns:
        out["segment_id"] = (out["timestamp_s"] // 20).astype(int).map(lambda x: f"seg_time_{x:03d}")
    return out


st.title("RoadPulse AI - NJBDA Edition")
st.caption("Computer vision for road hazard detection, risk scoring, and budget-aware repair prioritization.")

if "run_history" not in st.session_state:
    st.session_state["run_history"] = []
if "demo_video_path" not in st.session_state:
    st.session_state["demo_video_path"] = ""
if "demo_gps_path" not in st.session_state:
    st.session_state["demo_gps_path"] = ""

with st.sidebar:
    st.header("NJBDA Scenario")
    scenario_name = st.selectbox("Scenario Profile", list(SCENARIOS.keys()), index=0)
    selected_county = st.selectbox("County Focus", COUNTIES, index=6)
    scenario = get_scenario(scenario_name)
    st.caption(scenario.notes)

    if st.button("Generate Demo Assets"):
        Path("artifacts").mkdir(parents=True, exist_ok=True)
        demo_video = Path("artifacts") / "synthetic_nj_drive.mp4"
        demo_gps = Path("artifacts") / "synthetic_nj_drive_gps.csv"
        create_synthetic_video(demo_video)
        create_gps_csv(demo_gps)
        st.session_state["demo_video_path"] = str(demo_video)
        st.session_state["demo_gps_path"] = str(demo_gps)
        st.success("Demo assets created in artifacts/.")

    st.header("Pipeline Controls")
    backend = st.selectbox("Detector Backend", ["mock", "yolo"], index=0)
    model_path = ""
    if backend == "yolo":
        model_path = st.text_input("YOLO model path (.pt)", value="models/roadpulse.pt")
    confidence_threshold = st.slider("Confidence Threshold", 0.05, 0.95, 0.35, 0.05)
    sample_every_n = st.slider("Sample Every N Frames", 1, 30, 5, 1)
    dedupe_pixels = st.slider("Dedupe Pixel Radius", 20, 300, 80, 5)
    dedupe_seconds = st.slider("Dedupe Time Window (s)", 0.1, 10.0, 2.0, 0.1)
    max_frames = st.slider("Max Processed Frames", 50, 3000, 400, 50)

    st.header("Risk Context")
    traffic_index = st.slider("Traffic Index", 0.0, 1.0, scenario.traffic_index, 0.05)
    weather_index = st.slider("Weather Risk Index", 0.0, 1.0, scenario.weather_index, 0.05)
    school_zone_index = st.slider("School-Zone Exposure", 0.0, 1.0, scenario.school_zone_index, 0.05)
    equity_index = st.slider("Equity Priority Index", 0.0, 1.0, scenario.equity_index, 0.05)

    st.header("Risk Weights")
    severity_weight = st.slider("Severity Weight", 0.0, 1.0, 0.2, 0.05)
    traffic_weight = st.slider("Traffic Weight", 0.0, 1.0, 0.3, 0.05)
    weather_weight = st.slider("Weather Weight", 0.0, 1.0, 0.2, 0.05)
    school_weight = st.slider("School-Zone Weight", 0.0, 1.0, 0.15, 0.05)
    equity_weight = st.slider("Equity Weight", 0.0, 1.0, 0.15, 0.05)

    st.header("Priority Queue")
    top_k = st.slider("Top K Segments", 3, 50, scenario.top_k, 1)
    budget = st.number_input(
        "Repair Budget (USD)",
        min_value=5000.0,
        max_value=5000000.0,
        value=float(scenario.budget_usd),
        step=5000.0,
    )
    equity_boost = st.checkbox("Enable Equity Boost", value=True)

source_mode = st.radio("Input Source", ["Upload your own assets", "Use generated demo assets"], horizontal=True)
video_file = None
gps_file = None
demo_video_path = st.session_state.get("demo_video_path", "")
demo_gps_path = st.session_state.get("demo_gps_path", "")

if source_mode == "Upload your own assets":
    video_file = st.file_uploader("Upload Road Video", type=["mp4", "avi", "mov", "mkv"])
    gps_file = st.file_uploader("Upload GPS CSV (optional)", type=["csv"])
else:
    if demo_video_path and Path(demo_video_path).exists():
        st.success(f"Using demo video: {demo_video_path}")
    else:
        st.warning("No demo video found yet. Click `Generate Demo Assets` in sidebar.")

col_l, col_r = st.columns([2, 1])
with col_l:
    run_video = st.button("Run Full Video Analysis", type="primary")
with col_r:
    show_features = st.checkbox("Show 45 Features", value=False)

if show_features:
    st.subheader("Implemented Features (45)")
    st.dataframe(pd.DataFrame({"feature": FEATURES, "status": ["implemented"] * len(FEATURES)}), use_container_width=True)

if backend == "mock":
    detector = MockRoadDamageDetector()
else:
    try:
        detector = YoloRoadDamageDetector(model_path=model_path, conf_threshold=confidence_threshold)
    except Exception as exc:
        st.error(f"YOLO backend unavailable: {exc}")
        detector = MockRoadDamageDetector()

gps_df = pd.DataFrame()
if source_mode == "Upload your own assets":
    if gps_file is not None:
        try:
            gps_df = normalize_gps_dataframe(pd.read_csv(gps_file))
            st.success(f"Loaded GPS points: {len(gps_df)}")
        except Exception as exc:
            st.error(f"Invalid GPS CSV: {exc}")
            gps_df = pd.DataFrame()
else:
    if demo_gps_path and Path(demo_gps_path).exists():
        gps_df = normalize_gps_dataframe(pd.read_csv(demo_gps_path))
        st.info(f"Loaded demo GPS points: {len(gps_df)}")

camera_df = _run_camera_snapshot(detector, confidence_threshold)
if not camera_df.empty:
    st.info(f"Snapshot detections: {len(camera_df)}")
    st.dataframe(camera_df, use_container_width=True)

if run_video:
    if source_mode == "Upload your own assets" and video_file is None:
        st.error("Upload a video first.")
    elif source_mode == "Use generated demo assets" and (not demo_video_path or not Path(demo_video_path).exists()):
        st.error("Generate demo assets first from sidebar.")
    else:
        cfg = PipelineConfig(
            confidence_threshold=confidence_threshold,
            sample_every_n_frames=sample_every_n,
            dedupe_pixel_radius=dedupe_pixels,
            dedupe_time_window_s=dedupe_seconds,
            max_frames=max_frames,
        )
        weights = RiskWeights(
            severity_weight=severity_weight,
            traffic_weight=traffic_weight,
            weather_weight=weather_weight,
            school_zone_weight=school_weight,
            equity_weight=equity_weight,
        )
        with st.spinner("Processing video..."):
            if source_mode == "Upload your own assets":
                video_path = _write_uploaded_to_temp(video_file)
            else:
                video_path = Path(demo_video_path)
            result = run_pipeline(
                video_path=video_path,
                detector=detector,
                cfg=cfg,
                risk_weights=weights,
                gps_df=gps_df if not gps_df.empty else None,
                weather_index=weather_index,
                traffic_index=traffic_index,
                school_zone_index=school_zone_index,
                equity_index=equity_index,
            )

        detections_df = result.detections_df
        if detections_df.empty:
            st.warning("No detections found with current settings.")
        else:
            detections_df = _ensure_segment_id(detections_df)
            detections_df = apply_county_risk_adjustment(detections_df, selected_county)
            priority_cfg = PrioritizationConfig(top_k=top_k, budget_usd=budget, equity_boost_enabled=equity_boost)
            priority_df = build_priority_queue(detections_df, priority_cfg)
            metrics = summarize_detections(detections_df)
            kpis = compute_kpis(detections_df, priority_df)

            st.subheader("Core Metrics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Detections", int(metrics["detection_count"]))
            c2.metric("Avg Severity", f"{metrics['avg_severity']:.1f}")
            c3.metric("Avg Risk", f"{metrics['avg_risk']:.1f}")
            c4.metric("High Risk Events", int(metrics["high_risk_count"]))

            st.subheader("Operational KPIs")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Hours Saved", f"{kpis['hours_saved']:.1f}h")
            k2.metric("Repair Budget Used", f"${kpis['estimated_repairs_cost']:,.0f}")
            k3.metric("Damage Avoided (Proxy)", f"${kpis['estimated_vehicle_damage_avoided']:,.0f}")
            k4.metric("CO2 Avoided (Proxy)", f"{kpis['co2_kg_avoided']:.0f} kg")

            tabs = st.tabs(["Overview", "Quality", "Segments", "NJBDA Pitch Cards"])

            with tabs[0]:
                dist_df = hazard_distribution(detections_df)
                tl_df = risk_timeline(detections_df)
                left, right = st.columns(2)
                with left:
                    st.plotly_chart(
                        px.bar(dist_df, x="hazard_class", y="count", title="Hazard Distribution"),
                        use_container_width=True,
                    )
                with right:
                    st.plotly_chart(
                        px.line(tl_df, x="bucket_min", y="avg_risk", markers=True, title="Risk Timeline (minutes)"),
                        use_container_width=True,
                    )
                st.subheader("Preview Frames")
                for i, img in enumerate(result.preview_frames[:6]):
                    st.image(img, caption=f"Annotated frame {i + 1}", use_container_width=True)

            with tabs[1]:
                rb_df = risk_band_breakdown(detections_df)
                cb_df = confidence_buckets(detections_df)
                cl, cr = st.columns(2)
                with cl:
                    st.plotly_chart(px.bar(rb_df, x="risk_band", y="count", title="Risk Band Breakdown"), use_container_width=True)
                with cr:
                    st.plotly_chart(
                        px.bar(cb_df, x="confidence_bucket", y="count", title="Confidence Bucket Diagnostics"),
                        use_container_width=True,
                    )
                heat_df = class_risk_heatmap(detections_df)
                if not heat_df.empty:
                    st.plotly_chart(
                        px.density_heatmap(
                            heat_df,
                            x="risk_band",
                            y="hazard_class",
                            z="count",
                            color_continuous_scale="Oranges",
                            title="Class vs Risk Heatmap",
                        ),
                        use_container_width=True,
                    )

            with tabs[2]:
                st.subheader("Priority Queue")
                class_filter = st.multiselect(
                    "Filter Hazard Classes",
                    sorted(detections_df["hazard_class"].unique().tolist()),
                    default=sorted(detections_df["hazard_class"].unique().tolist()),
                )
                min_risk = st.slider("Minimum Risk Filter", 0, 100, 0, 1)
                filtered = detections_df[
                    detections_df["hazard_class"].isin(class_filter) & (detections_df["risk_score"] >= min_risk)
                ]
                st.dataframe(filtered.head(700), use_container_width=True)
                st.dataframe(priority_df, use_container_width=True)
                seg_df = segment_summary(detections_df)
                st.subheader("Segment Leaderboard")
                st.dataframe(seg_df.head(20), use_container_width=True)

                repaired_segments = st.slider(
                    "What-if: Repaired Segments",
                    1,
                    max(1, len(priority_df)),
                    min(5, max(1, len(priority_df))),
                    1,
                )
                sim = simulate_repair_impact(detections_df, priority_df, repaired_segments=repaired_segments)
                st.metric(
                    "Simulated Risk Drop",
                    f"{sim['risk_drop_pct']:.1f}%",
                    delta=f"{sim['baseline_avg_risk']:.1f} -> {sim['post_repair_avg_risk']:.1f}",
                )
                _render_map(priority_df)

            with tabs[3]:
                st.markdown("### Intervention Recommendations")
                for card in recommendation_cards(detections_df):
                    st.info(card)
                st.markdown("### NJBDA Judge-Friendly Talking Points")
                st.write(
                    "- AI is not only detecting defects, it is producing a budget-feasible repair queue.\n"
                    "- County/scenario context shows policy relevance beyond one street or one campus.\n"
                    "- Dashboard outputs are legible to engineering, business, and public-policy audiences.\n"
                    "- Sustainability and safety are quantified via risk reduction and CO2 proxy metrics."
                )
                st.markdown("### Data Templates")
                st.write(
                    "- `data/templates/nj_gps_template.csv`\n"
                    "- `data/templates/nj_context_profiles.csv`\n"
                    "- `data/templates/nj_county_priority_weights.csv`"
                )

            with st.expander("Model Assumptions and Transparency"):
                st.write(
                    "Current demo uses either a heuristic detector (mock) or optional YOLO model. "
                    "Economic and CO2 values are proxies for poster-level comparison; they should be "
                    "recalibrated with agency cost data for deployment."
                )

            _download_buttons(detections_df, priority_df, metrics)
            st.session_state["run_history"].append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "scenario": scenario_name,
                    "county": selected_county,
                    "detections": int(metrics["detection_count"]),
                    "avg_risk": round(metrics["avg_risk"], 2),
                    "top_segment": None if priority_df.empty else str(priority_df.iloc[0]["segment_id"]),
                }
            )

if st.session_state["run_history"]:
    st.subheader("Session Run History")
    st.dataframe(pd.DataFrame(st.session_state["run_history"]), use_container_width=True)

