from __future__ import annotations

from fastapi.testclient import TestClient

from app.api_server import app, build_benchmark_payload, build_dashboard_payload


def test_build_dashboard_payload_has_map_and_fairness_data() -> None:
    payload = build_dashboard_payload(
        budget=2000000.0,
        fairness_tolerance=0.05,
        min_county_coverage=0,
        optimizer_method="greedy",
    )
    assert "rows" in payload
    assert len(payload["rows"]) > 0
    first = payload["rows"][0]
    assert "lat" in first
    assert "lon" in first
    assert "top_drivers" in first
    assert "fairness_comparison" in payload
    assert "county_spend_comparison" in payload["fairness_comparison"]


def test_build_benchmark_payload_has_ablation_table() -> None:
    payload = build_benchmark_payload()
    assert "ablation_accuracy_table" in payload
    assert len(payload["ablation_accuracy_table"]) >= 5


def test_api_endpoints_serve_dashboard_and_frontend() -> None:
    client = TestClient(app)
    root = client.get("/")
    assert root.status_code == 200
    assert "LeadSense NJ Dashboard" in root.text

    dash = client.get("/api/dashboard")
    assert dash.status_code == 200
    body = dash.json()
    assert "rows" in body
    assert len(body["rows"]) > 0
