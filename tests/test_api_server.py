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


def test_api_dashboard_supports_county_scoping() -> None:
    client = TestClient(app)
    dash = client.get("/api/dashboard", params={"county": "Essex", "row_limit": 200})
    assert dash.status_code == 200
    body = dash.json()
    assert body["row_scope_county"] == "Essex"
    assert body["rows_returned"] <= 200
    assert all(str(row.get("county", "")).lower() == "essex" for row in body["rows"])


def test_api_ai_status_and_copilot_fallback(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app)

    status = client.get("/api/ai/status")
    assert status.status_code == 200
    status_body = status.json()
    assert status_body["enabled"] is False
    assert isinstance(status_body["model"], str)

    dash = client.get("/api/dashboard", params={"row_limit": 100})
    geoid = str(dash.json()["rows"][0]["geoid"])
    copilot = client.post(
        "/api/ai/copilot",
        json={
            "geoid": geoid,
            "question": "What actions should NJBDA prioritize for this area?",
            "budget": 2000000.0,
            "fairness_tolerance": 0.05,
            "optimizer_method": "greedy",
        },
    )
    assert copilot.status_code == 200
    body = copilot.json()
    assert body["geoid"] == geoid
    assert body["ai_used"] is False
    assert isinstance(body["answer"], str)
    assert len(body["answer"]) > 50


def test_api_ai_portfolio_fallback(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app)
    portfolio = client.post(
        "/api/ai/portfolio",
        json={
            "goal": "Maximize risk reduction but keep fairness strong and ensure at least one block per county.",
            "budget": 2000000.0,
            "fairness_tolerance": 0.05,
            "optimizer_method": "greedy",
            "county": "all",
        },
    )
    assert portfolio.status_code == 200
    body = portfolio.json()
    assert body["ai_used"] is False
    assert body["candidate_count"] > 0
    assert len(body["selected_rows"]) > 0
    assert "objective_profile" in body
    assert "portfolio_delta" in body


def test_api_ai_patterns_endpoint() -> None:
    client = TestClient(app)
    resp = client.post(
        "/api/ai/patterns",
        json={
            "budget": 2000000.0,
            "fairness_tolerance": 0.05,
            "optimizer_method": "greedy",
            "county": "all",
            "row_limit": 1000,
            "max_clusters": 6,
            "max_outliers": 10,
            "max_watchlist": 10,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["rows_analyzed"] > 0
    assert "summary" in body
    assert "hotspot_clusters" in body
    assert "outliers" in body
    assert "rising_watchlist" in body
