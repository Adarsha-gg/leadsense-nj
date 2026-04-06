from __future__ import annotations

from dataclasses import asdict
from dataclasses import is_dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi import Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from leadsense_nj.baseline import fit_tabular_logistic
from leadsense_nj.demo import build_demo_snapshot
from leadsense_nj.explainability import top_feature_drivers
from leadsense_nj.optimization import optimize_replacement_plan, optimize_replacement_plan_ilp
from leadsense_nj.preprocessing import build_feature_table
from leadsense_nj.research import run_model_research_benchmark
from leadsense_nj.target import with_elevated_risk_label

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "block_group_features_sample.csv"
WEB_DIR = PROJECT_ROOT / "web"


def _risk_band(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.45:
        return "MEDIUM"
    return "LOW"


def _normalize_value(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _normalize_value(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _normalize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    if isinstance(value, tuple):
        return [_normalize_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _serialize_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    safe_df = df.copy()
    for col in safe_df.columns:
        if safe_df[col].dtype.kind in {"M", "m"}:
            safe_df[col] = safe_df[col].astype(str)
    records = safe_df.to_dict(orient="records")
    return _normalize_value(records)


def _build_driver_map(df: pd.DataFrame) -> dict[str, list[dict[str, float | str]]]:
    labeled = with_elevated_risk_label(df)
    model, _ = fit_tabular_logistic(labeled, epochs=600, learning_rate=0.1)
    out: dict[str, list[dict[str, float | str]]] = {}
    for _, row in labeled.iterrows():
        geoid = str(row["geoid"])
        drivers = top_feature_drivers(model, row, top_k=3)
        out[geoid] = [{"feature": str(name), "score": float(score)} for name, score in drivers]
    return out


def _compute_fairness_comparison(
    scored_df: pd.DataFrame,
    *,
    budget: float,
    fairness_tolerance: float,
    min_county_coverage: int,
    optimizer_method: str,
) -> dict[str, Any]:
    optimizer = optimize_replacement_plan_ilp if optimizer_method == "ilp" else optimize_replacement_plan
    selected_fair, summary_fair = optimizer(
        scored_df,
        budget=budget,
        fairness_tolerance=fairness_tolerance,
        min_county_coverage=min_county_coverage,
    )
    selected_no_fair, summary_no_fair = optimizer(
        scored_df,
        budget=budget,
        fairness_tolerance=1.0,
        min_county_coverage=min_county_coverage,
    )

    fair_spend = (
        selected_fair.groupby("county", dropna=False)["replacement_cost"].sum().rename("with_fairness_spend")
        if not selected_fair.empty
        else pd.Series(dtype=float)
    )
    no_fair_spend = (
        selected_no_fair.groupby("county", dropna=False)["replacement_cost"].sum().rename("without_fairness_spend")
        if not selected_no_fair.empty
        else pd.Series(dtype=float)
    )
    county_table = pd.concat([fair_spend, no_fair_spend], axis=1).fillna(0.0).reset_index()
    county_table["spend_delta"] = county_table["with_fairness_spend"] - county_table["without_fairness_spend"]
    county_table = county_table.sort_values("county").reset_index(drop=True)

    return {
        "with_fairness": {
            "summary": _normalize_value(summary_fair),
            "selected_rows": _serialize_df(selected_fair),
        },
        "without_fairness": {
            "summary": _normalize_value(summary_no_fair),
            "selected_rows": _serialize_df(selected_no_fair),
        },
        "county_spend_comparison": _serialize_df(county_table),
    }


def build_dashboard_payload(
    *,
    budget: float = 35000.0,
    fairness_tolerance: float = 0.05,
    min_county_coverage: int = 0,
    optimizer_method: str = "ilp",
) -> dict[str, Any]:
    base_df = build_feature_table(DEFAULT_DATA_PATH)
    snapshot = build_demo_snapshot(
        base_df,
        budget=budget,
        fairness_tolerance=fairness_tolerance,
        min_county_coverage=min_county_coverage,
        optimizer_method=optimizer_method,
    )

    driver_map = _build_driver_map(base_df)
    scored = snapshot.scored_df.copy()
    scored["geoid"] = scored["geoid"].astype(str)
    scored["risk_band"] = scored["risk_score"].map(lambda x: _risk_band(float(x)))
    scored["top_drivers"] = scored["geoid"].map(driver_map)

    trend_cols = [col for col in scored.columns if col.startswith("q") and col.endswith("_lead_ppb")]
    trend_cols = sorted(trend_cols, key=lambda c: int(c[1:].split("_")[0])) if trend_cols else []
    scored["lead_trend"] = scored[trend_cols].apply(
        lambda row: [float(pd.to_numeric(v, errors="coerce")) for v in row.tolist()],
        axis=1,
    ) if trend_cols else [[] for _ in range(len(scored))]

    fairness = _compute_fairness_comparison(
        scored,
        budget=budget,
        fairness_tolerance=fairness_tolerance,
        min_county_coverage=min_county_coverage,
        optimizer_method=optimizer_method,
    )

    return {
        "params": {
            "budget": float(budget),
            "fairness_tolerance": float(fairness_tolerance),
            "min_county_coverage": int(min_county_coverage),
            "optimizer_method": str(optimizer_method),
        },
        "rows": _serialize_df(scored),
        "selected_rows": _serialize_df(snapshot.selected_df),
        "optimization_summary": _normalize_value(snapshot.optimization_summary),
        "policy_briefs": _normalize_value(snapshot.policy_briefs),
        "comparison_metrics": _normalize_value(snapshot.comparison_metrics),
        "fairness_comparison": fairness,
    }


@lru_cache(maxsize=1)
def build_benchmark_payload() -> dict[str, Any]:
    artifact_path = PROJECT_ROOT / "artifacts" / "research" / "benchmark_results.json"
    if artifact_path.exists():
        try:
            report = json.loads(artifact_path.read_text(encoding="utf-8"))
            if isinstance(report, dict) and "ablation_accuracy_table" in report:
                return _normalize_value(report)
        except Exception:
            pass

    research_path = PROJECT_ROOT / "data" / "processed" / "nj_research_features.csv"
    dataset_path = research_path if research_path.exists() else DEFAULT_DATA_PATH
    df = build_feature_table(dataset_path)
    report = run_model_research_benchmark(
        df,
        n_splits=5 if len(df) >= 1000 else 3,
        threshold=0.5,
        random_state=42,
        max_rows=2500 if len(df) > 2500 else None,
    )
    return _normalize_value(report)


app = FastAPI(title="LeadSense NJ API", version="0.1.0")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/api/health")
def api_health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/benchmark")
def api_benchmark() -> dict[str, Any]:
    return build_benchmark_payload()


@app.get("/api/dashboard")
def api_dashboard(
    budget: float = Query(default=35000.0, ge=10000.0, le=1000000.0),
    fairness_tolerance: float = Query(default=0.05, ge=0.0, le=1.0),
    min_county_coverage: int = Query(default=0, ge=0, le=10),
    optimizer_method: str = Query(default="ilp", pattern="^(ilp|greedy)$"),
) -> dict[str, Any]:
    return build_dashboard_payload(
        budget=budget,
        fairness_tolerance=fairness_tolerance,
        min_county_coverage=min_county_coverage,
        optimizer_method=optimizer_method,
    )


@app.get("/")
def root() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
