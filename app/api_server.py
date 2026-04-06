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
from fastapi import HTTPException
from fastapi import Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydantic import Field

from leadsense_nj.ai_assistant import configured_model
from leadsense_nj.ai_assistant import generate_block_answer
from leadsense_nj.ai_assistant import is_ai_enabled
from leadsense_nj.demo import build_demo_snapshot
from leadsense_nj.metrics import compute_model_vs_historical_metrics
from leadsense_nj.optimization import optimize_replacement_plan, optimize_replacement_plan_ilp
from leadsense_nj.policy_brief import generate_policy_brief
from leadsense_nj.preprocessing import build_feature_table
from leadsense_nj.research import run_model_research_benchmark

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "block_group_features_sample.csv"
RESEARCH_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "nj_research_features.csv"
WEB_DIR = PROJECT_ROOT / "web"


class AICopilotRequest(BaseModel):
    geoid: str = Field(min_length=1, max_length=40)
    question: str = Field(min_length=1, max_length=4000)
    budget: float = Field(default=2000000.0, ge=10000.0, le=100000000.0)
    fairness_tolerance: float = Field(default=0.05, ge=0.0, le=1.0)
    min_county_coverage: int = Field(default=0, ge=0, le=10)
    optimizer_method: str = Field(default="greedy", pattern="^(ilp|greedy)$")


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


def _fast_top_drivers(row: pd.Series) -> list[dict[str, float | str]]:
    def num(col: str, default: float = 0.0) -> float:
        return float(pd.to_numeric(row.get(col, default), errors="coerce"))

    housing_age = max(0.0, (2000.0 - num("median_housing_year", 2000.0)) / 100.0)
    candidates = [
        ("lead_90p_ppb", max(0.0, num("lead_90p_ppb")) / 20.0),
        ("pct_housing_pre_1950", max(0.0, num("pct_housing_pre_1950"))),
        ("pws_action_level_exceedance_5y", max(0.0, num("pws_action_level_exceedance_5y"))),
        ("pws_any_sample_gt15_3y", max(0.0, num("pws_any_sample_gt15_3y"))),
        ("poverty_rate", max(0.0, num("poverty_rate"))),
        ("minority_share", max(0.0, num("minority_share"))),
        ("housing_age_proxy", housing_age),
        ("winter_freeze_thaw_days", max(0.0, num("winter_freeze_thaw_days")) / 100.0),
    ]
    top = sorted(candidates, key=lambda x: abs(x[1]), reverse=True)[:3]
    return [{"feature": str(name), "score": float(score)} for name, score in top]


def _compute_fairness_comparison(
    scored_df: pd.DataFrame,
    *,
    budget: float,
    fairness_tolerance: float,
    min_county_coverage: int,
    optimizer_method: str,
) -> dict[str, Any]:
    effective_optimizer = optimizer_method
    if len(scored_df) > 1000 and optimizer_method == "ilp":
        effective_optimizer = "greedy"
    optimizer = optimize_replacement_plan_ilp if effective_optimizer == "ilp" else optimize_replacement_plan
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

    same_plan = (
        len(selected_fair) == len(selected_no_fair)
        and len(
            set(selected_fair.get("geoid", pd.Series(dtype=str)).astype(str).tolist())
            .symmetric_difference(set(selected_no_fair.get("geoid", pd.Series(dtype=str)).astype(str).tolist()))
        )
        == 0
    )
    fairness_mode = "requested_tolerance"
    stress_target = None
    if same_plan and not scored_df.empty:
        global_share = float(pd.to_numeric(scored_df["minority_share"], errors="coerce").fillna(0.0).mean())
        baseline_share = float(summary_no_fair.achieved_minority_share)
        stress_target = float(
            min(
                0.95,
                max(
                    global_share + 0.20,
                    baseline_share + 0.05,
                    0.60,
                ),
            )
        )
        selected_stress, summary_stress = optimizer(
            scored_df,
            budget=budget,
            fairness_tolerance=fairness_tolerance,
            fairness_target_override=stress_target,
            min_county_coverage=min_county_coverage,
        )
        if not selected_stress.empty:
            selected_fair, summary_fair = selected_stress, summary_stress
            fairness_mode = "stress_override"

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
    if "with_fairness_spend" not in county_table.columns:
        county_table["with_fairness_spend"] = 0.0
    if "without_fairness_spend" not in county_table.columns:
        county_table["without_fairness_spend"] = 0.0
    if "county" not in county_table.columns:
        first_col = county_table.columns[0]
        county_table = county_table.rename(columns={first_col: "county"})
    county_table["spend_delta"] = county_table["with_fairness_spend"] - county_table["without_fairness_spend"]
    county_table = county_table.sort_values("county").reset_index(drop=True)

    return {
        "effective_optimizer": effective_optimizer,
        "fairness_mode": fairness_mode,
        "stress_target": stress_target,
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


def _selected_with_summary(
    scored_df: pd.DataFrame,
    *,
    budget: float,
    fairness_tolerance: float,
    min_county_coverage: int,
    optimizer_method: str,
) -> tuple[pd.DataFrame, Any]:
    effective_optimizer = optimizer_method
    if len(scored_df) > 1000 and optimizer_method == "ilp":
        effective_optimizer = "greedy"
    optimizer = optimize_replacement_plan_ilp if effective_optimizer == "ilp" else optimize_replacement_plan
    return optimizer(
        scored_df,
        budget=budget,
        fairness_tolerance=fairness_tolerance,
        min_county_coverage=min_county_coverage,
    )


def _policy_briefs_from_selected(selected_df: pd.DataFrame) -> dict[str, str]:
    briefs: dict[str, str] = {}
    if selected_df.empty:
        return briefs
    for _, row in selected_df.sort_values("priority_rank", ascending=True).iterrows():
        geoid = str(row["geoid"])
        raw_drivers = row.get("top_drivers", [])
        drivers: list[tuple[str, float]] = []
        if isinstance(raw_drivers, list):
            for item in raw_drivers:
                if isinstance(item, dict):
                    name = str(item.get("feature", "unknown_feature"))
                    score = float(pd.to_numeric(item.get("score", 0.0), errors="coerce"))
                    drivers.append((name, score))
        if not drivers:
            drivers = [("risk_score", float(pd.to_numeric(row.get("risk_score", 0.0), errors="coerce")))]
        rank_value = pd.to_numeric(row.get("priority_rank", 0), errors="coerce")
        rank_int = int(rank_value) if pd.notna(rank_value) else 0
        briefs[geoid] = generate_policy_brief(
            geoid=geoid,
            county=str(row.get("county", "Unknown")),
            municipality=str(row.get("municipality", geoid)),
            risk_score=float(pd.to_numeric(row.get("risk_score", 0.0), errors="coerce")),
            uncertainty_std=float(pd.to_numeric(row.get("risk_uncertainty", 0.0), errors="coerce")),
            top_drivers=drivers,
            replacement_rank=rank_int,
            replacement_cost=float(pd.to_numeric(row.get("replacement_cost", 0.0), errors="coerce")),
        )
    return briefs


@lru_cache(maxsize=1)
def _build_scored_state_cache() -> dict[str, Any]:
    dataset_path = RESEARCH_DATA_PATH if RESEARCH_DATA_PATH.exists() else DEFAULT_DATA_PATH
    base_df = build_feature_table(dataset_path)
    large_mode = len(base_df) >= 1000

    snapshot = build_demo_snapshot(
        base_df,
        budget=2000000.0,
        fairness_tolerance=0.05,
        min_county_coverage=0,
        optimizer_method="greedy",
        baseline_epochs=35 if large_mode else 700,
        baseline_learning_rate=0.08 if large_mode else 0.1,
        ensemble_models=2 if large_mode else 12,
        ensemble_epochs=25 if large_mode else 350,
        ensemble_learning_rate=0.08 if large_mode else 0.1,
    )

    scored = snapshot.scored_df.copy()
    scored["geoid"] = scored["geoid"].astype(str)
    scored["risk_band"] = scored["risk_score"].map(lambda x: _risk_band(float(x)))
    scored["top_drivers"] = scored.apply(_fast_top_drivers, axis=1)

    trend_cols = [col for col in scored.columns if col.startswith("q") and col.endswith("_lead_ppb")]
    trend_cols = sorted(trend_cols, key=lambda c: int(c[1:].split("_")[0])) if trend_cols else []
    scored["lead_trend"] = (
        scored[trend_cols].apply(
            lambda row: [float(pd.to_numeric(v, errors="coerce")) for v in row.tolist()],
            axis=1,
        )
        if trend_cols
        else [[] for _ in range(len(scored))]
    )
    scored = scored.sort_values("risk_score", ascending=False).reset_index(drop=True)

    comparison_metrics = compute_model_vs_historical_metrics(scored, model_threshold=0.5, ece_bins=5)
    counties = sorted({str(c) for c in scored["county"].dropna().astype(str).tolist() if str(c).strip()})

    return {
        "dataset_path": str(dataset_path),
        "dataset_rows": int(len(base_df)),
        "scored_df": scored,
        "comparison_metrics": _normalize_value(comparison_metrics),
        "available_counties": counties,
    }


def build_dashboard_payload(
    *,
    budget: float = 35000.0,
    fairness_tolerance: float = 0.05,
    min_county_coverage: int = 0,
    optimizer_method: str = "ilp",
) -> dict[str, Any]:
    scored_state = _build_scored_state_cache()
    scored = scored_state["scored_df"].copy()
    selected_df, optimization_summary = _selected_with_summary(
        scored,
        budget=budget,
        fairness_tolerance=fairness_tolerance,
        min_county_coverage=min_county_coverage,
        optimizer_method=optimizer_method,
    )
    policy_briefs = _policy_briefs_from_selected(selected_df)

    fairness = _compute_fairness_comparison(
        scored,
        budget=budget,
        fairness_tolerance=fairness_tolerance,
        min_county_coverage=min_county_coverage,
        optimizer_method=optimizer_method,
    )

    benchmark = build_benchmark_payload()
    return {
        "params": {
            "budget": float(budget),
            "fairness_tolerance": float(fairness_tolerance),
            "min_county_coverage": int(min_county_coverage),
            "optimizer_method": str(optimizer_method),
        },
        "dataset_path": str(scored_state["dataset_path"]),
        "dataset_rows": int(scored_state["dataset_rows"]),
        "available_counties": list(scored_state["available_counties"]),
        "rows": _serialize_df(scored),
        "selected_rows": _serialize_df(selected_df),
        "optimization_summary": _normalize_value(optimization_summary),
        "policy_briefs": _normalize_value(policy_briefs),
        "comparison_metrics": _normalize_value(scored_state["comparison_metrics"]),
        "cv_metrics": {
            "historical_accuracy_mean": float(benchmark["historical"]["accuracy"]["mean"]),
            "fusion_accuracy_mean": float(benchmark["fusion"]["accuracy"]["mean"]),
            "graph_accuracy_mean": float(benchmark["graph"]["accuracy"]["mean"]),
            "fusion_auroc_mean": float(benchmark["fusion"]["auroc"]["mean"]),
            "fusion_auprc_mean": float(benchmark["fusion"]["auprc"]["mean"]),
        },
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


@lru_cache(maxsize=64)
def _cached_dashboard_payload(
    budget: float,
    fairness_tolerance: float,
    min_county_coverage: int,
    optimizer_method: str,
) -> dict[str, Any]:
    return build_dashboard_payload(
        budget=budget,
        fairness_tolerance=fairness_tolerance,
        min_county_coverage=min_county_coverage,
        optimizer_method=optimizer_method,
    )


def _apply_row_scope(
    payload: dict[str, Any],
    *,
    county: str | None,
    row_limit: int,
) -> dict[str, Any]:
    rows = payload.get("rows", [])
    selected_rows = payload.get("selected_rows", [])
    if county:
        county_lc = county.strip().lower()
        rows = [row for row in rows if str(row.get("county", "")).strip().lower() == county_lc]
        selected_rows = [row for row in selected_rows if str(row.get("county", "")).strip().lower() == county_lc]
    rows_total = len(rows)
    if row_limit > 0 and rows_total > row_limit:
        rows = rows[:row_limit]
    selected_geoids = {str(row.get("geoid")) for row in selected_rows}
    policy_briefs_all = payload.get("policy_briefs", {})
    scoped_policy_briefs = {k: v for k, v in policy_briefs_all.items() if str(k) in selected_geoids}

    scoped = dict(payload)
    scoped["rows"] = rows
    scoped["selected_rows"] = selected_rows
    scoped["policy_briefs"] = scoped_policy_briefs
    scoped["rows_total_available"] = rows_total
    scoped["rows_returned"] = len(rows)
    scoped["row_limit_applied"] = int(row_limit)
    scoped["row_scope_county"] = county
    return scoped


@app.get("/api/health")
def api_health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/benchmark")
def api_benchmark() -> dict[str, Any]:
    return build_benchmark_payload()


@app.get("/api/ai/status")
def api_ai_status() -> dict[str, Any]:
    return {
        "enabled": bool(is_ai_enabled()),
        "model": configured_model(),
    }


@app.post("/api/ai/copilot")
def api_ai_copilot(request: AICopilotRequest) -> dict[str, Any]:
    payload = _cached_dashboard_payload(
        budget=float(round(request.budget, 4)),
        fairness_tolerance=float(round(request.fairness_tolerance, 4)),
        min_county_coverage=int(request.min_county_coverage),
        optimizer_method=str(request.optimizer_method),
    )

    geoid = str(request.geoid).strip()
    rows = payload.get("rows", [])
    row = next((item for item in rows if str(item.get("geoid")) == geoid), None)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Area not found for geoid={geoid}")

    selected_geoids = {str(item.get("geoid")) for item in payload.get("selected_rows", [])}
    fairness_summary = (
        payload.get("fairness_comparison", {}).get("with_fairness", {}).get("summary", {})
    )
    optimization_summary = payload.get("optimization_summary", {})
    result = generate_block_answer(
        block_row=row,
        question=request.question,
        selected=geoid in selected_geoids,
        fairness_summary=fairness_summary,
        optimization_summary=optimization_summary,
    )
    return {
        "geoid": geoid,
        **result,
    }


@app.get("/api/dashboard")
def api_dashboard(
    budget: float = Query(default=2000000.0, ge=10000.0, le=100000000.0),
    fairness_tolerance: float = Query(default=0.05, ge=0.0, le=1.0),
    min_county_coverage: int = Query(default=0, ge=0, le=10),
    optimizer_method: str = Query(default="greedy", pattern="^(ilp|greedy)$"),
    county: str | None = Query(default=None, max_length=100),
    row_limit: int = Query(default=1200, ge=100, le=7000),
) -> dict[str, Any]:
    payload = _cached_dashboard_payload(
        budget=float(round(budget, 4)),
        fairness_tolerance=float(round(fairness_tolerance, 4)),
        min_county_coverage=int(min_county_coverage),
        optimizer_method=str(optimizer_method),
    )
    county_filter = county.strip() if county and county.strip() and county.strip().lower() != "all" else None
    return _apply_row_scope(payload, county=county_filter, row_limit=int(row_limit))


@app.get("/")
def root() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
