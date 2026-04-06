from __future__ import annotations

import json
import os
import re
from typing import Any

import pandas as pd

DEFAULT_MODEL = "gpt-4.1-mini"
OBJECTIVE_WEIGHTS_KEYS = ("risk_reduction", "equity", "cost_efficiency", "certainty")


def configured_model() -> str:
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip()
    return model or DEFAULT_MODEL


def is_ai_enabled() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def _to_float(value: Any, default: float = 0.0) -> float:
    cast = pd.to_numeric(value, errors="coerce")
    if pd.isna(cast):
        return default
    return float(cast)


def _to_int(value: Any, default: int = 0) -> int:
    cast = pd.to_numeric(value, errors="coerce")
    if pd.isna(cast):
        return default
    return int(cast)


def _fallback_answer(
    *,
    block_row: dict[str, Any],
    question: str,
    selected: bool,
    fairness_summary: dict[str, Any],
    optimization_summary: dict[str, Any],
) -> str:
    municipality = str(block_row.get("municipality") or block_row.get("geoid"))
    county = str(block_row.get("county") or "Unknown")
    risk_pct = _to_float(block_row.get("risk_score")) * 100.0
    uncertainty = _to_float(block_row.get("risk_uncertainty"))
    cost = _to_float(block_row.get("replacement_cost"))
    minority_pct = _to_float(block_row.get("minority_share")) * 100.0
    rank = _to_int(block_row.get("priority_rank"), default=-1)
    selected_text = "Yes" if selected else "No"
    fairness_minority = _to_float(fairness_summary.get("achieved_minority_share")) * 100.0
    fairness_selected = _to_int(fairness_summary.get("selected_count"))
    budget_spent = _to_float(optimization_summary.get("total_cost"))
    budget_total = _to_float(optimization_summary.get("budget"))
    drivers = block_row.get("top_drivers") or []
    driver_text = ", ".join(
        f"{str(item.get('feature', 'unknown'))} ({_to_float(item.get('score')):+.3f})"
        for item in drivers[:3]
        if isinstance(item, dict)
    )
    if not driver_text:
        driver_text = "no driver details available"

    rank_text = f"{rank}" if rank > 0 else "not selected"
    return (
        "AI fallback mode.\n\n"
        f"Question: {question}\n\n"
        f"Area: {municipality}, {county} County ({block_row.get('geoid')})\n"
        f"Selected for replacement: {selected_text} (priority rank: {rank_text})\n"
        f"Predicted risk: {risk_pct:.1f}% (+/- {uncertainty:.3f})\n"
        f"Estimated replacement cost: ${cost:,.0f}\n"
        f"Minority share: {minority_pct:.1f}%\n"
        f"Top drivers: {driver_text}\n\n"
        f"Current plan budget usage: ${budget_spent:,.0f} / ${budget_total:,.0f}\n"
        f"Fairness plan selects {fairness_selected} areas with achieved minority share {fairness_minority:.1f}%.\n"
    )


def _extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    chunks: list[str] = []
    output = getattr(response, "output", None)
    if isinstance(output, list):
        for item in output:
            content = getattr(item, "content", None)
            if content is None and isinstance(item, dict):
                content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                text = getattr(part, "text", None)
                if text is None and isinstance(part, dict):
                    text = part.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
    return "\n".join(chunks).strip()


def generate_block_answer(
    *,
    block_row: dict[str, Any],
    question: str,
    selected: bool,
    fairness_summary: dict[str, Any],
    optimization_summary: dict[str, Any],
) -> dict[str, Any]:
    question_clean = str(question or "").strip()
    if not question_clean:
        question_clean = "Give a concise risk explanation and action plan for this area."

    fallback = _fallback_answer(
        block_row=block_row,
        question=question_clean,
        selected=selected,
        fairness_summary=fairness_summary,
        optimization_summary=optimization_summary,
    )
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = configured_model()
    if not api_key:
        return {
            "ai_enabled": False,
            "ai_used": False,
            "model": None,
            "answer": fallback,
            "fallback_reason": "OPENAI_API_KEY is not set.",
        }

    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - import failure path
        return {
            "ai_enabled": True,
            "ai_used": False,
            "model": model,
            "answer": fallback,
            "fallback_reason": f"OpenAI SDK unavailable: {exc}",
        }

    municipality = str(block_row.get("municipality") or block_row.get("geoid"))
    county = str(block_row.get("county") or "Unknown")
    selected_text = "Yes" if selected else "No"
    context = {
        "geoid": str(block_row.get("geoid")),
        "municipality": municipality,
        "county": county,
        "risk_score": _to_float(block_row.get("risk_score")),
        "risk_uncertainty": _to_float(block_row.get("risk_uncertainty")),
        "replacement_cost": _to_float(block_row.get("replacement_cost")),
        "minority_share": _to_float(block_row.get("minority_share")),
        "selected_for_replacement": selected_text,
        "priority_rank": _to_int(block_row.get("priority_rank"), default=0),
        "top_drivers": block_row.get("top_drivers") or [],
        "lead_trend": block_row.get("lead_trend") or [],
        "fairness_summary": fairness_summary,
        "optimization_summary": optimization_summary,
    }

    system_prompt = (
        "You are LeadSense NJ Copilot. Answer with concrete public-health planning guidance grounded only in the "
        "provided area context. If the user asks for actions, split into immediate (0-3 months) and medium-term "
        "(3-12 months). Keep response concise and factual."
    )
    user_prompt = (
        f"User question: {question_clean}\n\n"
        "Area context (JSON):\n"
        f"{context}\n\n"
        "Return a concise answer with: 1) explanation of risk factors, 2) recommended actions, "
        "3) any caveat about model uncertainty/fairness."
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            temperature=0.2,
            max_output_tokens=550,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = _extract_output_text(response)
        if not answer:
            raise RuntimeError("Model returned an empty response.")
        return {
            "ai_enabled": True,
            "ai_used": True,
            "model": model,
            "answer": answer,
            "fallback_reason": None,
        }
    except Exception as exc:
        return {
            "ai_enabled": True,
            "ai_used": False,
            "model": model,
            "answer": fallback,
            "fallback_reason": f"OpenAI request failed: {exc}",
        }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _normalize_objective_weights(raw: dict[str, Any] | None) -> dict[str, float]:
    base = {key: 0.0 for key in OBJECTIVE_WEIGHTS_KEYS}
    if isinstance(raw, dict):
        for key in OBJECTIVE_WEIGHTS_KEYS:
            base[key] = max(0.0, _to_float(raw.get(key), default=0.0))
    total = sum(base.values())
    if total <= 0:
        base = {
            "risk_reduction": 0.55,
            "equity": 0.2,
            "cost_efficiency": 0.15,
            "certainty": 0.1,
        }
        total = sum(base.values())
    return {k: float(v / total) for k, v in base.items()}


def _heuristic_objective_profile(
    *,
    prompt: str,
    available_counties: list[str],
    default_fairness_tolerance: float,
) -> dict[str, Any]:
    text = str(prompt or "").strip().lower()
    weights = {
        "risk_reduction": 0.55,
        "equity": 0.2,
        "cost_efficiency": 0.15,
        "certainty": 0.1,
    }
    fairness_tolerance = float(default_fairness_tolerance)
    min_county_coverage = 0

    if any(token in text for token in ("equity", "fairness", "underserved", "minority", "justice")):
        weights["equity"] += 0.22
        weights["risk_reduction"] -= 0.08
        fairness_tolerance = min(fairness_tolerance, 0.03)
    if any(token in text for token in ("cost", "budget", "efficient", "roi", "value")):
        weights["cost_efficiency"] += 0.2
        weights["risk_reduction"] -= 0.05
    if any(token in text for token in ("certainty", "confidence", "reliable", "defensible")):
        weights["certainty"] += 0.15
        weights["risk_reduction"] -= 0.05
    if any(token in text for token in ("high risk", "urgent", "hotspot", "max risk")):
        weights["risk_reduction"] += 0.2
    if any(token in text for token in ("statewide", "all counties", "every county", "county coverage")):
        min_county_coverage = 1

    focus_counties: list[str] = []
    for county in available_counties:
        if county.lower() in text:
            focus_counties.append(county)

    return {
        "goal_label": "Heuristic objective profile",
        "weights": _normalize_objective_weights(weights),
        "fairness_tolerance": max(0.0, min(0.2, fairness_tolerance)),
        "min_county_coverage": int(max(0, min(3, min_county_coverage))),
        "focus_counties": focus_counties,
        "rationale": "Objective derived with rule-based parser because AI model output was unavailable.",
    }


def _sanitize_objective(
    data: dict[str, Any],
    *,
    available_counties: list[str],
    default_fairness_tolerance: float,
) -> dict[str, Any]:
    goal_label = str(data.get("goal_label") or "AI objective profile").strip()[:80]
    fairness_tolerance = _to_float(data.get("fairness_tolerance"), default_fairness_tolerance)
    fairness_tolerance = max(0.0, min(0.2, fairness_tolerance))
    min_county_coverage = _to_int(data.get("min_county_coverage"), default=0)
    min_county_coverage = int(max(0, min(3, min_county_coverage)))
    weights = _normalize_objective_weights(data.get("weights") if isinstance(data.get("weights"), dict) else None)
    focus_raw = data.get("focus_counties")
    focus_counties = []
    if isinstance(focus_raw, list):
        allowed = {county.lower(): county for county in available_counties}
        for item in focus_raw:
            county = str(item).strip()
            match = allowed.get(county.lower())
            if match and match not in focus_counties:
                focus_counties.append(match)
    rationale = str(data.get("rationale") or "Structured objective generated from natural language policy goal.")
    return {
        "goal_label": goal_label,
        "weights": weights,
        "fairness_tolerance": fairness_tolerance,
        "min_county_coverage": min_county_coverage,
        "focus_counties": focus_counties,
        "rationale": rationale,
    }


def generate_portfolio_objective(
    *,
    prompt: str,
    available_counties: list[str],
    default_fairness_tolerance: float = 0.05,
) -> dict[str, Any]:
    prompt_clean = str(prompt or "").strip()
    if not prompt_clean:
        prompt_clean = "Prioritize highest-risk areas while maintaining fairness across communities."

    fallback_objective = _heuristic_objective_profile(
        prompt=prompt_clean,
        available_counties=available_counties,
        default_fairness_tolerance=default_fairness_tolerance,
    )
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = configured_model()
    if not api_key:
        return {
            "ai_enabled": False,
            "ai_used": False,
            "model": None,
            "objective": fallback_objective,
            "fallback_reason": "OPENAI_API_KEY is not set.",
        }

    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - import failure path
        return {
            "ai_enabled": True,
            "ai_used": False,
            "model": model,
            "objective": fallback_objective,
            "fallback_reason": f"OpenAI SDK unavailable: {exc}",
        }

    system_prompt = (
        "You are LeadSense NJ Portfolio AI. Convert policy goals to optimization objective settings. "
        "Return ONLY strict JSON with fields: goal_label, weights, fairness_tolerance, min_county_coverage, "
        "focus_counties, rationale. weights must include risk_reduction, equity, cost_efficiency, certainty."
    )
    user_prompt = (
        f"Policy goal:\n{prompt_clean}\n\n"
        f"Available NJ counties:\n{available_counties}\n\n"
        "Rules:\n"
        "- fairness_tolerance in [0, 0.2]\n"
        "- min_county_coverage in [0, 3]\n"
        "- focus_counties must be subset of available counties\n"
        "- weights can be any non-negative numbers\n"
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            temperature=0.1,
            max_output_tokens=500,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        output_text = _extract_output_text(response)
        parsed = _extract_json_object(output_text)
        if not parsed:
            raise RuntimeError("Could not parse JSON objective profile from model response.")
        objective = _sanitize_objective(
            parsed,
            available_counties=available_counties,
            default_fairness_tolerance=default_fairness_tolerance,
        )
        return {
            "ai_enabled": True,
            "ai_used": True,
            "model": model,
            "objective": objective,
            "fallback_reason": None,
        }
    except Exception as exc:
        return {
            "ai_enabled": True,
            "ai_used": False,
            "model": model,
            "objective": fallback_objective,
            "fallback_reason": f"OpenAI request failed: {exc}",
        }
