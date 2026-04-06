from __future__ import annotations

import os
from typing import Any

import pandas as pd

DEFAULT_MODEL = "gpt-4.1-mini"


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
