from __future__ import annotations

from leadsense_nj.ai_assistant import generate_block_answer


def test_generate_block_answer_fallback_without_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    row = {
        "geoid": "340130001001",
        "municipality": "Test Township",
        "county": "Mercer",
        "risk_score": 0.73,
        "risk_uncertainty": 0.08,
        "replacement_cost": 27500,
        "minority_share": 0.42,
        "priority_rank": 3,
        "top_drivers": [
            {"feature": "lead_90p_ppb", "score": 0.9},
            {"feature": "pct_housing_pre_1950", "score": 0.6},
        ],
    }
    result = generate_block_answer(
        block_row=row,
        question="What should we do first here?",
        selected=True,
        fairness_summary={"achieved_minority_share": 0.51, "selected_count": 120},
        optimization_summary={"total_cost": 1900000, "budget": 2000000},
    )
    assert result["ai_enabled"] is False
    assert result["ai_used"] is False
    assert "fallback" in result["answer"].lower()
    assert "Test Township" in result["answer"]
