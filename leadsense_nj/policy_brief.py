from __future__ import annotations

from typing import Iterable


def _risk_band(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.45:
        return "MEDIUM"
    return "LOW"


def generate_policy_brief(
    *,
    geoid: str,
    county: str,
    municipality: str,
    risk_score: float,
    uncertainty_std: float,
    top_drivers: Iterable[tuple[str, float]],
    replacement_rank: int,
    replacement_cost: float,
) -> str:
    drivers_text = "\n".join(f"- {name}: {value:+.3f}" for name, value in top_drivers)
    band = _risk_band(risk_score)

    return (
        f"Policy Brief - Census Block Group {geoid}\n\n"
        f"Location: {municipality}, {county} County, New Jersey.\n"
        f"Risk Assessment: {band} risk ({risk_score:.3f} +/- {uncertainty_std:.3f}).\n"
        f"Priority Rank: #{replacement_rank} with estimated replacement cost ${replacement_cost:,.0f}.\n\n"
        "Primary risk drivers:\n"
        f"{drivers_text}\n\n"
        "Immediate action:\n"
        "1. Schedule targeted lead sampling outreach and confirm high-risk service line inventory.\n"
        "2. Prioritize replacement planning funds for this block group in the next budget window.\n\n"
        "Long-term action:\n"
        "1. Integrate this block group into county-wide replacement sequencing with equity monitoring.\n"
        "2. Track quarterly chemistry metrics and re-score to validate risk reduction after interventions.\n"
    )
