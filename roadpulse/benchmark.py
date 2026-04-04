from __future__ import annotations

import pandas as pd


def compute_kpis(detections_df: pd.DataFrame, priority_df: pd.DataFrame) -> dict[str, float]:
    if detections_df.empty:
        return {
            "manual_inspection_hours": 0.0,
            "ai_inspection_hours": 0.0,
            "hours_saved": 0.0,
            "estimated_repairs_cost": 0.0,
            "estimated_vehicle_damage_avoided": 0.0,
            "roi_ratio": 0.0,
            "co2_kg_avoided": 0.0,
        }

    detections = len(detections_df)
    manual_hours = detections * 0.14
    ai_hours = detections * 0.03
    hours_saved = max(0.0, manual_hours - ai_hours)

    repair_cost = 0.0
    if not priority_df.empty:
        repair_cost = float(priority_df["estimated_repair_cost_usd"].sum())

    # Lightweight proxy assumptions for demo economics.
    vehicle_damage_avoided = float(detections_df["risk_score"].mean() * detections * 2.8)
    roi = 0.0 if repair_cost <= 0 else vehicle_damage_avoided / repair_cost

    # Co2 proxy from avoided congestion + smoother travel.
    co2_kg_avoided = float(detections_df["severity"].mean() * detections * 0.11)

    return {
        "manual_inspection_hours": round(manual_hours, 2),
        "ai_inspection_hours": round(ai_hours, 2),
        "hours_saved": round(hours_saved, 2),
        "estimated_repairs_cost": round(repair_cost, 2),
        "estimated_vehicle_damage_avoided": round(vehicle_damage_avoided, 2),
        "roi_ratio": round(roi, 2),
        "co2_kg_avoided": round(co2_kg_avoided, 2),
    }


def recommendation_cards(detections_df: pd.DataFrame) -> list[str]:
    if detections_df.empty:
        return ["No detections yet. Run analysis to generate intervention recommendations."]

    mix = detections_df["hazard_class"].value_counts(normalize=True)
    cards: list[str] = []

    if float(mix.get("standing_water", 0.0)) > 0.18:
        cards.append("Drainage-first intervention: prioritize culvert/ditch cleanup and flood-channel fixes.")
    if float(mix.get("pothole", 0.0)) > 0.25:
        cards.append("Rapid-response pothole crews: schedule patching window within 7 days on top-risk segments.")
    if float(mix.get("alligator_crack", 0.0)) > 0.15:
        cards.append("Structural resurfacing candidate: recurring fatigue cracking suggests deeper rehabilitation.")
    if float(mix.get("faded_lane_marking", 0.0)) > 0.20:
        cards.append("Safety striping action: re-mark high-risk corridors and school-adjacent roads.")
    if not cards:
        cards.append("Balanced hazard mix: use budget-constrained queue and monitor trend weekly.")
    return cards

