from __future__ import annotations

import pandas as pd

from leadsense_nj.optimization import optimize_replacement_plan, optimize_replacement_plan_ilp


def _candidate_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "geoid": [
                "340010001001",
                "340030010002",
                "340050023001",
                "340070045003",
                "340090112001",
                "340110008002",
            ],
            "county": ["Atlantic", "Bergen", "Burlington", "Camden", "Cape May", "Cumberland"],
            "risk_score": [0.90, 0.35, 0.28, 0.95, 0.40, 0.78],
            "replacement_cost": [13000, 9000, 8500, 14000, 9200, 12000],
            "minority_share": [0.72, 0.20, 0.18, 0.84, 0.23, 0.66],
        }
    )


def test_optimize_replacement_plan_respects_budget() -> None:
    selected, summary = optimize_replacement_plan(_candidate_df(), budget=30000)
    assert summary.total_cost <= 30000
    assert len(selected) == summary.selected_count


def test_optimize_replacement_plan_tracks_fairness() -> None:
    selected, summary = optimize_replacement_plan(_candidate_df(), budget=36000, fairness_tolerance=0.02)
    if len(selected) > 0:
        assert summary.achieved_minority_share >= summary.fairness_target


def test_optimize_replacement_plan_county_coverage_seed() -> None:
    selected, _ = optimize_replacement_plan(_candidate_df(), budget=25000, min_county_coverage=1)
    # Budget is limited, but at least one county should be represented.
    assert selected["county"].nunique() >= 1


def test_optimize_replacement_plan_drops_non_finite_rows() -> None:
    df = _candidate_df()
    df["replacement_cost"] = df["replacement_cost"].astype(float)
    df.loc[0, "risk_score"] = float("nan")
    df.loc[1, "replacement_cost"] = float("inf")
    selected, summary = optimize_replacement_plan(df, budget=35000)
    assert summary.total_cost <= 35000
    assert selected["risk_score"].isna().sum() == 0


def test_optimize_replacement_plan_ilp_respects_budget() -> None:
    selected, summary = optimize_replacement_plan_ilp(_candidate_df(), budget=35000, fairness_tolerance=0.05)
    assert summary.total_cost <= 35000
    assert len(selected) == summary.selected_count
