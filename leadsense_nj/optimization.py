from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class OptimizationSummary:
    selected_count: int
    total_cost: float
    total_risk_reduced: float
    budget: float
    fairness_target: float
    achieved_minority_share: float


def _can_meet_fairness(
    minority_sum: float,
    selected_count: int,
    candidate_minority: float,
    fairness_target: float,
) -> bool:
    next_count = selected_count + 1
    next_share = (minority_sum + candidate_minority) / next_count
    return next_share >= fairness_target


def optimize_replacement_plan(
    df: pd.DataFrame,
    budget: float,
    *,
    fairness_tolerance: float = 0.05,
    min_county_coverage: int = 0,
    risk_col: str = "risk_score",
    cost_col: str = "replacement_cost",
    minority_col: str = "minority_share",
    county_col: str = "county",
    geoid_col: str = "geoid",
) -> tuple[pd.DataFrame, OptimizationSummary]:
    required = {risk_col, cost_col, minority_col, county_col, geoid_col}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required optimization columns: {missing}")
    if budget <= 0:
        raise ValueError("Budget must be > 0")
    if min_county_coverage < 0:
        raise ValueError("min_county_coverage must be >= 0")

    work = df.copy()
    work[risk_col] = pd.to_numeric(work[risk_col], errors="raise")
    work[cost_col] = pd.to_numeric(work[cost_col], errors="raise")
    work[minority_col] = pd.to_numeric(work[minority_col], errors="raise")
    if (work[cost_col] <= 0).any():
        raise ValueError("All replacement costs must be > 0.")

    work["risk_per_dollar"] = work[risk_col] / work[cost_col]
    work = work.sort_values("risk_per_dollar", ascending=False).reset_index(drop=True)

    global_minority_share = float(work[minority_col].mean())
    fairness_target = max(0.0, global_minority_share - fairness_tolerance)

    selected_rows: list[pd.Series] = []
    selected_geoids: set[str] = set()
    total_cost = 0.0
    minority_sum = 0.0

    def try_add(row: pd.Series, enforce_fairness: bool) -> bool:
        nonlocal total_cost, minority_sum
        geoid = str(row[geoid_col])
        row_cost = float(row[cost_col])
        row_minority = float(row[minority_col])
        if geoid in selected_geoids:
            return False
        if total_cost + row_cost > budget:
            return False
        if enforce_fairness and not _can_meet_fairness(minority_sum, len(selected_rows), row_minority, fairness_target):
            return False

        selected_geoids.add(geoid)
        selected_rows.append(row)
        total_cost += row_cost
        minority_sum += row_minority
        return True

    # Phase 1: Optional county coverage seed.
    if min_county_coverage > 0:
        for county, county_df in work.groupby(county_col):
            added = 0
            for _, row in county_df.sort_values("risk_per_dollar", ascending=False).iterrows():
                if added >= min_county_coverage:
                    break
                if try_add(row, enforce_fairness=False):
                    added += 1

    # Phase 2: Greedy fill with fairness constraint.
    for _, row in work.iterrows():
        try_add(row, enforce_fairness=True)

    selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    if selected_df.empty:
        summary = OptimizationSummary(
            selected_count=0,
            total_cost=0.0,
            total_risk_reduced=0.0,
            budget=float(budget),
            fairness_target=fairness_target,
            achieved_minority_share=0.0,
        )
        return selected_df, summary

    selected_df["priority_rank"] = selected_df[risk_col].rank(ascending=False, method="first").astype(int)
    selected_df = selected_df.sort_values("priority_rank").reset_index(drop=True)

    achieved_share = float(selected_df[minority_col].mean())
    summary = OptimizationSummary(
        selected_count=int(len(selected_df)),
        total_cost=float(selected_df[cost_col].sum()),
        total_risk_reduced=float(selected_df[risk_col].sum()),
        budget=float(budget),
        fairness_target=fairness_target,
        achieved_minority_share=achieved_share,
    )
    return selected_df, summary
