from __future__ import annotations

from typing import Iterable

import pandas as pd

from leadsense_nj.baseline import TabularBaselineModel


def compute_linear_contributions(model: TabularBaselineModel, df: pd.DataFrame) -> pd.DataFrame:
    standardized = model._transform(df)  # pylint: disable=protected-access
    contrib = standardized * model.weights
    return pd.DataFrame(contrib, columns=model.feature_columns, index=df.index)


def top_feature_drivers(
    model: TabularBaselineModel,
    row: pd.Series,
    *,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    row_df = pd.DataFrame([row])
    contrib = compute_linear_contributions(model, row_df).iloc[0]
    ranked = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
    top = ranked.iloc[:top_k]
    return list(top.items())


def format_driver_lines(drivers: Iterable[tuple[str, float]]) -> str:
    lines = []
    for feature, score in drivers:
        sign = "+" if score >= 0 else "-"
        lines.append(f"- {feature}: {sign}{abs(score):.3f}")
    return "\n".join(lines)
