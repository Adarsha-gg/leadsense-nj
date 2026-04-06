from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leadsense_nj.config import DataConfig
from leadsense_nj.baseline import fit_tabular_logistic
from leadsense_nj.demo import build_demo_snapshot
from leadsense_nj.explainability import top_feature_drivers
from leadsense_nj.optimization import optimize_replacement_plan
from leadsense_nj.policy_brief import generate_policy_brief
from leadsense_nj.preprocessing import build_feature_table
from leadsense_nj.target import with_elevated_risk_label
from leadsense_nj.uncertainty import expected_calibration_error, train_bootstrap_ensemble


def run_feature_01_checks() -> None:
    df = build_feature_table()
    if df.empty:
        raise RuntimeError("F01 failed: feature table is empty.")

    print("F01 checks passed.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Source: {DataConfig().default_feature_table_path}")


def run_feature_02_checks() -> None:
    df = build_feature_table()
    labeled = with_elevated_risk_label(df)
    if "risk_label" not in labeled.columns:
        raise RuntimeError("F02 failed: risk_label column missing.")
    if not labeled["risk_label"].isin([0, 1]).all():
        raise RuntimeError("F02 failed: risk_label contains values other than 0/1.")
    prevalence = float(labeled["risk_label"].mean())
    if prevalence <= 0.0 or prevalence >= 1.0:
        raise RuntimeError("F02 failed: label prevalence collapsed to all-zero or all-one.")

    print("F02 checks passed.")
    print(f"Positive label prevalence: {prevalence:.3f}")


def run_feature_03_checks() -> None:
    df = with_elevated_risk_label(build_feature_table())
    model, losses = fit_tabular_logistic(df, epochs=600, learning_rate=0.1)
    if losses[-1] >= losses[0]:
        raise RuntimeError("F03 failed: training loss did not decrease.")

    proba = model.predict_proba(df)
    if not ((proba >= 0.0) & (proba <= 1.0)).all():
        raise RuntimeError("F03 failed: model produced out-of-bound probabilities.")

    preds = model.predict(df)
    accuracy = float((preds == df["risk_label"]).mean())
    if accuracy < 0.65:
        raise RuntimeError(f"F03 failed: training accuracy too low ({accuracy:.3f}).")

    print("F03 checks passed.")
    print(f"Training accuracy: {accuracy:.3f}")


def run_feature_04_checks() -> None:
    df = with_elevated_risk_label(build_feature_table())
    ensemble = train_bootstrap_ensemble(df, n_models=12, epochs=350, learning_rate=0.1, seed=11)
    mean, std = ensemble.predict_mean_std(df)
    ece = expected_calibration_error(df["risk_label"].to_numpy(), mean, n_bins=5)

    if not ((mean >= 0.0) & (mean <= 1.0)).all():
        raise RuntimeError("F04 failed: uncertainty mean predictions out of range [0, 1].")
    if not (std >= 0.0).all():
        raise RuntimeError("F04 failed: uncertainty std values below 0.")
    if float(std.mean()) <= 0.0:
        raise RuntimeError("F04 failed: uncertainty collapsed to zero across all rows.")
    if not (0.0 <= ece <= 1.0):
        raise RuntimeError(f"F04 failed: ECE out of valid range: {ece:.3f}")

    print("F04 checks passed.")
    print(f"Average uncertainty std: {float(std.mean()):.4f}")
    print(f"ECE: {ece:.4f}")


def run_feature_05_checks() -> None:
    df = with_elevated_risk_label(build_feature_table())
    model, _ = fit_tabular_logistic(df, epochs=700, learning_rate=0.1)
    scored = df.copy()
    scored["risk_score"] = model.predict_proba(scored)
    scored["replacement_cost"] = 10000 + (scored["pct_housing_pre_1950"] * 8000) + (scored["lead_90p_ppb"] * 200)
    scored["minority_share"] = scored["poverty_rate"].clip(0.0, 1.0)

    selected, summary = optimize_replacement_plan(
        scored,
        budget=35000,
        fairness_tolerance=0.05,
        min_county_coverage=0,
    )

    if summary.total_cost > summary.budget + 1e-6:
        raise RuntimeError("F05 failed: optimization exceeded budget.")
    if summary.selected_count <= 0:
        raise RuntimeError("F05 failed: optimization selected no block groups.")
    if summary.achieved_minority_share < summary.fairness_target - 1e-6:
        raise RuntimeError("F05 failed: fairness target not achieved.")

    print("F05 checks passed.")
    print(f"Selected blocks: {summary.selected_count}")
    print(f"Total cost: {summary.total_cost:.2f} / Budget: {summary.budget:.2f}")


def run_feature_06_checks() -> None:
    df = with_elevated_risk_label(build_feature_table())
    model, _ = fit_tabular_logistic(df, epochs=700, learning_rate=0.1)
    ensemble = train_bootstrap_ensemble(df, n_models=12, epochs=300, learning_rate=0.1, seed=17)
    mean, std = ensemble.predict_mean_std(df)

    row = df.iloc[0]
    drivers = top_feature_drivers(model, row, top_k=3)
    brief = generate_policy_brief(
        geoid=str(row["geoid"]),
        county=str(row["county"]),
        municipality=str(row["municipality"]),
        risk_score=float(mean[0]),
        uncertainty_std=float(std[0]),
        top_drivers=drivers,
        replacement_rank=1,
        replacement_cost=12500.0,
    )
    if len(drivers) != 3:
        raise RuntimeError("F06 failed: top driver extraction did not return 3 drivers.")
    if "Immediate action" not in brief or "Long-term action" not in brief:
        raise RuntimeError("F06 failed: policy brief is missing required action sections.")
    if len(brief) < 250:
        raise RuntimeError("F06 failed: policy brief too short to be actionable.")

    print("F06 checks passed.")
    print(f"Policy brief length: {len(brief)} characters")


def run_feature_07_checks() -> None:
    df = build_feature_table()
    snapshot = build_demo_snapshot(df, budget=35000, fairness_tolerance=0.05, min_county_coverage=0)
    if snapshot.scored_df.empty:
        raise RuntimeError("F07 failed: demo snapshot has empty scored table.")
    if "risk_score" not in snapshot.scored_df.columns:
        raise RuntimeError("F07 failed: risk_score missing in demo snapshot.")
    if snapshot.optimization_summary.total_cost > snapshot.optimization_summary.budget + 1e-6:
        raise RuntimeError("F07 failed: demo snapshot optimization exceeds budget.")

    print("F07 checks passed.")
    print(f"Snapshot rows: {len(snapshot.scored_df)}")
    print(f"Policy briefs: {len(snapshot.policy_briefs)}")


if __name__ == "__main__":
    run_feature_01_checks()
    run_feature_02_checks()
    run_feature_03_checks()
    run_feature_04_checks()
    run_feature_05_checks()
    run_feature_06_checks()
    run_feature_07_checks()
