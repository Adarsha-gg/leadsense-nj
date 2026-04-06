from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api_server import build_benchmark_payload, build_dashboard_payload
from leadsense_nj.config import DataConfig
from leadsense_nj.baseline import fit_tabular_logistic
from leadsense_nj.demo import build_demo_snapshot
from leadsense_nj.explainability import top_feature_drivers
from leadsense_nj.ingestion import ensure_real_data_cache, validate_acs_block_group_frame, validate_epa_pws_lead_signal_frame
from leadsense_nj.infrastructure import build_county_proxy_edge_list, load_edge_list, validate_edge_list
from leadsense_nj.optimization import optimize_replacement_plan, optimize_replacement_plan_ilp
from leadsense_nj.policy_brief import generate_policy_brief
from leadsense_nj.preprocessing import build_feature_table
from leadsense_nj.research import run_model_research_benchmark
from leadsense_nj.satellite import ensure_sentinel_feature_cache, validate_sentinel_feature_frame
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

    selected_ilp, summary_ilp = optimize_replacement_plan_ilp(
        scored,
        budget=35000,
        fairness_tolerance=0.05,
        min_county_coverage=0,
    )
    if summary_ilp.total_cost > summary_ilp.budget + 1e-6:
        raise RuntimeError("F05 failed: ILP optimization exceeded budget.")
    print(f"ILP selected blocks: {len(selected_ilp)} | cost: {summary_ilp.total_cost:.2f}")


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
    snapshot = build_demo_snapshot(
        df,
        budget=35000,
        fairness_tolerance=0.05,
        min_county_coverage=0,
        optimizer_method="ilp",
    )
    if snapshot.scored_df.empty:
        raise RuntimeError("F07 failed: demo snapshot has empty scored table.")
    if "risk_score" not in snapshot.scored_df.columns:
        raise RuntimeError("F07 failed: risk_score missing in demo snapshot.")
    if snapshot.optimization_summary.total_cost > snapshot.optimization_summary.budget + 1e-6:
        raise RuntimeError("F07 failed: demo snapshot optimization exceeds budget.")
    if not (0.0 <= snapshot.comparison_metrics.historical.accuracy <= 1.0):
        raise RuntimeError("F07 failed: historical accuracy out of range.")
    if not (0.0 <= snapshot.comparison_metrics.model.accuracy <= 1.0):
        raise RuntimeError("F07 failed: model accuracy out of range.")
    if not (0.0 <= snapshot.comparison_metrics.model_ece <= 1.0):
        raise RuntimeError("F07 failed: model ECE out of range.")

    print("F07 checks passed.")
    print(f"Snapshot rows: {len(snapshot.scored_df)}")
    print(f"Policy briefs: {len(snapshot.policy_briefs)}")
    print(
        f"Historical acc: {snapshot.comparison_metrics.historical.accuracy:.3f} | "
        f"Model acc: {snapshot.comparison_metrics.model.accuracy:.3f}"
    )


def run_feature_08_checks() -> None:
    df = build_feature_table()
    report = run_model_research_benchmark(df, n_splits=3, threshold=0.5, random_state=42)
    if report["n_folds"] < 2:
        raise RuntimeError("F08 failed: insufficient CV folds.")
    for model_key in ["historical", "fusion", "graph"]:
        acc = report[model_key]["accuracy"]["mean"]
        if not (0.0 <= acc <= 1.0):
            raise RuntimeError(f"F08 failed: {model_key} accuracy out of range.")
    print("F08 checks passed.")
    print(
        f"Historical acc mean: {report['historical']['accuracy']['mean']:.3f} | "
        f"Fusion acc mean: {report['fusion']['accuracy']['mean']:.3f} | "
        f"Graph acc mean: {report['graph']['accuracy']['mean']:.3f}"
    )


def run_feature_09_checks() -> None:
    df = build_feature_table()
    report = run_model_research_benchmark(df, n_splits=3, threshold=0.5, random_state=42)

    required_variants = ["historical", "baseline_tabular", "tabular", "tabular_temporal", "fusion", "graph"]
    for variant in required_variants:
        if variant not in report:
            raise RuntimeError(f"F09 failed: missing benchmark variant '{variant}'.")
        for metric_name in ["accuracy", "auroc", "auprc"]:
            value = report[variant][metric_name]["mean"]
            if not (0.0 <= value <= 1.0):
                raise RuntimeError(f"F09 failed: {variant} {metric_name} mean out of range ({value}).")

    ablation_rows = report.get("ablation_accuracy_table", [])
    if len(ablation_rows) < 5:
        raise RuntimeError("F09 failed: ablation accuracy table is incomplete.")

    print("F09 checks passed.")
    print(
        f"Graph AUROC mean: {report['graph']['auroc']['mean']:.3f} | "
        f"Graph AUPRC mean: {report['graph']['auprc']['mean']:.3f} | "
        f"Ablation rows: {len(ablation_rows)}"
    )


def run_feature_10_checks() -> None:
    artifacts = ensure_real_data_cache(
        cache_dir=PROJECT_ROOT / "data" / "cache",
        acs_year=2022,
        refresh=False,
        max_violation_rows=10000,
        timeout_seconds=60,
    )

    acs_df = pd.read_csv(artifacts.acs_path, dtype={"geoid": str})
    validate_acs_block_group_frame(acs_df)

    pws_df = pd.read_csv(artifacts.epa_pws_summary_path)
    validate_epa_pws_lead_signal_frame(pws_df)
    if len(pws_df) < 100:
        raise RuntimeError("F10 failed: EPA PWS summary is unexpectedly small.")

    print("F10 checks passed.")
    print(f"ACS rows: {len(acs_df)}")
    print(f"EPA PWS rows: {len(pws_df)}")


def run_feature_11_checks() -> None:
    df = with_elevated_risk_label(build_feature_table())
    edge_path = PROJECT_ROOT / "data" / "processed" / "graph_edges_sample.csv"
    if edge_path.exists():
        file_edges = load_edge_list(edge_path)
        validate_edge_list(file_edges)

    edges = build_county_proxy_edge_list(df, k_within_county=2)
    validate_edge_list(edges)
    if edges.empty:
        raise RuntimeError("F11 failed: infrastructure edge list is empty.")

    report = run_model_research_benchmark(df, n_splits=3, threshold=0.5, random_state=42)
    if "graph_knn" not in report or "graph_infrastructure" not in report:
        raise RuntimeError("F11 failed: missing graph variant metrics in benchmark report.")
    delta = report.get("improvement_graph_infrastructure_over_graph_knn_accuracy")
    if delta is None:
        raise RuntimeError("F11 failed: missing infrastructure-vs-knn delta metric.")
    if not (-1.0 <= float(delta) <= 1.0):
        raise RuntimeError("F11 failed: invalid infrastructure-vs-knn delta metric.")

    print("F11 checks passed.")
    print(f"Infrastructure edges: {len(edges)}")
    print(f"Infra graph acc mean: {report['graph_infrastructure']['accuracy']['mean']:.3f}")
    print(f"Infra - KNN acc delta: {float(delta):.3f}")


def run_feature_12_checks() -> None:
    df = build_feature_table()
    artifacts = ensure_sentinel_feature_cache(
        df,
        cache_dir=PROJECT_ROOT / "data" / "cache",
        features_filename="sentinel_features_sample.csv",
        metadata_filename="sentinel_features_metadata.json",
        refresh=False,
        start_date="2024-04-01",
        end_date="2024-10-31",
        items_per_block=1,
        max_cloud_cover=60.0,
        bbox_half_size_deg=0.01,
        timeout_seconds=60,
        allow_fallback=True,
    )
    sat_df = pd.read_csv(artifacts.features_path, dtype={"geoid": str})
    validate_sentinel_feature_frame(sat_df)

    merged = df.merge(sat_df, on="geoid", how="left")
    missing = merged["s2_cloud_cover_mean"].isna().sum()
    if missing > 0:
        raise RuntimeError(f"F12 failed: missing Sentinel features for {missing} block groups.")

    live_or_fallback = set(sat_df["s2_source"].astype(str).unique().tolist())
    if not live_or_fallback:
        raise RuntimeError("F12 failed: Sentinel cache has no source tags.")

    print("F12 checks passed.")
    print(f"Sentinel cache rows: {len(sat_df)}")
    print(f"Sentinel source tags: {sorted(live_or_fallback)}")


def run_feature_13_checks() -> None:
    dashboard = build_dashboard_payload(
        budget=35000.0,
        fairness_tolerance=0.05,
        min_county_coverage=0,
        optimizer_method="ilp",
    )
    benchmark = build_benchmark_payload()

    if not dashboard.get("rows"):
        raise RuntimeError("F13 failed: dashboard payload contains no rows.")
    first = dashboard["rows"][0]
    for col in ["lat", "lon", "risk_score", "top_drivers"]:
        if col not in first:
            raise RuntimeError(f"F13 failed: dashboard row missing '{col}'.")

    fair = dashboard.get("fairness_comparison", {})
    if "with_fairness" not in fair or "without_fairness" not in fair:
        raise RuntimeError("F13 failed: fairness comparison blocks missing.")
    if "county_spend_comparison" not in fair:
        raise RuntimeError("F13 failed: county spend comparison missing.")

    if len(benchmark.get("ablation_accuracy_table", [])) < 5:
        raise RuntimeError("F13 failed: benchmark ablation table is incomplete.")

    for path in [PROJECT_ROOT / "web" / "index.html", PROJECT_ROOT / "web" / "app.js", PROJECT_ROOT / "web" / "styles.css"]:
        if not path.exists():
            raise RuntimeError(f"F13 failed: missing frontend file {path}.")

    print("F13 checks passed.")
    print(f"Dashboard rows: {len(dashboard['rows'])}")
    print(f"Ablation rows: {len(benchmark['ablation_accuracy_table'])}")


if __name__ == "__main__":
    run_feature_01_checks()
    run_feature_02_checks()
    run_feature_03_checks()
    run_feature_04_checks()
    run_feature_05_checks()
    run_feature_06_checks()
    run_feature_07_checks()
    run_feature_08_checks()
    run_feature_09_checks()
    run_feature_10_checks()
    run_feature_11_checks()
    run_feature_12_checks()
    run_feature_13_checks()
