from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from leadsense_nj.demo import build_demo_snapshot
from leadsense_nj.preprocessing import build_feature_table
from leadsense_nj.research import run_model_research_benchmark


@st.cache_data(show_spinner=False)
def _cached_ablation_report(df: pd.DataFrame) -> dict:
    return run_model_research_benchmark(df, n_splits=3, threshold=0.5, random_state=42)


st.set_page_config(page_title="LeadSense NJ", layout="wide")
st.title("LeadSense NJ")
st.caption("Risk scoring, uncertainty, fairness-aware replacement prioritization, and policy briefs.")

default_data_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "block_group_features_sample.csv"
budget = st.sidebar.slider("Budget (USD)", min_value=10000, max_value=100000, value=35000, step=1000)
fairness_tolerance = st.sidebar.slider("Fairness tolerance", min_value=0.0, max_value=0.2, value=0.05, step=0.01)
min_county_coverage = st.sidebar.slider("Minimum county seed coverage", min_value=0, max_value=2, value=0, step=1)
optimizer_method = st.sidebar.selectbox("Optimizer", options=["ilp", "greedy"], index=0)

df = build_feature_table(default_data_path)
snapshot = build_demo_snapshot(
    df,
    budget=float(budget),
    fairness_tolerance=float(fairness_tolerance),
    min_county_coverage=int(min_county_coverage),
    optimizer_method=str(optimizer_method),
)

tab1, tab2, tab3, tab4 = st.tabs(["Risk Overview", "Optimizer", "Policy Briefs", "Metrics"])

with tab1:
    st.subheader("Risk + Uncertainty Table")
    st.dataframe(
        snapshot.scored_df[
            [
                "geoid",
                "county",
                "municipality",
                "risk_score",
                "risk_uncertainty",
                "risk_label",
            ]
        ]
    )

with tab2:
    st.subheader("Replacement Plan")
    summary = snapshot.optimization_summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Selected Blocks", summary.selected_count)
    col2.metric("Total Cost", f"${summary.total_cost:,.0f}")
    col3.metric("Risk Reduced", f"{summary.total_risk_reduced:.3f}")
    st.caption(
        f"Fairness target: {summary.fairness_target:.3f} | "
        f"Achieved minority share: {summary.achieved_minority_share:.3f}"
    )
    st.dataframe(
        snapshot.selected_df[
            [
                "priority_rank",
                "geoid",
                "county",
                "municipality",
                "risk_score",
                "replacement_cost",
                "minority_share",
            ]
        ]
        if not snapshot.selected_df.empty
        else pd.DataFrame()
    )

with tab3:
    st.subheader("Generated Policy Briefs")
    if not snapshot.policy_briefs:
        st.info("No policy briefs were generated for the current budget/constraints.")
    else:
        geoid = st.selectbox("Select GEOID", options=list(snapshot.policy_briefs.keys()))
        st.text(snapshot.policy_briefs[geoid])

with tab4:
    st.subheader("Historical vs Model Performance")
    metrics = snapshot.comparison_metrics
    benchmark = _cached_ablation_report(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Historical Accuracy", f"{metrics.historical.accuracy * 100:.1f}%")
    c2.metric("Model Accuracy", f"{metrics.model.accuracy * 100:.1f}%")
    c3.metric("Accuracy Lift", f"{metrics.accuracy_delta_model_minus_historical * 100:.1f} pts")

    c4, c5, c6 = st.columns(3)
    c4.metric("Model Precision", f"{metrics.model.precision * 100:.1f}%")
    c5.metric("Model Recall", f"{metrics.model.recall * 100:.1f}%")
    c6.metric("Model F1", f"{metrics.model.f1 * 100:.1f}%")

    c7, c8, c9 = st.columns(3)
    c7.metric("Historical Positive Rate", f"{metrics.historical.positive_rate * 100:.1f}%")
    c8.metric("Model Positive Rate", f"{metrics.model.positive_rate * 100:.1f}%")
    c9.metric("Model ECE", f"{metrics.model_ece:.3f}")

    c10, c11 = st.columns(2)
    c10.metric("Historical AUROC", f"{metrics.historical_auroc:.3f}")
    c11.metric("Model AUROC", f"{metrics.model_auroc:.3f}")

    c12, c13 = st.columns(2)
    c12.metric("Historical AUPRC", f"{metrics.historical_auprc:.3f}")
    c13.metric("Model AUPRC", f"{metrics.model_auprc:.3f}")

    st.caption(
        f"Model threshold: {metrics.model_threshold:.2f} | "
        f"Brier score: {metrics.model_brier:.3f}"
    )

    matrix_df = pd.DataFrame(
        [
            {
                "method": "historical",
                "tp": metrics.historical.tp,
                "fp": metrics.historical.fp,
                "tn": metrics.historical.tn,
                "fn": metrics.historical.fn,
            },
            {
                "method": "model",
                "tp": metrics.model.tp,
                "fp": metrics.model.fp,
                "tn": metrics.model.tn,
                "fn": metrics.model.fn,
            },
        ]
    )
    st.dataframe(matrix_df, use_container_width=True)

    st.subheader("Ablation Benchmark")
    ablation_df = pd.DataFrame(benchmark.get("ablation_accuracy_table", []))
    if not ablation_df.empty:
        ablation_df = ablation_df.rename(
            columns={
                "model": "Model",
                "accuracy_mean": "Accuracy Mean",
                "accuracy_std": "Accuracy Std",
                "auroc_mean": "AUROC Mean",
                "auprc_mean": "AUPRC Mean",
            }
        )
        st.dataframe(ablation_df, use_container_width=True)
    else:
        st.info("No ablation rows available.")
