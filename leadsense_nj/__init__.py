"""LeadSense NJ package."""

from leadsense_nj.baseline import TabularBaselineModel, fit_tabular_logistic
from leadsense_nj.demo import DemoSnapshot, build_demo_snapshot
from leadsense_nj.explainability import compute_linear_contributions, format_driver_lines, top_feature_drivers
from leadsense_nj.graph_model import (
    GraphEnhancedRiskModel,
    build_knn_adjacency,
    graph_mean_aggregate,
    train_graph_enhanced_model,
)
from leadsense_nj.metrics import (
    BinaryClassificationMetrics,
    ModelVsHistoricalMetrics,
    compute_binary_metrics,
    compute_model_vs_historical_metrics,
    compute_probabilistic_metrics,
    historical_signal_prediction,
)
from leadsense_nj.multimodal import FusionRiskModel, build_fusion_feature_table, build_temporal_features, train_fusion_model
from leadsense_nj.optimization import OptimizationSummary, optimize_replacement_plan, optimize_replacement_plan_ilp
from leadsense_nj.policy_brief import generate_policy_brief
from leadsense_nj.preprocessing import build_feature_table
from leadsense_nj.research import run_model_research_benchmark, spatial_kfold_splits
from leadsense_nj.target import construct_elevated_risk_label, with_elevated_risk_label
from leadsense_nj.uncertainty import (
    BootstrappedRiskEnsemble,
    expected_calibration_error,
    train_bootstrap_ensemble,
)

__all__ = [
    "BootstrappedRiskEnsemble",
    "TabularBaselineModel",
    "FusionRiskModel",
    "GraphEnhancedRiskModel",
    "build_feature_table",
    "build_demo_snapshot",
    "build_fusion_feature_table",
    "build_knn_adjacency",
    "build_temporal_features",
    "compute_binary_metrics",
    "compute_linear_contributions",
    "compute_model_vs_historical_metrics",
    "compute_probabilistic_metrics",
    "construct_elevated_risk_label",
    "BinaryClassificationMetrics",
    "expected_calibration_error",
    "fit_tabular_logistic",
    "format_driver_lines",
    "generate_policy_brief",
    "historical_signal_prediction",
    "ModelVsHistoricalMetrics",
    "OptimizationSummary",
    "optimize_replacement_plan",
    "optimize_replacement_plan_ilp",
    "graph_mean_aggregate",
    "run_model_research_benchmark",
    "spatial_kfold_splits",
    "DemoSnapshot",
    "top_feature_drivers",
    "train_fusion_model",
    "train_graph_enhanced_model",
    "train_bootstrap_ensemble",
    "with_elevated_risk_label",
]
