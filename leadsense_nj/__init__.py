"""LeadSense NJ package."""

from leadsense_nj.baseline import TabularBaselineModel, fit_tabular_logistic
from leadsense_nj.explainability import compute_linear_contributions, format_driver_lines, top_feature_drivers
from leadsense_nj.optimization import OptimizationSummary, optimize_replacement_plan
from leadsense_nj.policy_brief import generate_policy_brief
from leadsense_nj.preprocessing import build_feature_table
from leadsense_nj.target import construct_elevated_risk_label, with_elevated_risk_label
from leadsense_nj.uncertainty import (
    BootstrappedRiskEnsemble,
    expected_calibration_error,
    train_bootstrap_ensemble,
)

__all__ = [
    "BootstrappedRiskEnsemble",
    "TabularBaselineModel",
    "build_feature_table",
    "compute_linear_contributions",
    "construct_elevated_risk_label",
    "expected_calibration_error",
    "fit_tabular_logistic",
    "format_driver_lines",
    "generate_policy_brief",
    "OptimizationSummary",
    "optimize_replacement_plan",
    "top_feature_drivers",
    "train_bootstrap_ensemble",
    "with_elevated_risk_label",
]
