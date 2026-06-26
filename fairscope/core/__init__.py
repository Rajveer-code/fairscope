"""Core statistical primitives for fairscope.

Public API for subgroup-stratified, calibration-aware fairness auditing:
DeLong AUC confidence intervals and tests, a stratified bootstrap AUC test, calibration
error and recalibration, multiple-comparison corrections, and subgroup fairness metrics.
"""

from .bootstrap import bootstrap_auc_test
from .calibration import (
    ece_by_group,
    expected_calibration_error,
    isotonic_recalibrate,
    maximum_calibration_error,
    reliability_diagram,
    temperature_scale,
)
from .correction import benjamini_hochberg, bonferroni
from .delong import (
    delong_auc_ci,
    delong_by_group,
    delong_paired_test,
    delong_unpaired_test,
)
from .metrics import disparate_impact, equalized_odds_difference, subgroup_metrics

__all__ = [
    # AUC confidence intervals and tests (delong)
    "delong_auc_ci",
    "delong_paired_test",
    "delong_unpaired_test",
    "delong_by_group",
    # AUC significance via stratified bootstrap
    "bootstrap_auc_test",
    # calibration error and recalibration
    "expected_calibration_error",
    "maximum_calibration_error",
    "ece_by_group",
    "reliability_diagram",
    "temperature_scale",
    "isotonic_recalibrate",
    # multiple-comparison correction
    "bonferroni",
    "benjamini_hochberg",
    # subgroup fairness metrics
    "disparate_impact",
    "equalized_odds_difference",
    "subgroup_metrics",
]
