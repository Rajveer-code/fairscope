"""Cross-Platform Fairness Evaluation (CPFE) for NLP, built on fairscope.core.

Public API: the five-axis ``CPFEProtocol``/``CPFEReport``, the multiclass metric and
significance primitives (axes 1-4), and the attribution-stability functions (axis 5).
"""

from .attribution import jaccard_topk, token_saliency
from .cross_platform import CPFEProtocol, CPFEReport
from .metrics import (
    macro_auc,
    macro_f1,
    multiclass_ece,
    per_class_disparate_impact,
    per_class_equalized_odds,
)
from .significance import bootstrap_macro_auc_test

__all__ = [
    "CPFEProtocol",
    "CPFEReport",
    "macro_auc",
    "macro_f1",
    "multiclass_ece",
    "per_class_disparate_impact",
    "per_class_equalized_odds",
    "bootstrap_macro_auc_test",
    "jaccard_topk",
    "token_saliency",
]
