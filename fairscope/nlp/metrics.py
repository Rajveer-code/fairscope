"""Multiclass metrics for the CPFE protocol (axes 1, 2, 4 primitives).

Pure functions that reuse ``fairscope.core`` where the definition is shared. The
confidence-accuracy ECE follows the formula in the CPFE paper (Pall & Yadav), and is
distinct from ``core.expected_calibration_error`` (binary prob-vs-frequency calibration).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from ..core import disparate_impact, equalized_odds_difference


def _check_probs(y_true, probs):
    y = np.asarray(y_true)
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2:
        raise ValueError("probs must be a 2-D array of shape (n_samples, n_classes)")
    if len(y) != p.shape[0]:
        raise ValueError("y_true and probs must have the same number of rows")
    if np.any(np.isnan(p)):
        raise ValueError("probs contain NaN")
    return y, p


def macro_auc(y_true, probs):
    """Macro one-vs-rest AUC. Requires every class present in ``y_true``."""
    y, p = _check_probs(y_true, probs)
    return float(roc_auc_score(y, p, multi_class="ovr", average="macro"))


def macro_f1(y_true, probs):
    """Macro F1 of the argmax predictions."""
    y, p = _check_probs(y_true, probs)
    return float(f1_score(y, p.argmax(axis=1), average="macro"))


def multiclass_ece(y_true, probs, n_bins=10):
    """Confidence-accuracy Expected Calibration Error (Guo et al. 2017; CPFE paper):
    ``ECE = sum_m (|B_m|/n) * |acc(B_m) - conf(B_m)|`` with ``conf = max prob`` and
    ``acc = top-1 correct``."""
    y, p = _check_probs(y_true, probs)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(conf, edges[1:-1])
    n = len(y)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if np.any(m):
            ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def per_class_disparate_impact(probs_a, probs_b, n_classes):
    """Symmetric DI per class between two platforms, reusing ``core.disparate_impact`` with
    the class binarized (``pred == c``) and platform as the two-group label."""
    pa = np.asarray(probs_a).argmax(axis=1)
    pb = np.asarray(probs_b).argmax(axis=1)
    groups = np.array(["a"] * len(pa) + ["b"] * len(pb))
    out = {}
    for c in range(n_classes):
        ypred = np.concatenate([(pa == c).astype(int), (pb == c).astype(int)])
        out[c] = disparate_impact(ypred, groups, "a", "b")
    return out


def per_class_equalized_odds(y_a, probs_a, y_b, probs_b, n_classes):
    """EOD per class between two platforms (``|TPR_c(A) - TPR_c(B)|``), reusing
    ``core.equalized_odds_difference``. A class with no positive labels in a platform is
    returned as ``None`` (TPR undefined)."""
    pa = np.asarray(probs_a).argmax(axis=1)
    pb = np.asarray(probs_b).argmax(axis=1)
    ya, yb = np.asarray(y_a), np.asarray(y_b)
    groups = np.array(["a"] * len(ya) + ["b"] * len(yb))
    out = {}
    for c in range(n_classes):
        yt = np.concatenate([(ya == c).astype(int), (yb == c).astype(int)])
        yp = np.concatenate([(pa == c).astype(int), (pb == c).astype(int)])
        try:
            out[c] = equalized_odds_difference(yt, yp, groups, "a", "b")
        except ValueError:
            out[c] = None  # a platform has no examples of class c
    return out
