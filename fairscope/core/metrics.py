"""Subgroup performance and fairness metrics.

Disparate impact is the symmetric ratio of positive-prediction rates, and equalized odds
difference is the absolute true-positive-rate gap, both as defined in the CPFE paper
(Pall & Yadav). Note: this EOD follows the paper's definition |TPR_a - TPR_b|; some
toolkits (e.g. Fairlearn) instead report max(|TPR diff|, |FPR diff|).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score

from ._utils import _as_1d_arrays, _check_binary_labels, _check_binary_values


def _group_mask(groups, g):
    mask = groups == g
    if not np.any(mask):
        raise ValueError(f"group {g!r} not present in `groups`")
    return mask


def disparate_impact(y_pred, groups, group_a, group_b):
    """Symmetric disparate impact between two groups: min(rate_a/rate_b, rate_b/rate_a).

    ``rate = P(y_pred == 1 | group)`` over hard 0/1 predictions. Result is in (0, 1];
    < 0.80 violates the four-fifths rule and < 0.50 is a severe disparity (interpretation
    per the CPFE paper). If both rates are 0 the groups are treated as equal (returns 1.0);
    if exactly one rate is 0 the disparity is maximal (returns 0.0).

    Examples
    --------
    >>> import numpy as np
    >>> y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    >>> g = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    >>> round(disparate_impact(y_pred, g, "A", "B"), 3)
    0.5
    """
    y_pred, groups = _as_1d_arrays(y_pred, groups)
    yp = _check_binary_values(y_pred)
    ra = float(np.mean(yp[_group_mask(groups, group_a)] == 1))
    rb = float(np.mean(yp[_group_mask(groups, group_b)] == 1))
    if ra == 0.0 and rb == 0.0:
        return 1.0
    if ra == 0.0 or rb == 0.0:
        return 0.0
    return float(min(ra / rb, rb / ra))


def equalized_odds_difference(y_true, y_pred, groups, group_a, group_b):
    """|TPR_a - TPR_b|, the equalized odds difference as defined in the CPFE paper.

    ``TPR = P(y_pred == 1 | y_true == 1, group)``. Raises if either group has no
    positive labels (TPR undefined).
    """
    y_true, y_pred, groups = _as_1d_arrays(y_true, y_pred, groups)
    yt = _check_binary_values(y_true)
    yp = _check_binary_values(y_pred)

    def _tpr(g):
        mask = _group_mask(groups, g)
        pos = (yt == 1) & mask
        if pos.sum() == 0:
            raise ValueError(f"group {g!r} has no positive labels; TPR is undefined")
        return float(np.mean(yp[pos] == 1))

    return float(abs(_tpr(group_a) - _tpr(group_b)))


def subgroup_metrics(y_true, y_score, groups, threshold=0.5):
    """Per-subgroup discrimination metrics: ``{group: {auc, brier, f1, n}}``.

    ``y_score`` are predicted probabilities in [0, 1] (Brier assumes a probability);
    ``threshold`` binarizes them for F1. Raises if any subgroup is single-class (AUC
    undefined) or if ``y_score`` contains NaN.
    """
    y_true, y_score, groups = _as_1d_arrays(y_true, y_score, groups)
    y = _check_binary_labels(y_true)
    s = np.asarray(y_score, dtype=float)
    if np.any(np.isnan(s)):
        raise ValueError("y_score contains NaN")
    preds = (s >= threshold).astype(int)
    out = {}
    for g in np.unique(groups):
        mask = groups == g
        yt = y[mask]
        if len(np.unique(yt)) < 2:
            raise ValueError(f"subgroup {g!r} has a single class; AUC undefined")
        out[g] = {
            "auc": float(roc_auc_score(yt, s[mask])),
            "brier": float(brier_score_loss(yt, s[mask])),
            "f1": float(f1_score(yt, preds[mask], zero_division=0)),
            "n": int(mask.sum()),
        }
    return out
