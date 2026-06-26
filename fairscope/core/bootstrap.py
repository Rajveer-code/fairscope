"""Stratified bootstrap test for the difference between two AUCs.

This is the significance procedure used for the cross-platform macro-AUC comparison in the
CPFE protocol (stratified bootstrap standard errors, B = 2000 by default). It complements
``delong`` (analytic per-subgroup CIs).
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from ._utils import _as_1d_arrays, _check_binary_labels, _check_scores


def bootstrap_auc_test(y_true, score_a, score_b, n_boot=2000, random_state=None):
    """Stratified bootstrap test of (AUC_a - AUC_b) on the same samples.

    Resamples positives and negatives separately (preserving class balance). Returns a
    dict: auc_a, auc_b, delta, se, z, p_value, ci_lower, ci_upper, n_boot.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 2, 400)
    >>> a = rng.random(400) + 0.8 * y
    >>> b = rng.random(400) + 0.05 * y
    >>> bootstrap_auc_test(y, a, b, n_boot=200, random_state=0)["delta"] > 0
    True
    """
    y_true, score_a, score_b = _as_1d_arrays(y_true, score_a, score_b)
    y = _check_binary_labels(y_true)
    a = _check_scores(score_a)
    b = _check_scores(score_b)
    rng = np.random.default_rng(random_state)

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    auc_a = float(roc_auc_score(y, a))
    auc_b = float(roc_auc_score(y, b))
    observed = auc_a - auc_b

    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        bp = rng.choice(pos_idx, size=pos_idx.size, replace=True)
        bn = rng.choice(neg_idx, size=neg_idx.size, replace=True)
        idx = np.concatenate([bp, bn])
        yb = y[idx]
        deltas[i] = roc_auc_score(yb, a[idx]) - roc_auc_score(yb, b[idx])

    se = float(deltas.std(ddof=1))
    z = observed / se if se > 0 else 0.0
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "delta": observed,
        "se": se,
        "z": float(z),
        "p_value": float(2.0 * stats.norm.sf(abs(z))),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "n_boot": n_boot,
    }
