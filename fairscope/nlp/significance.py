"""Axis 3 of CPFE: unpaired bootstrap comparison of macro one-vs-rest AUC across two
platforms (independent test sets), as in the CPFE paper (stratified bootstrap standard
errors, B = 2000 by default, combined for a normal-approximation z-test).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .metrics import macro_auc


def _bootstrap_se(y, probs, n_boot, rng):
    """Stratified (by class) bootstrap standard error of the macro AUC."""
    y = np.asarray(y)
    probs = np.asarray(probs, dtype=float)
    class_idx = [np.flatnonzero(y == c) for c in np.unique(y)]
    aucs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(ci, size=ci.size, replace=True) for ci in class_idx])
        aucs[i] = macro_auc(y[idx], probs[idx])
    return float(aucs.std(ddof=1))


def bootstrap_macro_auc_test(y_a, probs_a, y_b, probs_b, n_boot=2000, random_state=None):
    """Compare macro AUC across two platforms (independent test sets).

    Each platform's macro-AUC standard error is estimated by a class-stratified bootstrap;
    the errors are combined for an unpaired z-test. Returns a dict: auc_a, auc_b, delta,
    se, z, p_value, n_boot.
    """
    rng = np.random.default_rng(random_state)
    probs_a = np.asarray(probs_a, dtype=float)
    probs_b = np.asarray(probs_b, dtype=float)
    auc_a = macro_auc(y_a, probs_a)
    auc_b = macro_auc(y_b, probs_b)
    se_a = _bootstrap_se(y_a, probs_a, n_boot, rng)
    se_b = _bootstrap_se(y_b, probs_b, n_boot, rng)
    delta = auc_a - auc_b
    se = float(np.sqrt(se_a**2 + se_b**2))
    z = delta / se if se > 0 else 0.0
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "delta": delta,
        "se": se,
        "z": float(z),
        "p_value": float(2.0 * stats.norm.sf(abs(z))),
        "n_boot": n_boot,
    }
