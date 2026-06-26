"""Multiple-comparison corrections. Outputs match
``statsmodels.stats.multitest.multipletests`` for methods 'bonferroni' and 'fdr_bh'.
"""

from __future__ import annotations

import numpy as np


def _check_pvals(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be a 1-D array")
    if np.any(np.isnan(p)):
        raise ValueError("p_values contain NaN")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must lie in [0, 1]")
    return p


def bonferroni(p_values, alpha=0.05):
    """Bonferroni correction. Returns adjusted, reject, threshold.

    Examples
    --------
    >>> bonferroni([0.01, 0.5], alpha=0.05)["adjusted"].tolist()
    [0.02, 1.0]
    """
    p = _check_pvals(p_values)
    m = len(p)
    return {
        "adjusted": np.minimum(p * m, 1.0),
        "reject": p <= alpha / m,
        "threshold": alpha / m,
    }


def benjamini_hochberg(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction (step-up). Returns adjusted, reject."""
    p = _check_pvals(p_values)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    adj_sorted = ranked * m / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adj_sorted
    return {"adjusted": adjusted, "reject": adjusted <= alpha}
