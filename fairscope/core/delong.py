"""DeLong (1988) confidence intervals and tests for AUC, via the fast midrank
algorithm of Sun & Xu (2014).

References
----------
DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under
two or more correlated ROC curves. Biometrics, 44(3), 837-845.
Sun, X., & Xu, W. (2014). Fast implementation of DeLong's algorithm for comparing the
areas under correlated ROC curves. IEEE Signal Processing Letters, 21(11), 1389-1393.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._utils import _as_1d_arrays, _check_binary_labels, _check_scores


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks (average ranks for ties), O(n log n)."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    midrank = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midrank[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=float)
    out[order] = midrank
    return out


def _fast_delong(sorted_scores: np.ndarray, n_pos: int):
    """Sun & Xu (2014). ``sorted_scores`` has shape (k, N) with the n_pos positives
    first. Returns (aucs: shape (k,), covariance: shape (k, k))."""
    k, total = sorted_scores.shape
    n_neg = total - n_pos
    pos = sorted_scores[:, :n_pos]
    neg = sorted_scores[:, n_pos:]
    tx = np.empty((k, n_pos))
    ty = np.empty((k, n_neg))
    tz = np.empty((k, total))
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(sorted_scores[r])
    aucs = tz[:, :n_pos].sum(axis=1) / n_pos / n_neg - (n_pos + 1.0) / (2.0 * n_neg)
    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / n_pos + sy / n_neg
    return aucs, np.atleast_2d(cov)


def _prepare(y: np.ndarray, score_list):
    order = np.argsort(-y, kind="mergesort")  # label 1 (positives) first
    n_pos = int((y == 1).sum())
    sorted_scores = np.vstack([np.asarray(s)[order] for s in score_list])
    return sorted_scores, n_pos


def delong_auc_ci(y_true, y_score, alpha=0.05):
    """AUC with a DeLong (1 - alpha) normal-approximation confidence interval.

    Returns a dict: auc, ci_lower, ci_upper, se, n_pos, n_neg.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    >>> s = np.array([0.9, 0.8, 0.7, 0.4, 0.6, 0.5, 0.3, 0.2])
    >>> round(delong_auc_ci(y, s)["auc"], 3)
    0.875
    """
    y_true, y_score = _as_1d_arrays(y_true, y_score)
    y = _check_binary_labels(y_true)
    s = _check_scores(y_score)
    sorted_scores, n_pos = _prepare(y, [s])
    aucs, cov = _fast_delong(sorted_scores, n_pos)
    auc = float(aucs[0])
    var = float(cov[0, 0])
    se = float(np.sqrt(var)) if var > 0 else 0.0
    z = float(stats.norm.ppf(1 - alpha / 2.0))
    return {
        "auc": auc,
        "ci_lower": max(0.0, auc - z * se),
        "ci_upper": min(1.0, auc + z * se),
        "se": se,
        "n_pos": n_pos,
        "n_neg": len(y) - n_pos,
    }


def delong_paired_test(y_true, score_a, score_b):
    """Covariance-aware paired DeLong test for two scores on the SAME samples.

    Returns: auc_a, auc_b, delta, z, p_value.
    """
    y_true, score_a, score_b = _as_1d_arrays(y_true, score_a, score_b)
    y = _check_binary_labels(y_true)
    a = _check_scores(score_a)
    b = _check_scores(score_b)
    sorted_scores, n_pos = _prepare(y, [a, b])
    aucs, cov = _fast_delong(sorted_scores, n_pos)
    auc_a, auc_b = float(aucs[0]), float(aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    se = float(np.sqrt(var)) if var > 0 else 0.0
    delta = auc_a - auc_b
    z = delta / se if se > 0 else 0.0
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "delta": delta,
        "z": float(z),
        "p_value": float(2.0 * stats.norm.sf(abs(z))),
    }


def delong_unpaired_test(y_true_a, score_a, y_true_b, score_b):
    """Unpaired DeLong test for two INDEPENDENT samples (the cross-platform case)."""
    a = delong_auc_ci(y_true_a, score_a)
    b = delong_auc_ci(y_true_b, score_b)
    delta = a["auc"] - b["auc"]
    se = float(np.sqrt(a["se"] ** 2 + b["se"] ** 2))
    z = delta / se if se > 0 else 0.0
    return {
        "auc_a": a["auc"],
        "auc_b": b["auc"],
        "delta": delta,
        "z": float(z),
        "p_value": float(2.0 * stats.norm.sf(abs(z))),
    }


def delong_by_group(y_true, y_score, groups, alpha=0.05):
    """Per-subgroup DeLong AUC CIs. Returns {group: ci_dict}."""
    y_true, y_score, groups = _as_1d_arrays(y_true, y_score, groups)
    out = {}
    for g in np.unique(groups):
        mask = groups == g
        try:
            out[g] = delong_auc_ci(y_true[mask], y_score[mask], alpha=alpha)
        except ValueError as exc:
            raise ValueError(f"subgroup {g!r}: {exc}") from exc
    return out
