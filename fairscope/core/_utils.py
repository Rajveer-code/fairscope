"""Shared input validation. Fail loudly and early; never fail silently."""

from __future__ import annotations

import numpy as np


def _as_1d_arrays(*arrays):
    """Coerce inputs to 1-D numpy arrays of equal length; raise otherwise."""
    out = [np.asarray(a) for a in arrays]
    n = len(out[0])
    for a in out:
        if a.ndim != 1:
            raise ValueError("inputs must be 1-D arrays")
        if len(a) != n:
            raise ValueError("inputs must all have the same length")
    return out


def _check_binary_values(y):
    """Validate labels are binary 0/1 with no NaN. Single-class is allowed."""
    y = np.asarray(y)
    if np.any(np.isnan(y.astype(float))):
        raise ValueError("y_true contains NaN")
    uniq = np.unique(y)
    if not set(uniq.tolist()).issubset({0, 1}):
        raise ValueError(f"y_true must be binary 0/1; got values {uniq.tolist()}")
    return y.astype(int)


def _check_binary_labels(y):
    """Like _check_binary_values, but require BOTH classes present (AUC needs both)."""
    y = _check_binary_values(y)
    if len(np.unique(y)) < 2:
        raise ValueError("y_true must contain both classes (0 and 1)")
    return y


def _check_scores(s):
    """Validate scores are finite floats."""
    s = np.asarray(s, dtype=float)
    if np.any(np.isnan(s)):
        raise ValueError("scores contain NaN")
    return s
