"""Calibration analysis and a subgroup-stratified interface to standard recalibration.

ECE/MCE bin the predicted positive-class probability (equal-width or equal-frequency) and
compare each bin's mean probability to its observed frequency. ``temperature_scale``
(Guo et al., 2017) and ``isotonic_recalibrate`` (Zadrozny & Elkan, 2002) are standard
recalibration methods; the contribution here is the per-subgroup interface and pre/post-ECE
reporting, not the methods themselves.

References
----------
Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural
networks. ICML.
Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass
probability estimates. KDD.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize
from sklearn.isotonic import IsotonicRegression

from ._utils import _as_1d_arrays, _check_binary_values, _check_scores


def _bin_edges(y_prob, n_bins, strategy):
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = 0.0, 1.0
        return edges
    raise ValueError("strategy must be 'uniform' or 'quantile'")


def _binned_gaps(y_true, y_prob, n_bins, strategy):
    edges = _bin_edges(y_prob, n_bins, strategy)
    idx = np.digitize(y_prob, edges[1:-1], right=False)
    n = len(y_prob)
    gaps, weights = [], []
    for b in range(len(edges) - 1):
        mask = idx == b
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        gaps.append(abs(acc - conf))
        weights.append(mask.sum() / n)
    return np.array(gaps), np.array(weights)


def expected_calibration_error(y_true, y_prob, n_bins=10, strategy="uniform"):
    """Weighted mean |observed frequency - mean predicted probability| across bins.

    Returns 0.0 if there is no data. ``strategy`` is 'uniform' (equal-width) or 'quantile'
    (equal-frequency) bins.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0, 0, 1, 1])
    >>> p = np.array([0.0, 0.0, 1.0, 1.0])
    >>> expected_calibration_error(y, p)
    0.0
    """
    y_true, y_prob = _as_1d_arrays(y_true, y_prob)
    y = _check_binary_values(y_true)
    p = _check_scores(y_prob)
    gaps, weights = _binned_gaps(y, p, n_bins, strategy)
    return float(np.sum(weights * gaps)) if gaps.size else 0.0


def maximum_calibration_error(y_true, y_prob, n_bins=10, strategy="uniform"):
    """Maximum |observed frequency - mean predicted probability| over bins."""
    y_true, y_prob = _as_1d_arrays(y_true, y_prob)
    y = _check_binary_values(y_true)
    p = _check_scores(y_prob)
    gaps, _ = _binned_gaps(y, p, n_bins, strategy)
    return float(np.max(gaps)) if gaps.size else 0.0


def ece_by_group(y_true, y_prob, groups, n_bins=10, strategy="uniform"):
    """Per-subgroup Expected Calibration Error. Returns ``{group: ece}``."""
    y_true, y_prob, groups = _as_1d_arrays(y_true, y_prob, groups)
    out = {}
    for g in np.unique(groups):
        mask = groups == g
        out[g] = expected_calibration_error(y_true[mask], y_prob[mask], n_bins, strategy)
    return out


def reliability_diagram(y_true, y_prob, groups=None, n_bins=10):
    """Reliability diagram (mean predicted probability vs observed frequency).

    Returns a matplotlib ``Figure``. If ``groups`` is given, draws one curve per subgroup.
    """
    import matplotlib.pyplot as plt

    y_true, y_prob = _as_1d_arrays(y_true, y_prob)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")

    def _curve(yt, yp, label):
        edges = np.linspace(0, 1, n_bins + 1)
        idx = np.digitize(yp, edges[1:-1])
        xs, ys = [], []
        for b in range(n_bins):
            m = idx == b
            if np.any(m):
                xs.append(float(np.mean(yp[m])))
                ys.append(float(np.mean(yt[m])))
        ax.plot(xs, ys, marker="o", label=label)

    if groups is None:
        _curve(np.asarray(y_true), np.asarray(y_prob), "model")
    else:
        groups = np.asarray(groups)
        for g in np.unique(groups):
            m = groups == g
            _curve(np.asarray(y_true)[m], np.asarray(y_prob)[m], f"group={g}")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.legend()
    return fig


def temperature_scale(logits, y_true, max_iter=200):
    """Fit a single temperature T>0 by minimizing NLL (Guo et al., 2017).

    ``logits`` may be 1-D (binary positive-class logit) or 2-D (n, n_classes). Returns
    ``(T, calibrated_probabilities)``; calibrated probabilities are 1-D for 1-D input.
    """
    logits = np.asarray(logits, dtype=float)
    y = np.asarray(y_true).astype(int)
    logits2 = np.column_stack([np.zeros_like(logits), logits]) if logits.ndim == 1 else logits

    def _nll(log_t):
        t = np.exp(log_t[0])  # parameterize as exp() to keep T > 0
        z = logits2 / t
        z = z - z.max(axis=1, keepdims=True)
        logsumexp = np.log(np.exp(z).sum(axis=1))
        logp = z[np.arange(len(y)), y] - logsumexp
        return -float(np.mean(logp))

    res = optimize.minimize(
        _nll,
        x0=[0.0],
        method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 1e-4, "fatol": 1e-6},
    )
    temp = float(np.exp(res.x[0]))
    z = logits2 / temp
    z = z - z.max(axis=1, keepdims=True)
    probs = np.exp(z)
    probs /= probs.sum(axis=1, keepdims=True)
    calibrated = probs[:, 1] if logits.ndim == 1 else probs
    return temp, calibrated


def isotonic_recalibrate(probs, y_true):
    """Fit isotonic regression mapping predicted prob -> calibrated prob
    (Zadrozny & Elkan, 2002). Returns ``(fitted_model, calibrated_probabilities)``."""
    probs, y_true = _as_1d_arrays(probs, y_true)
    p = _check_scores(probs)
    y = _check_binary_values(y_true)
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrated = model.fit_transform(p, y)
    return model, calibrated
