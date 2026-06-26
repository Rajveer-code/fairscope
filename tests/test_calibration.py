import matplotlib
import numpy as np

matplotlib.use("Agg")

from fairscope.core.calibration import (  # noqa: E402
    ece_by_group,
    expected_calibration_error,
    isotonic_recalibrate,
    maximum_calibration_error,
    reliability_diagram,
    temperature_scale,
)


def test_ece_near_zero_for_perfectly_calibrated():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert expected_calibration_error(y, p, n_bins=10) < 0.02


def test_ece_high_for_confident_but_wrong():
    p = np.full(1000, 0.99)
    y = np.zeros(1000, dtype=int)
    assert expected_calibration_error(y, p, n_bins=10) > 0.9


def test_quantile_strategy_runs():
    rng = np.random.default_rng(3)
    p = rng.random(2000)
    y = (rng.random(2000) < p).astype(int)
    assert expected_calibration_error(y, p, n_bins=10, strategy="quantile") < 0.05


def test_invalid_strategy_raises():
    import pytest

    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0, 1]), np.array([0.2, 0.8]), strategy="bogus")


def test_mce_at_least_ece():
    rng = np.random.default_rng(1)
    p = rng.random(2000)
    y = (rng.random(2000) < p * 0.5).astype(int)  # miscalibrated
    assert maximum_calibration_error(y, p) >= expected_calibration_error(y, p) - 1e-12


def test_ece_by_group_returns_per_group():
    rng = np.random.default_rng(2)
    n = 2000
    p = rng.random(n)
    y = (rng.random(n) < p).astype(int)
    g = np.array(["A"] * 1000 + ["B"] * 1000)
    out = ece_by_group(y, p, g)
    assert set(out.keys()) == {"A", "B"}


def test_temperature_scaling_reduces_ece():
    # Genuine overconfidence: labels drawn from a MODERATE true logit, but the model
    # reports logits inflated by 3x. The NLL-optimal temperature is therefore ~3 (> 1),
    # and rescaling recovers calibration.
    rng = np.random.default_rng(0)
    n = 8000
    logit_true = rng.normal(0.0, 1.5, n)
    p_true = 1.0 / (1.0 + np.exp(-logit_true))
    y = (rng.random(n) < p_true).astype(int)
    logits = 3.0 * logit_true  # overconfident by a factor of 3
    p_before = 1.0 / (1.0 + np.exp(-logits))
    ece_before = expected_calibration_error(y, p_before, n_bins=15)
    temp, p_after = temperature_scale(logits, y)
    ece_after = expected_calibration_error(y, p_after, n_bins=15)
    assert ece_after < ece_before
    assert temp > 1.0


def test_isotonic_reduces_ece():
    rng = np.random.default_rng(1)
    n = 5000
    y = rng.integers(0, 2, n)
    base = 0.8 * (2 * y - 1) + 0.3 * rng.standard_normal(n)
    p = 1.0 / (1.0 + np.exp(-4.0 * base))  # overconfident
    ece_before = expected_calibration_error(y, p, n_bins=15)
    _, p_after = isotonic_recalibrate(p, y)
    ece_after = expected_calibration_error(y, p_after, n_bins=15)
    assert ece_after <= ece_before


def test_reliability_diagram_returns_figure():
    rng = np.random.default_rng(0)
    n = 500
    y = rng.integers(0, 2, n)
    p = rng.random(n)
    fig = reliability_diagram(y, p, n_bins=10)
    assert fig is not None
    assert len(fig.axes) == 1


def test_reliability_diagram_grouped_returns_figure():
    rng = np.random.default_rng(4)
    n = 600
    y = rng.integers(0, 2, n)
    p = rng.random(n)
    g = np.array(["A"] * 300 + ["B"] * 300)
    fig = reliability_diagram(y, p, groups=g, n_bins=10)
    assert fig is not None
