import numpy as np
import pytest

from fairscope.core.metrics import (
    disparate_impact,
    equalized_odds_difference,
    subgroup_metrics,
)


def test_disparate_impact_symmetric_and_known():
    # A positive rate 0.5, B positive rate 0.25 -> DI = 0.5
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    g = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    assert abs(disparate_impact(y_pred, g, "A", "B") - 0.5) < 1e-9
    assert abs(disparate_impact(y_pred, g, "B", "A") - 0.5) < 1e-9


def test_disparate_impact_equal_rates_is_one():
    y_pred = np.array([1, 0, 1, 0])
    g = np.array(["A", "A", "B", "B"])
    assert disparate_impact(y_pred, g, "A", "B") == 1.0


def test_disparate_impact_one_zero_rate_is_zero():
    y_pred = np.array([1, 1, 0, 0])  # A rate 1.0, B rate 0.0
    g = np.array(["A", "A", "B", "B"])
    assert disparate_impact(y_pred, g, "A", "B") == 0.0


def test_disparate_impact_both_zero_rate_is_one():
    y_pred = np.array([0, 0, 0, 0])
    g = np.array(["A", "A", "B", "B"])
    assert disparate_impact(y_pred, g, "A", "B") == 1.0


def test_disparate_impact_missing_group_raises():
    y_pred = np.array([1, 0, 1, 0])
    g = np.array(["A", "A", "B", "B"])
    with pytest.raises(ValueError):
        disparate_impact(y_pred, g, "A", "C")


def test_equalized_odds_difference_known():
    y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1, 1, 1, 0, 0])  # A TPR=1.0, B TPR=0.5
    g = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    assert abs(equalized_odds_difference(y_true, y_pred, g, "A", "B") - 0.5) < 1e-9


def test_equalized_odds_difference_zero_when_equal():
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 0, 1, 0])  # both groups TPR = 0.5
    g = np.array(["A", "A", "B", "B"])
    assert equalized_odds_difference(y_true, y_pred, g, "A", "B") == 0.0


def test_equalized_odds_no_positive_labels_raises():
    y_true = np.array([1, 1, 0, 0])  # group B has no positive labels
    y_pred = np.array([1, 0, 1, 0])
    g = np.array(["A", "A", "B", "B"])
    with pytest.raises(ValueError):
        equalized_odds_difference(y_true, y_pred, g, "A", "B")


def test_subgroup_metrics_keys_and_ranges():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    s = np.clip(rng.random(n) + 0.3 * y, 0.0, 1.0)
    g = np.array(["A"] * 200 + ["B"] * 200)
    out = subgroup_metrics(y, s, g)
    assert set(out["A"]) == {"auc", "brier", "f1", "n"}
    assert 0.0 <= out["A"]["auc"] <= 1.0
    assert out["A"]["n"] == 200


def test_subgroup_metrics_single_class_raises():
    y = np.array([1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4])
    g = np.array(["A"] * 4)
    with pytest.raises(ValueError):
        subgroup_metrics(y, s, g)


def test_subgroup_metrics_nan_score_raises():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, np.nan, 0.3, 0.4])
    g = np.array(["A", "A", "A", "A"])
    with pytest.raises(ValueError):
        subgroup_metrics(y, s, g)


def test_subgroup_metrics_single_class_subgroup_raises():
    # Overall y has both classes; subgroup B is single-class (per-group guard).
    y = np.array([0, 1, 0, 1, 1, 1, 1, 1])
    s = np.array([0.2, 0.8, 0.3, 0.7, 0.6, 0.6, 0.6, 0.6])
    g = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    with pytest.raises(ValueError, match="single class"):
        subgroup_metrics(y, s, g)
