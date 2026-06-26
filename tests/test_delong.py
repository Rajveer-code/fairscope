import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from fairscope.core.delong import (
    delong_auc_ci,
    delong_by_group,
    delong_paired_test,
    delong_unpaired_test,
)


def test_auc_matches_literature_toy():
    # 4 positives, 4 negatives; 14/16 concordant pairs -> AUC = 0.875
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.4, 0.6, 0.5, 0.3, 0.2])
    res = delong_auc_ci(y, s)
    assert abs(res["auc"] - 0.875) < 1e-3


def test_auc_matches_sklearn_on_random_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    s = rng.random(500) + 0.3 * y
    res = delong_auc_ci(y, s)
    assert abs(res["auc"] - roc_auc_score(y, s)) < 1e-9


def test_ci_brackets_auc_and_se_positive():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    s = rng.random(400) + 0.2 * y
    r = delong_auc_ci(y, s, alpha=0.05)
    assert r["ci_lower"] <= r["auc"] <= r["ci_upper"]
    assert r["se"] > 0
    assert r["n_pos"] + r["n_neg"] == 400


def test_paired_identical_scores_not_significant():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    s = rng.random(300) + 0.2 * y
    r = delong_paired_test(y, s, s.copy())
    assert abs(r["delta"]) < 1e-12
    assert r["p_value"] > 0.99


def test_paired_detects_real_difference():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 800)
    strong = rng.random(800) + 0.9 * y
    weak = rng.random(800) + 0.05 * y
    r = delong_paired_test(y, strong, weak)
    assert r["delta"] > 0
    assert r["p_value"] < 0.05


def test_unpaired_two_independent_samples():
    rng = np.random.default_rng(4)
    ya = rng.integers(0, 2, 400)
    sa = rng.random(400) + 0.9 * ya
    yb = rng.integers(0, 2, 400)
    sb = rng.random(400) + 0.05 * yb
    r = delong_unpaired_test(ya, sa, yb, sb)
    assert r["delta"] > 0
    assert r["p_value"] < 0.05


def test_one_class_input_raises():
    y = np.ones(10, dtype=int)
    s = np.linspace(0, 1, 10)
    with pytest.raises(ValueError):
        delong_auc_ci(y, s)


def test_by_group_returns_ci_per_group():
    rng = np.random.default_rng(5)
    n = 400
    y = rng.integers(0, 2, n)
    s = rng.random(n) + 0.2 * y
    g = np.array(["A"] * 200 + ["B"] * 200)
    out = delong_by_group(y, s, g)
    assert set(out.keys()) == {"A", "B"}
    assert "auc" in out["A"] and "ci_lower" in out["A"]
