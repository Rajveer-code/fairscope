import numpy as np

from fairscope.core.bootstrap import bootstrap_auc_test


def test_recovers_planted_difference():
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 2, n)
    strong = rng.random(n) + 0.8 * y
    weak = rng.random(n) + 0.05 * y
    r = bootstrap_auc_test(y, strong, weak, n_boot=500, random_state=42)
    assert r["delta"] > 0
    assert r["p_value"] < 0.05


def test_deterministic_with_seed():
    rng = np.random.default_rng(1)
    n = 300
    y = rng.integers(0, 2, n)
    a = rng.random(n) + 0.3 * y
    b = rng.random(n)
    r1 = bootstrap_auc_test(y, a, b, n_boot=200, random_state=7)
    r2 = bootstrap_auc_test(y, a, b, n_boot=200, random_state=7)
    assert r1["se"] == r2["se"]
    assert r1["p_value"] == r2["p_value"]


def test_identical_models_not_significant():
    rng = np.random.default_rng(2)
    n = 400
    y = rng.integers(0, 2, n)
    s = rng.random(n) + 0.3 * y
    r = bootstrap_auc_test(y, s, s.copy(), n_boot=200, random_state=0)
    assert abs(r["delta"]) < 1e-12
    assert r["p_value"] > 0.5
