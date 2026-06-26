import numpy as np

from fairscope.nlp.significance import bootstrap_macro_auc_test


def _probs(rng, y, n_classes, strength):
    logits = rng.normal(0, 1, (len(y), n_classes))
    logits[np.arange(len(y)), y] += strength
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def test_detects_cross_platform_degradation():
    rng = np.random.default_rng(0)
    ya = rng.integers(0, 4, 600)
    yb = rng.integers(0, 4, 600)
    pa = _probs(rng, ya, 4, strength=3.0)  # strong (reference)
    pb = _probs(rng, yb, 4, strength=0.6)  # degraded (shifted platform)
    r = bootstrap_macro_auc_test(ya, pa, yb, pb, n_boot=400, random_state=1)
    assert r["auc_a"] > r["auc_b"]
    assert r["delta"] > 0
    assert r["p_value"] < 0.05


def test_deterministic_with_seed():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 4, 400)
    p = _probs(rng, y, 4, 2.0)
    r1 = bootstrap_macro_auc_test(y, p, y, p, n_boot=200, random_state=9)
    r2 = bootstrap_macro_auc_test(y, p, y, p, n_boot=200, random_state=9)
    assert r1["se"] == r2["se"]
    assert r1["p_value"] == r2["p_value"]


def test_identical_platforms_not_significant():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 4, 500)
    p = _probs(rng, y, 4, 2.0)
    r = bootstrap_macro_auc_test(y, p, y, p, n_boot=300, random_state=0)
    assert abs(r["delta"]) < 1e-12
    assert r["p_value"] > 0.5
