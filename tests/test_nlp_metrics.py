import numpy as np

from fairscope.nlp.metrics import (
    macro_auc,
    macro_f1,
    multiclass_ece,
    per_class_disparate_impact,
    per_class_equalized_odds,
)


def _probs_from_labels(rng, y, n_classes, strength):
    """Class probabilities that favor the true class by `strength`."""
    logits = rng.normal(0, 1, size=(len(y), n_classes))
    logits[np.arange(len(y)), y] += strength
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def test_macro_auc_high_for_strong_signal():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 800)
    p = _probs_from_labels(rng, y, 4, strength=3.0)
    assert macro_auc(y, p) > 0.9


def test_macro_auc_near_chance_for_no_signal():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 800)
    p = _probs_from_labels(rng, y, 4, strength=0.0)
    assert 0.4 < macro_auc(y, p) < 0.6


def test_macro_f1_in_range():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 4, 400)
    p = _probs_from_labels(rng, y, 4, strength=2.0)
    assert 0.0 <= macro_f1(y, p) <= 1.0


def _build_probs(rng, conf, correct, n_classes):
    """Probabilities whose top-class probability is exactly `conf`, predicted class is the
    true class when `correct` else a wrong class. Lets us control calibration directly."""
    n = len(conf)
    y = rng.integers(0, n_classes, n)
    pred = np.where(correct, y, (y + 1) % n_classes)
    rest = (1.0 - conf) / (n_classes - 1)
    probs = np.repeat(rest[:, None], n_classes, axis=1)
    probs[np.arange(n), pred] = conf
    return y, probs


def test_multiclass_ece_near_zero_when_calibrated_high_when_overconfident():
    rng = np.random.default_rng(3)
    n = 8000
    conf = rng.uniform(0.5, 1.0, n)  # top-class confidence
    y_cal, p_cal = _build_probs(rng, conf, rng.random(n) < conf, 4)  # acc == conf
    y_over, p_over = _build_probs(rng, conf, rng.random(n) < conf * 0.5, 4)  # acc << conf
    ece_cal = multiclass_ece(y_cal, p_cal)
    ece_over = multiclass_ece(y_over, p_over)
    assert ece_cal < 0.05  # well calibrated -> small gap
    assert ece_over > 0.15  # overconfident -> large gap
    assert ece_over > ece_cal


def test_per_class_disparate_impact_symmetric_in_bounds():
    rng = np.random.default_rng(4)
    pa = _probs_from_labels(rng, rng.integers(0, 4, 300), 4, 2.0)
    pb = _probs_from_labels(rng, rng.integers(0, 4, 300), 4, 2.0)
    di = per_class_disparate_impact(pa, pb, n_classes=4)
    assert set(di.keys()) == {0, 1, 2, 3}
    assert all(0.0 <= v <= 1.0 for v in di.values())


def test_per_class_equalized_odds_zero_for_identical():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 4, 400)
    p = _probs_from_labels(rng, y, 4, 2.0)
    eod = per_class_equalized_odds(y, p, y, p, n_classes=4)
    assert all(v is not None and abs(v) < 1e-9 for v in eod.values())


def test_macro_auc_rejects_non_2d_probs():
    import pytest

    with pytest.raises(ValueError):
        macro_auc(np.array([0, 1, 2, 3]), np.array([0.1, 0.2, 0.3, 0.4]))


def test_check_probs_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        macro_auc(np.array([0, 1, 2]), np.zeros((4, 4)))


def test_check_probs_nan_raises():
    import pytest

    p = np.full((4, 4), 0.25)
    p[0, 0] = np.nan
    with pytest.raises(ValueError):
        macro_auc(np.array([0, 1, 2, 3]), p)


def test_per_class_equalized_odds_missing_class_is_none():
    # platform B has no examples of class 3 -> EOD for class 3 is undefined (None)
    rng = np.random.default_rng(6)
    ya = rng.integers(0, 4, 200)
    pa = _probs_from_labels(rng, ya, 4, 2.0)
    yb = rng.integers(0, 3, 200)  # classes 0, 1, 2 only
    pb = _probs_from_labels(rng, yb, 4, 2.0)
    eod = per_class_equalized_odds(ya, pa, yb, pb, n_classes=4)
    assert eod[3] is None
