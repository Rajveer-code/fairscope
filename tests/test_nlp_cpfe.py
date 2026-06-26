import numpy as np
import pytest

from fairscope.nlp.cross_platform import CPFEProtocol, CPFEReport


def _probs(rng, y, n_classes, strength):
    logits = rng.normal(0, 1, (len(y), n_classes))
    logits[np.arange(len(y)), y] += strength
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _shifted(rng, y, n_classes):
    """Overconfident-but-wrong predictions: confident toward a (usually wrong) class, so
    discrimination degrades AND calibration worsens (conf >> accuracy) cross-platform."""
    fake = (y + rng.integers(1, n_classes, len(y))) % n_classes
    logits = rng.normal(0, 1, (len(y), n_classes))
    logits[np.arange(len(y)), fake] += 3.0
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _data(rng):
    yk = rng.integers(0, 4, 600)
    yt = rng.integers(0, 4, 500)
    return {
        "kaggle": {"y_true": yk, "probs": _probs(rng, yk, 4, 3.0)},
        "twitter": {"y_true": yt, "probs": _shifted(rng, yt, 4)},  # injected shift
    }


def _run(rng, **kw):
    return CPFEProtocol(_data(rng), reference="kaggle", n_classes=4, n_boot=200, **kw).run()


def test_protocol_detects_shift_axis1_and_axis2():
    df = _run(np.random.default_rng(0)).to_dataframe().set_index("platform")
    assert df.loc["twitter", "macro_auc"] < df.loc["kaggle", "macro_auc"]  # axis 1
    assert df.loc["twitter", "ece"] > df.loc["kaggle", "ece"]  # axis 2


def test_axis3_significance_present_and_bonferroni_corrected():
    report = _run(np.random.default_rng(1))
    sig = report.significance["twitter"]
    assert sig["delta"] > 0 and sig["p_adjusted"] <= 1.0
    assert "reject" in sig


def test_deployment_readiness_structured_per_axis_verdict():
    report = _run(np.random.default_rng(2))
    v = report.deployment_readiness()
    assert "twitter" in v
    assert isinstance(v["twitter"]["ready"], bool)
    axes = v["twitter"]["axes"]
    assert {"discrimination", "calibration", "equity"}.issubset(axes)
    for ax in axes.values():
        assert {"pass", "value", "threshold", "source", "reason"}.issubset(ax)
    assert "illustrative" in axes["discrimination"]["source"]
    assert "P4" in axes["calibration"]["source"]
    assert "P4" in axes["equity"]["source"]
    assert "diagnostic" in report.deployment_readiness.__doc__.lower()


def test_delta_auc_threshold_is_configurable_and_illustrative():
    strict = _run(np.random.default_rng(5), delta_auc_pct_max=1.0)
    assert strict.deployment_readiness()["twitter"]["axes"]["discrimination"]["pass"] is False
    lax = _run(np.random.default_rng(5), delta_auc_pct_max=100.0)
    assert lax.deployment_readiness()["twitter"]["axes"]["discrimination"]["pass"] is True


def test_reference_not_in_data_raises():
    with pytest.raises(ValueError, match="reference"):
        CPFEProtocol(_data(np.random.default_rng(3)), reference="nope", n_classes=4)


def test_report_type_and_equity():
    report = _run(np.random.default_rng(4))
    assert isinstance(report, CPFEReport)
    assert "twitter" in report.equity
    assert "disparate_impact" in report.equity["twitter"]
