import os

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from fairscope.healthcare import HealthcareFairnessAudit, HealthcareReport  # noqa: E402


def _planted(rng, n_per_group, sep_young, sep_old):
    """Two disjoint groups; within each, positives/negatives separated by `sep` so the
    within-group AUC ~ Phi(sep / sqrt(2)). Larger sep -> higher AUC."""

    def block(sep, label):
        half = n_per_group // 2
        y = np.r_[np.ones(half, int), np.zeros(half, int)]
        raw = np.r_[rng.normal(sep, 1.0, half), rng.normal(0.0, 1.0, half)]
        s = 1.0 / (1.0 + np.exp(-raw))
        g = np.array([label] * (2 * half))
        return y, s, g

    y1, s1, g1 = block(sep_young, "young")
    y0, s0, g0 = block(sep_old, "old")
    return (
        np.concatenate([y1, y0]),
        np.concatenate([s1, s0]),
        np.concatenate([g1, g0]),
    )


def test_from_scores_recovers_planted_gap():
    rng = np.random.default_rng(0)
    y, s, age = _planted(rng, 600, sep_young=1.3, sep_old=0.4)
    report = HealthcareFairnessAudit.from_scores(y, s, {"age": age}).run()
    df = report.to_dataframe()
    auc_young = df[(df.attribute == "age") & (df.group == "young")]["auc"].iloc[0]
    auc_old = df[(df.attribute == "age") & (df.group == "old")]["auc"].iloc[0]
    assert auc_young > auc_old


def test_report_dataframe_has_expected_columns():
    rng = np.random.default_rng(1)
    y, s, age = _planted(rng, 200, 1.0, 0.5)
    report = HealthcareFairnessAudit.from_scores(y, s, {"age": age}).run()
    df = report.to_dataframe()
    assert {
        "attribute",
        "group",
        "n",
        "auc",
        "ci_lower",
        "ci_upper",
        "ece",
        "brier",
        "f1",
    }.issubset(df.columns)
    assert isinstance(report, HealthcareReport)


def test_summary_returns_string_without_printing(capsys):
    rng = np.random.default_rng(2)
    y, s, age = _planted(rng, 400, 1.4, 0.3)
    report = HealthcareFairnessAudit.from_scores(y, s, {"age": age}).run()
    text = report.summary()
    assert isinstance(text, str)
    assert "age" in text
    assert capsys.readouterr().out == ""  # no print side effect


def test_model_path_uses_predict_proba():
    rng = np.random.default_rng(3)
    n = 400
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    sex = np.where(rng.random(n) < 0.5, "m", "f")

    class _Stub:
        def predict_proba(self, X):
            p = 1.0 / (1.0 + np.exp(-X[:, 0]))
            return np.column_stack([1 - p, p])

    report = HealthcareFairnessAudit(_Stub(), X, y, {"sex": sex}).run()
    assert not report.to_dataframe().empty


def test_missing_predict_proba_raises():
    with pytest.raises(ValueError, match="predict_proba"):
        HealthcareFairnessAudit(
            object(),
            np.zeros((4, 2)),
            np.array([0, 1, 0, 1]),
            {"g": np.array(["a", "a", "b", "b"])},
        ).run()


def test_single_class_subgroup_raises_attribute_named_error():
    # Overall y has both classes; subgroup "old" is single-class. run() must raise a clear
    # error that names the protected attribute, not a bare low-level message.
    y = np.array([0, 1, 0, 1, 1, 1, 1, 1])
    s = np.array([0.2, 0.8, 0.3, 0.7, 0.6, 0.6, 0.6, 0.6])
    age = np.array(["young", "young", "young", "young", "old", "old", "old", "old"])
    with pytest.raises(ValueError, match="age"):
        HealthcareFairnessAudit.from_scores(y, s, {"age": age}).run()


FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "healthcare_subsample.csv")


def test_golden_fixture_reproduces_published_direction_and_magnitude():
    """Regression test on a SYNTHETIC subsample (NOT real BRFSS data). Asserts the published
    DIRECTION (elderly AUC < young) and APPROXIMATE MAGNITUDE (gap ~= 0.135 within a
    tolerance band), plus significance. The published p<0.001 was computed on n=1,285,783;
    this fixture is ~1,200 synthetic rows, so significance is asserted at the conservative
    threshold p<0.01. See tests/fixtures/README.md."""
    df = pd.read_csv(FIXTURE, comment="#")
    report = HealthcareFairnessAudit.from_scores(
        df["y_true"].to_numpy(),
        df["y_score"].to_numpy(),
        {"age_group": df["age_group"].to_numpy()},
    ).run()
    out = report.to_dataframe().set_index("group")
    auc_young = out.loc["young", "auc"]
    auc_elderly = out.loc["elderly", "auc"]
    gap = auc_young - auc_elderly
    assert auc_elderly < auc_young  # direction
    assert 0.105 <= gap <= 0.165  # approximate magnitude (published 0.135)
    assert report.results["age_group"]["p_adjusted"][0] < 0.01  # significance


def _small_report():
    rng = np.random.default_rng(7)
    y, s, age = _planted(rng, 300, 1.2, 0.5)
    return HealthcareFairnessAudit.from_scores(y, s, {"age": age}).run()


def test_plot_auc_forest_returns_figure():
    report = _small_report()
    fig = report.plot_auc_forest()
    assert fig is not None
    assert len(fig.axes) >= 1
    # attribute-filter path
    fig2 = report.plot_auc_forest(attribute="age")
    assert fig2 is not None


def test_plot_calibration_returns_real_reliability_figure():
    fig = _small_report().plot_calibration()
    assert fig is not None
    assert len(fig.axes) >= 1
    # reliability_diagram draws the perfect-calibration diagonal plus a curve per group
    assert len(fig.axes[0].lines) >= 2


def test_to_pdf_writes_nonempty_file(tmp_path):
    path = tmp_path / "report.pdf"
    _small_report().to_pdf(str(path))
    assert path.exists()
    assert path.stat().st_size > 0


def test_shap_summary_without_shap_raises_informative(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _no_shap(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("shap not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_shap)

    rng = np.random.default_rng(8)
    X = rng.normal(size=(50, 3))
    y = (X[:, 0] > 0).astype(int)

    class _Stub:
        def predict_proba(self, X):
            p = 1.0 / (1.0 + np.exp(-X[:, 0]))
            return np.column_stack([1 - p, p])

    audit = HealthcareFairnessAudit(_Stub(), X, y, {"g": np.array(["a"] * 25 + ["b"] * 25)})
    with pytest.raises(ImportError, match=r"fairscope\[shap\]"):
        audit.shap_summary()
