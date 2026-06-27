import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from fairscope.federated import FederatedFairnessAudit, FederatedReport


def _node(rng, n, sep):
    half = n // 2
    y = np.r_[np.ones(half, int), np.zeros(half, int)]
    raw = np.r_[rng.normal(sep, 1.0, half), rng.normal(0.0, 1.0, half)]
    s = 1.0 / (1.0 + np.exp(-raw))
    return y, s


def _nodes(rng):
    return {
        "site_a": _node(rng, 800, 1.4),  # strong
        "site_b": _node(rng, 800, 1.0),
        "site_c": _node(rng, 800, 0.4),  # weak -> cross-node gap
    }


def test_per_node_table_and_columns():
    rng = np.random.default_rng(0)
    report = FederatedFairnessAudit(_nodes(rng)).run()
    df = report.to_dataframe()
    assert {"node", "n", "auc", "ci_lower", "ci_upper", "ece", "brier", "f1"}.issubset(df.columns)
    assert isinstance(report, FederatedReport)


def test_cross_node_disparity_detected():
    rng = np.random.default_rng(1)
    report = FederatedFairnessAudit(_nodes(rng)).run()
    df = report.to_dataframe().set_index("node")
    assert df.loc["site_a", "auc"] > df.loc["site_c", "auc"]
    d = report.disparity()
    assert d["max_auc_gap"] > 0
    assert d["worst_pair"][0] in {"site_a", "site_c"}
    assert d["worst_pair"][1] in {"site_a", "site_c"}


def test_summary_returns_text_and_flags_gap():
    rng = np.random.default_rng(2)
    text = FederatedFairnessAudit(_nodes(rng)).run().summary()
    assert isinstance(text, str) and "site_" in text


def test_single_class_node_raises():
    with pytest.raises(ValueError):
        FederatedFairnessAudit({"bad": (np.ones(10, int), np.linspace(0, 1, 10))}).run()
