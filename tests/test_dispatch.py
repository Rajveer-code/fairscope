import numpy as np
import pytest

import fairscope


class _Stub:
    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-X[:, 0]))
        return np.column_stack([1 - p, p])


def test_dispatch_routes_healthcare():
    from fairscope.healthcare import HealthcareFairnessAudit

    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] > 0).astype(int)
    g = np.array(["a"] * 50 + ["b"] * 50)

    audit = fairscope.FairnessAudit(
        _Stub(), domain="healthcare", X_test=X, y_test=y, protected_attr={"g": g}
    )
    assert isinstance(audit, HealthcareFairnessAudit)
    assert not audit.run().to_dataframe().empty


def test_dispatch_routes_federated():
    from fairscope.federated import FederatedFairnessAudit

    rng = np.random.default_rng(1)

    def _node(sep):
        y = np.r_[np.ones(100, int), np.zeros(100, int)]
        raw = np.r_[rng.normal(sep, 1, 100), rng.normal(0, 1, 100)]
        return y, 1.0 / (1.0 + np.exp(-raw))

    audit = fairscope.FairnessAudit(
        None, domain="federated", node_data={"a": _node(1.2), "b": _node(0.5)}
    )
    assert isinstance(audit, FederatedFairnessAudit)
    assert not audit.run().to_dataframe().empty


def test_dispatch_unknown_domain_raises():
    with pytest.raises(ValueError, match="domain"):
        fairscope.FairnessAudit(None, domain="nope")


def test_fairness_audit_is_exported():
    assert hasattr(fairscope, "FairnessAudit")
    assert "FairnessAudit" in fairscope.__all__
