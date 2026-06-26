import numpy as np
import pytest

import fairscope
import fairscope.nlp as nlp


def _data(rng):
    yk = rng.integers(0, 4, 200)
    yt = rng.integers(0, 4, 200)

    def probs(y, strength):
        logits = rng.normal(0, 1, (len(y), 4))
        logits[np.arange(len(y)), y] += strength
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    return {
        "kaggle": {"y_true": yk, "probs": probs(yk, 3.0)},
        "twitter": {"y_true": yt, "probs": probs(yt, 0.5)},
    }


def test_nlp_public_api_exported():
    expected = {
        "CPFEProtocol",
        "CPFEReport",
        "macro_auc",
        "macro_f1",
        "multiclass_ece",
        "per_class_disparate_impact",
        "per_class_equalized_odds",
        "bootstrap_macro_auc_test",
        "jaccard_topk",
        "token_saliency",
    }
    assert expected.issubset(set(nlp.__all__))
    for name in expected:
        assert hasattr(nlp, name)


def test_dispatch_routes_nlp():
    from fairscope.nlp import CPFEProtocol

    rng = np.random.default_rng(0)
    audit = fairscope.FairnessAudit(
        None, domain="nlp", platform_data=_data(rng), reference="kaggle", n_classes=4
    )
    assert isinstance(audit, CPFEProtocol)


def test_dispatch_unknown_domain_still_raises():
    with pytest.raises(ValueError, match="domain"):
        fairscope.FairnessAudit(None, domain="bogus")
