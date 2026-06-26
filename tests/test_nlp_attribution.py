import pytest

from fairscope.nlp.attribution import jaccard_topk, token_saliency


def test_jaccard_identical_is_one():
    a = {"sad": 0.9, "tired": 0.5, "hopeless": 0.8, "the": 0.1}
    assert jaccard_topk(a, dict(a), k=3) == 1.0


def test_jaccard_disjoint_is_zero():
    a = {"sad": 0.9, "tired": 0.8}
    b = {"market": 0.9, "stock": 0.8}
    assert jaccard_topk(a, b, k=2) == 0.0


def test_jaccard_partial_overlap():
    a = {"sad": 0.9, "tired": 0.8, "down": 0.7}
    b = {"sad": 0.95, "anxious": 0.8, "down": 0.6}
    # top-3 A={sad,tired,down}, B={sad,anxious,down}; intersection {sad,down}=2, union=4
    assert jaccard_topk(a, b, k=3) == 0.5


def test_jaccard_empty_inputs():
    assert jaccard_topk({}, {}, k=5) == 0.0


def test_token_saliency_without_captum_raises(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _no_nlp(name, *args, **kwargs):
        if name in ("captum", "torch") or name.startswith("captum."):
            raise ImportError("no nlp extra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_nlp)
    with pytest.raises(ImportError, match=r"fairscope\[nlp\]"):
        token_saliency(model=None, tokenizer=None, text="x")
