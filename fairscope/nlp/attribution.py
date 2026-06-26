"""Axis 5 of CPFE: attribution stability via Jaccard overlap of top-K gradient-saliency
token sets across platforms (CPFE paper). The Jaccard computation is dependency-free; the
gradient-saliency extraction uses Captum and requires ``pip install fairscope[nlp]``.
"""

from __future__ import annotations


def jaccard_topk(saliency_a, saliency_b, k):
    """Jaccard overlap of the top-k tokens by saliency: ``|topK(A) ∩ topK(B)| / |union|``.

    ``saliency_*`` map token -> saliency score. Returns 0.0 if both are empty.

    Examples
    --------
    >>> jaccard_topk({"a": 0.9, "b": 0.8}, {"a": 0.7, "c": 0.6}, k=2)
    0.3333333333333333
    """
    top_a = set(sorted(saliency_a, key=saliency_a.get, reverse=True)[:k])
    top_b = set(sorted(saliency_b, key=saliency_b.get, reverse=True)[:k])
    union = top_a | top_b
    return len(top_a & top_b) / len(union) if union else 0.0


def token_saliency(model, tokenizer, text, target=None):
    """Per-token gradient saliency ``s_i = ‖∂P(y|x)/∂E_i‖₂`` via Captum (optional).
    Requires ``pip install fairscope[nlp]``. Returns ``{token: saliency}``."""
    try:
        import captum  # noqa: F401  (captum depends on torch; one import gates the extra)
    except ImportError as exc:
        raise ImportError(
            "token_saliency requires the optional dependency: pip install fairscope[nlp]"
        ) from exc
    return _captum_token_saliency(model, tokenizer, text, target)  # pragma: no cover


def _captum_token_saliency(model, tokenizer, text, target):  # pragma: no cover - needs nlp extra
    from captum.attr import Saliency

    enc = tokenizer(text, return_tensors="pt")
    embeddings = model.get_input_embeddings()(enc["input_ids"])
    embeddings.requires_grad_(True)

    def forward(emb):
        return model(inputs_embeds=emb, attention_mask=enc["attention_mask"]).logits.softmax(-1)

    tgt = target if target is not None else int(forward(embeddings).argmax())
    grads = Saliency(forward).attribute(embeddings, target=tgt, abs=False)
    scores = grads.norm(dim=-1).squeeze(0).detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
    agg = {}
    for tok, score in zip(tokens, scores):
        agg[tok] = max(agg.get(tok, 0.0), float(score))
    return agg
