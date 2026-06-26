"""Reproducible source of the CPFE numbers reported in paper/paper.md Section 4.2.

SYNTHETIC two-platform fixture (NOT real data): a high-signal reference platform ("kaggle")
and a platform ("twitter") with an injected over-confident-but-wrong shift. Deterministic.

Run: python paper/cpfe_demo.py
"""

import numpy as np

from fairscope.nlp.attribution import jaccard_topk
from fairscope.nlp.cross_platform import CPFEProtocol

SEED = 20260627


def _probs(rng, y, strength, n_classes=4):
    lg = rng.normal(0, 1, (len(y), n_classes))
    lg[np.arange(len(y)), y] += strength
    e = np.exp(lg - lg.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _shifted(rng, y, n_classes=4):
    fake = (y + rng.integers(1, n_classes, len(y))) % n_classes
    lg = rng.normal(0, 1, (len(y), n_classes))
    lg[np.arange(len(y)), fake] += 3.0
    e = np.exp(lg - lg.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def main():
    rng = np.random.default_rng(SEED)
    yk = rng.integers(0, 4, 3000)
    yt = rng.integers(0, 4, 2500)
    data = {
        "kaggle": {"y_true": yk, "probs": _probs(rng, yk, 3.0)},
        "twitter": {"y_true": yt, "probs": _shifted(rng, yt)},
    }
    rep = CPFEProtocol(data, reference="kaggle", n_classes=4, n_boot=1000).run()
    df = rep.to_dataframe().set_index("platform")
    sig = rep.significance["twitter"]
    di = rep.equity["twitter"]["disparate_impact"]
    sa = {"sad": 0.9, "tired": 0.8, "hopeless": 0.7, "down": 0.6, "empty": 0.5}
    sb = {"market": 0.9, "price": 0.8, "stock": 0.7, "hopeless": 0.6, "rally": 0.5}

    print(
        f"axis1 macro_auc: kaggle={df.loc['kaggle', 'macro_auc']:.3f} "
        f"twitter={df.loc['twitter', 'macro_auc']:.3f} "
        f"delta_auc_pct={df.loc['twitter', 'delta_auc_pct']:.1f}"
    )
    print(
        f"axis1 macro_f1: kaggle={df.loc['kaggle', 'macro_f1']:.3f} "
        f"twitter={df.loc['twitter', 'macro_f1']:.3f}"
    )
    print(f"axis2 ece: kaggle={df.loc['kaggle', 'ece']:.3f} twitter={df.loc['twitter', 'ece']:.3f}")
    print(
        f"axis3 sig: delta={sig['delta']:.3f} z={sig['z']:.1f} "
        f"p_adjusted={sig['p_adjusted']:.2e} reject={sig['reject']}"
    )
    print("axis4 di: " + " ".join(f"c{c}={v:.3f}" for c, v in di.items()))
    print(f"axis5 jaccard@5={jaccard_topk(sa, sb, 5):.3f}")


if __name__ == "__main__":
    main()
