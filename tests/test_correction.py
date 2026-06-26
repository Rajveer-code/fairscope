import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from fairscope.core.correction import benjamini_hochberg, bonferroni


def test_bonferroni_matches_statsmodels():
    p = np.array([0.001, 0.01, 0.02, 0.04, 0.2, 0.5])
    reject, padj, _, _ = multipletests(p, alpha=0.05, method="bonferroni")
    out = bonferroni(p, alpha=0.05)
    assert np.allclose(out["adjusted"], padj)
    assert np.array_equal(out["reject"], reject)
    assert np.isclose(out["threshold"], 0.05 / len(p))


def test_bh_matches_statsmodels():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.2, 0.6, 0.007])
    reject, padj, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    out = benjamini_hochberg(p, alpha=0.05)
    assert np.allclose(out["adjusted"], padj)
    assert np.array_equal(out["reject"], reject)


def test_rejects_out_of_range_pvalues():
    with pytest.raises(ValueError):
        bonferroni([0.1, 1.5])
