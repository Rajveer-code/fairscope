"""One-call cross-node fairness audit. Composes fairscope.core; invents no statistics.

Per node: DeLong AUC confidence interval -> Expected Calibration Error -> Brier/F1.
Across nodes: max-min AUC gap and Bonferroni-corrected pairwise (unpaired) DeLong tests.
An optional per-node recalibration step reports pre/post ECE. Mirrors the cross-node
evaluation in the privacy-preserving federated-learning study.

IMPORTANT: this module audits per-node PREDICTIONS only. It does not train models, perform
secure aggregation, or provide any privacy guarantee.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score

from ..core import (
    bonferroni,
    delong_auc_ci,
    delong_unpaired_test,
    expected_calibration_error,
)


class FederatedFairnessAudit:
    """Audit per-node predictions for cross-node fairness.

    Parameters
    ----------
    node_data : dict ``{node_name: (y_true, y_score)}`` where ``y_score`` is the
        positive-class probability for that node's evaluation samples.
    n_bins : int, ECE bin count.
    alpha : float, significance level for DeLong CIs and the Bonferroni correction.
    """

    def __init__(self, node_data, *, n_bins=10, alpha=0.05):
        self.node_data = {
            k: (np.asarray(y), np.asarray(s, dtype=float)) for k, (y, s) in node_data.items()
        }
        self.n_bins = n_bins
        self.alpha = alpha

    def run(self) -> FederatedReport:
        per_node = {}
        for node, (y, s) in self.node_data.items():
            if len(np.unique(y)) < 2:
                raise ValueError(f"node {node!r} has a single class; AUC undefined")
            ci = delong_auc_ci(y, s, alpha=self.alpha)
            ece = expected_calibration_error(y, s, n_bins=self.n_bins)
            per_node[node] = {
                "ci": ci,
                "ece": float(ece),
                "n": int(len(y)),
                "brier": float(brier_score_loss(y, s)),
                "f1": float(f1_score(y, (s >= 0.5).astype(int), zero_division=0)),
            }
        nodes = sorted(per_node)
        pairs, pvals = [], []
        for a, b in combinations(nodes, 2):
            ya, sa = self.node_data[a]
            yb, sb = self.node_data[b]
            res = delong_unpaired_test(ya, sa, yb, sb)
            pairs.append((a, b))
            pvals.append(res["p_value"])
        corrected = bonferroni(np.array(pvals), alpha=self.alpha) if pvals else None
        return FederatedReport(
            per_node,
            pairs,
            pvals,
            corrected,
            node_data=self.node_data,
            alpha=self.alpha,
            n_bins=self.n_bins,
        )


class FederatedReport:
    """Holds cross-node audit results; renders tables (and plots/PDF in later tasks)."""

    def __init__(self, per_node, pairs, pvals, corrected, *, node_data, alpha, n_bins):
        self.per_node = per_node
        self.pairs = pairs
        self.pvals = pvals
        self.corrected = corrected
        self._node_data = node_data
        self.alpha = alpha
        self.n_bins = n_bins

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for node, r in self.per_node.items():
            ci = r["ci"]
            rows.append(
                {
                    "node": node,
                    "n": r["n"],
                    "auc": ci["auc"],
                    "ci_lower": ci["ci_lower"],
                    "ci_upper": ci["ci_upper"],
                    "ece": r["ece"],
                    "brier": r["brier"],
                    "f1": r["f1"],
                }
            )
        return pd.DataFrame(rows)

    def disparity(self) -> dict:
        """Cross-node AUC disparity summary."""
        aucs = {n: r["ci"]["auc"] for n, r in self.per_node.items()}
        hi = max(aucs, key=aucs.get)
        lo = min(aucs, key=aucs.get)
        return {
            "max_auc_gap": aucs[hi] - aucs[lo],
            "best": hi,
            "worst": lo,
            "worst_pair": (lo, hi),
        }

    def summary(self) -> str:
        lines = [self.to_dataframe().to_string(index=False)]
        d = self.disparity()
        lines.append(f"cross-node AUC gap: {d['best']} vs {d['worst']} = {d['max_auc_gap']:.3f}")
        if self.corrected is not None:
            for (a, b), padj, rej in zip(
                self.pairs, self.corrected["adjusted"], self.corrected["reject"]
            ):
                if rej:
                    lines.append(f"  {a} vs {b}: Bonferroni-adjusted p={padj:.4g} (significant)")
        return "\n".join(lines)
