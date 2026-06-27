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
    isotonic_recalibrate,
    temperature_scale,
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

    def recalibrate(self, method="temperature") -> dict:
        """Recalibrate each node on its own (y, score) and report pre/post ECE.

        ``method`` is 'temperature' (Guo et al., 2017) or 'isotonic' (Zadrozny &
        Elkan, 2002) — both standard methods from ``fairscope.core``. Temperature
        scaling operates on logits, so the per-node probabilities are converted to
        logits first. Returns ``{node: {"ece_pre": float, "ece_post": float}}``.

        NOTE: this fits and evaluates on the same per-node data (an in-sample
        diagnostic). For a deployment estimate, recalibrate on a held-out split.
        """
        if method not in ("temperature", "isotonic"):
            raise ValueError(f"unknown method: {method!r}; use 'temperature' or 'isotonic'")
        out = {}
        for node, (y, s) in self._node_data.items():
            pre = expected_calibration_error(y, s, n_bins=self.n_bins)
            if method == "temperature":
                p = np.clip(s, 1e-7, 1 - 1e-7)
                logits = np.log(p / (1 - p))
                _, s_cal = temperature_scale(logits, y)
            else:  # isotonic
                _, s_cal = isotonic_recalibrate(s, y)
            post = expected_calibration_error(y, s_cal, n_bins=self.n_bins)
            out[node] = {"ece_pre": float(pre), "ece_post": float(post)}
        return out

    def plot_auc_forest(self):
        """Forest plot of per-node AUC with DeLong 95% CIs. Returns a Figure."""
        import matplotlib.pyplot as plt

        df = self.to_dataframe()
        ys = np.arange(len(df))
        xerr = np.vstack([df.auc - df.ci_lower, df.ci_upper - df.auc])
        fig, ax = plt.subplots(figsize=(6, 0.5 * len(df) + 1.5))
        ax.errorbar(df.auc, ys, xerr=xerr, fmt="o", capsize=3)
        ax.set_yticks(ys)
        ax.set_yticklabels(df.node)
        ax.set_xlabel("AUC (95% DeLong CI)")
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_title("Per-node discrimination")
        fig.tight_layout()
        return fig

    def plot_calibration(self):
        """Per-node reliability curves drawn with ``core.reliability_diagram``
        (federated retains each node's (y, score), so these are true curves).
        Returns a Figure."""
        from ..core import reliability_diagram

        ys, ss, labels = [], [], []
        for node, (y, s) in self._node_data.items():
            ys.append(np.asarray(y))
            ss.append(np.asarray(s))
            labels.append(np.full(len(y), node))
        return reliability_diagram(
            np.concatenate(ys),
            np.concatenate(ss),
            groups=np.concatenate(labels),
            n_bins=self.n_bins,
        )

    def to_pdf(self, path):
        """Write a multi-page PDF: summary, per-node AUC forest, per-node calibration.
        Uses matplotlib only (no extra dependency)."""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(path) as pdf:
            fig0, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            ax.text(0.02, 0.98, self.summary(), family="monospace", va="top", fontsize=8)
            pdf.savefig(fig0)
            plt.close(fig0)

            forest = self.plot_auc_forest()
            pdf.savefig(forest)
            plt.close(forest)

            cal = self.plot_calibration()
            cal.axes[0].set_title("Per-node calibration")
            pdf.savefig(cal)
            plt.close(cal)

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
