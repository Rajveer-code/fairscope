"""One-call clinical fairness audit. Composes fairscope.core; invents no statistics.

Pipeline per protected attribute: per-subgroup DeLong AUC CIs -> per-subgroup ECE ->
per-subgroup Brier/F1 -> Bonferroni-corrected pairwise (unpaired) DeLong tests across the
attribute's subgroups. Mirrors the analysis in the diabetes paper (IEEE CIPHER 2026).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from ..core import (
    bonferroni,
    delong_by_group,
    delong_unpaired_test,
    ece_by_group,
    subgroup_metrics,
)


class HealthcareFairnessAudit:
    """Audit a fitted classifier (or precomputed scores) for subgroup fairness.

    Parameters
    ----------
    model : object with ``predict_proba`` (positive-class probability in column 1), or
        ``None`` when using :meth:`from_scores`.
    X_test, y_test : test features and binary labels.
    protected_attr : dict ``{attribute_name: 1-D array of subgroup labels per sample}``.
    """

    def __init__(self, model, X_test, y_test, protected_attr, *, n_bins=10, alpha=0.05):
        self.model = model
        self.X_test = X_test
        self.y_test = np.asarray(y_test)
        self.protected_attr = protected_attr
        self.n_bins = n_bins
        self.alpha = alpha
        self._scores = None

    @classmethod
    def from_scores(cls, y_true, y_score, protected_attr, *, n_bins=10, alpha=0.05):
        """Build an audit from precomputed positive-class probabilities (no model)."""
        obj = cls(None, None, y_true, protected_attr, n_bins=n_bins, alpha=alpha)
        obj._scores = np.asarray(y_score, dtype=float)
        return obj

    def _get_scores(self):
        if self._scores is not None:
            return self._scores
        if self.model is None or not hasattr(self.model, "predict_proba"):
            raise ValueError(
                "model must implement predict_proba(); or use "
                "HealthcareFairnessAudit.from_scores(...)"
            )
        return np.asarray(self.model.predict_proba(self.X_test))[:, 1]

    def run(self) -> HealthcareReport:
        """Run the audit and return a :class:`HealthcareReport`.

        Raises a clear, attribute-named ``ValueError`` if a subgroup is single-class
        (AUC undefined) rather than letting a low-level error surface.
        """
        y = self.y_test
        s = self._get_scores()
        results = {}
        for name, groups in self.protected_attr.items():
            groups = np.asarray(groups)
            try:
                cis = delong_by_group(y, s, groups, alpha=self.alpha)
                eces = ece_by_group(y, s, groups, n_bins=self.n_bins)
                mets = subgroup_metrics(y, s, groups)
            except ValueError as exc:
                raise ValueError(f"protected attribute {name!r}: {exc}") from exc
            vals = sorted(np.unique(groups).tolist())
            pairs, pvals, deltas = [], [], []
            for a, b in combinations(vals, 2):
                ma, mb = groups == a, groups == b
                res = delong_unpaired_test(y[ma], s[ma], y[mb], s[mb])
                pairs.append((a, b))
                pvals.append(res["p_value"])
                deltas.append(res["delta"])
            corrected = bonferroni(np.array(pvals), alpha=self.alpha) if pvals else None
            results[name] = {
                "groups": vals,
                "ci": cis,
                "ece": eces,
                "metrics": mets,
                "pairs": pairs,
                "delta": deltas,
                "p_value": pvals,
                "p_adjusted": corrected["adjusted"].tolist() if corrected else [],
                "reject": corrected["reject"].tolist() if corrected else [],
            }
        return HealthcareReport(
            results, y, s, self.protected_attr, alpha=self.alpha, n_bins=self.n_bins
        )


class HealthcareReport:
    """Holds audit results and the raw (y, score, groups) needed to render reliability
    curves; provides tables (here), and plots/PDF (see plotting methods)."""

    def __init__(self, results, y_true, y_score, protected_attr, *, alpha=0.05, n_bins=10):
        self.results = results
        self.y_true = np.asarray(y_true)
        self.y_score = np.asarray(y_score)
        self.protected_attr = protected_attr
        self.alpha = alpha
        self.n_bins = n_bins

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for attr, r in self.results.items():
            for g in r["groups"]:
                ci = r["ci"][g]
                m = r["metrics"][g]
                rows.append(
                    {
                        "attribute": attr,
                        "group": g,
                        "n": m["n"],
                        "auc": ci["auc"],
                        "ci_lower": ci["ci_lower"],
                        "ci_upper": ci["ci_upper"],
                        "ece": r["ece"][g],
                        "brier": m["brier"],
                        "f1": m["f1"],
                    }
                )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Return a human-readable summary string (no print side effect)."""
        df = self.to_dataframe()
        lines = [df.to_string(index=False)]
        for attr, r in self.results.items():
            aucs = {g: r["ci"][g]["auc"] for g in r["groups"]}
            hi, lo = max(aucs, key=aucs.get), min(aucs, key=aucs.get)
            lines.append(
                f"[{attr}] largest AUC gap: {hi} {aucs[hi]:.3f} vs "
                f"{lo} {aucs[lo]:.3f} (delta={aucs[hi] - aucs[lo]:.3f})"
            )
            for (a, b), padj, rej in zip(r["pairs"], r["p_adjusted"], r["reject"]):
                if rej:
                    lines.append(f"  {a} vs {b}: Bonferroni-adjusted p={padj:.4g} (significant)")
        return "\n".join(lines)
