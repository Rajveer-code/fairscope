"""The five-axis Cross-Platform Fairness Evaluation (CPFE) protocol (Pall & Yadav).

Axes 1-4 run on precomputed per-platform outputs (no torch); attribution stability
(axis 5) is provided separately via ``fairscope.nlp.attribution`` behind the optional
``fairscope[nlp]`` extra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import bonferroni
from .metrics import (
    macro_auc,
    macro_f1,
    multiclass_ece,
    per_class_disparate_impact,
    per_class_equalized_odds,
)
from .significance import bootstrap_macro_auc_test

# Reference bands EXPLICITLY STATED in the CPFE paper (descriptive diagnostic, NOT
# regulatory standards). P4 declines to set a delta-AUC decision threshold (Section 6.6),
# so that one is a constructor argument with an illustrative default instead.
DI_FOUR_FIFTHS = 0.80  # P4 Sec 4.4: DI < 0.80 violates the four-fifths rule
DI_SEVERE = 0.50  # P4 Sec 4.4: DI < 0.50 is a severe disparity
ECE_WELL_CALIBRATED = 0.10  # P4 Suppl. Fig. S2: ECE < 0.10 well-calibrated
ECE_MODERATE = 0.20  # P4 Suppl. Fig. S2: ECE > 0.20 moderate miscalibration
JACCARD_INSTABILITY = 0.20  # P4 Suppl. Fig. S7: J < 0.20 attribution instability


class CPFEProtocol:
    """Run the CPFE five-axis evaluation over precomputed per-platform outputs.

    Parameters
    ----------
    platform_data : dict ``{name: {"y_true": array, "probs": (n, n_classes) array}}``.
    reference : the within-platform name (e.g. the training platform).
    n_classes : number of classes.
    delta_auc_pct_max : ILLUSTRATIVE macro-AUC-drop screening limit (percent) used by
        ``CPFEReport.deployment_readiness``. NOT a published cutoff: P4 Section 6.6 declines
        to set one (observed drops were 28.6-39.5%); the default echoes that ">30%" magnitude
        and is labelled illustrative everywhere it surfaces.
    """

    def __init__(
        self,
        platform_data,
        reference,
        n_classes,
        *,
        n_bins=10,
        alpha=0.05,
        n_boot=2000,
        delta_auc_pct_max=30.0,
    ):
        if reference not in platform_data:
            raise ValueError(f"reference platform {reference!r} not in platform_data")
        self.platform_data = platform_data
        self.reference = reference
        self.n_classes = n_classes
        self.n_bins = n_bins
        self.alpha = alpha
        self.n_boot = n_boot
        self.delta_auc_pct_max = delta_auc_pct_max

    def run(self) -> CPFEReport:
        ref = self.platform_data[self.reference]
        ref_auc = macro_auc(ref["y_true"], ref["probs"])
        others = [p for p in self.platform_data if p != self.reference]

        performance = {}
        for name, d in self.platform_data.items():
            a = macro_auc(d["y_true"], d["probs"])
            performance[name] = {
                "macro_auc": a,
                "macro_f1": macro_f1(d["y_true"], d["probs"]),
                "ece": multiclass_ece(d["y_true"], d["probs"], self.n_bins),
                "delta_auc_pct": 100.0 * (a - ref_auc) / ref_auc,
            }

        significance, equity, raw_p = {}, {}, []
        for name in others:
            d = self.platform_data[name]
            sig = bootstrap_macro_auc_test(
                ref["y_true"],
                ref["probs"],
                d["y_true"],
                d["probs"],
                n_boot=self.n_boot,
                random_state=0,
            )
            significance[name] = sig
            raw_p.append(sig["p_value"])
            equity[name] = {
                "disparate_impact": per_class_disparate_impact(
                    ref["probs"], d["probs"], self.n_classes
                ),
                "equalized_odds": per_class_equalized_odds(
                    ref["y_true"], ref["probs"], d["y_true"], d["probs"], self.n_classes
                ),
            }
        if raw_p:
            adj = bonferroni(np.array(raw_p), alpha=self.alpha)
            for name, padj, rej in zip(others, adj["adjusted"], adj["reject"]):
                significance[name]["p_adjusted"] = float(padj)
                significance[name]["reject"] = bool(rej)

        return CPFEReport(performance, significance, equity, self.reference, self.delta_auc_pct_max)


class CPFEReport:
    """Holds the five-axis results and renders tables and a deployment-readiness diagnostic."""

    def __init__(self, performance, significance, equity, reference, delta_auc_pct_max):
        self.performance = performance
        self.significance = significance
        self.equity = equity
        self.reference = reference
        self.delta_auc_pct_max = delta_auc_pct_max

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{"platform": name, **m} for name, m in self.performance.items()])

    def deployment_readiness(self):
        """Structured per-axis, per-platform screening DIAGNOSTIC -- NOT a deployment
        decision. Following the CPFE paper (Sections 6.5-6.6), cross-platform degradation
        is an informative diagnostic, not definitive evidence of bias.

        Thresholds: calibration uses P4's stated ECE bands (Suppl. Fig. S2); equity uses
        P4's four-fifths rule; discrimination uses an ILLUSTRATIVE ``delta_auc_pct_max``
        (P4 Section 6.6 declines to set a published cutoff). Returns
        ``{platform: {"ready": bool, "axes": {axis: {pass, value, threshold, source, reason}}}}``.
        """
        verdict = {}
        for name, perf in self.performance.items():
            if name == self.reference:
                continue
            drop = -perf["delta_auc_pct"]
            ece = perf["ece"]
            di = self.equity.get(name, {}).get("disparate_impact", {})
            violations = sorted(c for c, v in di.items() if v < DI_FOUR_FIFTHS)
            severe = sorted(c for c, v in di.items() if v < DI_SEVERE)
            axes = {
                "discrimination": {
                    "pass": drop <= self.delta_auc_pct_max,
                    "value": drop,
                    "threshold": self.delta_auc_pct_max,
                    "source": "illustrative (not a published cutoff; P4 Section 6.6)",
                    "reason": f"macro-AUC drop {drop:.1f}% vs reference "
                    f"(illustrative limit {self.delta_auc_pct_max:.0f}%)",
                },
                "calibration": {
                    "pass": ece < ECE_WELL_CALIBRATED,
                    "value": ece,
                    "threshold": ECE_WELL_CALIBRATED,
                    "source": "P4 Suppl. Fig. S2",
                    "reason": f"ECE {ece:.3f} (well-calibrated < {ECE_WELL_CALIBRATED}; "
                    f"moderate miscalibration > {ECE_MODERATE})",
                },
                "equity": {
                    "pass": len(violations) == 0,
                    "value": {"violations": violations, "severe": severe},
                    "threshold": DI_FOUR_FIFTHS,
                    "source": "P4 four-fifths rule (Sec 4.4)",
                    "reason": (
                        f"four-fifths violations (DI<{DI_FOUR_FIFTHS}) for classes "
                        f"{violations}; severe (<{DI_SEVERE}) {severe}"
                    )
                    if violations
                    else "no four-fifths violations",
                },
            }
            verdict[name] = {"ready": all(a["pass"] for a in axes.values()), "axes": axes}
        return verdict
