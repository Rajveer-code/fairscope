"""One-call lending fairness audit.

Annual approval-gap analysis composes fairscope.core and is purely descriptive (no causal
claim). Subgroup CATE estimation (Task 2) wraps econml under stated DML assumptions.
Mirrors the analyses in P1 (Causal Forest DML, HMDA) and P2 (descriptive disparities).
Ships no HMDA data and no model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import disparate_impact


class LendingFairnessAudit:
    """Audit mortgage-approval outcomes for annual subgroup disparities.

    Parameters
    ----------
    approved : 1-D binary array (1 = approved).
    group : 1-D array of protected-group labels, aligned with ``approved``.
    year : 1-D array of years, aligned with ``approved``.
    reference : the group label to compare every other group against. Must be present
        in each year.
    alpha : significance level (reserved for the CATE step).
    """

    def __init__(self, *, approved, group, year, reference, alpha=0.05):
        self.approved = np.asarray(approved)
        self.group = np.asarray(group)
        self.year = np.asarray(year)
        self.reference = reference
        self.alpha = alpha

    @classmethod
    def from_outcomes(cls, approved, group, year, *, reference, alpha=0.05):
        """Build an audit from precomputed approval outcomes."""
        return cls(approved=approved, group=group, year=year, reference=reference, alpha=alpha)

    def _binary_treatment(self):
        return (self.group != self.reference).astype(int)

    def estimate_cate(
        self,
        X,
        *,
        treatment=None,
        outcome=None,
        model_y=None,
        model_t=None,
        n_estimators=500,
        random_state=0,
    ):
        """Per-subgroup conditional average treatment effect (CATE) of the protected
        attribute on approval, via Causal Forest DML (``econml.dml.CausalForestDML``,
        as in P1).

        The CAUSAL CLAIM IS CONDITIONAL on the DML assumptions (unconfoundedness given
        the supplied features ``X``, and overlap). This estimates an effect under those
        assumptions; it does not, on its own, prove discrimination.

        Parameters
        ----------
        X : array (n, k) of heterogeneity features.
        treatment : binary array; defaults to ``group != reference``.
        outcome : binary array; defaults to the ``approved`` outcomes.
        model_y, model_t : nuisance estimators; default to random forests.

        Returns ``{"ate", "effect", "effect_interval"}``. Requires the optional
        dependency: ``pip install fairscope[lending]``.
        """
        try:
            from econml.dml import CausalForestDML
        except ImportError as exc:  # optional dependency
            raise ImportError(
                "Subgroup CATE requires the optional dependency: " "pip install fairscope[lending]"
            ) from exc
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        T = self._binary_treatment() if treatment is None else np.asarray(treatment)
        Y = self.approved if outcome is None else np.asarray(outcome)
        X = np.asarray(X)
        est = CausalForestDML(
            model_y=model_y or RandomForestRegressor(random_state=random_state),
            model_t=model_t or RandomForestClassifier(random_state=random_state),
            discrete_treatment=True,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        est.fit(Y, T, X=X)
        lo, hi = est.effect_interval(X, alpha=self.alpha)
        return {
            "ate": float(est.ate(X)),
            "effect": est.effect(X),
            "effect_interval": (lo, hi),
        }

    def run(self) -> LendingReport:
        rows = []
        for yr in sorted(np.unique(self.year).tolist()):
            m = self.year == yr
            approved_y = self.approved[m]
            group_y = self.group[m]
            for g in sorted(np.unique(group_y).tolist()):
                sel = group_y == g
                rate = float(approved_y[sel].mean())
                di = float(disparate_impact(approved_y, group_y, g, self.reference))
                rows.append(
                    {
                        "year": yr,
                        "group": g,
                        "n": int(sel.sum()),
                        "approval_rate": rate,
                        "disparate_impact": di,
                    }
                )
        return LendingReport(pd.DataFrame(rows), reference=self.reference, alpha=self.alpha)


class LendingReport:
    """Holds the annual approval-gap table (CATE results are returned separately)."""

    def __init__(self, df, *, reference, alpha=0.05):
        self._df = df
        self.reference = reference
        self.alpha = alpha

    def to_dataframe(self) -> pd.DataFrame:
        return self._df.copy()

    def summary(self) -> str:
        lines = [self._df.to_string(index=False)]
        nonref = self._df[self._df.group != self.reference]
        if not nonref.empty:
            worst = nonref.nsmallest(1, "disparate_impact").iloc[0]
            lines.append(
                f"largest gap: {worst.group} in {int(worst.year)} "
                f"DI={worst.disparate_impact:.3f} (approval {worst.approval_rate:.3f})"
            )
        return "\n".join(lines)
