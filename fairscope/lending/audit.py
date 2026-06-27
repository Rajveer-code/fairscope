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
