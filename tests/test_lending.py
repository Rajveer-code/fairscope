import numpy as np
import pandas as pd

from fairscope.lending import LendingFairnessAudit, LendingReport


def _planted_annual(rng, years, n_per_year, base_rate, minority_penalty):
    """Synthetic: minority approval rate = base_rate - minority_penalty[year]."""
    rows = []
    for i, yr in enumerate(years):
        for grp, rate in (
            ("reference", base_rate),
            ("minority", base_rate - minority_penalty[i]),
        ):
            approved = (rng.random(n_per_year) < rate).astype(int)
            rows.append(pd.DataFrame({"year": yr, "group": grp, "approved": approved}))
    return pd.concat(rows, ignore_index=True)


def test_annual_gap_detects_direction_and_columns():
    rng = np.random.default_rng(0)
    df = _planted_annual(rng, [2019, 2020, 2021], 4000, 0.70, [0.10, 0.15, 0.20])
    report = LendingFairnessAudit.from_outcomes(
        df["approved"].to_numpy(),
        df["group"].to_numpy(),
        df["year"].to_numpy(),
        reference="reference",
    ).run()
    out = report.to_dataframe()
    assert {"year", "group", "n", "approval_rate", "disparate_impact"}.issubset(out.columns)
    di = out[out.group == "minority"].sort_values("year")["disparate_impact"].to_numpy()
    assert (di < 1.0).all()  # symmetric DI < 1 whenever rates differ
    assert di[0] > di[-1]  # gap widens over the years
    # direction is readable from approval_rate
    minority = out[out.group == "minority"].sort_values("year")
    reference = out[out.group == "reference"].sort_values("year")
    assert (minority["approval_rate"].to_numpy() < reference["approval_rate"].to_numpy()).all()
    assert isinstance(report, LendingReport)


def test_summary_returns_text():
    rng = np.random.default_rng(1)
    df = _planted_annual(rng, [2020, 2021], 3000, 0.65, [0.12, 0.18])
    text = (
        LendingFairnessAudit.from_outcomes(
            df["approved"].to_numpy(),
            df["group"].to_numpy(),
            df["year"].to_numpy(),
            reference="reference",
        )
        .run()
        .summary()
    )
    assert isinstance(text, str) and "minority" in text
