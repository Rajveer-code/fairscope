import os

import numpy as np
import pandas as pd
import pytest

from fairscope.lending import LendingFairnessAudit, LendingReport

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "lending_subsample.csv")


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


def test_estimate_cate_missing_econml_raises(monkeypatch):
    import builtins

    real = builtins.__import__

    def _no_econml(name, *a, **k):
        if name.startswith("econml"):
            raise ImportError("no econml")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_econml)
    rng = np.random.default_rng(2)
    n = 200
    audit = LendingFairnessAudit.from_outcomes(
        rng.integers(0, 2, n),
        np.where(rng.random(n) < 0.5, "minority", "reference"),
        np.full(n, 2021),
        reference="reference",
    )
    with pytest.raises(ImportError, match=r"fairscope\[lending\]"):
        audit.estimate_cate(np.zeros((n, 2)))


def test_estimate_cate_recovers_planted_effect():
    pytest.importorskip("econml")
    rng = np.random.default_rng(3)
    n = 2000
    X = rng.normal(size=(n, 3))  # heterogeneity covariates
    T = (rng.random(n) < 0.5).astype(int)  # protected indicator (treatment)
    tau = 0.8 * (X[:, 0] > 0)  # heterogeneous planted effect on the X0>0 stratum
    Y = (0.3 + 0.5 * X[:, 0] + tau * T + rng.normal(scale=0.3, size=n) > 0.5).astype(int)
    audit = LendingFairnessAudit.from_outcomes(
        Y, np.where(T == 1, "minority", "reference"), np.full(n, 2021), reference="reference"
    )
    # defaults: treatment derives from group != reference, outcome from approved (P1 style)
    res = audit.estimate_cate(X, n_estimators=200)
    assert res["ate"] != 0
    lo, hi = res["effect_interval"]
    assert len(lo) == len(hi) == n
    strong = X[:, 0] > 0
    assert np.median(res["effect"][strong]) > 0  # right sign on the strong stratum


def test_golden_fixture_reproduces_widening_gap():
    """Regression test on a SYNTHETIC subsample (NOT real HMDA). Asserts the planted
    direction (minority approval < reference), a four-fifths violation, and a WIDENING
    annual gap. The published HMDA runs used millions of records; this fixture is ~1,500
    synthetic rows."""
    df = pd.read_csv(FIXTURE, comment="#")
    out = (
        LendingFairnessAudit.from_outcomes(
            df["approved"].to_numpy(),
            df["group"].to_numpy(),
            df["year"].to_numpy(),
            reference="reference",
        )
        .run()
        .to_dataframe()
    )
    mino = out[out.group == "minority"].sort_values("year")
    di = mino["disparate_impact"].to_numpy()
    assert (di < 0.80).all()  # four-fifths violated every year
    assert di[0] > di[1] > di[2]  # gap widens year over year
    assert di[-1] < 0.50  # severe disparity by the final year
    ref = out[out.group == "reference"].sort_values("year")["approval_rate"].to_numpy()
    assert (mino["approval_rate"].to_numpy() < ref).all()  # direction in the rates


def test_golden_fixture_cate_sign():
    pytest.importorskip("econml")
    df = pd.read_csv(FIXTURE, comment="#")
    audit = LendingFairnessAudit.from_outcomes(
        df["approved"].to_numpy(),
        df["group"].to_numpy(),
        df["year"].to_numpy(),
        reference="reference",
    )
    res = audit.estimate_cate(df[["x0", "x1", "x2"]].to_numpy(), n_estimators=100)
    x0 = df["x0"].to_numpy()
    eff = res["effect"]
    assert res["ate"] < 0  # minority treatment lowers approval
    # heterogeneity: more negative where x0 > 0 (the planted stratum)
    assert np.median(eff[x0 > 0]) < np.median(eff[x0 <= 0])
