# Lending audit

`LendingFairnessAudit` is a one-call mortgage-lending fairness audit. It has two parts:

1. a **descriptive** annual approval-gap analysis (composes `core`, no causal claim), and
2. an optional **subgroup CATE** estimate of the protected attribute's effect on approval,
   via Causal Forest DML.

## Annual approval-gap analysis

For each year and group, it reports the approval rate and the symmetric **disparate impact**
versus a reference group (`core.disparate_impact`). Group direction stays readable from the
`approval_rate` column.

```python
from fairscope.lending import LendingFairnessAudit

report = LendingFairnessAudit.from_outcomes(
    approved, group, year, reference="reference"
).run()

report.to_dataframe()   # year, group, n, approval_rate, disparate_impact
report.summary()        # text summary; flags the worst gap
```

## Subgroup CATE (optional, `fairscope[lending]`)

`estimate_cate` wraps `econml.dml.CausalForestDML` (the estimator used in the lending paper)
to estimate the conditional average treatment effect of the protected attribute on approval.
The treatment defaults to `group != reference` and the outcome to the approval labels.

```python
audit = LendingFairnessAudit.from_outcomes(approved, group, year, reference="reference")
res = audit.estimate_cate(X)        # X: heterogeneity features
res["ate"], res["effect"], res["effect_interval"]
```

!!! warning "Causal claims are conditional"
    The CATE is valid **only under the DML assumptions** — unconfoundedness given the
    supplied features `X`, and overlap. It estimates an effect under those assumptions; it
    does not, on its own, prove discrimination. `econml` is an optional dependency
    (`pip install fairscope[lending]`).

## Not in the library: RDD / DiD

The lending papers also use regression-discontinuity (RDD) and difference-in-differences
(DiD) identification strategies. These are **paper-specific** and are intentionally **not**
shipped as `fairscope` primitives (one paper's appendix RDD/DiD also failed its validity
tests). They are illustrated, clearly caveated, in the lending replication notebook — as the
papers' strategies on synthetic data, not as reusable audit functions.

Route it through the top-level dispatcher with
`FairnessAudit(None, domain="lending", approved=..., group=..., year=..., reference=...)`.
