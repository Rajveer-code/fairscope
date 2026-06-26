# Getting started

## Install

```bash
pip install fairscope
```

## A real audit in a few lines

The example below is **actually runnable** — it uses the synthetic fairness fixture committed
to the repository. Clone the repo and run it from the repository root:

```bash
git clone https://github.com/Rajveer-code/fairscope
cd fairscope
pip install -e .
```

```python
import pandas as pd

from fairscope.healthcare import HealthcareFairnessAudit

# A SYNTHETIC fixture committed in the repo (NOT real BRFSS data). It is seed-generated to
# reproduce the direction and approximate magnitude of a published age-disparity finding
# (elderly < young AUC). See tests/fixtures/README.md.
df = pd.read_csv("tests/fixtures/healthcare_subsample.csv", comment="#")

report = HealthcareFairnessAudit.from_scores(
    df["y_true"].to_numpy(),
    df["y_score"].to_numpy(),
    {"age_group": df["age_group"].to_numpy()},
).run()

# Per-subgroup AUC with DeLong CIs, ECE, Brier, F1:
print(report.to_dataframe())

# A readable summary that flags the largest gap and any Bonferroni-significant difference:
print(report.summary())
```

Running this prints a per-subgroup table and a summary reporting that the **elderly** group
has a substantially lower AUC than the **young** group, with the gap flagged as statistically
significant after Bonferroni correction — the audit recovering the published finding's
direction on the synthetic fixture.

!!! note
    The fixture is synthetic and small; it demonstrates the audit *pipeline*. It does not
    reproduce the full published run (which used a cohort of ~1.28M records). See
    [`tests/fixtures/README.md`](https://github.com/Rajveer-code/fairscope/blob/main/tests/fixtures/README.md).

## Using your own model

If you have a fitted scikit-learn-style classifier with `predict_proba`, pass it directly:

```python
from fairscope import FairnessAudit

report = FairnessAudit(
    model, domain="healthcare", X_test=X_test, y_test=y_test,
    protected_attr={"age_group": age_groups, "sex": sex},
).run()
```

For cross-platform NLP evaluation, see the [CPFE protocol](cpfe.md).
