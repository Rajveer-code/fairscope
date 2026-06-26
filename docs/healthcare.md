# Healthcare audit

`HealthcareFairnessAudit` is a one-call clinical fairness audit. It composes the `core`
primitives — it invents no statistics — into a per-subgroup report for each protected
attribute.

## What it computes

For each protected attribute, and each subgroup within it:

- per-subgroup **AUC with a DeLong confidence interval**
- per-subgroup **Expected Calibration Error**
- per-subgroup **Brier score** and **F1**
- **Bonferroni-corrected pairwise** (unpaired) DeLong tests across the attribute's subgroups
  (subgroups are disjoint samples, so the comparison is unpaired)

## Two entry points

```python
from fairscope.healthcare import HealthcareFairnessAudit

# 1) From a fitted model with predict_proba:
report = HealthcareFairnessAudit(
    model, X_test, y_test, protected_attr={"age_group": ages, "sex": sex}
).run()

# 2) From precomputed positive-class probabilities (no model needed):
report = HealthcareFairnessAudit.from_scores(
    y_true, y_score, protected_attr={"age_group": ages}
).run()
```

## The report

```python
report.to_dataframe()      # tidy per-subgroup table
report.summary()           # text summary; flags the largest gap and significant differences
report.plot_auc_forest()   # forest plot of per-subgroup AUC with DeLong CIs
report.plot_calibration()  # reliability curves per subgroup
report.to_pdf("report.pdf")  # multi-page PDF (matplotlib only)
```

An optional SHAP feature summary is available via `audit.shap_summary()` when the
`fairscope[shap]` extra is installed and a model is provided.

See [Getting started](getting-started.md) for a runnable example on the committed fixture.
