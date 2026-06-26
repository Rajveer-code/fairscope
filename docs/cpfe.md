# Cross-Platform Fairness Evaluation (CPFE)

CPFE is `fairscope`'s novel contribution: a five-axis protocol for evaluating what happens
when a model trained on one data source is deployed on another. A single AUC number cannot
distinguish the ways a model degrades across platforms; CPFE evaluates five orthogonal axes,
comparing each non-training platform against a reference.

## The five axes

1. **Discriminative performance** — macro one-vs-rest AUC and macro F1, with the relative
   change ΔAUC%.
2. **Calibration** — confidence–accuracy Expected Calibration Error per platform.
3. **Statistical significance** — a stratified bootstrap standard error on the macro-AUC
   difference between platforms (independent test sets), with Bonferroni correction.
4. **Prediction equity** — symmetric disparate impact and equalized-odds difference per class,
   treating platform membership as the group.
5. **Attribution stability** — Jaccard overlap of the top-K gradient-saliency token sets
   across platforms (requires `pip install fairscope[nlp]`).

## Usage

The protocol runs on **precomputed per-platform outputs** (axes 1–4 need no deep-learning
dependencies):

```python
from fairscope.nlp import CPFEProtocol

platform_data = {
    "reference": {"y_true": y_ref, "probs": probs_ref},   # (n, n_classes)
    "deployment": {"y_true": y_dep, "probs": probs_dep},
}
report = CPFEProtocol(platform_data, reference="reference", n_classes=4).run()

report.to_dataframe()          # macro AUC/F1, ECE, ΔAUC% per platform
report.significance            # bootstrap macro-AUC test + Bonferroni-adjusted p
report.equity                  # per-class disparate impact and equalized odds
report.deployment_readiness()  # structured per-axis diagnostic
```

## `deployment_readiness()` — a diagnostic, not a decision

`deployment_readiness()` returns a structured per-axis verdict
(`{platform: {ready, axes: {axis: {pass, value, threshold, source, reason}}}}`). It is
explicitly a **screening diagnostic, not a deployment decision or compliance verdict**.

Each axis reports the threshold it used and that threshold's **source**:

- **Equity** uses the four-fifths rule (disparate impact < 0.80 is a violation, < 0.50 severe)
  — a stated reference value.
- **Calibration** uses stated reference bands (ECE < 0.10 well-calibrated, > 0.20 moderate
  miscalibration).
- **Discrimination** uses an **illustrative, user-configurable** ΔAUC limit
  (`delta_auc_pct_max`) — *not* a published decision threshold; the underlying study
  explicitly declines to set one.

Cross-platform degradation is an informative signal that warrants platform-specific
validation; it is not, by itself, evidence of algorithmic bias.
