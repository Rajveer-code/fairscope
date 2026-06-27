# Federated audit

`FederatedFairnessAudit` is a one-call **cross-node** (federated / multi-site) fairness
audit. It composes the `core` primitives — it invents no statistics — into a per-node and
across-node report.

!!! warning "Audit layer only"
    This module **audits per-node predictions** of an already-trained model. It does **not**
    perform federated training, secure aggregation, or differential privacy, and it makes
    **no privacy guarantee**. The privacy-preserving training is the modelling paper's
    contribution; this is the fairness audit on top of its outputs.

## What it computes

For each node:

- per-node **AUC with a DeLong confidence interval**
- per-node **Expected Calibration Error**, **Brier score**, and **F1**

Across nodes:

- the **max−min AUC gap** and the worst pair (`disparity()`)
- **Bonferroni-corrected pairwise** (unpaired) DeLong tests (nodes are disjoint samples)

Optionally, per-node **recalibration** (temperature scaling or isotonic regression) with
pre/post ECE.

## Entry point

```python
import numpy as np
from fairscope.federated import FederatedFairnessAudit

node_data = {
    "site_a": (y_a, score_a),   # (y_true, positive-class probability) per node
    "site_b": (y_b, score_b),
    "site_c": (y_c, score_c),
}
report = FederatedFairnessAudit(node_data).run()
```

## The report

```python
report.to_dataframe()        # tidy per-node table
report.disparity()           # {max_auc_gap, best, worst, worst_pair}
report.summary()             # text summary; flags the largest cross-node gap + significant pairs
report.recalibrate("temperature")  # {node: {ece_pre, ece_post}}; or "isotonic"
report.plot_auc_forest()     # per-node AUC with DeLong CIs
report.plot_calibration()    # true per-node reliability curves (core.reliability_diagram)
report.to_pdf("federated_report.pdf")  # multi-page PDF (matplotlib only)
```

`recalibrate` fits on each node's own `(y, score)` and reports pre/post ECE — an in-sample
diagnostic. For a deployment estimate, recalibrate on a held-out split.

Route it through the top-level dispatcher with
`FairnessAudit(None, domain="federated", node_data=node_data)`.
