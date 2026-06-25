# fairscope — Design Overview

`fairscope` is a Python library for subgroup-stratified, calibration-aware fairness
auditing of machine-learning models. It packages statistical methods from peer-reviewed
work into a tested, documented API. It ports established methods rather than inventing new
ones; only the Cross-Platform Fairness Evaluation protocol is presented as novel.

## The gap it fills

Mainstream fairness toolkits (AIF360, Fairlearn) provide group fairness metrics, but not —
as first-class, subgroup-stratified functions — DeLong confidence intervals for
per-subgroup AUC, per-subgroup Expected Calibration Error, or significance testing of
subgroup performance differences. `fairscope` provides these, plus a calibration-aware,
cross-platform, and federated audit surface.

Any capability comparison shipped in the README is verified directly against the installed
versions of the other toolkits before it is published; claims that no longer hold are
softened or dropped.

## Methods and sources

Each function cites its source in its docstring.

- **DeLong confidence intervals for AUC** — DeLong et al. (1988); fast midrank algorithm of
  Sun & Xu (2014).
- **Bootstrap AUC significance test** — stratified bootstrap standard errors, with
  Bonferroni and Benjamini–Hochberg multiple-comparison correction.
- **Expected Calibration Error, Maximum Calibration Error, reliability diagrams** — standard
  binned calibration analysis, computed per subgroup.
- **Subgroup-stratified interface to standard recalibration** — temperature scaling
  (Guo et al., 2017) and isotonic regression (Zadrozny & Elkan, 2002). These are standard
  methods; the contribution is the per-subgroup interface and pre/post-ECE reporting.
- **Subgroup metrics** — AUC, Brier score, F1, symmetric disparate impact, and equalized
  odds difference.
- **Cross-Platform Fairness Evaluation (CPFE)** — a five-axis protocol covering
  discriminative performance, calibration, statistical significance, prediction equity, and
  attribution stability (gradient-saliency token-set Jaccard overlap).
- **Subgroup conditional effects** — Causal Forest / Double Machine Learning estimation of
  heterogeneous effects, with an across-time gap summary.
- **Federated / per-node audit** — per-node DeLong intervals and cross-node disparity.

## Module / API design

```
fairscope/
├── core/        # delong, bootstrap, calibration (+ recalibration), correction, metrics
├── healthcare/  # one-call clinical fairness audit + report
├── nlp/         # CPFE five-axis protocol + gradient-saliency attribution stability
├── lending/     # subgroup CATE/DML effects + annual gap analysis
├── federated/   # per-node DeLong + cross-node disparity + recalibration
└── visualize/   # forest plots, reliability diagrams, heatmaps
```

A top-level `FairnessAudit(model, domain=...)` dispatches to the relevant domain class.
Heavy NLP dependencies (torch, transformers, captum) are an optional extra
(`pip install fairscope[nlp]`); SHAP is `fairscope[shap]`. The base install stays light.

Every public function uses full type hints, NumPy-style docstrings with a runnable example,
explicit input validation, and a fixed `random_state` wherever results are stochastic.

## Test strategy

Regression tests are the credibility anchor of the project.

- Where an authoritative reference value exists (DeLong 1988; `statsmodels` multiple-testing
  routines), tests assert against it.
- Synthetic fixtures with known closed-form answers cover edge cases (perfect calibration,
  one-class subgroups, planted effects).
- Small committed data fixtures are labeled subsamples that reproduce the *direction and
  approximate magnitude* of published results within a stated tolerance — they do not
  reproduce full datasets, and their docstrings say so.
- CI runs `pytest` with coverage (enforced overall floor ≥70%, target ≥85% on `core/`),
  plus `ruff` and `black`, across Python 3.9–3.12.

## Roadmap

1. `core/` statistical primitives and their tests.
2. `healthcare/` audit and report; first PyPI release.
3. `nlp/` CPFE protocol.
4. `lending/` and `federated/` audits.
5. Documentation site, an interactive demo, and a software paper.

Datasets are not bundled. `scripts/fetch_data.py` downloads public datasets on demand for
the full replication notebooks.

## License & citation

MIT. If you use `fairscope`, please cite it via [`CITATION.cff`](../CITATION.cff).
