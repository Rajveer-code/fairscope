# fairscope

[![CI](https://github.com/Rajveer-code/fairscope/actions/workflows/ci.yml/badge.svg)](https://github.com/Rajveer-code/fairscope/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-live-brightgreen.svg)](https://rajveer-code.github.io/fairscope/)

**Subgroup-stratified, calibration-aware fairness auditing for machine-learning models — grounded in peer-reviewed methods.**

📖 **Documentation:** <https://rajveer-code.github.io/fairscope/>

`fairscope` packages statistical machinery that mainstream fairness toolkits do not expose as
first-class, subgroup-stratified functions, and adds one novel protocol on top:

- **DeLong confidence intervals** for per-subgroup AUC (fast midrank algorithm).
- **Per-subgroup Expected/Maximum Calibration Error** with reliability diagrams.
- **Significance testing** of subgroup performance gaps (paired/unpaired DeLong, stratified
  bootstrap) with **Bonferroni / Benjamini–Hochberg** correction.
- A subgroup-stratified **interface to standard recalibration** — temperature scaling
  (Guo et al. 2017) and isotonic regression (Zadrozny & Elkan 2002), with pre/post-ECE.
- A novel five-axis **Cross-Platform Fairness Evaluation (CPFE)** protocol.
- One-call **domain audits**: `healthcare`, `lending`, `federated`.

Only the CPFE protocol is presented as novel. Every other function ports a documented method
and cites its source; the recalibration methods are standard, and the contribution there is the
per-subgroup interface and pre/post-ECE reporting.

> **Status — v0.3.0.** All five modules (`core`, `healthcare`, `nlp`/CPFE, `federated`,
> `lending`) are implemented, tested, and released. 100% line coverage on the statistical core;
> CI green across Python 3.9–3.12. See [`docs/DESIGN.md`](docs/DESIGN.md) for methods and design.

## Install

```bash
pip install fairscope
```

Releases are uploaded to PyPI by the maintainer; if a version isn't available there yet,
install from source or from the [release assets](https://github.com/Rajveer-code/fairscope/releases):

```bash
git clone https://github.com/Rajveer-code/fairscope
cd fairscope
pip install -e ".[dev]"
pytest
```

The base install is light (NumPy, SciPy, scikit-learn, pandas, matplotlib). Optional extras:
`fairscope[nlp]` (torch, transformers, captum), `fairscope[lending]` (econml),
`fairscope[shap]`, `fairscope[docs]`.

## Quickstart

```python
from fairscope.healthcare import HealthcareFairnessAudit

# y_true : binary outcomes
# y_score: the model's positive-class probabilities
# age_group: a protected attribute, aligned row-for-row
report = HealthcareFairnessAudit.from_scores(
    y_true, y_score, {"age_group": age_group}
).run()

print(report.summary())     # per-subgroup AUC (DeLong CI), ECE, Brier, F1; flags the largest gap
report.to_dataframe()       # tidy per-subgroup table
report.plot_auc_forest()    # forest plot of per-subgroup AUC with DeLong intervals
```

Every domain is also reachable through one dispatcher,
`FairnessAudit(model, domain=...)`, with `domain` in `{"healthcare", "nlp", "federated",
"lending"}`. A runnable end-to-end example on a committed synthetic fixture is in the
[getting-started guide](https://rajveer-code.github.io/fairscope/getting-started/) and in
[`notebooks/`](notebooks/).

## Modules

| Module | Purpose | Status |
|---|---|---|
| `core/` | DeLong CI, bootstrap-AUC test, ECE/MCE + reliability, multiple-testing correction, subgroup metrics | ✅ shipped |
| `healthcare/` | one-call clinical fairness audit + report (tables, forest & reliability plots, PDF, optional SHAP) | ✅ shipped |
| `nlp/` | CPFE five-axis cross-platform protocol (centerpiece) + Captum attribution stability | ✅ shipped |
| `federated/` | per-node DeLong + cross-node disparity + per-node recalibration | ✅ shipped |
| `lending/` | annual approval-gap + subgroup CATE (Causal Forest DML) | ✅ shipped |

Plotting (forest plots, reliability diagrams) currently lives in the domain reports.
`lending`'s CATE estimation needs the optional `fairscope[lending]` extra (`econml`). The
`federated` module audits per-node predictions only — it performs no training and provides no
privacy guarantee.

## How it differs from AIF360 / Fairlearn

`fairscope` is complementary to AIF360 and Fairlearn, not a replacement: those toolkits do bias
*mitigation*; `fairscope` does uncertainty-aware *measurement*. The table below was verified by
inspecting the installed public APIs of **AIF360 0.6.1** and **Fairlearn 0.14.0** (checked
2026-06; re-confirm if versions change).

| Capability | AIF360 | Fairlearn | fairscope |
|---|:---:|:---:|:---:|
| Per-subgroup AUC confidence interval (DeLong) | no | no\* | yes |
| Per-subgroup Expected Calibration Error | no | no | yes |
| Subgroup significance test + multiple-comparison correction | no | no | yes |
| Subgroup-stratified recalibration (temperature / isotonic) | partial† | no | yes |
| Cross-platform five-axis protocol (CPFE) | no | no | yes (novel) |
| Per-node / federated audit | no | no | yes |
| Bias-mitigation algorithms | yes | yes | out of scope |

\* Fairlearn's `MetricFrame` computes per-subgroup AUC *point estimates* (e.g.
`roc_auc_score_group_min`), but provides no analytic (DeLong) confidence interval.
† AIF360 ships `CalibratedEqOddsPostprocessing` (calibration-aware equalized-odds
postprocessing), not a general per-subgroup temperature/isotonic recalibration interface.

**Closest related work — `meval`** (Sutariya & Petersen, 2025,
[arXiv:2512.17409](https://arxiv.org/abs/2512.17409)): a statistical toolbox for stratified,
fine-grained model-performance analysis that *also* provides subgroup metric uncertainty and
multiple-comparison corrections (with a medical-imaging focus). `fairscope` overlaps with it on
uncertainty + significance; what `fairscope` adds is the specific DeLong AUC intervals, the
per-subgroup calibration **and recalibration** interface, the five-axis cross-platform CPFE
protocol, and one-call domain audits (healthcare / lending / federated).

## Engineering

- **Test-driven**, with regression tests anchored to authoritative reference values where they
  exist (DeLong's worked example; `statsmodels` multiple-testing routines).
- **100% line coverage** on the statistical core; CI runs pytest + coverage, ruff, and black
  across Python 3.9–3.12, and executes the replication notebooks via `nbmake`.
- Full type hints, NumPy-style docstrings with runnable examples, and explicit input validation
  (an AUC on a single-class subgroup raises rather than returning a meaningless value).
- Committed fixtures are **small, synthetic, and labelled as such**; no datasets or trained
  models are bundled.

## Grounded in published research

`fairscope` ports methods from the author's peer-reviewed and under-review papers; it invents no
new mathematics. Each function cites its source. Full venues and identifiers are in
[`CITATION.cff`](CITATION.cff).

- Diabetes risk prediction with external validation + fairness analysis (XGBoost, NHANES→BRFSS) — IEEE CIPHER, 2026.
- A five-axis Cross-Platform Fairness Evaluation for mental-health NLP — under review.
- Privacy-preserving federated learning for diabetes risk across heterogeneous nodes — under review.
- Heterogeneous racial effects in mortgage approval (Causal Forest Double Machine Learning, HMDA) — under review.
- Racial disparities in mortgage lending (RDD / DiD / decomposition, HMDA) — under review.

## Citation

If you use `fairscope`, please cite it via [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE) © 2026 Rajveer Singh Pall · ORCID [0009-0001-6762-6134](https://orcid.org/0009-0001-6762-6134)
