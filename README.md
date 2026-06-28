# fairscope

[![CI](https://github.com/Rajveer-code/fairscope/actions/workflows/ci.yml/badge.svg)](https://github.com/Rajveer-code/fairscope/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-live-brightgreen.svg)](https://rajveer-code.github.io/fairscope/)

📖 **Documentation:** <https://rajveer-code.github.io/fairscope/>

**Subgroup-stratified, calibration-aware fairness auditing for ML models — grounded in peer-reviewed methods.**

`fairscope` packages statistical machinery that mainstream fairness toolkits do not
provide as first-class, subgroup-stratified functions: DeLong confidence intervals for
per-subgroup AUC, Expected Calibration Error per group, bootstrap significance testing
of subgroup performance differences, a subgroup-stratified interface to *standard*
recalibration (temperature scaling — Guo et al. 2017; isotonic regression — Zadrozny &
Elkan 2002), and a novel five-axis **Cross-Platform Fairness Evaluation (CPFE)** protocol.

Only the CPFE protocol is presented as novel. The recalibration methods are standard; the
contribution there is the per-subgroup interface and pre/post-ECE reporting.

> **Status: v0.2.0.** `core`, `healthcare`, and the CPFE `nlp` protocol are implemented and
> tested (100% coverage, CI on Python 3.9–3.12). `lending/` and `federated/` are planned.
> See [`docs/DESIGN.md`](docs/DESIGN.md) for the methods, API design, and roadmap.

## Install

```bash
pip install fairscope
```

Releases are uploaded to PyPI by the maintainer; if a version isn't available there yet,
install from source or from the release assets:

```bash
git clone https://github.com/Rajveer-code/fairscope
cd fairscope
pip install -e ".[dev]"
pytest
```

Wheels and source distributions are attached to every
[GitHub release](https://github.com/Rajveer-code/fairscope/releases).

Heavy NLP dependencies (torch, transformers, captum) install via
`pip install fairscope[nlp]`; SHAP via `fairscope[shap]`; docs tooling via
`fairscope[docs]`. The base install stays light. See the
[documentation site](https://rajveer-code.github.io/fairscope/) for a runnable
getting-started example.

## Modules

| Module | Purpose | Status |
|---|---|---|
| `core/` | DeLong CI, bootstrap-AUC test, ECE/MCE + reliability, multiple-testing correction, subgroup metrics | ✅ shipped |
| `healthcare/` | one-call clinical fairness audit + report (tables, forest & reliability plots, PDF, optional SHAP) | ✅ shipped |
| `nlp/` | CPFE 5-axis cross-platform protocol (centerpiece) + Captum attribution stability | ✅ shipped |
| `federated/` | per-node DeLong + cross-node disparity + recalibration | ✅ shipped |
| `lending/` | annual approval-gap + subgroup CATE (Causal Forest DML) | ✅ shipped |

Plotting (forest plots, reliability diagrams) currently lives in the domain reports.
`lending`'s CATE estimation needs the optional `fairscope[lending]` extra (`econml`).

## How it differs from AIF360 / Fairlearn

`fairscope` is complementary to AIF360 and Fairlearn, not a replacement: those toolkits do
bias *mitigation*; `fairscope` does uncertainty-aware *measurement*. The table below was
verified by inspecting the installed public APIs of **AIF360 0.6.1** and **Fairlearn 0.14.0**
(checked 2026-06; re-confirm if versions change).

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

## Grounded in published research

`fairscope` ports methods from the author's peer-reviewed and under-review papers; it
invents no new mathematics. Each function cites its source. Full venues and identifiers
are in [`CITATION.cff`](CITATION.cff).

- Diabetes risk prediction with external validation + fairness analysis (XGBoost, NHANES→BRFSS) — IEEE CIPHER, 2026.
- A five-axis Cross-Platform Fairness Evaluation for mental-health NLP — under review.
- Privacy-preserving federated learning for diabetes risk across heterogeneous nodes — under review.
- Heterogeneous racial effects in mortgage approval (Causal Forest Double Machine Learning, HMDA) — under review.
- Racial disparities in mortgage lending (RDD / DiD / decomposition, HMDA) — under review.

## Citation

If you use `fairscope`, please cite it via [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE) © 2026 Rajveer Singh Pall · ORCID [0009-0001-6762-6134](https://orcid.org/0009-0001-6762-6134)
