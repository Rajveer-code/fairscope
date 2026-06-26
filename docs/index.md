# fairscope

**Subgroup-stratified, calibration-aware fairness auditing for machine-learning models — grounded in peer-reviewed methods.**

`fairscope` packages statistical machinery that mainstream fairness toolkits do not provide
as first-class, subgroup-stratified functions:

- **DeLong confidence intervals** for per-subgroup AUC
- **Expected Calibration Error** per subgroup, with reliability diagrams
- **Bootstrap significance testing** of subgroup performance differences, with
  multiple-comparison correction
- a subgroup-stratified **interface to standard recalibration** (temperature scaling,
  isotonic regression)
- a novel five-axis **Cross-Platform Fairness Evaluation (CPFE)** protocol
- per-class fairness metrics (symmetric disparate impact, equalized odds difference)

It ports established methods rather than inventing new ones; only the CPFE protocol is
presented as novel, and every function cites its source. `fairscope` is for rigorous,
uncertainty-aware *measurement* — it is complementary to mitigation-focused toolkits such as
AIF360 and Fairlearn, not a replacement.

## Install

```bash
pip install fairscope
```

Heavy NLP dependencies (torch, transformers, captum) install via `pip install fairscope[nlp]`;
SHAP via `pip install fairscope[shap]`. The base install stays light.

## Where to go next

- [Getting started](getting-started.md) — a real, runnable audit in a few lines.
- [Healthcare audit](healthcare.md) — the one-call clinical fairness report.
- [CPFE protocol](cpfe.md) — the five-axis cross-platform evaluation.
- [API reference](api.md) — every public function.

Source and releases: [github.com/Rajveer-code/fairscope](https://github.com/Rajveer-code/fairscope) · MIT license.
