---
title: "fairscope: subgroup-stratified, calibration-aware fairness auditing for machine-learning models"
author: "Rajveer Singh Pall (ORCID 0009-0001-6762-6134)"
keywords: [algorithmic fairness, model auditing, calibration, AUC confidence intervals, cross-platform generalisation, open-source software]
arxiv-categories: [cs.LG, cs.CY]
date: 2026-06-27
note: "DRAFT for review — not submitted. Markdown to be converted to arXiv LaTeX before submission."
---

# Abstract

Fairness audits in applied machine learning repeatedly require statistical machinery that
mainstream toolkits do not expose as first-class, subgroup-stratified functions:
confidence intervals for per-subgroup AUC, per-subgroup calibration error, and significance
tests for differences in subgroup performance. We present **fairscope**, an open-source,
test-driven Python library that packages these peer-reviewed methods behind a small, typed
API, and introduces one novel module — a five-axis **Cross-Platform Fairness Evaluation
(CPFE)** protocol that jointly characterises discriminative, calibration, statistical-significance,
prediction-equity, and attribution-stability failures when a model is deployed on a
distribution different from the one it was trained on. fairscope is an *engineering* contribution
grounded in prior published work: every statistical primitive ports a documented method and
cites its source; only the CPFE protocol is presented as novel. The library ships with 100%
line coverage on its statistical core, continuous integration across Python 3.9–3.12, and
small committed fixtures that regression-test the *direction and approximate magnitude* of
published findings. It is available on PyPI (`pip install fairscope`) under the MIT license.

# 1. The gap

Two widely used fairness toolkits anchor current practice. AIF360 [Bellamy et al. 2019]
provides a broad set of group-fairness metrics and bias-mitigation algorithms; Fairlearn
[Bird et al. 2020] focuses on constraint-based mitigation and disparity metrics. Both are
valuable, and fairscope is complementary rather than competitive. However, recurring needs
in applied audits are not available in either as first-class, subgroup-stratified functions:

- **Uncertainty on subgroup discrimination.** Reporting a per-subgroup AUC without a
  confidence interval invites over-interpretation of noise. The DeLong estimator
  [DeLong et al. 1988] gives an analytic AUC variance, but neither toolkit exposes a
  per-subgroup DeLong interval as a primary function.
- **Calibration parity.** Two subgroups can have equal accuracy yet unequal calibration —
  predicted probabilities that are systematically more reliable for one group. Per-subgroup
  Expected Calibration Error (ECE) is not a first-class output of either toolkit.
- **Significance of subgroup gaps.** Deciding whether an observed subgroup performance gap
  is distinguishable from sampling noise requires a paired or unpaired test with
  multiple-comparison control; this is left to the user.

A 2025 toolbox, *meval* [Sutariya & Petersen 2025], moves in a similar direction by providing fine-grained,
stratified performance analysis, and we cite it as the most closely related recent work; the
precise overlap should be assessed against its current release. fairscope's distinguishing
surface is the combination of subgroup DeLong intervals, per-subgroup calibration and
*recalibration*, multiple-comparison-corrected significance, the five-axis cross-platform
protocol, and a per-node federated audit, packaged as a single tested library.

*(Verification note: the capability comparison in Section 6 is to be re-confirmed against the
installed versions of AIF360 and Fairlearn, and meval's documented coverage, before
submission. Claims that no longer hold will be softened or removed.)*

# 2. Design

fairscope is organised as a small set of focused modules built on numpy, scipy, and
scikit-learn, with optional heavy dependencies isolated behind extras.

```
fairscope/
  core/        delong, bootstrap, calibration (+ recalibration), correction, metrics
  healthcare/  one-call clinical fairness audit + report
  nlp/         CPFE five-axis protocol + gradient-saliency attribution stability
  lending/     subgroup CATE/DML effects + annual gap analysis   (planned)
  federated/   per-node DeLong + cross-node disparity            (planned)
  visualize/   forest plots, reliability diagrams, heatmaps
```

**Statistical primitives (`core`), each citing its source.** `delong` implements AUC
confidence intervals and paired/unpaired tests using the O(n log n) midrank algorithm of
Sun & Xu [2014] for the DeLong [1988] variance; `bootstrap` provides a stratified bootstrap
AUC-difference test; `calibration` computes ECE, maximum calibration error, and reliability
diagrams, and exposes a subgroup-stratified *interface* to two standard recalibration
methods — temperature scaling [Guo et al. 2017] and isotonic regression
[Zadrozny & Elkan 2002] (the methods are standard; the contribution is the per-subgroup
interface and pre/post-ECE reporting); `correction` implements Bonferroni and
Benjamini–Hochberg [1995] corrections, validated to match `statsmodels`; `metrics` provides
subgroup AUC/Brier/F1, symmetric disparate impact, and equalized-odds difference.

**Engineering discipline.** Every public function has full type hints, a NumPy-style
docstring with a runnable example, explicit input validation (no silent failures: an AUC on
a single-class subgroup raises rather than returning a meaningless value), and a fixed
`random_state` wherever results are stochastic. Regression tests are the project's
credibility anchor: where an authoritative reference value exists (DeLong's worked example;
`statsmodels` multiple-testing routines), tests assert against it. The statistical core has
100% line coverage; CI runs pytest, coverage, ruff, and black across Python 3.9–3.12.
Heavy NLP dependencies (torch, transformers, captum) install only via `pip install
fairscope[nlp]`; SHAP via `fairscope[shap]`. Datasets are not bundled.

# 3. The CPFE protocol

The one novel module is the **Cross-Platform Fairness Evaluation (CPFE)** protocol. Models
trained on one data source (e.g. a curated corpus) and deployed on another (e.g. a different
social-media platform) can degrade in ways a single AUC number cannot distinguish. CPFE
evaluates five orthogonal axes, treating each non-training platform against a reference:

1. **Discriminative performance** — macro one-vs-rest AUC and macro F1, within- and
   cross-platform, with the relative change ΔAUC%.
2. **Calibration** — confidence–accuracy ECE per platform (the standard multiclass ECE of
   Guo et al. [2017]).
3. **Statistical significance** — a stratified **bootstrap** standard error on the macro-AUC
   difference between platforms (independent test sets), with Bonferroni correction across
   platform pairs.
4. **Prediction equity** — symmetric disparate impact and equalized-odds difference per
   class, treating platform membership as the group.
5. **Attribution stability** — Jaccard overlap of the top-K gradient-saliency token sets
   across platforms, K ∈ {5, 10, 15, 20}.

A `CPFEReport.deployment_readiness()` method returns a structured per-axis screening
diagnostic. It is explicitly a diagnostic, not a deployment decision or compliance verdict.
Each axis reports its value, the threshold applied, and the threshold's *source*: the
disparate-impact four-fifths rule and the calibration bands are reference values stated in
the underlying study; the discrimination ΔAUC limit is an *illustrative, user-configurable*
parameter, because that study explicitly declines to set a validated ΔAUC decision threshold.

# 4. Worked replications

We report two demonstrations. The first reproduces a *published* finding's direction and
magnitude; the second exercises the CPFE protocol on a synthetic shift. **No real
restricted datasets or trained models are distributed; committed fixtures are small and
explicitly labelled.**

**4.1 Healthcare audit (direction + magnitude of a published result).**
The author's published diabetes study [Pall et al. 2026, IEEE CIPHER] reports, on an
external validation cohort (BRFSS, n = 1,285,783), a substantially lower AUC for elderly
adults (≥60) than for younger adults — 0.607 versus 0.742, a gap of 0.135 with p < 0.001.
fairscope's `HealthcareFairnessAudit` is regression-tested on a **synthetic fixture
calibrated to reproduce that direction and approximate magnitude** (it is not real BRFSS
data and does not reproduce the full run): the audit recovers a young-vs-elderly AUC gap of
0.133 (young 0.764, elderly 0.631) with a Bonferroni-adjusted p < 0.01. The test asserts the
gap within a tolerance band and a conservative significance threshold, and documents that
the published p < 0.001 was obtained on the full 1.28M-row cohort, not the fixture.

**4.2 CPFE protocol (synthetic cross-platform shift).**
On a synthetic two-platform fixture — a high-signal reference platform and a platform with
an injected over-confident-but-wrong shift — the protocol's five axes behave as designed
(these are the library's actual outputs on the fixture, not real-world measurements):
macro-AUC falls from 0.997 to 0.341 (ΔAUC −65.8%); macro-F1 falls from 0.958 to 0.019; ECE
rises from 0.172 to 0.776; the bootstrap macro-AUC difference is significant (Bonferroni-
adjusted p ≪ 0.001); and a synthetic attribution comparison with near-disjoint token
sets yields J@5 = 0.11. Notably, the prediction-equity axis shows *no* four-fifths violation
on this fixture (per-class disparate impact 0.97–0.99), because the degraded predictions
remain roughly proportional across classes — illustrating that the five axes capture
genuinely distinct failure modes, and that a single discrimination metric would have masked
which dimensions failed. These figures are reproducible via `paper/cpfe_demo.py` (seed
20260627).

# 5. Comparison with existing toolkits

*(Provisional — to be re-verified against installed package versions before submission, per
Section 1.)*

| Capability | AIF360 | Fairlearn | fairscope |
|---|---|---|---|
| Per-subgroup AUC with DeLong CI | not first-class | not first-class | yes |
| Per-subgroup Expected Calibration Error | not first-class | not first-class | yes |
| Subgroup significance test (+ multiple-comparison correction) | not first-class | not first-class | yes |
| Subgroup-stratified recalibration interface | partial | partial | yes |
| Cross-platform five-axis protocol (CPFE) | no | no | yes (novel) |
| Per-node / federated audit | no | no | planned |
| Bias-mitigation algorithms | yes | yes | out of scope |

fairscope does not implement bias-mitigation or constraint-based training; for those, AIF360
and Fairlearn remain the appropriate tools. fairscope's scope is rigorous, uncertainty-aware
*measurement*.

# 6. Limitations

- The committed healthcare fixture is **synthetic**, calibrated to a published direction and
  magnitude; it is a pipeline regression test, not an empirical result.
- CPFE's per-axis screening thresholds are descriptive: the disparate-impact and calibration
  bands are reference values from a single study, and the ΔAUC limit is illustrative and
  user-configurable, not a validated decision threshold. None should be treated as
  regulatory criteria.
- Gradient-saliency attribution is a local linear approximation and can be unstable across
  seeds; the Jaccard axis should be read as consistent-with-instability, not as definitive.
- The capability comparison (Section 5) requires confirmation against current toolkit
  releases before publication.

# 7. Availability

Source: `https://github.com/Rajveer-code/fairscope` (MIT license). Install: `pip install
fairscope`; NLP extras via `pip install fairscope[nlp]`. The release is archived and
citable via `CITATION.cff`. Continuous integration, 100% statistical-core coverage, and a
public, incremental commit history accompany the code.

# References

- Bellamy, R. K. E., et al. (2019). AI Fairness 360: An extensible toolkit for detecting and mitigating algorithmic bias. *IBM Journal of Research and Development*, 63(4/5). arXiv:1810.01943.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.
- Bird, S., et al. (2020). Fairlearn: A toolkit for assessing and improving fairness in AI. *Microsoft Technical Report* MSR-TR-2020-32.
- DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics*, 44(3), 837–845.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*, PMLR 70:1321–1330.
- Sutariya, D., & Petersen, E. (2025). meval: A Statistical Toolbox for Fine-Grained Model Performance Analysis. arXiv:2512.17409.
- Pall, R. S., Yadav, S., Bhalerao, S., Sahu, S., Ahluwalia, R., & Awadhiya, B. (2026). Comprehensive Evaluation of Machine Learning for Type 2 Diabetes Risk Prediction: Large-Scale External Validation and Fairness Analysis. *IEEE CIPHER 2026*. DOI: 10.1109/CIPHER70417.2026.11523789.
- Sun, X., & Xu, W. (2014). Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves. *IEEE Signal Processing Letters*, 21(11), 1389–1393.
- Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. *KDD*, 694–699.

<!-- CPFE protocol origin: the five-axis design is the author's; the cross-platform
mental-health NLP study that motivates it is under review and is intentionally NOT cited
with specific unpublished figures here. Add the citation once that paper is public. -->
