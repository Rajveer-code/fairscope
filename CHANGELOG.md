# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-06-27

### Added
- `fairscope.federated`: `FederatedFairnessAudit` + `FederatedReport` — a cross-node
  (federated / multi-site) audit composing `core`: per-node DeLong AUC CIs, ECE, Brier and
  F1; cross-node disparity (max−min AUC gap and Bonferroni-corrected pairwise unpaired
  DeLong tests); optional per-node recalibration (temperature/isotonic) with pre/post ECE;
  per-node AUC forest, reliability curves, and PDF export. Audits per-node predictions only
  — no training and no privacy guarantee. Routed via `FairnessAudit(model, domain="federated", ...)`.
- `fairscope.lending`: `LendingFairnessAudit` + `LendingReport` — a descriptive annual
  approval-gap analysis (symmetric disparate impact per year, composing `core`) plus an
  optional subgroup CATE via Causal Forest DML (`estimate_cate`, `econml.dml.CausalForestDML`).
  Causal claims are conditional on the DML assumptions. `econml` is the optional
  `fairscope[lending]` extra. Routed via `FairnessAudit(model, domain="lending", ...)`.
- Documentation pages for the federated and lending modules, and an auto-generated API
  reference for both.
- Replication notebooks `notebooks/03_lending_replication.ipynb` and
  `notebooks/04_federated_replication.ipynb` (synthetic; executed in CI via `nbmake`).

## [0.2.0] - 2026-06-27

### Added
- `fairscope.nlp`: the five-axis Cross-Platform Fairness Evaluation (CPFE) protocol —
  `CPFEProtocol` + `CPFEReport` (macro AUC/F1 and ΔAUC%, multiclass ECE, bootstrap
  macro-AUC significance with Bonferroni correction, per-class disparate impact and
  equalized odds), a structured per-axis `deployment_readiness()` diagnostic using P4's
  stated reference bands (with an illustrative, configurable ΔAUC limit), and
  gradient-saliency Jaccard attribution stability (`token_saliency` behind
  `fairscope[nlp]`). Routed via `FairnessAudit(model, domain="nlp", ...)`.
- Documentation site (MkDocs Material + mkdocstrings) published to GitHub Pages:
  getting-started with a runnable example on the synthetic fixture, CPFE and healthcare
  guides, and an auto-generated API reference. <https://rajveer-code.github.io/fairscope/>
- Replication notebooks (`notebooks/01_healthcare_replication.ipynb`,
  `notebooks/02_nlp_cpfe_demo.ipynb`) executed in CI via `nbmake`.

## [0.1.0] - 2026-06-26

### Added
- Phase 0 public skeleton: package layout, MIT `LICENSE`, `pyproject.toml` (hatchling),
  GitHub Actions CI (matrix py3.9–3.12), `pre-commit` config, `CITATION.cff`, README
  skeleton, and the design overview in `docs/DESIGN.md`.
- `fairscope.core`: DeLong AUC confidence intervals and paired/unpaired tests
  (`delong.py`), a stratified bootstrap AUC test (`bootstrap.py`), Expected/Maximum
  Calibration Error with reliability diagrams plus temperature-scaling and isotonic
  recalibration (`calibration.py`), Bonferroni and Benjamini–Hochberg corrections
  (`correction.py`), and subgroup metrics with symmetric disparate impact and equalized
  odds difference (`metrics.py`). 100% test coverage on `core/`.
- `fairscope.healthcare`: `HealthcareFairnessAudit` + `HealthcareReport` — a one-call
  clinical fairness audit composing `core/` (per-subgroup DeLong CIs, ECE, Bonferroni-
  corrected pairwise tests, Brier/F1), with report tables, an AUC forest plot,
  reliability-curve plots, multi-page PDF export (matplotlib only), and an optional SHAP
  summary (`fairscope[shap]`). A synthetic, seed-generated golden fixture regression-tests
  the published direction and approximate magnitude (elderly < young AUC, gap ≈ 0.135).
- Top-level `FairnessAudit(model, domain=...)` dispatcher (healthcare implemented).

[0.3.0]: https://github.com/Rajveer-code/fairscope/releases/tag/v0.3.0
[0.2.0]: https://github.com/Rajveer-code/fairscope/releases/tag/v0.2.0
[0.1.0]: https://github.com/Rajveer-code/fairscope/releases/tag/v0.1.0
