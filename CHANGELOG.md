# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[0.1.0]: https://github.com/Rajveer-code/fairscope/releases/tag/v0.1.0
