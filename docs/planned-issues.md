# First Issues to open (Phase 0 → Phase 1)

Open these as real GitHub Issues once the repo is public. They form the public,
reviewer-visible task trail. Suggested `gh` commands are at the bottom.

1. **Phase 1 — `core/delong.py`: DeLong AUC CI + per-group CIs**
   Implement `delong_auc_ci`, `delong_paired_test`, `delong_unpaired_test`,
   `delong_by_group`. Use the O(n log n) midrank algorithm (Sun & Xu 2014); cite
   DeLong 1988. Test against the literature value (8-point toy set, AUC≈0.875).

2. **Phase 1 — `core/bootstrap.py`: bootstrap-AUC significance test (CPFE Axis 3)**
   `bootstrap_auc_test(y, score_a, score_b, n_boot=2000, random_state=...)`. This is the
   method P4 actually used for pairwise macro-AUC significance (stratified bootstrap SE +
   Bonferroni), not paired DeLong. Seed-deterministic.

3. **Phase 1 — `core/calibration.py`: ECE/MCE, reliability diagrams, + remediation**
   `expected_calibration_error` (M bins, uniform|quantile), `maximum_calibration_error`,
   `ece_by_group`, `reliability_diagram`; plus `temperature_scale` (P4) and
   `isotonic_recalibrate` (P6). Multiclass = one-vs-rest ECE per class.

4. **Phase 1 — `core/correction.py`: Bonferroni + Benjamini-Hochberg**
   Match `statsmodels.stats.multitest` exactly in tests.

5. **Phase 1 — `core/metrics.py`: subgroup metrics**
   Subgroup `auc`, `brier`, `f1`, symmetric `disparate_impact`,
   `equalized_odds_difference` — defined exactly as in P4; cite it.

6. **Verification gate — confirm AIF360/Fairlearn/meval gap claims**
   Before writing the README comparison table, `pip install aif360 fairlearn`, inspect
   current public APIs, and confirm each gap claim. Soften/drop anything no longer true.
   Confirm what `meval` actually covers. Do not ship unverified comparison claims.

7. **Phase 2 prep — healthcare golden fixture (BRFSS subsample)**
   Build a small committed CSV fixture (a labeled subsample, NOT the full set) that
   reproduces the *direction + approximate magnitude* of the published result: elderly
   AUC < young AUC, gap ≈ 0.135 (tolerance ±0.02), Bonferroni p < 0.001.

---

## Suggested `gh` commands

```bash
gh issue create --title "Phase 1: core/delong.py — DeLong AUC CI + per-group" \
  --body "Implement delong_auc_ci/paired/unpaired/by_group via Sun & Xu 2014 midrank; cite DeLong 1988; test vs 8-point toy AUC=0.875."
gh issue create --title "Phase 1: core/bootstrap.py — bootstrap-AUC test (CPFE Axis 3)" \
  --body "bootstrap_auc_test(B=2000, stratified, Bonferroni). This is P4's actual Axis-3 method, not paired DeLong. Seed-deterministic."
gh issue create --title "Phase 1: core/calibration.py — ECE/MCE + reliability + temperature/isotonic" \
  --body "ECE (uniform|quantile), MCE, ece_by_group, reliability_diagram, temperature_scale (P4), isotonic_recalibrate (P6). OvR for multiclass."
gh issue create --title "Phase 1: core/correction.py — Bonferroni + BH" \
  --body "Match statsmodels.stats.multitest exactly in tests."
gh issue create --title "Phase 1: core/metrics.py — subgroup AUC/Brier/F1/DI/EOD" \
  --body "Symmetric disparate impact + equalized odds difference exactly as defined in P4; cite it."
gh issue create --title "Verify AIF360/Fairlearn/meval gap claims before README table" \
  --body "pip install aif360 fairlearn; inspect public APIs; confirm or soften each claim; confirm meval coverage. No unverified comparison claims."
gh issue create --title "Phase 2 prep: healthcare golden fixture (BRFSS subsample, labeled)" \
  --body "Small committed CSV subsample reproducing direction+magnitude: elderly<young AUC, gap~0.135 (+/-0.02), Bonferroni p<0.001. Docstring states it is a subsample."
```
