# Contributing to fairscope

Thanks for your interest. `fairscope` is built incrementally, one module per phase,
with regression tests as the credibility anchor.

## Development setup

```bash
git clone https://github.com/Rajveer-code/fairscope
cd fairscope
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
pytest
```

## Standards

- **Every function:** full type hints; NumPy-style docstring with a runnable `Examples`
  block; input validation with clear errors; no silent failures; deterministic given a
  seed (expose `random_state` on anything stochastic).
- **Tests are mandatory.** Where an authoritative reference value exists (e.g. DeLong
  1988, `statsmodels` multitest), test against it — not against a paper. Where testing
  against a paper, assert *direction + approximate magnitude* within a stated tolerance
  and document any discrepancy. Committed data fixtures are small subsamples, labeled as
  such in the docstring; they do not reproduce a full published run.
- **Formatting/linting:** `black` + `ruff` (run automatically by `pre-commit`).
- **Coverage:** enforced overall package floor ≥70% (`--cov-fail-under=70` in CI); target ≥85% on `core/`.

## Honesty rules (non-negotiable)

1. No invented mathematics — every method ports a published paper and cites it.
2. No fabricated user counts, stars, downloads, or adoption claims anywhere.
3. If a paper is ambiguous about a parameter, open an issue and ask — do not invent it.
4. If a method does not match what a paper actually contains, name the function for what
   it generically does and flag the mismatch.

## Commits

Small, logical, self-contained commits — never batch a whole phase into one commit. Each
commit message should describe a real design decision or fix.
