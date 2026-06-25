"""Phase 0 smoke test: the package imports and exposes a valid version string.

This is intentionally minimal. Real regression tests (the project's credibility
anchor) arrive with each module in Phases 1+.
"""

import fairscope


def test_version_is_a_nonempty_string():
    assert isinstance(fairscope.__version__, str)
    assert fairscope.__version__


def test_package_imports():
    assert hasattr(fairscope, "__version__")
