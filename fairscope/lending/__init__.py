"""Mortgage-lending fairness auditing built on fairscope.core (+ optional econml)."""

from .audit import LendingFairnessAudit, LendingReport

__all__ = ["LendingFairnessAudit", "LendingReport"]
