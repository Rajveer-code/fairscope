"""fairscope: subgroup-stratified, calibration-aware fairness auditing for ML models.

A peer-reviewed-method-backed Python library that fills documented gaps in mainstream
fairness toolkits (AIF360, Fairlearn): per-subgroup DeLong confidence intervals,
per-subgroup Expected Calibration Error, calibration-aware fairness, a five-axis
Cross-Platform Fairness Evaluation (CPFE) protocol, and per-node federated audits.

See ``docs/DESIGN.md`` for the methods, API design, and roadmap.
"""

__version__ = "0.0.1.dev0"

__all__ = ["FairnessAudit", "__version__"]


def FairnessAudit(model, domain, **kwargs):
    """Route to a domain-specific fairness audit.

    Parameters
    ----------
    model : the fitted classifier (or ``None`` with precomputed scores via the domain API).
    domain : str
        The audit domain. Implemented: ``"healthcare"``.
    **kwargs : passed through to the domain audit class.

    Examples
    --------
    >>> import fairscope
    >>> callable(fairscope.FairnessAudit)
    True
    """
    if domain == "healthcare":
        from .healthcare import HealthcareFairnessAudit

        return HealthcareFairnessAudit(model, **kwargs)
    raise ValueError(
        f"unknown or unimplemented domain: {domain!r}; available domains: 'healthcare'"
    )
