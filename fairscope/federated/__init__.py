"""Cross-node (federated / multi-site) fairness auditing built on fairscope.core.

AUDITS per-node predictions of an already-trained model. It does NOT perform federated
training and provides NO privacy guarantees.
"""

from .audit import FederatedFairnessAudit, FederatedReport

__all__ = ["FederatedFairnessAudit", "FederatedReport"]
