"""P2P Sub-Orchestrators.

January 2026: Created as part of P2POrchestrator decomposition.

This package contains sub-orchestrators that handle specific domains:
- LeadershipOrchestrator: Leader election, fence tokens, consensus
- PeerNetworkOrchestrator: Peer discovery, SWIM integration, voters
- SyncOrchestrator: Data sync, manifests, multi-transport
- JobOrchestrator: Job spawning, scheduling, rate limiting

Each orchestrator is composed into the main P2POrchestrator using
the composition pattern for clear separation of concerns.
"""

from scripts.p2p.orchestrators.base_orchestrator import (
    BaseOrchestrator,
    HealthCheckResult,
)
from scripts.p2p.orchestrators.leadership_orchestrator import LeadershipOrchestrator
from scripts.p2p.orchestrators.peer_network_orchestrator import PeerNetworkOrchestrator

__all__ = [
    "BaseOrchestrator",
    "HealthCheckResult",
    "LeadershipOrchestrator",
    "PeerNetworkOrchestrator",
    # Future exports:
    # "SyncOrchestrator",
    # "JobOrchestrator",
]
