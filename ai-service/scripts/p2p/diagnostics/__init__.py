"""P2P Diagnostic Instrumentation Package.

Provides comprehensive tracking for:
- Peer state transitions (ALIVE/DEAD with reasons)
- Connection failure categorization
- Probe effectiveness and false positive detection
"""

from .peer_state_tracker import (
    PeerState,
    DeathReason,
    StateTransition,
    PeerStateTracker,
)
from .connection_failure_tracker import (
    FailureType,
    ConnectionFailure,
    ConnectionFailureTracker,
)
from .probe_tracker import (
    ProbeResult,
    ProbeEffectivenessTracker,
)

__all__ = [
    # Peer state
    "PeerState",
    "DeathReason",
    "StateTransition",
    "PeerStateTracker",
    # Connection failures
    "FailureType",
    "ConnectionFailure",
    "ConnectionFailureTracker",
    # Probe effectiveness
    "ProbeResult",
    "ProbeEffectivenessTracker",
]
