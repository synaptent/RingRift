"""Track peer state transitions with reasons and flap detection."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger("p2p.diagnostics.state")


class PeerState(Enum):
    """Possible states for a peer node."""

    ALIVE = "alive"
    DEAD = "dead"
    SUSPECTED = "suspected"
    PROBING = "probing"


class DeathReason(Enum):
    """Reasons why a peer was marked as dead."""

    PROBE_TIMEOUT = "probe_timeout"
    GOSSIP_SUSPECT = "gossip_suspect"
    CONNECTION_REFUSED = "connection_refused"
    CIRCUIT_OPEN = "circuit_open"
    HEARTBEAT_TIMEOUT = "heartbeat_timeout"
    MANUAL = "manual"
    UNKNOWN = "unknown"


@dataclass
class StateTransition:
    """Record of a peer state transition."""

    node_id: str
    from_state: PeerState
    to_state: PeerState
    reason: DeathReason | None
    timestamp: float
    details: dict[str, Any] = field(default_factory=dict)


class PeerStateTracker:
    """Tracks all peer state transitions for diagnostics.

    Features:
    - Records every ALIVE<->DEAD transition with reason
    - Detects flapping peers (too many transitions in short window)
    - Provides diagnostic summary for debugging
    """

    def __init__(
        self,
        flap_window: float = 300.0,
        flap_threshold: int = 4,
        max_history: int = 10000,
    ) -> None:
        """Initialize the tracker.

        Args:
            flap_window: Time window in seconds to detect flapping (default 5 min)
            flap_threshold: Number of transitions to consider flapping (default 4)
            max_history: Maximum transitions to keep in memory
        """
        self._transitions: deque[StateTransition] = deque(maxlen=max_history)
        self._current_state: dict[str, PeerState] = {}
        self._flap_window = flap_window
        self._flap_threshold = flap_threshold

    def record_transition(
        self,
        node_id: str,
        to_state: PeerState,
        reason: DeathReason | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record a state transition for a peer.

        Args:
            node_id: The peer's node ID
            to_state: The new state
            reason: Reason for death (if transitioning to DEAD)
            details: Additional context (latency, error message, etc.)
        """
        from_state = self._current_state.get(node_id, PeerState.DEAD)

        # Skip if no actual transition
        if from_state == to_state:
            return

        transition = StateTransition(
            node_id=node_id,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            timestamp=time.time(),
            details=details or {},
        )
        self._transitions.append(transition)
        self._current_state[node_id] = to_state

        # Log every transition for visibility
        reason_str = f" reason={reason.value}" if reason else ""
        details_str = ""
        if details:
            # Include key details in log
            if "latency_ms" in details:
                details_str += f" latency={details['latency_ms']:.0f}ms"
            if "error" in details:
                details_str += f" error={str(details['error'])[:50]}"

        logger.info(
            f"PEER_STATE: {node_id[:20]} {from_state.value}->{to_state.value}"
            f"{reason_str}{details_str}"
        )

        # Warn if peer is flapping
        if self.is_flapping(node_id):
            logger.warning(f"PEER_FLAPPING: {node_id} has excessive state transitions")

    def get_state(self, node_id: str) -> PeerState:
        """Get current state of a peer."""
        return self._current_state.get(node_id, PeerState.DEAD)

    def is_flapping(self, node_id: str) -> bool:
        """Check if peer is flapping (too many transitions recently).

        A peer is considered flapping if it has >= flap_threshold transitions
        within the flap_window period.
        """
        now = time.time()
        recent = [
            t
            for t in self._transitions
            if t.node_id == node_id and now - t.timestamp < self._flap_window
        ]
        return len(recent) >= self._flap_threshold

    def get_flapping_peers(self) -> list[str]:
        """Return list of currently flapping peers."""
        # Get unique node IDs from recent transitions
        now = time.time()
        recent_nodes = set(
            t.node_id
            for t in self._transitions
            if now - t.timestamp < self._flap_window
        )
        return [nid for nid in recent_nodes if self.is_flapping(nid)]

    def get_recent_transitions(
        self, node_id: str | None = None, window: float = 300.0
    ) -> list[StateTransition]:
        """Get recent transitions, optionally filtered by node.

        Args:
            node_id: Filter to specific node (None for all)
            window: Time window in seconds (default 5 min)
        """
        now = time.time()
        transitions = [t for t in self._transitions if now - t.timestamp < window]
        if node_id:
            transitions = [t for t in transitions if t.node_id == node_id]
        return transitions

    def get_diagnostics(self) -> dict[str, Any]:
        """Return diagnostic summary.

        Returns dict with:
        - total_transitions_5min: Count of transitions in last 5 min
        - flapping_peers: List of currently flapping peer IDs
        - death_reasons: Count by reason in last 5 min
        - alive_count: Number of peers currently alive
        - dead_count: Number of peers currently dead
        - recent_deaths: Last 10 deaths with details
        """
        now = time.time()
        recent = [t for t in self._transitions if now - t.timestamp < 300]

        # Count by reason
        reason_counts: dict[str, int] = {}
        for t in recent:
            if t.reason:
                reason_counts[t.reason.value] = reason_counts.get(t.reason.value, 0) + 1

        # Get recent deaths for detailed view
        recent_deaths = [
            {
                "node_id": t.node_id[:20],
                "reason": t.reason.value if t.reason else "unknown",
                "ago_sec": int(now - t.timestamp),
                "details": t.details,
            }
            for t in reversed(list(recent))
            if t.to_state == PeerState.DEAD
        ][:10]

        return {
            "total_transitions_5min": len(recent),
            "flapping_peers": self.get_flapping_peers(),
            "death_reasons": reason_counts,
            "alive_count": sum(
                1 for s in self._current_state.values() if s == PeerState.ALIVE
            ),
            "dead_count": sum(
                1 for s in self._current_state.values() if s == PeerState.DEAD
            ),
            "recent_deaths": recent_deaths,
        }

    def clear(self) -> None:
        """Clear all tracking data."""
        self._transitions.clear()
        self._current_state.clear()
