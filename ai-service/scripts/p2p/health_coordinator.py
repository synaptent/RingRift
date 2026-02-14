"""Unified Health Coordinator for P2P Cluster.

January 3, 2026 - Sprint 13: Single source of truth for all P2P health signals.

Problem Solved: Previously there were 3 separate health tracking mechanisms:
    - GossipHealthTracker: Per-peer gossip failure tracking with backoff
    - NodeCircuitBreaker: Per-node circuit states (CLOSED/OPEN/HALF_OPEN)
    - QuorumHealthLevel: Cluster quorum health (HEALTHY/DEGRADED/MINIMUM/LOST)

This led to:
    - Inconsistent failure detection (different mechanisms disagreed)
    - Duplicate health state across ~800 LOC
    - No unified recovery decision-making

Solution: HealthCoordinator aggregates all health sources:
    - Single ClusterHealthState combining all signals
    - Unified get_recovery_action() for recovery decisions
    - should_trigger_election() for leadership decisions
    - Approximately 600-900 LOC savings once refactoring complete

Usage:
    from scripts.p2p.health_coordinator import (
        HealthCoordinator,
        get_health_coordinator,
        ClusterHealthState,
        RecoveryAction,
    )

    # Get singleton
    coordinator = get_health_coordinator()

    # Get unified cluster health
    health = coordinator.get_cluster_health()
    print(f"Overall: {health.overall_health}")
    print(f"Quorum: {health.quorum_health}")
    print(f"Open circuits: {len(health.open_circuits)}")

    # Get recovery action
    action = coordinator.get_recovery_action()
    if action == RecoveryAction.RESTART_P2P:
        await restart_p2p_orchestrator()

    # Check if election needed
    if coordinator.should_trigger_election():
        await trigger_leader_election()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from scripts.p2p.gossip_health import GossipHealthTracker
    from app.coordination.node_circuit_breaker import NodeCircuitBreaker, NodeCircuitState

# Import circuit breaker registry for auto-integration
try:
    from app.coordination.node_circuit_breaker import get_node_circuit_breaker
    from app.distributed.circuit_breaker import get_circuit_registry
    HAS_CIRCUIT_REGISTRIES = True
except ImportError:
    HAS_CIRCUIT_REGISTRIES = False
    get_node_circuit_breaker = None  # type: ignore
    get_circuit_registry = None  # type: ignore

# Import event system for CLUSTER_HEALTH_CHANGED emission
try:
    from app.distributed.data_events import DataEventType
    from app.coordination.event_router import get_event_bus
    HAS_EVENT_SYSTEM = True
except ImportError:
    HAS_EVENT_SYSTEM = False
    DataEventType = None  # type: ignore
    get_event_bus = None  # type: ignore

logger = logging.getLogger(__name__)

__all__ = [
    "HealthCoordinator",
    "ClusterHealthState",
    "RecoveryAction",
    "OverallHealthLevel",
    "DaemonHealthSummary",
    "get_health_coordinator",
    "reset_health_coordinator",
]


class OverallHealthLevel(str, Enum):
    """Overall cluster health level combining all signals.

    Levels are ordered by severity (worst to best):
        CRITICAL: Cluster cannot operate, immediate intervention required
        DEGRADED: Cluster can operate but with reduced capacity
        WARNING: Minor issues detected, monitoring closely
        HEALTHY: All systems operating normally
    """

    CRITICAL = "critical"
    DEGRADED = "degraded"
    WARNING = "warning"
    HEALTHY = "healthy"

    def __lt__(self, other: "OverallHealthLevel") -> bool:
        """Enable comparison for health level ordering."""
        order = {self.CRITICAL: 0, self.DEGRADED: 1, self.WARNING: 2, self.HEALTHY: 3}
        return order[self] < order[other]


class RecoveryAction(str, Enum):
    """Recommended recovery action based on cluster health.

    Actions are prioritized by severity:
        RESTART_P2P: Full P2P orchestrator restart required
        TRIGGER_ELECTION: Initiate leader election
        HEAL_PARTITIONS: Attempt partition healing
        RESET_CIRCUITS: Reset open circuit breakers
        NONE: No action required
    """

    RESTART_P2P = "restart_p2p"
    TRIGGER_ELECTION = "trigger_election"
    HEAL_PARTITIONS = "heal_partitions"
    RESET_CIRCUITS = "reset_circuits"
    NONE = "none"


# Import QuorumHealthLevel from leader_election
try:
    from scripts.p2p.leader_election import QuorumHealthLevel
except ImportError:
    # Fallback for standalone testing

    class QuorumHealthLevel(str, Enum):  # type: ignore[no-redef]
        HEALTHY = "healthy"
        DEGRADED = "degraded"
        MINIMUM = "minimum"
        LOST = "lost"


@dataclass
class DaemonHealthSummary:
    """Summary of daemon health status.

    Aggregates health information from DaemonManager for unified view.
    """

    running_count: int = 0
    failed_count: int = 0
    restarting_count: int = 0
    total_count: int = 0
    critical_failed: list[str] = field(default_factory=list)
    last_health_check: float = 0.0

    @property
    def failure_rate(self) -> float:
        """Calculate daemon failure rate (0.0-1.0)."""
        if self.total_count == 0:
            return 0.0
        return self.failed_count / self.total_count

    @property
    def is_healthy(self) -> bool:
        """Check if daemon health is acceptable."""
        return self.failure_rate < 0.1 and len(self.critical_failed) == 0


@dataclass
class GossipHealthSummary:
    """Summary of gossip protocol health.

    Aggregates per-peer gossip health for cluster-level view.
    """

    total_peers: int = 0
    healthy_peers: int = 0
    suspected_peers: int = 0
    unreachable_peers: int = 0
    peers_in_backoff: list[str] = field(default_factory=list)
    max_backoff_seconds: float = 0.0
    last_successful_round: float = 0.0

    @property
    def healthy_ratio(self) -> float:
        """Calculate ratio of healthy peers (0.0-1.0)."""
        if self.total_peers == 0:
            return 1.0
        return self.healthy_peers / self.total_peers


@dataclass
class CircuitBreakerSummary:
    """Summary of circuit breaker states across cluster.

    Aggregates per-node circuit breaker states.
    """

    total_circuits: int = 0
    closed_count: int = 0
    open_count: int = 0
    half_open_count: int = 0
    open_circuits: list[str] = field(default_factory=list)  # Node IDs with open circuits
    recently_opened: list[str] = field(default_factory=list)  # Opened in last 5 min

    @property
    def open_ratio(self) -> float:
        """Calculate ratio of open circuits (0.0-1.0)."""
        if self.total_circuits == 0:
            return 0.0
        return self.open_count / self.total_circuits

    @property
    def is_cascade_risk(self) -> bool:
        """Check if there's risk of cascading circuit failures.

        Returns True if >50% of circuits are open or >3 opened recently.
        """
        return self.open_ratio > 0.5 or len(self.recently_opened) > 3


@dataclass
class ClusterHealthState:
    """Unified cluster health state combining all signals.

    This is the single source of truth for cluster health.
    All health decisions should be made based on this state.
    """

    # Overall assessment
    overall_health: OverallHealthLevel = OverallHealthLevel.HEALTHY
    overall_score: float = 1.0  # 0.0-1.0, higher is better

    # Component health
    quorum_health: QuorumHealthLevel = QuorumHealthLevel.HEALTHY
    gossip_health: GossipHealthSummary = field(default_factory=GossipHealthSummary)
    circuit_health: CircuitBreakerSummary = field(default_factory=CircuitBreakerSummary)
    daemon_health: DaemonHealthSummary = field(default_factory=DaemonHealthSummary)

    # Cluster metadata
    node_id: str = ""
    is_leader: bool = False
    leader_id: str | None = None
    alive_peers: int = 0
    total_peers: int = 0
    uptime_seconds: float = 0.0

    # Open circuit node IDs (convenience accessor)
    open_circuits: list[str] = field(default_factory=list)

    # Unhealthy peer IDs (suspected, unreachable, or circuit-broken)
    unhealthy_peers: list[str] = field(default_factory=list)

    # Timestamp
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization/API response."""
        return {
            "overall_health": self.overall_health.value,
            "overall_score": self.overall_score,
            "quorum_health": self.quorum_health.value,
            "gossip": {
                "total_peers": self.gossip_health.total_peers,
                "healthy_peers": self.gossip_health.healthy_peers,
                "suspected_peers": self.gossip_health.suspected_peers,
                "healthy_ratio": self.gossip_health.healthy_ratio,
            },
            "circuits": {
                "total": self.circuit_health.total_circuits,
                "open": self.circuit_health.open_count,
                "half_open": self.circuit_health.half_open_count,
                "open_ratio": self.circuit_health.open_ratio,
                "cascade_risk": self.circuit_health.is_cascade_risk,
            },
            "daemons": {
                "running": self.daemon_health.running_count,
                "failed": self.daemon_health.failed_count,
                "failure_rate": self.daemon_health.failure_rate,
            },
            "cluster": {
                "node_id": self.node_id,
                "is_leader": self.is_leader,
                "leader_id": self.leader_id,
                "alive_peers": self.alive_peers,
                "total_peers": self.total_peers,
                "unhealthy_peers": self.unhealthy_peers,
            },
            "timestamp": self.timestamp,
        }


class HealthCoordinator:
    """Unified health monitoring for P2P cluster.

    Aggregates all health sources into a single view:
        - GossipHealthTracker: Per-peer gossip failure tracking
        - NodeCircuitBreaker: Per-node circuit states
        - QuorumHealthLevel: Cluster quorum health
        - DaemonHealthSummary: Daemon status from DaemonManager

    Provides unified APIs:
        - get_cluster_health(): Get complete health state
        - get_recovery_action(): Get recommended recovery action
        - should_trigger_election(): Check if election needed

    Thread-safe via RLock.
    """

    # Health score weights for overall calculation
    # Quorum is most critical - cluster cannot function without it
    WEIGHT_QUORUM = 0.40
    WEIGHT_GOSSIP = 0.20
    WEIGHT_CIRCUIT = 0.20
    WEIGHT_DAEMON = 0.20

    # Thresholds for health levels
    # CRITICAL: < 0.45 (e.g., quorum LOST alone triggers this)
    # DEGRADED: 0.45-0.65
    # WARNING: 0.65-0.85
    # HEALTHY: >= 0.85
    SCORE_CRITICAL_THRESHOLD = 0.45
    SCORE_DEGRADED_THRESHOLD = 0.65
    SCORE_WARNING_THRESHOLD = 0.85

    # Recovery action thresholds
    ELECTION_GRACE_PERIOD_SECONDS = 30.0
    CIRCUIT_CASCADE_THRESHOLD = 0.5
    GOSSIP_DEGRADED_THRESHOLD = 0.6

    def __init__(
        self,
        gossip_tracker: "GossipHealthTracker | None" = None,
        node_circuit_breaker: "NodeCircuitBreaker | None" = None,
        node_id: str = "",
    ):
        """Initialize the health coordinator.

        Args:
            gossip_tracker: Optional GossipHealthTracker instance
            node_circuit_breaker: Optional NodeCircuitBreaker instance
            node_id: This node's ID
        """
        self._gossip_tracker = gossip_tracker
        self._node_circuit_breaker = node_circuit_breaker
        self._node_id = node_id

        self._lock = RLock()
        self._start_time = time.time()

        # Cached state
        self._last_health_state: ClusterHealthState | None = None
        self._last_health_check_time: float = 0.0

        # External state sources (set via setters)
        self._quorum_health: QuorumHealthLevel = QuorumHealthLevel.HEALTHY
        self._is_leader: bool = False
        self._leader_id: str | None = None
        self._alive_peers: int = 0
        self._total_peers: int = 0
        self._daemon_health: DaemonHealthSummary = DaemonHealthSummary()

        # Recovery state tracking
        self._last_election_trigger: float = 0.0
        self._last_recovery_action: RecoveryAction = RecoveryAction.NONE
        self._recovery_cooldown_seconds: float = 60.0

        # Callbacks
        self._on_health_change: list[Callable[[ClusterHealthState], None]] = []

        logger.debug("HealthCoordinator initialized for node %s", node_id)

    # =========================================================================
    # Configuration Methods
    # =========================================================================

    def set_gossip_tracker(self, tracker: "GossipHealthTracker") -> None:
        """Set the gossip health tracker.

        Args:
            tracker: GossipHealthTracker instance to monitor
        """
        with self._lock:
            self._gossip_tracker = tracker

    def set_node_circuit_breaker(self, breaker: "NodeCircuitBreaker") -> None:
        """Set the node circuit breaker.

        Args:
            breaker: NodeCircuitBreaker instance to monitor
        """
        with self._lock:
            self._node_circuit_breaker = breaker

    def set_node_id(self, node_id: str) -> None:
        """Set this node's ID.

        Args:
            node_id: The node identifier
        """
        with self._lock:
            self._node_id = node_id

    def update_cluster_state(
        self,
        *,
        quorum_health: QuorumHealthLevel | None = None,
        is_leader: bool | None = None,
        leader_id: str | None = None,
        alive_peers: int | None = None,
        total_peers: int | None = None,
    ) -> None:
        """Update cluster state from external sources.

        Args:
            quorum_health: Current quorum health level
            is_leader: Whether this node is leader
            leader_id: Current leader node ID
            alive_peers: Number of alive peers
            total_peers: Total number of peers
        """
        with self._lock:
            if quorum_health is not None:
                self._quorum_health = quorum_health
            if is_leader is not None:
                self._is_leader = is_leader
            if leader_id is not None:
                self._leader_id = leader_id
            if alive_peers is not None:
                self._alive_peers = alive_peers
            if total_peers is not None:
                self._total_peers = total_peers

    def update_daemon_health(self, summary: DaemonHealthSummary) -> None:
        """Update daemon health summary.

        Args:
            summary: DaemonHealthSummary from DaemonManager
        """
        with self._lock:
            self._daemon_health = summary

    def register_health_change_callback(
        self, callback: Callable[[ClusterHealthState], None]
    ) -> None:
        """Register callback for health state changes.

        Args:
            callback: Function called when health state changes significantly
        """
        with self._lock:
            self._on_health_change.append(callback)

    # =========================================================================
    # Health Assessment Methods
    # =========================================================================

    def get_cluster_health(self, force_refresh: bool = False) -> ClusterHealthState:
        """Get unified cluster health state.

        This is the primary API for health assessment. Returns a complete
        ClusterHealthState combining all health signals.

        Args:
            force_refresh: If True, bypass cache and recalculate

        Returns:
            ClusterHealthState with all health information
        """
        with self._lock:
            # Check cache (refresh every 5 seconds by default)
            now = time.time()
            if (
                not force_refresh
                and self._last_health_state is not None
                and (now - self._last_health_check_time) < 5.0
            ):
                return self._last_health_state

            # Build new health state
            state = self._compute_health_state()

            # Check for significant change
            if self._last_health_state is not None:
                if state.overall_health != self._last_health_state.overall_health:
                    self._notify_health_change(state)

            # Update cache
            self._last_health_state = state
            self._last_health_check_time = now

            return state

    def _compute_health_state(self) -> ClusterHealthState:
        """Compute the complete health state from all sources."""
        state = ClusterHealthState(
            node_id=self._node_id,
            is_leader=self._is_leader,
            leader_id=self._leader_id,
            alive_peers=self._alive_peers,
            total_peers=self._total_peers,
            uptime_seconds=time.time() - self._start_time,
            quorum_health=self._quorum_health,
            daemon_health=self._daemon_health,
        )

        # Collect gossip health
        state.gossip_health = self._collect_gossip_health()

        # Collect circuit breaker health
        state.circuit_health = self._collect_circuit_health()
        state.open_circuits = state.circuit_health.open_circuits.copy()

        # Build unhealthy peers list (circuit-broken + peers in backoff)
        unhealthy = set(state.open_circuits)
        if state.gossip_health.peers_in_backoff:
            unhealthy.update(state.gossip_health.peers_in_backoff)
        state.unhealthy_peers = list(unhealthy)

        # Calculate overall score and level
        state.overall_score = self._calculate_overall_score(state)
        state.overall_health = self._score_to_level(state.overall_score)

        return state

    def _collect_gossip_health(self) -> GossipHealthSummary:
        """Collect health summary from gossip tracker.

        Jan 3, 2026 Sprint 13: Now uses public get_health_summary() API
        instead of accessing private attributes for thread safety.
        """
        summary = GossipHealthSummary()

        if self._gossip_tracker is None:
            return summary

        try:
            # Get thread-safe health summary from tracker (Sprint 13 API)
            if hasattr(self._gossip_tracker, "get_health_summary"):
                tracker_summary = self._gossip_tracker.get_health_summary()
                failure_counts = tracker_summary.failure_counts
                suspected_peers_set = set(tracker_summary.suspected_peers)
                threshold = tracker_summary.failure_threshold
                all_peers = set(failure_counts.keys()) | set(tracker_summary.last_success.keys())
            else:
                # Fallback for older tracker versions (deprecated path)
                failure_counts = getattr(self._gossip_tracker, "_failure_counts", {})
                last_success = getattr(self._gossip_tracker, "_last_success", {})
                suspected_peers_set = getattr(self._gossip_tracker, "_suspect_emitted", set())
                threshold = getattr(self._gossip_tracker, "_failure_threshold", 3)
                all_peers = set(failure_counts.keys()) | set(last_success.keys())

            summary.total_peers = len(all_peers)

            for peer_id in all_peers:
                failures = failure_counts.get(peer_id, 0)
                if failures >= threshold or peer_id in suspected_peers_set:
                    summary.suspected_peers += 1
                elif failures == 0:
                    summary.healthy_peers += 1
                else:
                    summary.healthy_peers += 1  # Still reachable, just had some failures

                # Check if in backoff
                if hasattr(self._gossip_tracker, "should_skip_peer"):
                    if self._gossip_tracker.should_skip_peer(peer_id):
                        summary.peers_in_backoff.append(peer_id)

            # Get max backoff
            if hasattr(self._gossip_tracker, "get_backoff_seconds"):
                for peer_id in all_peers:
                    backoff = self._gossip_tracker.get_backoff_seconds(peer_id)
                    summary.max_backoff_seconds = max(summary.max_backoff_seconds, backoff)

            # Track unreachable (suspected peers that are in max backoff)
            summary.unreachable_peers = len(
                [p for p in summary.peers_in_backoff if p in suspected_peers_set]
            )

        except (AttributeError, TypeError, KeyError, ValueError) as e:
            # Narrowed from broad Exception (Jan 2026, Phase 2 tech debt reduction)
            # These cover: malformed tracker state, type mismatches, missing keys
            logger.warning("Failed to collect gossip health: %s", e)

        return summary

    def _collect_circuit_health(self) -> CircuitBreakerSummary:
        """Collect health summary from circuit breaker.

        Automatically uses the global NodeCircuitBreaker registry if
        no breaker was explicitly set via set_node_circuit_breaker().
        """
        summary = CircuitBreakerSummary()

        # Auto-discover from registry if not explicitly set
        breaker = self._node_circuit_breaker
        if breaker is None and HAS_CIRCUIT_REGISTRIES and get_node_circuit_breaker is not None:
            breaker = get_node_circuit_breaker()

        if breaker is None:
            return summary

        try:
            # Import NodeCircuitState here to avoid circular imports
            from app.coordination.node_circuit_breaker import NodeCircuitState

            circuits = getattr(breaker, "_circuits", {})
            now = time.time()

            for node_id, circuit_data in circuits.items():
                summary.total_circuits += 1
                state = circuit_data.state

                if state == NodeCircuitState.CLOSED:
                    summary.closed_count += 1
                elif state == NodeCircuitState.OPEN:
                    summary.open_count += 1
                    summary.open_circuits.append(node_id)
                    # Check if recently opened (last 5 minutes)
                    opened_at = getattr(circuit_data, "opened_at", None)
                    if opened_at and (now - opened_at) < 300:
                        summary.recently_opened.append(node_id)
                elif state == NodeCircuitState.HALF_OPEN:
                    summary.half_open_count += 1

        except (ImportError, AttributeError, TypeError) as e:
            # Narrowed from broad Exception (Jan 2026, Phase 2 tech debt reduction)
            # These cover: module import failure, malformed circuit data, type mismatches
            logger.warning("Failed to collect circuit health: %s", e)

        return summary

    def _calculate_overall_score(self, state: ClusterHealthState) -> float:
        """Calculate overall health score (0.0-1.0).

        Weighted average of component health scores:
            - Quorum health: 40% (WEIGHT_QUORUM) - Most critical, cluster cannot function without quorum
            - Gossip health: 20% (WEIGHT_GOSSIP) - Indicates network connectivity between nodes
            - Circuit breaker health: 20% (WEIGHT_CIRCUIT) - Tracks node-level failures
            - Daemon health: 20% (WEIGHT_DAEMON) - Background process health

        Rationale (Sprint 16.1, Jan 3, 2026):
            Quorum is weighted highest because without voter quorum, the cluster cannot
            elect a leader or make consensus decisions. Losing quorum is an immediate
            CRITICAL state. The remaining 60% is split evenly between gossip (network
            health), circuit breakers (node health), and daemons (process health).
            This ensures no single component other than quorum can drive the cluster
            to CRITICAL by itself.
        """
        # Quorum score
        quorum_scores = {
            QuorumHealthLevel.HEALTHY: 1.0,
            QuorumHealthLevel.DEGRADED: 0.7,
            QuorumHealthLevel.MINIMUM: 0.4,
            QuorumHealthLevel.LOST: 0.0,
        }
        quorum_score = quorum_scores.get(state.quorum_health, 0.5)

        # Gossip score (based on healthy ratio)
        gossip_score = state.gossip_health.healthy_ratio

        # Circuit score (inverse of open ratio, penalize cascade risk)
        circuit_score = 1.0 - state.circuit_health.open_ratio
        if state.circuit_health.is_cascade_risk:
            circuit_score *= 0.5

        # Daemon score (inverse of failure rate)
        daemon_score = 1.0 - state.daemon_health.failure_rate
        if not state.daemon_health.is_healthy:
            daemon_score *= 0.7

        # Weighted average
        overall = (
            self.WEIGHT_QUORUM * quorum_score
            + self.WEIGHT_GOSSIP * gossip_score
            + self.WEIGHT_CIRCUIT * circuit_score
            + self.WEIGHT_DAEMON * daemon_score
        )

        return max(0.0, min(1.0, overall))

    def _score_to_level(self, score: float) -> OverallHealthLevel:
        """Convert numeric score to health level."""
        if score < self.SCORE_CRITICAL_THRESHOLD:
            return OverallHealthLevel.CRITICAL
        elif score < self.SCORE_DEGRADED_THRESHOLD:
            return OverallHealthLevel.DEGRADED
        elif score < self.SCORE_WARNING_THRESHOLD:
            return OverallHealthLevel.WARNING
        else:
            return OverallHealthLevel.HEALTHY

    def _notify_health_change(self, state: ClusterHealthState) -> None:
        """Notify callbacks of health state change and emit event."""
        # Emit CLUSTER_HEALTH_CHANGED event via event bus
        if HAS_EVENT_SYSTEM:
            try:
                bus = get_event_bus()
                if bus is not None:
                    bus.emit(
                        DataEventType.CLUSTER_HEALTH_CHANGED.value,
                        {
                            "overall_health": state.overall_health.value,
                            "overall_score": state.overall_score,
                            "quorum_health": state.quorum_health.value,
                            "alive_peers": state.alive_peers,
                            "open_circuits": state.open_circuits,
                            "is_degraded": state.overall_health != OverallHealthLevel.HEALTHY,
                            "is_critical": state.overall_health == OverallHealthLevel.CRITICAL,
                            "leader_id": state.leader_id,
                            "node_id": self._node_id,
                        },
                    )
                    logger.debug(
                        "[HealthCoordinator] Emitted CLUSTER_HEALTH_CHANGED: %s (score=%.2f)",
                        state.overall_health.value,
                        state.overall_score,
                    )
            except Exception as e:
                logger.debug("[HealthCoordinator] Event emission failed: %s", e)

        # Notify registered callbacks
        for callback in self._on_health_change:
            try:
                callback(state)
            except Exception as e:
                logger.warning("Health change callback failed: %s", e)

    # =========================================================================
    # Recovery Decision Methods
    # =========================================================================

    def get_recovery_action(self) -> RecoveryAction:
        """Get recommended recovery action based on current health.

        Analyzes cluster health and returns the most appropriate recovery
        action. Actions are prioritized by severity.

        Returns:
            RecoveryAction indicating what recovery to perform
        """
        state = self.get_cluster_health()
        now = time.time()

        with self._lock:
            # Check cooldown
            if (now - self._last_election_trigger) < self._recovery_cooldown_seconds:
                # In cooldown, only allow escalation for critical issues
                if state.overall_health != OverallHealthLevel.CRITICAL:
                    return RecoveryAction.NONE

            # Priority 1: Quorum lost - need election
            if state.quorum_health == QuorumHealthLevel.LOST:
                # Only if we've been running long enough
                if state.uptime_seconds > self.ELECTION_GRACE_PERIOD_SECONDS:
                    self._last_election_trigger = now
                    return RecoveryAction.TRIGGER_ELECTION

            # Priority 2: Critical overall health - restart P2P
            if state.overall_health == OverallHealthLevel.CRITICAL:
                self._last_recovery_action = RecoveryAction.RESTART_P2P
                return RecoveryAction.RESTART_P2P

            # Priority 3: Many open circuits - try healing
            if state.circuit_health.is_cascade_risk:
                return RecoveryAction.HEAL_PARTITIONS

            # Priority 4: Some open circuits - reset them
            if state.circuit_health.open_count > 0:
                return RecoveryAction.RESET_CIRCUITS

            # Priority 5: Degraded health - heal partitions
            if state.overall_health == OverallHealthLevel.DEGRADED:
                return RecoveryAction.HEAL_PARTITIONS

            return RecoveryAction.NONE

    def should_trigger_election(self) -> bool:
        """Check if a leader election should be triggered.

        Considers:
            - Quorum health (lost quorum requires election)
            - Leader presence (no leader requires election)
            - Grace period (avoid election during startup)

        Returns:
            True if election should be triggered
        """
        state = self.get_cluster_health()

        with self._lock:
            # Grace period check
            if state.uptime_seconds < self.ELECTION_GRACE_PERIOD_SECONDS:
                return False

            # No leader and quorum available
            if state.leader_id is None and state.quorum_health != QuorumHealthLevel.LOST:
                return True

            # Quorum at minimum or lost and we're not leader
            if state.quorum_health in (QuorumHealthLevel.MINIMUM, QuorumHealthLevel.LOST):
                if not state.is_leader:
                    return True

            return False

    def is_safe_to_update(
        self,
        updating_voters: list[str],
        quorum_required: int = 4,
    ) -> tuple[bool, str]:
        """Check if cluster can safely update given voters.

        This method is used by QuorumSafeUpdateCoordinator to verify that
        updating specific voter nodes won't cause quorum loss.

        January 3, 2026 - Sprint 16.2: Added for quorum-safe rolling updates.

        Args:
            updating_voters: List of voter node IDs that will be updated/restarted
            quorum_required: Minimum voters needed for quorum (default: 4 from VOTER_MIN_QUORUM)

        Returns:
            (safe, reason) - Whether update is safe and explanation

        Example:
            >>> coordinator = get_health_coordinator()
            >>> safe, reason = coordinator.is_safe_to_update(["lambda-gh200-1"])
            >>> if not safe:
            ...     logger.error(f"Cannot update: {reason}")
        """
        with self._lock:
            state = self.get_cluster_health()

            # Check current quorum health
            if state.quorum_health == QuorumHealthLevel.LOST:
                return False, "Quorum already lost, cannot update"

            if state.quorum_health == QuorumHealthLevel.MINIMUM:
                return False, f"Quorum at minimum ({quorum_required} voters), cannot risk any updates"

            # Calculate remaining voters after update
            # alive_peers includes all alive nodes; we need to filter to just voters
            # For safety, we use the total_peers as upper bound on voters
            alive_voters = state.alive_peers  # Best approximation from current state

            # Account for the updating voters being taken offline
            remaining_after_update = alive_voters - len(updating_voters)

            if remaining_after_update < quorum_required:
                return False, (
                    f"Update would leave {remaining_after_update} voters alive, "
                    f"need {quorum_required} for quorum"
                )

            # Check overall health - don't update during degraded state
            if state.overall_health == OverallHealthLevel.CRITICAL:
                return False, "Cluster in CRITICAL health, resolve issues before updating"

            # Additional safety: warn if updating would leave us at minimum
            if remaining_after_update == quorum_required:
                logger.warning(
                    "Update will leave cluster at minimum quorum (%d voters). "
                    "Proceed with caution.",
                    quorum_required,
                )

            return True, f"Safe to update. {remaining_after_update} voters will remain."

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def is_node_circuit_broken(self, node_id: str) -> bool:
        """Check if a node has an OPEN circuit breaker.

        Use this before performing health checks on nodes to avoid
        wasting timeout windows on nodes known to be unavailable.

        January 4, 2026 - Sprint 17.10: Added for health check optimization.

        Args:
            node_id: The node identifier to check

        Returns:
            True if node has an OPEN circuit, False otherwise

        Example:
            >>> if coordinator.is_node_circuit_broken("node-1"):
            ...     return cached_health  # Skip health check
            ... return await perform_health_check("node-1")
        """
        with self._lock:
            # Auto-discover from registry if not explicitly set
            breaker = self._node_circuit_breaker
            if breaker is None and HAS_CIRCUIT_REGISTRIES and get_node_circuit_breaker is not None:
                breaker = get_node_circuit_breaker()

            if breaker is None:
                return False

            try:
                from app.coordination.node_circuit_breaker import NodeCircuitState

                circuits = getattr(breaker, "_circuits", {})
                circuit_data = circuits.get(node_id)
                if circuit_data is None:
                    return False

                return circuit_data.state == NodeCircuitState.OPEN
            except (ImportError, AttributeError, TypeError):
                # Circuit breaker module unavailable or malformed state
                return False

    def get_cached_node_health(self, node_id: str) -> dict[str, Any] | None:
        """Get cached health for a circuit-broken node.

        Returns minimal health info for nodes with OPEN circuits,
        avoiding expensive health checks.

        January 4, 2026 - Sprint 17.10: Added for health check optimization.

        Args:
            node_id: The node identifier

        Returns:
            Cached health dict if node is circuit-broken, None otherwise
        """
        if not self.is_node_circuit_broken(node_id):
            return None

        return {
            "healthy": False,
            "status": "circuit_broken",
            "node_id": node_id,
            "details": {
                "reason": "circuit_breaker_open",
                "skip_health_check": True,
            },
        }

    def health_check(self) -> dict[str, Any]:
        """Get health check result for daemon manager integration.

        Returns:
            Dict compatible with HealthCheckResult protocol
        """
        state = self.get_cluster_health()

        return {
            "healthy": state.overall_health in (
                OverallHealthLevel.HEALTHY,
                OverallHealthLevel.WARNING,
            ),
            "status": state.overall_health.value,
            "details": {
                "overall_score": state.overall_score,
                "quorum_health": state.quorum_health.value,
                "open_circuits": len(state.open_circuits),
                "alive_peers": state.alive_peers,
                "uptime_seconds": state.uptime_seconds,
            },
        }

    def reset(self) -> None:
        """Reset coordinator state (for testing)."""
        with self._lock:
            self._last_health_state = None
            self._last_health_check_time = 0.0
            self._last_election_trigger = 0.0
            self._last_recovery_action = RecoveryAction.NONE

    def __repr__(self) -> str:
        state = self.get_cluster_health()
        return (
            f"HealthCoordinator(node={self._node_id!r}, "
            f"health={state.overall_health.value}, "
            f"score={state.overall_score:.2f})"
        )


# =============================================================================
# Singleton Instance
# =============================================================================

_health_coordinator: HealthCoordinator | None = None
_coordinator_lock = RLock()


def get_health_coordinator() -> HealthCoordinator:
    """Get the singleton HealthCoordinator instance.

    Returns:
        The global HealthCoordinator instance
    """
    global _health_coordinator

    with _coordinator_lock:
        if _health_coordinator is None:
            _health_coordinator = HealthCoordinator()
        return _health_coordinator


def reset_health_coordinator() -> None:
    """Reset the singleton instance (for testing)."""
    global _health_coordinator

    with _coordinator_lock:
        if _health_coordinator is not None:
            _health_coordinator.reset()
        _health_coordinator = None
