"""Leader Election Logic Mixin.

Extracted from p2p_orchestrator.py for modularity.
This mixin provides core leader election and voter quorum logic.

Usage:
    class P2POrchestrator(LeaderElectionMixin, ...):
        pass

Phase 2.2 extraction - Dec 26, 2025
Refactored to use P2PMixinBase - Dec 27, 2025
P5.4: Added Raft leader integration - Dec 30, 2025
Phase 9.2: Added Prometheus quorum metrics - Jan 3, 2026

Leader Election Modes (controlled by CONSENSUS_MODE):
- "bully": Traditional bully algorithm with voter quorum (default)
- "raft": Use Raft leader as authoritative P2P leader (when available)
- "hybrid": Use Raft for work queue, bully for leadership

When CONSENSUS_MODE="raft" and Raft is initialized, the Raft leader
becomes the P2P leader. This provides:
- Single source of truth for leadership
- Reduced election churn (Raft handles leader changes)
- Strong consistency with work queue operations

Prometheus Metrics (Jan 3, 2026):
- ringrift_quorum_health_level: Current quorum health (0=LOST, 1=MINIMUM, 2=DEGRADED, 3=HEALTHY)
- ringrift_quorum_alive_voters: Number of alive voters
- ringrift_quorum_total_voters: Total configured voters
- ringrift_quorum_margin: Margin above required quorum (alive - required)
"""

from __future__ import annotations

import asyncio

from app.core.async_context import safe_create_task
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from scripts.p2p.p2p_mixin_base import P2PMixinBase

# Jan 13, 2026: Voter config management for strict quorum enforcement
try:
    from app.coordination.voter_config_types import QuorumHealth, QuorumResult
    from scripts.p2p.managers.voter_config_manager import (
        get_voter_config_manager,
        STRICT_QUORUM_ENABLED,
    )
    HAS_VOTER_CONFIG_MANAGER = True
except ImportError:
    HAS_VOTER_CONFIG_MANAGER = False
    QuorumResult = None  # type: ignore
    QuorumHealth = None  # type: ignore
    STRICT_QUORUM_ENABLED = False


# Jan 3, 2026: Quorum health levels for proactive monitoring
class QuorumHealthLevel(str, Enum):
    """Quorum health state for proactive monitoring.

    Enables early warning when voter quorum is degrading, before complete failure.

    Levels:
        HEALTHY: Sufficient margin above minimum quorum (>= quorum + 2)
        DEGRADED: At risk, one failure away from MINIMUM (== quorum + 1)
        MINIMUM: Exactly at quorum - no room for failure (== quorum)
        LOST: Below quorum - cluster cannot make progress (< quorum)
    """
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    MINIMUM = "minimum"
    LOST = "lost"


# Jan 20, 2026: Per-failure event tracking for partition grace period
# Fixes trap where global _last_healthy_quorum_at could reference much older
# recovery, masking current quorum loss
@dataclass
class QuorumFailureEvent:
    """Tracks a specific quorum failure event for grace period calculation.

    The partition grace period should apply to THIS failure event, not globally.
    When quorum recovers (even briefly), the failure event resets.

    Attributes:
        timestamp: When this specific failure started
        previous_level: What level we were at before dropping to LOST
        failure_id: Unique ID for logging/debugging
    """
    timestamp: float
    previous_level: QuorumHealthLevel
    failure_id: str = field(default_factory=lambda: uuid4().hex[:8])

if TYPE_CHECKING:
    from threading import RLock

    from scripts.p2p.models import NodeInfo
    from scripts.p2p.types import NodeRole

logger = logging.getLogger(__name__)

# Prometheus metrics (optional dependency)
try:
    from prometheus_client import REGISTRY, Gauge

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    REGISTRY = None
    Gauge = None  # type: ignore


def _get_or_create_gauge(
    name: str,
    documentation: str,
    labelnames: list[str],
) -> Any:
    """Get an existing Prometheus gauge or create a new one.

    Handles re-registration gracefully (e.g., during hot reload).
    """
    if not HAS_PROMETHEUS:
        return None
    try:
        # Try to get existing metric
        return REGISTRY._names_to_collectors.get(name)
    except (AttributeError, KeyError):
        pass
    try:
        return Gauge(name, documentation, labelnames)
    except ValueError:
        # Already registered - get from registry
        for collector in REGISTRY._names_to_collectors.values():
            if hasattr(collector, "_name") and collector._name == name:
                return collector
        return None


# Quorum health level gauge (0=LOST, 1=MINIMUM, 2=DEGRADED, 3=HEALTHY)
PROM_QUORUM_HEALTH_LEVEL = _get_or_create_gauge(
    "ringrift_quorum_health_level",
    "Current quorum health level (0=LOST, 1=MINIMUM, 2=DEGRADED, 3=HEALTHY)",
    ["node_id"],
)

# Quorum voter counts
PROM_ALIVE_VOTERS = _get_or_create_gauge(
    "ringrift_quorum_alive_voters",
    "Number of alive voters in the cluster",
    ["node_id"],
)


# Jan 3, 2026 Sprint 13.3: Leader election latency tracking
def _get_or_create_histogram(
    name: str,
    documentation: str,
    labelnames: list[str],
    buckets: tuple[float, ...],
) -> Any:
    """Get an existing Prometheus histogram or create a new one."""
    if not HAS_PROMETHEUS:
        return None
    try:
        from prometheus_client import Histogram
        # Try to get existing metric
        return REGISTRY._names_to_collectors.get(name)
    except (AttributeError, KeyError):
        pass
    try:
        from prometheus_client import Histogram
        return Histogram(name, documentation, labelnames, buckets=buckets)
    except ValueError:
        # Already registered - get from registry
        for collector in REGISTRY._names_to_collectors.values():
            if hasattr(collector, "_name") and collector._name == name:
                return collector
        return None


# Leader election latency histogram
# Buckets: 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s, 30s, 60s
PROM_ELECTION_LATENCY = _get_or_create_histogram(
    "ringrift_leader_election_latency_seconds",
    "Time taken for leader election to complete",
    ["node_id", "outcome"],  # outcome: won, lost, timeout, adopted
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

PROM_TOTAL_VOTERS = _get_or_create_gauge(
    "ringrift_quorum_total_voters",
    "Total number of configured voters in the cluster",
    ["node_id"],
)

PROM_QUORUM_MARGIN = _get_or_create_gauge(
    "ringrift_quorum_margin",
    "Margin above quorum requirement (alive - required)",
    ["node_id"],
)

# Map QuorumHealthLevel to numeric values for Prometheus
_HEALTH_LEVEL_VALUES = {
    "lost": 0,
    "minimum": 1,
    "degraded": 2,
    "healthy": 3,
}


# Load constants with fallbacks using base class helper
# Jan 13, 2026: Changed VOTER_MIN_QUORUM from 3 to 2 for simplified 3-voter setup
# With 3 voters, quorum=2 allows 1 failure (simple majority)
_CONSTANTS = P2PMixinBase._load_config_constants({
    "VOTER_MIN_QUORUM": 2,
    "CONSENSUS_MODE": "bully",
    "RAFT_ENABLED": False,
    # Jan 7, 2026: Increased from 30s to 60s to prevent split-brain during cluster startup
    "LEADER_LEASE_EXPIRY_GRACE_SECONDS": 60,  # Stale leader alerting
})

# Jan 2026: Environment variable override for quorum threshold
# Set RINGRIFT_VOTER_MIN_QUORUM=2 to allow 2-of-N quorum (lower threshold)
# This is useful when cluster stability is prioritized over split-brain prevention
_env_quorum = os.environ.get("RINGRIFT_VOTER_MIN_QUORUM", "").strip()
VOTER_MIN_QUORUM = int(_env_quorum) if _env_quorum.isdigit() else _CONSTANTS["VOTER_MIN_QUORUM"]
CONSENSUS_MODE = _CONSTANTS["CONSENSUS_MODE"]
RAFT_ENABLED = _CONSTANTS["RAFT_ENABLED"]
LEADER_LEASE_EXPIRY_GRACE_SECONDS = _CONSTANTS["LEADER_LEASE_EXPIRY_GRACE_SECONDS"]

# Session 17.48: Single-node fallback when quorum is lost for extended period
# After this timeout, a node can become its own leader for local operations
# to prevent the cluster from being stuck for hours without leadership
# January 8, 2026: Reduced from 600s to 180s for faster autonomous recovery
# January 12, 2026: Reduced from 180s to 60s for faster quorum loss recovery
# January 13, 2026: Made configurable via RINGRIFT_SINGLE_NODE_FALLBACK_TIMEOUT
# January 13, 2026: Increased default from 60s to 180s to prevent premature split-brain
# during transient network issues - gives partition healing time to run
_env_single_node_timeout = os.environ.get("RINGRIFT_SINGLE_NODE_FALLBACK_TIMEOUT", "").strip()
SINGLE_NODE_FALLBACK_TIMEOUT = int(_env_single_node_timeout) if _env_single_node_timeout.isdigit() else 180

# January 13, 2026: Allow disabling single-node fallback for strict quorum enforcement
# Set RINGRIFT_ALLOW_SINGLE_NODE=false to disable (default: true)
ALLOW_SINGLE_NODE_FALLBACK = os.environ.get("RINGRIFT_ALLOW_SINGLE_NODE", "true").lower() != "false"

# January 13, 2026: Partition tolerance - stale quorum grace period
# When network partitions occur, voters may become temporarily unreachable.
# This grace period allows the cluster to continue operating based on
# last-known quorum state, preventing quorum flip-flopping.
# Set RINGRIFT_QUORUM_STALE_GRACE to customize (default: 60 seconds)
# January 13, 2026: Increased default from 30s to 60s for better partition tolerance
_env_stale_grace = os.environ.get("RINGRIFT_QUORUM_STALE_GRACE", "").strip()
QUORUM_STALE_GRACE_SECONDS = int(_env_stale_grace) if _env_stale_grace.isdigit() else 60

# Phase 3.2 (January 2026): Dynamic voter management
# Enable via RINGRIFT_P2P_DYNAMIC_VOTER=true
#
# FEATURE STATUS: DISABLED BY DEFAULT (Sprint 3.5, Jan 2, 2026)
# This feature auto-promotes non-voter nodes to voters when the voter quorum
# is at risk (alive voters <= quorum + 1). This provides automatic failover
# when voter nodes fail.
#
# WHY DISABLED:
# - Cluster stability: Dynamic voter changes can cause split-brain scenarios
#   during network partitions if not carefully tuned.
# - Raft mode has its own membership management.
# - The current voter set (7 nodes across providers) is stable.
#
# ENABLE WHEN:
# - You have a stable cluster with reliable network connectivity
# - You understand the risks of automatic voter promotion
# - You've tested the feature in a non-production environment
#
# CONFIGURATION:
# - RINGRIFT_P2P_DYNAMIC_VOTER=true - Enable dynamic voter management
# - RINGRIFT_P2P_DYNAMIC_VOTER_PROMOTION_DELAY=60 - Seconds to wait before promotion
#
import os
DYNAMIC_VOTER_ENABLED = os.environ.get("RINGRIFT_P2P_DYNAMIC_VOTER", "false").lower() == "true"
VOTER_QUORUM_MARGIN = 1  # Promote when alive voters <= quorum_required + margin
VOTER_MIN_UPTIME_SECONDS = 300.0  # Candidate must have 5+ minutes uptime
VOTER_MAX_ERROR_RATE = 0.10  # Candidate must have <10% error rate

# Import promotion delay from constants (with fallback)
try:
    from app.p2p.constants import DYNAMIC_VOTER_PROMOTION_DELAY
except ImportError:
    DYNAMIC_VOTER_PROMOTION_DELAY = 60

# Jan 2026: Circuit breaker for voter promotion
# Prevents voter churn during cluster instability
VOTER_PROMOTION_CIRCUIT_BREAKER_FAILURES = 3  # Open CB after 3 failed promotions

# Use centralized timeout from LoopTimeouts
try:
    from scripts.p2p.loops.loop_constants import LoopTimeouts
    VOTER_PROMOTION_CIRCUIT_BREAKER_TIMEOUT = LoopTimeouts.VOTER_PROMOTION_CB
except ImportError:
    VOTER_PROMOTION_CIRCUIT_BREAKER_TIMEOUT = 300  # 5 minutes fallback

VOTER_PROMOTION_COOLOFF_PERIOD = VOTER_PROMOTION_CIRCUIT_BREAKER_TIMEOUT  # Match CB timeout


class VoterPromotionCircuitBreaker:
    """Circuit breaker to prevent voter churn during instability.

    Jan 2026: Added for Phase 1 - Dynamic Voter Promotion safety.

    Opens the circuit breaker when:
    - Multiple promotion attempts fail in quick succession
    - Cluster health drops below threshold

    When open, voter promotions are blocked until:
    - Timeout expires, OR
    - Manual reset is triggered
    """

    def __init__(
        self,
        failure_threshold: int = VOTER_PROMOTION_CIRCUIT_BREAKER_FAILURES,
        timeout_seconds: float = VOTER_PROMOTION_CIRCUIT_BREAKER_TIMEOUT,
    ):
        self._failure_threshold = failure_threshold
        self._timeout_seconds = timeout_seconds
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._opened_at = 0.0
        self._last_promotion_time = 0.0

    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is open (blocking promotions)."""
        if self._opened_at == 0.0:
            return False
        # Check if timeout has expired
        if time.time() - self._opened_at > self._timeout_seconds:
            self._reset()
            return False
        return True

    @property
    def cooloff_active(self) -> bool:
        """Check if cooloff period is active after a recent promotion."""
        if self._last_promotion_time == 0.0:
            return False
        return time.time() - self._last_promotion_time < VOTER_PROMOTION_COOLOFF_PERIOD

    def record_failure(self) -> None:
        """Record a promotion failure."""
        now = time.time()
        # Reset failure count if last failure was long ago
        if now - self._last_failure_time > self._timeout_seconds:
            self._failure_count = 0
        self._failure_count += 1
        self._last_failure_time = now

        if self._failure_count >= self._failure_threshold:
            self._opened_at = now
            logger.warning(
                f"[VoterPromotion] Circuit breaker OPENED after {self._failure_count} failures"
            )

    def record_success(self) -> None:
        """Record a successful promotion."""
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._last_promotion_time = time.time()
        if self._opened_at > 0:
            logger.info("[VoterPromotion] Circuit breaker CLOSED after successful promotion")
            self._opened_at = 0.0

    def _reset(self) -> None:
        """Reset the circuit breaker."""
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._opened_at = 0.0

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status for monitoring."""
        now = time.time()
        return {
            "is_open": self.is_open,
            "cooloff_active": self.cooloff_active,
            "failure_count": self._failure_count,
            "failure_threshold": self._failure_threshold,
            "time_since_last_failure": now - self._last_failure_time if self._last_failure_time else None,
            "time_until_reset": max(0, self._timeout_seconds - (now - self._opened_at)) if self._opened_at else None,
            "cooloff_remaining": max(0, VOTER_PROMOTION_COOLOFF_PERIOD - (now - self._last_promotion_time)) if self._last_promotion_time else None,
        }


class LeaderElectionMixin(P2PMixinBase):
    """Mixin providing core leader election logic.

    Inherits from P2PMixinBase for shared peer counting helpers.

    Requires the implementing class to have:
    State:
    - node_id: str - This node's ID
    - role: NodeRole - Current node role
    - leader_id: str | None - Current leader's ID
    - leader_lease_id: str - Active lease ID
    - leader_lease_expires: float - Lease expiry timestamp
    - last_lease_renewal: float - Last lease renewal time
    - voter_node_ids: list[str] - Configured voters
    - voter_grant_leader_id: str - Voter grant recipient
    - voter_grant_lease_id: str - Voter grant lease ID
    - voter_grant_expires: float - Voter grant expiry
    - peers_lock: RLock - Lock for peers dict
    - peers: dict[str, NodeInfo] - Active peers

    Methods:
    - _start_election() - Start new election
    - _save_state() - Persist state changes
    """

    MIXIN_TYPE = "leader_election"

    # Type hints for IDE support (implemented by P2POrchestrator)
    node_id: str
    role: Any  # NodeRole
    leader_id: str | None
    leader_lease_id: str
    leader_lease_expires: float
    last_lease_renewal: float
    voter_node_ids: list[str]
    voter_grant_leader_id: str
    voter_grant_lease_id: str
    voter_grant_expires: float
    peers_lock: "RLock"
    peers: dict[str, Any]  # dict[str, NodeInfo]

    # Jan 2026: Circuit breaker for voter promotion (singleton per mixin instance)
    _voter_promotion_cb: VoterPromotionCircuitBreaker | None = None

    # Jan 3, 2026 Sprint 13.3: Leader election latency tracking
    _election_started_at: float = 0.0
    _last_election_latency_seconds: float = 0.0
    # Jan 2026: Use deque(maxlen=10) for bounded rolling window (prevents memory leak)
    _election_latencies: deque[float] | None = None
    _elections_completed: int = 0
    _elections_won: int = 0
    _elections_lost: int = 0
    _elections_timeout: int = 0

    def _start_election_timing(self) -> None:
        """Mark the start of an election for latency tracking.

        Jan 3, 2026 Sprint 13.3: Call this at the beginning of _start_election().
        """
        self._election_started_at = time.time()

    def _record_election_latency(self, outcome: str) -> float:
        """Record election completion and calculate latency.

        Jan 3, 2026 Sprint 13.3: Call this when election completes.

        Args:
            outcome: One of 'won', 'lost', 'timeout', 'adopted'

        Returns:
            The election latency in seconds, or 0.0 if no start time recorded.
        """
        if self._election_started_at <= 0:
            return 0.0

        latency = time.time() - self._election_started_at
        self._last_election_latency_seconds = latency
        self._election_started_at = 0.0  # Reset for next election

        # Update rolling window (keep last 10)
        # Jan 2026: Use deque(maxlen=10) for automatic bounded size
        if not hasattr(self, "_election_latencies") or self._election_latencies is None:
            self._election_latencies = deque(maxlen=10)
        self._election_latencies.append(latency)  # deque auto-removes oldest when full

        # Update counters
        self._elections_completed = getattr(self, "_elections_completed", 0) + 1
        if outcome == "won":
            self._elections_won = getattr(self, "_elections_won", 0) + 1
        elif outcome == "lost":
            self._elections_lost = getattr(self, "_elections_lost", 0) + 1
        elif outcome == "timeout":
            self._elections_timeout = getattr(self, "_elections_timeout", 0) + 1

        # Record Prometheus metric
        if HAS_PROMETHEUS and PROM_ELECTION_LATENCY is not None:
            try:
                node_id = getattr(self, "node_id", "unknown")
                PROM_ELECTION_LATENCY.labels(node_id=node_id, outcome=outcome).observe(latency)
            except Exception:
                pass  # Don't let metrics failures affect elections

        self._log_info(
            f"[Election] Completed in {latency:.2f}s, outcome={outcome}"
        )
        return latency

    def get_election_latency_stats(self) -> dict[str, Any]:
        """Get election latency statistics for /status endpoint.

        Jan 3, 2026 Sprint 13.3: Returns latency stats for observability.

        Returns:
            Dict with last latency, average, min, max, and counts.
        """
        latencies = getattr(self, "_election_latencies", []) or []

        stats = {
            "last_latency_seconds": getattr(self, "_last_election_latency_seconds", 0.0),
            "elections_completed": getattr(self, "_elections_completed", 0),
            "elections_won": getattr(self, "_elections_won", 0),
            "elections_lost": getattr(self, "_elections_lost", 0),
            "elections_timeout": getattr(self, "_elections_timeout", 0),
            "election_in_progress": getattr(self, "_election_started_at", 0.0) > 0,
        }

        if latencies:
            stats["avg_latency_seconds"] = sum(latencies) / len(latencies)
            stats["min_latency_seconds"] = min(latencies)
            stats["max_latency_seconds"] = max(latencies)
            stats["latency_samples"] = len(latencies)
        else:
            stats["avg_latency_seconds"] = 0.0
            stats["min_latency_seconds"] = 0.0
            stats["max_latency_seconds"] = 0.0
            stats["latency_samples"] = 0

        # Add current election duration if in progress
        started_at = getattr(self, "_election_started_at", 0.0)
        if started_at > 0:
            stats["current_election_duration_seconds"] = time.time() - started_at

        return stats

    # Jan 7, 2026: Election timeout threshold for auto-escalation
    ELECTION_TIMEOUT_THRESHOLD_SECONDS: float = 30.0

    def _check_election_timeout(self) -> dict[str, Any]:
        """Check if current election has exceeded timeout and trigger escalation.

        Jan 7, 2026: Added for automatic recovery from stuck elections.

        When an election has been in progress for longer than ELECTION_TIMEOUT_THRESHOLD_SECONDS
        (default 30s), this method:
        1. Emits ELECTION_TIMEOUT_DETECTED event for observability
        2. Records the timeout in election statistics
        3. Returns escalation recommendation

        Returns:
            Dict with:
            - election_stuck: True if election exceeds timeout
            - duration_seconds: How long election has been running
            - should_escalate: True if escalation is recommended
            - action_taken: Description of any action taken
        """
        result = {
            "election_stuck": False,
            "duration_seconds": 0.0,
            "should_escalate": False,
            "action_taken": None,
        }

        started_at = getattr(self, "_election_started_at", 0.0)
        if started_at <= 0:
            return result  # No election in progress

        duration = time.time() - started_at
        result["duration_seconds"] = duration

        if duration < self.ELECTION_TIMEOUT_THRESHOLD_SECONDS:
            return result  # Election still within timeout

        result["election_stuck"] = True
        result["should_escalate"] = True

        # Log warning about stuck election
        self._log_warning(
            f"[Election] TIMEOUT: Election in progress for {duration:.1f}s "
            f"(threshold: {self.ELECTION_TIMEOUT_THRESHOLD_SECONDS}s) - escalating"
        )

        # Emit event for observability
        self._safe_emit_event("ELECTION_TIMEOUT_DETECTED", {
            "node_id": self.node_id,
            "duration_seconds": duration,
            "threshold_seconds": self.ELECTION_TIMEOUT_THRESHOLD_SECONDS,
            "current_leader": self.leader_id,
            "timestamp": time.time(),
        })

        # Record timeout in statistics
        self._elections_timeout = getattr(self, "_elections_timeout", 0) + 1

        # Reset election timer to allow retry
        self._election_started_at = 0.0
        result["action_taken"] = "election_timer_reset"

        return result

    def _get_voter_promotion_cb(self) -> VoterPromotionCircuitBreaker:
        """Get or create the voter promotion circuit breaker."""
        if self._voter_promotion_cb is None:
            self._voter_promotion_cb = VoterPromotionCircuitBreaker()
        return self._voter_promotion_cb

    # Jan 3, 2026: Quorum health level tracking for proactive monitoring
    _last_quorum_health_level: QuorumHealthLevel | None = None

    # Session 17.48: Single-node fallback when quorum is lost for extended period
    _quorum_lost_at: float | None = None  # Timestamp when quorum was first lost
    _single_node_mode: bool = False  # Whether we're operating in single-node fallback

    # January 13, 2026: Partition tolerance - track last-healthy quorum state
    _last_healthy_quorum_at: float | None = None  # Timestamp when quorum was last healthy
    _last_healthy_voter_list: list[str] | None = None  # Last known alive voters when healthy

    # Jan 20, 2026: Per-failure event tracking for partition grace period
    # Fixes trap where global timestamp could mask current failures
    _current_failure_event: QuorumFailureEvent | None = None

    def _calculate_dynamic_quorum(self, alive_voters: int) -> int:
        """Calculate quorum dynamically based on alive voters.

        January 13, 2026: Added for dynamic quorum that adjusts based on actual
        alive voters rather than fixed constants. This allows the cluster to
        make progress even when some voters are offline.

        Args:
            alive_voters: Number of currently alive voters

        Returns:
            Dynamic quorum requirement:
            - 3+ alive: (alive // 2) + 1 (standard majority)
            - 2 alive: 2 (need both)
            - 1 alive: 1 (emergency single-node mode)
            - 0 alive: 0 (handled by caller)

        The dynamic quorum ensures:
        - With 5 alive: need 3 (majority)
        - With 4 alive: need 3 (majority)
        - With 3 alive: need 2 (majority)
        - With 2 alive: need 2 (both)
        - With 1 alive: need 1 (emergency)
        """
        if alive_voters >= 3:
            return (alive_voters // 2) + 1
        elif alive_voters == 2:
            return 2  # Need both
        elif alive_voters == 1:
            return 1  # Emergency single-node mode
        else:
            return 0  # No quorum possible

    def _has_voter_quorum(self) -> bool:
        """Return True if we currently see enough voter nodes alive.

        January 13, 2026: Updated to use dynamic quorum based on alive voters.
        Previously used fixed VOTER_MIN_QUORUM which could fail when many voters
        were offline even if a majority of alive voters agreed.
        """
        voters = list(getattr(self, "voter_node_ids", []) or [])
        if not voters:
            return True

        # Count alive voters
        alive = self._count_alive_peers(voters)

        # Use dynamic quorum based on alive voters
        # This allows progress when some voters are offline
        # Fallback: also check against fixed minimum for safety
        dynamic_quorum = self._calculate_dynamic_quorum(alive)
        fixed_quorum = min(VOTER_MIN_QUORUM, len(voters))

        # Use the more permissive of the two (dynamic is more adaptive)
        effective_quorum = min(dynamic_quorum, fixed_quorum)

        return alive >= effective_quorum

    def _get_voter_config_version(self) -> int:
        """Get current voter config version for sync protocol.

        Jan 20, 2026: Added for automated voter config synchronization.
        Returns 0 if no config manager available (graceful degradation).
        """
        try:
            from scripts.p2p.managers.voter_config_manager import get_voter_config_manager
            manager = get_voter_config_manager()
            config = manager.get_current()
            return config.version if config else 0
        except (ImportError, AttributeError):
            return 0

    def _get_voter_config_hash(self) -> str:
        """Get current voter config hash (first 16 chars) for drift detection.

        Jan 20, 2026: Added for automated voter config synchronization.
        Returns empty string if no config manager available.
        """
        try:
            from scripts.p2p.managers.voter_config_manager import get_voter_config_manager
            manager = get_voter_config_manager()
            config = manager.get_current()
            return config.sha256_hash[:16] if config else ""
        except (ImportError, AttributeError):
            return ""

    def _check_quorum_health(self) -> QuorumHealthLevel:
        """Proactive quorum health check with early warning and single-node fallback.

        Jan 3, 2026: Added for proactive quorum degradation monitoring.
        Session 17.48: Added single-node fallback when quorum is lost for extended period.
        January 13, 2026: Added partition tolerance with stale quorum grace period.

        When quorum is lost for longer than SINGLE_NODE_FALLBACK_TIMEOUT (60 seconds),
        the node enters single-node mode and returns MINIMUM instead of LOST. This allows
        the node to become its own leader for local operations, preventing the cluster
        from being stuck for hours without leadership.

        Partition tolerance: When quorum drops to LOST, we check if we recently had
        healthy quorum (within QUORUM_STALE_GRACE_SECONDS). If so, we return DEGRADED
        instead of LOST to prevent flip-flopping during network partitions.

        Returns:
            QuorumHealthLevel indicating current quorum state:
            - HEALTHY: >= quorum + 2 voters alive
            - DEGRADED: == quorum + 1 voters alive, OR partition tolerance grace period
            - MINIMUM: == quorum voters alive (no room for failure), OR single-node fallback
            - LOST: < quorum voters alive (cluster cannot make progress)
        """
        # Jan 20, 2026: CRITICAL - Take atomic snapshot of voters at start
        # All calculations must use this snapshot to prevent race conditions
        # where voter list changes mid-calculation causing split-brain
        voters = list(getattr(self, "voter_node_ids", []) or [])
        if not voters:
            return QuorumHealthLevel.HEALTHY

        # Use snapshot values throughout - never re-read self.voter_node_ids
        total = len(voters)
        alive = self._count_alive_peers(voters)

        # January 13, 2026: Use dynamic quorum for more adaptive resilience
        # Dynamic quorum adjusts based on alive voters (majority of alive, not total)
        dynamic_quorum = self._calculate_dynamic_quorum(alive)
        fixed_quorum = min(VOTER_MIN_QUORUM, total)
        quorum = min(dynamic_quorum, fixed_quorum)  # Use more permissive

        # Get list of alive voters for tracking
        alive_voter_list = self._get_alive_peer_list(voters)

        # Determine health level based on margin above quorum
        if alive < quorum:
            level = QuorumHealthLevel.LOST
        elif alive == quorum:
            level = QuorumHealthLevel.MINIMUM
        elif alive == quorum + 1:
            level = QuorumHealthLevel.DEGRADED
        else:  # alive >= quorum + 2
            level = QuorumHealthLevel.HEALTHY

        # January 13, 2026: Track last-healthy state for partition tolerance
        # Jan 20, 2026: Only track when NOT LOST - prevents grace period updates during failure
        if level in (QuorumHealthLevel.HEALTHY, QuorumHealthLevel.DEGRADED, QuorumHealthLevel.MINIMUM):
            self._last_healthy_quorum_at = time.time()
            self._last_healthy_voter_list = alive_voter_list
            # Jan 20, 2026: Clear failure event on recovery (even brief recovery resets grace)
            if self._current_failure_event is not None:
                logger.debug(
                    f"Quorum recovered to {level.value}, clearing failure event "
                    f"{self._current_failure_event.failure_id}"
                )
                self._current_failure_event = None

        # January 13, 2026: Partition tolerance - stale quorum grace period
        # Jan 20, 2026: REFACTORED - Track per-failure event, not global timestamp
        # This fixes the trap where brief recoveries kept extending grace period
        if level == QuorumHealthLevel.LOST and QUORUM_STALE_GRACE_SECONDS > 0:
            # Start tracking this failure event if not already
            if self._current_failure_event is None:
                last_level = getattr(self, "_last_quorum_health_level", None) or QuorumHealthLevel.HEALTHY
                self._current_failure_event = QuorumFailureEvent(
                    timestamp=time.time(),
                    previous_level=last_level,
                )
                logger.debug(
                    f"Started tracking failure event {self._current_failure_event.failure_id} "
                    f"(dropped from {last_level.value} to LOST)"
                )

            # Check grace period for THIS failure event
            time_in_failure = time.time() - self._current_failure_event.timestamp
            if time_in_failure < QUORUM_STALE_GRACE_SECONDS:
                # Within grace period for this specific failure - treat as DEGRADED
                logger.info(
                    f"Quorum lost ({alive}/{total}) but within grace period for failure "
                    f"{self._current_failure_event.failure_id} "
                    f"({time_in_failure:.1f}s < {QUORUM_STALE_GRACE_SECONDS}s), "
                    f"treating as DEGRADED"
                )
                level = QuorumHealthLevel.DEGRADED

        # Session 17.48: Single-node fallback when quorum is lost for extended period
        # January 13, 2026: Respect ALLOW_SINGLE_NODE_FALLBACK setting
        if level == QuorumHealthLevel.LOST:
            # Track when quorum was first lost
            if self._quorum_lost_at is None:
                self._quorum_lost_at = time.time()
                if ALLOW_SINGLE_NODE_FALLBACK:
                    logger.warning(
                        f"Quorum lost ({alive}/{total} voters alive, need {quorum}), "
                        f"single-node fallback in {SINGLE_NODE_FALLBACK_TIMEOUT}s"
                    )
                else:
                    logger.warning(
                        f"Quorum lost ({alive}/{total} voters alive, need {quorum}), "
                        f"single-node fallback DISABLED"
                    )

            # Check if we should enter single-node fallback mode
            # Only if ALLOW_SINGLE_NODE_FALLBACK is enabled
            if ALLOW_SINGLE_NODE_FALLBACK:
                lost_duration = time.time() - self._quorum_lost_at
                if lost_duration >= SINGLE_NODE_FALLBACK_TIMEOUT:
                    if not self._single_node_mode:
                        # Jan 20, 2026: Priority-based coordination to prevent split-brain
                        # Only the node with lowest ID among alive peers can enter single-node mode
                        # This is deterministic - if all nodes agree on alive peers, they agree on priority
                        can_enter = self._can_enter_single_node_mode()
                        if can_enter:
                            logger.warning(
                                f"Quorum lost for {lost_duration:.0f}s (> {SINGLE_NODE_FALLBACK_TIMEOUT}s), "
                                f"enabling single-node fallback mode (won priority check)"
                            )
                            self._single_node_mode = True
                        else:
                            logger.info(
                                f"Quorum lost for {lost_duration:.0f}s but another node has priority "
                                f"for single-node mode, staying LOST"
                            )
                    # Return MINIMUM instead of LOST to allow local leadership (only if we entered)
                    if self._single_node_mode:
                        level = QuorumHealthLevel.MINIMUM
        else:
            # Quorum restored - reset tracking
            if self._quorum_lost_at is not None or self._single_node_mode:
                if self._single_node_mode:
                    logger.info(
                        f"Quorum restored ({alive}/{total} voters alive), "
                        f"exiting single-node fallback mode"
                    )
                self._quorum_lost_at = None
                self._single_node_mode = False

        # Always update Prometheus metrics (counts can change without level change)
        self._update_quorum_metrics(level, alive, total, quorum)

        # Emit event if health level changed
        old_level = getattr(self, "_last_quorum_health_level", None)
        if old_level is not None and level != old_level:
            self._on_quorum_health_changed(old_level, level, alive, total, quorum)

        self._last_quorum_health_level = level
        return level

    def _update_quorum_metrics(
        self,
        level: QuorumHealthLevel,
        alive: int,
        total: int,
        quorum: int,
    ) -> None:
        """Update Prometheus metrics for quorum health.

        Jan 3, 2026: Added for Phase 9.2 - Quorum health observability.

        Args:
            level: Current quorum health level
            alive: Number of alive voters
            total: Total number of voters
            quorum: Required quorum count
        """
        if not HAS_PROMETHEUS:
            return

        node_id = getattr(self, "node_id", "unknown")
        labels = {"node_id": node_id}

        try:
            if PROM_QUORUM_HEALTH_LEVEL is not None:
                level_value = _HEALTH_LEVEL_VALUES.get(level.value, 0)
                PROM_QUORUM_HEALTH_LEVEL.labels(**labels).set(level_value)

            if PROM_ALIVE_VOTERS is not None:
                PROM_ALIVE_VOTERS.labels(**labels).set(alive)

            if PROM_TOTAL_VOTERS is not None:
                PROM_TOTAL_VOTERS.labels(**labels).set(total)

            if PROM_QUORUM_MARGIN is not None:
                margin = alive - quorum
                PROM_QUORUM_MARGIN.labels(**labels).set(margin)
        except Exception as e:
            # Don't let metrics failures affect quorum logic
            logger.debug(f"Failed to update quorum metrics: {e}")

    def _on_quorum_health_changed(
        self,
        old_level: QuorumHealthLevel,
        new_level: QuorumHealthLevel,
        alive: int,
        total: int,
        quorum: int,
    ) -> None:
        """Handle quorum health level change with event emission.

        Jan 3, 2026: Proactive alerting for quorum degradation.
        """
        # Update Prometheus metrics
        self._update_quorum_metrics(new_level, alive, total, quorum)

        # Log appropriately based on severity
        if new_level == QuorumHealthLevel.LOST:
            logger.error(
                f"QUORUM LOST: {alive}/{total} voters alive, need {quorum}. "
                "Cluster cannot elect new leaders!"
            )
        elif new_level == QuorumHealthLevel.MINIMUM:
            logger.warning(
                f"QUORUM AT MINIMUM: {alive}/{total} voters alive (exactly {quorum}). "
                "Any voter failure will cause quorum loss!"
            )
        elif new_level == QuorumHealthLevel.DEGRADED:
            logger.warning(
                f"Quorum degraded: {alive}/{total} voters alive ({quorum} required). "
                "One failure from minimum quorum."
            )
        elif old_level in (QuorumHealthLevel.LOST, QuorumHealthLevel.MINIMUM):
            logger.info(
                f"Quorum recovered to {new_level.value}: {alive}/{total} voters alive."
            )

        # Emit event via base class helper
        self._safe_emit_event("QUORUM_HEALTH_CHANGED", {
            "old_level": old_level.value,
            "new_level": new_level.value,
            "alive_voters": alive,
            "total_voters": total,
            "quorum_required": quorum,
            "node_id": self.node_id,
            "timestamp": time.time(),
        })

    def get_quorum_health_status(self) -> dict[str, Any]:
        """Get detailed quorum health status for monitoring.

        Jan 3, 2026: Comprehensive quorum health info for /status endpoint.

        Returns:
            Dict with health level, counts, and thresholds
        """
        voters = list(getattr(self, "voter_node_ids", []) or [])
        total = len(voters)
        quorum = min(VOTER_MIN_QUORUM, total) if voters else 0
        alive = self._count_alive_peers(voters) if voters else 0
        level = self._check_quorum_health()

        # January 13, 2026: Include single-node fallback status
        single_node_mode = getattr(self, "_single_node_mode", False)
        quorum_lost_at = getattr(self, "_quorum_lost_at", None)
        lost_duration = (time.time() - quorum_lost_at) if quorum_lost_at else 0.0

        return {
            "health_level": level.value,
            "alive_voters": alive,
            "total_voters": total,
            "quorum_required": quorum,
            "margin": alive - quorum,
            "is_healthy": level == QuorumHealthLevel.HEALTHY,
            "is_at_risk": level in (QuorumHealthLevel.DEGRADED, QuorumHealthLevel.MINIMUM),
            "is_lost": level == QuorumHealthLevel.LOST,
            # January 13, 2026: Single-node fallback status
            "single_node_fallback": {
                "enabled": ALLOW_SINGLE_NODE_FALLBACK,
                "active": single_node_mode,
                "timeout_seconds": SINGLE_NODE_FALLBACK_TIMEOUT,
                "quorum_lost_duration": lost_duration,
                "time_until_fallback": max(0, SINGLE_NODE_FALLBACK_TIMEOUT - lost_duration) if quorum_lost_at and not single_node_mode else None,
            },
            # January 13, 2026: Partition tolerance status
            # Jan 20, 2026: Added per-failure event tracking
            "partition_tolerance": {
                "grace_period_seconds": QUORUM_STALE_GRACE_SECONDS,
                "last_healthy_at": getattr(self, "_last_healthy_quorum_at", None),
                "last_healthy_voters": getattr(self, "_last_healthy_voter_list", None),
                "time_since_healthy": (time.time() - getattr(self, "_last_healthy_quorum_at", time.time())) if getattr(self, "_last_healthy_quorum_at", None) else None,
                # Jan 20, 2026: Per-failure event tracking
                "current_failure_event": {
                    "failure_id": self._current_failure_event.failure_id if self._current_failure_event else None,
                    "started_at": self._current_failure_event.timestamp if self._current_failure_event else None,
                    "duration": time.time() - self._current_failure_event.timestamp if self._current_failure_event else None,
                    "previous_level": self._current_failure_event.previous_level.value if self._current_failure_event else None,
                } if True else {},  # Always include, even if None
                "within_grace_period": (
                    self._current_failure_event is not None
                    and (time.time() - self._current_failure_event.timestamp) < QUORUM_STALE_GRACE_SECONDS
                ),
            },
        }

    def _can_enter_single_node_mode(self) -> bool:
        """Check if this node should enter single-node mode (priority-based).

        Jan 20, 2026: Prevents split-brain where multiple nodes simultaneously
        enter single-node mode. Uses deterministic priority based on node_id.

        Priority rules:
        1. If no other peers are alive, we can enter
        2. If other peers exist, only the lowest node_id alphabetically can enter
        3. This ensures exactly one node enters single-node mode

        Returns:
            True if this node has priority to enter single-node mode
        """
        node_id = getattr(self, "node_id", None)
        if not node_id:
            return False

        # Feb 2026: Don't enter single-node mode if a preferred leader is configured
        # and this isn't it. Prevents non-preferred nodes from self-electing when
        # they temporarily lose contact with the preferred leader.
        preferred = getattr(self, "_preferred_leader_id", None)
        if preferred and preferred != node_id:
            logger.info(
                f"Suppressing single-node mode: preferred leader '{preferred}' "
                f"configured (we are '{node_id}')"
            )
            return False

        # Get all peer IDs from peers dict (with lock for thread safety)
        peers_lock = getattr(self, "peers_lock", None)
        peers = getattr(self, "peers", {})

        if peers_lock:
            with peers_lock:
                all_peer_ids = list(peers.keys())
        else:
            all_peer_ids = list(peers.keys())

        # Get alive peers from all known peers
        alive_peers = self._get_alive_peer_list(all_peer_ids) if all_peer_ids else []

        # Remove self from alive list
        alive_others = [p for p in alive_peers if p != node_id]

        if not alive_others:
            # No other peers alive - we can enter single-node mode
            logger.debug(f"No other peers alive, can enter single-node mode")
            return True

        # Check if another node already claims to be in single-node mode via gossip
        # This provides additional protection if gossip is working
        try:
            for peer_id, peer_info in peers.items():
                if peer_id == node_id:
                    continue
                # Check if peer has single_node_mode flag in their shared state
                if hasattr(peer_info, "single_node_mode") and peer_info.single_node_mode:
                    logger.info(
                        f"Peer {peer_id} already in single-node mode, deferring"
                    )
                    return False
        except (AttributeError, TypeError):
            pass  # peers dict not available or wrong format

        # Priority check: only lowest node_id can enter
        # This is deterministic - all nodes will agree if they see the same alive set
        all_candidates = sorted([node_id] + alive_others)
        priority_node = all_candidates[0]

        if priority_node == node_id:
            logger.debug(f"Have priority for single-node mode (lowest of {all_candidates})")
            return True
        else:
            logger.debug(
                f"Deferring to {priority_node} for single-node mode (we are {node_id})"
            )
            return False

    def _release_voter_grant_if_self(self) -> None:
        """Release our voter-side lease grant when stepping down.

        This shortens failover time when the leader voluntarily steps down (e.g.
        lost quorum) by not forcing other candidates to wait for the full lease
        TTL to expire.
        """
        if str(getattr(self, "voter_grant_leader_id", "") or "") != self.node_id:
            return
        self.voter_grant_leader_id = ""
        self.voter_grant_lease_id = ""
        self.voter_grant_expires = 0.0

    async def _notify_voters_lease_revoked(self) -> int:
        """Notify all voters to revoke cached lease grants.

        Jan 1, 2026: Phase 3B-C fix for leadership stability.

        When stepping down from leadership, this notifies voters to clear their
        cached grants. This prevents the 60s timeout waiting for lease expiry.

        Returns:
            Number of voters successfully notified
        """
        import aiohttp
        from aiohttp import ClientTimeout

        voter_node_ids = list(getattr(self, "voter_node_ids", []) or [])
        if not voter_node_ids:
            return 0

        # Get current epoch
        lease_epoch = int(getattr(self, "_lease_epoch", 0) or 0)

        cleared_count = 0
        # Jan 2026: Use centralized timeout for election requests
        try:
            timeout = ClientTimeout(total=LoopTimeouts.ELECTION_REQUEST)
        except NameError:
            timeout = ClientTimeout(total=3)  # Fallback if import failed

        # Jan 3, 2026: Copy peers under lock before async operations
        # This prevents race conditions when peers dict is modified concurrently
        with self.peers_lock:
            peers_snapshot = dict(self.peers)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for voter_id in voter_node_ids:
                if voter_id == self.node_id:
                    # Release our own grant synchronously
                    self._release_voter_grant_if_self()
                    cleared_count += 1
                    continue

                # Get voter peer info from snapshot (lock-safe)
                peer = peers_snapshot.get(voter_id)
                if not peer or not peer.is_alive():
                    continue

                try:
                    url = self._url_for_peer(peer, "/election/lease_revoke")
                    resp = await session.post(
                        url,
                        json={
                            "leader_id": self.node_id,
                            "epoch": lease_epoch,
                        },
                        headers=self._auth_headers(),
                    )
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("cleared"):
                            cleared_count += 1
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass  # Expected during step-down

        logger.info(f"Notified {cleared_count}/{len(voter_node_ids)} voters of lease revocation")
        return cleared_count

    # _is_leader_lease_valid: Implemented in P2POrchestrator with additional grace period logic

    def _get_voter_quorum_status(self) -> dict[str, Any]:
        """Get detailed voter quorum status for debugging.

        Returns:
            Dict with alive/total voters, quorum met status
        """
        voters = list(getattr(self, "voter_node_ids", []) or [])
        if not voters:
            return {"voters": [], "alive": 0, "total": 0, "quorum_met": True}

        quorum = min(VOTER_MIN_QUORUM, len(voters))

        # Use base class helper for getting alive peer list
        alive_voters = self._get_alive_peer_list(voters)

        return {
            "voters": voters,
            "alive": len(alive_voters),
            "alive_list": alive_voters,
            "total": len(voters),
            "quorum_required": quorum,
            "quorum_met": len(alive_voters) >= quorum,
        }

    def get_priority_voters_for_recovery(self) -> list[str]:
        """Get priority-ordered list of voters for quorum recovery.

        January 13, 2026: Fix B - Use last-healthy voter list to prioritize
        reconnection to voters that were recently alive.

        When quorum is lost, this method returns voters in priority order:
        1. Voters that were alive in the last healthy state
        2. All other configured voters

        This helps the P2P layer prioritize reconnection to voters that are
        most likely to restore quorum quickly.

        Returns:
            List of voter node IDs ordered by recovery priority
        """
        voters = list(getattr(self, "voter_node_ids", []) or [])
        if not voters:
            return []

        # Get last-healthy voter list
        last_healthy_voters = getattr(self, "_last_healthy_voter_list", None) or []

        # Build priority list: last-healthy voters first, then others
        priority_list: list[str] = []
        seen: set[str] = set()

        # Add last-healthy voters first (they're most likely to restore quorum)
        for voter_id in last_healthy_voters:
            if voter_id in voters and voter_id not in seen:
                priority_list.append(voter_id)
                seen.add(voter_id)

        # Add remaining voters
        for voter_id in voters:
            if voter_id not in seen:
                priority_list.append(voter_id)
                seen.add(voter_id)

        return priority_list

    def get_quorum_recovery_status(self) -> dict[str, Any]:
        """Get status information for quorum recovery operations.

        January 13, 2026: Provides information to help with quorum recovery.

        Returns:
            Dict with recovery-relevant information
        """
        voters = list(getattr(self, "voter_node_ids", []) or [])
        quorum = min(VOTER_MIN_QUORUM, len(voters)) if voters else 0
        alive = self._count_alive_peers(voters) if voters else 0
        last_healthy_voters = getattr(self, "_last_healthy_voter_list", None) or []

        # Calculate how many more voters needed
        voters_needed = max(0, quorum - alive)

        # Check which last-healthy voters are currently offline
        offline_healthy_voters = [v for v in last_healthy_voters if v not in self._get_alive_peer_list(voters)]

        return {
            "quorum_required": quorum,
            "current_alive": alive,
            "voters_needed_for_quorum": voters_needed,
            "priority_voters": self.get_priority_voters_for_recovery(),
            "last_healthy_voters": last_healthy_voters,
            "offline_healthy_voters": offline_healthy_voters,
            "recovery_possible": len(offline_healthy_voters) >= voters_needed,
        }

    # =========================================================================
    # Phase 3.2: Dynamic Voter Management (January 2026)
    # =========================================================================

    def _should_promote_voter(self) -> bool:
        """Check if a new voter should be promoted to maintain quorum margin.

        Returns True when:
        - Dynamic voter management is enabled
        - Alive voters are at or below quorum + margin threshold
        - Not using Raft for leadership (Raft has its own membership)

        Returns:
            True if a new voter should be promoted
        """
        if not DYNAMIC_VOTER_ENABLED:
            return False

        # Don't mess with voters when using Raft
        if self._use_raft_for_leadership():
            return False

        voters = list(getattr(self, "voter_node_ids", []) or [])
        if not voters:
            return False

        quorum = min(VOTER_MIN_QUORUM, len(voters))
        threshold = quorum + VOTER_QUORUM_MARGIN

        alive = self._count_alive_peers(voters)
        return alive <= threshold

    def _rank_voter_candidates(self) -> list[dict[str, Any]]:
        """Rank non-voter peers as candidates for voter promotion.

        Candidates are ranked by:
        1. Uptime (longer is better)
        2. Error rate (lower is better)
        3. Connectivity (more peers visible is better)

        Returns:
            List of candidate dicts sorted best-to-worst:
            [{"node_id": str, "score": float, "uptime": float, "error_rate": float}, ...]
        """
        voters = set(getattr(self, "voter_node_ids", []) or [])
        candidates: list[dict[str, Any]] = []
        now = time.time()

        with self.peers_lock:
            peers = dict(self.peers)

        for node_id, peer in peers.items():
            # Skip existing voters
            if node_id in voters:
                continue

            # Jan 7, 2026: Skip SWIM protocol entries (IP:7947 format)
            # These leak from SWIM layer and should not be promoted to voters
            if self._is_swim_peer_id(node_id):
                continue

            # Skip dead/unhealthy peers
            if not peer.is_alive():
                continue

            # Calculate uptime
            first_seen = getattr(peer, "first_seen", now)
            uptime = now - first_seen

            # Skip peers without minimum uptime
            if uptime < VOTER_MIN_UPTIME_SECONDS:
                continue

            # Get error rate from peer stats (default to 0 if not available)
            error_count = getattr(peer, "error_count", 0)
            request_count = getattr(peer, "request_count", 1)  # Avoid div by 0
            error_rate = error_count / max(request_count, 1)

            # Skip peers with high error rates
            if error_rate > VOTER_MAX_ERROR_RATE:
                continue

            # Get connectivity score (number of peers this node sees)
            peer_count = getattr(peer, "peer_count", 0)

            # Calculate composite score (higher is better)
            # Normalize factors: uptime in hours, 1-error_rate, peer_count
            uptime_score = min(uptime / 3600, 24)  # Cap at 24 hours
            reliability_score = (1.0 - error_rate) * 10
            connectivity_score = min(peer_count, 20)  # Cap at 20 peers

            score = uptime_score + reliability_score + connectivity_score

            candidates.append({
                "node_id": node_id,
                "score": score,
                "uptime": uptime,
                "error_rate": error_rate,
                "peer_count": peer_count,
            })

        # Sort by score descending (best first)
        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates

    async def _promote_to_voter(self, node_id: str) -> bool:
        """Promote a node to voter status.

        This adds the node to voter_node_ids and broadcasts the update
        to the cluster via gossip.

        Args:
            node_id: The node ID to promote

        Returns:
            True if promotion was successful
        """
        # Jan 8, 2026: Reject SWIM peer IDs to prevent voter list pollution
        if self._is_swim_peer_id(node_id):
            self._log_warning(f"Cannot promote {node_id}: SWIM peer ID not allowed as voter")
            return False

        # Validate node exists and is healthy
        with self.peers_lock:
            peer = self.peers.get(node_id)
            if not peer or not peer.is_alive():
                self._log_warning(f"Cannot promote {node_id}: peer not found or not alive")
                return False

        # Add to voter list
        if self.voter_node_ids is None:
            self.voter_node_ids = []

        if node_id in self.voter_node_ids:
            self._log_info(f"Node {node_id} is already a voter")
            return True

        self.voter_node_ids.append(node_id)

        # Log and emit event
        self._log_info(
            f"[DynamicVoter] Promoted {node_id} to voter "
            f"(total voters: {len(self.voter_node_ids)})"
        )

        self._safe_emit_event("VOTER_PROMOTED", {
            "node_id": node_id,
            "total_voters": len(self.voter_node_ids),
            "quorum_required": min(VOTER_MIN_QUORUM, len(self.voter_node_ids)),
        })

        # Save state
        if hasattr(self, "_save_state"):
            self._save_state()

        # Broadcast voter change via gossip
        await self._broadcast_voter_change("promote", node_id)

        return True

    async def _maybe_promote_voter(self) -> bool:
        """Check quorum margin and promote a voter if needed.

        This is the main entry point for dynamic voter management.
        Call this periodically (e.g., in membership loop or health check).

        Jan 2026: Added circuit breaker and cooloff protection to prevent
        voter churn during cluster instability.

        Returns:
            True if a voter was promoted
        """
        if not self._should_promote_voter():
            return False

        # Jan 2026: Check circuit breaker - block promotions during instability
        cb = self._get_voter_promotion_cb()
        if cb.is_open:
            self._log_debug(
                "[DynamicVoter] Circuit breaker OPEN - skipping promotion check"
            )
            return False

        # Jan 2026: Check cooloff period - prevent rapid successive promotions
        if cb.cooloff_active:
            self._log_debug(
                "[DynamicVoter] Cooloff period active - skipping promotion"
            )
            return False

        # Get ranked candidates
        candidates = self._rank_voter_candidates()
        if not candidates:
            self._log_warning(
                "[DynamicVoter] Quorum at risk but no suitable candidates for promotion"
            )
            return False

        # Promote the best candidate
        best = candidates[0]
        self._log_info(
            f"[DynamicVoter] Promoting {best['node_id']} (score={best['score']:.2f}, "
            f"uptime={best['uptime']:.0f}s, error_rate={best['error_rate']:.2%})"
        )

        success = await self._promote_to_voter(best["node_id"])

        # Jan 2026: Record result in circuit breaker
        if success:
            cb.record_success()
        else:
            cb.record_failure()

        return success

    async def _broadcast_voter_change(self, action: str, node_id: str) -> int:
        """Broadcast voter list change to all peers.

        Args:
            action: "promote" or "demote"
            node_id: The affected node

        Returns:
            Number of peers successfully notified
        """
        import aiohttp
        from aiohttp import ClientTimeout

        notified = 0
        # Jan 2026: Use centralized timeout for leader probes
        try:
            timeout = ClientTimeout(total=LoopTimeouts.LEADER_PROBE)
        except NameError:
            timeout = ClientTimeout(total=5)  # Fallback if import failed

        payload = {
            "action": action,
            "node_id": node_id,
            "voters": list(self.voter_node_ids or []),
            "source_node": self.node_id,
            "timestamp": time.time(),
        }

        with self.peers_lock:
            peers = list(self.peers.values())

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for peer in peers:
                if not peer.is_alive():
                    continue

                try:
                    url = self._url_for_peer(peer, "/election/voter_update")
                    resp = await session.post(
                        url,
                        json=payload,
                        headers=self._auth_headers(),
                    )
                    if resp.status == 200:
                        notified += 1
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                    pass  # Expected for some nodes

        self._log_info(f"[DynamicVoter] Notified {notified} peers of voter {action}")
        return notified

    def _check_dynamic_voter_health(self) -> dict[str, Any]:
        """Return health status for dynamic voter management.

        Returns:
            Dict with dynamic voter status and metrics
        """
        voters = list(getattr(self, "voter_node_ids", []) or [])
        quorum = min(VOTER_MIN_QUORUM, len(voters)) if voters else 0
        alive = self._count_alive_peers(voters) if voters else 0

        candidates = self._rank_voter_candidates() if DYNAMIC_VOTER_ENABLED else []

        return {
            "dynamic_voter_enabled": DYNAMIC_VOTER_ENABLED,
            "total_voters": len(voters),
            "alive_voters": alive,
            "quorum_required": quorum,
            "margin_threshold": quorum + VOTER_QUORUM_MARGIN,
            "should_promote": self._should_promote_voter(),
            "candidate_count": len(candidates),
            "top_candidate": candidates[0] if candidates else None,
        }

    def _check_leader_consistency(self) -> tuple[bool, str]:
        """Check for inconsistent leadership state.

        Returns:
            (is_consistent, reason) - True if state is consistent
        """
        # Import NodeRole lazily
        try:
            from scripts.p2p.types import NodeRole
        except ImportError:
            from enum import Enum

            class NodeRole(str, Enum):
                LEADER = "leader"
                FOLLOWER = "follower"

        if self.leader_id == self.node_id and self.role != NodeRole.LEADER:
            return False, "leader_id=self but role!=leader"
        if self.leader_id != self.node_id and self.role == NodeRole.LEADER:
            return False, "role=leader but leader_id!=self"
        return True, "consistent"

    def _check_lease_expiry(self) -> dict[str, Any]:
        """Check if the current leader's lease has expired without stepdown.

        January 2, 2026: Added for stale leader alerting (Sprint 3).

        This method checks if the known leader's lease has expired beyond the
        configured grace period. If expired, it emits a LEADER_LEASE_EXPIRED
        event to alert monitoring systems.

        Returns:
            dict with:
            - lease_stale: True if lease expired beyond grace
            - lease_remaining: Seconds until expiry (negative if expired)
            - grace_exceeded_by: Seconds beyond grace (0 if not exceeded)
            - event_emitted: True if LEADER_LEASE_EXPIRED was emitted this check
        """
        now = time.time()
        lease_expires = getattr(self, "leader_lease_expires", 0.0) or 0.0
        leader_id = self.leader_id or ""

        # Calculate time since expiry
        lease_remaining = lease_expires - now
        grace_exceeded_by = max(0, -lease_remaining - LEADER_LEASE_EXPIRY_GRACE_SECONDS)

        result = {
            "lease_stale": False,
            "lease_remaining": lease_remaining,
            "grace_exceeded_by": grace_exceeded_by,
            "event_emitted": False,
            "leader_id": leader_id,
        }

        # Skip if no leader known
        if not leader_id:
            return result

        # Skip if we ARE the leader (we handle our own stepdown)
        if leader_id == self.node_id:
            return result

        # Check if expired beyond grace
        if grace_exceeded_by > 0:
            result["lease_stale"] = True

            # Log warning
            self._log_warning(
                f"Leader {leader_id} lease expired {grace_exceeded_by:.1f}s ago "
                f"(lease_expires={lease_expires:.1f}, grace={LEADER_LEASE_EXPIRY_GRACE_SECONDS}s)"
            )

            # Emit event (async-safe via _safe_emit_event)
            self._safe_emit_event("LEADER_LEASE_EXPIRED", {
                "leader_id": leader_id,
                "lease_expiry_time": lease_expires,
                "current_time": now,
                "grace_seconds": LEADER_LEASE_EXPIRY_GRACE_SECONDS,
                "expired_by_seconds": -lease_remaining,
            })
            result["event_emitted"] = True

        return result

    def _has_voter_consensus_on_leader(self, proposed_leader: str) -> bool:
        """Check if voter quorum agrees on the proposed leader.

        This prevents split-brain scenarios where network partitions cause
        different nodes to see different leaders. Leadership is only valid
        if a quorum of voters agrees on the SAME leader.

        Args:
            proposed_leader: The node ID to validate as leader

        Returns:
            True if quorum of voters agrees on this leader
        """
        voters = list(getattr(self, "voter_node_ids", []) or [])
        if not voters:
            return True  # No voters configured = single-node mode

        quorum = min(VOTER_MIN_QUORUM, len(voters))
        voting_for_proposed = 0

        with self.peers_lock:
            peers = dict(self.peers)

        for node_id in voters:
            if node_id == self.node_id:
                # Count self's vote
                if self.leader_id == proposed_leader:
                    voting_for_proposed += 1
            else:
                peer = peers.get(node_id)
                if peer and peer.is_alive() and getattr(peer, "leader_id", None) == proposed_leader:
                    voting_for_proposed += 1

        has_consensus = voting_for_proposed >= quorum
        if not has_consensus:
            self._log_warning(
                f"No consensus on leader {proposed_leader}: "
                f"{voting_for_proposed}/{quorum} voters agree"
            )
        return has_consensus

    def _count_votes_for_leader(self, leader_id: str) -> int:
        """Count how many voters recognize this leader.

        Args:
            leader_id: The leader to count votes for

        Returns:
            Number of voters agreeing on this leader
        """
        voters = list(getattr(self, "voter_node_ids", []) or [])
        if not voters:
            return 1

        vote_count = 0
        with self.peers_lock:
            peers = dict(self.peers)

        for node_id in voters:
            if node_id == self.node_id:
                if self.leader_id == leader_id:
                    vote_count += 1
            else:
                peer = peers.get(node_id)
                if peer and peer.is_alive() and getattr(peer, "leader_id", None) == leader_id:
                    vote_count += 1

        return vote_count

    def _detect_split_brain(self) -> dict[str, Any] | None:
        """Detect if cluster is in split-brain state and trigger resolution.

        Split-brain occurs when voters report different leaders.
        This is a critical situation that must be resolved.

        Returns:
            None if no split-brain, otherwise dict with details:
            {
                "leaders_seen": {"leader1": [voter1, voter2], "leader2": [voter3]},
                "severity": "critical" | "warning",
                "recommended_action": "force_election" | "wait"
            }
        """
        voters = list(getattr(self, "voter_node_ids", []) or [])
        if not voters:
            return None

        leaders_seen: dict[str, list[str]] = {}

        with self.peers_lock:
            peers = dict(self.peers)

        # Collect leader reports from all alive voters
        for node_id in voters:
            if node_id == self.node_id:
                leader = self.leader_id or ""
            else:
                peer = peers.get(node_id)
                if not peer or not peer.is_alive():
                    continue
                leader = getattr(peer, "leader_id", "") or ""

            if leader:
                if leader not in leaders_seen:
                    leaders_seen[leader] = []
                leaders_seen[leader].append(node_id)

        # Check for split-brain
        if len(leaders_seen) <= 1:
            # Feb 2026 (2a): Populate leader_consensus_id on unanimous agreement
            if len(leaders_seen) == 1:
                consensus_leader = list(leaders_seen.keys())[0]
                if hasattr(self, "self_info") and self.self_info:
                    self.self_info.leader_consensus_id = consensus_leader
            return None  # No split-brain

        # Multiple leaders detected - this is split-brain
        severity = "critical" if len(leaders_seen) >= 3 else "warning"
        self._log_error(
            f"SPLIT-BRAIN DETECTED: {len(leaders_seen)} different leaders seen: "
            f"{list(leaders_seen.keys())}"
        )

        # Emit SPLIT_BRAIN_DETECTED event
        self._safe_emit_event("SPLIT_BRAIN_DETECTED", {
            "leaders_seen": list(leaders_seen.keys()),
            "voter_count": len(voters),
            "severity": severity,
        })

        # ENFORCEMENT: Trigger resolution immediately
        self._resolve_split_brain(leaders_seen)

        return {
            "leaders_seen": leaders_seen,
            "severity": severity,
            "recommended_action": "force_election" if severity == "critical" else "wait",
        }

    def _resolve_split_brain(self, leaders_seen: dict[str, list[str]]) -> None:
        """Resolve split-brain by demoting self if not the canonical leader.

        Canonical leader is determined by:
        1. Highest peer count (most votes)
        2. Lowest node_id as tiebreaker (deterministic)

        Args:
            leaders_seen: Dict mapping leader_id to list of voters recognizing them
        """
        # Import NodeRole lazily
        try:
            from scripts.p2p.types import NodeRole
        except ImportError:
            from enum import Enum

            class NodeRole(str, Enum):
                LEADER = "leader"
                FOLLOWER = "follower"

        # Only relevant if we think we're the leader
        if self.role != NodeRole.LEADER:
            return

        # Find canonical leader: highest vote count, then lowest node_id
        canonical_leader = None
        max_votes = 0
        for leader_id, voters in leaders_seen.items():
            vote_count = len(voters)
            if vote_count > max_votes or (vote_count == max_votes and (
                canonical_leader is None or leader_id < canonical_leader
            )):
                canonical_leader = leader_id
                max_votes = vote_count

        if canonical_leader is None:
            return

        # Feb 2026 (2a): Set consensus leader to the canonical choice
        if hasattr(self, "self_info") and self.self_info:
            self.self_info.leader_consensus_id = canonical_leader

        # If we're not the canonical leader, step down
        if canonical_leader != self.node_id:
            # Feb 2026: Don't step down if forced leader override is active
            _forced_sb = getattr(self, "_forced_leader_override", False)
            _lease_sb = time.time() < getattr(self, "leader_lease_expires", 0)
            if _forced_sb and _lease_sb:
                self._log_warning(
                    f"SPLIT-BRAIN RESOLUTION: Ignoring demotion to {canonical_leader} "
                    f"(forced leader override active for {self.node_id})"
                )
                return
            self._log_warning(
                f"SPLIT-BRAIN RESOLUTION: Demoting self ({self.node_id}) in favor of "
                f"canonical leader {canonical_leader} (has {max_votes} votes)"
            )

            # Step down from leadership
            self.role = NodeRole.FOLLOWER
            self.leader_id = canonical_leader
            self.leader_lease_id = ""
            self.leader_lease_expires = 0.0
            self.last_lease_renewal = 0.0
            self._release_voter_grant_if_self()

            # Save state and emit event
            if hasattr(self, "_save_state"):
                self._save_state()

            self._safe_emit_event("SPLIT_BRAIN_RESOLVED", {
                "demoted_node": self.node_id,
                "canonical_leader": canonical_leader,
                "canonical_votes": max_votes,
            })
        else:
            self._log_info(
                f"SPLIT-BRAIN RESOLUTION: This node ({self.node_id}) is the canonical leader"
            )

    # =========================================================================
    # P5.4: Raft Leader Integration (Dec 30, 2025)
    # =========================================================================

    def _use_raft_for_leadership(self) -> bool:
        """Check if Raft should be used for leadership decisions.

        Returns True when:
        - CONSENSUS_MODE is "raft" (not "bully" or "hybrid")
        - Raft is enabled and initialized
        - This node is a voter (non-voters use bully as fallback)

        In "hybrid" mode, Raft handles work queue but bully handles leadership.
        In "raft" mode, Raft leader is the authoritative P2P leader.

        Returns:
            True if Raft should determine leadership
        """
        # Only use Raft for leadership in "raft" mode (not "hybrid")
        if CONSENSUS_MODE != "raft":
            return False

        # Must have Raft enabled and initialized
        if not RAFT_ENABLED:
            return False

        # Check if Raft is initialized (set by ConsensusMixin)
        if not getattr(self, "_raft_initialized", False):
            return False

        # Only voters should use Raft for leadership
        # Non-voters use bully algorithm as fallback
        if self.node_id not in (self.voter_node_ids or []):
            return False

        return True

    def _get_raft_leader_node_id(self) -> str | None:
        """Get the node ID of the current Raft leader.

        Extracts the Raft leader address and maps it back to a node ID
        by looking up peers with matching addresses.

        Returns:
            Node ID of the Raft leader, or None if unavailable
        """
        raft_wq = getattr(self, "_raft_work_queue", None)
        if raft_wq is None:
            return None

        try:
            leader_addr = raft_wq._getLeader()
            if leader_addr is None:
                return None

            leader_addr_str = str(leader_addr)

            # Check if we are the leader
            advertise_host = getattr(self, "advertise_host", "")
            raft_port = 4321  # Default from constants
            try:
                from scripts.p2p.constants import RAFT_BIND_PORT
                raft_port = RAFT_BIND_PORT
            except ImportError:
                pass

            self_addr = f"{advertise_host}:{raft_port}"
            if leader_addr_str == self_addr:
                return self.node_id

            # Look up peer by Raft address
            # The address is in format "host:raft_port"
            leader_host = leader_addr_str.rsplit(":", 1)[0]

            with self.peers_lock:
                for node_id, peer in self.peers.items():
                    # Check tailscale_ip or host
                    peer_ip = getattr(peer, "tailscale_ip", None) or getattr(peer, "host", None)
                    if peer_ip == leader_host:
                        return node_id

            self._log_debug(f"Could not map Raft leader address {leader_addr_str} to node ID")
            return None

        except Exception as e:
            self._log_debug(f"Error getting Raft leader: {e}")
            return None

    def _sync_leader_from_raft(self) -> bool:
        """Synchronize P2P leader with Raft leader.

        When Raft is used for leadership, this method updates the P2P
        leader_id to match the Raft leader. This ensures work queue
        operations and leadership are consistent.

        Should be called periodically (e.g., in health check or membership loop).

        Returns:
            True if leader was synced/updated, False if no change or error
        """
        if not self._use_raft_for_leadership():
            return False

        raft_leader_id = self._get_raft_leader_node_id()
        if raft_leader_id is None:
            # No Raft leader elected yet - don't change anything
            return False

        # Check if leader changed
        current_leader = self.leader_id
        if current_leader == raft_leader_id:
            return False  # No change

        # Import NodeRole lazily
        try:
            from scripts.p2p.types import NodeRole
        except ImportError:
            from enum import Enum

            class NodeRole(str, Enum):
                LEADER = "leader"
                FOLLOWER = "follower"

        # Feb 2026: Don't override forced leadership with Raft leader
        _forced_raft = getattr(self, "_forced_leader_override", False)
        _lease_raft = time.time() < getattr(self, "leader_lease_expires", 0)
        if _forced_raft and _lease_raft and raft_leader_id != self.node_id:
            self._log_info(
                f"[Raft] Ignoring Raft leader {raft_leader_id} "
                f"(forced leader override active for {self.node_id})"
            )
            return False

        old_leader = self.leader_id
        self.leader_id = raft_leader_id

        # Update our role based on whether we're the Raft leader
        if raft_leader_id == self.node_id:
            if self.role != NodeRole.LEADER:
                self._log_info(f"[Raft] Becoming leader (Raft leader elected)")
                self.role = NodeRole.LEADER
                self.leader_lease_expires = time.time() + 300  # Raft handles leases
                self._safe_emit_event("LEADER_ELECTED", {
                    "leader_id": self.node_id,
                    "source": "raft",
                })
                # Jan 9, 2026: Broadcast leadership to all peers for fast propagation
                if hasattr(self, "_broadcast_leader_to_all_peers"):
                    epoch = getattr(self, "cluster_epoch", 0)
                    if hasattr(self, "_leadership_sm") and self._leadership_sm:
                        epoch = getattr(self._leadership_sm, "epoch", epoch)
                    safe_create_task(
                        self._broadcast_leader_to_all_peers(
                            self.node_id,
                            epoch,
                            self.leader_lease_expires,
                        ),
                        name="election-broadcast-leadership",
                    )
        else:
            if self.role == NodeRole.LEADER:
                self._log_info(f"[Raft] Stepping down, new leader: {raft_leader_id}")
                self.role = NodeRole.FOLLOWER
                self.leader_lease_id = ""
                self.leader_lease_expires = 0.0
                self._release_voter_grant_if_self()

        # Save state
        if hasattr(self, "_save_state"):
            self._save_state()

        # Emit leader change event
        self._safe_emit_event("P2P_LEADER_CHANGED", {
            "old_leader": old_leader,
            "new_leader": raft_leader_id,
            "source": "raft_sync",
        })

        self._log_info(f"[Raft] Leader synced from Raft: {old_leader} -> {raft_leader_id}")
        return True

    def get_authoritative_leader(self) -> str | None:
        """Get the authoritative leader ID, checking Raft first.

        This is the preferred method for getting the current leader.
        It checks Raft leader first (if using Raft for leadership),
        then falls back to the bully-elected leader.

        Returns:
            Node ID of the authoritative leader, or None if no leader
        """
        if self._use_raft_for_leadership():
            raft_leader = self._get_raft_leader_node_id()
            if raft_leader is not None:
                return raft_leader

        # Fall back to bully-elected leader
        return self.leader_id

    def election_health_check(self) -> dict[str, Any]:
        """Return health status for leader election subsystem.

        Also runs split-brain detection as a side effect, triggering automatic
        resolution if multiple leaders are detected.

        December 2025: Added split-brain detection integration to ensure it runs
        periodically with health checks, enabling automatic resolution.

        December 30, 2025 (P5.4): Added Raft leader sync. When using Raft for
        leadership, syncs P2P leader with Raft leader on each health check.

        January 2026 (Phase 3.2): Added dynamic voter management. When enabled,
        automatically promotes healthy nodes to voter status when quorum is at risk.

        Returns:
            dict with is_healthy, role, leader_id, quorum status, split_brain status
        """
        # December 30, 2025 (P5.4): Sync leader from Raft if applicable
        # This ensures P2P leader stays in sync with Raft leader
        raft_leader_synced = self._sync_leader_from_raft()
        using_raft = self._use_raft_for_leadership()

        has_quorum = self._has_voter_quorum()
        voter_count = len(self.voter_node_ids) if self.voter_node_ids else 0
        alive_voters = self._count_alive_peers(self.voter_node_ids or [])
        lease_remaining = max(0, self.leader_lease_expires - time.time())

        # Jan 3, 2026: Proactive quorum health monitoring with event emission
        # This tracks and emits QUORUM_HEALTH_CHANGED events on level transitions
        quorum_health_level = self._check_quorum_health()

        # December 2025: Run split-brain detection (triggers resolution if needed)
        # Skip split-brain detection when using Raft - Raft handles this
        split_brain_info = None
        if not using_raft:
            split_brain_info = self._detect_split_brain()
        has_split_brain = split_brain_info is not None

        # January 2026 (Phase 3.2): Dynamic voter management health info
        dynamic_voter_info = self._check_dynamic_voter_health()

        # January 2026 (Sprint 3): Check for stale leader lease expiry
        lease_expiry_info = self._check_lease_expiry()
        has_stale_leader = lease_expiry_info.get("lease_stale", False)

        # Unhealthy if:
        # 1. No quorum and we should have voters (unless using Raft), OR
        # 2. Split-brain detected (critical severity), OR
        # 3. Leader lease expired without stepdown (stale leader)
        # When using Raft, quorum is handled by Raft consensus
        if using_raft:
            is_healthy = not has_split_brain and not has_stale_leader
        else:
            is_healthy = (has_quorum or voter_count == 0) and not has_split_brain and not has_stale_leader

        result = {
            "is_healthy": is_healthy,
            "role": str(self.role) if self.role else "unknown",
            "leader_id": self.leader_id,
            "authoritative_leader": self.get_authoritative_leader(),
            "has_quorum": has_quorum,
            "voter_count": voter_count,
            "alive_voters": alive_voters,
            "lease_remaining_seconds": lease_remaining,
            "split_brain_detected": has_split_brain,
            "split_brain_info": split_brain_info,
            # P5.4: Raft leader integration status
            "using_raft_for_leadership": using_raft,
            "raft_leader_synced": raft_leader_synced,
            # Phase 3.2: Dynamic voter management
            "dynamic_voter": dynamic_voter_info,
            # Sprint 3: Stale leader alerting
            "stale_leader_detected": has_stale_leader,
            "lease_expiry_info": lease_expiry_info,
            # Jan 3, 2026: Quorum health level for proactive monitoring
            "quorum_health_level": quorum_health_level.value,
            "quorum_at_risk": quorum_health_level in (
                QuorumHealthLevel.DEGRADED,
                QuorumHealthLevel.MINIMUM,
                QuorumHealthLevel.LOST,
            ),
        }

        # Add Raft leader info if available
        if using_raft:
            result["raft_leader_node_id"] = self._get_raft_leader_node_id()

        # Jan 3, 2026 Sprint 13.3: Add election latency stats
        result["election_latency"] = self.get_election_latency_stats()

        # Jan 7, 2026: Check for stuck elections and auto-escalate
        election_timeout_info = self._check_election_timeout()
        result["election_timeout"] = election_timeout_info
        if election_timeout_info.get("election_stuck"):
            result["is_healthy"] = False

        return result

    def health_check(self) -> dict[str, Any]:
        """Return health status for leader election mixin (DaemonManager integration).

        December 2025: Added for unified health check interface.
        Uses base class helper for standardized response format.

        Returns:
            dict with healthy status, message, and details
        """
        status = self.election_health_check()
        is_healthy = status.get("is_healthy", False)
        role = status.get("role", "unknown")
        leader = status.get("leader_id", "none")
        message = f"Election (role={role}, leader={leader})" if is_healthy else "No quorum"
        return self._build_health_response(is_healthy, message, status)

    # =========================================================================
    # Phase 17.2: Leader Candidate Tracking (Session 17.33, Jan 5, 2026)
    # =========================================================================

    # Class-level attribute for top leader candidates
    _top_leader_candidates: list[dict[str, Any]] = []
    _candidates_updated_at: float = 0.0

    def _rank_leader_candidates(self) -> list[dict[str, Any]]:
        """Rank all alive peers as potential leader candidates.

        Session 17.33 Phase 17.2: Ranks nodes by their leader eligibility using
        the bully algorithm logic (higher priority node ID wins). Also considers
        health metrics like uptime and error rate for ranking.

        In bully algorithm, any node can become leader - the node with
        "highest" priority (typically lowest node ID in our implementation)
        wins the election. We track multiple candidates to enable faster
        failover by pre-probing backup candidates.

        Returns:
            List of candidate dicts sorted by priority (best first):
            [{"node_id": str, "priority": int, "is_alive": bool, "uptime": float}, ...]
        """
        candidates: list[dict[str, Any]] = []
        now = time.time()

        # Add ourselves as a candidate
        candidates.append({
            "node_id": self.node_id,
            "priority": self._compute_leader_priority(self.node_id),
            "is_alive": True,
            "is_self": True,
            "uptime": now - getattr(self, "_start_time", now),
            "is_voter": self.node_id in (self.voter_node_ids or []),
        })

        # Add all alive peers as candidates
        with self.peers_lock:
            peers = dict(self.peers)

        for node_id, peer in peers.items():
            if node_id == self.node_id:
                continue

            is_alive = peer.is_alive() if hasattr(peer, "is_alive") else False

            # Calculate uptime
            first_seen = getattr(peer, "first_seen", now)
            uptime = now - first_seen

            candidates.append({
                "node_id": node_id,
                "priority": self._compute_leader_priority(node_id),
                "is_alive": is_alive,
                "is_self": False,
                "uptime": uptime,
                "is_voter": node_id in (self.voter_node_ids or []),
            })

        # Sort by:
        # 1. Voters first (they're more critical for elections)
        # 2. Alive nodes first
        # 3. Higher priority (lower number = higher priority in bully)
        # 4. Longer uptime as tiebreaker
        candidates.sort(
            key=lambda c: (
                not c["is_voter"],     # Voters first
                not c["is_alive"],      # Alive first
                c["priority"],          # Lower priority number = better
                -c["uptime"],           # Longer uptime = better
            )
        )

        return candidates

    def _compute_leader_priority(self, node_id: str) -> int:
        """Compute leader priority for a node.

        In bully algorithm, we typically use node ID for deterministic ordering.
        Lower return value = higher priority = more likely to become leader.

        Args:
            node_id: The node ID to compute priority for

        Returns:
            Priority value (lower is better)
        """
        # Use hash of node_id for consistent ordering
        # We negate to make "higher" hash values have higher priority
        return hash(node_id) % 1000000

    def update_leader_candidates(self) -> list[dict[str, Any]]:
        """Update and return the top-3 leader candidates.

        Session 17.33 Phase 17.2: This method computes the current top
        leader candidates and caches them. LeaderProbeLoop can use this
        to probe backup candidates in parallel with the current leader.

        Returns:
            List of top-3 candidates (or fewer if not enough nodes)
        """
        all_candidates = self._rank_leader_candidates()

        # Filter to only alive candidates
        alive_candidates = [c for c in all_candidates if c["is_alive"]]

        # Take top 3
        top_candidates = alive_candidates[:3]

        # Cache the result
        self._top_leader_candidates = top_candidates
        self._candidates_updated_at = time.time()

        return top_candidates

    def get_leader_candidates(self, max_count: int = 3) -> list[dict[str, Any]]:
        """Get the current top leader candidates for probing.

        Session 17.33 Phase 17.2: Used by LeaderProbeLoop to get backup
        candidates to probe in parallel with the current leader.

        Args:
            max_count: Maximum number of candidates to return (default: 3)

        Returns:
            List of candidate dicts with node_id, priority, is_alive, etc.
        """
        # Update candidates if stale (>30 seconds old)
        if time.time() - self._candidates_updated_at > 30.0:
            self.update_leader_candidates()

        return self._top_leader_candidates[:max_count]

    def get_backup_leader_candidates(self) -> list[str]:
        """Get node IDs of backup leader candidates (excluding current leader).

        Session 17.33 Phase 17.2: Returns node IDs that could become leader
        if the current leader fails. Useful for pre-probing to enable
        faster failover.

        Returns:
            List of node IDs that are not the current leader
        """
        current_leader = self.leader_id
        candidates = self.get_leader_candidates()

        return [
            c["node_id"]
            for c in candidates
            if c["node_id"] != current_leader and c["is_alive"]
        ]


# Convenience functions for external use
def check_quorum(
    voters: list[str],
    alive_peers: dict[str, Any],
    self_node_id: str,
) -> bool:
    """Standalone quorum check function.

    Args:
        voters: List of voter node IDs
        alive_peers: Dict of alive peer NodeInfo objects
        self_node_id: This node's ID

    Returns:
        True if quorum is met
    """
    if not voters:
        return True

    quorum = min(VOTER_MIN_QUORUM, len(voters))
    alive = 0
    for node_id in voters:
        if node_id == self_node_id:
            alive += 1
            continue
        peer = alive_peers.get(node_id)
        if peer and hasattr(peer, "is_alive") and peer.is_alive():
            alive += 1
    return alive >= quorum


def check_quorum_strict(
    voters: list[str],
    alive_peers: dict[str, Any],
    self_node_id: str,
) -> tuple[bool, str]:
    """Strict quorum check with detailed result.

    Jan 13, 2026: Phase 2 of P2P Cluster Stability Plan
    This function provides strict quorum enforcement when enabled.

    When RINGRIFT_STRICT_QUORUM_ENFORCEMENT=true:
    - Elections CANNOT proceed without proper quorum
    - Returns detailed reason for any failures
    - Uses VoterConfigManager for versioned config

    When strict mode is disabled (default):
    - Falls back to permissive check_quorum() behavior
    - Still provides detailed logging for monitoring

    Args:
        voters: List of voter node IDs
        alive_peers: Dict of alive peer NodeInfo objects
        self_node_id: This node's ID

    Returns:
        Tuple of (quorum_ok: bool, reason: str)
        - If quorum_ok is True, reason is "OK"
        - If quorum_ok is False, reason explains the failure
    """
    # Try to use VoterConfigManager for strict enforcement
    if HAS_VOTER_CONFIG_MANAGER:
        try:
            manager = get_voter_config_manager()

            # Build list of alive voter IDs
            alive_voter_ids = []
            for node_id in voters:
                if node_id == self_node_id:
                    alive_voter_ids.append(node_id)
                    continue
                peer = alive_peers.get(node_id)
                if peer and hasattr(peer, "is_alive") and peer.is_alive():
                    alive_voter_ids.append(node_id)

            # Use strict quorum check
            result, reason = manager.check_quorum_strict(alive_voter_ids)

            if result == QuorumResult.OK:
                return True, "OK"

            # Strict mode blocks elections on any non-OK result
            if STRICT_QUORUM_ENABLED:
                logger.warning(f"[STRICT QUORUM] Election blocked: {result.value} - {reason}")
                return False, reason

            # Permissive mode: log but allow
            logger.debug(f"[QUORUM] {result.value}: {reason} (strict mode disabled)")

        except Exception as e:
            logger.warning(f"[QUORUM] VoterConfigManager error: {e}, using fallback")

    # Fall back to simple quorum check
    has_quorum = check_quorum(voters, alive_peers, self_node_id)

    if has_quorum:
        return True, "OK"
    else:
        alive_count = sum(
            1 for v in voters
            if v == self_node_id or (alive_peers.get(v) and alive_peers[v].is_alive())
        )
        return False, f"Quorum lost: {alive_count}/{len(voters)} voters alive"


def should_block_election(
    voters: list[str],
    alive_peers: dict[str, Any],
    self_node_id: str,
) -> tuple[bool, str]:
    """Check if an election should be blocked due to quorum issues.

    This is the primary election gating function. It returns True
    if the election should NOT proceed.

    Args:
        voters: List of voter node IDs
        alive_peers: Dict of alive peer NodeInfo objects
        self_node_id: This node's ID

    Returns:
        Tuple of (should_block: bool, reason: str)
    """
    quorum_ok, reason = check_quorum_strict(voters, alive_peers, self_node_id)
    return (not quorum_ok, reason)
