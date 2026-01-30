"""Health Metrics Manager for P2P Orchestrator.

January 2026: Phase 9 Aggressive Decomposition - Extracts health scoring,
peer health tracking, and monitoring loops from the monolithic p2p_orchestrator.py.

This manager handles:
- Manager health validation at startup
- Peer health scoring and circuit breaker integration
- P2P sync result recording for reputation tracking
- Cluster health snapshot logging
- Event loop latency monitoring

Dependencies:
- Orchestrator reference for peers, circuit breakers, health scores
- ThreadPoolExecutor for timeout-protected health checks
- PeerHealthState from state_manager for persistence
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scripts.p2p_orchestrator import P2POrchestrator

logger = logging.getLogger(__name__)

# Singleton instance
_health_metrics_manager: "HealthMetricsManager | None" = None

# Constants
STARTUP_GRACE_PERIOD = 10.0  # seconds
MANAGER_HEALTH_TIMEOUT = 2.0  # seconds per manager
SNAPSHOT_INTERVAL = 60  # seconds
LATENCY_CHECK_INTERVAL = 5.0  # seconds
LATENCY_WARNING_THRESHOLD = 1.0  # seconds
LATENCY_CRITICAL_THRESHOLD = 5.0  # seconds


@dataclass
class HealthMetricsConfig:
    """Configuration for HealthMetricsManager."""

    health_check_timeout: float = MANAGER_HEALTH_TIMEOUT
    startup_grace_period: float = STARTUP_GRACE_PERIOD
    snapshot_interval: float = SNAPSHOT_INTERVAL
    latency_check_interval: float = LATENCY_CHECK_INTERVAL
    latency_warning_threshold: float = LATENCY_WARNING_THRESHOLD
    latency_critical_threshold: float = LATENCY_CRITICAL_THRESHOLD


@dataclass
class HealthMetricsStats:
    """Statistics tracked by HealthMetricsManager."""

    health_checks_run: int = 0
    managers_healthy: int = 0
    managers_degraded: int = 0
    latency_warnings: int = 0
    latency_criticals: int = 0
    sync_successes: int = 0
    sync_failures: int = 0
    snapshots_logged: int = 0


@dataclass
class PeerCircuitBreaker:
    """Per-peer circuit breaker for fault isolation.

    Tracks consecutive failures and opens circuit after threshold to prevent
    wasting time on unreliable peers.
    """

    peer_id: str
    failure_count: int = 0
    success_count: int = 0
    opened_at: float = 0.0
    cooldown_seconds: float = 300.0  # 5 minutes
    failure_threshold: int = 3

    def record_success(self) -> None:
        """Record successful interaction, reset failure count."""
        self.success_count += 1
        self.failure_count = 0
        self.opened_at = 0.0

    def record_failure(self) -> None:
        """Record failed interaction, potentially open circuit."""
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold and self.opened_at == 0.0:
            self.opened_at = time.time()

    def should_allow_request(self) -> bool:
        """Check if requests should be allowed to this peer."""
        if self.opened_at == 0.0:
            return True
        # Allow if cooldown has elapsed
        if time.time() - self.opened_at > self.cooldown_seconds:
            return True
        return False


@dataclass
class PeerHealthScore:
    """Composite health score for peer selection.

    Tracks multiple dimensions: success rate, latency, request volume.
    """

    peer_id: str
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0
    degraded_threshold: float = 0.7

    @property
    def request_count(self) -> int:
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        if self.request_count == 0:
            return 1.0
        return self.success_count / self.request_count

    @property
    def avg_latency_ms(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count

    @property
    def composite_score(self) -> float:
        """Calculate composite score (0-1, higher is better)."""
        # Weighted combination of success rate and latency
        # Success rate dominates, latency is secondary
        latency_penalty = min(1.0, self.avg_latency_ms / 5000.0)  # 5s = max penalty
        return max(0.0, self.success_rate * 0.8 + (1.0 - latency_penalty) * 0.2)

    def record_request(self, *, success: bool, latency_ms: float = 0.0) -> None:
        """Record a request result."""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.total_latency_ms += latency_ms

    def is_degraded(self) -> bool:
        """Check if peer health is degraded."""
        return self.composite_score < self.degraded_threshold


class HealthMetricsManager:
    """Manages health metrics collection and peer health scoring.

    Extracted from P2POrchestrator in January 2026 (Phase 9) to improve
    modularity and testability.
    """

    def __init__(
        self,
        config: HealthMetricsConfig | None = None,
        orchestrator: "P2POrchestrator | None" = None,
    ):
        """Initialize HealthMetricsManager.

        Args:
            config: Configuration options
            orchestrator: Reference to P2P orchestrator for state access
        """
        self.config = config or HealthMetricsConfig()
        self._orchestrator = orchestrator
        self._stats = HealthMetricsStats()

        # Health check executor (singleton, shared across calls)
        self._health_check_executor: ThreadPoolExecutor | None = None

        # Per-peer tracking (can be injected or created locally)
        self._peer_circuit_breakers: dict[str, PeerCircuitBreaker] = {}
        self._peer_health_scores: dict[str, PeerHealthScore] = {}

        # Legacy circuit breaker dict (for backward compatibility)
        self._p2p_circuit_breaker: dict[str, dict[str, Any]] = {}
        self._p2p_sync_metrics: dict[str, int] = {"success": 0, "failure": 0, "bytes": 0}

        # Background task references
        self._snapshot_task: asyncio.Task | None = None
        self._latency_task: asyncio.Task | None = None

    def _get_health_check_executor(self) -> ThreadPoolExecutor:
        """Get or create singleton health check executor."""
        if self._health_check_executor is None:
            self._health_check_executor = ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="health_check_",
            )
        return self._health_check_executor

    def validate_manager_health(self) -> dict[str, Any]:
        """Validate health of all P2P managers at startup.

        Checks that all managers initialized correctly and are healthy.
        This catches initialization issues early rather than at first use.

        Returns:
            dict with manager health status and overall healthy flag
        """
        if not self._orchestrator:
            return {"all_healthy": False, "error": "No orchestrator reference"}

        self._stats.health_checks_run += 1

        managers = [
            ("state_manager", getattr(self._orchestrator, "state_manager", None)),
            ("node_selector", getattr(self._orchestrator, "node_selector", None)),
            ("sync_planner", getattr(self._orchestrator, "sync_planner", None)),
            ("selfplay_scheduler", getattr(self._orchestrator, "selfplay_scheduler", None)),
            ("job_manager", getattr(self._orchestrator, "job_manager", None)),
            ("training_coordinator", getattr(self._orchestrator, "training_coordinator", None)),
            ("loop_manager", self._orchestrator._get_loop_manager() if hasattr(self._orchestrator, "_get_loop_manager") else None),
            ("job_orchestration", getattr(self._orchestrator, "job_orchestration", None)),
        ]

        status: dict[str, Any] = {
            "managers": {},
            "all_healthy": True,
            "unhealthy_count": 0,
            "timestamp": time.time(),
        }

        # Startup grace period handling
        uptime = time.time() - getattr(self._orchestrator, "start_time", time.time())
        in_grace_period = uptime < self.config.startup_grace_period
        if in_grace_period:
            status["in_startup_grace_period"] = True
            status["grace_period_remaining"] = round(self.config.startup_grace_period - uptime, 1)

        executor = self._get_health_check_executor()

        def _safe_health_check(manager: Any) -> Any:
            """Call health_check with timeout protection."""
            return manager.health_check()

        for name, manager in managers:
            try:
                if manager is None:
                    status["managers"][name] = {"status": "not_initialized", "error": "Manager is None"}
                    status["all_healthy"] = False
                    status["unhealthy_count"] += 1
                elif hasattr(manager, "health_check"):
                    # Use ThreadPoolExecutor with timeout to prevent blocking
                    try:
                        future = executor.submit(_safe_health_check, manager)
                        health = future.result(timeout=self.config.health_check_timeout)
                    except FuturesTimeout:
                        logger.warning(
                            f"[HealthMetrics] Manager {name} health check timed out "
                            f"after {self.config.health_check_timeout}s"
                        )
                        status["managers"][name] = {
                            "status": "timeout",
                            "error": f"Health check timed out after {self.config.health_check_timeout}s",
                        }
                        status["all_healthy"] = False
                        status["unhealthy_count"] += 1
                        continue

                    # Handle both dict and HealthCheckResult return types
                    healthy_statuses = ("healthy", "ready", "running", "starting", "initializing")
                    if hasattr(health, "status"):
                        is_healthy = str(health.status).lower() in healthy_statuses
                        health_status = str(health.status)
                    else:
                        is_healthy = health.get("status") in healthy_statuses
                        health_status = health.get("status", "unknown")

                    status["managers"][name] = {
                        "status": health_status,
                        "operations": health.get("operations_count", 0) if isinstance(health, dict) else 0,
                        "errors": health.get("errors_count", 0) if isinstance(health, dict) else 0,
                    }

                    if not is_healthy:
                        if in_grace_period:
                            status["managers"][name]["status"] = "starting"
                            status["managers"][name]["original_status"] = health_status
                        else:
                            status["all_healthy"] = False
                            status["unhealthy_count"] += 1
                else:
                    status["managers"][name] = {"status": "initialized", "health_check": "not_available"}

            except Exception as e:  # noqa: BLE001
                logger.error(
                    f"[HealthMetrics] Manager {name} health check failed: {type(e).__name__}: {e}",
                    exc_info=True,
                )
                status["managers"][name] = {
                    "status": "error",
                    "error": str(e),
                    "exception_type": type(e).__name__,
                }
                status["all_healthy"] = False
                status["unhealthy_count"] += 1

        # Update stats
        if status["all_healthy"]:
            self._stats.managers_healthy += 1
        else:
            self._stats.managers_degraded += 1

        # Log results
        manager_count = len(managers)
        if status["all_healthy"]:
            if in_grace_period:
                starting = [n for n, s in status["managers"].items() if s.get("status") == "starting"]
                if starting:
                    logger.info(
                        f"[HealthMetrics] Manager health: {manager_count - len(starting)}/{manager_count} healthy, "
                        f"{len(starting)} starting (grace period: {status.get('grace_period_remaining', 0):.0f}s remaining)"
                    )
                else:
                    logger.info(f"[HealthMetrics] Manager health: all {manager_count} managers healthy")
            else:
                logger.info(f"[HealthMetrics] Manager health: all {manager_count} managers healthy")
        else:
            unhealthy = [
                n
                for n, s in status["managers"].items()
                if s.get("status") not in ("healthy", "initialized", "ready", "running", "starting")
            ]
            logger.warning(f"[HealthMetrics] Manager health: {len(unhealthy)}/{manager_count} unhealthy: {unhealthy}")

        return status

    def health_check(self) -> dict[str, Any]:
        """Return health check result for daemon protocol compliance.

        Returns:
            HealthCheckResult-compatible dict with overall health status
        """
        manager_health = self.validate_manager_health()

        uptime_seconds = time.time() - getattr(self._orchestrator, "start_time", time.time()) if self._orchestrator else 0

        # Count active peers
        active_peers = 0
        total_peers = 0
        if self._orchestrator and hasattr(self._orchestrator, "peers"):
            total_peers = len(self._orchestrator.peers)
            active_peers = sum(
                1
                for p in self._orchestrator.peers.values()
                if time.time() - getattr(p, "last_heartbeat", 0) < 120
            )

        details = {
            "node_id": getattr(self._orchestrator, "node_id", "unknown") if self._orchestrator else "unknown",
            "active_peers": active_peers,
            "total_peers": total_peers,
            "uptime_seconds": uptime_seconds,
            "managers_healthy": manager_health.get("all_healthy", False),
            "unhealthy_managers": manager_health.get("unhealthy_count", 0),
            "stats": {
                "health_checks_run": self._stats.health_checks_run,
                "latency_warnings": self._stats.latency_warnings,
                "sync_successes": self._stats.sync_successes,
                "sync_failures": self._stats.sync_failures,
            },
        }

        is_healthy = manager_health.get("all_healthy", False)
        if uptime_seconds < 10:
            is_healthy = True
            message = "HealthMetricsManager starting up"
        elif not is_healthy:
            message = f"Unhealthy: {manager_health.get('unhealthy_count', 0)} managers degraded"
        else:
            message = f"Healthy, {active_peers} peers active"

        return {
            "healthy": is_healthy,
            "status": "healthy" if is_healthy else "degraded",
            "message": message,
            "details": details,
        }

    def collect_peer_health_states(self) -> list:
        """Collect peer health states from circuit breakers and gossip tracker.

        Returns:
            List of PeerHealthState objects for persistence
        """
        from scripts.p2p.managers.state_manager import PeerHealthState

        if not self._orchestrator:
            return []

        health_states = []

        # Get circuit breaker states
        circuit_states: dict[str, dict[str, Any]] = {}
        try:
            from app.coordination.node_circuit_breaker import get_node_circuit_breaker

            breaker = get_node_circuit_breaker("health_check")
            for node_id, cb_status in breaker.get_all_states().items():
                circuit_states[node_id] = {
                    "state": cb_status.state.value,
                    "failure_count": cb_status.failure_count,
                    "opened_at": cb_status.opened_at or 0.0,
                    "last_failure": cb_status.last_failure_time or 0.0,
                }
        except ImportError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[HealthMetrics] Error collecting circuit states: {e}")

        # Get gossip health tracker states
        gossip_failures: dict[str, int] = {}
        try:
            tracker = getattr(self._orchestrator, "_gossip_health_tracker", None)
            if tracker:
                for node_id in tracker.get_suspected_peers():
                    gossip_failures[node_id] = tracker.get_failure_count(node_id)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[HealthMetrics] Error collecting gossip states: {e}")

        # Use lock-free PeerSnapshot for read-only access
        peer_snapshot = getattr(self._orchestrator, "_peer_snapshot", None)
        if not peer_snapshot:
            return health_states

        peers_snapshot = peer_snapshot.get_snapshot()
        node_id = getattr(self._orchestrator, "node_id", "")

        for peer_node_id, peer in peers_snapshot.items():
            if peer_node_id == node_id:
                continue

            # Determine peer state
            is_retired = getattr(peer, "retired", False)
            is_alive = peer.is_alive() if hasattr(peer, "is_alive") else True

            if is_retired:
                peer_state = "retired"
            elif not is_alive:
                peer_state = "dead"
            else:
                peer_state = "alive"

            # Get circuit info
            circuit_info = circuit_states.get(peer_node_id, {})
            gossip_fail_count = gossip_failures.get(peer_node_id, 0)

            # Adjust state if circuit is open
            if circuit_info.get("state") == "open" and peer_state == "alive":
                peer_state = "suspect"

            health_states.append(
                PeerHealthState(
                    node_id=peer_node_id,
                    state=peer_state,
                    failure_count=circuit_info.get("failure_count", 0),
                    gossip_failure_count=gossip_fail_count,
                    last_seen=getattr(peer, "last_heartbeat", 0.0) or 0.0,
                    last_failure=circuit_info.get("last_failure", 0.0),
                    circuit_state=circuit_info.get("state", "closed"),
                    circuit_opened_at=circuit_info.get("opened_at", 0.0),
                )
            )

        return health_states

    def apply_loaded_peer_health(self, peer_health_states: dict) -> None:
        """Apply loaded peer health state to circuit breakers and gossip tracker.

        Args:
            peer_health_states: Dict of node_id -> PeerHealthState
        """
        if not self._orchestrator:
            return

        try:
            from app.coordination.node_circuit_breaker import get_node_circuit_breaker

            breaker = get_node_circuit_breaker("health_check")
            restored_circuits = 0
            node_id = getattr(self._orchestrator, "node_id", "")

            for peer_node_id, health_state in peer_health_states.items():
                if peer_node_id == node_id:
                    continue

                # Restore circuit breaker state if circuit was open
                if health_state.circuit_state == "open":
                    breaker.force_open(peer_node_id)
                    restored_circuits += 1
                    logger.debug(
                        f"[HealthMetrics] Restored open circuit for {peer_node_id} "
                        f"(failures: {health_state.failure_count})"
                    )

                # Update peer's last_seen if we have fresh data
                peers = getattr(self._orchestrator, "peers", {})
                if peer_node_id in peers and health_state.last_seen > 0:
                    peer = peers[peer_node_id]
                    if hasattr(peer, "last_heartbeat"):
                        if health_state.last_seen > (peer.last_heartbeat or 0):
                            peer.last_heartbeat = health_state.last_seen

            if restored_circuits > 0:
                logger.info(f"[HealthMetrics] Restored {restored_circuits} open circuit breakers from state")

            # Restore gossip health tracker state if available
            tracker = getattr(self._orchestrator, "_gossip_health_tracker", None)
            if tracker:
                for peer_node_id, health_state in peer_health_states.items():
                    if health_state.gossip_failure_count >= 5:
                        for _ in range(health_state.gossip_failure_count):
                            tracker.record_gossip_failure(peer_node_id)

        except ImportError:
            logger.debug("[HealthMetrics] Node circuit breaker not available for health state restoration")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[HealthMetrics] Error applying peer health state: {e}")

    def get_peer_health_score(self, peer_id: str) -> float:
        """Calculate health score for a peer (0-100, higher is healthier).

        Considers multiple factors: resource usage, failures, NAT status, GPU.

        Args:
            peer_id: ID of the peer to score

        Returns:
            Health score from 0-100
        """
        if not self._orchestrator:
            return 0.0

        peers = getattr(self._orchestrator, "peers", {})
        peers_lock = getattr(self._orchestrator, "peers_lock", None)

        if peers_lock:
            with peers_lock:
                peer = peers.get(peer_id)
        else:
            peer = peers.get(peer_id)

        if not peer or not (hasattr(peer, "is_alive") and peer.is_alive()):
            return 0.0

        # Check per-peer circuit breaker first
        breaker = self._peer_circuit_breakers.get(peer_id)
        if breaker and not breaker.should_allow_request():
            return 0.0

        # Use PeerHealthScore if available
        health_score = self._peer_health_scores.get(peer_id)
        if health_score:
            base_score = health_score.composite_score * 100.0
        else:
            base_score = 100.0

        score = base_score

        # Penalize high resource usage
        cpu = float(getattr(peer, "cpu_percent", 0) or 0)
        memory = float(getattr(peer, "memory_percent", 0) or 0)
        disk = float(getattr(peer, "disk_percent", 0) or 0)

        score -= cpu * 0.3
        score -= memory * 0.2
        score -= max(0, disk - 50) * 0.5

        # Penalize consecutive failures
        failures = int(getattr(peer, "consecutive_failures", 0) or 0)
        score -= failures * 10

        # Penalize NAT-blocked peers
        if getattr(peer, "nat_blocked", False):
            score -= 20

        # Bonus for GPU nodes
        if getattr(peer, "has_gpu", False):
            score += 10

        # Legacy circuit breaker check
        legacy_breaker = self._p2p_circuit_breaker.get(peer_id, {})
        if legacy_breaker.get("open_until", 0) > time.time():
            score = 0

        return max(0.0, min(100.0, score))

    def record_p2p_sync_result(self, peer_id: str, success: bool, latency_ms: float = 0.0) -> None:
        """Record P2P sync result for circuit breaker, metrics, and reputation.

        Args:
            peer_id: ID of the peer
            success: Whether the sync succeeded
            latency_ms: Latency of the sync operation in milliseconds
        """
        # Use new PeerCircuitBreaker class
        if peer_id not in self._peer_circuit_breakers:
            self._peer_circuit_breakers[peer_id] = PeerCircuitBreaker(peer_id=peer_id)
        peer_breaker = self._peer_circuit_breakers[peer_id]

        # Use new PeerHealthScore class
        if peer_id not in self._peer_health_scores:
            self._peer_health_scores[peer_id] = PeerHealthScore(peer_id=peer_id)
        health_score = self._peer_health_scores[peer_id]

        # Record for new classes
        health_score.record_request(success=success, latency_ms=latency_ms)
        if success:
            peer_breaker.record_success()
            self._stats.sync_successes += 1
        else:
            peer_breaker.record_failure()
            self._stats.sync_failures += 1

        # Legacy dict-based breaker (for backward compatibility)
        breaker = self._p2p_circuit_breaker.get(peer_id, {"failures": 0, "open_until": 0})

        # Record for reputation tracking (if orchestrator available)
        # Jan 30, 2026: Use network orchestrator directly
        if self._orchestrator and hasattr(self._orchestrator, "network"):
            self._orchestrator.network.record_peer_interaction(peer_id, success, "sync")

        if success:
            breaker["failures"] = 0
            breaker["open_until"] = 0
            self._p2p_sync_metrics["success"] += 1
        else:
            breaker["failures"] = breaker.get("failures", 0) + 1
            self._p2p_sync_metrics["failure"] += 1

            # Open circuit after 3 failures
            if breaker["failures"] >= 3:
                breaker["open_until"] = time.time() + 300
                logger.info(f"[HealthMetrics] CIRCUIT BREAKER: Opening circuit for {peer_id} (3 failures)")

        self._p2p_circuit_breaker[peer_id] = breaker

        # Log if peer health is degraded
        if health_score.is_degraded():
            logger.warning(
                f"[HealthMetrics] PEER_HEALTH_DEGRADED: {peer_id} score={health_score.composite_score:.2f}, "
                f"success_rate={health_score.success_rate:.2f}"
            )

    def get_peer_health_summary(self) -> dict[str, Any]:
        """Get peer health summary for P2P stability monitoring.

        Returns:
            Dict with transitions, flapping peers, suspected count, timeout disagreements
        """
        result: dict[str, Any] = {
            "transitions_5min": 0,
            "flapping_peers": [],
            "suspected_count": 0,
            "timeout_disagreements": [],
        }

        if not self._orchestrator:
            return result

        # Get diagnostics from peer state tracker
        tracker = getattr(self._orchestrator, "_peer_state_tracker", None)
        if tracker:
            try:
                diagnostics = tracker.get_diagnostics()
                result["transitions_5min"] = diagnostics.get("total_transitions_5min", 0)
                result["flapping_peers"] = diagnostics.get("flapping_peers", [])

                # Count suspected peers
                try:
                    from scripts.p2p.diagnostics.peer_state_tracker import PeerState

                    suspected_count = 0
                    current_states = getattr(tracker, "_current_state", {})
                    for state in current_states.values():
                        if state == PeerState.SUSPECTED:
                            suspected_count += 1
                    result["suspected_count"] = suspected_count
                except ImportError:
                    pass
            except Exception as e:  # noqa: BLE001
                result["tracker_error"] = str(e)

        # Check for timeout disagreements
        try:
            self_info = getattr(self._orchestrator, "self_info", None)
            my_timeout = getattr(self_info, "effective_timeout", 0.0) if self_info else 180.0
            if not my_timeout:
                my_timeout = 180.0

            disagreements = []
            peers = getattr(self._orchestrator, "peers", {})
            for peer in peers.values():
                if not (hasattr(peer, "is_alive") and peer.is_alive()):
                    continue
                peer_timeout = getattr(peer, "effective_timeout", 0.0)
                if peer_timeout > 0:
                    ratio = peer_timeout / my_timeout if my_timeout > 0 else 1.0
                    if ratio > 1.3 or ratio < 0.7:
                        disagreements.append({
                            "node_id": peer.node_id,
                            "their_timeout": round(peer_timeout, 1),
                            "our_timeout": round(my_timeout, 1),
                            "ratio": round(ratio, 2),
                        })
            result["timeout_disagreements"] = disagreements
        except Exception as e:  # noqa: BLE001
            result["timeout_error"] = str(e)

        return result

    async def cluster_health_snapshot_loop(self) -> None:
        """Periodically log cluster health snapshots for debugging.

        Interval: configurable, default 60 seconds
        """
        await asyncio.sleep(30)  # Initial delay to let cluster stabilize

        while True:
            try:
                if self._orchestrator and hasattr(self._orchestrator, "_log_cluster_health_snapshot"):
                    self._orchestrator._log_cluster_health_snapshot()
                    self._stats.snapshots_logged += 1
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[HealthMetrics] Snapshot loop error: {e}")

            await asyncio.sleep(self.config.snapshot_interval)

    async def event_loop_latency_monitor(self) -> None:
        """Monitor event loop responsiveness to detect blocking operations.

        When asyncio.sleep(0.1) takes significantly longer than 100ms, it indicates
        the event loop was blocked by a synchronous operation.
        """
        EXPECTED_SLEEP = 0.1

        await asyncio.sleep(10)  # Initial delay to let startup complete

        consecutive_blocks = 0
        while True:
            try:
                start = time.monotonic()
                await asyncio.sleep(EXPECTED_SLEEP)
                actual = time.monotonic() - start
                latency = actual - EXPECTED_SLEEP

                if latency > self.config.latency_critical_threshold:
                    consecutive_blocks += 1
                    self._stats.latency_criticals += 1
                    logger.error(
                        f"[HealthMetrics] CRITICAL: Event loop blocked for {latency:.2f}s "
                        f"(expected {EXPECTED_SLEEP}s, actual {actual:.2f}s). "
                        f"Consecutive blocks: {consecutive_blocks}. "
                        f"Likely cause: synchronous SQLite or file I/O."
                    )
                    # Emit event for external monitoring
                    if self._orchestrator and hasattr(self._orchestrator, "_safe_emit_event"):
                        try:
                            self._orchestrator._safe_emit_event(
                                "EVENT_LOOP_BLOCKED",
                                {
                                    "node_id": getattr(self._orchestrator, "node_id", "unknown"),
                                    "latency_seconds": latency,
                                    "consecutive_blocks": consecutive_blocks,
                                    "severity": "critical",
                                },
                            )
                        except Exception:  # noqa: BLE001
                            pass
                elif latency > self.config.latency_warning_threshold:
                    consecutive_blocks += 1
                    self._stats.latency_warnings += 1
                    logger.warning(
                        f"[HealthMetrics] Event loop blocked for {latency:.2f}s "
                        f"(expected {EXPECTED_SLEEP}s, actual {actual:.2f}s). "
                        f"May indicate synchronous I/O operation."
                    )
                else:
                    if consecutive_blocks > 0:
                        logger.info(f"[HealthMetrics] Recovered after {consecutive_blocks} blocked iterations")
                    consecutive_blocks = 0

            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[HealthMetrics] Monitor error: {e}")

            await asyncio.sleep(self.config.latency_check_interval)

    async def start_background_loops(self) -> None:
        """Start background monitoring loops."""
        if self._snapshot_task is None:
            self._snapshot_task = asyncio.create_task(self.cluster_health_snapshot_loop())
        if self._latency_task is None:
            self._latency_task = asyncio.create_task(self.event_loop_latency_monitor())

    async def stop_background_loops(self) -> None:
        """Stop background monitoring loops."""
        if self._snapshot_task:
            self._snapshot_task.cancel()
            try:
                await self._snapshot_task
            except asyncio.CancelledError:
                pass
            self._snapshot_task = None

        if self._latency_task:
            self._latency_task.cancel()
            try:
                await self._latency_task
            except asyncio.CancelledError:
                pass
            self._latency_task = None

    def shutdown(self) -> None:
        """Shutdown the manager and cleanup resources."""
        if self._health_check_executor:
            self._health_check_executor.shutdown(wait=False)
            self._health_check_executor = None


# Singleton accessors
def create_health_metrics_manager(
    config: HealthMetricsConfig | None = None,
    orchestrator: "P2POrchestrator | None" = None,
) -> HealthMetricsManager:
    """Create a new HealthMetricsManager instance.

    Args:
        config: Configuration options
        orchestrator: Reference to P2P orchestrator

    Returns:
        New HealthMetricsManager instance
    """
    return HealthMetricsManager(config=config, orchestrator=orchestrator)


def get_health_metrics_manager() -> HealthMetricsManager | None:
    """Get the singleton HealthMetricsManager instance.

    Returns:
        The singleton instance, or None if not set
    """
    return _health_metrics_manager


def set_health_metrics_manager(manager: HealthMetricsManager) -> None:
    """Set the singleton HealthMetricsManager instance.

    Args:
        manager: The manager instance to set as singleton
    """
    global _health_metrics_manager
    _health_metrics_manager = manager


def reset_health_metrics_manager() -> None:
    """Reset the singleton HealthMetricsManager instance."""
    global _health_metrics_manager
    if _health_metrics_manager:
        _health_metrics_manager.shutdown()
    _health_metrics_manager = None
