"""Idle Resource Daemon (December 2025 - Phase 20).

Monitors cluster for idle GPU resources and automatically spawns selfplay jobs
to maximize resource utilization.

Key features:
- Monitors GPU utilization across cluster nodes
- Spawns selfplay jobs on idle GPUs (>15min at <10% utilization)
- Matches board size to GPU memory capacity
- Queue-depth aware scaling (more aggressive when queue is deep)
- Integrates with FeedbackAccelerator for config prioritization

Usage:
    from app.coordination.idle_resource_daemon import IdleResourceDaemon

    daemon = IdleResourceDaemon()
    await daemon.start()
"""

from __future__ import annotations

__all__ = [
    "ClusterIdleState",
    "ConfigSpawnHistory",
    "IdleResourceConfig",
    "IdleResourceDaemon",
    "NodeIdleState",
    "NodeSpawnHistory",
    "NodeStatus",
    "SpawnAttempt",
    "SpawnStats",
]

import asyncio
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config.cluster_config import get_host_provider
from app.config.env import env
from app.coordination.protocols import (
    CoordinatorStatus,
    register_coordinator,
    unregister_coordinator,
)
from app.core.async_context import safe_create_task
# December 2025: Use consolidated daemon stats base class
from app.coordination.daemon_stats import JobDaemonStats

# SSH fallback for node discovery when P2P is unavailable (Dec 2025)
try:
    from app.sync.cluster_hosts import get_cluster_nodes as get_configured_hosts, ClusterNode
    from app.execution.executor import SSHExecutor
    HAS_SSH_FALLBACK = True
except ImportError:
    HAS_SSH_FALLBACK = False
    get_configured_hosts = None
    ClusterNode = None
    SSHExecutor = None

logger = logging.getLogger(__name__)

# Job scheduler integration (Phase 21.2 - Dec 2025)
try:
    from app.coordination.job_scheduler import (
        JobPriority,
        PriorityJobScheduler,
        ScheduledJob,
        get_scheduler,
    )
    HAS_JOB_SCHEDULER = True
except ImportError:
    HAS_JOB_SCHEDULER = False
    get_scheduler = None
    PriorityJobScheduler = None
    ScheduledJob = None
    JobPriority = None

# Circuit breaker integration (Phase 4 - December 2025)
# Prevents cascading failures when cluster operations are failing
try:
    from app.distributed.circuit_breaker import get_operation_breaker
    HAS_CIRCUIT_BREAKER = True
except ImportError:
    HAS_CIRCUIT_BREAKER = False
    get_operation_breaker = None

# Unified backpressure monitoring (Phase 21.5 - December 2025)
# Consolidates queue, training, disk, sync, and memory pressure signals
try:
    from app.coordination.backpressure import (
        BackpressureMonitor,
        get_backpressure_monitor,
    )
    HAS_BACKPRESSURE = True
except ImportError:
    HAS_BACKPRESSURE = False
    BackpressureMonitor = None
    get_backpressure_monitor = None

# Job stall detection (Phase 21.5 - December 2025)
# Detects stalled jobs and manages node penalties with exponential backoff
try:
    from app.coordination.stall_detection import (
        JobStallDetector,
        get_stall_detector,
    )
    HAS_STALL_DETECTION = True
except ImportError:
    HAS_STALL_DETECTION = False
    JobStallDetector = None
    get_stall_detector = None


@dataclass
class IdleResourceConfig:
    """Configuration for idle resource monitoring."""
    enabled: bool = True
    # Reduced from 300s (5min) to 15s for faster detection (Dec 2025)
    check_interval_seconds: int = 15  # 15 seconds (was 5 minutes, then 1 minute)
    idle_threshold_percent: float = 10.0  # <10% GPU utilization
    # Reduced from 900s (15min) to 120s (2min) for faster spawning (Dec 2025)
    idle_duration_seconds: int = 120  # 2 minutes before spawning (was 15 minutes)
    # Base max concurrent spawns (scaled dynamically based on idle nodes)
    max_concurrent_spawns: int = 4
    # Maximum absolute cap for spawns (Phase 2B.2 - December 2025)
    max_spawns_cap: int = 40
    # Board size to GPU memory mapping (GB)
    gpu_memory_thresholds: dict[str, int] = field(default_factory=lambda: {
        "hexagonal_4p": 80,   # Largest board, 4 players
        "hexagonal_2p": 48,
        "square19_4p": 48,
        "square19_2p": 24,
        "square8_4p": 12,
        "hex8_4p": 12,
        "square8_2p": 8,
        "hex8_2p": 8,
    })
    # P11-CRITICAL-2: Minimum free GPU memory buffer (GB) before spawning
    # This prevents OOM errors when spawning new selfplay processes
    min_free_gpu_memory_buffer_gb: float = 4.0
    # Default games per spawn
    default_games_per_spawn: int = 100
    # Queue depth thresholds for scaling
    high_queue_depth: int = 20
    medium_queue_depth: int = 10
    # Queue backpressure threshold (December 2025) - stop spawning above this
    max_queue_depth: int = 100
    # Training backlog threshold in hours - stop spawning if too much unprocessed data
    max_pending_training_hours: float = 24.0
    # Dec 26 2025: Max selfplay processes per node before skipping spawns
    # This prevents runaway process accumulation
    max_selfplay_processes_per_node: int = 50

    @classmethod
    def from_env(cls) -> IdleResourceConfig:
        """Load configuration from centralized env config."""
        config = cls()
        config.enabled = env.idle_resource_enabled
        # Faster detection: reduced from 300s to 60s (Dec 2025)
        config.check_interval_seconds = env.idle_check_interval
        config.idle_threshold_percent = env.idle_threshold
        # Faster spawning: reduced from 900s to 120s (Dec 2025)
        config.idle_duration_seconds = env.idle_duration
        return config


@dataclass
class NodeStatus:
    """Status of a cluster node."""
    node_id: str
    host: str
    gpu_utilization: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_memory_used_gb: float = 0.0
    last_seen: float = 0.0
    idle_since: float = 0.0  # When node became idle
    active_jobs: int = 0
    provider: str = "unknown"


@dataclass
class SpawnAttempt:
    """Record of a single spawn attempt (December 2025 - spawn tracking)."""
    node_id: str
    config_key: str
    games: int
    timestamp: float
    success: bool
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class NodeSpawnHistory:
    """Per-node spawn history for failure tracking (December 2025)."""
    node_id: str
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    consecutive_failures: int = 0
    last_attempt_time: float = 0.0
    last_success_time: float = 0.0
    last_failure_time: float = 0.0
    backoff_until: float = 0.0  # Skip spawning until this time
    last_error: str | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)."""
        if self.total_attempts == 0:
            return 1.0  # No history = optimistic
        return self.successful_attempts / self.total_attempts


@dataclass
class ConfigSpawnHistory:
    """Per-config spawn history (December 2025)."""
    config_key: str
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    games_spawned: int = 0
    last_success_time: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)."""
        if self.total_attempts == 0:
            return 1.0
        return self.successful_attempts / self.total_attempts


@dataclass
class SpawnStats(JobDaemonStats):
    """Statistics for spawn operations.

    December 2025: Now extends JobDaemonStats for consistent tracking.
    Inherits: jobs_processed, jobs_succeeded, jobs_failed, is_healthy(), etc.
    """

    # IdleResource-specific fields
    games_spawned: int = 0

    # Backward compatibility aliases
    @property
    def total_spawns(self) -> int:
        """Alias for jobs_processed (backward compatibility)."""
        return self.jobs_processed

    @property
    def successful_spawns(self) -> int:
        """Alias for jobs_succeeded (backward compatibility)."""
        return self.jobs_succeeded

    @property
    def failed_spawns(self) -> int:
        """Alias for jobs_failed (backward compatibility)."""
        return self.jobs_failed

    @property
    def last_spawn_time(self) -> float:
        """Alias for last_job_time (backward compatibility)."""
        return self.last_job_time

    def record_spawn_success(self, games: int = 0) -> None:
        """Record a successful spawn."""
        self.record_job_success()
        if games > 0:
            self.games_spawned += games

    def record_spawn_failure(self, error: str) -> None:
        """Record a failed spawn."""
        self.record_job_failure(error)


@dataclass
class NodeIdleState:
    """Idle state for a single node - broadcasted to cluster (December 2025).

    This enables cluster-wide visibility into which nodes have idle resources,
    allowing better coordination of job distribution and load balancing.
    """
    node_id: str
    host: str
    is_idle: bool
    gpu_utilization: float
    gpu_memory_free_gb: float
    gpu_memory_total_gb: float
    idle_duration_seconds: float
    recommended_config: str
    provider: str
    timestamp: float = field(default_factory=time.time)
    active_jobs: int = 0


@dataclass
class ClusterIdleState:
    """Aggregated cluster-wide idle state (December 2025).

    Provides a complete picture of idle resources across the cluster,
    enabling optimal job placement decisions.
    """
    total_nodes: int
    idle_nodes: int
    total_idle_gpu_memory_gb: float
    nodes: list[NodeIdleState] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def idle_ratio(self) -> float:
        """Fraction of nodes that are idle."""
        return self.idle_nodes / max(1, self.total_nodes)

    @property
    def has_idle_capacity(self) -> bool:
        """Check if cluster has any idle capacity."""
        return self.idle_nodes > 0


class IdleResourceDaemon:
    """Daemon that monitors idle resources and spawns selfplay jobs.

    Continuously scans cluster for underutilized GPUs and automatically
    spawns selfplay jobs to maximize resource usage.
    """

    def __init__(self, config: IdleResourceConfig | None = None):
        self.config = config or IdleResourceConfig.from_env()
        self.node_id = socket.gethostname()
        self._running = False
        self._stats = SpawnStats()
        self._monitor_task: asyncio.Task | None = None
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_spawns)

        # Track node states
        self._node_states: dict[str, NodeStatus] = {}

        # Spawn tracking (December 2025)
        self._node_spawn_history: dict[str, NodeSpawnHistory] = {}
        self._config_spawn_history: dict[str, ConfigSpawnHistory] = {}
        self._recent_spawn_attempts: list[SpawnAttempt] = []  # Last 100 attempts
        self._max_recent_attempts: int = 100

        # Backoff configuration
        self._base_backoff_seconds: float = 60.0  # 1 minute initial backoff
        self._max_backoff_seconds: float = 1800.0  # 30 minutes max backoff
        self._max_consecutive_failures: int = 5  # Cap for exponential backoff

        # Cluster-wide state tracking (December 2025)
        self._cluster_idle_states: dict[str, NodeIdleState] = {}
        self._last_broadcast_time: float = 0.0
        self._broadcast_interval: float = 30.0  # Broadcast local state every 30s
        self._state_stale_threshold: float = 120.0  # States older than 2min are stale

        # CoordinatorProtocol state
        self._coordinator_status = CoordinatorStatus.INITIALIZING
        self._start_time: float = 0.0
        self._events_processed: int = 0
        self._errors_count: int = 0
        self._last_error: str = ""

        logger.info(
            f"IdleResourceDaemon initialized: node={self.node_id}, "
            f"interval={self.config.check_interval_seconds}s, "
            f"idle_threshold={self.config.idle_threshold_percent}%"
        )

    def _get_dynamic_max_spawns(self) -> int:
        """Calculate max concurrent spawns based on idle node count.

        December 2025 - Phase 2B.2: Scale job spawning proportionally
        to idle capacity instead of fixed 4.

        December 2025 - Backpressure integration: Reduce spawns when
        any node is experiencing resource pressure.

        Returns:
            Max spawns: base * (idle_nodes / 4), capped at max_spawns_cap,
            reduced by backpressure factor
        """
        # Count idle nodes
        idle_nodes = sum(
            1 for node in self._node_states.values()
            if node.gpu_utilization < self.config.idle_threshold_percent
            and node.idle_since > 0
        )

        if idle_nodes <= 0:
            return self.config.max_concurrent_spawns

        # Scale: 2 spawns per idle node, minimum of base (4), max of cap (40)
        scaled = max(
            self.config.max_concurrent_spawns,
            min(idle_nodes * 2, self.config.max_spawns_cap)
        )

        # Apply backpressure reduction (December 2025)
        reduction = getattr(self, "_backpressure_spawn_reduction", 1.0)
        if reduction < 1.0:
            reduced = int(scaled * reduction)
            if reduced < scaled:
                logger.debug(
                    f"[IdleResourceDaemon] Backpressure reducing spawns: {scaled} → {reduced}"
                )
            scaled = max(1, reduced)  # Always allow at least 1 spawn

        # Apply cluster health reduction (December 2025 - P2P integration)
        cluster_health = getattr(self, "_cluster_health_reduction", 1.0)
        if cluster_health < 1.0:
            health_reduced = int(scaled * cluster_health)
            if health_reduced < scaled:
                logger.debug(
                    f"[IdleResourceDaemon] Cluster health reducing spawns: {scaled} → {health_reduced}"
                )
            scaled = max(1, health_reduced)

        return scaled

    def is_running(self) -> bool:
        """Check if the daemon is running."""
        return self._running

    def _is_node_in_backoff(self, node_id: str) -> bool:
        """Check if node is currently in backoff period (December 2025)."""
        history = self._node_spawn_history.get(node_id)
        if not history:
            return False
        return time.time() < history.backoff_until

    def _get_node_backoff_remaining(self, node_id: str) -> float:
        """Get remaining backoff seconds for a node."""
        history = self._node_spawn_history.get(node_id)
        if not history:
            return 0.0
        remaining = history.backoff_until - time.time()
        return max(0.0, remaining)

    def _calculate_backoff(self, consecutive_failures: int) -> float:
        """Calculate exponential backoff with cap (December 2025).

        Backoff = base * 2^(failures-1), capped at max_backoff.
        """
        capped_failures = min(consecutive_failures, self._max_consecutive_failures)
        backoff = self._base_backoff_seconds * (2 ** max(0, capped_failures - 1))
        return min(backoff, self._max_backoff_seconds)

    def _record_spawn_attempt(
        self,
        node_id: str,
        config_key: str,
        games: int,
        success: bool,
        error: str | None = None,
        duration: float = 0.0,
    ) -> None:
        """Record a spawn attempt for tracking (December 2025)."""
        now = time.time()

        # Record in recent attempts (ring buffer)
        attempt = SpawnAttempt(
            node_id=node_id,
            config_key=config_key,
            games=games,
            timestamp=now,
            success=success,
            error=error,
            duration_seconds=duration,
        )
        self._recent_spawn_attempts.append(attempt)
        if len(self._recent_spawn_attempts) > self._max_recent_attempts:
            self._recent_spawn_attempts.pop(0)

        # Update node history
        if node_id not in self._node_spawn_history:
            self._node_spawn_history[node_id] = NodeSpawnHistory(node_id=node_id)

        node_history = self._node_spawn_history[node_id]
        node_history.total_attempts += 1
        node_history.last_attempt_time = now

        if success:
            node_history.successful_attempts += 1
            node_history.last_success_time = now
            node_history.consecutive_failures = 0  # Reset on success
            node_history.backoff_until = 0.0  # Clear backoff
        else:
            node_history.failed_attempts += 1
            node_history.last_failure_time = now
            node_history.consecutive_failures += 1
            node_history.last_error = error
            # Apply exponential backoff
            backoff = self._calculate_backoff(node_history.consecutive_failures)
            node_history.backoff_until = now + backoff
            logger.warning(
                f"[IdleResourceDaemon] Node {node_id} spawn failed "
                f"(consecutive: {node_history.consecutive_failures}), "
                f"backoff for {backoff:.0f}s"
            )

        # Update config history
        if config_key not in self._config_spawn_history:
            self._config_spawn_history[config_key] = ConfigSpawnHistory(config_key=config_key)

        config_history = self._config_spawn_history[config_key]
        config_history.total_attempts += 1

        if success:
            config_history.successful_attempts += 1
            config_history.last_success_time = now
            config_history.games_spawned += games
        else:
            config_history.failed_attempts += 1

    def get_spawn_history(self) -> dict[str, Any]:
        """Get comprehensive spawn history (December 2025)."""
        now = time.time()

        # Aggregate node stats
        node_stats = {}
        for node_id, history in self._node_spawn_history.items():
            node_stats[node_id] = {
                "total_attempts": history.total_attempts,
                "success_rate": round(history.success_rate, 3),
                "consecutive_failures": history.consecutive_failures,
                "in_backoff": self._is_node_in_backoff(node_id),
                "backoff_remaining_seconds": round(self._get_node_backoff_remaining(node_id), 0),
                "last_error": history.last_error,
            }

        # Aggregate config stats
        config_stats = {}
        for config_key, history in self._config_spawn_history.items():
            config_stats[config_key] = {
                "total_attempts": history.total_attempts,
                "success_rate": round(history.success_rate, 3),
                "games_spawned": history.games_spawned,
            }

        # Recent attempt summary
        recent_window = 300  # Last 5 minutes
        recent_attempts = [
            a for a in self._recent_spawn_attempts
            if now - a.timestamp < recent_window
        ]
        recent_success = sum(1 for a in recent_attempts if a.success)
        recent_failed = len(recent_attempts) - recent_success

        return {
            "overall": {
                "total_spawns": self._stats.total_spawns,
                "successful_spawns": self._stats.successful_spawns,
                "failed_spawns": self._stats.failed_spawns,
                "success_rate": round(
                    self._stats.successful_spawns / max(1, self._stats.total_spawns), 3
                ),
                "games_spawned": self._stats.games_spawned,
            },
            "recent_5min": {
                "attempts": len(recent_attempts),
                "successful": recent_success,
                "failed": recent_failed,
            },
            "nodes_in_backoff": sum(
                1 for n in self._node_spawn_history
                if self._is_node_in_backoff(n)
            ),
            "nodes": node_stats,
            "configs": config_stats,
        }

    async def start(self) -> None:
        """Start the idle resource daemon."""
        # December 2025: Coordinator-only mode check
        # This daemon spawns selfplay processes - should NEVER run on coordinator nodes
        from app.config.env import env
        if env.is_coordinator or not env.selfplay_enabled:
            self._coordinator_status = CoordinatorStatus.STOPPED
            logger.info(
                f"IdleResourceDaemon skipped on coordinator node: {env.node_id} "
                f"(is_coordinator={env.is_coordinator}, selfplay_enabled={env.selfplay_enabled})"
            )
            return

        if not self.config.enabled:
            self._coordinator_status = CoordinatorStatus.STOPPED
            logger.info("IdleResourceDaemon disabled by config")
            return

        if self._coordinator_status == CoordinatorStatus.RUNNING:
            return  # Already running

        self._running = True
        self._coordinator_status = CoordinatorStatus.RUNNING
        self._start_time = time.time()
        logger.info(f"Starting IdleResourceDaemon on {self.node_id}")

        # Register with coordinator protocol
        try:
            register_coordinator("idle_resource", self)
        except Exception as e:
            logger.debug(f"Failed to register coordinator: {e}")

        # December 2025: Subscribe to backpressure events
        self._wire_backpressure_events()

        # December 2025: Subscribe to P2P cluster health events
        self._wire_p2p_health_events()

        # December 2025: Subscribe to cluster-wide idle state broadcast
        self._wire_cluster_state_events()

        # December 2025: Subscribe to selfplay priority updates from feedback loop
        self._wire_selfplay_target_events()

        # Start monitoring loop
        self._monitor_task = safe_create_task(
            self._monitor_loop(),
            name="idle_resource_monitor"
        )

    def _wire_backpressure_events(self) -> None:
        """Subscribe to backpressure events for cluster-wide coordination.

        December 2025: When any node experiences resource pressure, we should
        reduce spawning cluster-wide to avoid overwhelming the pipeline.
        """
        try:
            from app.coordination.event_router import DataEventType, get_router

            router = get_router()

            def _on_backpressure_activated(event: Any) -> None:
                """Handle BACKPRESSURE_ACTIVATED - reduce spawn rate."""
                payload = event if isinstance(event, dict) else getattr(event, "payload", {})
                node_id = payload.get("node_id", "unknown")
                resource_type = payload.get("resource_type", "unknown")
                usage_pct = payload.get("usage_pct", 0)

                logger.info(
                    f"[IdleResourceDaemon] Backpressure activated on {node_id}: "
                    f"{resource_type} at {usage_pct:.1f}%"
                )

                # Reduce max concurrent spawns temporarily
                if not hasattr(self, "_backpressure_active"):
                    self._backpressure_active = set()
                self._backpressure_active.add(node_id)

                # Apply global backoff when any node is under pressure
                if len(self._backpressure_active) > 0:
                    self._backpressure_spawn_reduction = 0.5  # 50% reduction

            def _on_backpressure_released(event: Any) -> None:
                """Handle BACKPRESSURE_RELEASED - resume normal rate."""
                payload = event if isinstance(event, dict) else getattr(event, "payload", {})
                node_id = payload.get("node_id", "unknown")

                logger.info(f"[IdleResourceDaemon] Backpressure released on {node_id}")

                if hasattr(self, "_backpressure_active") and node_id in self._backpressure_active:
                    self._backpressure_active.discard(node_id)

                # Resume normal spawning when all nodes are clear
                if hasattr(self, "_backpressure_active") and len(self._backpressure_active) == 0:
                    self._backpressure_spawn_reduction = 1.0  # No reduction

            router.subscribe(DataEventType.BACKPRESSURE_ACTIVATED.value, _on_backpressure_activated)
            router.subscribe(DataEventType.BACKPRESSURE_RELEASED.value, _on_backpressure_released)

            # Initialize backpressure state
            self._backpressure_active: set[str] = set()
            self._backpressure_spawn_reduction: float = 1.0

            logger.info(
                "[IdleResourceDaemon] Subscribed to backpressure events "
                "(BACKPRESSURE_ACTIVATED, BACKPRESSURE_RELEASED)"
            )

        except ImportError:
            logger.debug("[IdleResourceDaemon] Event router not available, backpressure handling disabled")
        except Exception as e:
            logger.warning(f"[IdleResourceDaemon] Failed to wire backpressure events: {e}")

    def _wire_p2p_health_events(self) -> None:
        """Subscribe to P2P cluster health events.

        December 2025: Prevents spawning jobs on unhealthy/failing nodes.
        This closes a critical gap where idle_resource_daemon could spawn
        work on nodes that are in the process of failing or recovering.

        Events handled:
        - NODE_UNHEALTHY: Mark node as unavailable for spawning
        - NODE_RECOVERED: Re-enable node for spawning
        - P2P_CLUSTER_UNHEALTHY: Reduce spawning cluster-wide
        """
        try:
            from app.coordination.event_router import DataEventType, get_router

            router = get_router()

            # Track unhealthy nodes - skip them during spawn decisions
            if not hasattr(self, "_unhealthy_nodes"):
                self._unhealthy_nodes: set[str] = set()

            def _on_node_unhealthy(event: Any) -> None:
                """Handle NODE_UNHEALTHY - mark node as unavailable."""
                payload = event if isinstance(event, dict) else getattr(event, "payload", {})
                node_id = payload.get("node_id", "")
                reason = payload.get("reason", "unknown")

                if node_id:
                    self._unhealthy_nodes.add(node_id)
                    logger.warning(
                        f"[IdleResourceDaemon] Node {node_id} marked unhealthy: {reason}. "
                        f"Will skip spawning on this node."
                    )

            def _on_node_recovered(event: Any) -> None:
                """Handle NODE_RECOVERED - re-enable node for spawning."""
                payload = event if isinstance(event, dict) else getattr(event, "payload", {})
                node_id = payload.get("node_id", "")

                if node_id and node_id in self._unhealthy_nodes:
                    self._unhealthy_nodes.discard(node_id)
                    logger.info(
                        f"[IdleResourceDaemon] Node {node_id} recovered. "
                        f"Re-enabled for job spawning."
                    )

            def _on_cluster_unhealthy(event: Any) -> None:
                """Handle P2P_CLUSTER_UNHEALTHY - reduce spawn rate cluster-wide."""
                payload = event if isinstance(event, dict) else getattr(event, "payload", {})
                healthy_nodes = payload.get("healthy_nodes", 0)
                total_nodes = payload.get("total_nodes", 0)

                logger.warning(
                    f"[IdleResourceDaemon] Cluster unhealthy: {healthy_nodes}/{total_nodes} nodes. "
                    f"Reducing spawn rate."
                )

                # Apply cluster-wide spawn reduction
                if not hasattr(self, "_cluster_health_reduction"):
                    self._cluster_health_reduction = 1.0

                if total_nodes > 0:
                    # Scale spawn rate by healthy node ratio
                    self._cluster_health_reduction = max(0.2, healthy_nodes / total_nodes)
                else:
                    self._cluster_health_reduction = 0.5

            def _on_cluster_healthy(event: Any) -> None:
                """Handle P2P_CLUSTER_HEALTHY - restore normal spawn rate."""
                logger.info("[IdleResourceDaemon] Cluster healthy. Restoring normal spawn rate.")
                if hasattr(self, "_cluster_health_reduction"):
                    self._cluster_health_reduction = 1.0

            # Subscribe to P2P health events
            if hasattr(DataEventType, 'NODE_UNHEALTHY'):
                router.subscribe(DataEventType.NODE_UNHEALTHY.value, _on_node_unhealthy)
            if hasattr(DataEventType, 'NODE_RECOVERED'):
                router.subscribe(DataEventType.NODE_RECOVERED.value, _on_node_recovered)
            if hasattr(DataEventType, 'P2P_CLUSTER_UNHEALTHY'):
                router.subscribe(DataEventType.P2P_CLUSTER_UNHEALTHY.value, _on_cluster_unhealthy)
            if hasattr(DataEventType, 'P2P_CLUSTER_HEALTHY'):
                router.subscribe(DataEventType.P2P_CLUSTER_HEALTHY.value, _on_cluster_healthy)

            # Initialize health tracking
            self._unhealthy_nodes: set[str] = set()
            self._cluster_health_reduction: float = 1.0

            logger.info(
                "[IdleResourceDaemon] Subscribed to P2P health events "
                "(NODE_UNHEALTHY, NODE_RECOVERED, P2P_CLUSTER_UNHEALTHY, P2P_CLUSTER_HEALTHY)"
            )

        except ImportError:
            logger.debug("[IdleResourceDaemon] Event router not available, P2P health handling disabled")
        except Exception as e:
            logger.warning(f"[IdleResourceDaemon] Failed to wire P2P health events: {e}")

    def _wire_cluster_state_events(self) -> None:
        """Subscribe to cluster-wide idle state broadcast events.

        December 2025: Enables cluster-wide visibility into idle resources.
        Each node broadcasts its idle state periodically, allowing:
        - Better job placement decisions
        - Cluster-wide resource optimization
        - Visibility into total idle capacity

        Events:
        - IDLE_STATE_BROADCAST: Receive idle state from other nodes
        - IDLE_STATE_REQUEST: Respond with local idle state
        """
        try:
            from app.coordination.event_router import DataEventType, get_router

            router = get_router()

            def _on_idle_state_broadcast(event: Any) -> None:
                """Handle IDLE_STATE_BROADCAST from other nodes."""
                payload = event if isinstance(event, dict) else getattr(event, "payload", {})
                node_id = payload.get("node_id", "")

                if not node_id or node_id == self.node_id:
                    return  # Ignore self or invalid

                try:
                    # Parse and store remote node's idle state
                    state = NodeIdleState(
                        node_id=node_id,
                        host=payload.get("host", ""),
                        is_idle=payload.get("is_idle", False),
                        gpu_utilization=payload.get("gpu_utilization", 0.0),
                        gpu_memory_free_gb=payload.get("gpu_memory_free_gb", 0.0),
                        gpu_memory_total_gb=payload.get("gpu_memory_total_gb", 0.0),
                        idle_duration_seconds=payload.get("idle_duration_seconds", 0.0),
                        recommended_config=payload.get("recommended_config", ""),
                        provider=payload.get("provider", "unknown"),
                        timestamp=payload.get("timestamp", time.time()),
                        active_jobs=payload.get("active_jobs", 0),
                    )
                    self._cluster_idle_states[node_id] = state

                    # Clean up stale states
                    self._prune_stale_cluster_states()

                except Exception as e:
                    logger.debug(f"[IdleResourceDaemon] Failed to parse idle state from {node_id}: {e}")

            def _on_idle_state_request(event: Any) -> None:
                """Handle IDLE_STATE_REQUEST - respond with our current state."""
                # Trigger immediate broadcast of local state
                safe_create_task(
                    self._broadcast_local_state(force=True),
                    name="idle_state_response"
                )

            # Subscribe to cluster state events
            if hasattr(DataEventType, 'IDLE_STATE_BROADCAST'):
                router.subscribe(DataEventType.IDLE_STATE_BROADCAST.value, _on_idle_state_broadcast)
            if hasattr(DataEventType, 'IDLE_STATE_REQUEST'):
                router.subscribe(DataEventType.IDLE_STATE_REQUEST.value, _on_idle_state_request)

            logger.info(
                "[IdleResourceDaemon] Subscribed to cluster idle state events "
                "(IDLE_STATE_BROADCAST, IDLE_STATE_REQUEST)"
            )

        except ImportError:
            logger.debug("[IdleResourceDaemon] Event router not available, cluster state disabled")
        except Exception as e:
            logger.warning(f"[IdleResourceDaemon] Failed to wire cluster state events: {e}")

    def _wire_selfplay_target_events(self) -> None:
        """Subscribe to SELFPLAY_TARGET_UPDATED events from feedback loop.

        December 2025: When the feedback loop detects quality degradation,
        regression, or promotion failures, it emits SELFPLAY_TARGET_UPDATED
        to request priority selfplay for specific configs. This daemon should
        prioritize spawning selfplay jobs for those configs on idle GPUs.

        Events handled:
        - SELFPLAY_TARGET_UPDATED: Update config priority for job spawning
        """
        try:
            from app.coordination.event_router import DataEventType, get_router

            router = get_router()

            # Track priority configs - these should be preferred when spawning
            if not hasattr(self, "_priority_configs"):
                self._priority_configs: dict[str, dict[str, Any]] = {}

            def _on_selfplay_target_updated(event: Any) -> None:
                """Handle SELFPLAY_TARGET_UPDATED - prioritize config for spawning."""
                payload = event if isinstance(event, dict) else getattr(event, "payload", {})
                config_key = payload.get("config_key") or payload.get("config", "")
                priority = payload.get("priority", "normal")
                reason = payload.get("reason", "feedback_loop")
                target_jobs = payload.get("target_jobs") or payload.get("target_games")
                exploration_boost = payload.get("exploration_boost", 1.0)

                if not config_key:
                    return

                # Store priority config with metadata
                self._priority_configs[config_key] = {
                    "priority": priority,
                    "reason": reason,
                    "target_jobs": target_jobs,
                    "exploration_boost": exploration_boost,
                    "timestamp": time.time(),
                }

                logger.info(
                    f"[IdleResourceDaemon] SELFPLAY_TARGET_UPDATED: {config_key} "
                    f"priority={priority} reason={reason}"
                    + (f" target_jobs={target_jobs}" if target_jobs else "")
                    + (f" exploration_boost={exploration_boost:.1f}x" if exploration_boost > 1.0 else "")
                )

                # If priority is urgent, trigger immediate spawn check
                if priority == "urgent":
                    logger.info(
                        f"[IdleResourceDaemon] Urgent priority for {config_key}, "
                        f"triggering immediate spawn check"
                    )
                    # Could trigger immediate spawn here, but the regular loop
                    # will pick it up quickly (15s interval by default)

            # Subscribe to the event
            if hasattr(DataEventType, 'SELFPLAY_TARGET_UPDATED'):
                router.subscribe(DataEventType.SELFPLAY_TARGET_UPDATED.value, _on_selfplay_target_updated)
                logger.info(
                    "[IdleResourceDaemon] Subscribed to SELFPLAY_TARGET_UPDATED "
                    "(feedback loop priority updates)"
                )
            else:
                logger.debug("[IdleResourceDaemon] SELFPLAY_TARGET_UPDATED event type not available")

        except ImportError:
            logger.debug("[IdleResourceDaemon] Event router not available, selfplay target updates disabled")
        except Exception as e:
            logger.warning(f"[IdleResourceDaemon] Failed to wire selfplay target events: {e}")

    def get_priority_configs(self) -> dict[str, dict[str, Any]]:
        """Get current priority configs for spawning decisions.

        Returns:
            Dict mapping config_key to priority metadata (priority, reason, timestamp, etc.)
        """
        if not hasattr(self, "_priority_configs"):
            return {}

        # Prune stale priorities (older than 30 minutes)
        now = time.time()
        stale_threshold = 30 * 60  # 30 minutes
        stale_keys = [
            key for key, meta in self._priority_configs.items()
            if now - meta.get("timestamp", 0) > stale_threshold
        ]
        for key in stale_keys:
            del self._priority_configs[key]
            logger.debug(f"[IdleResourceDaemon] Pruned stale priority for {key}")

        return self._priority_configs.copy()

    def _prune_stale_cluster_states(self) -> None:
        """Remove cluster states older than the stale threshold."""
        now = time.time()
        stale_nodes = [
            node_id for node_id, state in self._cluster_idle_states.items()
            if now - state.timestamp > self._state_stale_threshold
        ]
        for node_id in stale_nodes:
            del self._cluster_idle_states[node_id]
            logger.debug(f"[IdleResourceDaemon] Pruned stale idle state for {node_id}")

    async def _broadcast_local_state(self, force: bool = False) -> None:
        """Broadcast this node's idle state to the cluster.

        Args:
            force: If True, broadcast immediately regardless of interval.
        """
        now = time.time()

        # Rate limit broadcasts unless forced
        if not force and now - self._last_broadcast_time < self._broadcast_interval:
            return

        self._last_broadcast_time = now

        try:
            from app.coordination.event_router import DataEventType, get_router

            router = get_router()

            # Get local node's current state
            local_state = self._get_local_idle_state()
            if local_state is None:
                return

            # Broadcast to cluster
            if hasattr(DataEventType, 'IDLE_STATE_BROADCAST'):
                await router.publish(
                    DataEventType.IDLE_STATE_BROADCAST.value,
                    {
                        "node_id": local_state.node_id,
                        "host": local_state.host,
                        "is_idle": local_state.is_idle,
                        "gpu_utilization": local_state.gpu_utilization,
                        "gpu_memory_free_gb": local_state.gpu_memory_free_gb,
                        "gpu_memory_total_gb": local_state.gpu_memory_total_gb,
                        "idle_duration_seconds": local_state.idle_duration_seconds,
                        "recommended_config": local_state.recommended_config,
                        "provider": local_state.provider,
                        "timestamp": local_state.timestamp,
                        "active_jobs": local_state.active_jobs,
                    },
                    source="idle_resource_daemon",
                )

        except ImportError:
            pass  # Event router not available
        except Exception as e:
            logger.debug(f"[IdleResourceDaemon] Failed to broadcast local state: {e}")

    def _get_local_idle_state(self) -> NodeIdleState | None:
        """Get this node's current idle state."""
        # Check if we have local GPU info
        local_node = self._node_states.get(self.node_id)

        if local_node is None:
            # Try to get local GPU info
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.total,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(',')
                    if len(parts) >= 3:
                        gpu_util = float(parts[0].strip())
                        mem_total = float(parts[1].strip()) / 1024.0
                        mem_used = float(parts[2].strip()) / 1024.0

                        local_node = NodeStatus(
                            node_id=self.node_id,
                            host=socket.gethostname(),
                            gpu_utilization=gpu_util,
                            gpu_memory_total_gb=mem_total,
                            gpu_memory_used_gb=mem_used,
                            last_seen=time.time(),
                        )
            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                # nvidia-smi not available or timed out - expected on non-GPU nodes
                logger.debug(f"[IdleResourceDaemon] GPU query unavailable: {e}")
            except (ValueError, IndexError) as e:
                # Failed to parse nvidia-smi output
                logger.debug(f"[IdleResourceDaemon] GPU output parse error: {e}")

        if local_node is None:
            return None

        now = time.time()
        is_idle = local_node.gpu_utilization < self.config.idle_threshold_percent
        idle_duration = now - local_node.idle_since if local_node.idle_since > 0 else 0.0

        return NodeIdleState(
            node_id=self.node_id,
            host=local_node.host,
            is_idle=is_idle,
            gpu_utilization=local_node.gpu_utilization,
            gpu_memory_free_gb=local_node.gpu_memory_total_gb - local_node.gpu_memory_used_gb,
            gpu_memory_total_gb=local_node.gpu_memory_total_gb,
            idle_duration_seconds=idle_duration,
            recommended_config=self._get_recommended_config(local_node) if is_idle else "",
            provider=self._detect_provider(self.node_id),
            timestamp=now,
            active_jobs=local_node.active_jobs,
        )

    def get_cluster_idle_state(self) -> ClusterIdleState:
        """Get aggregated cluster-wide idle state.

        Returns:
            ClusterIdleState with all known node states.
        """
        # Prune stale states first
        self._prune_stale_cluster_states()

        # Include local state
        local_state = self._get_local_idle_state()
        all_states = list(self._cluster_idle_states.values())
        if local_state is not None:
            all_states.append(local_state)

        idle_nodes = [s for s in all_states if s.is_idle]
        total_idle_memory = sum(s.gpu_memory_free_gb for s in idle_nodes)

        return ClusterIdleState(
            total_nodes=len(all_states),
            idle_nodes=len(idle_nodes),
            total_idle_gpu_memory_gb=total_idle_memory,
            nodes=all_states,
            timestamp=time.time(),
        )

    async def stop(self) -> None:
        """Stop the idle resource daemon."""
        if self._coordinator_status == CoordinatorStatus.STOPPED:
            return  # Already stopped

        self._coordinator_status = CoordinatorStatus.STOPPING
        logger.info("Stopping IdleResourceDaemon...")
        self._running = False

        # Stop monitor task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # Unregister coordinator
        try:
            unregister_coordinator("idle_resource")
        except Exception as e:
            logger.debug(f"[IdleResourceDaemon] Failed to unregister coordinator: {e}")

        self._coordinator_status = CoordinatorStatus.STOPPED
        logger.info(
            f"IdleResourceDaemon stopped. Stats: "
            f"{self._stats.successful_spawns}/{self._stats.total_spawns} spawns, "
            f"{self._stats.games_spawned} games"
        )

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_and_spawn()
                await asyncio.sleep(self.config.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._errors_count += 1
                self._last_error = str(e)
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(60)  # Back off on error

    async def _check_and_spawn(self) -> None:
        """Check for idle nodes and spawn selfplay jobs.

        December 2025 - Phase 2B.2: Dynamically scales spawning based on
        idle node count instead of fixed max of 4.

        December 2025 - Phase 2C.4: Integrates with SelfplayScheduler for
        priority-based config selection.
        """
        try:
            # December 2025: Broadcast local idle state to cluster
            await self._broadcast_local_state()

            # Update SelfplayScheduler priorities before spawning
            await self._update_scheduler_priorities()

            # Phase 21.5: Refresh backpressure signal for accurate spawn decisions
            await self._refresh_backpressure_signal()

            # December 2025: Enforce process limits before spawning
            # This actively kills excess processes on nodes with runaway counts
            await self._enforce_process_limits()

            # Get cluster status
            nodes = await self._get_cluster_nodes()

            if not nodes:
                logger.debug("No cluster nodes found")
                return

            # Get queue depth for scaling decisions
            queue_depth = await self._get_queue_depth()

            # Get dynamic max spawns based on current idle capacity
            max_spawns = self._get_dynamic_max_spawns()

            # Collect nodes that need spawning
            spawn_candidates = [
                node for node in nodes
                if self._should_spawn(node, queue_depth)
            ]

            if not spawn_candidates:
                return

            # Emit IDLE_RESOURCE_DETECTED events for each candidate (Dec 2025 Phase 2)
            # This allows SelfplayOrchestrator and other components to react
            await self._emit_idle_resource_events(spawn_candidates)

            # Log scaling decision
            logger.info(
                f"[IdleResourceDaemon] Spawn check: {len(spawn_candidates)} candidates, "
                f"max_spawns={max_spawns} (dynamic), queue_depth={queue_depth}"
            )

            # Spawn up to max_spawns jobs concurrently
            spawn_tasks = []
            for node in spawn_candidates[:max_spawns]:
                spawn_tasks.append(self._spawn_selfplay(node))

            if spawn_tasks:
                results = await asyncio.gather(*spawn_tasks, return_exceptions=True)
                successful = sum(1 for r in results if r is True)
                logger.info(
                    f"[IdleResourceDaemon] Spawned {successful}/{len(spawn_tasks)} jobs"
                )

        except Exception as e:
            logger.warning(f"Check and spawn error: {e}")

    async def _update_scheduler_priorities(self) -> None:
        """Update SelfplayScheduler priorities for informed config selection.

        December 2025 - Phase 2C.4: Ensure scheduler has fresh priorities
        based on curriculum weights, Elo velocities, and feedback signals.
        """
        try:
            from app.coordination.selfplay_scheduler import get_selfplay_scheduler

            scheduler = get_selfplay_scheduler()
            await scheduler._update_priorities()

        except ImportError:
            pass  # Scheduler not available
        except Exception as e:
            logger.debug(f"[IdleResourceDaemon] Failed to update scheduler priorities: {e}")

    async def _refresh_backpressure_signal(self) -> None:
        """Refresh unified backpressure signal for accurate spawn decisions.

        Phase 21.5 - December 2025: Updates the cached BackpressureSignal so that
        _should_spawn() can use fresh pressure metrics. This runs asynchronously
        to collect metrics from queue monitor, daemon manager, P2P, and GPU memory.
        """
        if not HAS_BACKPRESSURE or not get_backpressure_monitor:
            return

        try:
            monitor = get_backpressure_monitor()
            signal = await monitor.get_signal(force_refresh=True)

            # Log pressure summary periodically
            if signal.overall_pressure > 0.3:
                logger.info(
                    f"[IdleResourceDaemon] Backpressure: overall={signal.overall_pressure:.2f} "
                    f"(queue={signal.queue_pressure:.2f}, training={signal.training_pressure:.2f}, "
                    f"disk={signal.disk_pressure:.2f}), multiplier={signal.spawn_rate_multiplier:.2f}"
                )
        except Exception as e:
            logger.debug(f"[IdleResourceDaemon] Failed to refresh backpressure: {e}")

    async def _get_cluster_nodes(self) -> list[NodeStatus]:
        """Get status of all cluster nodes.

        Dec 2025: Added SSH fallback for nodes not in P2P. This ensures we can
        discover and spawn on nodes even if their P2P daemon isn't running.
        """
        nodes: list[NodeStatus] = []
        seen_hosts: set[str] = set()

        # Phase 1: Try P2P first (preferred for nodes with P2P running)
        nodes.extend(await self._get_p2p_nodes(seen_hosts))

        # Phase 2: SSH fallback for configured nodes not in P2P
        if HAS_SSH_FALLBACK:
            ssh_nodes = await self._get_ssh_fallback_nodes(seen_hosts)
            nodes.extend(ssh_nodes)

        return nodes

    async def _get_p2p_nodes(self, seen_hosts: set[str]) -> list[NodeStatus]:
        """Get nodes from P2P orchestrator."""
        nodes: list[NodeStatus] = []

        try:
            from app.coordination.p2p_integration import get_p2p_orchestrator

            p2p = get_p2p_orchestrator()
            if p2p is None:
                return nodes

            # Get cluster status
            status = await p2p.get_status()
            if not status:
                return nodes

            # Parse alive peers
            alive_peers = status.get("alive_peers", [])
            if isinstance(alive_peers, int):
                # Just a count, not detailed info
                return nodes

            for peer_info in alive_peers:
                if isinstance(peer_info, dict):
                    node = NodeStatus(
                        node_id=peer_info.get("node_id", ""),
                        host=peer_info.get("host", ""),
                        gpu_utilization=peer_info.get("gpu_utilization", 0.0),
                        gpu_memory_total_gb=peer_info.get("gpu_memory_total", 0.0),
                        gpu_memory_used_gb=peer_info.get("gpu_memory_used", 0.0),
                        last_seen=peer_info.get("last_seen", time.time()),
                        active_jobs=peer_info.get("active_jobs", 0),
                        provider=peer_info.get("provider", "unknown"),
                    )
                    nodes.append(node)
                    self._update_node_state(node)
                    # Track by both node_id and host to avoid duplicates
                    seen_hosts.add(node.node_id)
                    if node.host:
                        seen_hosts.add(node.host)

        except ImportError:
            logger.debug("P2P orchestrator not available")
        except Exception as e:
            logger.debug(f"Failed to get cluster nodes from P2P: {e}")

        return nodes

    async def _get_ssh_fallback_nodes(self, exclude: set[str]) -> list[NodeStatus]:
        """Get nodes via SSH for hosts not discovered via P2P.

        Dec 2025: Discovers nodes from distributed_hosts.yaml that aren't
        in the P2P cluster, checks their GPU status via SSH.

        Args:
            exclude: Set of node IDs/hosts to skip (already discovered via P2P).

        Returns:
            List of NodeStatus for SSH-discovered nodes with GPUs.
        """
        nodes: list[NodeStatus] = []

        if not HAS_SSH_FALLBACK or get_configured_hosts is None:
            return nodes

        try:
            configured_hosts = get_configured_hosts()
        except Exception as e:
            logger.debug(f"[IdleResourceDaemon] Failed to load configured hosts: {e}")
            return nodes

        # Filter to active hosts with GPUs that aren't already discovered
        candidates = [
            (name, host) for name, host in configured_hosts.items()
            if host.is_active
            and host.gpu  # Has GPU configured
            and name not in exclude
            and (host.best_ip is None or host.best_ip not in exclude)
        ]

        if not candidates:
            return nodes

        logger.debug(
            f"[IdleResourceDaemon] SSH fallback: checking {len(candidates)} nodes "
            f"not in P2P: {[n for n, _ in candidates[:5]]}..."
        )

        # Check nodes concurrently (limit concurrency to avoid overwhelming)
        semaphore = asyncio.Semaphore(5)

        async def check_node(name: str, host: ClusterNode) -> NodeStatus | None:
            async with semaphore:
                return await self._check_node_via_ssh(name, host)

        tasks = [check_node(name, host) for name, host in candidates]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, NodeStatus):
                nodes.append(result)
                self._update_node_state(result)
            elif isinstance(result, Exception):
                logger.debug(f"[IdleResourceDaemon] SSH check failed: {result}")

        if nodes:
            logger.info(
                f"[IdleResourceDaemon] SSH fallback discovered {len(nodes)} nodes: "
                f"{[n.node_id for n in nodes]}"
            )

        return nodes

    async def _check_node_via_ssh(
        self, name: str, host: ClusterNode
    ) -> NodeStatus | None:
        """Check a single node's GPU status via SSH.

        Args:
            name: Node name from config.
            host: ClusterNode with SSH connection info.

        Returns:
            NodeStatus if node is reachable and has GPU, None otherwise.
        """
        if SSHExecutor is None or host.best_ip is None:
            return None

        try:
            executor = SSHExecutor(
                host=host.best_ip,
                user=host.ssh_user,
                port=host.ssh_port,
                key_path=host.ssh_key,
                connect_timeout=5,
                max_retries=1,
            )

            # Quick GPU check via nvidia-smi
            result = await executor.run(
                "nvidia-smi --query-gpu=utilization.gpu,memory.total,memory.used "
                "--format=csv,noheader,nounits 2>/dev/null || echo 'no-gpu'",
                timeout=10,
            )

            if not result.success or "no-gpu" in result.stdout:
                return None

            # Parse nvidia-smi output: "util%, mem_total, mem_used"
            # Example: "5, 24576, 1234"
            lines = result.stdout.strip().split('\n')
            if not lines or not lines[0]:
                return None

            parts = lines[0].split(',')
            if len(parts) < 3:
                return None

            try:
                gpu_util = float(parts[0].strip())
                mem_total_mb = float(parts[1].strip())
                mem_used_mb = float(parts[2].strip())
            except ValueError:
                return None

            return NodeStatus(
                node_id=name,
                host=host.best_ip,
                gpu_utilization=gpu_util,
                gpu_memory_total_gb=mem_total_mb / 1024.0,
                gpu_memory_used_gb=mem_used_mb / 1024.0,
                last_seen=time.time(),
                active_jobs=0,  # Unknown via SSH
                provider=self._detect_provider(name),
            )

        except Exception as e:
            logger.debug(f"[IdleResourceDaemon] SSH check failed for {name}: {e}")
            return None

    def _detect_provider(self, node_name: str) -> str:
        """Detect provider from node name.

        Uses the consolidated get_host_provider() from cluster_config.
        """
        return get_host_provider(node_name)

    def _update_node_state(self, node: NodeStatus) -> None:
        """Update tracked node state for idle duration tracking."""
        now = time.time()
        existing = self._node_states.get(node.node_id)

        if existing:
            # Check if node transitioned to idle
            if node.gpu_utilization < self.config.idle_threshold_percent:
                if existing.gpu_utilization >= self.config.idle_threshold_percent:
                    # Just became idle
                    node.idle_since = now
                else:
                    # Still idle, preserve idle_since
                    node.idle_since = existing.idle_since
            else:
                # Not idle
                node.idle_since = 0.0
        else:
            # New node
            if node.gpu_utilization < self.config.idle_threshold_percent:
                node.idle_since = now

        self._node_states[node.node_id] = node

    async def _emit_idle_resource_events(self, idle_nodes: list[NodeStatus]) -> None:
        """Emit IDLE_RESOURCE_DETECTED events for idle nodes.

        December 2025 - Phase 2: Wire orphaned event handler.
        This enables SelfplayOrchestrator and other components to react
        to idle resources for better cluster utilization.

        Args:
            idle_nodes: List of nodes with idle GPU resources.
        """
        try:
            from app.distributed.data_events import emit_idle_resource_detected
        except ImportError:
            logger.debug("[IdleResourceDaemon] data_events not available for emit")
            return

        now = time.time()
        for node in idle_nodes:
            try:
                # Calculate idle duration
                idle_duration = now - node.idle_since if node.idle_since > 0 else 0.0

                # Determine recommended config based on GPU memory
                recommended_config = self._get_recommended_config(node)

                await emit_idle_resource_detected(
                    node_id=node.node_id,
                    host=node.host,
                    gpu_utilization=node.gpu_utilization,
                    gpu_memory_gb=node.gpu_memory_total_gb - node.gpu_memory_used_gb,
                    idle_duration_seconds=idle_duration,
                    recommended_config=recommended_config,
                    source="idle_resource_daemon",
                )
            except Exception as e:
                logger.debug(f"[IdleResourceDaemon] Failed to emit event for {node.node_id}: {e}")

    def _get_recommended_config(self, node: NodeStatus) -> str:
        """Get recommended board config based on GPU memory.

        Args:
            node: Node status with GPU memory info.

        Returns:
            Recommended config key (e.g., 'hex8_2p') or empty string.
        """
        available_gb = node.gpu_memory_total_gb - node.gpu_memory_used_gb

        # Match GPU memory to largest board it can handle
        for config, required_gb in sorted(
            self.config.gpu_memory_thresholds.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            if available_gb >= required_gb + self.config.min_free_gpu_memory_buffer_gb:
                return config

        return "hex8_2p"  # Default to smallest config

    async def _get_queue_depth(self) -> int:
        """Get current queue depth for scaling decisions."""
        try:
            from app.coordination.job_scheduler import get_job_scheduler

            scheduler = get_job_scheduler()
            if scheduler:
                return scheduler.get_queue_depth()
        except ImportError:
            pass  # Expected if job_scheduler not available
        except Exception as e:
            logger.debug(f"[IdleResourceDaemon] Failed to get queue depth: {e}")

        return 0  # Default to no queue

    def _get_pending_training_hours(self) -> float:
        """Get estimated hours of unprocessed training data.

        December 2025: Used for training backlog backpressure. If training
        can't keep up with selfplay, we should pause data generation.

        Returns:
            Estimated hours of training data pending processing, or 0.0 if unknown.
        """
        try:
            from app.distributed.data_catalog import get_data_catalog

            catalog = get_data_catalog()
            # Get total samples waiting for training
            pending_samples = catalog.get_pending_sample_count()

            # Estimate processing rate: ~50k samples/hour on typical GPU
            samples_per_hour = 50000
            pending_hours = pending_samples / samples_per_hour

            return pending_hours

        except ImportError:
            pass  # DataCatalog not available
        except AttributeError:
            pass  # get_pending_sample_count not implemented
        except Exception as e:
            logger.debug(f"[IdleResourceDaemon] Failed to get pending training hours: {e}")

        return 0.0  # Default to no backlog

    async def _enforce_process_limits(self) -> None:
        """Kill excess selfplay processes on nodes exceeding limits.

        December 2025: Actively kills runaway selfplay processes to prevent
        resource exhaustion. Uses SSH fallback when P2P job tracking isn't accurate.

        Called before spawn decisions to maintain healthy process counts.
        """
        if not HAS_SSH_FALLBACK or SSHExecutor is None or get_configured_hosts is None:
            return

        max_per_node = self.config.max_selfplay_processes_per_node

        try:
            hosts = get_configured_hosts()
        except Exception as e:
            logger.debug(f"[IdleResourceDaemon] Failed to get cluster hosts: {e}")
            return

        for name, host in hosts.items():
            if host.best_ip is None:
                continue

            try:
                executor = SSHExecutor(
                    host=host.best_ip,
                    user=host.ssh_user,
                    port=host.ssh_port,
                    key_path=host.ssh_key,
                    connect_timeout=5,
                    max_retries=1,
                )

                # Count selfplay/gpu_parallel processes
                count_result = await executor.run(
                    "pgrep -c -f 'selfplay|gpu_parallel' 2>/dev/null || echo 0",
                    timeout=10,
                )

                if not count_result.success:
                    continue

                try:
                    process_count = int(count_result.stdout.strip())
                except ValueError:
                    continue

                if process_count <= max_per_node:
                    continue

                excess = process_count - max_per_node
                logger.warning(
                    f"[IdleResourceDaemon] Node {name} has {process_count} processes "
                    f"(max {max_per_node}), killing {excess} oldest"
                )

                # Kill oldest processes first (sorted by elapsed time)
                # ps -eo pid,etime,cmd sorts by start time, oldest first
                kill_cmd = (
                    f"ps -eo pid,etime,cmd --sort=etime 2>/dev/null | "
                    f"grep -E 'selfplay|gpu_parallel' | "
                    f"grep -v grep | "
                    f"head -n {excess} | "
                    f"awk '{{print $1}}' | "
                    f"xargs -r kill -9 2>/dev/null || true"
                )

                kill_result = await executor.run(kill_cmd, timeout=30)

                if kill_result.success:
                    logger.info(
                        f"[IdleResourceDaemon] Killed {excess} excess processes on {name}"
                    )
                    self._stats.failed_spawns += 0  # Track cleanup (no stat for this yet)
                else:
                    logger.warning(
                        f"[IdleResourceDaemon] Failed to kill processes on {name}: "
                        f"{kill_result.stderr}"
                    )

            except Exception as e:
                logger.debug(f"[IdleResourceDaemon] Process check failed for {name}: {e}")

    def _should_spawn(self, node: NodeStatus, queue_depth: int) -> bool:
        """Decide whether to spawn selfplay on a node."""
        now = time.time()

        # =======================================================================
        # P2P Node Health Check (December 2025 - Critical Gap Fix)
        # =======================================================================
        # Skip nodes that are marked unhealthy by P2P cluster health events
        unhealthy_nodes = getattr(self, "_unhealthy_nodes", set())
        if node.node_id in unhealthy_nodes:
            logger.debug(
                f"[IdleResourceDaemon] Skipping {node.node_id}: marked unhealthy by P2P"
            )
            return False

        # Also check by host if node_id doesn't match
        if node.host and node.host in unhealthy_nodes:
            logger.debug(
                f"[IdleResourceDaemon] Skipping {node.node_id} (host {node.host}): "
                f"marked unhealthy by P2P"
            )
            return False

        # =======================================================================
        # Stall Detection Check (Phase 21.5 - December 2025)
        # =======================================================================
        # Skip nodes that are penalized due to previous job stalls
        if HAS_STALL_DETECTION and get_stall_detector:
            try:
                detector = get_stall_detector()
                if detector.is_node_penalized(node.node_id):
                    remaining = detector.get_penalty_remaining(node.node_id)
                    logger.debug(
                        f"[IdleResourceDaemon] Skipping {node.node_id}: "
                        f"stall penalty active ({remaining:.0f}s remaining)"
                    )
                    return False
                # Also check if node is unhealthy due to too many stalls
                if detector.is_node_unhealthy(node.node_id):
                    logger.warning(
                        f"[IdleResourceDaemon] Skipping {node.node_id}: "
                        f"marked unhealthy due to repeated stalls"
                    )
                    return False
            except Exception as e:
                logger.debug(f"[IdleResourceDaemon] Stall detector check failed: {e}")

        # =======================================================================
        # Unified Backpressure Check (Phase 21.5 - December 2025)
        # =======================================================================
        # Use unified backpressure signal for comprehensive pressure monitoring
        if HAS_BACKPRESSURE and get_backpressure_monitor:
            try:
                monitor = get_backpressure_monitor()
                # Use cached signal (non-blocking) since this is a sync method
                signal = monitor.get_cached_signal()
                if signal is not None:
                    if signal.should_pause:
                        logger.info(
                            f"[IdleResourceDaemon] Unified backpressure pause: "
                            f"pressure={signal.overall_pressure:.2f}, "
                            f"skipping spawn on {node.node_id}"
                        )
                        return False
                    elif signal.spawn_rate_multiplier < 0.5:
                        # Probabilistically skip based on spawn rate multiplier
                        import random
                        if random.random() > signal.spawn_rate_multiplier:
                            logger.debug(
                                f"[IdleResourceDaemon] Backpressure throttle: "
                                f"multiplier={signal.spawn_rate_multiplier:.2f}, "
                                f"skipping {node.node_id}"
                            )
                            return False
            except Exception as e:
                logger.debug(f"[IdleResourceDaemon] Backpressure check failed: {e}")

        # =======================================================================
        # Queue Backpressure Fallback (December 2025)
        # =======================================================================
        # Simple queue depth check as fallback if unified backpressure unavailable
        if queue_depth > self.config.max_queue_depth:
            logger.info(
                f"[IdleResourceDaemon] Queue backpressure: depth {queue_depth} > "
                f"max {self.config.max_queue_depth}, skipping spawn on {node.node_id}"
            )
            return False

        # Check training data backlog (prevent generating data faster than training)
        pending_hours = self._get_pending_training_hours()
        if pending_hours > self.config.max_pending_training_hours:
            logger.info(
                f"[IdleResourceDaemon] Training backlog: {pending_hours:.1f}h > "
                f"max {self.config.max_pending_training_hours}h, skipping spawn"
            )
            return False

        # Check if node is in backoff from previous failures (December 2025)
        if self._is_node_in_backoff(node.node_id):
            remaining = self._get_node_backoff_remaining(node.node_id)
            logger.debug(
                f"[IdleResourceDaemon] Skipping {node.node_id}: "
                f"in backoff for {remaining:.0f}s more"
            )
            return False

        # Check if node is idle long enough
        if node.idle_since <= 0:
            return False

        idle_duration = now - node.idle_since

        # Adjust threshold based on queue depth
        if queue_depth > self.config.high_queue_depth:
            # More aggressive spawning when queue is deep
            threshold = self.config.idle_threshold_percent * 3  # 30%
            required_idle_time = self.config.idle_duration_seconds / 3  # 5 minutes
        elif queue_depth > self.config.medium_queue_depth:
            threshold = self.config.idle_threshold_percent * 2  # 20%
            required_idle_time = self.config.idle_duration_seconds / 2  # 7.5 minutes
        else:
            threshold = self.config.idle_threshold_percent  # 10%
            required_idle_time = self.config.idle_duration_seconds  # 15 minutes

        # Check conditions
        if node.gpu_utilization > threshold:
            return False

        if idle_duration < required_idle_time:
            return False

        # Dec 26 2025: Enforce process limit - don't spawn if node at capacity
        # Note: active_jobs may be 0 for nodes where P2P tracking isn't perfect,
        # but this still protects against spawning on nodes that report high counts
        if node.active_jobs >= self.config.max_selfplay_processes_per_node:
            logger.debug(
                f"[IdleResourceDaemon] Node {node.node_id} at process limit "
                f"({node.active_jobs}/{self.config.max_selfplay_processes_per_node})"
            )
            return False

        return True

    def _select_config_for_gpu(self, gpu_memory_gb: float) -> str:
        """Select appropriate board config for GPU memory.

        December 2025 - Phase 2C.4: Now uses SelfplayScheduler priorities
        to select the highest-priority config that fits the GPU.
        """
        # Get configs that fit this GPU's memory
        valid_configs = {
            config_key for config_key, required_memory
            in self.config.gpu_memory_thresholds.items()
            if gpu_memory_gb >= required_memory
        }

        if not valid_configs:
            return "hex8_2p"  # Smallest config as fallback

        # Try to get priority from SelfplayScheduler
        try:
            from app.coordination.selfplay_scheduler import get_selfplay_scheduler

            scheduler = get_selfplay_scheduler()
            # Get priority configs using public API (Dec 2025: replaced private access)
            # Uses cached priorities, safe for sync context
            sorted_priorities = scheduler.get_priority_configs_sync(
                filter_configs=valid_configs
            )

            # Return highest priority config that fits this GPU
            if sorted_priorities:
                config_key, priority_score = sorted_priorities[0]
                logger.debug(
                    f"[IdleResourceDaemon] Selected {config_key} "
                    f"(priority={priority_score:.2f}) for {gpu_memory_gb:.0f}GB GPU"
                )
                return config_key

        except ImportError:
            logger.debug("[IdleResourceDaemon] SelfplayScheduler not available, using memory-based selection")
        except Exception as e:
            logger.debug(f"[IdleResourceDaemon] SelfplayScheduler query failed: {e}")

        # Fallback: Sort by memory requirement descending, pick largest that fits
        sorted_configs = sorted(
            self.config.gpu_memory_thresholds.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for config_key, required_memory in sorted_configs:
            if gpu_memory_gb >= required_memory:
                return config_key

        # Default to smallest config
        return "hex8_2p"

    async def _spawn_selfplay(self, node: NodeStatus) -> bool:
        """Spawn a selfplay job on the given node."""
        async with self._semaphore:
            self._stats.total_spawns += 1
            start_time = time.time()
            config_key = "unknown"
            games = self.config.default_games_per_spawn

            try:
                # Phase 4: Check circuit breaker before spawning
                # Prevents cascading failures when cluster operations are failing
                if HAS_CIRCUIT_BREAKER and get_operation_breaker:
                    breaker = get_operation_breaker()
                    if not breaker.can_execute("selfplay_spawn"):
                        logger.debug(
                            f"[IdleResourceDaemon] Circuit open for selfplay_spawn, "
                            f"skipping {node.node_id}"
                        )
                        return False

                config_key = self._select_config_for_gpu(node.gpu_memory_total_gb)

                # P11-CRITICAL-2: Check free GPU memory before spawning
                # This prevents OOM errors by ensuring adequate VRAM headroom
                required_memory = self.config.gpu_memory_thresholds.get(config_key, 8)
                free_memory = node.gpu_memory_total_gb - node.gpu_memory_used_gb
                min_required = required_memory + self.config.min_free_gpu_memory_buffer_gb

                if free_memory < min_required:
                    logger.info(
                        f"[IdleResourceDaemon] Skipping {node.node_id}: insufficient VRAM. "
                        f"Free={free_memory:.1f}GB, required={min_required:.1f}GB "
                        f"(config={config_key} needs {required_memory}GB + "
                        f"{self.config.min_free_gpu_memory_buffer_gb}GB buffer)"
                    )
                    self._stats.failed_spawns += 1
                    return False

                # Get multiplier from FeedbackAccelerator
                try:
                    from app.training.feedback_accelerator import get_selfplay_multiplier
                    multiplier = get_selfplay_multiplier(config_key)
                    games = int(games * multiplier)
                except ImportError:
                    pass

                logger.info(
                    f"[IdleResourceDaemon] Spawning selfplay on {node.node_id}: "
                    f"config={config_key}, games={games}, "
                    f"gpu_memory={node.gpu_memory_total_gb:.0f}GB, "
                    f"free={free_memory:.1f}GB"
                )

                # Phase 21.2: Also schedule via PriorityJobScheduler for tracking
                if HAS_JOB_SCHEDULER and get_scheduler:
                    try:
                        scheduler = get_scheduler()
                        if scheduler:
                            # Parse config key for job config
                            parts = config_key.rsplit("_", 1)
                            board_type = parts[0] if parts else config_key
                            num_players = int(parts[1].replace("p", "")) if len(parts) > 1 else 2

                            job = ScheduledJob(
                                job_type="selfplay",
                                priority=JobPriority.NORMAL,  # Idle-spawned jobs are normal priority
                                config={
                                    "board_type": board_type,
                                    "num_players": num_players,
                                    "games": games,
                                    "config_key": config_key,
                                    "source": "idle_resource_daemon",
                                },
                                host_preference=node.node_id,
                                requires_gpu=True,
                                estimated_duration_seconds=games * 10.0,  # ~10s per game estimate
                            )
                            scheduler.schedule(job)
                            logger.debug(
                                f"[IdleResourceDaemon] Scheduled job via JobScheduler: {config_key}"
                            )
                    except Exception as e:
                        logger.debug(f"[IdleResourceDaemon] JobScheduler integration failed: {e}")

                # Phase 21.5: Register job with stall detector for progress tracking
                job_id = f"selfplay_{node.node_id}_{config_key}_{int(start_time)}"
                if HAS_STALL_DETECTION and get_stall_detector:
                    try:
                        detector = get_stall_detector()
                        detector.register_job(job_id, node.node_id)
                    except Exception as e:
                        logger.debug(f"[IdleResourceDaemon] Stall detector registration failed: {e}")

                # Spawn via P2P job distribution
                success = await self._distribute_job(node, config_key, games)
                duration = time.time() - start_time

                if success:
                    self._stats.successful_spawns += 1
                    self._stats.games_spawned += games
                    self._stats.last_spawn_time = time.time()

                    # Record successful attempt (December 2025)
                    self._record_spawn_attempt(
                        node_id=node.node_id,
                        config_key=config_key,
                        games=games,
                        success=True,
                        duration=duration,
                    )

                    # Phase 4: Record circuit breaker success
                    if HAS_CIRCUIT_BREAKER and get_operation_breaker:
                        get_operation_breaker().record_success("selfplay_spawn")

                    # Phase 21.5: Mark job complete for stall detector (reduces node penalty)
                    if HAS_STALL_DETECTION and get_stall_detector:
                        try:
                            detector = get_stall_detector()
                            detector.complete_job(job_id, success=True)
                        except Exception as e:
                            logger.debug(f"[IdleResourceDaemon] Stall detector completion failed: {e}")

                    # Emit event
                    self._emit_spawn_event(node, config_key, games)

                    # Reset idle tracking for this node
                    if node.node_id in self._node_states:
                        self._node_states[node.node_id].idle_since = 0.0

                    return True
                else:
                    self._stats.failed_spawns += 1
                    # Record failed attempt (December 2025)
                    self._record_spawn_attempt(
                        node_id=node.node_id,
                        config_key=config_key,
                        games=games,
                        success=False,
                        error="P2P job distribution returned failure",
                        duration=duration,
                    )
                    # Phase 4: Record circuit breaker failure
                    if HAS_CIRCUIT_BREAKER and get_operation_breaker:
                        get_operation_breaker().record_failure("selfplay_spawn")

                    # Phase 21.5: Report stall to detector (applies node penalty)
                    if HAS_STALL_DETECTION and get_stall_detector:
                        try:
                            detector = get_stall_detector()
                            detector.report_stall(job_id, node.node_id, duration)
                        except Exception as e:
                            logger.debug(f"[IdleResourceDaemon] Stall detector report failed: {e}")

                    return False

            except Exception as e:
                self._stats.failed_spawns += 1
                self._stats.last_error = str(e)
                duration = time.time() - start_time
                # Record failed attempt with exception details (December 2025)
                self._record_spawn_attempt(
                    node_id=node.node_id,
                    config_key=config_key,
                    games=games,
                    success=False,
                    error=str(e),
                    duration=duration,
                )
                # Phase 4: Record circuit breaker failure on exception
                if HAS_CIRCUIT_BREAKER and get_operation_breaker:
                    get_operation_breaker().record_failure("selfplay_spawn")

                # Phase 21.5: Report stall on exception (applies node penalty)
                if HAS_STALL_DETECTION and get_stall_detector:
                    try:
                        detector = get_stall_detector()
                        detector.report_stall(job_id, node.node_id, duration)
                    except Exception as ex:
                        logger.debug(f"[IdleResourceDaemon] Stall detector report failed: {ex}")

                logger.error(f"Failed to spawn selfplay on {node.node_id}: {e}")
                return False

    async def _distribute_job(
        self,
        node: NodeStatus,
        config_key: str,
        games: int,
    ) -> bool:
        """Distribute a selfplay job to a node.

        Dec 2025: Added SSH fallback when P2P is unavailable. This allows
        spawning jobs on nodes even if their P2P daemon isn't running.
        """
        # Parse config key first (needed for both methods)
        parts = config_key.rsplit("_", 1)
        if len(parts) != 2:
            logger.warning(f"Invalid config key: {config_key}")
            return False

        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

        # Phase 1: Try P2P first (preferred)
        p2p_success = await self._distribute_job_via_p2p(
            node, board_type, num_players, games
        )
        if p2p_success:
            return True

        # Phase 2: SSH fallback for nodes discovered via SSH
        if HAS_SSH_FALLBACK:
            return await self._distribute_job_via_ssh(
                node, board_type, num_players, games
            )

        return False

    async def _distribute_job_via_p2p(
        self,
        node: NodeStatus,
        board_type: str,
        num_players: int,
        games: int,
    ) -> bool:
        """Distribute job via P2P orchestrator."""
        try:
            from app.coordination.p2p_integration import get_p2p_orchestrator

            p2p = get_p2p_orchestrator()
            if p2p is None:
                return False

            # Submit job via P2P
            # NOTE: Must use "engine_mode" (not "engine") to match P2P orchestrator API
            job_spec = {
                "type": "selfplay",
                "board_type": board_type,
                "num_players": num_players,
                "num_games": games,
                "engine_mode": "gumbel-mcts",  # GPU-accelerated Gumbel MCTS
                "target_node": node.node_id,
            }

            result = await p2p.submit_job(job_spec)
            return result.get("success", False)

        except ImportError:
            logger.debug("P2P orchestrator not available for job distribution")
            return False
        except Exception as e:
            logger.debug(f"P2P job distribution failed: {e}")
            return False

    async def _distribute_job_via_ssh(
        self,
        node: NodeStatus,
        board_type: str,
        num_players: int,
        games: int,
    ) -> bool:
        """Distribute job via SSH when P2P is unavailable.

        Dec 2025: SSH-based job spawn for nodes not in P2P cluster.
        Spawns selfplay as a background process on the remote node.

        Args:
            node: Target node info.
            board_type: Board type (e.g., 'hex8', 'square8').
            num_players: Number of players.
            games: Number of games to run.

        Returns:
            True if job was spawned successfully.
        """
        if not HAS_SSH_FALLBACK or SSHExecutor is None or get_configured_hosts is None:
            return False

        try:
            # Get SSH config for this node
            configured_hosts = get_configured_hosts()
            host_config = configured_hosts.get(node.node_id)

            if host_config is None:
                # Try to find by IP
                for name, cfg in configured_hosts.items():
                    if cfg.best_ip == node.host:
                        host_config = cfg
                        break

            if host_config is None or host_config.best_ip is None:
                logger.debug(
                    f"[IdleResourceDaemon] No SSH config for {node.node_id}, "
                    "cannot distribute via SSH"
                )
                return False

            executor = SSHExecutor(
                host=host_config.best_ip,
                user=host_config.ssh_user,
                port=host_config.ssh_port,
                key_path=host_config.ssh_key,
                connect_timeout=10,
                max_retries=2,
            )

            # Build selfplay command
            # Use nohup to detach from SSH session
            ringrift_path = host_config.ringrift_path or "~/ringrift/ai-service"

            # Expand ~ in path
            if ringrift_path.startswith("~"):
                ringrift_path = ringrift_path.replace("~", "$HOME", 1)

            selfplay_cmd = (
                f"cd {ringrift_path} && "
                f"PYTHONPATH=. nohup python scripts/selfplay.py "
                f"--board {board_type} --num-players {num_players} "
                f"--num-games {games} --engine gumbel-mcts "
                f"> /tmp/selfplay_{board_type}_{num_players}p_{int(time.time())}.log 2>&1 &"
            )

            logger.info(
                f"[IdleResourceDaemon] SSH spawn on {node.node_id}: "
                f"{board_type}_{num_players}p x{games} games"
            )

            result = await executor.run(selfplay_cmd, timeout=30)

            if result.success:
                logger.info(
                    f"[IdleResourceDaemon] SSH spawn successful on {node.node_id}"
                )
                return True
            else:
                logger.warning(
                    f"[IdleResourceDaemon] SSH spawn failed on {node.node_id}: "
                    f"{result.stderr}"
                )
                return False

        except Exception as e:
            logger.warning(f"[IdleResourceDaemon] SSH job distribution failed: {e}")
            return False

    def _emit_spawn_event(
        self,
        node: NodeStatus,
        config_key: str,
        games: int,
    ) -> None:
        """Emit event for selfplay spawn."""
        try:
            from app.coordination.event_router import get_router, DataEventType

            router = get_router()
            # Phase 22.2 fix: Use publish_sync instead of emit (which doesn't exist)
            router.publish_sync(
                DataEventType.P2P_SELFPLAY_SCALED.value
                if hasattr(DataEventType, 'P2P_SELFPLAY_SCALED')
                else "p2p_selfplay_scaled",
                {
                    "node_id": node.node_id,
                    "config": config_key,
                    "games": games,
                    "reason": "idle_resource",
                    "gpu_memory_gb": node.gpu_memory_total_gb,
                    "timestamp": time.time(),
                },
                source="idle_resource_daemon",
            )
        except Exception as e:
            logger.debug(f"Could not publish spawn event: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get daemon statistics."""
        # Calculate success rate
        success_rate = 0.0
        if self._stats.total_spawns > 0:
            success_rate = self._stats.successful_spawns / self._stats.total_spawns

        # Count nodes in backoff
        nodes_in_backoff = sum(
            1 for n in self._node_spawn_history
            if self._is_node_in_backoff(n)
        )

        # Get cluster-wide idle state (December 2025)
        cluster_state = self.get_cluster_idle_state()

        return {
            "running": self._running,
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
            "total_spawns": self._stats.total_spawns,
            "successful_spawns": self._stats.successful_spawns,
            "failed_spawns": self._stats.failed_spawns,
            "success_rate": round(success_rate, 3),
            "games_spawned": self._stats.games_spawned,
            "last_spawn_time": self._stats.last_spawn_time,
            "last_error": self._stats.last_error,
            "tracked_nodes": len(self._node_states),
            "nodes_in_backoff": nodes_in_backoff,
            "errors_count": self._errors_count,
            # Cluster-wide idle state (December 2025)
            "cluster": {
                "total_nodes": cluster_state.total_nodes,
                "idle_nodes": cluster_state.idle_nodes,
                "idle_ratio": round(cluster_state.idle_ratio, 3),
                "total_idle_gpu_memory_gb": round(cluster_state.total_idle_gpu_memory_gb, 1),
                "has_idle_capacity": cluster_state.has_idle_capacity,
            },
        }

    # CoordinatorProtocol methods
    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        return {
            "healthy": self._running,
            "status": self._coordinator_status.value,
            "stats": self.get_stats(),
        }

    def get_status(self) -> CoordinatorStatus:
        """Get coordinator status."""
        return self._coordinator_status
