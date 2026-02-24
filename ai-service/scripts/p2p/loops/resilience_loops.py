"""Resilience Loops for P2P Orchestrator.

December 2025: Background loops for self-healing and predictive monitoring.

Loops:
- SelfHealingLoop: Recovers stuck jobs and cleans up stale processes
- PredictiveMonitoringLoop: Proactive monitoring and alerting

Usage:
    from scripts.p2p.loops import SelfHealingLoop, PredictiveMonitoringLoop

    healing_loop = SelfHealingLoop(
        is_leader=lambda: orchestrator.role == NodeRole.LEADER,
        get_health_manager=lambda: get_health_manager(),
        get_work_queue=lambda: get_work_queue(),
        cleanup_stale_processes=lambda: orchestrator._cleanup_stale_processes(),
    )
    await healing_loop.run_forever()
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Protocol

from app.core.async_context import safe_create_task

from .base import BaseLoop
from .loop_constants import LoopIntervals

logger = logging.getLogger(__name__)

# Backward-compat alias (Sprint 10: use LoopIntervals.STALE_PROCESS_CHECK instead)
STALE_PROCESS_CHECK_INTERVAL = LoopIntervals.STALE_PROCESS_CHECK


# =============================================================================
# Type Protocols for Dependency Injection
# =============================================================================


class HealthManagerProtocol(Protocol):
    """Protocol for health manager operations."""

    def set_work_queue(self, work_queue: Any) -> None:
        """Set the work queue for health checks."""
        ...

    def find_stuck_jobs(self, work_items: list[Any]) -> list[tuple[Any, float]]:
        """Find stuck jobs from running work items."""
        ...

    async def recover_stuck_job(self, work_item: Any, expected_timeout: float) -> Any:
        """Attempt to recover a stuck job."""
        ...


class WorkQueueProtocol(Protocol):
    """Protocol for work queue operations."""

    def get_queue_status(self) -> dict[str, Any]:
        """Get current queue status."""
        ...


class PeerProtocol(Protocol):
    """Protocol for peer information."""

    node_id: str

    def is_alive(self) -> bool:
        """Check if peer is alive."""
        ...


class AlertManagerProtocol(Protocol):
    """Protocol for predictive alert manager."""

    def record_disk_usage(self, node_id: str, pct: float) -> None:
        """Record disk usage metric."""
        ...

    def record_memory_usage(self, node_id: str, pct: float) -> None:
        """Record memory usage metric."""
        ...

    def record_queue_depth(self, depth: int) -> None:
        """Record work queue depth."""
        ...

    async def run_all_checks(
        self,
        node_ids: list[str],
        model_ids: list[str],
        last_training_time: float,
    ) -> list[Any]:
        """Run all predictive checks and return alerts."""
        ...


class NotifierProtocol(Protocol):
    """Protocol for webhook notifier."""

    async def send(
        self,
        title: str,
        message: str,
        level: str,
        fields: dict[str, str],
        node_id: str,
    ) -> None:
        """Send a notification."""
        ...


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SelfHealingConfig:
    """Configuration for self-healing loop."""

    healing_interval_seconds: float = 60.0
    stale_process_check_interval_seconds: float = 300.0
    initial_delay_seconds: float = 45.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.healing_interval_seconds <= 0:
            raise ValueError("healing_interval_seconds must be > 0")
        if self.stale_process_check_interval_seconds <= 0:
            raise ValueError("stale_process_check_interval_seconds must be > 0")
        if self.initial_delay_seconds < 0:
            raise ValueError("initial_delay_seconds must be >= 0")


@dataclass
class PredictiveMonitoringConfig:
    """Configuration for predictive monitoring loop."""

    monitoring_interval_seconds: float = 300.0
    initial_delay_seconds: float = 90.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.monitoring_interval_seconds <= 0:
            raise ValueError("monitoring_interval_seconds must be > 0")
        if self.initial_delay_seconds < 0:
            raise ValueError("initial_delay_seconds must be >= 0")


@dataclass
class SplitBrainDetectionConfig:
    """Configuration for split-brain detection loop.

    December 2025 P2P Hardening: Detect network partitions that could
    cause multiple leaders in the cluster.

    December 29, 2025: Added partition recovery settings for 48-hour
    autonomous operation.
    """

    detection_interval_seconds: float = 60.0
    initial_delay_seconds: float = 120.0
    request_timeout_seconds: float = 5.0
    min_peers_for_detection: int = 3

    # December 29, 2025: Partition recovery settings (48-hour autonomous operation)
    # Using PartitionRecoveryDefaults from coordination_defaults.py
    # Jan 2026: Reduced alert threshold from 1800s (30min) to 600s (10min) for faster alerting
    partition_alert_threshold_seconds: int = 600  # 10 min before alert
    partition_resync_delay_seconds: int = 60  # Wait after partition heals before resync
    min_peers_for_healthy: int = 3  # Minimum peers for healthy cluster

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.detection_interval_seconds <= 0:
            raise ValueError("detection_interval_seconds must be > 0")
        if self.initial_delay_seconds < 0:
            raise ValueError("initial_delay_seconds must be >= 0")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be > 0")
        if self.min_peers_for_detection < 1:
            raise ValueError("min_peers_for_detection must be >= 1")

    @classmethod
    def from_defaults(cls) -> "SplitBrainDetectionConfig":
        """Create config from PartitionRecoveryDefaults."""
        try:
            from app.config.coordination_defaults import PartitionRecoveryDefaults
            return cls(
                partition_alert_threshold_seconds=PartitionRecoveryDefaults.PARTITION_ALERT_THRESHOLD,
                partition_resync_delay_seconds=PartitionRecoveryDefaults.RESYNC_DELAY_SECONDS,
                min_peers_for_healthy=PartitionRecoveryDefaults.MIN_PEERS_FOR_HEALTHY,
            )
        except ImportError:
            return cls()


# =============================================================================
# SelfHealingLoop
# =============================================================================


class SelfHealingLoop(BaseLoop):
    """Background loop for self-healing: recover stuck jobs and unhealthy nodes.

    Detects jobs that have exceeded their expected timeout and automatically
    terminates and reschedules them. Stuck job recovery only runs on the leader,
    but stale process cleanup runs on all nodes.

    Key responsibilities:
    1. Clean up stale processes (all nodes)
    2. Find and recover stuck jobs (leader only)
    3. Report recovery statistics
    """

    def __init__(
        self,
        is_leader: Callable[[], bool],
        get_health_manager: Callable[[], HealthManagerProtocol | None],
        get_work_queue: Callable[[], WorkQueueProtocol | None],
        cleanup_stale_processes: Callable[[], int],
        config: SelfHealingConfig | None = None,
        restart_stopped_loops: Callable[[], Coroutine[Any, Any, dict[str, bool]]] | None = None,
    ):
        """Initialize self-healing loop.

        Args:
            is_leader: Callback returning True if this node is leader
            get_health_manager: Callback returning health manager instance
            get_work_queue: Callback returning work queue instance
            cleanup_stale_processes: Callback to clean up stale processes
            config: Loop configuration
            restart_stopped_loops: Optional callback to restart stopped loops (Jan 2026)
        """
        self.config = config or SelfHealingConfig()
        super().__init__(
            name="self_healing",
            interval=self.config.healing_interval_seconds,
        )
        self._is_leader = is_leader
        self._get_health_manager = get_health_manager
        self._get_work_queue = get_work_queue
        self._cleanup_stale_processes = cleanup_stale_processes
        self._restart_stopped_loops = restart_stopped_loops

        # Statistics
        self._stale_processes_cleaned = 0
        self._stuck_jobs_recovered = 0
        self._loops_restarted = 0
        self._last_stale_check = 0.0
        self._last_loop_restart_check = 0.0

    async def _on_start(self) -> None:
        """Initial delay before starting healing."""
        logger.info("Self-healing loop starting...")
        await asyncio.sleep(self.config.initial_delay_seconds)
        logger.info("Self-healing loop started")

    async def _run_once(self) -> None:
        """Execute one healing iteration."""
        now = time.time()

        # Stale process cleanup runs on ALL nodes (not just leader)
        # Jan 23, 2026: Wrap blocking subprocess calls in asyncio.to_thread()
        if now - self._last_stale_check >= self.config.stale_process_check_interval_seconds:
            try:
                killed = await asyncio.to_thread(self._cleanup_stale_processes)
                if killed > 0:
                    self._stale_processes_cleaned += killed
                    logger.info(f"[SelfHealing] Cleaned up {killed} stale processes")
            except Exception as e:
                logger.debug(f"[SelfHealing] Stale process cleanup error: {e}")
            self._last_stale_check = now

        # Jan 2026: Loop auto-restart - check every 5 minutes for stopped loops
        # This ensures 48h autonomous operation by recovering crashed loops
        loop_restart_interval = 300.0  # 5 minutes
        if self._restart_stopped_loops and now - self._last_loop_restart_check >= loop_restart_interval:
            try:
                results = await self._restart_stopped_loops()
                restarted_count = sum(1 for success in results.values() if success)
                if restarted_count > 0:
                    self._loops_restarted += restarted_count
                    logger.info(f"[SelfHealing] Restarted {restarted_count} stopped loops: {list(results.keys())}")
            except Exception as e:
                logger.debug(f"[SelfHealing] Loop restart error: {e}")
            self._last_loop_restart_check = now

        # Only leader performs job recovery
        if not self._is_leader():
            return

        health_manager = self._get_health_manager()
        if health_manager is None:
            return

        # Wire up work queue
        wq = self._get_work_queue()
        if wq is None:
            return

        health_manager.set_work_queue(wq)

        # Get running work items
        status = wq.get_queue_status()
        running_items = status.get("running", [])

        # Convert to WorkItem objects for stuck job detection
        try:
            from app.coordination.work_queue import WorkItem

            work_items = []
            for item_dict in running_items:
                with contextlib.suppress(Exception):
                    work_items.append(WorkItem.from_dict(item_dict))

            # Find stuck jobs
            stuck_jobs = health_manager.find_stuck_jobs(work_items)

            for work_item, expected_timeout in stuck_jobs:
                running_time = time.time() - work_item.started_at
                logger.warning(
                    f"[SelfHealing] Detected stuck job {work_item.work_id} "
                    f"on {work_item.claimed_by} "
                    f"(running {running_time:.0f}s > expected {expected_timeout * 1.5:.0f}s)"
                )
                result = await health_manager.recover_stuck_job(work_item, expected_timeout)
                if hasattr(result, "value") and result.value == "success":
                    self._stuck_jobs_recovered += 1
                    logger.info(f"[SelfHealing] Recovered stuck job {work_item.work_id}")

        except ImportError:
            logger.debug("[SelfHealing] WorkItem not available")

    def get_healing_stats(self) -> dict[str, Any]:
        """Get self-healing statistics."""
        return {
            "stale_processes_cleaned": self._stale_processes_cleaned,
            "stuck_jobs_recovered": self._stuck_jobs_recovered,
            "loops_restarted": self._loops_restarted,
            "last_stale_check": self._last_stale_check,
            "last_loop_restart_check": self._last_loop_restart_check,
            **self.stats.to_dict(),
        }

    def health_check(self) -> Any:
        """Check self-healing loop health for DaemonManager integration.

        Jan 2026: Added specialized health check for resilience monitoring.

        Returns:
            HealthCheckResult with healing-specific metrics
        """
        try:
            from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
        except ImportError:
            return {
                "healthy": self._running,
                "status": "running" if self._running else "stopped",
                "message": f"SelfHealingLoop {'running' if self._running else 'stopped'}",
                "details": self.get_healing_stats(),
            }

        if not self._running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="SelfHealingLoop is stopped",
            )

        # Check base loop health first
        base_health = super().health_check()
        if not base_health.healthy:
            return base_health

        # Add healing-specific details
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="SelfHealingLoop operational",
            details={
                "stale_processes_cleaned": self._stale_processes_cleaned,
                "stuck_jobs_recovered": self._stuck_jobs_recovered,
                "loops_restarted": self._loops_restarted,
                "is_leader": self._is_leader(),
                "total_runs": self.stats.total_runs,
                "success_rate": f"{self.stats.success_rate:.1f}%",
            },
        )


# =============================================================================
# PredictiveMonitoringLoop
# =============================================================================


class PredictiveMonitoringLoop(BaseLoop):
    """Background loop for proactive monitoring and alerting.

    Predicts issues before they occur and sends proactive alerts.
    Only runs on the leader node.

    Key responsibilities:
    1. Collect metrics from peers (disk, memory)
    2. Record queue depth
    3. Run predictive alert checks
    4. Send alerts via webhook notifier
    """

    def __init__(
        self,
        is_leader: Callable[[], bool],
        get_alert_manager: Callable[[], AlertManagerProtocol | None],
        get_work_queue: Callable[[], WorkQueueProtocol | None],
        get_peers: Callable[[], list[PeerProtocol]],
        get_notifier: Callable[[], NotifierProtocol | None],
        get_production_models: Callable[[], tuple[list[str], float]] | None = None,
        config: PredictiveMonitoringConfig | None = None,
    ):
        """Initialize predictive monitoring loop.

        Args:
            is_leader: Callback returning True if this node is leader
            get_alert_manager: Callback returning predictive alert manager
            get_work_queue: Callback returning work queue instance
            get_peers: Callback returning list of peer objects
            get_notifier: Callback returning webhook notifier
            get_production_models: Optional callback returning (model_ids, last_training_time)
            config: Loop configuration
        """
        self.config = config or PredictiveMonitoringConfig()
        super().__init__(
            name="predictive_monitoring",
            interval=self.config.monitoring_interval_seconds,
        )
        self._is_leader = is_leader
        self._get_alert_manager = get_alert_manager
        self._get_work_queue = get_work_queue
        self._get_peers = get_peers
        self._get_notifier = get_notifier
        self._get_production_models = get_production_models

        # Statistics
        self._alerts_sent = 0
        self._checks_performed = 0

    async def _on_start(self) -> None:
        """Initial delay before starting monitoring."""
        logger.info("Predictive monitoring loop starting...")
        await asyncio.sleep(self.config.initial_delay_seconds)
        logger.info("Predictive monitoring loop started")

    async def _run_once(self) -> None:
        """Execute one monitoring iteration."""
        # Only leader performs monitoring
        if not self._is_leader():
            return

        alert_manager = self._get_alert_manager()
        if alert_manager is None:
            return

        # Collect metrics from peers
        peers = self._get_peers()
        for peer in peers:
            if not peer.is_alive():
                continue

            # Record disk usage
            disk_pct = float(getattr(peer, "disk_percent", 0) or 0)
            if disk_pct > 0:
                alert_manager.record_disk_usage(peer.node_id, disk_pct)

            # Record memory usage
            mem_pct = float(getattr(peer, "mem_percent", 0) or 0)
            if mem_pct > 0:
                alert_manager.record_memory_usage(peer.node_id, mem_pct)

        # Record queue depth
        wq = self._get_work_queue()
        if wq is not None:
            status = wq.get_queue_status()
            pending = status.get("by_status", {}).get("pending", 0)
            alert_manager.record_queue_depth(pending)

        # Get production models
        node_ids = [p.node_id for p in peers if p.is_alive()]
        model_ids: list[str] = []
        last_training = time.time() - 3600  # Default to 1 hour ago

        if self._get_production_models:
            try:
                model_ids, last_training = self._get_production_models()
            except Exception as e:
                logger.debug(f"[PredictiveMonitoring] Model lookup failed: {e}")
        else:
            # Try to get from model registry
            model_ids, last_training = self._get_models_from_registry()

        # Run all checks
        self._checks_performed += 1
        alerts = await alert_manager.run_all_checks(
            node_ids=node_ids,
            model_ids=model_ids,
            last_training_time=last_training,
        )

        # Send alerts via webhook notifier
        notifier = self._get_notifier()
        if notifier is not None:
            for alert in alerts:
                try:
                    level = "warning" if getattr(alert, "severity", None) and alert.severity.value == "warning" else "error"
                    await notifier.send(
                        title=f"Proactive Alert: {alert.alert_type.value}",
                        message=alert.message,
                        level=level,
                        fields={
                            "action": getattr(alert, "action", ""),
                            "target": getattr(alert, "target_id", ""),
                        },
                        node_id=getattr(alert, "target_id", "unknown"),
                    )
                    self._alerts_sent += 1
                except Exception as e:
                    logger.warning(f"[PredictiveMonitoring] Failed to send alert: {e}")

    def _get_models_from_registry(self) -> tuple[list[str], float]:
        """Get production models from model registry.

        Returns:
            Tuple of (model_ids, last_training_time)
        """
        model_ids: list[str] = []
        last_training = time.time() - 3600  # Default to 1 hour ago

        try:
            from app.training.model_registry import ModelRegistry, ModelStage

            registry = ModelRegistry()
            production_models = registry.get_versions_by_stage(ModelStage.PRODUCTION)
            model_ids = [f"{m['model_id']}_v{m['version']}" for m in production_models]

            # Get last training time from most recently updated model
            if production_models:
                from datetime import datetime

                latest_update = max(
                    datetime.fromisoformat(m["updated_at"].replace("Z", "+00:00"))
                    for m in production_models
                    if m.get("updated_at")
                )
                last_training = latest_update.timestamp()
        except Exception as e:
            logger.debug(f"[PredictiveMonitoring] Model registry lookup failed: {e}")

        return model_ids, last_training

    def get_monitoring_stats(self) -> dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "alerts_sent": self._alerts_sent,
            "checks_performed": self._checks_performed,
            **self.stats.to_dict(),
        }

    def health_check(self) -> Any:
        """Check predictive monitoring loop health for DaemonManager integration.

        Jan 2026: Added specialized health check for predictive monitoring.

        Returns:
            HealthCheckResult with monitoring-specific metrics
        """
        try:
            from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
        except ImportError:
            return {
                "healthy": self._running,
                "status": "running" if self._running else "stopped",
                "message": f"PredictiveMonitoringLoop {'running' if self._running else 'stopped'}",
                "details": self.get_monitoring_stats(),
            }

        if not self._running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="PredictiveMonitoringLoop is stopped",
            )

        # Check base loop health first
        base_health = super().health_check()
        if not base_health.healthy:
            return base_health

        # Non-leader idle state is valid
        if not self._is_leader():
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.PAUSED,
                message="PredictiveMonitoringLoop idle (not leader)",
                details={
                    "is_leader": False,
                    "checks_performed": self._checks_performed,
                },
            )

        # Add monitoring-specific details
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="PredictiveMonitoringLoop operational",
            details={
                "alerts_sent": self._alerts_sent,
                "checks_performed": self._checks_performed,
                "is_leader": True,
                "total_runs": self.stats.total_runs,
                "success_rate": f"{self.stats.success_rate:.1f}%",
            },
        )


# =============================================================================
# SplitBrainDetectionLoop (December 2025 P2P Hardening)
# =============================================================================


class SplitBrainDetectionLoop(BaseLoop):
    """Background loop for detecting split-brain conditions.

    Periodically polls all known peers for their leader_id and detects
    if multiple leaders exist in the cluster (indicating network partition).

    December 2025 P2P Hardening: This loop runs on all nodes and emits
    SPLIT_BRAIN_DETECTED events when multiple leaders are detected.

    Key responsibilities:
    1. Poll peers for their leader_id
    2. Detect multiple leaders (split-brain condition)
    3. Emit alerts and log critical warnings
    4. Track detection statistics
    """

    def __init__(
        self,
        get_peers: Callable[[], dict[str, Any]],
        get_peer_endpoint: Callable[[str], str | None],
        get_own_leader_id: Callable[[], str | None],
        get_cluster_epoch: Callable[[], int],
        on_split_brain_detected: Callable[[list[str], int], Coroutine[Any, Any, None]] | None = None,
        config: SplitBrainDetectionConfig | None = None,
    ):
        """Initialize split-brain detection loop.

        Args:
            get_peers: Callback returning dict of peer_id -> peer_info
            get_peer_endpoint: Callback returning HTTP endpoint for a peer
            get_own_leader_id: Callback returning this node's view of leader_id
            get_cluster_epoch: Callback returning current cluster epoch
            on_split_brain_detected: Optional async callback when split-brain detected
            config: Loop configuration
        """
        self.config = config or SplitBrainDetectionConfig()
        super().__init__(
            name="split_brain_detection",
            interval=self.config.detection_interval_seconds,
        )
        self._get_peers = get_peers
        self._get_peer_endpoint = get_peer_endpoint
        self._get_own_leader_id = get_own_leader_id
        self._get_cluster_epoch = get_cluster_epoch
        self._on_split_brain_detected = on_split_brain_detected

        # Statistics
        self._detections = 0
        self._checks_performed = 0
        self._last_detection_time: float = 0.0
        self._last_leaders_seen: list[str] = []

        # December 29, 2025: Partition tracking for 48-hour autonomous operation
        self._partition_start_time: float = 0.0  # When partition was first detected
        self._partition_alert_emitted: bool = False  # Track if we emitted alert
        self._last_healthy_time: float = time.time()  # Last time we saw healthy cluster

    async def _on_start(self) -> None:
        """Initial delay before starting detection."""
        logger.info("Split-brain detection loop starting...")
        await asyncio.sleep(self.config.initial_delay_seconds)
        logger.info("Split-brain detection loop started")

    async def _run_once(self) -> None:
        """Execute one detection iteration."""
        self._checks_performed += 1
        peers = self._get_peers()

        # Need minimum number of peers for meaningful detection
        if len(peers) < self.config.min_peers_for_detection:
            return

        # Collect leader_id from all reachable peers
        leaders_seen: dict[str, list[str]] = {}  # leader_id -> list of nodes reporting it
        own_leader = self._get_own_leader_id()
        if own_leader:
            leaders_seen.setdefault(own_leader, []).append("self")

        # Poll peers in parallel
        async def poll_peer(peer_id: str) -> tuple[str, str | None]:
            """Poll a single peer for its leader_id."""
            endpoint = self._get_peer_endpoint(peer_id)
            if not endpoint:
                return peer_id, None

            try:
                import aiohttp

                timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_seconds)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    url = f"{endpoint}/status"
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return peer_id, data.get("leader_id")
            except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as e:
                # Network errors - peer unreachable, skip
                logger.debug(f"Could not poll peer {peer_id}: {e}")
            return peer_id, None

        # Poll all peers
        tasks = [poll_peer(peer_id) for peer_id in peers.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, tuple):
                peer_id, leader_id = result
                if leader_id:
                    leaders_seen.setdefault(leader_id, []).append(peer_id)

        # Check for split-brain: multiple distinct leaders
        unique_leaders = list(leaders_seen.keys())

        # Feb 2026: Filter out leaders reported by tiny minorities (likely stale gossip).
        # A leader must be reported by at least 3 nodes (or 25% of responding peers,
        # whichever is smaller) to be considered a real partition participant.
        total_responding = sum(len(v) for v in leaders_seen.values())
        min_reporters = min(3, max(1, total_responding // 4))
        significant_leaders = {
            leader: reporters
            for leader, reporters in leaders_seen.items()
            if len(reporters) >= min_reporters
        }

        if len(significant_leaders) > 1:
            self._detections += 1
            self._last_detection_time = time.time()
            self._last_leaders_seen = list(significant_leaders.keys())

            logger.critical(
                f"[SplitBrain] DETECTED: {len(significant_leaders)} leaders in cluster! "
                f"Leaders: {list(significant_leaders.keys())}"
            )

            # Log which nodes report which leader
            for leader_id, reporters in significant_leaders.items():
                logger.warning(
                    f"[SplitBrain] Leader '{leader_id}' reported by: {reporters}"
                )

            # Emit event
            try:
                from app.distributed.data_events import DataEventType, DataEvent

                event = DataEvent(
                    event_type=DataEventType.SPLIT_BRAIN_DETECTED,
                    payload={
                        "leaders_seen": list(significant_leaders.keys()),
                        "leader_reporters": {k: v for k, v in significant_leaders.items()},
                        "cluster_epoch": self._get_cluster_epoch(),
                        "total_peers_polled": len(peers),
                        "detection_time": self._last_detection_time,
                    },
                    source="SplitBrainDetectionLoop",
                )
                # Try to publish via event bus (fire-and-forget)
                try:
                    from app.coordination.event_router import get_event_bus
                    bus = get_event_bus()
                    if bus:
                        safe_create_task(bus.publish(event), name="resilience-split-brain-event-publish")
                except ImportError:
                    pass
            except ImportError:
                pass

            # Call callback if provided
            if self._on_split_brain_detected:
                try:
                    await self._on_split_brain_detected(
                        list(significant_leaders.keys()), self._get_cluster_epoch()
                    )
                except Exception as e:
                    logger.error(f"[SplitBrain] Callback error: {e}")

            # December 29, 2025: Track partition duration and emit alerts
            await self._track_partition_duration()

        elif len(unique_leaders) > 1:
            # Feb 2026: Stale gossip - some peers report old leader, not a real partition
            stale_leaders = {
                leader: reporters
                for leader, reporters in leaders_seen.items()
                if len(reporters) < min_reporters
            }
            if stale_leaders:
                logger.info(
                    f"[SplitBrain] Stale gossip detected (not a partition): "
                    f"{len(unique_leaders)} leader views, but minority leaders "
                    f"{list(stale_leaders.keys())} only reported by "
                    f"{sum(len(v) for v in stale_leaders.values())} peers "
                    f"(threshold={min_reporters})"
                )
            # Heal any in-progress partition tracking since this isn't a real split
            await self._handle_partition_healed()

        else:
            # Cluster is healthy (single leader)
            await self._handle_partition_healed()

    async def _track_partition_duration(self) -> None:
        """Track how long partition has lasted and emit alerts.

        December 29, 2025: Part of 48-hour autonomous operation.
        """
        now = time.time()

        # Start tracking if not already
        if self._partition_start_time == 0.0:
            self._partition_start_time = now
            logger.warning("[SplitBrain] Partition started, tracking duration")

        # Calculate partition duration
        duration = now - self._partition_start_time
        threshold = self.config.partition_alert_threshold_seconds

        # Emit alert if partition exceeds threshold
        if duration >= threshold and not self._partition_alert_emitted:
            self._partition_alert_emitted = True
            logger.critical(
                f"[SplitBrain] PARTITION ALERT: Duration {duration/60:.1f} minutes "
                f"exceeds threshold {threshold/60:.1f} minutes"
            )

            # Emit health alert event
            try:
                from app.distributed.data_events import DataEventType, DataEvent
                from app.coordination.event_router import get_event_bus

                event = DataEvent(
                    event_type=DataEventType.HEALTH_ALERT,
                    payload={
                        "alert_type": "partition_duration_exceeded",
                        "partition_duration_seconds": duration,
                        "threshold_seconds": threshold,
                        "leaders_seen": self._last_leaders_seen,
                    },
                    source="SplitBrainDetectionLoop",
                )
                bus = get_event_bus()
                if bus:
                    # bus.publish() is async, create task to avoid blocking
                    safe_create_task(bus.publish(event), name="resilience-split-brain-alert")
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"[SplitBrain] Failed to emit alert event: {e}")

    async def _handle_partition_healed(self) -> None:
        """Handle partition recovery and trigger resync.

        December 29, 2025: Part of 48-hour autonomous operation.
        """
        now = time.time()

        # Check if we were previously partitioned
        if self._partition_start_time > 0.0:
            partition_duration = now - self._partition_start_time

            logger.info(
                f"[SplitBrain] Partition HEALED after {partition_duration/60:.1f} minutes"
            )

            # Wait before triggering resync (let cluster stabilize)
            await asyncio.sleep(self.config.partition_resync_delay_seconds)

            # Trigger cluster resync
            await self._trigger_partition_resync()

            # Reset partition tracking
            self._partition_start_time = 0.0
            self._partition_alert_emitted = False

        # Update last healthy time
        self._last_healthy_time = now

    async def _trigger_partition_resync(self) -> None:
        """Trigger cluster-wide resync after partition heals.

        December 29, 2025: Part of 48-hour autonomous operation.
        """
        logger.info("[SplitBrain] Triggering cluster resync after partition healed")

        try:
            from app.distributed.data_events import DataEventType, DataEvent
            from app.coordination.event_router import get_event_bus

            event = DataEvent(
                event_type=DataEventType.SYNC_REQUEST,
                payload={
                    "sync_type": "partition_recovery",
                    "priority": "high",
                    "source": "split_brain_detection",
                    "partition_duration_seconds": time.time() - self._last_healthy_time,
                },
                source="SplitBrainDetectionLoop",
            )
            bus = get_event_bus()
            if bus:
                # bus.publish() is async, create task to avoid blocking
                safe_create_task(bus.publish(event), name="resilience-split-brain-sync-request")
                logger.info("[SplitBrain] Emitted SYNC_REQUEST for partition recovery")
        except ImportError:
            logger.debug("[SplitBrain] Event system not available, skipping resync trigger")
        except Exception as e:
            logger.warning(f"[SplitBrain] Failed to trigger partition resync: {e}")

    def get_detection_stats(self) -> dict[str, Any]:
        """Get split-brain detection statistics."""
        # Calculate partition duration if currently partitioned
        partition_duration = 0.0
        if self._partition_start_time > 0.0:
            partition_duration = time.time() - self._partition_start_time

        return {
            "detections": self._detections,
            "checks_performed": self._checks_performed,
            "last_detection_time": self._last_detection_time,
            "last_leaders_seen": self._last_leaders_seen,
            # December 29, 2025: Partition tracking stats
            "partition_active": self._partition_start_time > 0.0,
            "partition_duration_seconds": partition_duration,
            "partition_alert_emitted": self._partition_alert_emitted,
            "last_healthy_time": self._last_healthy_time,
            **self.stats.to_dict(),
        }

    def health_check(self) -> Any:
        """Check split-brain detection loop health for DaemonManager integration.

        Jan 2026: Added specialized health check for partition detection.

        Returns:
            HealthCheckResult with partition-specific metrics
        """
        try:
            from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
        except ImportError:
            return {
                "healthy": self._running,
                "status": "running" if self._running else "stopped",
                "message": f"SplitBrainDetectionLoop {'running' if self._running else 'stopped'}",
                "details": self.get_detection_stats(),
            }

        if not self._running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="SplitBrainDetectionLoop is stopped",
            )

        # Check base loop health first
        base_health = super().health_check()
        if not base_health.healthy:
            return base_health

        # Check for active partition - this is a critical condition
        partition_duration = 0.0
        if self._partition_start_time > 0.0:
            partition_duration = time.time() - self._partition_start_time

            # Active partition is a critical condition
            if partition_duration >= self.config.partition_alert_threshold_seconds:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.ERROR,
                    message=f"SPLIT-BRAIN ACTIVE: Partition detected for {partition_duration/60:.1f} minutes",
                    details={
                        "partition_active": True,
                        "partition_duration_seconds": partition_duration,
                        "leaders_seen": self._last_leaders_seen,
                        "detections": self._detections,
                    },
                )

            # Partition active but under threshold - degraded
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message=f"Partition detected ({partition_duration:.0f}s)",
                details={
                    "partition_active": True,
                    "partition_duration_seconds": partition_duration,
                    "leaders_seen": self._last_leaders_seen,
                    "detections": self._detections,
                },
            )

        # No active partition - healthy
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="SplitBrainDetectionLoop operational",
            details={
                "partition_active": False,
                "detections": self._detections,
                "checks_performed": self._checks_performed,
                "last_healthy_time": self._last_healthy_time,
                "total_runs": self.stats.total_runs,
                "success_rate": f"{self.stats.success_rate:.1f}%",
            },
        )


__all__ = [
    # Configuration
    "SelfHealingConfig",
    "PredictiveMonitoringConfig",
    "SplitBrainDetectionConfig",
    # Loops
    "SelfHealingLoop",
    "PredictiveMonitoringLoop",
    "SplitBrainDetectionLoop",
    # Constants
    "STALE_PROCESS_CHECK_INTERVAL",
]
