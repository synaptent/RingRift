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

from .base import BaseLoop

logger = logging.getLogger(__name__)

# Constants
STALE_PROCESS_CHECK_INTERVAL = 300  # 5 minutes between stale process checks


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
    ):
        """Initialize self-healing loop.

        Args:
            is_leader: Callback returning True if this node is leader
            get_health_manager: Callback returning health manager instance
            get_work_queue: Callback returning work queue instance
            cleanup_stale_processes: Callback to clean up stale processes
            config: Loop configuration
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

        # Statistics
        self._stale_processes_cleaned = 0
        self._stuck_jobs_recovered = 0
        self._last_stale_check = 0.0

    async def _on_start(self) -> None:
        """Initial delay before starting healing."""
        logger.info("Self-healing loop starting...")
        await asyncio.sleep(self.config.initial_delay_seconds)
        logger.info("Self-healing loop started")

    async def _run_once(self) -> None:
        """Execute one healing iteration."""
        now = time.time()

        # Stale process cleanup runs on ALL nodes (not just leader)
        if now - self._last_stale_check >= self.config.stale_process_check_interval_seconds:
            try:
                killed = self._cleanup_stale_processes()
                if killed > 0:
                    self._stale_processes_cleaned += killed
                    logger.info(f"[SelfHealing] Cleaned up {killed} stale processes")
            except Exception as e:
                logger.debug(f"[SelfHealing] Stale process cleanup error: {e}")
            self._last_stale_check = now

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
            "last_stale_check": self._last_stale_check,
            **self.stats.to_dict(),
        }


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


__all__ = [
    # Configuration
    "SelfHealingConfig",
    "PredictiveMonitoringConfig",
    # Loops
    "SelfHealingLoop",
    "PredictiveMonitoringLoop",
    # Constants
    "STALE_PROCESS_CHECK_INTERVAL",
]
