"""Training Activity Detection Daemon.

Monitors cluster for training activity and triggers priority data sync
before training starts. This ensures training nodes have fresh data.

December 2025: Restored from deprecated cluster_data_sync.py.
The pattern was lost during consolidation but is critical for ensuring
training doesn't start with stale data.

Usage:
    from app.coordination.training_activity_daemon import (
        TrainingActivityDaemon,
        get_training_activity_daemon,
    )

    # Start the daemon
    daemon = get_training_activity_daemon()
    await daemon.start()

    # Check training status
    training_nodes = daemon.get_training_nodes()
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from app.coordination.base_daemon import BaseDaemon, DaemonConfig
from app.coordination.health_check_helper import HealthCheckHelper
from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

logger = logging.getLogger(__name__)


@dataclass
class TrainingActivityConfig(DaemonConfig):
    """Configuration for training activity detection."""

    # Check interval (seconds)
    check_interval_seconds: int = 30

    # Enable signal handling for graceful shutdown with final sync
    handle_signals: bool = True

    # Whether to trigger priority sync when training detected
    trigger_priority_sync: bool = True

    # Process patterns to detect as training
    training_process_patterns: list[str] = field(
        default_factory=lambda: [
            "app.training.train",
            "train.py",
            "training_loop",
        ]
    )

    @classmethod
    def from_env(cls, prefix: str = "RINGRIFT_TRAINING_ACTIVITY") -> "TrainingActivityConfig":
        """Load configuration from environment variables."""
        import os

        config = super().from_env(prefix)

        # Training-specific env vars
        if os.environ.get(f"{prefix}_TRIGGER_SYNC"):
            config.trigger_priority_sync = os.environ.get(f"{prefix}_TRIGGER_SYNC", "1") == "1"

        return config


class TrainingActivityDaemon(BaseDaemon[TrainingActivityConfig]):
    """Detects training activity and triggers priority sync.

    This daemon monitors for training activity across the cluster and
    ensures training nodes have fresh data before training starts.

    Features:
    - Detects training via P2P status (running_jobs, processes)
    - Detects local training via process monitoring
    - Triggers priority sync when new training detected
    - Emits TRAINING_STARTED events for coordination
    - Graceful shutdown on SIGTERM with final sync
    """

    def __init__(self, config: TrainingActivityConfig | None = None):
        super().__init__(config)
        self._training_nodes: set[str] = set()
        self._last_check_time: float = 0.0
        self._syncs_triggered: int = 0

    def _get_default_config(self) -> TrainingActivityConfig:
        return TrainingActivityConfig.from_env()

    def _get_daemon_name(self) -> str:
        return "TrainingActivityDaemon"

    async def _on_graceful_shutdown(self) -> None:
        """Trigger final sync before shutdown.

        Override of BaseDaemon hook - performs priority sync to ensure
        no training data is lost when the daemon is terminated.
        """
        if self.config.trigger_priority_sync:
            logger.info(f"[{self._get_daemon_name()}] Triggering final sync before shutdown")
            await self._trigger_priority_sync("termination")

    async def _run_cycle(self) -> None:
        """Main daemon cycle - check for training activity."""
        self._last_check_time = time.time()

        # Check P2P cluster for training activity
        training_detected = await self._check_p2p_training()

        # Also check local processes (via thread pool to avoid blocking)
        if await asyncio.to_thread(self.detect_local_training):
            training_detected.add(self.node_id)

        # Detect new training nodes
        new_training = training_detected - self._training_nodes
        if new_training:
            logger.info(f"[{self._get_daemon_name()}] New training detected on nodes: {new_training}")
            await self._on_training_detected(new_training)

        # Detect training completion
        completed_training = self._training_nodes - training_detected
        if completed_training:
            logger.info(f"[{self._get_daemon_name()}] Training completed on nodes: {completed_training}")

        self._training_nodes = training_detected

        # December 29, 2025: Update SyncRouter with training-active nodes
        # This enables priority sync to training nodes (reduces data staleness)
        self._update_sync_router_training_nodes(training_detected)

    def _update_sync_router_training_nodes(self, training_detected: set[str]) -> None:
        """Update SyncRouter with training-active nodes for priority sync.

        December 29, 2025: Added to enable +50 priority boost for nodes
        actively running training jobs.

        Args:
            training_detected: Set of node IDs currently running training
        """
        try:
            from app.coordination.sync_router import get_sync_router

            router = get_sync_router()
            router.update_training_active_nodes(training_detected)
        except ImportError:
            # SyncRouter not available - expected in minimal environments
            logger.debug(f"[{self._get_daemon_name()}] SyncRouter not available for training node update")
        except Exception as e:
            # Non-critical - log and continue
            logger.debug(f"[{self._get_daemon_name()}] Failed to update sync router: {e}")

    async def _check_p2p_training(self) -> set[str]:
        """Check P2P status for training activity."""
        training_detected: set[str] = set()

        try:
            status = await self._get_p2p_status()
            if not status:
                return training_detected

            peers = status.get("peers", {})
            for node_id, info in peers.items():
                # Check running_jobs for training
                running_jobs = info.get("running_jobs", [])
                for job in running_jobs:
                    job_type = job.get("type", "")
                    if "train" in job_type.lower():
                        training_detected.add(node_id)
                        break

                # Check processes for training patterns
                processes = info.get("processes", [])
                for proc in processes:
                    proc_str = str(proc).lower()
                    for pattern in self.config.training_process_patterns:
                        if pattern.lower() in proc_str:
                            training_detected.add(node_id)
                            break

        except Exception as e:
            logger.debug(f"[{self._get_daemon_name()}] P2P status check failed: {e}")

        return training_detected

    async def _get_p2p_status(self) -> dict[str, Any] | None:
        """Get P2P cluster status."""
        try:
            import aiohttp

            from app.config.cluster_config import get_p2p_port
            from app.config.coordination_defaults import TransportDefaults

            p2p_port = get_p2p_port()
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://localhost:{p2p_port}/status",
                    timeout=aiohttp.ClientTimeout(total=TransportDefaults.HTTP_TIMEOUT),
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except ImportError:
            # aiohttp not installed - expected in some environments
            logger.debug("aiohttp not available for P2P status check")
        except (asyncio.TimeoutError, OSError, ConnectionError) as e:
            # Connection failures - expected when P2P daemon not running
            logger.debug(f"P2P status check failed (expected): {e}")
        except Exception as e:
            # Unexpected errors - log at warning level
            logger.warning(f"Unexpected error checking P2P status: {e}")
        return None

    async def _on_training_detected(self, nodes: set[str]) -> None:
        """Handle detection of new training activity."""
        # Emit TRAINING_STARTED event
        await self._emit_training_started(nodes)

        # Trigger priority sync if enabled
        if self.config.trigger_priority_sync:
            await self._trigger_priority_sync(",".join(nodes))

    async def _emit_training_started(self, nodes: set[str]) -> None:
        """Emit TRAINING_STARTED event for coordination."""
        try:
            from app.distributed.data_events import emit_training_started

            for node_id in nodes:
                await emit_training_started(
                    node_id=node_id,
                    source="TrainingActivityDaemon",
                )
        except Exception as e:
            logger.debug(f"[{self._get_daemon_name()}] Failed to emit training started: {e}")

    async def _trigger_priority_sync(self, reason: str) -> None:
        """Trigger priority data sync for training nodes."""
        try:
            from app.coordination.sync_facade import get_sync_facade

            facade = get_sync_facade()
            logger.info(f"[{self._get_daemon_name()}] Triggering priority sync (reason: {reason})")

            await facade.trigger_priority_sync(
                reason=f"training_detected:{reason}",
                data_type="games",
            )
            self._syncs_triggered += 1

        except Exception as e:
            logger.error(f"[{self._get_daemon_name()}] Priority sync failed: {e}")
            # Emit sync failure event
            try:
                from app.distributed.data_events import emit_data_sync_failed

                await emit_data_sync_failed(
                    host=reason,
                    error=str(e),
                    source="TrainingActivityDaemon",
                )
            except (ImportError, RuntimeError, OSError) as emit_err:
                logger.debug(
                    f"[{self._get_daemon_name()}] Failed to emit sync failure event: {emit_err}"
                )

    def detect_local_training(self) -> bool:
        """Check if training is running locally.

        Returns:
            True if local training process detected
        """
        try:
            for pattern in self.config.training_process_patterns:
                result = subprocess.run(
                    ["pgrep", "-f", pattern],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return True
        except subprocess.TimeoutExpired as e:
            # pgrep taking too long - skip this cycle
            logger.debug(f"[{self._get_daemon_name()}] Local training check timed out: {e}")
        except subprocess.SubprocessError as e:
            # pgrep failed for process-related reason
            logger.debug(f"[{self._get_daemon_name()}] Local training check subprocess error: {e}")
        except (OSError, FileNotFoundError) as e:
            # pgrep not available or permission denied
            logger.debug(f"[{self._get_daemon_name()}] Local training check OS error: {e}")
        return False

    def get_training_nodes(self) -> set[str]:
        """Get set of currently detected training nodes."""
        return self._training_nodes.copy()

    def health_check(self) -> HealthCheckResult:
        """Return health check result for daemon protocol.

        Override of BaseDaemon.health_check() with training-specific details.
        """
        details = {
            "running": self._running,
            "training_nodes": list(self._training_nodes),
            "syncs_triggered": self._syncs_triggered,
            "last_check_time": self._last_check_time,
            "uptime_seconds": self.uptime_seconds,
            "cycles_completed": self._cycles_completed,
            "errors_count": self._errors_count,
        }

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message=f"{self._get_daemon_name()} is not running",
                details=details,
            )

        # Check for high error rate using HealthCheckHelper
        is_healthy, msg = HealthCheckHelper.check_error_rate(
            errors=self._errors_count,
            cycles=self._cycles_completed,
            threshold=0.5,
        )
        if not is_healthy:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=msg,
                details=details,
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"{self._get_daemon_name()} healthy, tracking {len(self._training_nodes)} training nodes",
            details=details,
        )

    def get_status(self) -> dict[str, Any]:
        """Get daemon status for monitoring."""
        health = self.health_check()
        return {
            "name": self._get_daemon_name(),
            "running": self._running,
            "uptime_seconds": self.uptime_seconds,
            "config": {
                "check_interval": self.config.check_interval_seconds,
                "trigger_priority_sync": self.config.trigger_priority_sync,
            },
            "health": {
                "healthy": health.healthy,
                "status": health.status.value if hasattr(health.status, "value") else str(health.status),
                "message": health.message,
            },
            **health.details,
        }


# =============================================================================
# Singleton Access
# =============================================================================

_daemon_instance: TrainingActivityDaemon | None = None


def get_training_activity_daemon() -> TrainingActivityDaemon:
    """Get the singleton TrainingActivityDaemon instance."""
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = TrainingActivityDaemon()
    return _daemon_instance


def reset_training_activity_daemon() -> None:
    """Reset the singleton (for testing)."""
    global _daemon_instance
    _daemon_instance = None
