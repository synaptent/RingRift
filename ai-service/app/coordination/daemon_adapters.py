"""Daemon Adapters - Wrappers for integrating daemons with DaemonManager.

This module provides adapters that wrap existing daemon classes to make them
compatible with the centralized DaemonManager lifecycle management.

Each adapter:
1. Implements a consistent interface for DaemonManager
2. Optionally acquires an OrchestratorRole for exclusive execution
3. Handles startup, shutdown, and health checks uniformly
4. Reports status to the DaemonManager

Usage:
    from app.coordination.daemon_adapters import (
        DistillationDaemonAdapter,
        PromotionDaemonAdapter,
        get_daemon_adapter,
    )

    # Get an adapter for a daemon type
    adapter = get_daemon_adapter(DaemonType.DISTILLATION)

    # Use with DaemonManager
    manager = get_daemon_manager()
    manager.register_factory(
        DaemonType.DISTILLATION,
        adapter.run,
        depends_on=adapter.depends_on,
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .daemon_manager import DaemonType
from .orchestrator_registry import OrchestratorRole, get_registry

logger = logging.getLogger(__name__)


@dataclass
class DaemonAdapterConfig:
    """Configuration for daemon adapters."""

    # Role acquisition
    acquire_role: bool = True  # Whether to acquire an OrchestratorRole
    role_timeout_seconds: float = 300.0  # Max time to wait for role

    # Health checks
    health_check_interval: float = 60.0
    unhealthy_threshold: int = 3  # Consecutive failures before unhealthy

    # Restart policy
    auto_restart: bool = True
    max_restarts: int = 5
    restart_delay_seconds: float = 5.0

    # Polling intervals
    poll_interval_seconds: float = 30.0


class DaemonAdapter(ABC):
    """Base class for daemon adapters.

    Adapters wrap existing daemon implementations to provide a consistent
    interface for the DaemonManager.
    """

    def __init__(self, config: DaemonAdapterConfig | None = None):
        self.config = config or DaemonAdapterConfig()
        self._running = False
        self._healthy = True
        self._unhealthy_count = 0
        self._start_time: float = 0.0
        self._daemon_instance: Any = None

    @property
    @abstractmethod
    def daemon_type(self) -> DaemonType:
        """Get the DaemonType this adapter handles."""
        ...

    @property
    def role(self) -> OrchestratorRole | None:
        """Get the OrchestratorRole required by this daemon (if any)."""
        return None

    @property
    def depends_on(self) -> list[DaemonType]:
        """Get list of DaemonTypes this daemon depends on."""
        return []

    @abstractmethod
    async def _create_daemon(self) -> Any:
        """Create the daemon instance. Override in subclasses."""
        ...

    @abstractmethod
    async def _run_daemon(self, daemon: Any) -> None:
        """Run the daemon main loop. Override in subclasses."""
        ...

    async def _health_check(self) -> bool:
        """Check daemon health. Override for custom health checks."""
        return self._running and self._daemon_instance is not None

    async def run(self) -> None:
        """Main entry point for DaemonManager integration."""
        # Acquire role if required
        if self.config.acquire_role and self.role:
            registry = get_registry()
            if not registry.wait_for_role(
                self.role,
                timeout=self.config.role_timeout_seconds,
            ):
                logger.warning(
                    f"[{self.__class__.__name__}] Could not acquire role {self.role.value}"
                )
                return

        try:
            self._running = True
            self._start_time = time.time()

            # Create daemon instance
            self._daemon_instance = await self._create_daemon()
            if self._daemon_instance is None:
                logger.error(f"[{self.__class__.__name__}] Failed to create daemon")
                return

            logger.info(f"[{self.__class__.__name__}] Started")

            # Run daemon with health monitoring
            await self._run_with_health_monitoring()

        except asyncio.CancelledError:
            logger.info(f"[{self.__class__.__name__}] Cancelled")
            raise
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Error: {e}")
            raise
        finally:
            self._running = False
            if self.config.acquire_role and self.role:
                get_registry().release_role()

    async def _run_with_health_monitoring(self) -> None:
        """Run daemon with periodic health checks."""
        health_task = asyncio.create_task(self._health_monitor_loop())

        try:
            await self._run_daemon(self._daemon_instance)
        finally:
            health_task.cancel()
            try:
                await health_task
            except asyncio.CancelledError:
                pass

    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        while self._running:
            try:
                is_healthy = await self._health_check()
                if is_healthy:
                    self._healthy = True
                    self._unhealthy_count = 0
                else:
                    self._unhealthy_count += 1
                    if self._unhealthy_count >= self.config.unhealthy_threshold:
                        self._healthy = False
                        logger.warning(
                            f"[{self.__class__.__name__}] Marked unhealthy after "
                            f"{self._unhealthy_count} failed checks"
                        )
            except Exception as e:
                logger.debug(f"[{self.__class__.__name__}] Health check error: {e}")
                self._unhealthy_count += 1

            await asyncio.sleep(self.config.health_check_interval)

    def get_status(self) -> dict[str, Any]:
        """Get adapter status."""
        return {
            "daemon_type": self.daemon_type.value,
            "role": self.role.value if self.role else None,
            "running": self._running,
            "healthy": self._healthy,
            "unhealthy_count": self._unhealthy_count,
            "uptime_seconds": time.time() - self._start_time if self._running else 0,
            "has_instance": self._daemon_instance is not None,
        }


class DistillationDaemonAdapter(DaemonAdapter):
    """Adapter for the distillation daemon."""

    @property
    def daemon_type(self) -> DaemonType:
        return DaemonType.DISTILLATION

    @property
    def role(self) -> OrchestratorRole:
        return OrchestratorRole.DISTILLATION_LEADER

    async def _create_daemon(self) -> Any:
        try:
            from app.training.distillation_daemon import DistillationDaemon

            return DistillationDaemon()
        except ImportError:
            logger.warning("[DistillationDaemonAdapter] DistillationDaemon not available")
            return None

    async def _run_daemon(self, daemon: Any) -> None:
        if hasattr(daemon, "run"):
            await daemon.run()
        elif hasattr(daemon, "start"):
            await daemon.start()
        else:
            # Fallback: run indefinitely
            while self._running:
                await asyncio.sleep(self.config.poll_interval_seconds)


class PromotionDaemonAdapter(DaemonAdapter):
    """Adapter for the unified promotion daemon."""

    @property
    def daemon_type(self) -> DaemonType:
        return DaemonType.UNIFIED_PROMOTION

    @property
    def role(self) -> OrchestratorRole:
        return OrchestratorRole.PROMOTION_LEADER

    async def _create_daemon(self) -> Any:
        try:
            from app.training.unified_promotion_daemon import UnifiedPromotionDaemon

            return UnifiedPromotionDaemon()
        except ImportError:
            logger.warning("[PromotionDaemonAdapter] UnifiedPromotionDaemon not available")
            return None

    async def _run_daemon(self, daemon: Any) -> None:
        if hasattr(daemon, "run"):
            await daemon.run()
        elif hasattr(daemon, "start"):
            await daemon.start()
        else:
            while self._running:
                await asyncio.sleep(self.config.poll_interval_seconds)


class ExternalDriveSyncAdapter(DaemonAdapter):
    """Adapter for external drive sync daemon."""

    @property
    def daemon_type(self) -> DaemonType:
        return DaemonType.EXTERNAL_DRIVE_SYNC

    @property
    def role(self) -> OrchestratorRole:
        return OrchestratorRole.EXTERNAL_SYNC_LEADER

    async def _create_daemon(self) -> Any:
        try:
            from app.distributed.external_drive_sync import ExternalDriveSyncDaemon

            return ExternalDriveSyncDaemon()
        except ImportError:
            logger.warning("[ExternalDriveSyncAdapter] ExternalDriveSyncDaemon not available")
            return None

    async def _run_daemon(self, daemon: Any) -> None:
        if hasattr(daemon, "run"):
            await daemon.run()
        elif hasattr(daemon, "start"):
            await daemon.start()
        else:
            while self._running:
                await asyncio.sleep(self.config.poll_interval_seconds)


class VastCpuPipelineAdapter(DaemonAdapter):
    """Adapter for Vast.ai CPU pipeline daemon."""

    @property
    def daemon_type(self) -> DaemonType:
        return DaemonType.VAST_CPU_PIPELINE

    @property
    def role(self) -> OrchestratorRole:
        return OrchestratorRole.VAST_PIPELINE_LEADER

    async def _create_daemon(self) -> Any:
        try:
            from app.distributed.vast_cpu_pipeline import VastCpuPipelineDaemon

            return VastCpuPipelineDaemon()
        except ImportError:
            logger.warning("[VastCpuPipelineAdapter] VastCpuPipelineDaemon not available")
            return None

    async def _run_daemon(self, daemon: Any) -> None:
        if hasattr(daemon, "run"):
            await daemon.run()
        elif hasattr(daemon, "start"):
            await daemon.start()
        else:
            while self._running:
                await asyncio.sleep(self.config.poll_interval_seconds)


class ClusterDataSyncAdapter(DaemonAdapter):
    """Adapter for cluster-wide data sync daemon.

    Ensures game databases are synchronized to all cluster nodes with
    adequate storage, excluding development machines.
    """

    @property
    def daemon_type(self) -> DaemonType:
        return DaemonType.CLUSTER_DATA_SYNC

    @property
    def role(self) -> OrchestratorRole:
        return OrchestratorRole.CLUSTER_DATA_SYNC_LEADER

    async def _create_daemon(self) -> Any:
        try:
            from app.coordination.cluster_data_sync import ClusterDataSyncDaemon

            return ClusterDataSyncDaemon()
        except ImportError:
            logger.warning("[ClusterDataSyncAdapter] ClusterDataSyncDaemon not available")
            return None

    async def _run_daemon(self, daemon: Any) -> None:
        if hasattr(daemon, "run"):
            await daemon.run()
        elif hasattr(daemon, "start"):
            await daemon.start()
        else:
            while self._running:
                await asyncio.sleep(self.config.poll_interval_seconds)

    async def _health_check(self) -> bool:
        """Check if sync daemon is healthy."""
        if not self._daemon_instance:
            return False
        # Check if it's running and has synced recently (within 2 intervals)
        from app.coordination.cluster_data_sync import SYNC_INTERVAL_SECONDS
        stats = self._daemon_instance.stats
        if not stats.get("running"):
            return False
        last_sync = stats.get("last_sync_time", 0)
        if time.time() - last_sync > SYNC_INTERVAL_SECONDS * 2:
            return False
        return True


# =============================================================================
# Adapter Registry
# =============================================================================

_ADAPTER_CLASSES: dict[DaemonType, type[DaemonAdapter]] = {
    DaemonType.DISTILLATION: DistillationDaemonAdapter,
    DaemonType.UNIFIED_PROMOTION: PromotionDaemonAdapter,
    DaemonType.EXTERNAL_DRIVE_SYNC: ExternalDriveSyncAdapter,
    DaemonType.VAST_CPU_PIPELINE: VastCpuPipelineAdapter,
    DaemonType.CLUSTER_DATA_SYNC: ClusterDataSyncAdapter,
}


def get_daemon_adapter(
    daemon_type: DaemonType,
    config: DaemonAdapterConfig | None = None,
) -> DaemonAdapter | None:
    """Get an adapter for a daemon type.

    Args:
        daemon_type: The type of daemon
        config: Optional adapter configuration

    Returns:
        DaemonAdapter instance or None if not available
    """
    adapter_class = _ADAPTER_CLASSES.get(daemon_type)
    if adapter_class:
        return adapter_class(config)
    return None


def register_adapter_class(
    daemon_type: DaemonType,
    adapter_class: type[DaemonAdapter],
) -> None:
    """Register a custom adapter class for a daemon type.

    Args:
        daemon_type: The daemon type
        adapter_class: The adapter class to register
    """
    _ADAPTER_CLASSES[daemon_type] = adapter_class


def get_available_adapters() -> list[DaemonType]:
    """Get list of daemon types with available adapters."""
    return list(_ADAPTER_CLASSES.keys())


def register_all_adapters_with_manager() -> dict[DaemonType, bool]:
    """Register all available adapters with the DaemonManager.

    Returns:
        Dict mapping DaemonType to registration success
    """
    from .daemon_manager import get_daemon_manager

    manager = get_daemon_manager()
    results: dict[DaemonType, bool] = {}

    for daemon_type, adapter_class in _ADAPTER_CLASSES.items():
        try:
            adapter = adapter_class()
            manager.register_factory(
                daemon_type,
                adapter.run,
                depends_on=adapter.depends_on,
                health_check_interval=adapter.config.health_check_interval,
                auto_restart=adapter.config.auto_restart,
                max_restarts=adapter.config.max_restarts,
            )
            results[daemon_type] = True
            logger.info(f"Registered adapter for {daemon_type.value}")
        except Exception as e:
            logger.error(f"Failed to register adapter for {daemon_type.value}: {e}")
            results[daemon_type] = False

    return results


__all__ = [
    "DaemonAdapter",
    "DaemonAdapterConfig",
    "DistillationDaemonAdapter",
    "ExternalDriveSyncAdapter",
    "PromotionDaemonAdapter",
    "VastCpuPipelineAdapter",
    "get_available_adapters",
    "get_daemon_adapter",
    "register_adapter_class",
    "register_all_adapters_with_manager",
]
