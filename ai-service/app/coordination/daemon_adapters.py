"""Daemon Adapters - Wrappers for integrating daemons with DaemonManager.

This module provides adapters that wrap existing daemon classes to make them
compatible with the centralized DaemonManager lifecycle management.

December 2025 Consolidation:
- Replaced 9 near-identical adapter classes with data-driven ConfigurableDaemonAdapter
- ~450 LOC saved through configuration-based approach
- Legacy adapter classes preserved as thin wrappers for backward compatibility

Each adapter:
1. Implements a consistent interface for DaemonManager
2. Optionally acquires an OrchestratorRole for exclusive execution
3. Handles startup, shutdown, and health checks uniformly
4. Reports status to the DaemonManager

Usage:
    from app.coordination.daemon_adapters import (
        get_daemon_adapter,
        ConfigurableDaemonAdapter,
        DaemonAdapterSpec,
    )

    # Get an adapter for a daemon type (uses ADAPTER_SPECS registry)
    adapter = get_daemon_adapter(DaemonType.DISTILLATION)

    # Or create a custom adapter via spec
    spec = DaemonAdapterSpec(
        daemon_type=DaemonType.MY_DAEMON,
        module_path="app.my_module",
        class_name="MyDaemon",
        role=OrchestratorRole.MY_LEADER,
    )
    adapter = ConfigurableDaemonAdapter(spec)
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from app.core.async_context import safe_create_task

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

    def health_check(self) -> "HealthCheckResult":
        """Check adapter health for CoordinatorProtocol compliance.

        December 2025: Added for unified daemon health monitoring.
        Delegates to wrapped daemon's health_check() if available.
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        # If daemon has its own health_check(), delegate to it
        if self._daemon_instance is not None and hasattr(self._daemon_instance, "health_check"):
            return self._daemon_instance.health_check()

        # Otherwise, check adapter state
        if not self._running:
            return HealthCheckResult(
                healthy=True,  # Stopped is not unhealthy
                status=CoordinatorStatus.STOPPED,
                message=f"{self.__class__.__name__} not running",
            )

        if not self._healthy:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"{self.__class__.__name__} unhealthy (count: {self._unhealthy_count})",
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"{self.__class__.__name__} running",
        )

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
        health_task = safe_create_task(
            self._health_monitor_loop(),
            name="daemon_health_monitor",
        )

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

# =============================================================================
# Configurable Daemon Adapter (December 2025 Consolidation)
# =============================================================================


@dataclass(frozen=True)
class DaemonAdapterSpec:
    """Specification for a daemon adapter.

    Enables data-driven adapter configuration instead of per-daemon subclasses.
    December 2025: Replaces 9 near-identical adapter classes.
    """

    daemon_type: DaemonType
    module_path: str  # e.g., "app.training.distillation_daemon"
    class_name: str  # e.g., "DistillationDaemon"
    role: OrchestratorRole | None = None
    depends_on: tuple[DaemonType, ...] = ()
    deprecated: bool = False
    deprecated_message: str = ""
    # Custom health check: attribute name on daemon that returns bool
    health_check_attr: str | None = None


class ConfigurableDaemonAdapter(DaemonAdapter):
    """Generic daemon adapter configured via DaemonAdapterSpec.

    December 2025: Consolidates 9 near-identical adapter classes into one
    configuration-driven implementation. Saves ~450 LOC.

    Usage:
        spec = ADAPTER_SPECS[DaemonType.DISTILLATION]
        adapter = ConfigurableDaemonAdapter(spec)
        await adapter.run()
    """

    def __init__(
        self,
        spec: DaemonAdapterSpec,
        config: DaemonAdapterConfig | None = None,
    ):
        super().__init__(config)
        self._spec = spec

    @property
    def daemon_type(self) -> DaemonType:
        return self._spec.daemon_type

    @property
    def role(self) -> OrchestratorRole | None:
        return self._spec.role

    @property
    def depends_on(self) -> list[DaemonType]:
        return list(self._spec.depends_on)

    async def _create_daemon(self) -> Any:
        # Emit deprecation warning if applicable
        if self._spec.deprecated:
            warnings.warn(
                self._spec.deprecated_message or f"{self._spec.class_name} is deprecated",
                DeprecationWarning,
                stacklevel=3,
            )

        try:
            module = importlib.import_module(self._spec.module_path)
            daemon_class = getattr(module, self._spec.class_name)
            return daemon_class()
        except ImportError as e:
            logger.warning(
                f"[{self.__class__.__name__}] {self._spec.class_name} not available: {e}"
            )
            return None
        except AttributeError as e:
            logger.warning(
                f"[{self.__class__.__name__}] {self._spec.class_name} not found in "
                f"{self._spec.module_path}: {e}"
            )
            return None

    async def _run_daemon(self, daemon: Any) -> None:
        if hasattr(daemon, "start"):
            await daemon.start()
            # Wait while daemon is running
            while hasattr(daemon, "is_running") and daemon.is_running():
                await asyncio.sleep(self.config.poll_interval_seconds)
        elif hasattr(daemon, "run"):
            await daemon.run()
        else:
            # Fallback: run indefinitely
            while self._running:
                await asyncio.sleep(self.config.poll_interval_seconds)

    async def _health_check(self) -> bool:
        if not self._daemon_instance:
            return False

        # Use custom health check attribute if specified
        if self._spec.health_check_attr:
            attr = getattr(self._daemon_instance, self._spec.health_check_attr, None)
            if callable(attr):
                result = attr()
                return bool(result)
            return bool(attr)

        # Default health checks
        if hasattr(self._daemon_instance, "is_running"):
            return self._daemon_instance.is_running()
        if hasattr(self._daemon_instance, "_running"):
            return self._daemon_instance._running
        return True


# =============================================================================
# Adapter Specifications Registry
# =============================================================================

ADAPTER_SPECS: dict[DaemonType, DaemonAdapterSpec] = {
    DaemonType.DISTILLATION: DaemonAdapterSpec(
        daemon_type=DaemonType.DISTILLATION,
        module_path="app.training.distillation_daemon",
        class_name="DistillationDaemon",
        role=OrchestratorRole.DISTILLATION_LEADER,
    ),
    DaemonType.UNIFIED_PROMOTION: DaemonAdapterSpec(
        daemon_type=DaemonType.UNIFIED_PROMOTION,
        module_path="app.training.unified_promotion_daemon",
        class_name="UnifiedPromotionDaemon",
        role=OrchestratorRole.PROMOTION_LEADER,
    ),
    DaemonType.EXTERNAL_DRIVE_SYNC: DaemonAdapterSpec(
        daemon_type=DaemonType.EXTERNAL_DRIVE_SYNC,
        module_path="app.distributed.external_drive_sync",
        class_name="ExternalDriveSyncDaemon",
        role=OrchestratorRole.EXTERNAL_SYNC_LEADER,
    ),
    DaemonType.VAST_CPU_PIPELINE: DaemonAdapterSpec(
        daemon_type=DaemonType.VAST_CPU_PIPELINE,
        module_path="app.distributed.vast_cpu_pipeline",
        class_name="VastCpuPipelineDaemon",
        role=OrchestratorRole.VAST_PIPELINE_LEADER,
    ),
    DaemonType.CLUSTER_DATA_SYNC: DaemonAdapterSpec(
        daemon_type=DaemonType.CLUSTER_DATA_SYNC,
        module_path="app.coordination.auto_sync_daemon",
        class_name="AutoSyncDaemon",
        role=OrchestratorRole.CLUSTER_DATA_SYNC_LEADER,
    ),
    DaemonType.AUTO_SYNC: DaemonAdapterSpec(
        daemon_type=DaemonType.AUTO_SYNC,
        module_path="app.coordination.auto_sync_daemon",
        class_name="AutoSyncDaemon",
        role=None,  # Runs on all nodes
    ),
    DaemonType.NPZ_DISTRIBUTION: DaemonAdapterSpec(
        daemon_type=DaemonType.NPZ_DISTRIBUTION,
        module_path="app.coordination.unified_distribution_daemon",
        class_name="UnifiedDistributionDaemon",
        role=None,
        deprecated=True,
        deprecated_message=(
            "NPZ_DISTRIBUTION is deprecated. Use MODEL_DISTRIBUTION with "
            "UnifiedDistributionDaemon instead."
        ),
    ),
    DaemonType.ORPHAN_DETECTION: DaemonAdapterSpec(
        daemon_type=DaemonType.ORPHAN_DETECTION,
        module_path="app.coordination.orphan_detection_daemon",
        class_name="OrphanDetectionDaemon",
        role=None,
    ),
    DaemonType.DATA_CLEANUP: DaemonAdapterSpec(
        daemon_type=DaemonType.DATA_CLEANUP,
        module_path="app.coordination.data_cleanup_daemon",
        class_name="DataCleanupDaemon",
        role=None,
    ),
}


# =============================================================================
# Legacy Adapter Classes (Backward Compatibility)
# =============================================================================
# These classes are preserved for backward compatibility with existing code
# that imports specific adapter classes. They are thin wrappers around
# ConfigurableDaemonAdapter.


def _create_legacy_adapter(daemon_type: DaemonType) -> type[DaemonAdapter]:
    """Create a legacy adapter class for backward compatibility."""
    spec = ADAPTER_SPECS.get(daemon_type)
    if not spec:
        raise ValueError(f"No spec found for {daemon_type}")

    class LegacyAdapter(ConfigurableDaemonAdapter):
        """Backward-compatible daemon adapter wrapping a DaemonType specification.

        Generated by _create_legacy_adapter for existing code that uses the
        old XxxDaemonAdapter naming convention.
        """

        def __init__(self, config: DaemonAdapterConfig | None = None):
            super().__init__(spec, config)

    return LegacyAdapter


# Legacy classes - thin wrappers for backward compatibility
DistillationDaemonAdapter = _create_legacy_adapter(DaemonType.DISTILLATION)
PromotionDaemonAdapter = _create_legacy_adapter(DaemonType.UNIFIED_PROMOTION)
ExternalDriveSyncAdapter = _create_legacy_adapter(DaemonType.EXTERNAL_DRIVE_SYNC)
VastCpuPipelineAdapter = _create_legacy_adapter(DaemonType.VAST_CPU_PIPELINE)
ClusterDataSyncAdapter = _create_legacy_adapter(DaemonType.CLUSTER_DATA_SYNC)
AutoSyncDaemonAdapter = _create_legacy_adapter(DaemonType.AUTO_SYNC)
NPZDistributionDaemonAdapter = _create_legacy_adapter(DaemonType.NPZ_DISTRIBUTION)
OrphanDetectionDaemonAdapter = _create_legacy_adapter(DaemonType.ORPHAN_DETECTION)
DataCleanupDaemonAdapter = _create_legacy_adapter(DaemonType.DATA_CLEANUP)


# =============================================================================
# Adapter Registry (uses legacy adapter classes for compatibility)
# =============================================================================

_ADAPTER_CLASSES: dict[DaemonType, type[DaemonAdapter]] = {
    DaemonType.DISTILLATION: DistillationDaemonAdapter,
    DaemonType.UNIFIED_PROMOTION: PromotionDaemonAdapter,
    DaemonType.EXTERNAL_DRIVE_SYNC: ExternalDriveSyncAdapter,
    DaemonType.VAST_CPU_PIPELINE: VastCpuPipelineAdapter,
    DaemonType.CLUSTER_DATA_SYNC: ClusterDataSyncAdapter,
    DaemonType.AUTO_SYNC: AutoSyncDaemonAdapter,
    DaemonType.NPZ_DISTRIBUTION: NPZDistributionDaemonAdapter,
    DaemonType.ORPHAN_DETECTION: OrphanDetectionDaemonAdapter,
    DaemonType.DATA_CLEANUP: DataCleanupDaemonAdapter,
}


def get_daemon_adapter(
    daemon_type: DaemonType,
    config: DaemonAdapterConfig | None = None,
) -> DaemonAdapter | None:
    """Get an adapter for a daemon type.

    December 2025: Now uses ConfigurableDaemonAdapter via ADAPTER_SPECS.

    Args:
        daemon_type: The type of daemon
        config: Optional adapter configuration

    Returns:
        DaemonAdapter instance or None if not available
    """
    # First check ADAPTER_SPECS for data-driven adapters
    spec = ADAPTER_SPECS.get(daemon_type)
    if spec:
        return ConfigurableDaemonAdapter(spec, config)

    # Fallback to legacy registry
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


def register_adapter_spec(spec: DaemonAdapterSpec) -> None:
    """Register a daemon adapter specification.

    December 2025: Preferred method for adding new adapters.

    Args:
        spec: The adapter specification
    """
    ADAPTER_SPECS[spec.daemon_type] = spec


def get_available_adapters() -> list[DaemonType]:
    """Get list of daemon types with available adapters."""
    return list(set(_ADAPTER_CLASSES.keys()) | set(ADAPTER_SPECS.keys()))


def register_all_adapters_with_manager() -> dict[DaemonType, bool]:
    """Register all available adapters with the DaemonManager.

    Returns:
        Dict mapping DaemonType to registration success
    """
    from .daemon_manager import get_daemon_manager

    manager = get_daemon_manager()
    results: dict[DaemonType, bool] = {}

    # Register from ADAPTER_SPECS (preferred)
    for daemon_type, spec in ADAPTER_SPECS.items():
        try:
            adapter = ConfigurableDaemonAdapter(spec)
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
    # Core classes
    "DaemonAdapter",
    "DaemonAdapterConfig",
    "DaemonAdapterSpec",
    "ConfigurableDaemonAdapter",
    # Legacy adapter classes (backward compatibility)
    "AutoSyncDaemonAdapter",
    "ClusterDataSyncAdapter",
    "DataCleanupDaemonAdapter",
    "DistillationDaemonAdapter",
    "ExternalDriveSyncAdapter",
    "NPZDistributionDaemonAdapter",
    "OrphanDetectionDaemonAdapter",
    "PromotionDaemonAdapter",
    "VastCpuPipelineAdapter",
    # Registry and functions
    "ADAPTER_SPECS",
    "get_available_adapters",
    "get_daemon_adapter",
    "register_adapter_class",
    "register_adapter_spec",
    "register_all_adapters_with_manager",
]
