"""Unified Daemon Manager - Coordinates lifecycle of all background services.

Provides centralized management for all daemons and background services:
- Sync daemons (data, model, elo)
- Health check services
- Event watchers
- Background pipelines

Features:
- Unified start/stop lifecycle
- Health monitoring with auto-restart
- Graceful shutdown handling
- Integration with OrchestratorRegistry
- Status reporting

Usage:
    from app.coordination.daemon_manager import (
        DaemonManager,
        get_daemon_manager,
        DaemonType,
    )

    # Get the singleton manager
    manager = get_daemon_manager()

    # Start all daemons
    await manager.start_all()

    # Start specific daemon
    await manager.start(DaemonType.SYNC_COORDINATOR)

    # Get status
    status = manager.get_status()

    # Graceful shutdown
    await manager.shutdown()
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import logging
import signal
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class DaemonType(Enum):
    """Types of daemons that can be managed."""
    # Sync daemons
    SYNC_COORDINATOR = "sync_coordinator"
    HIGH_QUALITY_SYNC = "high_quality_sync"
    ELO_SYNC = "elo_sync"
    MODEL_SYNC = "model_sync"

    # Health/monitoring
    HEALTH_CHECK = "health_check"
    CLUSTER_MONITOR = "cluster_monitor"
    QUEUE_MONITOR = "queue_monitor"

    # Event processing
    EVENT_ROUTER = "event_router"
    CROSS_PROCESS_POLLER = "cross_process_poller"

    # Pipeline daemons
    DATA_PIPELINE = "data_pipeline"
    TRAINING_WATCHER = "training_watcher"
    SELFPLAY_COORDINATOR = "selfplay_coordinator"

    # P2P services
    P2P_BACKEND = "p2p_backend"
    GOSSIP_SYNC = "gossip_sync"
    DATA_SERVER = "data_server"


class DaemonState(Enum):
    """State of a daemon."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"
    RESTARTING = "restarting"


@dataclass
class DaemonInfo:
    """Information about a registered daemon."""
    daemon_type: DaemonType
    state: DaemonState = DaemonState.STOPPED
    task: asyncio.Task | None = None
    start_time: float = 0.0
    restart_count: int = 0
    last_error: str | None = None
    health_check_interval: float = 60.0
    auto_restart: bool = True
    max_restarts: int = 5
    restart_delay: float = 5.0

    # Dependencies
    depends_on: list[DaemonType] = field(default_factory=list)

    @property
    def uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        if self.state == DaemonState.RUNNING and self.start_time > 0:
            return time.time() - self.start_time
        return 0.0


@dataclass
class DaemonManagerConfig:
    """Configuration for DaemonManager."""
    auto_start: bool = False  # Auto-start all daemons on init
    health_check_interval: float = 30.0  # Global health check interval
    shutdown_timeout: float = 10.0  # Max time to wait for graceful shutdown
    auto_restart_failed: bool = True  # Auto-restart failed daemons
    max_restart_attempts: int = 5  # Max restart attempts per daemon


class DaemonManager:
    """Unified manager for all background daemons and services.

    Provides centralized lifecycle management, health monitoring, and
    coordinated shutdown for all background services.
    """

    _instance: DaemonManager | None = None

    def __init__(self, config: DaemonManagerConfig | None = None):
        self.config = config or DaemonManagerConfig()
        self._daemons: dict[DaemonType, DaemonInfo] = {}
        self._factories: dict[DaemonType, Callable[[], Coroutine[Any, Any, None]]] = {}
        self._running = False
        self._health_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()
        self._lock = asyncio.Lock()

        # Register cleanup
        atexit.register(self._sync_shutdown)

        # Register default factories
        self._register_default_factories()

    @classmethod
    def get_instance(cls, config: DaemonManagerConfig | None = None) -> DaemonManager:
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        if cls._instance is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(cls._instance.shutdown())
            except RuntimeError:
                cls._instance._sync_shutdown()
        cls._instance = None

    def _register_default_factories(self) -> None:
        """Register default daemon factories."""
        # Sync daemons
        self.register_factory(DaemonType.SYNC_COORDINATOR, self._create_sync_coordinator)
        self.register_factory(DaemonType.HIGH_QUALITY_SYNC, self._create_high_quality_sync)
        self.register_factory(DaemonType.ELO_SYNC, self._create_elo_sync)

        # Event processing
        self.register_factory(DaemonType.EVENT_ROUTER, self._create_event_router)
        self.register_factory(DaemonType.CROSS_PROCESS_POLLER, self._create_cross_process_poller)

        # Health monitoring
        self.register_factory(DaemonType.HEALTH_CHECK, self._create_health_check)
        self.register_factory(DaemonType.QUEUE_MONITOR, self._create_queue_monitor)

        # P2P services
        self.register_factory(DaemonType.GOSSIP_SYNC, self._create_gossip_sync)
        self.register_factory(DaemonType.DATA_SERVER, self._create_data_server)

    def register_factory(
        self,
        daemon_type: DaemonType,
        factory: Callable[[], Coroutine[Any, Any, None]],
        depends_on: list[DaemonType] | None = None,
        health_check_interval: float = 60.0,
        auto_restart: bool = True,
        max_restarts: int = 5,
    ) -> None:
        """Register a factory function for creating a daemon.

        Args:
            daemon_type: Type of daemon
            factory: Async function that runs the daemon
            depends_on: List of daemons that must be running first
            health_check_interval: Health check interval for this daemon
            auto_restart: Whether to auto-restart on failure
            max_restarts: Maximum restart attempts
        """
        self._factories[daemon_type] = factory
        self._daemons[daemon_type] = DaemonInfo(
            daemon_type=daemon_type,
            depends_on=depends_on or [],
            health_check_interval=health_check_interval,
            auto_restart=auto_restart,
            max_restarts=max_restarts,
        )

    async def start(self, daemon_type: DaemonType) -> bool:
        """Start a specific daemon.

        Args:
            daemon_type: Type of daemon to start

        Returns:
            True if started successfully
        """
        async with self._lock:
            info = self._daemons.get(daemon_type)
            if info is None:
                logger.error(f"Unknown daemon type: {daemon_type}")
                return False

            if info.state == DaemonState.RUNNING:
                logger.debug(f"{daemon_type.value} already running")
                return True

            # Check dependencies
            for dep in info.depends_on:
                dep_info = self._daemons.get(dep)
                if dep_info is None or dep_info.state != DaemonState.RUNNING:
                    logger.warning(f"Cannot start {daemon_type.value}: dependency {dep.value} not running")
                    return False

            # Get factory
            factory = self._factories.get(daemon_type)
            if factory is None:
                logger.error(f"No factory registered for {daemon_type.value}")
                return False

            # Start daemon
            info.state = DaemonState.STARTING
            try:
                info.task = asyncio.create_task(
                    self._run_daemon(daemon_type, factory),
                    name=f"daemon_{daemon_type.value}"
                )
                info.start_time = time.time()
                info.state = DaemonState.RUNNING
                logger.info(f"Started daemon: {daemon_type.value}")
                return True

            except Exception as e:
                info.state = DaemonState.FAILED
                info.last_error = str(e)
                logger.error(f"Failed to start {daemon_type.value}: {e}")
                return False

    async def _run_daemon(
        self,
        daemon_type: DaemonType,
        factory: Callable[[], Coroutine[Any, Any, None]]
    ) -> None:
        """Run a daemon with error handling and restart logic."""
        info = self._daemons[daemon_type]

        while not self._shutdown_event.is_set():
            try:
                await factory()
            except asyncio.CancelledError:
                logger.debug(f"{daemon_type.value} cancelled")
                break
            except Exception as e:
                info.last_error = str(e)
                logger.error(f"{daemon_type.value} failed: {e}")

                if not info.auto_restart:
                    info.state = DaemonState.FAILED
                    break

                if info.restart_count >= info.max_restarts:
                    logger.error(f"{daemon_type.value} exceeded max restarts, stopping")
                    info.state = DaemonState.FAILED
                    break

                # Restart
                info.restart_count += 1
                info.state = DaemonState.RESTARTING
                logger.info(f"Restarting {daemon_type.value} (attempt {info.restart_count})")
                await asyncio.sleep(info.restart_delay)
                info.state = DaemonState.RUNNING
                info.start_time = time.time()

        info.state = DaemonState.STOPPED

    async def stop(self, daemon_type: DaemonType) -> bool:
        """Stop a specific daemon.

        Args:
            daemon_type: Type of daemon to stop

        Returns:
            True if stopped successfully
        """
        async with self._lock:
            info = self._daemons.get(daemon_type)
            if info is None:
                return False

            if info.state == DaemonState.STOPPED:
                return True

            info.state = DaemonState.STOPPING

            if info.task is not None:
                info.task.cancel()
                try:
                    await asyncio.wait_for(info.task, timeout=self.config.shutdown_timeout)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout stopping {daemon_type.value}")
                except asyncio.CancelledError:
                    pass

            info.state = DaemonState.STOPPED
            info.task = None
            logger.info(f"Stopped daemon: {daemon_type.value}")
            return True

    async def start_all(self, types: list[DaemonType] | None = None) -> dict[DaemonType, bool]:
        """Start all (or specified) daemons in dependency order.

        Args:
            types: Specific daemon types to start (all if None)

        Returns:
            Dict mapping daemon type to start success
        """
        results: dict[DaemonType, bool] = {}
        types_to_start = types or list(self._factories.keys())

        # Sort by dependencies (topological sort)
        sorted_types = self._sort_by_dependencies(types_to_start)

        for daemon_type in sorted_types:
            results[daemon_type] = await self.start(daemon_type)

        # Start health check loop
        if not self._health_task or self._health_task.done():
            self._running = True
            self._health_task = asyncio.create_task(self._health_loop())

        return results

    async def stop_all(self) -> dict[DaemonType, bool]:
        """Stop all running daemons.

        Returns:
            Dict mapping daemon type to stop success
        """
        self._running = False
        results: dict[DaemonType, bool] = {}

        # Stop in reverse dependency order
        sorted_types = list(reversed(self._sort_by_dependencies(list(self._daemons.keys()))))

        for daemon_type in sorted_types:
            results[daemon_type] = await self.stop(daemon_type)

        return results

    async def shutdown(self) -> None:
        """Gracefully shutdown all daemons."""
        logger.info("DaemonManager shutting down...")
        self._shutdown_event.set()
        self._running = False

        # Stop health check
        if self._health_task:
            self._health_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_task

        # Stop all daemons
        await self.stop_all()

        logger.info("DaemonManager shutdown complete")

    def _sync_shutdown(self) -> None:
        """Synchronous shutdown for atexit."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(self.shutdown())
        except RuntimeError:
            # No running loop
            pass

    def _sort_by_dependencies(self, types: list[DaemonType]) -> list[DaemonType]:
        """Sort daemon types by dependencies (topological sort)."""
        result: list[DaemonType] = []
        visited: set[DaemonType] = set()
        visiting: set[DaemonType] = set()

        def visit(dt: DaemonType) -> None:
            if dt in visited:
                return
            if dt in visiting:
                logger.warning(f"Circular dependency detected for {dt.value}")
                return

            visiting.add(dt)
            info = self._daemons.get(dt)
            if info:
                for dep in info.depends_on:
                    if dep in types:
                        visit(dep)

            visiting.remove(dt)
            visited.add(dt)
            result.append(dt)

        for dt in types:
            visit(dt)

        return result

    async def _health_loop(self) -> None:
        """Background health check loop."""
        while self._running and not self._shutdown_event.is_set():
            try:
                await self._check_health()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _check_health(self) -> None:
        """Check health of all running daemons."""
        for daemon_type, info in self._daemons.items():
            if info.state != DaemonState.RUNNING:
                continue

            # Check if task is still alive
            if info.task is None or info.task.done():
                if info.task and info.task.exception():
                    info.last_error = str(info.task.exception())

                if self.config.auto_restart_failed and info.restart_count < info.max_restarts:
                    logger.warning(f"{daemon_type.value} died, restarting...")
                    await self.start(daemon_type)
                else:
                    info.state = DaemonState.FAILED

    def get_status(self) -> dict[str, Any]:
        """Get status of all daemons.

        Returns:
            Status dict with daemon states and metrics
        """
        daemon_status = {}
        for daemon_type, info in self._daemons.items():
            daemon_status[daemon_type.value] = {
                "state": info.state.value,
                "uptime_seconds": info.uptime_seconds,
                "restart_count": info.restart_count,
                "last_error": info.last_error,
                "auto_restart": info.auto_restart,
            }

        running_count = sum(1 for i in self._daemons.values() if i.state == DaemonState.RUNNING)
        failed_count = sum(1 for i in self._daemons.values() if i.state == DaemonState.FAILED)

        return {
            "running": self._running,
            "daemons": daemon_status,
            "summary": {
                "total": len(self._daemons),
                "running": running_count,
                "failed": failed_count,
                "stopped": len(self._daemons) - running_count - failed_count,
            },
        }

    def is_running(self, daemon_type: DaemonType) -> bool:
        """Check if a daemon is running."""
        info = self._daemons.get(daemon_type)
        return info is not None and info.state == DaemonState.RUNNING

    # =========================================================================
    # Default Daemon Factories
    # =========================================================================

    async def _create_sync_coordinator(self) -> None:
        """Create and run the sync coordinator daemon."""
        try:
            from app.distributed.sync_coordinator import SyncCategory, SyncCoordinator

            coordinator = SyncCoordinator.get_instance()
            await coordinator.start_background_sync(
                interval_seconds=300,
                categories=[SyncCategory.GAMES, SyncCategory.MODELS, SyncCategory.TRAINING],
            )
        except ImportError:
            logger.warning("SyncCoordinator not available")
            await asyncio.sleep(float('inf'))  # Sleep forever

    async def _create_high_quality_sync(self) -> None:
        """Create and run high-quality data sync watcher."""
        try:
            from app.distributed.sync_coordinator import wire_all_quality_events_to_sync

            _watcher = wire_all_quality_events_to_sync(
                sync_cooldown_seconds=60.0,
                min_quality_score=0.7,
                max_games_per_sync=500,
            )
            # Keep alive
            while True:
                await asyncio.sleep(3600)
        except ImportError:
            logger.warning("High-quality sync watcher not available")
            await asyncio.sleep(float('inf'))

    async def _create_elo_sync(self) -> None:
        """Create and run ELO sync daemon."""
        try:
            from app.tournament.elo_sync_manager import EloSyncManager

            manager = EloSyncManager.get_instance()
            await manager.start_sync_daemon(interval_seconds=60)
        except ImportError:
            logger.warning("EloSyncManager not available")
            await asyncio.sleep(float('inf'))

    async def _create_event_router(self) -> None:
        """Create and run the unified event router."""
        try:
            from app.coordination.event_router import UnifiedEventRouter

            router = UnifiedEventRouter.get_instance()
            await router.start()
        except ImportError:
            logger.warning("UnifiedEventRouter not available")
            await asyncio.sleep(float('inf'))

    async def _create_cross_process_poller(self) -> None:
        """Create and run cross-process event poller."""
        try:
            from app.coordination.cross_process_events import CrossProcessEventPoller

            poller = CrossProcessEventPoller()
            poller.start()  # start() is sync, runs in background thread
            # Keep this coroutine alive while the poller runs
            await asyncio.sleep(float('inf'))
        except ImportError:
            logger.warning("CrossProcessEventPoller not available")
            await asyncio.sleep(float('inf'))

    async def _create_health_check(self) -> None:
        """Create and run health check daemon."""
        try:
            from app.distributed.health_checks import HealthCheckDaemon

            daemon = HealthCheckDaemon()
            await daemon.start()
        except ImportError:
            logger.warning("HealthCheckDaemon not available")
            await asyncio.sleep(float('inf'))

    async def _create_queue_monitor(self) -> None:
        """Create and run queue monitor."""
        try:
            from app.coordination.queue_monitor import QueueMonitor

            monitor = QueueMonitor()
            await monitor.start()
        except ImportError:
            logger.warning("QueueMonitor not available")
            await asyncio.sleep(float('inf'))

    async def _create_gossip_sync(self) -> None:
        """Create and run gossip sync daemon."""
        try:
            from app.distributed.gossip_sync import GossipSyncDaemon

            daemon = GossipSyncDaemon()
            await daemon.start()
        except ImportError:
            logger.warning("GossipSyncDaemon not available")
            await asyncio.sleep(float('inf'))

    async def _create_data_server(self) -> None:
        """Create and run aria2 data server."""
        try:
            from app.distributed.sync_coordinator import SyncCoordinator

            coordinator = SyncCoordinator.get_instance()
            await coordinator.start_data_server(port=8766)
            # Keep alive
            while coordinator.is_data_server_running():
                await asyncio.sleep(60)
        except ImportError:
            logger.warning("Data server not available")
            await asyncio.sleep(float('inf'))


# Singleton accessor
_daemon_manager: DaemonManager | None = None


def get_daemon_manager(config: DaemonManagerConfig | None = None) -> DaemonManager:
    """Get the singleton DaemonManager instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        DaemonManager instance
    """
    global _daemon_manager
    if _daemon_manager is None:
        _daemon_manager = DaemonManager(config)
    return _daemon_manager


def reset_daemon_manager() -> None:
    """Reset the singleton (for testing)."""
    global _daemon_manager
    if _daemon_manager is not None:
        DaemonManager.reset_instance()
    _daemon_manager = None


# Signal handlers for graceful shutdown
def setup_signal_handlers() -> None:
    """Set up signal handlers for graceful shutdown."""
    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        manager = get_daemon_manager()
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(manager.shutdown())
        except RuntimeError:
            # No running loop
            manager._sync_shutdown()

    try:
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
    except Exception as e:
        logger.debug(f"Could not set up signal handlers: {e}")


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Data classes
    "DaemonInfo",
    # Main class
    "DaemonManager",
    "DaemonManagerConfig",
    "DaemonState",
    # Enums
    "DaemonType",
    # Functions
    "get_daemon_manager",
    "reset_daemon_manager",
    "setup_signal_handlers",
]
