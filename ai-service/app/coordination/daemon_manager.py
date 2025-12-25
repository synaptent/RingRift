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

from app.core.async_context import fire_and_forget, safe_create_task

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
    NODE_HEALTH_MONITOR = "node_health_monitor"

    # Event processing
    EVENT_ROUTER = "event_router"
    CROSS_PROCESS_POLLER = "cross_process_poller"
    DLQ_RETRY = "dlq_retry"

    # Pipeline daemons
    DATA_PIPELINE = "data_pipeline"
    TRAINING_WATCHER = "training_watcher"
    SELFPLAY_COORDINATOR = "selfplay_coordinator"

    # P2P services
    P2P_BACKEND = "p2p_backend"
    GOSSIP_SYNC = "gossip_sync"
    DATA_SERVER = "data_server"

    # Training enhancement daemons (December 2025)
    DISTILLATION = "distillation"
    UNIFIED_PROMOTION = "unified_promotion"
    EXTERNAL_DRIVE_SYNC = "external_drive_sync"
    VAST_CPU_PIPELINE = "vast_cpu_pipeline"

    # Continuous training loop (December 2025)
    CONTINUOUS_TRAINING_LOOP = "continuous_training_loop"

    # Cluster-wide data sync (December 2025)
    CLUSTER_DATA_SYNC = "cluster_data_sync"

    # Model distribution (December 2025) - auto-distribute models after promotion
    MODEL_DISTRIBUTION = "model_distribution"

    # Automated P2P data sync (December 2025)
    AUTO_SYNC = "auto_sync"

    # Training node watcher (December 2025 - Phase 6)
    TRAINING_NODE_WATCHER = "training_node_watcher"

    # Ephemeral sync for Vast.ai (December 2025 - Phase 4)
    EPHEMERAL_SYNC = "ephemeral_sync"

    # P2P auto-deployment (December 2025) - ensure P2P runs on all nodes
    P2P_AUTO_DEPLOY = "p2p_auto_deploy"

    # Replication monitor (December 2025) - monitor data replication health
    REPLICATION_MONITOR = "replication_monitor"

    # Tournament daemon (December 2025) - automatic tournament scheduling
    TOURNAMENT_DAEMON = "tournament_daemon"


class DaemonState(Enum):
    """State of a daemon."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"
    RESTARTING = "restarting"
    IMPORT_FAILED = "import_failed"  # Permanent failure due to missing imports


# Constants for recovery behavior
DAEMON_RESTART_RESET_AFTER = 3600  # Reset restart count after 1 hour of stability
MAX_RESTART_DELAY = 300  # Cap exponential backoff at 5 minutes


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

    # Stability tracking for restart count reset
    stable_since: float = 0.0  # When daemon became stable (no errors)
    last_failure_time: float = 0.0  # When the last failure occurred

    # Import error tracking
    import_error: str | None = None  # Specific import error message

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
    recovery_cooldown: float = 300.0  # Time before attempting to recover FAILED daemons


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
                fire_and_forget(
                    cls._instance.shutdown(),
                    name="daemon_manager_reset_shutdown",
                )
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

        # Continuous training
        self.register_factory(DaemonType.CONTINUOUS_TRAINING_LOOP, self._create_continuous_training_loop)

        # Auto sync (December 2025)
        self.register_factory(DaemonType.AUTO_SYNC, self._create_auto_sync)

        # Training node watcher (Phase 6, December 2025)
        self.register_factory(DaemonType.TRAINING_NODE_WATCHER, self._create_training_node_watcher)

        # Ephemeral sync for Vast.ai (Phase 4, December 2025)
        self.register_factory(DaemonType.EPHEMERAL_SYNC, self._create_ephemeral_sync)

        # Replication monitor (December 2025)
        self.register_factory(DaemonType.REPLICATION_MONITOR, self._create_replication_monitor)

        # Tournament daemon (December 2025)
        self.register_factory(DaemonType.TOURNAMENT_DAEMON, self._create_tournament_daemon)

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
                info.task = safe_create_task(
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
        """Run a daemon with error handling and restart logic.

        Features:
        - Import errors are treated as permanent failures (IMPORT_FAILED state)
        - Restart count resets after DAEMON_RESTART_RESET_AFTER seconds of stability
        - Exponential backoff for restart delays (capped at MAX_RESTART_DELAY)
        """
        info = self._daemons[daemon_type]

        while not self._shutdown_event.is_set():
            # Reset restart counter after period of stability
            if info.last_failure_time > 0:
                time_since_failure = time.time() - info.last_failure_time
                if time_since_failure > DAEMON_RESTART_RESET_AFTER:
                    if info.restart_count > 0:
                        logger.info(
                            f"{daemon_type.value} stable for {DAEMON_RESTART_RESET_AFTER}s, "
                            f"resetting restart count from {info.restart_count} to 0"
                        )
                        info.restart_count = 0
                        info.last_failure_time = 0.0

            try:
                info.stable_since = time.time()  # Mark start of stable period
                await factory()
            except asyncio.CancelledError:
                logger.debug(f"{daemon_type.value} cancelled")
                break
            except ImportError as e:
                # Import errors are permanent - require code/environment fix
                info.last_error = str(e)
                info.import_error = str(e)
                info.state = DaemonState.IMPORT_FAILED
                info.last_failure_time = time.time()
                logger.error(
                    f"{daemon_type.value} import failed permanently: {e}. "
                    f"Fix the import and restart the daemon manager."
                )
                # Don't retry import failures - they need manual intervention
                break
            except Exception as e:
                info.last_error = str(e)
                info.last_failure_time = time.time()
                info.stable_since = 0.0
                logger.error(f"{daemon_type.value} failed: {e}")

                if not info.auto_restart:
                    info.state = DaemonState.FAILED
                    break

                if info.restart_count >= info.max_restarts:
                    logger.error(f"{daemon_type.value} exceeded max restarts, stopping")
                    info.state = DaemonState.FAILED
                    break

                # Restart with exponential backoff
                info.restart_count += 1
                info.state = DaemonState.RESTARTING
                delay = min(info.restart_delay * (2 ** (info.restart_count - 1)), MAX_RESTART_DELAY)
                logger.info(f"Restarting {daemon_type.value} (attempt {info.restart_count}) in {delay:.1f}s")
                await asyncio.sleep(delay)
                info.state = DaemonState.RUNNING
                info.start_time = time.time()

        if info.state not in (DaemonState.FAILED, DaemonState.IMPORT_FAILED):
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

    async def restart_failed_daemon(
        self,
        daemon_type: DaemonType,
        force: bool = False,
    ) -> bool:
        """Restart a failed daemon, optionally resetting its restart count.

        This method allows manual recovery of daemons that have exceeded their
        restart limit or have import errors.

        Args:
            daemon_type: Type of daemon to restart
            force: If True, reset restart count and clear import error

        Returns:
            True if restart initiated successfully
        """
        async with self._lock:
            info = self._daemons.get(daemon_type)
            if info is None:
                logger.error(f"Unknown daemon type: {daemon_type}")
                return False

            if info.state not in (DaemonState.FAILED, DaemonState.IMPORT_FAILED, DaemonState.STOPPED):
                logger.warning(
                    f"Cannot restart {daemon_type.value}: state is {info.state.value}, "
                    f"expected FAILED, IMPORT_FAILED, or STOPPED"
                )
                return False

            # Import failures need force=True to retry
            if info.state == DaemonState.IMPORT_FAILED and not force:
                logger.warning(
                    f"{daemon_type.value} has import error: {info.import_error}. "
                    f"Use force=True to retry after fixing the import."
                )
                return False

            if force:
                logger.info(f"Force restarting {daemon_type.value}, resetting counters")
                info.restart_count = 0
                info.import_error = None
                info.last_error = None
                info.last_failure_time = 0.0

            logger.info(f"Restarting failed daemon: {daemon_type.value}")

        # Release lock before starting (start() acquires its own lock)
        return await self.start(daemon_type)

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
            self._health_task = safe_create_task(
                self._health_loop(),
                name="daemon_health_loop"
            )

        # Start daemon watchdog for active monitoring
        try:
            from app.coordination.daemon_watchdog import start_watchdog
            safe_create_task(start_watchdog(), name="daemon_watchdog")
            logger.info("Daemon watchdog started")
        except Exception as e:
            logger.warning(f"Failed to start daemon watchdog: {e}")

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

        # Stop watchdog first
        try:
            from app.coordination.daemon_watchdog import stop_watchdog
            await stop_watchdog()
        except Exception as e:
            logger.debug(f"Watchdog stop error (expected if not started): {e}")

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
                fire_and_forget(
                    self.shutdown(),
                    name="daemon_manager_atexit_shutdown",
                )
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
        """Check health of all daemons and attempt recovery of FAILED ones."""
        current_time = time.time()

        for daemon_type, info in list(self._daemons.items()):
            # Attempt recovery of FAILED daemons after cooldown period
            if info.state == DaemonState.FAILED:
                # Skip daemons with import errors - they can't be recovered
                if info.import_error:
                    continue

                time_since_failure = current_time - info.last_failure_time
                if time_since_failure >= self.config.recovery_cooldown:
                    logger.info(
                        f"Attempting recovery of {daemon_type.value} after "
                        f"{time_since_failure:.0f}s cooldown"
                    )
                    # Reset restart count to allow recovery attempts
                    info.restart_count = 0
                    info.state = DaemonState.STOPPED  # Reset state before restart
                    await self.start(daemon_type)
                continue

            if info.state != DaemonState.RUNNING:
                continue

            # Check if task is still alive
            if info.task is None or info.task.done():
                if info.task and info.task.exception():
                    info.last_error = str(info.task.exception())
                    info.last_failure_time = current_time

                if self.config.auto_restart_failed and info.restart_count < info.max_restarts:
                    logger.warning(f"{daemon_type.value} died, restarting...")
                    await self.start(daemon_type)
                else:
                    info.state = DaemonState.FAILED
                    info.last_failure_time = current_time

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
        except ImportError as e:
            logger.error(f"SyncCoordinator not available: {e}")
            raise  # Propagate error so DaemonManager marks as FAILED

    async def _create_high_quality_sync(self) -> None:
        """Create and run high-quality data sync watcher."""
        try:
            from app.distributed.sync_coordinator import wire_all_quality_events_to_sync

            wire_all_quality_events_to_sync(
                sync_cooldown_seconds=60.0,
                min_quality_score=0.7,
                max_games_per_sync=500,
            )
            # Keep daemon alive
            while True:
                await asyncio.sleep(3600)
        except ImportError as e:
            logger.error(f"High-quality sync watcher not available: {e}")
            raise  # Propagate error so DaemonManager marks as FAILED

    async def _create_elo_sync(self) -> None:
        """Create and run ELO sync daemon."""
        try:
            from app.tournament.elo_sync_manager import EloSyncManager

            manager = EloSyncManager.get_instance()
            await manager.start_sync_daemon(interval_seconds=60)
        except ImportError as e:
            logger.error(f"EloSyncManager not available: {e}")
            raise  # Propagate error so DaemonManager marks as FAILED

    async def _create_event_router(self) -> None:
        """Create and run the unified event router."""
        try:
            from app.coordination.event_router import UnifiedEventRouter

            router = UnifiedEventRouter.get_instance()
            await router.start()
        except ImportError as e:
            logger.error(f"UnifiedEventRouter not available: {e}")
            raise  # Propagate error so DaemonManager marks as FAILED

    async def _create_cross_process_poller(self) -> None:
        """Create and run cross-process event poller."""
        try:
            from app.coordination.event_router import CrossProcessEventPoller

            poller = CrossProcessEventPoller()
            poller.start()  # start() is sync, runs in background thread
            # Keep this coroutine alive while the poller runs
            while True:
                await asyncio.sleep(3600)  # Check every hour
        except ImportError as e:
            logger.error(f"CrossProcessEventPoller not available: {e}")
            raise  # Propagate error so DaemonManager marks as FAILED

    async def _create_health_check(self) -> None:
        """Create and run health check daemon."""
        try:
            from app.distributed.health_checks import HealthCheckDaemon

            daemon = HealthCheckDaemon()
            await daemon.start()
        except ImportError as e:
            logger.error(f"HealthCheckDaemon not available: {e}")
            raise  # Propagate error so DaemonManager marks as FAILED

    async def _create_queue_monitor(self) -> None:
        """Create and run queue monitor."""
        try:
            from app.coordination.queue_monitor import QueueMonitor

            monitor = QueueMonitor()
            await monitor.start()
        except ImportError as e:
            logger.error(f"QueueMonitor not available: {e}")
            raise  # Propagate error so DaemonManager marks as FAILED

    async def _create_gossip_sync(self) -> None:
        """Create and run gossip sync daemon."""
        try:
            from app.distributed.gossip_sync import GossipSyncDaemon

            daemon = GossipSyncDaemon()
            await daemon.start()
        except ImportError as e:
            logger.error(f"GossipSyncDaemon not available: {e}")
            raise  # Propagate error so DaemonManager marks as FAILED

    async def _create_data_server(self) -> None:
        """Create and run aria2 data server."""
        try:
            from app.distributed.sync_coordinator import SyncCoordinator

            coordinator = SyncCoordinator.get_instance()
            await coordinator.start_data_server(port=8766)
            # Keep alive
            while coordinator.is_data_server_running():
                await asyncio.sleep(60)
        except ImportError as e:
            logger.error(f"Data server not available: {e}")
            raise  # Propagate error so DaemonManager marks as FAILED

    async def _create_continuous_training_loop(self) -> None:
        """Create and run continuous training loop daemon."""
        try:
            from app.coordination.continuous_loop import (
                ContinuousTrainingLoop,
                LoopConfig,
            )

            # Default config - can be customized via environment or config file
            config = LoopConfig(
                configs=[("hex8", 2), ("square8", 2)],
                selfplay_games_per_iteration=1000,
                selfplay_engine="gumbel-mcts",
                max_iterations=0,  # Infinite
            )

            loop = ContinuousTrainingLoop(config)
            await loop.start()

            # Wait for loop to complete (or be stopped)
            while loop._running:
                await asyncio.sleep(10)

        except ImportError as e:
            logger.error(f"ContinuousTrainingLoop not available: {e}")
            raise  # Propagate error so DaemonManager marks as FAILED

    async def _create_auto_sync(self) -> None:
        """Create and run automated P2P data sync daemon (December 2025).

        This daemon orchestrates data synchronization across the cluster using:
        - Layer 1: Push-from-generator (immediate push to neighbors)
        - Layer 2: P2P gossip replication (eventual consistency)

        Excludes coordinator nodes (MacBooks) from receiving synced data.
        """
        try:
            from app.coordination.auto_sync_daemon import AutoSyncDaemon

            daemon = AutoSyncDaemon()
            await daemon.start()

            # Wait for daemon to complete (or be stopped)
            while daemon.is_running():
                await asyncio.sleep(10)

        except ImportError as e:
            logger.error(f"AutoSyncDaemon not available: {e}")
            raise  # Propagate error so DaemonManager marks as FAILED

    async def _create_training_node_watcher(self) -> None:
        """Create and run training node watcher daemon (Phase 6, December 2025).

        Monitors for training activity across the cluster and triggers
        priority sync to ensure training nodes have fresh data.
        """
        try:
            from app.coordination.cluster_data_sync import get_training_node_watcher

            watcher = get_training_node_watcher()
            await watcher.start()

            # Wait for watcher to complete (or be stopped)
            while watcher._running:
                await asyncio.sleep(10)

        except ImportError as e:
            logger.error(f"TrainingNodeWatcher not available: {e}")
            raise  # Propagate error so DaemonManager marks as FAILED

    async def _create_ephemeral_sync(self) -> None:
        """Create and run ephemeral sync daemon (Phase 4, December 2025).

        Provides aggressive sync for Vast.ai and spot instances with
        short termination notice (15-30 seconds).
        """
        try:
            from app.coordination.ephemeral_sync import get_ephemeral_sync_daemon

            daemon = get_ephemeral_sync_daemon()
            await daemon.start()

            # Wait for daemon to complete (or be stopped)
            while daemon._running:
                await asyncio.sleep(5)

        except ImportError as e:
            logger.error(f"EphemeralSyncDaemon not available: {e}")
            raise  # Propagate error so DaemonManager marks as FAILED

    async def _create_replication_monitor(self) -> None:
        """Create and run replication monitor daemon (December 2025).

        Monitors data replication health across the cluster and triggers
        emergency sync when data safety is at risk.
        """
        try:
            from app.coordination.replication_monitor import get_replication_monitor

            daemon = get_replication_monitor()
            await daemon.start()

            # Wait for daemon to complete (or be stopped)
            while daemon.is_running():
                await asyncio.sleep(10)

        except ImportError as e:
            logger.error(f"ReplicationMonitorDaemon not available: {e}")
            raise  # Propagate error so DaemonManager marks as FAILED

    async def _create_tournament_daemon(self) -> None:
        """Create and run tournament scheduling daemon (December 2025).

        Automatically schedules evaluation tournaments when:
        - New models are trained (TRAINING_COMPLETED event)
        - Periodic ladder tournaments (configurable interval)

        Integrates with EloService to update ratings and emits
        EVALUATION_COMPLETED events for downstream consumers.
        """
        try:
            from app.coordination.tournament_daemon import get_tournament_daemon

            daemon = get_tournament_daemon()
            await daemon.start()

            # Wait for daemon to complete (or be stopped)
            while daemon.is_running():
                await asyncio.sleep(10)

        except ImportError as e:
            logger.error(f"TournamentDaemon not available: {e}")
            raise  # Propagate error so DaemonManager marks as FAILED


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
            fire_and_forget(
                manager.shutdown(),
                name="daemon_manager_signal_shutdown",
            )
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
