"""Lifecycle Management for RingRift AI Service.

Provides unified lifecycle management combining:
- State machine transitions
- Initialization with dependencies
- Health checking
- Graceful shutdown

Usage:
    from app.core.lifecycle import LifecycleManager, Service

    class DatabaseService(Service):
        @property
        def name(self) -> str:
            return "database"

        async def on_start(self) -> None:
            self._pool = await create_pool(...)

        async def on_stop(self) -> None:
            await self._pool.close()

        async def check_health(self) -> HealthStatus:
            await self._pool.execute("SELECT 1")
            return HealthStatus.healthy()

    # Using LifecycleManager
    manager = LifecycleManager()
    manager.register(DatabaseService())
    manager.register(CacheService())  # depends on database

    await manager.start_all()
    # Later
    await manager.stop_all()
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.core.health import HealthCheck, HealthRegistry, HealthResult, HealthStatus, ProbeType

logger = logging.getLogger(__name__)

__all__ = [
    "LifecycleEvent",
    "LifecycleListener",
    "LifecycleManager",
    "Service",
    "ServiceState",
]


class ServiceState(Enum):
    """States for a managed service."""
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class LifecycleEvent:
    """Event emitted during lifecycle transitions."""
    service_name: str
    event_type: str  # started, stopped, failed, health_changed
    old_state: ServiceState | None
    new_state: ServiceState
    timestamp: float = field(default_factory=time.time)
    error: Exception | None = None
    details: dict[str, Any] = field(default_factory=dict)


class LifecycleListener(ABC):
    """Listener for lifecycle events."""

    @abstractmethod
    async def on_lifecycle_event(self, event: LifecycleEvent) -> None:
        """Handle a lifecycle event."""


class Service(HealthCheck, ABC):
    """Base class for managed services.

    Combines lifecycle management with health checking.
    Services define their startup/shutdown behavior and health checks.

    Example:
        class ApiServer(Service):
            @property
            def name(self) -> str:
                return "api_server"

            @property
            def dependencies(self) -> List[str]:
                return ["database", "cache"]

            async def on_start(self) -> None:
                self._server = await create_server()
                await self._server.start()

            async def on_stop(self) -> None:
                await self._server.shutdown()

            async def check_health(self) -> HealthStatus:
                if self._server.is_accepting:
                    return HealthStatus.healthy()
                return HealthStatus.unhealthy("Not accepting connections")
    """

    def __init__(self):
        self._state = ServiceState.CREATED
        self._error: Exception | None = None
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._dependencies_resolved: dict[str, Service] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique service name."""

    @property
    def dependencies(self) -> list[str]:
        """List of service names this depends on."""
        return []

    @property
    def state(self) -> ServiceState:
        """Current service state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if service is running."""
        return self._state == ServiceState.RUNNING

    @property
    def uptime(self) -> float | None:
        """Get service uptime in seconds."""
        if self._started_at and self._state == ServiceState.RUNNING:
            return time.time() - self._started_at
        return None

    @property
    def error(self) -> Exception | None:
        """Get error that caused failure."""
        return self._error

    @abstractmethod
    async def on_start(self) -> None:
        """Called when service should start.

        Initialize resources and start processing.
        Dependencies are available via _get_dependency().
        """

    @abstractmethod
    async def on_stop(self) -> None:
        """Called when service should stop.

        Clean up resources and stop processing.
        """

    @abstractmethod
    async def check_health(self) -> HealthStatus:
        """Check service health."""

    def _get_dependency(self, name: str) -> Service:
        """Get a resolved dependency."""
        if name not in self._dependencies_resolved:
            raise RuntimeError(f"Dependency '{name}' not resolved for {self.name}")
        return self._dependencies_resolved[name]

    def _set_dependency(self, name: str, service: Service) -> None:
        """Set a resolved dependency (called by manager)."""
        self._dependencies_resolved[name] = service

    async def _do_start(self) -> None:
        """Internal start method."""
        self._state = ServiceState.STARTING
        try:
            await self.on_start()
            self._state = ServiceState.RUNNING
            self._started_at = time.time()
            self._error = None
            logger.info(f"Service {self.name} started")
        except Exception as e:
            self._state = ServiceState.FAILED
            self._error = e
            logger.error(f"Service {self.name} failed to start: {e}")
            raise

    async def _do_stop(self) -> None:
        """Internal stop method."""
        self._state = ServiceState.STOPPING
        try:
            await self.on_stop()
            self._state = ServiceState.STOPPED
            self._stopped_at = time.time()
            logger.info(f"Service {self.name} stopped")
        except Exception as e:
            self._state = ServiceState.FAILED
            self._error = e
            logger.error(f"Service {self.name} failed to stop: {e}")
            raise

    def get_status(self) -> dict[str, Any]:
        """Get service status as dictionary."""
        return {
            "name": self.name,
            "state": self._state.value,
            "uptime": self.uptime,
            "error": str(self._error) if self._error else None,
            "dependencies": self.dependencies,
        }


class LifecycleManager:
    """Manages service lifecycles with dependency ordering.

    Handles:
    - Service registration and dependency resolution
    - Ordered startup respecting dependencies
    - Reverse-order shutdown
    - Health check aggregation
    - Signal handling for graceful shutdown

    Example:
        manager = LifecycleManager()

        # Register services
        manager.register(DatabaseService())
        manager.register(CacheService())
        manager.register(ApiService())

        # Start all (respects dependency order)
        await manager.start_all()

        # Health check endpoint
        health = await manager.check_health()

        # Graceful shutdown
        await manager.stop_all()
    """

    def __init__(
        self,
        health_timeout: float = 5.0,
        shutdown_timeout: float = 30.0,
    ):
        """Initialize the lifecycle manager.

        Args:
            health_timeout: Timeout for health checks
            shutdown_timeout: Timeout for service shutdown
        """
        self._services: dict[str, Service] = {}
        self._startup_order: list[str] = []
        self._listeners: list[LifecycleListener] = []
        self._health_registry = HealthRegistry(timeout=health_timeout)
        self._shutdown_timeout = shutdown_timeout
        self._shutting_down = False
        self._signal_handlers_installed = False

    def register(self, service: Service) -> None:
        """Register a service.

        Args:
            service: Service to register

        Raises:
            ValueError: If service with same name already registered
        """
        if service.name in self._services:
            raise ValueError(f"Service '{service.name}' already registered")

        self._services[service.name] = service
        self._health_registry.register(service.name, service)
        logger.debug(f"Registered service: {service.name}")

    def unregister(self, name: str) -> None:
        """Unregister a service."""
        self._services.pop(name, None)
        self._health_registry.unregister(name)

    def get(self, name: str) -> Service | None:
        """Get a service by name."""
        return self._services.get(name)

    def add_listener(self, listener: LifecycleListener) -> None:
        """Add a lifecycle event listener."""
        self._listeners.append(listener)

    def remove_listener(self, listener: LifecycleListener) -> None:
        """Remove a lifecycle event listener."""
        self._listeners.remove(listener)

    @property
    def services(self) -> dict[str, Service]:
        """Get all registered services."""
        return dict(self._services)

    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._shutting_down

    def compute_startup_order(self) -> list[str]:
        """Compute startup order via topological sort.

        Returns:
            List of service names in startup order

        Raises:
            ValueError: If dependencies cannot be resolved
        """
        # Build dependency graph
        in_degree: dict[str, int] = dict.fromkeys(self._services, 0)
        graph: dict[str, list[str]] = {name: [] for name in self._services}

        for name, service in self._services.items():
            for dep in service.dependencies:
                if dep not in self._services:
                    raise ValueError(
                        f"Service '{name}' depends on '{dep}' which is not registered"
                    )
                graph[dep].append(name)
                in_degree[name] += 1

        # Kahn's algorithm
        queue = [name for name, degree in in_degree.items() if degree == 0]
        order: list[str] = []

        while queue:
            queue.sort()  # Deterministic order
            name = queue.pop(0)
            order.append(name)

            for dependent in graph[name]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(order) != len(self._services):
            remaining = set(self._services.keys()) - set(order)
            raise ValueError(f"Circular dependency detected involving: {remaining}")

        self._startup_order = order
        return order

    async def start_all(
        self,
        parallel: bool = False,
        stop_on_failure: bool = True,
    ) -> list[str]:
        """Start all registered services.

        Args:
            parallel: Start independent services in parallel
            stop_on_failure: Stop startup if any service fails

        Returns:
            List of started service names
        """
        order = self.compute_startup_order()
        logger.info(f"Starting services: {' -> '.join(order)}")

        # Resolve dependencies
        for name in order:
            service = self._services[name]
            for dep_name in service.dependencies:
                service._set_dependency(dep_name, self._services[dep_name])

        started: list[str] = []

        if parallel:
            started = await self._start_parallel(order, stop_on_failure)
        else:
            started = await self._start_sequential(order, stop_on_failure)

        logger.info(f"Started {len(started)}/{len(order)} services")
        return started

    async def _start_sequential(
        self,
        order: list[str],
        stop_on_failure: bool,
    ) -> list[str]:
        """Start services sequentially."""
        started = []

        for name in order:
            service = self._services[name]
            old_state = service.state

            try:
                await service._do_start()
                started.append(name)
                await self._emit_event(LifecycleEvent(
                    service_name=name,
                    event_type="started",
                    old_state=old_state,
                    new_state=service.state,
                ))
            except Exception as e:
                await self._emit_event(LifecycleEvent(
                    service_name=name,
                    event_type="failed",
                    old_state=old_state,
                    new_state=service.state,
                    error=e,
                ))
                if stop_on_failure:
                    # Stop already started services
                    await self._stop_services(started)
                    raise

        return started

    async def _start_parallel(
        self,
        order: list[str],
        stop_on_failure: bool,
    ) -> list[str]:
        """Start independent services in parallel."""
        started: set[str] = set()
        pending = set(order)

        while pending:
            # Find services whose dependencies are all started
            ready = []
            for name in pending:
                service = self._services[name]
                if all(dep in started for dep in service.dependencies):
                    ready.append(name)

            if not ready:
                break

            # Start ready services in parallel
            results = await asyncio.gather(
                *[self._start_one(name) for name in ready],
                return_exceptions=True,
            )

            for name, result in zip(ready, results, strict=False):
                if isinstance(result, Exception):
                    if stop_on_failure:
                        await self._stop_services(list(started))
                        raise result
                else:
                    started.add(name)
                    pending.discard(name)

        return list(started)

    async def _start_one(self, name: str) -> None:
        """Start a single service."""
        service = self._services[name]
        old_state = service.state

        try:
            await service._do_start()
            await self._emit_event(LifecycleEvent(
                service_name=name,
                event_type="started",
                old_state=old_state,
                new_state=service.state,
            ))
        except Exception as e:
            await self._emit_event(LifecycleEvent(
                service_name=name,
                event_type="failed",
                old_state=old_state,
                new_state=service.state,
                error=e,
            ))
            raise

    async def stop_all(self, reverse: bool = True) -> None:
        """Stop all running services.

        Args:
            reverse: Stop in reverse startup order
        """
        self._shutting_down = True
        order = list(self._startup_order)
        if reverse:
            order.reverse()

        await self._stop_services(order)
        self._shutting_down = False

    async def _stop_services(self, names: list[str]) -> None:
        """Stop specified services."""
        logger.info(f"Stopping {len(names)} services")

        for name in names:
            service = self._services.get(name)
            if not service or service.state != ServiceState.RUNNING:
                continue

            old_state = service.state

            try:
                await asyncio.wait_for(
                    service._do_stop(),
                    timeout=self._shutdown_timeout,
                )
                await self._emit_event(LifecycleEvent(
                    service_name=name,
                    event_type="stopped",
                    old_state=old_state,
                    new_state=service.state,
                ))
            except asyncio.TimeoutError:
                logger.error(f"Service {name} shutdown timed out")
                service._state = ServiceState.FAILED
            except Exception as e:
                logger.error(f"Service {name} shutdown error: {e}")
                await self._emit_event(LifecycleEvent(
                    service_name=name,
                    event_type="failed",
                    old_state=old_state,
                    new_state=service.state,
                    error=e,
                ))

    async def restart(self, name: str) -> None:
        """Restart a specific service.

        Args:
            name: Service name to restart
        """
        service = self._services.get(name)
        if not service:
            raise KeyError(f"Service '{name}' not registered")

        logger.info(f"Restarting service: {name}")

        if service.state == ServiceState.RUNNING:
            await service._do_stop()

        await service._do_start()

    async def check_health(
        self,
        probe_type: ProbeType = ProbeType.READINESS,
    ) -> HealthResult:
        """Check health of all services.

        Args:
            probe_type: Type of health probe

        Returns:
            Aggregated health result
        """
        return await self._health_registry.check_all(probe_type=probe_type)

    async def _emit_event(self, event: LifecycleEvent) -> None:
        """Emit lifecycle event to listeners."""
        for listener in self._listeners:
            try:
                await listener.on_lifecycle_event(event)
            except Exception as e:
                logger.warning(f"Lifecycle listener error: {e}")

    def install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown.

        Handles SIGTERM and SIGINT for graceful shutdown.
        """
        if self._signal_handlers_installed:
            return

        def handler(signum: int, frame: Any) -> None:
            logger.info(f"Received signal {signum}, initiating shutdown")
            asyncio.create_task(self.stop_all())

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)
        self._signal_handlers_installed = True
        logger.debug("Signal handlers installed")

    def get_status(self) -> dict[str, Any]:
        """Get status of all services."""
        return {
            "services": {
                name: service.get_status()
                for name, service in self._services.items()
            },
            "startup_order": self._startup_order,
            "shutting_down": self._shutting_down,
        }


# Convenience context manager
class managed_lifecycle:
    """Context manager for lifecycle management.

    Usage:
        async with managed_lifecycle(manager) as m:
            # Services are running
            await do_work()
        # Services are stopped on exit
    """

    def __init__(
        self,
        manager: LifecycleManager,
        install_signals: bool = True,
    ):
        self._manager = manager
        self._install_signals = install_signals

    async def __aenter__(self) -> LifecycleManager:
        if self._install_signals:
            self._manager.install_signal_handlers()
        await self._manager.start_all()
        return self._manager

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self._manager.stop_all()
