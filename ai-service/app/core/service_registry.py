"""ServiceRegistry - Centralized registry for service singletons.

January 2026: Phase 4.5 - Unified service access.

This module provides a central registry for all service singletons in RingRift.
It replaces scattered global variables with a unified, thread-safe registry.

Benefits:
- Centralized service discovery
- Thread-safe access
- Easy testing (reset registry between tests)
- Lazy initialization
- Dependency tracking

Usage:
    from app.core.service_registry import (
        ServiceRegistry,
        get_service,
        register_service,
        get_work_queue,  # Backward-compat accessor
        get_event_router,  # Backward-compat accessor
    )

    # Register a service
    register_service("work_queue", my_work_queue)

    # Get a service
    wq = get_service("work_queue")

    # Or use typed accessors
    wq = get_work_queue()
    router = get_event_router()

    # Reset for testing
    ServiceRegistry.reset()

Migration:
    # Old pattern (deprecated)
    _work_queue: WorkQueue | None = None
    def get_work_queue() -> WorkQueue | None:
        global _work_queue
        if _work_queue is None:
            _work_queue = WorkQueue()
        return _work_queue

    # New pattern
    from app.core.service_registry import get_work_queue
    wq = get_work_queue()
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

if TYPE_CHECKING:
    from app.coordination.work_queue import WorkQueue
    from app.coordination.event_router import UnifiedEventRouter
    from app.coordination.daemon_manager import DaemonManager
    from app.coordination.unified_health_manager import UnifiedHealthManager

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ServiceEntry(Generic[T]):
    """Entry in the service registry."""
    name: str
    instance: T | None = None
    factory: Callable[[], T] | None = None
    initialized: bool = False


class ServiceRegistry:
    """Central registry for all service singletons.

    Thread-safe registry with lazy initialization support.
    Use the module-level functions for convenient access.
    """

    _instance: ServiceRegistry | None = None
    _lock = threading.RLock()  # Reentrant lock for nested calls

    def __init__(self):
        self._services: dict[str, ServiceEntry[Any]] = {}
        self._service_lock = threading.RLock()  # Reentrant lock for nested calls
        self._initialization_callbacks: list[Callable[[str, Any], None]] = []

    @classmethod
    def get_instance(cls) -> ServiceRegistry:
        """Get the singleton registry instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (for testing).

        This clears all registered services and resets the singleton.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance._services.clear()
            cls._instance = None

    def register(
        self,
        name: str,
        instance: T | None = None,
        factory: Callable[[], T] | None = None,
    ) -> None:
        """Register a service.

        Args:
            name: Service name (e.g., "work_queue", "event_router")
            instance: Pre-created instance (mutually exclusive with factory)
            factory: Factory function for lazy initialization

        Raises:
            ValueError: If neither instance nor factory is provided
        """
        if instance is None and factory is None:
            raise ValueError(f"Must provide instance or factory for service '{name}'")

        with self._service_lock:
            self._services[name] = ServiceEntry(
                name=name,
                instance=instance,
                factory=factory,
                initialized=instance is not None,
            )

        if instance is not None:
            self._notify_callbacks(name, instance)

    def get(self, name: str) -> Any | None:
        """Get a service by name.

        If a factory was registered, the service is lazily initialized.

        Args:
            name: Service name

        Returns:
            The service instance, or None if not registered
        """
        with self._service_lock:
            entry = self._services.get(name)
            if entry is None:
                return None

            if not entry.initialized and entry.factory is not None:
                try:
                    entry.instance = entry.factory()
                    entry.initialized = True
                    self._notify_callbacks(name, entry.instance)
                except Exception as e:
                    logger.error(f"Failed to initialize service '{name}': {e}")
                    return None

            return entry.instance

    def get_or_create(self, name: str, factory: Callable[[], T]) -> T:
        """Get a service, creating it if not registered.

        Args:
            name: Service name
            factory: Factory function to create if missing

        Returns:
            The service instance
        """
        with self._service_lock:
            entry = self._services.get(name)
            if entry is None:
                self.register(name, factory=factory)
                return self.get(name)  # type: ignore
            return self.get(name)  # type: ignore

    def is_registered(self, name: str) -> bool:
        """Check if a service is registered."""
        return name in self._services

    def list_services(self) -> list[str]:
        """List all registered service names."""
        return list(self._services.keys())

    def on_service_initialized(
        self, callback: Callable[[str, Any], None]
    ) -> None:
        """Register a callback for when services are initialized.

        The callback receives (service_name, service_instance).
        """
        self._initialization_callbacks.append(callback)

    def _notify_callbacks(self, name: str, instance: Any) -> None:
        """Notify all callbacks about service initialization."""
        for callback in self._initialization_callbacks:
            try:
                callback(name, instance)
            except Exception as e:
                logger.error(f"Service callback error for '{name}': {e}")


# =============================================================================
# Module-level convenience functions
# =============================================================================


def get_registry() -> ServiceRegistry:
    """Get the global service registry."""
    return ServiceRegistry.get_instance()


def register_service(
    name: str,
    instance: Any = None,
    factory: Callable[[], Any] | None = None,
) -> None:
    """Register a service in the global registry."""
    get_registry().register(name, instance, factory)


def get_service(name: str) -> Any | None:
    """Get a service from the global registry."""
    return get_registry().get(name)


# =============================================================================
# Typed accessors for common services (backward compatibility)
# =============================================================================


def get_work_queue() -> WorkQueue | None:
    """Get the global work queue.

    Returns:
        WorkQueue instance or None if not initialized
    """
    return get_service("work_queue")


def get_event_router() -> UnifiedEventRouter | None:
    """Get the global event router.

    Returns:
        UnifiedEventRouter instance or None if not initialized
    """
    return get_service("event_router")


def get_daemon_manager() -> DaemonManager | None:
    """Get the global daemon manager.

    Returns:
        DaemonManager instance or None if not initialized
    """
    return get_service("daemon_manager")


def get_health_manager() -> UnifiedHealthManager | None:
    """Get the global health manager.

    Returns:
        UnifiedHealthManager instance or None if not initialized
    """
    return get_service("health_manager")


# =============================================================================
# Service names (constants for type safety)
# =============================================================================


class ServiceName:
    """Standard service names."""
    WORK_QUEUE = "work_queue"
    EVENT_ROUTER = "event_router"
    DAEMON_MANAGER = "daemon_manager"
    HEALTH_MANAGER = "health_manager"
    TRAINING_COORDINATOR = "training_coordinator"
    SELFPLAY_SCHEDULER = "selfplay_scheduler"
    DATA_PIPELINE = "data_pipeline"
    FEEDBACK_LOOP = "feedback_loop"
    CLUSTER_TRANSPORT = "cluster_transport"
    SYNC_PLANNER = "sync_planner"
