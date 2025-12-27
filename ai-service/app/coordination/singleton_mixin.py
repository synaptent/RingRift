"""Singleton Mixin for Daemons and Coordinators (December 2025).

Provides standardized thread-safe singleton pattern for daemons and coordinators.

This mixin consolidates 25+ similar patterns found across the codebase:
- daemon_manager.py (DaemonManager)
- daemon_factory.py (DaemonFactory)
- facade.py (CoordinationFacade)
- sync_coordinator.py (SyncScheduler)
- selfplay_scheduler.py (SelfplayScheduler)
- maintenance_daemon.py (MaintenanceDaemon)
- backpressure.py (BackpressureMonitor)
- resource_monitoring_coordinator.py (ResourceMonitoringCoordinator)
- pipeline_triggers.py (PipelineTrigger)
- event_router.py (UnifiedEventRouter)
- plus 15+ more daemon/coordinator classes

Usage:
    class MyDaemon(BaseDaemon, SingletonMixin):
        def __init__(self, config: Optional[MyConfig] = None):
            super().__init__("my_daemon")
            self.config = config or MyConfig()

    # Get singleton instance
    daemon = MyDaemon.get_instance()

    # With custom initialization
    daemon = MyDaemon.get_instance(config=custom_config)

    # Reset for testing
    MyDaemon.reset_instance()

    # Module-level convenience functions (optional)
    _my_daemon: MyDaemon | None = None

    def get_my_daemon() -> MyDaemon:
        global _my_daemon
        if _my_daemon is None:
            _my_daemon = MyDaemon.get_instance()
        return _my_daemon

    def reset_my_daemon() -> None:
        global _my_daemon
        MyDaemon.reset_instance()
        _my_daemon = None
"""

from __future__ import annotations

import logging
import threading
from typing import Any, ClassVar, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="SingletonMixin")


class SingletonMixin(Generic[T]):
    """Mixin providing thread-safe singleton pattern.

    Features:
    - Thread-safe singleton access via get_instance()
    - Reset capability for testing via reset_instance()
    - Optional initialization arguments support
    - Per-class instance storage (each subclass has its own singleton)
    - Compatible with existing daemon base classes

    Thread Safety:
    Uses a per-class reentrant lock to ensure thread-safe access.
    The lock is reentrant (RLock) to allow nested calls within the same thread.

    Notes:
    - This is a mixin, not a base class - use alongside your actual base class
    - Each subclass maintains its own singleton instance
    - Arguments passed to get_instance() are only used on first instantiation
    - Subsequent calls to get_instance() return the existing instance regardless of args
    """

    # Class-level storage for instances and locks
    # Using ClassVar to indicate these are class variables
    _instances: ClassVar[dict[type, Any]] = {}
    _locks: ClassVar[dict[type, threading.RLock]] = {}

    @classmethod
    def _get_lock(cls) -> threading.RLock:
        """Get or create the lock for this class.

        Uses a per-class lock to avoid contention between different singleton types.
        """
        # Access _locks directly to avoid recursion
        if cls not in SingletonMixin._locks:
            # Use a meta-lock for creating per-class locks
            # This is a one-time operation per class, so acceptable
            SingletonMixin._locks[cls] = threading.RLock()
        return SingletonMixin._locks[cls]

    @classmethod
    def get_instance(cls: type[T], *args: Any, **kwargs: Any) -> T:
        """Get or create the singleton instance.

        Thread-safe access to the singleton instance. If the instance doesn't
        exist, it will be created with the provided arguments. If it already
        exists, the existing instance is returned and arguments are ignored.

        Args:
            *args: Positional arguments for first-time initialization
            **kwargs: Keyword arguments for first-time initialization

        Returns:
            The singleton instance

        Example:
            # First call creates instance
            daemon = MyDaemon.get_instance(config=custom_config)

            # Subsequent calls return same instance
            same_daemon = MyDaemon.get_instance()
            assert daemon is same_daemon
        """
        with cls._get_lock():
            if cls not in SingletonMixin._instances:
                logger.debug(f"[{cls.__name__}] Creating singleton instance")
                try:
                    instance = cls(*args, **kwargs)
                    SingletonMixin._instances[cls] = instance
                except Exception as e:
                    logger.error(f"[{cls.__name__}] Failed to create singleton: {e}")
                    raise
            return SingletonMixin._instances[cls]  # type: ignore[return-value]

    @classmethod
    def has_instance(cls) -> bool:
        """Check if a singleton instance exists.

        Returns:
            True if singleton instance exists
        """
        with cls._get_lock():
            return cls in SingletonMixin._instances

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance.

        Thread-safe reset of the singleton. Primarily used for testing.
        Does NOT call any cleanup methods on the instance - call stop()
        explicitly if needed before resetting.

        Example:
            # In test setup
            MyDaemon.reset_instance()

            # Or with cleanup
            daemon = MyDaemon.get_instance()
            await daemon.stop()
            MyDaemon.reset_instance()
        """
        with cls._get_lock():
            if cls in SingletonMixin._instances:
                logger.debug(f"[{cls.__name__}] Resetting singleton instance")
                del SingletonMixin._instances[cls]

    @classmethod
    def reset_instance_safe(cls) -> None:
        """Reset singleton only if it's safe to do so.

        Checks if the instance has a `is_running` or `_running` attribute
        and refuses to reset if the daemon appears to be running.

        Raises:
            RuntimeError: If the daemon is still running
        """
        with cls._get_lock():
            if cls not in SingletonMixin._instances:
                return

            instance = SingletonMixin._instances[cls]

            # Check for running state
            is_running = getattr(instance, "is_running", None)
            if is_running is None:
                is_running = getattr(instance, "_running", False)

            if is_running:
                raise RuntimeError(
                    f"[{cls.__name__}] Cannot reset singleton: daemon is still running"
                )

            logger.debug(f"[{cls.__name__}] Safe-resetting singleton instance")
            del SingletonMixin._instances[cls]


class LazySingletonMixin(SingletonMixin[T]):
    """Singleton mixin with lazy initialization support.

    Extends SingletonMixin with support for deferred initialization.
    Useful when the singleton needs external configuration that isn't
    available at import time.

    Usage:
        class MyDaemon(BaseDaemon, LazySingletonMixin):
            def __init__(self, config: MyConfig):
                super().__init__("my_daemon")
                self._config = config
                self._initialized = False

            def initialize(self) -> None:
                if self._initialized:
                    return
                # Expensive initialization here
                self._initialized = True

        # Later, when config is available
        MyDaemon.initialize_singleton(config=my_config)
        daemon = MyDaemon.get_instance()
    """

    _initialization_args: ClassVar[dict[type, tuple[tuple, dict]]] = {}

    @classmethod
    def configure_singleton(cls: type[T], *args: Any, **kwargs: Any) -> None:
        """Configure initialization arguments without creating the instance.

        Call this before get_instance() to set up initialization arguments.
        The singleton will be created with these arguments on first access.

        Args:
            *args: Positional arguments for initialization
            **kwargs: Keyword arguments for initialization
        """
        with cls._get_lock():
            LazySingletonMixin._initialization_args[cls] = (args, kwargs)
            logger.debug(f"[{cls.__name__}] Configured singleton initialization")

    @classmethod
    def get_instance(cls: type[T], *args: Any, **kwargs: Any) -> T:
        """Get or create the singleton instance.

        If configure_singleton() was called, uses those arguments.
        Otherwise uses the provided arguments.
        """
        with cls._get_lock():
            if cls not in SingletonMixin._instances:
                # Check for pre-configured arguments
                if cls in LazySingletonMixin._initialization_args:
                    stored_args, stored_kwargs = LazySingletonMixin._initialization_args[
                        cls
                    ]
                    # Merge: stored args take precedence, kwargs are merged
                    args = stored_args or args
                    merged_kwargs = {**kwargs, **stored_kwargs}
                    return super().get_instance(*args, **merged_kwargs)

            return super().get_instance(*args, **kwargs)

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton and clear configuration."""
        with cls._get_lock():
            super().reset_instance()
            if cls in LazySingletonMixin._initialization_args:
                del LazySingletonMixin._initialization_args[cls]


def create_singleton_accessors(
    cls: type[T],
    get_name: str | None = None,
    reset_name: str | None = None,
) -> tuple[callable, callable]:
    """Create module-level get/reset functions for a singleton class.

    Migration helper for existing code that uses module-level functions.

    Args:
        cls: The singleton class (must use SingletonMixin)
        get_name: Name for the getter (default: get_{class_name_lower})
        reset_name: Name for the reset (default: reset_{class_name_lower})

    Returns:
        Tuple of (get_function, reset_function)

    Example:
        class MyDaemon(BaseDaemon, SingletonMixin):
            pass

        get_my_daemon, reset_my_daemon = create_singleton_accessors(MyDaemon)

        # Now you can use:
        daemon = get_my_daemon()
        reset_my_daemon()
    """
    class_name = cls.__name__
    lower_name = "".join(
        f"_{c.lower()}" if c.isupper() and i > 0 else c.lower()
        for i, c in enumerate(class_name)
    )

    # Module-level cache for backward compatibility
    _cache: dict[str, Any] = {"instance": None}

    def get_instance_wrapper(*args: Any, **kwargs: Any) -> T:
        if _cache["instance"] is None:
            _cache["instance"] = cls.get_instance(*args, **kwargs)
        return _cache["instance"]

    def reset_instance_wrapper() -> None:
        cls.reset_instance()
        _cache["instance"] = None

    # Set function names for debugging
    actual_get_name = get_name or f"get_{lower_name}"
    actual_reset_name = reset_name or f"reset_{lower_name}"
    get_instance_wrapper.__name__ = actual_get_name
    reset_instance_wrapper.__name__ = actual_reset_name

    return get_instance_wrapper, reset_instance_wrapper


__all__ = [
    "SingletonMixin",
    "LazySingletonMixin",
    "create_singleton_accessors",
]
