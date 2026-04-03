"""Thread-safe singleton patterns for RingRift.

.. deprecated:: December 2025
    This module is deprecated. Use ``app.coordination.singleton_mixin`` instead:

    .. code-block:: python

        # Old (deprecated)
        from app.core.singleton_mixin import SingletonMixin, SingletonMeta

        # New (canonical)
        from app.coordination.singleton_mixin import SingletonMixin, SingletonMeta

    All exports are re-exported from the canonical module for backward compatibility.
    This module will be archived in Q2 2026.

Usage:
    # Method 1: Metaclass (preferred for new classes)
    class MyService(metaclass=SingletonMeta):
        def __init__(self, config: str = "default"):
            self.config = config

    # Method 2: Mixin (for existing classes)
    class MyRegistry(SingletonMixin):
        _instance: MyRegistry | None = None

        @classmethod
        def get_instance(cls) -> MyRegistry:
            return cls._get_or_create_instance()

    # Method 3: Decorator (simplest)
    @singleton
    class MyCache:
        pass

All patterns support:
    - MyClass.get_instance() - get or create singleton
    - MyClass.reset_instance() - clear singleton (for testing)
    - MyClass.has_instance() - check if instance exists
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import warnings
from typing import Any, TypeVar

# Emit deprecation warning on import
warnings.warn(
    "app.core.singleton_mixin is deprecated. "
    "Use app.coordination.singleton_mixin instead. "
    "This module will be archived in Q2 2026.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from canonical location
from app.coordination.singleton_mixin import (
    LazySingletonMixin as _CoordLazySingletonMixin,
    LazySingletonMixin,
    SingletonMeta as _CoordSingletonMeta,
    SingletonMeta,
    SingletonMixin as _CoordSingletonMixin,
    SingletonMixin,
    ThreadSafeSingletonMixin as _CoordThreadSafeSingletonMixin,
    ThreadSafeSingletonMixin,
    create_singleton_accessors,
    singleton,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="SingletonMixin")


def _run_optional_reset_hook(instance: Any, hook_name: str) -> None:
    """Best-effort compatibility hook runner for deprecated singleton reset."""
    hook = getattr(instance, hook_name, None)
    if hook is None:
        return
    try:
        result = hook()
        if inspect.isawaitable(result):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(result)
            else:
                logger.debug(
                    "Skipping async %s during sync singleton reset for %s",
                    hook_name,
                    type(instance).__name__,
                )
    except Exception:
        logger.debug(
            "Ignoring %s failure during deprecated singleton reset for %s",
            hook_name,
            type(instance).__name__,
            exc_info=True,
        )


class SingletonMixin(_CoordSingletonMixin[T]):
    """Backward-compatible deprecated wrapper over the canonical singleton mixin."""

    @classmethod
    def _get_or_create_instance(cls: type[T], *args: Any, **kwargs: Any) -> T:
        return cls.get_instance(*args, **kwargs)

    @classmethod
    def _clear_instance(cls) -> None:
        cls.reset_instance()

    @classmethod
    def reset_instance(cls) -> None:
        instance = _CoordSingletonMixin._instances.get(cls)
        if instance is not None:
            _run_optional_reset_hook(instance, "_save_state")
            _run_optional_reset_hook(instance, "_cleanup")
        with contextlib.suppress(Exception):
            super().reset_instance()


class ThreadSafeSingletonMixin(_CoordThreadSafeSingletonMixin[T]):
    """Backward-compatible deprecated wrapper over the thread-safe mixin."""

    @classmethod
    def _clear_instance(cls) -> None:
        cls.reset_instance()

    @classmethod
    def reset_instance(cls) -> None:
        instance = _CoordSingletonMixin._instances.get(cls)
        if instance is not None:
            _run_optional_reset_hook(instance, "_save_state")
            _run_optional_reset_hook(instance, "_cleanup")
        with contextlib.suppress(Exception):
            super().reset_instance()


class LazySingletonMixin(_CoordLazySingletonMixin[T]):
    """Backward-compatible deprecated wrapper over the lazy singleton mixin."""

    @classmethod
    def _get_or_create_instance(cls: type[T], *args: Any, **kwargs: Any) -> T:
        return cls.get_instance(*args, **kwargs)

    @classmethod
    def _clear_instance(cls) -> None:
        cls.reset_instance()

__all__ = [
    "SingletonMeta",
    "SingletonMixin",
    "ThreadSafeSingletonMixin",
    "LazySingletonMixin",
    "singleton",
    "create_singleton_accessors",
]
