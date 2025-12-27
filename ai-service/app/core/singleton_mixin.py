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

import warnings

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
    LazySingletonMixin,
    SingletonMeta,
    SingletonMixin,
    ThreadSafeSingletonMixin,
    create_singleton_accessors,
    singleton,
)

__all__ = [
    "SingletonMeta",
    "SingletonMixin",
    "ThreadSafeSingletonMixin",
    "LazySingletonMixin",
    "singleton",
    "create_singleton_accessors",
]
