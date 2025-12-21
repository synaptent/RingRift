"""Deprecation utilities for legacy replay code.

This module provides decorators and utilities to mark legacy functions
as deprecated, with clear migration paths to the canonical implementations.
"""

import functools
import warnings
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)


def deprecated_legacy(removal_target: str = "Q2 2026") -> Callable[[F], F]:
    """Decorator to mark legacy functions as deprecated.

    Args:
        removal_target: Target date/version for removal of this function.

    Example:
        @deprecated_legacy("Q2 2026")
        def normalize_legacy_state(state: GameState) -> GameState:
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__module__}.{func.__name__}() is deprecated. "
                f"This legacy function will be removed in {removal_target}. "
                f"See app/rules/legacy/CANONICAL_PATH.md for migration.",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator


def deprecated_class(removal_target: str = "Q2 2026"):
    """Decorator to mark entire classes as deprecated.

    Usage:
        @deprecated_class("Q2 2026")
        class LegacyMoveConverter:
            ...
    """
    def decorator(cls):
        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            warnings.warn(
                f"{cls.__module__}.{cls.__name__} is deprecated. "
                f"This legacy class will be removed in {removal_target}. "
                f"See app/rules/legacy/CANONICAL_PATH.md for migration.",
                DeprecationWarning,
                stacklevel=2,
            )
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls
    return decorator
