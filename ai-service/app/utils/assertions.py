"""Assertion and precondition utilities for debugging and validation.

This module provides assertion helpers that:
- Give better error messages than built-in assert
- Can be disabled in production for performance
- Support common validation patterns

Usage:
    from app.utils.assertions import require, ensure, check

    def process_game(game_id: str, player: int):
        require(game_id, "game_id is required")
        require(0 <= player <= 3, f"player must be 0-3, got {player}")

        # ... do work ...

        ensure(result is not None, "process_game must return a result")
        return result
"""

from __future__ import annotations

import os
from typing import Any, Optional, Type, TypeVar, Union

T = TypeVar("T")

# Check if assertions are enabled (disabled in production for performance)
ASSERTIONS_ENABLED = os.environ.get("RINGRIFT_ENV", "development").lower() != "production"


class AssertionError(Exception):
    """Raised when an assertion fails.

    This is a custom exception (not built-in AssertionError) so it can be
    caught separately and includes better context.
    """

    def __init__(self, message: str, context: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} ({ctx})"
        return self.message


class PreconditionError(AssertionError):
    """Raised when a precondition (require) fails."""
    pass


class PostconditionError(AssertionError):
    """Raised when a postcondition (ensure) fails."""
    pass


class InvariantError(AssertionError):
    """Raised when an invariant (check) fails."""
    pass


def require(
    condition: Any,
    message: str = "Precondition failed",
    **context,
) -> None:
    """Assert a precondition at function entry.

    Use this to validate function arguments.

    Args:
        condition: Condition that must be true
        message: Error message if condition is false
        **context: Additional context for error message

    Raises:
        PreconditionError: If condition is false

    Example:
        def divide(a: int, b: int) -> float:
            require(b != 0, "divisor cannot be zero", a=a, b=b)
            return a / b
    """
    if not ASSERTIONS_ENABLED:
        return

    if not condition:
        raise PreconditionError(message, context)


def ensure(
    condition: Any,
    message: str = "Postcondition failed",
    **context,
) -> None:
    """Assert a postcondition at function exit.

    Use this to validate function return values.

    Args:
        condition: Condition that must be true
        message: Error message if condition is false
        **context: Additional context for error message

    Raises:
        PostconditionError: If condition is false

    Example:
        def get_positive_value() -> int:
            result = compute_something()
            ensure(result > 0, "result must be positive", result=result)
            return result
    """
    if not ASSERTIONS_ENABLED:
        return

    if not condition:
        raise PostconditionError(message, context)


def check(
    condition: Any,
    message: str = "Invariant violated",
    **context,
) -> None:
    """Assert an invariant that should always be true.

    Use this for sanity checks in the middle of functions.

    Args:
        condition: Condition that must be true
        message: Error message if condition is false
        **context: Additional context for error message

    Raises:
        InvariantError: If condition is false

    Example:
        def process_items(items: list) -> list:
            result = []
            for item in items:
                processed = transform(item)
                check(processed is not None, "transform returned None", item=item)
                result.append(processed)
            return result
    """
    if not ASSERTIONS_ENABLED:
        return

    if not condition:
        raise InvariantError(message, context)


def require_type(
    value: Any,
    expected_type: Union[Type, tuple],
    name: str = "value",
) -> None:
    """Assert that a value has the expected type.

    Args:
        value: Value to check
        expected_type: Expected type or tuple of types
        name: Name of the value for error message

    Raises:
        PreconditionError: If value is not of expected type

    Example:
        def process_name(name: str) -> str:
            require_type(name, str, "name")
            return name.upper()
    """
    if not ASSERTIONS_ENABLED:
        return

    if not isinstance(value, expected_type):
        type_name = (
            expected_type.__name__
            if isinstance(expected_type, type)
            else " or ".join(t.__name__ for t in expected_type)
        )
        raise PreconditionError(
            f"{name} must be {type_name}, got {type(value).__name__}",
            {"value": repr(value)[:100]},
        )


def require_not_none(
    value: Optional[T],
    name: str = "value",
) -> T:
    """Assert that a value is not None and return it.

    Args:
        value: Value to check
        name: Name of the value for error message

    Returns:
        The value (with type narrowed to non-None)

    Raises:
        PreconditionError: If value is None

    Example:
        def process_user(user: Optional[User]) -> str:
            user = require_not_none(user, "user")
            return user.name
    """
    if value is None:
        raise PreconditionError(f"{name} must not be None")
    return value


def require_in_range(
    value: Union[int, float],
    min_val: Union[int, float],
    max_val: Union[int, float],
    name: str = "value",
    inclusive: bool = True,
) -> None:
    """Assert that a numeric value is within a range.

    Args:
        value: Value to check
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the value for error message
        inclusive: Whether bounds are inclusive (default True)

    Raises:
        PreconditionError: If value is out of range

    Example:
        def set_volume(level: int) -> None:
            require_in_range(level, 0, 100, "volume level")
            self._volume = level
    """
    if not ASSERTIONS_ENABLED:
        return

    if inclusive:
        in_range = min_val <= value <= max_val
        range_str = f"[{min_val}, {max_val}]"
    else:
        in_range = min_val < value < max_val
        range_str = f"({min_val}, {max_val})"

    if not in_range:
        raise PreconditionError(
            f"{name} must be in range {range_str}, got {value}",
            {"value": value, "min": min_val, "max": max_val},
        )


def require_non_empty(
    value: Any,
    name: str = "value",
) -> None:
    """Assert that a collection is not empty.

    Args:
        value: Collection to check (must support len())
        name: Name of the value for error message

    Raises:
        PreconditionError: If collection is empty

    Example:
        def process_items(items: list) -> int:
            require_non_empty(items, "items")
            return sum(items)
    """
    if not ASSERTIONS_ENABLED:
        return

    if len(value) == 0:
        raise PreconditionError(f"{name} must not be empty")


def unreachable(message: str = "This code should be unreachable") -> None:
    """Mark code that should never be reached.

    Use this in default cases of exhaustive switches, after return statements
    that should always happen, etc.

    Args:
        message: Description of why this is unreachable

    Raises:
        InvariantError: Always raised

    Example:
        def process_status(status: Status) -> str:
            if status == Status.ACTIVE:
                return "active"
            elif status == Status.INACTIVE:
                return "inactive"
            else:
                unreachable(f"Unknown status: {status}")
    """
    raise InvariantError(f"Unreachable code reached: {message}")


__all__ = [
    "AssertionError",
    "PreconditionError",
    "PostconditionError",
    "InvariantError",
    "require",
    "ensure",
    "check",
    "require_type",
    "require_not_none",
    "require_in_range",
    "require_non_empty",
    "unreachable",
    "ASSERTIONS_ENABLED",
]
