"""Standardized Result pattern for consistent success/failure handling.

This module provides a Result type that explicitly represents success or failure,
encouraging explicit error handling rather than exceptions for expected failures.

Usage:
    from app.utils.result import Result, Ok, Err

    def divide(a: int, b: int) -> Result[float, str]:
        if b == 0:
            return Err("Division by zero")
        return Ok(a / b)

    # Pattern matching style
    result = divide(10, 2)
    if result.is_ok:
        print(f"Result: {result.value}")
    else:
        print(f"Error: {result.error}")

    # Or with unwrap (raises if error)
    value = result.unwrap()

    # Or with default
    value = result.unwrap_or(0.0)

    # Chain operations
    result = (
        divide(10, 2)
        .map(lambda x: x * 2)
        .map(lambda x: int(x))
    )
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import (
    Any,
    Generic,
    TypeVar,
    Union,
)

T = TypeVar("T")  # Success type
E = TypeVar("E")  # Error type
U = TypeVar("U")  # Mapped type


class ResultError(Exception):
    """Raised when unwrapping a failed Result."""

    pass


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """Represents a successful result."""

    value: T

    @property
    def is_ok(self) -> bool:
        return True

    @property
    def is_err(self) -> bool:
        return False

    @property
    def error(self) -> None:
        return None

    def unwrap(self) -> T:
        """Get the success value."""
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Get the success value or a default."""
        return self.value

    def unwrap_or_else(self, f: Callable[[], T]) -> T:
        """Get the success value or compute a default."""
        return self.value

    def expect(self, msg: str) -> T:
        """Get the success value or raise with message."""
        return self.value

    def map(self, f: Callable[[T], U]) -> Result[U, Any]:
        """Transform the success value."""
        return Ok(f(self.value))

    def map_err(self, f: Callable[[Any], Any]) -> Result[T, Any]:
        """Transform the error (no-op for Ok)."""
        return self  # type: ignore

    def and_then(self, f: Callable[[T], Result[U, Any]]) -> Result[U, Any]:
        """Chain another operation that returns a Result."""
        return f(self.value)

    def or_else(self, f: Callable[[Any], Result[T, Any]]) -> Result[T, Any]:
        """Provide fallback (no-op for Ok)."""
        return self  # type: ignore

    def __iter__(self) -> Iterator[T]:
        """Allow iteration over success value."""
        yield self.value

    def __bool__(self) -> bool:
        """Ok is truthy."""
        return True

    def __repr__(self) -> str:
        return f"Ok({self.value!r})"


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    """Represents a failed result."""

    error: E

    @property
    def is_ok(self) -> bool:
        return False

    @property
    def is_err(self) -> bool:
        return True

    @property
    def value(self) -> None:
        return None

    def unwrap(self) -> Any:
        """Raises ResultError since this is an error."""
        raise ResultError(f"Called unwrap on Err: {self.error}")

    def unwrap_or(self, default: T) -> T:
        """Return the default since this is an error."""
        return default

    def unwrap_or_else(self, f: Callable[[], T]) -> T:
        """Compute and return default since this is an error."""
        return f()

    def expect(self, msg: str) -> Any:
        """Raises ResultError with custom message."""
        raise ResultError(f"{msg}: {self.error}")

    def map(self, f: Callable[[Any], U]) -> Result[U, E]:
        """No-op for error."""
        return self  # type: ignore

    def map_err(self, f: Callable[[E], U]) -> Result[Any, U]:
        """Transform the error value."""
        return Err(f(self.error))

    def and_then(self, f: Callable[[Any], Result[U, E]]) -> Result[U, E]:
        """No-op for error."""
        return self  # type: ignore

    def or_else(self, f: Callable[[E], Result[T, Any]]) -> Result[T, Any]:
        """Provide fallback result."""
        return f(self.error)

    def __iter__(self) -> Iterator[Any]:
        """Empty iteration for error."""
        return iter([])

    def __bool__(self) -> bool:
        """Err is falsy."""
        return False

    def __repr__(self) -> str:
        return f"Err({self.error!r})"


# Type alias for Result
Result = Union[Ok[T], Err[E]]


def result_from_exception(
    func: Callable[..., T],
    *args,
    catch: type = Exception,
    **kwargs,
) -> Result[T, str]:
    """Execute a function and wrap any exception as Err.

    Args:
        func: Function to execute
        *args: Positional arguments
        catch: Exception type(s) to catch
        **kwargs: Keyword arguments

    Returns:
        Ok(result) on success, Err(str(exception)) on failure

    Example:
        result = result_from_exception(int, "not a number")
        # Returns Err("invalid literal for int()...")
    """
    try:
        return Ok(func(*args, **kwargs))
    except catch as e:
        return Err(str(e))


def collect_results(results: list[Result[T, E]]) -> Result[list[T], E]:
    """Collect a list of Results into a Result of list.

    Returns Ok with all values if all results are Ok,
    otherwise returns the first Err encountered.

    Args:
        results: List of Result objects

    Returns:
        Ok(list of values) or first Err

    Example:
        results = [Ok(1), Ok(2), Ok(3)]
        collected = collect_results(results)  # Ok([1, 2, 3])

        results = [Ok(1), Err("fail"), Ok(3)]
        collected = collect_results(results)  # Err("fail")
    """
    values = []
    for r in results:
        if r.is_err:
            return r  # type: ignore
        values.append(r.value)
    return Ok(values)


def partition_results(
    results: list[Result[T, E]]
) -> tuple[list[T], list[E]]:
    """Partition results into successes and failures.

    Args:
        results: List of Result objects

    Returns:
        Tuple of (list of Ok values, list of Err values)

    Example:
        results = [Ok(1), Err("a"), Ok(2), Err("b")]
        oks, errs = partition_results(results)
        # oks = [1, 2], errs = ["a", "b"]
    """
    oks: list[T] = []
    errs: list[E] = []
    for r in results:
        if r.is_ok:
            oks.append(r.value)  # type: ignore
        else:
            errs.append(r.error)  # type: ignore
    return oks, errs


@dataclass
class OperationResult(Generic[T]):
    """A more detailed operation result with metadata.

    Use this for operations that need to report more than just
    success/failure, such as batch operations or complex workflows.

    Attributes:
        success: Whether the operation succeeded
        value: The result value (if successful)
        error: Error message (if failed)
        details: Additional details dict
        duration_ms: Operation duration in milliseconds
    """

    success: bool
    value: T | None = None
    error: str | None = None
    details: dict = field(default_factory=dict)
    duration_ms: float | None = None

    @classmethod
    def ok(cls, value: T, **details) -> OperationResult[T]:
        """Create a successful result."""
        return cls(success=True, value=value, details=details)

    @classmethod
    def fail(cls, error: str, **details) -> OperationResult[T]:
        """Create a failed result."""
        return cls(success=False, error=error, details=details)

    @property
    def is_ok(self) -> bool:
        return self.success

    @property
    def is_err(self) -> bool:
        return not self.success

    def to_result(self) -> Result[T, str]:
        """Convert to simple Result type."""
        if self.success:
            return Ok(self.value)  # type: ignore
        return Err(self.error or "Unknown error")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = {"success": self.success}
        if self.value is not None:
            d["value"] = self.value
        if self.error:
            d["error"] = self.error
        if self.details:
            d["details"] = self.details
        if self.duration_ms is not None:
            d["duration_ms"] = self.duration_ms
        return d


__all__ = [
    "Err",
    "Ok",
    "OperationResult",
    "Result",
    "ResultError",
    "collect_results",
    "partition_results",
    "result_from_exception",
]
