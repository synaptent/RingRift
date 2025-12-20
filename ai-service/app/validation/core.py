"""Core validation primitives.

Provides the base classes and functions for the validation system.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar, Union

__all__ = [
    "ValidationError",
    "ValidationResult",
    "Validator",
    "validate",
    "validate_all",
]

T = TypeVar("T")


class ValidationError(Exception):
    """Exception raised for validation failures."""

    def __init__(self, message: str, field: str | None = None):
        self.message = message
        self.field = field
        super().__init__(message)


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    field: str | None = None
    value: Any = None

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.valid

    @property
    def error_message(self) -> str:
        """Get combined error message."""
        return "; ".join(self.errors)

    @classmethod
    def ok(cls, value: Any = None) -> ValidationResult:
        """Create a successful result."""
        return cls(valid=True, value=value)

    @classmethod
    def fail(cls, error: str, field: str | None = None) -> ValidationResult:
        """Create a failed result."""
        return cls(valid=False, errors=[error], field=field)

    def merge(self, other: ValidationResult) -> ValidationResult:
        """Merge another result into this one."""
        return ValidationResult(
            valid=self.valid and other.valid,
            errors=self.errors + other.errors,
            field=self.field or other.field,
            value=self.value,
        )


# Validator type: callable that returns ValidationResult or bool
Validator = Callable[[Any], Union[ValidationResult, bool, str]]


def _normalize_result(
    result: Union[ValidationResult, bool, str, None],
    value: Any,
) -> ValidationResult:
    """Normalize various return types to ValidationResult."""
    if result is None:
        return ValidationResult.ok(value)
    if isinstance(result, ValidationResult):
        if result.value is None:
            result.value = value
        return result
    if isinstance(result, bool):
        return ValidationResult.ok(value) if result else ValidationResult.fail("Validation failed")
    if isinstance(result, str):
        # String is treated as error message
        return ValidationResult.fail(result)
    return ValidationResult.ok(value)


def validate(value: Any, *validators: Validator) -> ValidationResult:
    """Validate a value using one or more validators.

    Args:
        value: Value to validate
        *validators: Validator functions to apply

    Returns:
        ValidationResult with combined results

    Example:
        result = validate(age, is_positive, in_range(0, 120))
        if not result:
            print(result.errors)
    """
    result = ValidationResult.ok(value)

    for validator in validators:
        try:
            v_result = validator(value)
            normalized = _normalize_result(v_result, value)
            result = result.merge(normalized)
        except ValidationError as e:
            result = result.merge(ValidationResult.fail(e.message, e.field))
        except Exception as e:
            result = result.merge(ValidationResult.fail(str(e)))

    return result


def validate_all(
    items: list[Any],
    *validators: Validator,
    stop_on_first: bool = False,
) -> ValidationResult:
    """Validate a list of items.

    Args:
        items: Items to validate
        *validators: Validators to apply to each item
        stop_on_first: Stop on first error

    Returns:
        Combined ValidationResult
    """
    result = ValidationResult.ok()

    for i, item in enumerate(items):
        item_result = validate(item, *validators)
        if not item_result:
            item_result.errors = [f"Item {i}: {e}" for e in item_result.errors]
            result = result.merge(item_result)
            if stop_on_first:
                break

    return result
