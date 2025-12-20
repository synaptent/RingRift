"""Common validators for general use.

Provides validators for common validation scenarios:
- Range and numeric validators
- String validators
- Collection validators
- Type validators
"""

from __future__ import annotations

import re
from re import Pattern
from typing import Any, Union

from app.validation.core import ValidationResult, Validator

__all__ = [
    "each_item",
    # Collection validators
    "has_keys",
    "has_length",
    # Range validators
    "in_range",
    "is_instance",
    "is_non_negative",
    # String validators
    "is_not_empty",
    "is_positive",
    # Type validators
    "is_type",
    "matches_pattern",
    "max_length",
    "min_length",
]


# =============================================================================
# Range Validators
# =============================================================================

def in_range(
    min_val: Union[int, float],
    max_val: Union[int, float],
    inclusive: bool = True,
) -> Validator:
    """Create a validator that checks if value is in range.

    Args:
        min_val: Minimum value
        max_val: Maximum value
        inclusive: Whether bounds are inclusive

    Returns:
        Validator function
    """
    def validator(value: Any) -> ValidationResult:
        try:
            if inclusive:
                if min_val <= value <= max_val:
                    return ValidationResult.ok(value)
            else:
                if min_val < value < max_val:
                    return ValidationResult.ok(value)
            return ValidationResult.fail(
                f"Value {value} not in range [{min_val}, {max_val}]"
            )
        except TypeError:
            return ValidationResult.fail(f"Value {value} cannot be compared to range")

    return validator


def is_positive(value: Any) -> ValidationResult:
    """Validate that value is positive (> 0)."""
    try:
        if value > 0:
            return ValidationResult.ok(value)
        return ValidationResult.fail(f"Value {value} is not positive")
    except TypeError:
        return ValidationResult.fail(f"Value {value} cannot be checked for positivity")


def is_non_negative(value: Any) -> ValidationResult:
    """Validate that value is non-negative (>= 0)."""
    try:
        if value >= 0:
            return ValidationResult.ok(value)
        return ValidationResult.fail(f"Value {value} is negative")
    except TypeError:
        return ValidationResult.fail(f"Value {value} cannot be checked for non-negativity")


# =============================================================================
# String Validators
# =============================================================================

def is_not_empty(value: Any) -> ValidationResult:
    """Validate that value is not empty (None, '', [], {})."""
    if value is None:
        return ValidationResult.fail("Value is None")
    if isinstance(value, str) and not value.strip():
        return ValidationResult.fail("String is empty or whitespace")
    if hasattr(value, "__len__") and len(value) == 0:
        return ValidationResult.fail("Value is empty")
    return ValidationResult.ok(value)


def matches_pattern(
    pattern: Union[str, Pattern],
    flags: int = 0,
) -> Validator:
    """Create a validator that checks if value matches a regex pattern.

    Args:
        pattern: Regex pattern
        flags: Regex flags

    Returns:
        Validator function
    """
    compiled = re.compile(pattern, flags) if isinstance(pattern, str) else pattern

    def validator(value: Any) -> ValidationResult:
        if not isinstance(value, str):
            return ValidationResult.fail(f"Value {value} is not a string")
        if compiled.match(value):
            return ValidationResult.ok(value)
        return ValidationResult.fail(f"Value '{value}' does not match pattern")

    return validator


def max_length(length: int) -> Validator:
    """Create a validator that checks maximum length.

    Args:
        length: Maximum allowed length

    Returns:
        Validator function
    """
    def validator(value: Any) -> ValidationResult:
        try:
            if len(value) <= length:
                return ValidationResult.ok(value)
            return ValidationResult.fail(
                f"Length {len(value)} exceeds maximum {length}"
            )
        except TypeError:
            return ValidationResult.fail("Value has no length")

    return validator


def min_length(length: int) -> Validator:
    """Create a validator that checks minimum length.

    Args:
        length: Minimum required length

    Returns:
        Validator function
    """
    def validator(value: Any) -> ValidationResult:
        try:
            if len(value) >= length:
                return ValidationResult.ok(value)
            return ValidationResult.fail(
                f"Length {len(value)} is less than minimum {length}"
            )
        except TypeError:
            return ValidationResult.fail("Value has no length")

    return validator


# =============================================================================
# Collection Validators
# =============================================================================

def has_keys(*keys: str) -> Validator:
    """Create a validator that checks if dict has required keys.

    Args:
        *keys: Required keys

    Returns:
        Validator function
    """
    def validator(value: Any) -> ValidationResult:
        if not isinstance(value, dict):
            return ValidationResult.fail("Value is not a dictionary")

        missing = [k for k in keys if k not in value]
        if missing:
            return ValidationResult.fail(f"Missing keys: {', '.join(missing)}")
        return ValidationResult.ok(value)

    return validator


def has_length(
    min_len: int = 0,
    max_len: int = float("inf"),
) -> Validator:
    """Create a validator that checks collection length.

    Args:
        min_len: Minimum length
        max_len: Maximum length

    Returns:
        Validator function
    """
    def validator(value: Any) -> ValidationResult:
        try:
            length = len(value)
            if min_len <= length <= max_len:
                return ValidationResult.ok(value)
            return ValidationResult.fail(
                f"Length {length} not in range [{min_len}, {max_len}]"
            )
        except TypeError:
            return ValidationResult.fail("Value has no length")

    return validator


def each_item(*validators: Validator) -> Validator:
    """Create a validator that applies validators to each item in a collection.

    Args:
        *validators: Validators to apply to each item

    Returns:
        Validator function
    """
    from app.validation.core import validate

    def validator(value: Any) -> ValidationResult:
        if not hasattr(value, "__iter__"):
            return ValidationResult.fail("Value is not iterable")

        result = ValidationResult.ok(value)
        for i, item in enumerate(value):
            item_result = validate(item, *validators)
            if not item_result:
                item_result.errors = [f"[{i}]: {e}" for e in item_result.errors]
                result = result.merge(item_result)

        return result

    return validator


# =============================================================================
# Type Validators
# =============================================================================

def is_type(*types: type) -> Validator:
    """Create a validator that checks if value is one of the specified types.

    Args:
        *types: Allowed types

    Returns:
        Validator function
    """
    def validator(value: Any) -> ValidationResult:
        if type(value) in types:
            return ValidationResult.ok(value)
        type_names = ", ".join(t.__name__ for t in types)
        return ValidationResult.fail(
            f"Expected type {type_names}, got {type(value).__name__}"
        )

    return validator


def is_instance(*types: type) -> Validator:
    """Create a validator that checks if value is instance of specified types.

    Args:
        *types: Allowed types (uses isinstance)

    Returns:
        Validator function
    """
    def validator(value: Any) -> ValidationResult:
        if isinstance(value, types):
            return ValidationResult.ok(value)
        type_names = ", ".join(t.__name__ for t in types)
        return ValidationResult.fail(
            f"Expected instance of {type_names}, got {type(value).__name__}"
        )

    return validator
