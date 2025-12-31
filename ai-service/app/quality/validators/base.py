"""Base Validator Abstract Class.

Provides the abstract base class that all validators should inherit from.
Validators check data integrity and return ValidationResult objects.

December 30, 2025: Created as part of Priority 3.4 consolidation effort.

Usage:
    from app.quality.validators.base import BaseValidator, ValidatorConfig

    class MyValidator(BaseValidator):
        def _validate(self, data: Any) -> ValidationResult:
            # Custom validation logic
            if not data:
                return ValidationResult.invalid("Data is empty")
            return ValidationResult.valid()

    validator = MyValidator()
    result = validator.validate(my_data)
    if not result.is_valid:
        print(f"Errors: {result.errors}")
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.quality.types import ValidationResult

__all__ = [
    "BaseValidator",
    "ValidatorConfig",
    "ValidatorStats",
]

logger = logging.getLogger(__name__)


@dataclass
class ValidatorConfig:
    """Configuration for validators.

    Attributes:
        strict_mode: Whether to fail on first error (True) or collect all (False)
        warn_on_missing: Log warnings for missing optional fields
        max_errors: Maximum errors to collect before stopping (0 = unlimited)
    """

    strict_mode: bool = False
    warn_on_missing: bool = True
    max_errors: int = 0


@dataclass
class ValidatorStats:
    """Statistics for validator performance monitoring."""

    total_validated: int = 0
    passed: int = 0
    failed: int = 0
    total_time_ms: float = 0.0
    error_counts: dict[str, int] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        """Validation pass rate."""
        if self.total_validated == 0:
            return 0.0
        return self.passed / self.total_validated

    @property
    def avg_time_ms(self) -> float:
        """Average validation time in milliseconds."""
        if self.total_validated == 0:
            return 0.0
        return self.total_time_ms / self.total_validated

    def record_error(self, error_type: str) -> None:
        """Record an error occurrence."""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_validated": self.total_validated,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "avg_time_ms": self.avg_time_ms,
            "error_counts": self.error_counts,
        }


class BaseValidator(ABC):
    """Abstract base class for data validators.

    Provides common infrastructure for validation:
    - Consistent ValidationResult return type
    - Statistics tracking
    - Error collection with configurable limits
    - Logging integration

    Subclasses must implement:
    - _validate(data) -> ValidationResult: Core validation logic
    """

    # Class-level validator identification
    VALIDATOR_NAME: str = "base"
    VALIDATOR_VERSION: str = "1.0.0"

    def __init__(self, config: ValidatorConfig | None = None):
        """Initialize the validator.

        Args:
            config: Validator configuration
        """
        self.config = config or ValidatorConfig()
        self._stats = ValidatorStats()

    @property
    def stats(self) -> ValidatorStats:
        """Get validation statistics."""
        return self._stats

    def validate(self, data: Any) -> ValidationResult:
        """Validate data and return result.

        This is the main entry point. It handles timing, statistics,
        and logging around the core validation logic.

        Args:
            data: Data to validate (type depends on validator)

        Returns:
            ValidationResult with validity status and any errors/warnings
        """
        start_time = time.perf_counter()

        try:
            result = self._validate(data)
        except Exception as e:
            logger.error(f"{self.VALIDATOR_NAME} validation error: {e}")
            result = ValidationResult.invalid(f"Validation error: {e}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._record_result(result, elapsed_ms)

        return result

    @abstractmethod
    def _validate(self, data: Any) -> ValidationResult:
        """Core validation logic.

        Subclasses must implement this method.

        Args:
            data: Data to validate

        Returns:
            ValidationResult with validity status
        """
        ...

    def _record_result(self, result: ValidationResult, time_ms: float) -> None:
        """Record validation result for statistics."""
        self._stats.total_validated += 1
        self._stats.total_time_ms += time_ms

        if result.is_valid:
            self._stats.passed += 1
        else:
            self._stats.failed += 1
            for error in result.errors:
                # Extract error type from message for categorization
                error_type = error.split(":")[0] if ":" in error else "general"
                self._stats.record_error(error_type)

    def reset_stats(self) -> None:
        """Reset validation statistics."""
        self._stats = ValidatorStats()

    def get_info(self) -> dict[str, Any]:
        """Get validator information for debugging."""
        return {
            "name": self.VALIDATOR_NAME,
            "version": self.VALIDATOR_VERSION,
            "config": {
                "strict_mode": self.config.strict_mode,
                "warn_on_missing": self.config.warn_on_missing,
                "max_errors": self.config.max_errors,
            },
            "stats": self._stats.to_dict(),
        }


class PathValidator(BaseValidator):
    """Base class for validators that operate on file paths.

    Provides common path validation utilities.
    """

    VALIDATOR_NAME = "path"
    VALIDATOR_VERSION = "1.0.0"

    def validate(self, path: str | Path) -> ValidationResult:
        """Validate a file path.

        Args:
            path: File path to validate

        Returns:
            ValidationResult with validity status
        """
        path = Path(path) if isinstance(path, str) else path
        return super().validate(path)

    def _validate(self, path: Path) -> ValidationResult:
        """Validate the file exists and is readable."""
        if not path.exists():
            return ValidationResult.invalid(f"File not found: {path}")

        if not path.is_file():
            return ValidationResult.invalid(f"Not a file: {path}")

        return self._validate_file(path)

    @abstractmethod
    def _validate_file(self, path: Path) -> ValidationResult:
        """Validate file contents.

        Subclasses must implement this method.

        Args:
            path: Path to validated file

        Returns:
            ValidationResult with validity status
        """
        ...
