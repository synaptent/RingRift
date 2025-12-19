"""UnifiedDataValidator - Consolidated facade for all data validation (December 2025).

This module provides a single entry point for all data validation needs,
consolidating multiple validators into a unified interface:

- Training data validation (NPZ/HDF5)
- NNUE dataset validation
- Territory dataset validation
- Database integrity validation

Benefits:
- Consistent result types across all validation types
- Unified metrics and event emission
- Single import for all validation needs
- Graceful fallback when specific validators unavailable

Usage:
    from app.training.unified_data_validator import (
        UnifiedDataValidator,
        validate_training_data,
        validate_database,
        get_validator,
    )

    # Get singleton validator
    validator = get_validator()

    # Validate training data
    result = validator.validate_training_file("data/training.npz")
    if not result.is_valid:
        print(f"Validation failed: {result.summary()}")

    # Validate database
    result = validator.validate_database("data/games.db")

    # Use convenience functions
    result = validate_training_data("data/training.npz")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ValidationType(Enum):
    """Types of validation supported."""
    TRAINING_DATA = "training_data"  # NPZ/HDF5 training data
    NNUE_DATASET = "nnue_dataset"  # NNUE-specific dataset
    TERRITORY_DATASET = "territory_dataset"  # Territory JSONL
    DATABASE = "database"  # SQLite game database
    CONFIG = "config"  # Configuration validation


class ValidationSeverity(Enum):
    """Severity of validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class UnifiedValidationIssue:
    """A validation issue with unified format."""
    issue_type: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    sample_index: Optional[int] = None
    file_path: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_type": self.issue_type,
            "message": self.message,
            "severity": self.severity.value,
            "sample_index": self.sample_index,
            "file_path": self.file_path,
            "details": self.details,
        }


@dataclass
class UnifiedValidationResult:
    """Unified result from any validation operation."""
    is_valid: bool
    validation_type: ValidationType
    source_path: str
    total_items: int = 0
    items_with_issues: int = 0
    issues: List[UnifiedValidationIssue] = field(default_factory=list)
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_issue(
        self,
        issue_type: str,
        message: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        **kwargs,
    ) -> None:
        """Add a validation issue."""
        self.issues.append(UnifiedValidationIssue(
            issue_type=issue_type,
            message=message,
            severity=severity,
            **kwargs,
        ))
        if severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL):
            self.is_valid = False

    def summary(self) -> str:
        """Get a human-readable summary."""
        if self.is_valid:
            return (
                f"✓ Valid [{self.validation_type.value}]: "
                f"{self.total_items} items, {len(self.issues)} warnings"
            )

        # Count by severity
        errors = sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)
        critical = sum(1 for i in self.issues if i.severity == ValidationSeverity.CRITICAL)
        warnings = sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

        return (
            f"✗ Invalid [{self.validation_type.value}]: "
            f"{critical} critical, {errors} errors, {warnings} warnings "
            f"in {self.items_with_issues}/{self.total_items} items"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "validation_type": self.validation_type.value,
            "source_path": self.source_path,
            "total_items": self.total_items,
            "items_with_issues": self.items_with_issues,
            "issues": [i.to_dict() for i in self.issues],
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }


class UnifiedDataValidator:
    """Unified facade for all data validation needs.

    Provides a single entry point that delegates to appropriate validators
    based on data type, with consistent result handling.
    """

    _instance: Optional["UnifiedDataValidator"] = None

    def __init__(self):
        """Initialize the unified validator."""
        self._training_validator = None
        self._nnue_validator = None
        self._db_validator = None
        self._config_validator = None

        # Statistics
        self._validations_run = 0
        self._validations_passed = 0
        self._validations_failed = 0

    @classmethod
    def get_instance(cls) -> "UnifiedDataValidator":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_training_validator(self):
        """Lazy-load training data validator."""
        if self._training_validator is None:
            try:
                from app.training.data_validation import DataValidator
                self._training_validator = DataValidator()
            except ImportError:
                logger.debug("Training data validator not available")
        return self._training_validator

    def validate_training_file(
        self,
        path: Union[str, Path],
        **kwargs,
    ) -> UnifiedValidationResult:
        """Validate a training data file (NPZ/HDF5).

        Args:
            path: Path to the training data file

        Returns:
            UnifiedValidationResult with validation results
        """
        start = time.time()
        path = Path(path)

        result = UnifiedValidationResult(
            is_valid=True,
            validation_type=ValidationType.TRAINING_DATA,
            source_path=str(path),
        )

        validator = self._get_training_validator()
        if validator is None:
            result.add_issue(
                "validator_unavailable",
                "Training data validator not available",
                severity=ValidationSeverity.WARNING,
            )
            result.duration_seconds = time.time() - start
            return result

        try:
            # Use the underlying validator
            raw_result = validator.validate_npz(path)

            result.total_items = raw_result.total_samples
            result.items_with_issues = raw_result.samples_with_issues
            result.is_valid = raw_result.valid

            # Convert issues to unified format
            for issue in raw_result.issues:
                result.add_issue(
                    issue_type=issue.issue_type.value,
                    message=issue.message,
                    severity=ValidationSeverity.ERROR,
                    sample_index=issue.sample_index,
                    file_path=str(path),
                    details=issue.details,
                )

            # Preserve statistics
            if raw_result.policy_sum_stats:
                result.metadata["policy_sum_stats"] = raw_result.policy_sum_stats
            if raw_result.value_stats:
                result.metadata["value_stats"] = raw_result.value_stats

        except Exception as e:
            result.add_issue(
                "validation_error",
                f"Validation failed with error: {e}",
                severity=ValidationSeverity.CRITICAL,
            )

        result.duration_seconds = time.time() - start
        self._record_validation(result)
        return result

    def validate_database(
        self,
        path: Union[str, Path],
        sample_size: int = 100,
        **kwargs,
    ) -> UnifiedValidationResult:
        """Validate a SQLite game database.

        Args:
            path: Path to the database file
            sample_size: Number of games to sample for validation

        Returns:
            UnifiedValidationResult with validation results
        """
        start = time.time()
        path = Path(path)

        result = UnifiedValidationResult(
            is_valid=True,
            validation_type=ValidationType.DATABASE,
            source_path=str(path),
        )

        try:
            from app.training.nnue_dataset import validate_database_integrity

            is_valid, stats = validate_database_integrity(str(path))

            result.is_valid = is_valid
            result.total_items = stats.get("total_games", 0)
            result.metadata = stats

            if not is_valid:
                result.add_issue(
                    "integrity_check_failed",
                    stats.get("error", "Database integrity check failed"),
                    severity=ValidationSeverity.ERROR,
                )

        except ImportError:
            # Try alternate validation
            try:
                from app.db.validation import validate_database_summary

                summary = validate_database_summary(str(path), sample_size)
                result.total_items = summary.get("total_games", 0)
                result.metadata = summary

                if summary.get("invalid_games", 0) > 0:
                    result.is_valid = False
                    result.items_with_issues = summary["invalid_games"]
                    result.add_issue(
                        "invalid_games",
                        f"{summary['invalid_games']} games failed validation",
                        severity=ValidationSeverity.ERROR,
                    )

            except ImportError:
                result.add_issue(
                    "validator_unavailable",
                    "Database validator not available",
                    severity=ValidationSeverity.WARNING,
                )

        except Exception as e:
            result.add_issue(
                "validation_error",
                f"Database validation failed: {e}",
                severity=ValidationSeverity.CRITICAL,
            )

        result.duration_seconds = time.time() - start
        self._record_validation(result)
        return result

    def validate_nnue_dataset(
        self,
        samples: List[Any],
        feature_dim: int,
        **kwargs,
    ) -> UnifiedValidationResult:
        """Validate an NNUE dataset.

        Args:
            samples: List of NNUE samples
            feature_dim: Expected feature dimension

        Returns:
            UnifiedValidationResult with validation results
        """
        start = time.time()

        result = UnifiedValidationResult(
            is_valid=True,
            validation_type=ValidationType.NNUE_DATASET,
            source_path="<in-memory>",
            total_items=len(samples),
        )

        try:
            from app.training.nnue_dataset import validate_nnue_dataset

            raw_result = validate_nnue_dataset(samples, feature_dim, **kwargs)

            result.is_valid = raw_result.is_valid
            result.total_items = raw_result.total_samples
            result.items_with_issues = raw_result.invalid_samples

            if raw_result.feature_dim_errors > 0:
                result.add_issue(
                    "feature_dim_mismatch",
                    f"{raw_result.feature_dim_errors} samples have wrong feature dimensions",
                    severity=ValidationSeverity.ERROR,
                )

            if raw_result.value_range_errors > 0:
                result.add_issue(
                    "value_out_of_range",
                    f"{raw_result.value_range_errors} samples have value out of range",
                    severity=ValidationSeverity.ERROR,
                )

            if raw_result.nan_inf_errors > 0:
                result.add_issue(
                    "nan_inf_values",
                    f"{raw_result.nan_inf_errors} samples have NaN/Inf values",
                    severity=ValidationSeverity.CRITICAL,
                )

        except ImportError:
            result.add_issue(
                "validator_unavailable",
                "NNUE validator not available",
                severity=ValidationSeverity.WARNING,
            )
        except Exception as e:
            result.add_issue(
                "validation_error",
                f"NNUE validation failed: {e}",
                severity=ValidationSeverity.CRITICAL,
            )

        result.duration_seconds = time.time() - start
        self._record_validation(result)
        return result

    def validate_territory_dataset(
        self,
        path: Union[str, Path],
        max_errors: int = 50,
        **kwargs,
    ) -> UnifiedValidationResult:
        """Validate a territory dataset JSONL file.

        Args:
            path: Path to the JSONL file
            max_errors: Maximum errors to collect

        Returns:
            UnifiedValidationResult with validation results
        """
        start = time.time()
        path = Path(path)

        result = UnifiedValidationResult(
            is_valid=True,
            validation_type=ValidationType.TERRITORY_DATASET,
            source_path=str(path),
        )

        try:
            from app.training.territory_dataset_validation import validate_territory_dataset_file

            errors = validate_territory_dataset_file(str(path), max_errors=max_errors)

            result.is_valid = len(errors) == 0

            for line_num, error_msg in errors:
                result.add_issue(
                    "schema_error",
                    error_msg,
                    severity=ValidationSeverity.ERROR,
                    sample_index=line_num,
                    file_path=str(path),
                )

            result.items_with_issues = len(errors)

        except ImportError:
            result.add_issue(
                "validator_unavailable",
                "Territory validator not available",
                severity=ValidationSeverity.WARNING,
            )
        except Exception as e:
            result.add_issue(
                "validation_error",
                f"Territory validation failed: {e}",
                severity=ValidationSeverity.CRITICAL,
            )

        result.duration_seconds = time.time() - start
        self._record_validation(result)
        return result

    def validate(
        self,
        path: Union[str, Path],
        validation_type: Optional[ValidationType] = None,
        **kwargs,
    ) -> UnifiedValidationResult:
        """Validate any supported data type (auto-detects type from extension).

        Args:
            path: Path to the data file
            validation_type: Explicit validation type (auto-detected if None)

        Returns:
            UnifiedValidationResult with validation results
        """
        path = Path(path)

        # Auto-detect type from extension if not specified
        if validation_type is None:
            suffix = path.suffix.lower()
            if suffix in (".npz", ".hdf5", ".h5"):
                validation_type = ValidationType.TRAINING_DATA
            elif suffix in (".db", ".sqlite", ".sqlite3"):
                validation_type = ValidationType.DATABASE
            elif suffix in (".jsonl", ".json"):
                validation_type = ValidationType.TERRITORY_DATASET
            else:
                return UnifiedValidationResult(
                    is_valid=False,
                    validation_type=ValidationType.TRAINING_DATA,
                    source_path=str(path),
                    issues=[UnifiedValidationIssue(
                        issue_type="unknown_type",
                        message=f"Cannot auto-detect validation type for extension: {suffix}",
                        severity=ValidationSeverity.ERROR,
                    )],
                )

        # Dispatch to appropriate validator
        if validation_type == ValidationType.TRAINING_DATA:
            return self.validate_training_file(path, **kwargs)
        elif validation_type == ValidationType.DATABASE:
            return self.validate_database(path, **kwargs)
        elif validation_type == ValidationType.TERRITORY_DATASET:
            return self.validate_territory_dataset(path, **kwargs)
        else:
            return UnifiedValidationResult(
                is_valid=False,
                validation_type=validation_type,
                source_path=str(path),
                issues=[UnifiedValidationIssue(
                    issue_type="unsupported_type",
                    message=f"Validation type not supported: {validation_type}",
                    severity=ValidationSeverity.ERROR,
                )],
            )

    def _record_validation(self, result: UnifiedValidationResult) -> None:
        """Record validation result for statistics and events."""
        self._validations_run += 1
        if result.is_valid:
            self._validations_passed += 1
        else:
            self._validations_failed += 1

        # Emit event for monitoring
        self._emit_validation_event(result)

    def _emit_validation_event(self, result: UnifiedValidationResult) -> None:
        """Emit validation event for monitoring."""
        try:
            from app.distributed.data_events import (
                DataEvent,
                DataEventType,
                get_event_bus,
            )

            event = DataEvent(
                event_type=DataEventType.DATA_QUALITY_ALERT if not result.is_valid
                else DataEventType.QUALITY_CHECK_FAILED,  # Use existing event type
                payload={
                    "validation_type": result.validation_type.value,
                    "source_path": result.source_path,
                    "is_valid": result.is_valid,
                    "total_items": result.total_items,
                    "items_with_issues": result.items_with_issues,
                    "issue_count": len(result.issues),
                    "duration_seconds": result.duration_seconds,
                },
                source="unified_data_validator",
            )

            bus = get_event_bus()
            bus.publish_sync(event)

        except Exception as e:
            logger.debug(f"Failed to emit validation event: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            "validations_run": self._validations_run,
            "validations_passed": self._validations_passed,
            "validations_failed": self._validations_failed,
            "pass_rate": (
                self._validations_passed / self._validations_run
                if self._validations_run > 0 else 1.0
            ),
        }


# Singleton access
_validator: Optional[UnifiedDataValidator] = None


def get_validator() -> UnifiedDataValidator:
    """Get the global UnifiedDataValidator singleton."""
    global _validator
    if _validator is None:
        _validator = UnifiedDataValidator.get_instance()
    return _validator


# Convenience functions

def validate_training_data(path: Union[str, Path], **kwargs) -> UnifiedValidationResult:
    """Validate training data file."""
    return get_validator().validate_training_file(path, **kwargs)


def validate_database(path: Union[str, Path], **kwargs) -> UnifiedValidationResult:
    """Validate game database."""
    return get_validator().validate_database(path, **kwargs)


def validate_any(path: Union[str, Path], **kwargs) -> UnifiedValidationResult:
    """Validate any supported data type (auto-detects)."""
    return get_validator().validate(path, **kwargs)


__all__ = [
    "UnifiedDataValidator",
    "UnifiedValidationResult",
    "UnifiedValidationIssue",
    "ValidationType",
    "ValidationSeverity",
    "get_validator",
    "validate_training_data",
    "validate_database",
    "validate_any",
]
