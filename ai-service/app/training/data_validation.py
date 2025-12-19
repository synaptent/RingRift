"""Training Data Validation Module.

.. deprecated::
    For new code, prefer importing from unified_data_validator:

        from app.training.unified_data_validator import (
            get_validator,
            validate_training_data,
            UnifiedDataValidator,
            # Original exports also available for compatibility:
            DataValidator,
            DataValidatorConfig,
            validate_npz_file,
        )

    The unified validator provides a consistent interface across all
    validation types (training data, NNUE, database, etc.).

Validates self-play training data before it's used for training.
Catches corrupt data, policy/value issues, and feature anomalies.

Per Section 4.4 of the action plan, this module provides:
- Validation of individual samples and entire datasets
- Policy target validation (sum to 1)
- Value target validation (valid range)
- Feature validation (correct dimensions, no NaN/Inf)
- Deduplication support (via GameDeduplicator)

Usage:
    from app.training.data_validation import (
        DataValidator,
        GameDeduplicator,
        validate_npz_file,
    )

    # Validate a training data file
    result = validate_npz_file("data/training_data.npz")
    if not result.valid:
        print(f"Validation failed: {result.issues}")

    # Use deduplicator during data generation
    dedup = GameDeduplicator()
    for game in games:
        if not dedup.is_duplicate(game.moves):
            add_to_dataset(game)

See docs/COMPREHENSIVE_ACTION_PLAN_2025_12_17.md Section 4.4 for context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from app.utils.checksum_utils import compute_string_checksum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ValidationIssueType(Enum):
    """Types of validation issues."""
    POLICY_SUM_INVALID = "policy_sum_invalid"
    POLICY_NEGATIVE = "policy_negative"
    VALUE_OUT_OF_RANGE = "value_out_of_range"
    FEATURE_NAN = "feature_nan"
    FEATURE_INF = "feature_inf"
    DIMENSION_MISMATCH = "dimension_mismatch"
    EMPTY_POLICY = "empty_policy"
    MISSING_ARRAY = "missing_array"
    CORRUPT_FILE = "corrupt_file"


@dataclass
class ValidationIssue:
    """A single validation issue found in the data."""
    issue_type: ValidationIssueType
    message: str
    sample_index: Optional[int] = None
    details: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        if self.sample_index is not None:
            return f"[{self.issue_type.value}] Sample {self.sample_index}: {self.message}"
        return f"[{self.issue_type.value}] {self.message}"


@dataclass
class ValidationResult:
    """Result of validating training data."""
    valid: bool
    total_samples: int
    issues: List[ValidationIssue] = field(default_factory=list)

    # Statistics
    samples_with_issues: int = 0
    policy_sum_stats: Optional[Dict[str, float]] = None
    value_stats: Optional[Dict[str, float]] = None

    def add_issue(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        self.valid = False

    def summary(self) -> str:
        """Human-readable summary."""
        if self.valid:
            return f"✓ Valid: {self.total_samples} samples, no issues"

        issue_counts = {}
        for issue in self.issues:
            key = issue.issue_type.value
            issue_counts[key] = issue_counts.get(key, 0) + 1

        lines = [f"✗ Invalid: {len(self.issues)} issues in {self.samples_with_issues}/{self.total_samples} samples"]
        for issue_type, count in sorted(issue_counts.items()):
            lines.append(f"  - {issue_type}: {count}")
        return "\n".join(lines)


@dataclass
class DataValidatorConfig:
    """Configuration for data validation."""
    # Policy validation
    policy_sum_tolerance: float = 0.01  # Allow 1% deviation from 1.0
    allow_empty_policies: bool = True  # Terminal states may have empty policies

    # Value validation
    min_value: float = -1.0
    max_value: float = 1.0

    # Feature validation
    check_nan: bool = True
    check_inf: bool = True

    # Reporting
    max_issues_to_report: int = 100  # Don't overwhelm logs
    sample_rate: float = 1.0  # Check all samples by default


class DataValidator:
    """Validates self-play training data for quality issues.

    Checks for:
    - Policy targets summing to 1 (within tolerance)
    - Value targets within valid range [-1, 1]
    - No NaN or Inf values in features
    - Correct array dimensions
    """

    def __init__(self, config: Optional[DataValidatorConfig] = None):
        self.config = config or DataValidatorConfig()

    def validate_npz(self, path: Path) -> ValidationResult:
        """Validate an NPZ training data file.

        Args:
            path: Path to the .npz file

        Returns:
            ValidationResult with any issues found
        """
        path = Path(path)

        try:
            data = np.load(path, allow_pickle=True, mmap_mode='r')
        except Exception as e:
            result = ValidationResult(valid=False, total_samples=0)
            result.add_issue(ValidationIssue(
                issue_type=ValidationIssueType.CORRUPT_FILE,
                message=f"Failed to load file: {e}",
            ))
            return result

        return self.validate_arrays(dict(data))

    def validate_arrays(self, data: Dict[str, np.ndarray]) -> ValidationResult:
        """Validate training data arrays.

        Args:
            data: Dictionary of numpy arrays (features, values, policy_*, etc.)

        Returns:
            ValidationResult with any issues found
        """
        # Check required arrays exist
        required = ['values']
        for key in required:
            if key not in data:
                result = ValidationResult(valid=False, total_samples=0)
                result.add_issue(ValidationIssue(
                    issue_type=ValidationIssueType.MISSING_ARRAY,
                    message=f"Missing required array: {key}",
                ))
                return result

        values = data['values']
        total_samples = len(values)
        result = ValidationResult(valid=True, total_samples=total_samples)

        # Track samples with issues
        samples_with_issues: Set[int] = set()

        # Validate values
        value_issues = self._validate_values(values)
        for issue in value_issues:
            if len(result.issues) < self.config.max_issues_to_report:
                result.add_issue(issue)
            if issue.sample_index is not None:
                samples_with_issues.add(issue.sample_index)

        # Compute value statistics
        result.value_stats = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

        # Validate policies if present
        if 'policy_values' in data:
            policy_values = data['policy_values']
            policy_indices = data.get('policy_indices')

            policy_issues = self._validate_policies(policy_values, policy_indices)
            for issue in policy_issues:
                if len(result.issues) < self.config.max_issues_to_report:
                    result.add_issue(issue)
                if issue.sample_index is not None:
                    samples_with_issues.add(issue.sample_index)

        # Validate features if present
        if 'features' in data:
            features = data['features']
            feature_issues = self._validate_features(features)
            for issue in feature_issues:
                if len(result.issues) < self.config.max_issues_to_report:
                    result.add_issue(issue)
                if issue.sample_index is not None:
                    samples_with_issues.add(issue.sample_index)

        result.samples_with_issues = len(samples_with_issues)
        result.valid = len(result.issues) == 0

        return result

    def _validate_values(self, values: np.ndarray) -> List[ValidationIssue]:
        """Validate value targets."""
        issues = []

        # Check for out-of-range values
        out_of_range = np.where(
            (values < self.config.min_value) | (values > self.config.max_value)
        )[0]

        for idx in out_of_range[:self.config.max_issues_to_report]:
            issues.append(ValidationIssue(
                issue_type=ValidationIssueType.VALUE_OUT_OF_RANGE,
                message=f"Value {values[idx]:.4f} outside [{self.config.min_value}, {self.config.max_value}]",
                sample_index=int(idx),
                details={"value": float(values[idx])},
            ))

        # Check for NaN
        nan_indices = np.where(np.isnan(values))[0]
        for idx in nan_indices[:self.config.max_issues_to_report]:
            issues.append(ValidationIssue(
                issue_type=ValidationIssueType.FEATURE_NAN,
                message="Value is NaN",
                sample_index=int(idx),
            ))

        return issues

    def _validate_policies(
        self,
        policy_values: np.ndarray,
        policy_indices: Optional[np.ndarray] = None,
    ) -> List[ValidationIssue]:
        """Validate policy targets."""
        issues = []

        # For sparse policies, we need to compute sums per sample
        if policy_indices is not None:
            # Sparse format: policy_indices gives sample boundaries
            # This format varies - handle common cases
            pass
        else:
            # Dense policy format - each row should sum to 1
            if len(policy_values.shape) == 2:
                policy_sums = np.sum(policy_values, axis=1)

                # Check sums
                invalid_sums = np.where(
                    np.abs(policy_sums - 1.0) > self.config.policy_sum_tolerance
                )[0]

                # Filter out empty policies if allowed
                if self.config.allow_empty_policies:
                    invalid_sums = [
                        idx for idx in invalid_sums
                        if policy_sums[idx] > self.config.policy_sum_tolerance  # Not empty
                    ]

                for idx in invalid_sums[:self.config.max_issues_to_report]:
                    issues.append(ValidationIssue(
                        issue_type=ValidationIssueType.POLICY_SUM_INVALID,
                        message=f"Policy sum {policy_sums[idx]:.4f} != 1.0",
                        sample_index=int(idx),
                        details={"sum": float(policy_sums[idx])},
                    ))

                # Compute statistics
                self._policy_sum_stats = {
                    "min": float(np.min(policy_sums)),
                    "max": float(np.max(policy_sums)),
                    "mean": float(np.mean(policy_sums)),
                }

        # Check for negative probabilities
        # Skip check for object arrays (variable-length sparse format)
        if policy_values.dtype == object:
            return issues
        negative_indices = np.where(policy_values < 0)[0]
        reported = set()
        for idx in negative_indices:
            sample_idx = idx if len(policy_values.shape) == 1 else idx // policy_values.shape[1]
            if sample_idx not in reported and len(issues) < self.config.max_issues_to_report:
                issues.append(ValidationIssue(
                    issue_type=ValidationIssueType.POLICY_NEGATIVE,
                    message="Policy contains negative probability",
                    sample_index=int(sample_idx),
                ))
                reported.add(sample_idx)

        return issues

    def _validate_features(self, features: np.ndarray) -> List[ValidationIssue]:
        """Validate feature arrays."""
        issues = []

        if self.config.check_nan:
            nan_mask = np.isnan(features)
            if np.any(nan_mask):
                # Find first few samples with NaN
                if len(features.shape) > 1:
                    nan_samples = np.where(np.any(nan_mask, axis=tuple(range(1, len(features.shape)))))[0]
                else:
                    nan_samples = np.where(nan_mask)[0]

                for idx in nan_samples[:self.config.max_issues_to_report]:
                    issues.append(ValidationIssue(
                        issue_type=ValidationIssueType.FEATURE_NAN,
                        message="Features contain NaN values",
                        sample_index=int(idx),
                    ))

        if self.config.check_inf:
            inf_mask = np.isinf(features)
            if np.any(inf_mask):
                if len(features.shape) > 1:
                    inf_samples = np.where(np.any(inf_mask, axis=tuple(range(1, len(features.shape)))))[0]
                else:
                    inf_samples = np.where(inf_mask)[0]

                for idx in inf_samples[:self.config.max_issues_to_report]:
                    issues.append(ValidationIssue(
                        issue_type=ValidationIssueType.FEATURE_INF,
                        message="Features contain Inf values",
                        sample_index=int(idx),
                    ))

        return issues


class GameDeduplicator:
    """Deduplicate games based on move sequence hash.

    Used during data generation to avoid training on duplicate games
    which can lead to overfitting on specific positions.

    Usage:
        dedup = GameDeduplicator()

        for game in games:
            if not dedup.is_duplicate(game.moves):
                training_data.append(game)
            else:
                stats['duplicates'] += 1
    """

    def __init__(self, hash_prefix_length: int = 16):
        """Initialize deduplicator.

        Args:
            hash_prefix_length: Length of hash prefix to store (memory vs collision tradeoff)
        """
        self.seen_hashes: Set[str] = set()
        self.hash_prefix_length = hash_prefix_length
        self.total_checked = 0
        self.duplicates_found = 0

    def is_duplicate(self, moves: List[Any]) -> bool:
        """Check if a game (by move sequence) is a duplicate.

        Args:
            moves: List of moves in the game

        Returns:
            True if this move sequence was seen before
        """
        self.total_checked += 1
        game_hash = self._compute_game_hash(moves)

        if game_hash in self.seen_hashes:
            self.duplicates_found += 1
            return True

        self.seen_hashes.add(game_hash)
        return False

    def is_duplicate_by_hash(self, game_hash: str) -> bool:
        """Check if a game hash is a duplicate.

        Args:
            game_hash: Pre-computed game hash

        Returns:
            True if this hash was seen before
        """
        self.total_checked += 1
        truncated = game_hash[:self.hash_prefix_length]

        if truncated in self.seen_hashes:
            self.duplicates_found += 1
            return True

        self.seen_hashes.add(truncated)
        return False

    def _compute_game_hash(self, moves: List[Any]) -> str:
        """Compute hash of move sequence."""
        moves_str = "|".join(str(m) for m in moves)
        return compute_string_checksum(moves_str, truncate=self.hash_prefix_length)

    def reset(self) -> None:
        """Clear seen hashes and reset statistics."""
        self.seen_hashes.clear()
        self.total_checked = 0
        self.duplicates_found = 0

    def stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        return {
            "total_checked": self.total_checked,
            "duplicates_found": self.duplicates_found,
            "unique_games": len(self.seen_hashes),
            "duplicate_rate": self.duplicates_found / max(1, self.total_checked),
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def validate_npz_file(
    path: str,
    config: Optional[DataValidatorConfig] = None,
) -> ValidationResult:
    """Validate an NPZ training data file.

    Args:
        path: Path to the .npz file
        config: Optional validation configuration

    Returns:
        ValidationResult with any issues found
    """
    validator = DataValidator(config)
    return validator.validate_npz(Path(path))


def validate_training_data(
    data: Dict[str, np.ndarray],
    config: Optional[DataValidatorConfig] = None,
) -> ValidationResult:
    """Validate training data arrays.

    Args:
        data: Dictionary of numpy arrays
        config: Optional validation configuration

    Returns:
        ValidationResult with any issues found
    """
    validator = DataValidator(config)
    return validator.validate_arrays(data)


# =============================================================================
# Prometheus Metrics (Optional)
# =============================================================================

try:
    from prometheus_client import Counter, Gauge, Histogram

    VALIDATION_RUNS = Counter(
        'ringrift_data_validation_runs_total',
        'Total data validation runs',
        ['result']
    )

    VALIDATION_ISSUES = Counter(
        'ringrift_data_validation_issues_total',
        'Total validation issues found',
        ['issue_type']
    )

    VALIDATION_SAMPLES = Gauge(
        'ringrift_data_validation_samples',
        'Samples in last validation',
        ['status']
    )

    DUPLICATE_RATE = Gauge(
        'ringrift_data_duplicate_rate',
        'Rate of duplicate games detected'
    )

    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False


def record_validation_metrics(result: ValidationResult) -> None:
    """Record validation results to Prometheus metrics."""
    if not HAS_METRICS:
        return

    VALIDATION_RUNS.labels(result='valid' if result.valid else 'invalid').inc()
    VALIDATION_SAMPLES.labels(status='total').set(result.total_samples)
    VALIDATION_SAMPLES.labels(status='with_issues').set(result.samples_with_issues)

    for issue in result.issues:
        VALIDATION_ISSUES.labels(issue_type=issue.issue_type.value).inc()


def record_deduplication_metrics(dedup: GameDeduplicator) -> None:
    """Record deduplication stats to Prometheus metrics."""
    if not HAS_METRICS:
        return

    stats = dedup.stats()
    DUPLICATE_RATE.set(stats['duplicate_rate'])
