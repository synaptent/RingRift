"""Tests for app.training.data_validation module.

This module tests:
- ValidationIssueType enum
- ValidationIssue dataclass
- ValidationResult dataclass
- DataValidatorConfig dataclass
- DataValidator class
- GameDeduplicator class
- NPZHeaderValidation dataclass
- Convenience functions
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.training.data_validation import (
    DataValidator,
    DataValidatorConfig,
    GameDeduplicator,
    NPZHeaderValidation,
    ValidationIssue,
    ValidationIssueType,
    ValidationResult,
    is_npz_valid,
    validate_npz_file,
    validate_npz_header,
    validate_training_data,
)


# =============================================================================
# ValidationIssueType Tests
# =============================================================================


class TestValidationIssueType:
    """Tests for ValidationIssueType enum."""

    def test_all_issue_types_exist(self):
        """All expected issue types are defined."""
        expected_types = [
            "CORRUPT_FILE",
            "MISSING_ARRAY",
            "POLICY_SUM_INVALID",
            "POLICY_NEGATIVE",
            "VALUE_OUT_OF_RANGE",
            "FEATURE_NAN",
            "FEATURE_INF",
            "DIMENSION_MISMATCH",
            "EMPTY_DATA",
        ]
        for issue_type in expected_types:
            assert hasattr(ValidationIssueType, issue_type)

    def test_issue_type_values_are_strings(self):
        """Issue type values are strings for logging."""
        for issue_type in ValidationIssueType:
            assert isinstance(issue_type.value, str)


# =============================================================================
# ValidationIssue Tests
# =============================================================================


class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_create_with_required_fields(self):
        """Create issue with required fields only."""
        issue = ValidationIssue(
            issue_type=ValidationIssueType.FEATURE_NAN,
            message="NaN found in features",
        )
        assert issue.issue_type == ValidationIssueType.FEATURE_NAN
        assert issue.message == "NaN found in features"
        assert issue.sample_index is None
        assert issue.details is None

    def test_create_with_all_fields(self):
        """Create issue with all fields."""
        issue = ValidationIssue(
            issue_type=ValidationIssueType.VALUE_OUT_OF_RANGE,
            message="Value 1.5 outside [-1, 1]",
            sample_index=42,
            details={"value": 1.5},
        )
        assert issue.sample_index == 42
        assert issue.details == {"value": 1.5}


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_create_valid_result(self):
        """Create valid result with no issues."""
        result = ValidationResult(valid=True, total_samples=100)
        assert result.valid is True
        assert result.total_samples == 100
        assert len(result.issues) == 0
        assert result.samples_with_issues == 0

    def test_add_issue_sets_valid_false(self):
        """Adding an issue sets valid to False."""
        result = ValidationResult(valid=True, total_samples=100)
        issue = ValidationIssue(
            issue_type=ValidationIssueType.FEATURE_NAN,
            message="test",
        )
        result.add_issue(issue)
        assert result.valid is False
        assert len(result.issues) == 1

    def test_summary_for_valid_result(self):
        """Summary for valid result shows checkmark."""
        result = ValidationResult(valid=True, total_samples=100)
        summary = result.summary()
        assert "✓" in summary
        assert "100 samples" in summary
        assert "no issues" in summary

    def test_summary_for_invalid_result(self):
        """Summary for invalid result shows issues."""
        result = ValidationResult(valid=True, total_samples=100)
        result.add_issue(ValidationIssue(
            issue_type=ValidationIssueType.FEATURE_NAN,
            message="test",
            sample_index=0,
        ))
        result.samples_with_issues = 1
        summary = result.summary()
        assert "✗" in summary
        assert "1 issues" in summary


# =============================================================================
# DataValidatorConfig Tests
# =============================================================================


class TestDataValidatorConfig:
    """Tests for DataValidatorConfig dataclass."""

    def test_default_values(self):
        """Default configuration values are sensible."""
        config = DataValidatorConfig()
        assert config.policy_sum_tolerance == 0.01
        assert config.allow_empty_policies is True
        assert config.min_value == -1.0
        assert config.max_value == 1.0
        assert config.check_nan is True
        assert config.check_inf is True
        assert config.max_issues_to_report == 100

    def test_custom_values(self):
        """Custom configuration values override defaults."""
        config = DataValidatorConfig(
            policy_sum_tolerance=0.05,
            allow_empty_policies=False,
            max_issues_to_report=10,
        )
        assert config.policy_sum_tolerance == 0.05
        assert config.allow_empty_policies is False
        assert config.max_issues_to_report == 10


# =============================================================================
# DataValidator Tests
# =============================================================================


class TestDataValidator:
    """Tests for DataValidator class."""

    def test_init_with_default_config(self):
        """Initialize validator with default config."""
        validator = DataValidator()
        assert validator.config is not None

    def test_init_with_custom_config(self):
        """Initialize validator with custom config."""
        config = DataValidatorConfig(policy_sum_tolerance=0.1)
        validator = DataValidator(config)
        assert validator.config.policy_sum_tolerance == 0.1

    def test_validate_arrays_missing_values(self):
        """Detect missing values array."""
        validator = DataValidator()
        data = {"features": np.zeros((10, 5, 5))}  # No values array
        result = validator.validate_arrays(data)
        assert result.valid is False
        assert any(
            i.issue_type == ValidationIssueType.MISSING_ARRAY
            for i in result.issues
        )

    def test_validate_arrays_valid_data(self):
        """Valid data passes validation."""
        validator = DataValidator()
        data = {
            "values": np.random.uniform(-1, 1, 100),
            "features": np.random.randn(100, 5, 5),
            "policy_values": np.ones((100, 10)) / 10,
        }
        result = validator.validate_arrays(data)
        assert result.valid is True
        assert len(result.issues) == 0

    def test_validate_values_out_of_range(self):
        """Detect values outside [-1, 1] range."""
        validator = DataValidator()
        data = {
            "values": np.array([0.5, 1.5, -0.5, -1.5, 0.0]),  # 1.5 and -1.5 are OOR
        }
        result = validator.validate_arrays(data)
        assert result.valid is False
        oor_issues = [
            i for i in result.issues
            if i.issue_type == ValidationIssueType.VALUE_OUT_OF_RANGE
        ]
        assert len(oor_issues) == 2

    def test_validate_values_nan(self):
        """Detect NaN in values."""
        validator = DataValidator()
        data = {
            "values": np.array([0.5, np.nan, -0.5]),
        }
        result = validator.validate_arrays(data)
        assert result.valid is False
        nan_issues = [
            i for i in result.issues
            if i.issue_type == ValidationIssueType.FEATURE_NAN
        ]
        assert len(nan_issues) == 1

    def test_validate_policy_sum_invalid(self):
        """Detect policy sums not equal to 1."""
        validator = DataValidator()
        data = {
            "values": np.zeros(3),
            "policy_values": np.array([
                [0.5, 0.5],  # Sum = 1.0, valid
                [0.3, 0.3],  # Sum = 0.6, invalid
                [1.0, 1.0],  # Sum = 2.0, invalid
            ]),
        }
        result = validator.validate_arrays(data)
        assert result.valid is False
        sum_issues = [
            i for i in result.issues
            if i.issue_type == ValidationIssueType.POLICY_SUM_INVALID
        ]
        assert len(sum_issues) == 2

    def test_validate_policy_empty_allowed(self):
        """Empty policies allowed by default."""
        config = DataValidatorConfig(allow_empty_policies=True)
        validator = DataValidator(config)
        data = {
            "values": np.zeros(2),
            "policy_values": np.array([
                [0.5, 0.5],  # Valid
                [0.0, 0.0],  # Empty, should be allowed
            ]),
        }
        result = validator.validate_arrays(data)
        # No policy sum issues for empty
        sum_issues = [
            i for i in result.issues
            if i.issue_type == ValidationIssueType.POLICY_SUM_INVALID
        ]
        assert len(sum_issues) == 0

    def test_validate_policy_empty_not_allowed(self):
        """Empty policies rejected when configured."""
        config = DataValidatorConfig(allow_empty_policies=False)
        validator = DataValidator(config)
        data = {
            "values": np.zeros(2),
            "policy_values": np.array([
                [0.5, 0.5],  # Valid
                [0.0, 0.0],  # Empty, should be rejected
            ]),
        }
        result = validator.validate_arrays(data)
        sum_issues = [
            i for i in result.issues
            if i.issue_type == ValidationIssueType.POLICY_SUM_INVALID
        ]
        assert len(sum_issues) == 1

    def test_validate_policy_negative_values(self):
        """Detect negative policy values."""
        validator = DataValidator()
        data = {
            "values": np.zeros(2),
            "policy_values": np.array([
                [0.5, 0.5],
                [-0.1, 1.1],  # Negative value
            ]),
        }
        result = validator.validate_arrays(data)
        neg_issues = [
            i for i in result.issues
            if i.issue_type == ValidationIssueType.POLICY_NEGATIVE
        ]
        assert len(neg_issues) == 1

    def test_validate_features_nan(self):
        """Detect NaN in features."""
        validator = DataValidator()
        data = {
            "values": np.zeros(3),
            "features": np.array([
                [[1.0, 2.0], [3.0, 4.0]],
                [[np.nan, 2.0], [3.0, 4.0]],  # NaN here
                [[1.0, 2.0], [3.0, 4.0]],
            ]),
        }
        result = validator.validate_arrays(data)
        nan_issues = [
            i for i in result.issues
            if i.issue_type == ValidationIssueType.FEATURE_NAN
        ]
        assert len(nan_issues) >= 1

    def test_validate_features_inf(self):
        """Detect Inf in features."""
        validator = DataValidator()
        data = {
            "values": np.zeros(3),
            "features": np.array([
                [[1.0, 2.0], [3.0, 4.0]],
                [[np.inf, 2.0], [3.0, 4.0]],  # Inf here
                [[1.0, 2.0], [3.0, 4.0]],
            ]),
        }
        result = validator.validate_arrays(data)
        inf_issues = [
            i for i in result.issues
            if i.issue_type == ValidationIssueType.FEATURE_INF
        ]
        assert len(inf_issues) >= 1

    def test_validate_features_nan_check_disabled(self):
        """NaN check can be disabled."""
        config = DataValidatorConfig(check_nan=False)
        validator = DataValidator(config)
        data = {
            "values": np.zeros(1),
            "features": np.array([[[np.nan]]]),
        }
        result = validator.validate_arrays(data)
        nan_issues = [
            i for i in result.issues
            if i.issue_type == ValidationIssueType.FEATURE_NAN
        ]
        assert len(nan_issues) == 0

    def test_max_issues_to_report_limit(self):
        """Issues limited by max_issues_to_report."""
        config = DataValidatorConfig(max_issues_to_report=3)
        validator = DataValidator(config)
        data = {
            "values": np.array([2.0, 2.0, 2.0, 2.0, 2.0]),  # 5 out of range
        }
        result = validator.validate_arrays(data)
        assert len(result.issues) <= 3

    def test_value_stats_computed(self):
        """Value statistics are computed."""
        validator = DataValidator()
        data = {
            "values": np.array([0.0, 0.5, 1.0, -0.5, -1.0]),
        }
        result = validator.validate_arrays(data)
        assert result.value_stats is not None
        assert "min" in result.value_stats
        assert "max" in result.value_stats
        assert "mean" in result.value_stats
        assert result.value_stats["min"] == -1.0
        assert result.value_stats["max"] == 1.0

    def test_validate_npz_corrupt_file(self):
        """Handle corrupt/missing NPZ file."""
        validator = DataValidator()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            f.write(b"not a valid npz file")
            temp_path = f.name

        try:
            result = validator.validate_npz(Path(temp_path))
            assert result.valid is False
            assert any(
                i.issue_type == ValidationIssueType.CORRUPT_FILE
                for i in result.issues
            )
        finally:
            Path(temp_path).unlink()

    def test_validate_npz_valid_file(self):
        """Validate actual NPZ file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            np.savez(
                path,
                values=np.random.uniform(-1, 1, 100),
                features=np.random.randn(100, 5, 5),
            )
            validator = DataValidator()
            result = validator.validate_npz(path)
            assert result.valid is True


# =============================================================================
# GameDeduplicator Tests
# =============================================================================


class TestGameDeduplicator:
    """Tests for GameDeduplicator class."""

    def test_init_defaults(self):
        """Default initialization."""
        dedup = GameDeduplicator()
        assert dedup.hash_prefix_length == 16
        assert len(dedup.seen_hashes) == 0
        assert dedup.total_checked == 0
        assert dedup.duplicates_found == 0

    def test_init_custom_prefix_length(self):
        """Custom hash prefix length."""
        dedup = GameDeduplicator(hash_prefix_length=8)
        assert dedup.hash_prefix_length == 8

    def test_is_duplicate_unique_games(self):
        """Unique games are not duplicates."""
        dedup = GameDeduplicator()
        moves1 = ["a1", "b2", "c3"]
        moves2 = ["d4", "e5", "f6"]
        assert dedup.is_duplicate(moves1) is False
        assert dedup.is_duplicate(moves2) is False
        assert dedup.duplicates_found == 0
        assert dedup.total_checked == 2

    def test_is_duplicate_same_game(self):
        """Same game detected as duplicate."""
        dedup = GameDeduplicator()
        moves = ["a1", "b2", "c3"]
        assert dedup.is_duplicate(moves) is False  # First time
        assert dedup.is_duplicate(moves) is True  # Duplicate
        assert dedup.duplicates_found == 1

    def test_is_duplicate_by_hash(self):
        """Deduplication by pre-computed hash."""
        dedup = GameDeduplicator(hash_prefix_length=8)
        hash1 = "abcdef123456"
        hash2 = "abcdef123456"  # Same prefix
        hash3 = "xyz789000000"  # Different
        assert dedup.is_duplicate_by_hash(hash1) is False
        assert dedup.is_duplicate_by_hash(hash2) is True  # Same prefix
        assert dedup.is_duplicate_by_hash(hash3) is False

    def test_reset_clears_state(self):
        """Reset clears all state."""
        dedup = GameDeduplicator()
        dedup.is_duplicate(["a", "b"])
        dedup.is_duplicate(["a", "b"])  # Duplicate
        assert len(dedup.seen_hashes) == 1
        assert dedup.duplicates_found == 1

        dedup.reset()
        assert len(dedup.seen_hashes) == 0
        assert dedup.total_checked == 0
        assert dedup.duplicates_found == 0

    def test_stats_returns_metrics(self):
        """Stats returns deduplication metrics."""
        dedup = GameDeduplicator()
        dedup.is_duplicate(["a"])
        dedup.is_duplicate(["b"])
        dedup.is_duplicate(["a"])  # Duplicate

        stats = dedup.stats()
        assert stats["total_checked"] == 3
        assert stats["duplicates_found"] == 1
        assert stats["unique_games"] == 2
        assert stats["duplicate_rate"] == pytest.approx(1 / 3)

    def test_stats_empty(self):
        """Stats handles empty state."""
        dedup = GameDeduplicator()
        stats = dedup.stats()
        assert stats["total_checked"] == 0
        assert stats["duplicate_rate"] == 0.0  # Avoid division by zero


# =============================================================================
# NPZHeaderValidation Tests
# =============================================================================


class TestNPZHeaderValidation:
    """Tests for NPZHeaderValidation dataclass."""

    def test_valid_result(self):
        """Valid result with array info."""
        result = NPZHeaderValidation(
            valid=True,
            file_path="/path/to/file.npz",
            file_size_bytes=1024,
            array_info={"values": {"shape": "(100,)"}},
        )
        assert result.valid is True
        assert "✓" in result.summary()

    def test_invalid_result(self):
        """Invalid result with error."""
        result = NPZHeaderValidation(
            valid=False,
            file_path="/path/to/file.npz",
            error="File not found",
        )
        assert result.valid is False
        assert "✗" in result.summary()
        assert "File not found" in result.summary()


class TestValidateNpzHeader:
    """Tests for validate_npz_header function."""

    def test_nonexistent_file(self):
        """Handle nonexistent file."""
        result = validate_npz_header("/nonexistent/path.npz")
        assert result.valid is False
        assert "does not exist" in result.error

    def test_empty_file(self):
        """Handle empty file."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            temp_path = f.name
        try:
            result = validate_npz_header(temp_path)
            assert result.valid is False
            assert "empty" in result.error.lower()
        finally:
            Path(temp_path).unlink()

    def test_not_zip_file(self):
        """Handle non-ZIP file."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            f.write(b"not a zip file")
            temp_path = f.name
        try:
            result = validate_npz_header(temp_path)
            assert result.valid is False
            assert "ZIP" in result.error or "valid" in result.error.lower()
        finally:
            Path(temp_path).unlink()

    def test_valid_npz(self):
        """Validate valid NPZ file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            np.savez(path, values=np.zeros(10), features=np.ones((10, 5)))
            result = validate_npz_header(path)
            assert result.valid is True
            assert "values" in result.array_info
            assert "features" in result.array_info

    def test_required_arrays_present(self):
        """Check required arrays when present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            np.savez(path, values=np.zeros(10), features=np.ones((10, 5)))
            result = validate_npz_header(
                path,
                required_arrays=["values", "features"],
            )
            assert result.valid is True

    def test_required_arrays_missing(self):
        """Check required arrays when missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            np.savez(path, values=np.zeros(10))
            result = validate_npz_header(
                path,
                required_arrays=["values", "features"],
            )
            assert result.valid is False
            assert "Missing required" in result.error


class TestIsNpzValid:
    """Tests for is_npz_valid convenience function."""

    def test_valid_file(self):
        """Valid file returns True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            np.savez(path, values=np.zeros(10))
            assert is_npz_valid(path) is True

    def test_invalid_file(self):
        """Invalid file returns False."""
        result = is_npz_valid("/nonexistent.npz", log_errors=False)
        assert result is False


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestValidateNpzFile:
    """Tests for validate_npz_file convenience function."""

    def test_valid_file(self):
        """Validate valid NPZ file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            np.savez(
                path,
                values=np.random.uniform(-1, 1, 100),
                features=np.random.randn(100, 5, 5),
            )
            result = validate_npz_file(str(path))
            assert result.valid is True

    def test_with_custom_config(self):
        """Use custom configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            np.savez(path, values=np.array([0.0, 2.0]))  # 2.0 out of range
            config = DataValidatorConfig(max_value=1.0)
            result = validate_npz_file(str(path), config)
            assert result.valid is False


class TestValidateTrainingData:
    """Tests for validate_training_data convenience function."""

    def test_valid_data(self):
        """Validate valid training data."""
        data = {
            "values": np.random.uniform(-1, 1, 100),
            "features": np.random.randn(100, 5, 5),
        }
        result = validate_training_data(data)
        assert result.valid is True

    def test_invalid_data(self):
        """Validate invalid training data."""
        data = {
            "values": np.array([np.nan, 0.5]),  # NaN
        }
        result = validate_training_data(data)
        assert result.valid is False


# =============================================================================
# Prometheus Metrics Tests
# =============================================================================


class TestPrometheusMetrics:
    """Tests for Prometheus metrics recording."""

    def test_record_validation_metrics_no_prometheus(self):
        """Handle missing Prometheus gracefully."""
        from app.training import data_validation

        result = ValidationResult(valid=True, total_samples=100)

        # Should not raise even if Prometheus not available
        data_validation.record_validation_metrics(result)

    def test_record_deduplication_metrics_no_prometheus(self):
        """Handle missing Prometheus gracefully."""
        from app.training import data_validation

        dedup = GameDeduplicator()
        dedup.is_duplicate(["a", "b"])

        # Should not raise even if Prometheus not available
        data_validation.record_deduplication_metrics(dedup)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_data(self):
        """Handle empty arrays."""
        validator = DataValidator()
        data = {"values": np.array([])}
        result = validator.validate_arrays(data)
        assert result.total_samples == 0

    def test_single_sample(self):
        """Handle single sample."""
        validator = DataValidator()
        data = {"values": np.array([0.5])}
        result = validator.validate_arrays(data)
        assert result.valid is True
        assert result.total_samples == 1

    def test_sparse_policy_format_skipped(self):
        """Sparse policy format is handled."""
        validator = DataValidator()
        data = {
            "values": np.zeros(3),
            "policy_values": np.array([0.3, 0.7, 0.5, 0.5]),
            "policy_indices": np.array([0, 2, 4]),  # Boundaries
        }
        # Should not crash - sparse format skipped
        result = validator.validate_arrays(data)
        assert result is not None

    def test_object_dtype_policies_skipped(self):
        """Object dtype policies skip negative check."""
        validator = DataValidator()
        # Object arrays (variable length) should be skipped
        policies = np.empty(2, dtype=object)
        policies[0] = np.array([0.5, 0.5])
        policies[1] = np.array([0.3, 0.3, 0.4])
        data = {
            "values": np.zeros(2),
            "policy_values": policies,
        }
        result = validator.validate_arrays(data)
        # Should not crash on object dtype
        neg_issues = [
            i for i in result.issues
            if i.issue_type == ValidationIssueType.POLICY_NEGATIVE
        ]
        assert len(neg_issues) == 0

    def test_1d_features(self):
        """Handle 1D feature arrays."""
        validator = DataValidator()
        data = {
            "values": np.zeros(5),
            "features": np.array([np.nan, 1.0, 2.0, 3.0, 4.0]),
        }
        result = validator.validate_arrays(data)
        nan_issues = [
            i for i in result.issues
            if i.issue_type == ValidationIssueType.FEATURE_NAN
        ]
        assert len(nan_issues) >= 1

    def test_large_dataset_performance(self):
        """Validate large dataset efficiently."""
        validator = DataValidator()
        # Create large dataset
        n_samples = 10000
        data = {
            "values": np.random.uniform(-1, 1, n_samples),
            "features": np.random.randn(n_samples, 20, 20),
            "policy_values": np.ones((n_samples, 100)) / 100,
        }
        result = validator.validate_arrays(data)
        assert result.total_samples == n_samples
        assert result.valid is True
