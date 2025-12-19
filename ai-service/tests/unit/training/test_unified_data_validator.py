"""Tests for unified_data_validator.py - consolidated data validation."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestUnifiedDataValidatorImports:
    """Test that unified_data_validator provides correct imports."""

    def test_import_unified_api(self):
        """Test importing the unified API."""
        from app.training.unified_data_validator import (
            UnifiedDataValidator,
            UnifiedValidationResult,
            UnifiedValidationIssue,
            ValidationType,
            ValidationSeverity,
            get_validator,
            validate_training_data,
            validate_database,
            validate_any,
        )
        assert UnifiedDataValidator is not None
        assert get_validator is not None
        assert ValidationType is not None

    def test_import_legacy_reexports(self):
        """Test that legacy re-exports are available."""
        from app.training.unified_data_validator import (
            DataValidator,
            DataValidatorConfig,
            validate_npz_file,
        )
        # These may be None if original module not available
        # but they should be importable

    def test_import_territory_reexports(self):
        """Test territory validation re-exports."""
        try:
            from app.training.unified_data_validator import (
                validate_territory_example,
                validate_territory_dataset_file,
            )
        except ImportError:
            pytest.skip("Territory validation not available")

    def test_import_parity_reexports(self):
        """Test parity validation re-exports."""
        try:
            from app.training.unified_data_validator import (
                validate_parity,
                ParityValidationError,
                ParityDivergence,
                ParityMode,
            )
        except ImportError:
            pytest.skip("Parity validation not available")


class TestValidationType:
    """Test ValidationType enum."""

    def test_validation_types_exist(self):
        """Test that expected validation types exist."""
        from app.training.unified_data_validator import ValidationType

        assert hasattr(ValidationType, 'TRAINING_DATA')
        assert hasattr(ValidationType, 'NNUE_DATASET')
        assert hasattr(ValidationType, 'TERRITORY_DATASET')
        assert hasattr(ValidationType, 'DATABASE')
        assert hasattr(ValidationType, 'GAME_PARITY')


class TestValidationSeverity:
    """Test ValidationSeverity enum."""

    def test_severity_levels_exist(self):
        """Test that expected severity levels exist."""
        from app.training.unified_data_validator import ValidationSeverity

        assert hasattr(ValidationSeverity, 'INFO')
        assert hasattr(ValidationSeverity, 'WARNING')
        assert hasattr(ValidationSeverity, 'ERROR')
        assert hasattr(ValidationSeverity, 'CRITICAL')


class TestUnifiedValidationResult:
    """Test UnifiedValidationResult dataclass."""

    def test_create_valid_result(self):
        """Test creating a valid result."""
        from app.training.unified_data_validator import (
            UnifiedValidationResult,
            ValidationType,
        )

        result = UnifiedValidationResult(
            is_valid=True,
            validation_type=ValidationType.TRAINING_DATA,
            source_path="/path/to/file.npz",
            total_items=100,
        )

        assert result.is_valid is True
        assert result.validation_type == ValidationType.TRAINING_DATA
        assert result.total_items == 100

    def test_add_issue(self):
        """Test adding an issue to result."""
        from app.training.unified_data_validator import (
            UnifiedValidationResult,
            ValidationType,
            ValidationSeverity,
        )

        result = UnifiedValidationResult(
            is_valid=True,
            validation_type=ValidationType.TRAINING_DATA,
            source_path="/path/to/file.npz",
        )

        result.add_issue(
            issue_type="test_error",
            message="Test error message",
            severity=ValidationSeverity.ERROR,
        )

        assert len(result.issues) == 1
        assert result.is_valid is False  # Should become invalid on ERROR

    def test_summary(self):
        """Test getting summary string."""
        from app.training.unified_data_validator import (
            UnifiedValidationResult,
            ValidationType,
        )

        result = UnifiedValidationResult(
            is_valid=True,
            validation_type=ValidationType.TRAINING_DATA,
            source_path="/path/to/file.npz",
            total_items=100,
        )

        summary = result.summary()
        assert isinstance(summary, str)
        assert "Valid" in summary or "training_data" in summary

    def test_to_dict(self):
        """Test converting to dict."""
        from app.training.unified_data_validator import (
            UnifiedValidationResult,
            ValidationType,
        )

        result = UnifiedValidationResult(
            is_valid=True,
            validation_type=ValidationType.DATABASE,
            source_path="/path/to/db.sqlite",
            total_items=50,
        )

        data = result.to_dict()
        assert isinstance(data, dict)
        assert data["is_valid"] is True
        assert data["validation_type"] == "database"


class TestUnifiedDataValidator:
    """Test UnifiedDataValidator class."""

    def test_singleton_pattern(self):
        """Test that get_validator returns singleton."""
        from app.training.unified_data_validator import get_validator

        v1 = get_validator()
        v2 = get_validator()
        assert v1 is v2

    def test_validator_has_required_methods(self):
        """Test that validator has required interface methods."""
        from app.training.unified_data_validator import get_validator

        validator = get_validator()

        # Check key methods exist
        assert hasattr(validator, 'validate_training_file')
        assert hasattr(validator, 'validate_database')
        assert hasattr(validator, 'validate_nnue_dataset')
        assert hasattr(validator, 'validate_territory_dataset')
        assert hasattr(validator, 'validate_game_parity')
        assert hasattr(validator, 'validate')
        assert hasattr(validator, 'get_stats')

    def test_get_stats(self):
        """Test getting validation statistics."""
        from app.training.unified_data_validator import get_validator

        validator = get_validator()
        stats = validator.get_stats()

        assert isinstance(stats, dict)
        assert "validations_run" in stats
        assert "validations_passed" in stats
        assert "validations_failed" in stats


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_validate_training_data_exists(self):
        """Test that validate_training_data function exists."""
        from app.training.unified_data_validator import validate_training_data
        assert callable(validate_training_data)

    def test_validate_database_exists(self):
        """Test that validate_database function exists."""
        from app.training.unified_data_validator import validate_database
        assert callable(validate_database)

    def test_validate_any_exists(self):
        """Test that validate_any function exists."""
        from app.training.unified_data_validator import validate_any
        assert callable(validate_any)


class TestValidateGameParity:
    """Test validate_game_parity method."""

    def test_method_exists(self):
        """Test that validate_game_parity method exists."""
        from app.training.unified_data_validator import get_validator

        validator = get_validator()
        assert hasattr(validator, 'validate_game_parity')
        assert callable(validator.validate_game_parity)
