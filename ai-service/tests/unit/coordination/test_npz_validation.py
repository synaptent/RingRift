"""Tests for npz_validation.py - NPZ training data validation.

This module tests the NPZ validation utilities that catch corrupted
or malformed training data files before they cause training issues.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.coordination.npz_validation import (
    EXPECTED_DTYPES,
    MAX_REASONABLE_DIMENSION,
    MAX_REASONABLE_SAMPLES,
    NPZValidationResult,
    POLICY_PREFIXES,
    REQUIRED_ARRAYS,
    _get_expected_cells,
    quick_npz_check,
    validate_npz_for_training,
    validate_npz_structure,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_npz_path(tmp_path: Path) -> Path:
    """Create a temporary path for NPZ files."""
    return tmp_path / "test_data.npz"


@pytest.fixture
def valid_npz_file(tmp_npz_path: Path) -> Path:
    """Create a valid NPZ file with all required arrays."""
    n_samples = 100
    np.savez(
        tmp_npz_path,
        features=np.random.randn(n_samples, 61, 32).astype(np.float32),
        values=np.random.randn(n_samples, 2).astype(np.float32),
        policy_logits=np.random.randn(n_samples, 61).astype(np.float32),
        policy_mask=np.ones((n_samples, 61), dtype=bool),
    )
    return tmp_npz_path


@pytest.fixture
def minimal_valid_npz(tmp_npz_path: Path) -> Path:
    """Create a minimal valid NPZ with only required arrays."""
    n_samples = 50
    np.savez(
        tmp_npz_path,
        features=np.random.randn(n_samples, 64, 24).astype(np.float32),
        values=np.random.randn(n_samples, 2).astype(np.float32),
    )
    return tmp_npz_path


# =============================================================================
# NPZValidationResult Tests
# =============================================================================


class TestNPZValidationResult:
    """Tests for the NPZValidationResult dataclass."""

    def test_default_fields(self) -> None:
        """Test that default fields are initialized correctly."""
        result = NPZValidationResult(valid=True)
        assert result.valid is True
        assert result.sample_count == 0
        assert result.errors == []
        assert result.warnings == []
        assert result.array_shapes == {}
        assert result.array_dtypes == {}
        assert result.file_size == 0

    def test_invalid_result_with_errors(self) -> None:
        """Test invalid result with error messages."""
        result = NPZValidationResult(
            valid=False,
            errors=["Missing features array", "Missing values array"],
        )
        assert result.valid is False
        assert len(result.errors) == 2
        assert "Missing features array" in result.errors

    def test_valid_result_with_data(self) -> None:
        """Test valid result with populated data."""
        result = NPZValidationResult(
            valid=True,
            sample_count=1000,
            array_shapes={"features": (1000, 61, 32), "values": (1000, 2)},
            array_dtypes={"features": "float32", "values": "float32"},
            file_size=1024 * 1024,  # 1MB
        )
        assert result.valid is True
        assert result.sample_count == 1000
        assert result.array_shapes["features"] == (1000, 61, 32)

    def test_summary_valid_result(self) -> None:
        """Test summary method for valid result."""
        result = NPZValidationResult(
            valid=True,
            sample_count=5000,
            array_shapes={"features": (5000, 61, 32), "values": (5000, 2)},
            file_size=50 * 1024 * 1024,  # 50MB
        )
        summary = result.summary()
        assert "5000 samples" in summary
        assert "2 arrays" in summary
        assert "50.0MB" in summary

    def test_summary_invalid_result(self) -> None:
        """Test summary method for invalid result."""
        result = NPZValidationResult(
            valid=False,
            errors=["File not found", "Cannot open as NPZ"],
        )
        summary = result.summary()
        assert "Invalid NPZ" in summary
        assert "File not found" in summary
        assert "Cannot open as NPZ" in summary


# =============================================================================
# validate_npz_structure Tests
# =============================================================================


class TestValidateNpzStructure:
    """Tests for the validate_npz_structure function."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test validation of non-existent file."""
        result = validate_npz_structure(tmp_path / "nonexistent.npz")
        assert result.valid is False
        assert any("not found" in err.lower() for err in result.errors)

    def test_empty_file(self, tmp_npz_path: Path) -> None:
        """Test validation of empty file."""
        tmp_npz_path.touch()
        result = validate_npz_structure(tmp_npz_path)
        assert result.valid is False
        assert any("empty" in err.lower() for err in result.errors)

    def test_invalid_npz_format(self, tmp_npz_path: Path) -> None:
        """Test validation of non-NPZ file."""
        tmp_npz_path.write_text("not a valid npz file")
        result = validate_npz_structure(tmp_npz_path)
        assert result.valid is False
        assert any("cannot open" in err.lower() for err in result.errors)

    def test_valid_npz_file(self, valid_npz_file: Path) -> None:
        """Test validation of valid NPZ file."""
        result = validate_npz_structure(valid_npz_file)
        assert result.valid is True
        assert result.sample_count == 100
        assert "features" in result.array_shapes
        assert "values" in result.array_shapes
        assert result.file_size > 0

    def test_minimal_valid_npz(self, minimal_valid_npz: Path) -> None:
        """Test validation of minimal valid NPZ (no policy arrays)."""
        result = validate_npz_structure(minimal_valid_npz, require_policy=False)
        assert result.valid is True
        assert result.sample_count == 50

    def test_missing_required_arrays(self, tmp_npz_path: Path) -> None:
        """Test validation when required arrays are missing."""
        # Only save features, missing values
        np.savez(tmp_npz_path, features=np.random.randn(10, 64, 24).astype(np.float32))
        result = validate_npz_structure(tmp_npz_path)
        assert result.valid is False
        assert any("values" in err.lower() for err in result.errors)

    def test_missing_features_array(self, tmp_npz_path: Path) -> None:
        """Test validation when features array is missing."""
        np.savez(tmp_npz_path, values=np.random.randn(10, 2).astype(np.float32))
        result = validate_npz_structure(tmp_npz_path)
        assert result.valid is False
        assert any("features" in err.lower() for err in result.errors)

    def test_missing_policy_array_with_require_policy(self, minimal_valid_npz: Path) -> None:
        """Test warning when policy arrays are missing with require_policy=True."""
        result = validate_npz_structure(minimal_valid_npz, require_policy=True)
        assert result.valid is True  # Still valid, just a warning
        assert any("policy" in warn.lower() for warn in result.warnings)

    def test_inconsistent_sample_counts(self, tmp_npz_path: Path) -> None:
        """Test validation when arrays have different sample counts."""
        np.savez(
            tmp_npz_path,
            features=np.random.randn(100, 64, 24).astype(np.float32),
            values=np.random.randn(50, 2).astype(np.float32),  # Different count!
        )
        result = validate_npz_structure(tmp_npz_path)
        assert result.valid is False
        assert any("inconsistent" in err.lower() for err in result.errors)

    def test_exceeds_max_samples(self, tmp_npz_path: Path) -> None:
        """Test validation when sample count exceeds maximum."""
        # We can't actually create a file with 100M samples, so test with low max
        np.savez(
            tmp_npz_path,
            features=np.random.randn(1000, 64, 24).astype(np.float32),
            values=np.random.randn(1000, 2).astype(np.float32),
        )
        result = validate_npz_structure(tmp_npz_path, max_samples=500)
        assert result.valid is False
        assert any("exceeding maximum" in err.lower() for err in result.errors)

    def test_wrong_dtype_warning(self, tmp_npz_path: Path) -> None:
        """Test warning for unexpected data types."""
        np.savez(
            tmp_npz_path,
            features=np.random.randn(10, 64, 24).astype(np.float64),  # Wrong dtype
            values=np.random.randn(10, 2).astype(np.float32),
        )
        result = validate_npz_structure(tmp_npz_path, require_policy=False)
        assert result.valid is True
        assert any("dtype" in warn.lower() for warn in result.warnings)

    def test_file_size_tracked(self, valid_npz_file: Path) -> None:
        """Test that file size is correctly tracked."""
        result = validate_npz_structure(valid_npz_file)
        assert result.file_size > 0
        assert result.file_size == valid_npz_file.stat().st_size

    def test_array_shapes_tracked(self, valid_npz_file: Path) -> None:
        """Test that array shapes are correctly tracked."""
        result = validate_npz_structure(valid_npz_file)
        assert result.array_shapes["features"] == (100, 61, 32)
        assert result.array_shapes["values"] == (100, 2)

    def test_array_dtypes_tracked(self, valid_npz_file: Path) -> None:
        """Test that array dtypes are correctly tracked."""
        result = validate_npz_structure(valid_npz_file)
        assert result.array_dtypes["features"] == "float32"
        assert result.array_dtypes["values"] == "float32"

    def test_npz_with_no_arrays(self, tmp_npz_path: Path) -> None:
        """Test validation of NPZ with no arrays."""
        # Create an empty NPZ by saving empty dict
        np.savez(tmp_npz_path)
        result = validate_npz_structure(tmp_npz_path)
        assert result.valid is False
        assert any("no arrays" in err.lower() for err in result.errors)


# =============================================================================
# validate_npz_for_training Tests
# =============================================================================


class TestValidateNpzForTraining:
    """Tests for the validate_npz_for_training function."""

    def test_invalid_structure_propagated(self, tmp_path: Path) -> None:
        """Test that invalid structure errors are propagated."""
        result = validate_npz_for_training(tmp_path / "nonexistent.npz")
        assert result.valid is False

    def test_valid_hex8_file(self, tmp_npz_path: Path) -> None:
        """Test validation with correct hex8 board type."""
        n_samples = 50
        np.savez(
            tmp_npz_path,
            features=np.random.randn(n_samples, 61, 32).astype(np.float32),
            values=np.random.randn(n_samples, 2).astype(np.float32),
            policy_logits=np.random.randn(n_samples, 61).astype(np.float32),
        )
        result = validate_npz_for_training(tmp_npz_path, board_type="hex8", num_players=2)
        assert result.valid is True

    def test_valid_square8_file(self, tmp_npz_path: Path) -> None:
        """Test validation with correct square8 board type."""
        n_samples = 50
        np.savez(
            tmp_npz_path,
            features=np.random.randn(n_samples, 64, 32).astype(np.float32),
            values=np.random.randn(n_samples, 4).astype(np.float32),
            policy_logits=np.random.randn(n_samples, 64).astype(np.float32),
        )
        result = validate_npz_for_training(tmp_npz_path, board_type="square8", num_players=4)
        assert result.valid is True

    def test_wrong_board_type_warning(self, tmp_npz_path: Path) -> None:
        """Test warning when features don't match board type."""
        n_samples = 50
        np.savez(
            tmp_npz_path,
            features=np.random.randn(n_samples, 100, 32).astype(np.float32),  # Wrong size
            values=np.random.randn(n_samples, 2).astype(np.float32),
            policy_logits=np.random.randn(n_samples, 100).astype(np.float32),
        )
        result = validate_npz_for_training(tmp_npz_path, board_type="hex8")  # 61 cells
        assert result.valid is True
        assert any("may not match" in warn.lower() for warn in result.warnings)

    def test_wrong_num_players_warning(self, tmp_npz_path: Path) -> None:
        """Test warning when values don't match player count."""
        n_samples = 50
        np.savez(
            tmp_npz_path,
            features=np.random.randn(n_samples, 61, 32).astype(np.float32),
            values=np.random.randn(n_samples, 2).astype(np.float32),
            policy_logits=np.random.randn(n_samples, 61).astype(np.float32),
        )
        result = validate_npz_for_training(tmp_npz_path, num_players=4)  # Expects 4
        assert result.valid is True
        assert any("num_players" in warn.lower() for warn in result.warnings)

    def test_no_board_type_specified(self, valid_npz_file: Path) -> None:
        """Test validation without board type (no shape check)."""
        result = validate_npz_for_training(valid_npz_file)
        assert result.valid is True

    def test_unknown_board_type(self, valid_npz_file: Path) -> None:
        """Test validation with unknown board type."""
        result = validate_npz_for_training(valid_npz_file, board_type="unknown_board")
        assert result.valid is True  # Unknown board type skips check


# =============================================================================
# _get_expected_cells Tests
# =============================================================================


class TestGetExpectedCells:
    """Tests for the _get_expected_cells helper function."""

    def test_hex8_cells(self) -> None:
        """Test hex8 returns 61 cells."""
        assert _get_expected_cells("hex8") == 61

    def test_square8_cells(self) -> None:
        """Test square8 returns 64 cells."""
        assert _get_expected_cells("square8") == 64

    def test_square19_cells(self) -> None:
        """Test square19 returns 361 cells."""
        assert _get_expected_cells("square19") == 361

    def test_hexagonal_cells(self) -> None:
        """Test hexagonal returns 469 cells."""
        assert _get_expected_cells("hexagonal") == 469

    def test_unknown_board_type(self) -> None:
        """Test unknown board type returns None."""
        assert _get_expected_cells("unknown") is None

    def test_empty_string(self) -> None:
        """Test empty string returns None."""
        assert _get_expected_cells("") is None


# =============================================================================
# quick_npz_check Tests
# =============================================================================


class TestQuickNpzCheck:
    """Tests for the quick_npz_check function."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test quick check for non-existent file."""
        is_valid, error = quick_npz_check(tmp_path / "nonexistent.npz")
        assert is_valid is False
        assert "not found" in error.lower()

    def test_empty_file(self, tmp_npz_path: Path) -> None:
        """Test quick check for empty file."""
        tmp_npz_path.touch()
        is_valid, error = quick_npz_check(tmp_npz_path)
        assert is_valid is False
        assert "empty" in error.lower()

    def test_invalid_format(self, tmp_npz_path: Path) -> None:
        """Test quick check for invalid format."""
        tmp_npz_path.write_bytes(b"not a npz file content")
        is_valid, error = quick_npz_check(tmp_npz_path)
        assert is_valid is False
        assert error != ""

    def test_valid_npz(self, valid_npz_file: Path) -> None:
        """Test quick check for valid NPZ."""
        is_valid, error = quick_npz_check(valid_npz_file)
        assert is_valid is True
        assert error == ""

    def test_missing_features(self, tmp_npz_path: Path) -> None:
        """Test quick check when features array is missing."""
        np.savez(tmp_npz_path, values=np.random.randn(10, 2).astype(np.float32))
        is_valid, error = quick_npz_check(tmp_npz_path)
        assert is_valid is False
        assert "features" in error.lower()

    def test_missing_values(self, tmp_npz_path: Path) -> None:
        """Test quick check when values array is missing."""
        np.savez(tmp_npz_path, features=np.random.randn(10, 64, 24).astype(np.float32))
        is_valid, error = quick_npz_check(tmp_npz_path)
        assert is_valid is False
        assert "values" in error.lower()

    def test_empty_npz(self, tmp_npz_path: Path) -> None:
        """Test quick check for NPZ with no arrays."""
        np.savez(tmp_npz_path)
        is_valid, error = quick_npz_check(tmp_npz_path)
        assert is_valid is False
        assert "no arrays" in error.lower()


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_max_reasonable_samples(self) -> None:
        """Test MAX_REASONABLE_SAMPLES is set appropriately."""
        assert MAX_REASONABLE_SAMPLES == 100_000_000

    def test_max_reasonable_dimension(self) -> None:
        """Test MAX_REASONABLE_DIMENSION is set appropriately."""
        assert MAX_REASONABLE_DIMENSION == 1_000_000_000

    def test_required_arrays(self) -> None:
        """Test REQUIRED_ARRAYS contains expected arrays."""
        assert "features" in REQUIRED_ARRAYS
        assert "values" in REQUIRED_ARRAYS

    def test_policy_prefixes(self) -> None:
        """Test POLICY_PREFIXES contains expected prefixes."""
        assert "policy" in POLICY_PREFIXES

    def test_expected_dtypes_structure(self) -> None:
        """Test EXPECTED_DTYPES has expected structure."""
        assert "features" in EXPECTED_DTYPES
        assert "values" in EXPECTED_DTYPES
        assert "float32" in EXPECTED_DTYPES["features"]
        assert "float32" in EXPECTED_DTYPES["values"]


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_file(self, tmp_npz_path: Path) -> None:
        """Test validation with very small (1 sample) file."""
        np.savez(
            tmp_npz_path,
            features=np.random.randn(1, 64, 24).astype(np.float32),
            values=np.random.randn(1, 2).astype(np.float32),
        )
        result = validate_npz_structure(tmp_npz_path, require_policy=False)
        assert result.valid is True
        assert result.sample_count == 1

    def test_single_dimension_arrays(self, tmp_npz_path: Path) -> None:
        """Test handling of 1D arrays."""
        np.savez(
            tmp_npz_path,
            features=np.random.randn(100, 64, 24).astype(np.float32),
            values=np.random.randn(100, 2).astype(np.float32),
            weights=np.random.randn(100).astype(np.float32),  # 1D array
        )
        result = validate_npz_structure(tmp_npz_path, require_policy=False)
        assert result.valid is True

    def test_extra_arrays_allowed(self, tmp_npz_path: Path) -> None:
        """Test that extra arrays are allowed."""
        np.savez(
            tmp_npz_path,
            features=np.random.randn(50, 64, 24).astype(np.float32),
            values=np.random.randn(50, 2).astype(np.float32),
            extra_array=np.random.randn(50, 10).astype(np.float32),
            another_extra=np.random.randn(50).astype(np.int32),
        )
        result = validate_npz_structure(tmp_npz_path, require_policy=False)
        assert result.valid is True
        assert "extra_array" in result.array_shapes
        assert "another_extra" in result.array_shapes

    def test_boolean_array(self, tmp_npz_path: Path) -> None:
        """Test validation with boolean policy mask."""
        np.savez(
            tmp_npz_path,
            features=np.random.randn(50, 64, 24).astype(np.float32),
            values=np.random.randn(50, 2).astype(np.float32),
            policy_mask=np.ones((50, 64), dtype=bool),
        )
        result = validate_npz_structure(tmp_npz_path)
        assert result.valid is True
        assert result.array_dtypes["policy_mask"] == "bool"

    def test_int_arrays(self, tmp_npz_path: Path) -> None:
        """Test validation with integer arrays."""
        np.savez(
            tmp_npz_path,
            features=np.random.randn(50, 64, 24).astype(np.float32),
            values=np.random.randn(50, 2).astype(np.float32),
            move_indices=np.random.randint(0, 64, size=(50,), dtype=np.int32),
        )
        result = validate_npz_structure(tmp_npz_path, require_policy=False)
        assert result.valid is True
        assert result.array_dtypes["move_indices"] == "int32"

    def test_float16_precision(self, tmp_npz_path: Path) -> None:
        """Test validation with float16 arrays (memory efficient)."""
        np.savez(
            tmp_npz_path,
            features=np.random.randn(50, 64, 24).astype(np.float16),
            values=np.random.randn(50, 2).astype(np.float16),
        )
        result = validate_npz_structure(tmp_npz_path, require_policy=False)
        assert result.valid is True
        assert result.array_dtypes["features"] == "float16"
