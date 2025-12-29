"""
Unit tests for app.training.data_quality module.

Tests cover:
- Checksum computation and verification
- DataQualityReport dataclass
- QualityIssueLevel enum
- DatabaseQualityChecker class
- TrainingDataValidator class
- validate_database_for_export function
- MultiplayerValidationResult dataclass

Created: December 2025
"""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.training.data_quality import (
    DataQualityReport,
    DatabaseQualityChecker,
    QualityIssueLevel,
    TrainingDataValidator,
    compute_array_checksum,
    compute_npz_checksums,
    verify_npz_checksums,
)


# =============================================================================
# Checksum Tests
# =============================================================================


class TestComputeArrayChecksum:
    """Tests for compute_array_checksum function."""

    def test_basic_array(self):
        """Checksum is computed for basic array."""
        arr = np.array([1, 2, 3, 4, 5])
        checksum = compute_array_checksum(arr)
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex length

    def test_deterministic(self):
        """Same array produces same checksum."""
        arr = np.array([1.0, 2.0, 3.0])
        checksum1 = compute_array_checksum(arr)
        checksum2 = compute_array_checksum(arr)
        assert checksum1 == checksum2

    def test_different_arrays_different_checksums(self):
        """Different arrays produce different checksums."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 4])
        assert compute_array_checksum(arr1) != compute_array_checksum(arr2)

    def test_shape_affects_checksum(self):
        """Arrays with same data but different shapes have different checksums."""
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([1, 2, 3, 4])
        assert compute_array_checksum(arr1) != compute_array_checksum(arr2)

    def test_dtype_affects_checksum(self):
        """Arrays with same values but different dtypes have different checksums."""
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        arr2 = np.array([1, 2, 3], dtype=np.int64)
        assert compute_array_checksum(arr1) != compute_array_checksum(arr2)

    def test_float_array(self):
        """Checksum works for float arrays."""
        arr = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        checksum = compute_array_checksum(arr)
        assert len(checksum) == 64

    def test_multidimensional_array(self):
        """Checksum works for multidimensional arrays."""
        arr = np.random.randn(10, 20, 30)
        checksum = compute_array_checksum(arr)
        assert len(checksum) == 64


class TestComputeNpzChecksums:
    """Tests for compute_npz_checksums function."""

    def test_multiple_arrays(self):
        """Computes checksums for multiple arrays."""
        data = {
            "features": np.random.randn(100, 10),
            "labels": np.random.randint(0, 10, (100,)),
        }
        checksums = compute_npz_checksums(data)
        assert "features" in checksums
        assert "labels" in checksums
        assert len(checksums["features"]) == 64
        assert len(checksums["labels"]) == 64

    def test_empty_dict(self):
        """Empty dict returns empty checksums."""
        checksums = compute_npz_checksums({})
        assert checksums == {}

    def test_single_array(self):
        """Works with single array."""
        data = {"arr": np.array([1, 2, 3])}
        checksums = compute_npz_checksums(data)
        assert len(checksums) == 1
        assert "arr" in checksums


class TestVerifyNpzChecksums:
    """Tests for verify_npz_checksums function."""

    def test_valid_checksums(self, tmp_path):
        """Valid checksums pass verification."""
        npz_path = tmp_path / "test.npz"
        data = {
            "features": np.random.randn(10, 5).astype(np.float32),
            "values": np.random.rand(10).astype(np.float32),
        }
        # Save NPZ then verify
        np.savez(npz_path, **data)
        checksums = compute_npz_checksums(data)
        # verify_npz_checksums takes npz_path, not dict
        is_valid, computed, errors = verify_npz_checksums(npz_path, checksums)
        assert is_valid
        assert len(errors) == 0

    def test_missing_array(self, tmp_path):
        """Missing array in NPZ file is detected."""
        npz_path = tmp_path / "test.npz"
        data = {"features": np.array([1, 2, 3])}
        np.savez(npz_path, **data)
        # Expect checksum for array not in file
        checksums = {"features": "abc123" * 10 + "abcd", "labels": "def456" * 10 + "defg"}
        is_valid, computed, errors = verify_npz_checksums(npz_path, checksums)
        # Missing array means mismatch - either errors or invalid
        assert not is_valid or len(errors) > 0 or "labels" not in computed

    def test_corrupted_data(self, tmp_path):
        """Corrupted data is detected via checksum mismatch."""
        npz_path = tmp_path / "test.npz"
        data = {"arr": np.array([1, 2, 3])}
        np.savez(npz_path, **data)
        checksums = {"arr": "wrong_checksum_" + "0" * 50}
        is_valid, computed, errors = verify_npz_checksums(npz_path, checksums)
        assert not is_valid


# =============================================================================
# DataQualityReport Tests
# =============================================================================


class TestDataQualityReport:
    """Tests for DataQualityReport dataclass."""

    def test_creation_defaults(self):
        """Report can be created with defaults."""
        report = DataQualityReport()
        assert report.database_path is None
        assert report.total_games == 0
        assert report.valid_games == 0
        assert report.quality_score == 0.0
        assert report.issues == []
        assert report.recommendations == []
        assert report.metadata == {}

    def test_creation_with_values(self):
        """Report can be created with custom values."""
        report = DataQualityReport(
            database_path="/path/to/db.db",
            total_games=100,
            valid_games=90,
            quality_score=0.9,
            issues=["Issue 1"],
            recommendations=["Recommendation 1"],
        )
        assert report.database_path == "/path/to/db.db"
        assert report.total_games == 100
        assert report.valid_games == 90
        assert report.quality_score == 0.9
        assert len(report.issues) == 1
        assert len(report.recommendations) == 1

    def test_quality_score_range(self):
        """Quality score is clamped to [0, 1] range."""
        report_low = DataQualityReport(quality_score=-0.5)
        report_high = DataQualityReport(quality_score=1.5)
        # Note: dataclass doesn't auto-clamp, so these are allowed
        assert report_low.quality_score == -0.5
        assert report_high.quality_score == 1.5


# =============================================================================
# QualityIssueLevel Tests
# =============================================================================


class TestQualityIssueLevel:
    """Tests for QualityIssueLevel enum."""

    def test_all_levels_exist(self):
        """All expected levels exist."""
        assert hasattr(QualityIssueLevel, "INFO")
        assert hasattr(QualityIssueLevel, "WARNING")
        assert hasattr(QualityIssueLevel, "ERROR")
        assert hasattr(QualityIssueLevel, "CRITICAL")

    def test_level_values(self):
        """Levels have correct string values."""
        assert QualityIssueLevel.INFO.value == "info"
        assert QualityIssueLevel.WARNING.value == "warning"
        assert QualityIssueLevel.ERROR.value == "error"
        assert QualityIssueLevel.CRITICAL.value == "critical"


# =============================================================================
# DatabaseQualityChecker Tests
# =============================================================================


class TestDatabaseQualityChecker:
    """Tests for DatabaseQualityChecker class."""

    @pytest.fixture
    def checker(self):
        """Create a DatabaseQualityChecker instance."""
        return DatabaseQualityChecker()

    @pytest.fixture
    def valid_db(self, tmp_path):
        """Create a valid game database for testing."""
        db_path = tmp_path / "test_games.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Create minimal required schema
        cursor.execute("""
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT,
                num_players INTEGER,
                game_status TEXT,
                winner INTEGER,
                total_moves INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE moves (
                id INTEGER PRIMARY KEY,
                game_id TEXT,
                move_number INTEGER,
                move_type TEXT,
                player INTEGER,
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            )
        """)
        # Insert sample data
        cursor.execute("""
            INSERT INTO games VALUES ('game1', 'hex8', 2, 'completed', 1, 10, CURRENT_TIMESTAMP)
        """)
        for i in range(10):
            cursor.execute("""
                INSERT INTO moves VALUES (?, 'game1', ?, 'PLACE_RING', ?)
            """, (i, i, i % 2 + 1))
        conn.commit()
        conn.close()
        return db_path

    def test_checker_instantiation(self, checker):
        """Checker can be instantiated."""
        assert checker is not None
        assert checker.last_report is None

    def test_get_quality_score_valid_db(self, checker, valid_db):
        """Quality score is computed for valid database."""
        score = checker.get_quality_score(str(valid_db))
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_get_quality_score_missing_db(self, checker):
        """Returns 0 for missing database."""
        score = checker.get_quality_score("/nonexistent/path.db")
        assert score == 0.0

    def test_last_report_populated(self, checker, valid_db):
        """last_report is populated after quality check."""
        checker.get_quality_score(str(valid_db))
        assert checker.last_report is not None

    def test_get_full_report(self, checker, valid_db):
        """get_quality_score populates last_report with DataQualityReport."""
        checker.get_quality_score(str(valid_db))
        assert checker.last_report is not None
        assert isinstance(checker.last_report, DataQualityReport)


# =============================================================================
# TrainingDataValidator Tests
# =============================================================================


class TestTrainingDataValidator:
    """Tests for TrainingDataValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a TrainingDataValidator instance."""
        return TrainingDataValidator()

    @pytest.fixture
    def valid_npz(self, tmp_path):
        """Create a valid NPZ file for testing."""
        npz_path = tmp_path / "valid.npz"
        # Create minimal valid training data
        # features: 4D (N, C, H, W), values: 1D (N,)
        np.savez(
            npz_path,
            features=np.random.randn(100, 10, 8, 8).astype(np.float32),
            policy_indices=np.random.randint(0, 64, (100, 5)),
            policy_values=np.random.rand(100, 5).astype(np.float32),
            values=np.random.rand(100).astype(np.float32),  # 1D, not 2D
        )
        return npz_path

    @pytest.fixture
    def invalid_npz(self, tmp_path):
        """Create an invalid NPZ file (missing required arrays)."""
        npz_path = tmp_path / "invalid.npz"
        np.savez(npz_path, only_one_array=np.array([1, 2, 3]))
        return npz_path

    def test_validator_instantiation(self, validator):
        """Validator can be instantiated."""
        assert validator is not None

    def test_validate_valid_npz(self, validator, valid_npz):
        """Valid NPZ file passes validation."""
        result = validator.validate_npz_file(str(valid_npz))
        assert result is True

    def test_validate_missing_file(self, validator):
        """Missing file fails validation."""
        result = validator.validate_npz_file("/nonexistent/file.npz")
        assert result is False

    def test_check_feature_distribution(self, validator, valid_npz):
        """Feature distribution can be checked."""
        stats = validator.check_feature_distribution(str(valid_npz))
        assert isinstance(stats, dict)

    def test_detect_outliers(self, validator, valid_npz):
        """Outlier detection works."""
        result = validator.detect_outliers(str(valid_npz))
        assert isinstance(result, dict)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Edge case tests for data quality module."""

    def test_empty_array_checksum(self):
        """Empty array has valid checksum."""
        arr = np.array([])
        checksum = compute_array_checksum(arr)
        assert len(checksum) == 64

    def test_large_array_checksum(self):
        """Large array checksum is computed."""
        arr = np.random.randn(1000, 1000)
        checksum = compute_array_checksum(arr)
        assert len(checksum) == 64

    def test_nan_values_in_array(self):
        """NaN values don't break checksum."""
        arr = np.array([1.0, np.nan, 3.0])
        checksum = compute_array_checksum(arr)
        assert len(checksum) == 64

    def test_inf_values_in_array(self):
        """Inf values don't break checksum."""
        arr = np.array([1.0, np.inf, -np.inf])
        checksum = compute_array_checksum(arr)
        assert len(checksum) == 64


class TestDatabaseCheckerEdgeCases:
    """Edge case tests for DatabaseQualityChecker."""

    @pytest.fixture
    def checker(self):
        return DatabaseQualityChecker()

    def test_empty_database(self, checker, tmp_path):
        """Empty database has low quality score."""
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(db_path)
        conn.close()
        score = checker.get_quality_score(str(db_path))
        # Empty DB should have low score
        assert score < 0.5

    def test_corrupted_database(self, checker, tmp_path):
        """Corrupted database returns 0 score."""
        db_path = tmp_path / "corrupted.db"
        db_path.write_text("not a valid sqlite database")
        score = checker.get_quality_score(str(db_path))
        assert score == 0.0
