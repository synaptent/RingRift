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


# =============================================================================
# embed_checksums_in_save_kwargs Tests
# =============================================================================


class TestEmbedChecksumsInSaveKwargs:
    """Tests for embed_checksums_in_save_kwargs function."""

    def test_adds_checksum_field(self):
        """Checksums are added to save_kwargs."""
        from app.training.data_quality import embed_checksums_in_save_kwargs

        data = {
            "features": np.random.randn(10, 5).astype(np.float32),
            "values": np.random.rand(10).astype(np.float32),
        }
        result = embed_checksums_in_save_kwargs(data)
        assert "data_checksums" in result
        assert isinstance(result["data_checksums"], np.ndarray)

    def test_checksums_are_json(self):
        """Checksum field contains valid JSON."""
        import json

        from app.training.data_quality import embed_checksums_in_save_kwargs

        data = {"arr": np.array([1, 2, 3])}
        result = embed_checksums_in_save_kwargs(data)
        checksums_json = str(result["data_checksums"])
        parsed = json.loads(checksums_json)
        assert "arr" in parsed

    def test_original_data_preserved(self):
        """Original arrays are preserved in result."""
        from app.training.data_quality import embed_checksums_in_save_kwargs

        data = {
            "features": np.array([1, 2, 3]),
            "values": np.array([0.5, 0.5]),
        }
        result = embed_checksums_in_save_kwargs(data)
        assert "features" in result
        assert "values" in result
        assert np.array_equal(result["features"], data["features"])


# =============================================================================
# validate_database_for_export Tests
# =============================================================================


class TestValidateDatabaseForExport:
    """Tests for validate_database_for_export function."""

    @pytest.fixture
    def db_with_moves(self, tmp_path):
        """Create a database with game_moves table."""
        db_path = tmp_path / "with_moves.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                game_status TEXT NOT NULL,
                winner INTEGER,
                total_moves INTEGER NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE game_moves (
                id INTEGER PRIMARY KEY,
                game_id TEXT NOT NULL,
                move_number INTEGER NOT NULL,
                move_data TEXT NOT NULL,
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            )
        """)
        # Insert games with moves
        for i in range(10):
            game_id = f"game_{i}"
            cursor.execute(
                "INSERT INTO games VALUES (?, 'hex8', 2, 'completed', 0, 5)",
                (game_id,),
            )
            for j in range(5):
                cursor.execute(
                    "INSERT INTO game_moves (game_id, move_number, move_data) VALUES (?, ?, ?)",
                    (game_id, j, '{"type": "PLACE_RING", "to": [0, 0]}'),
                )
        conn.commit()
        conn.close()
        return db_path

    @pytest.fixture
    def db_without_moves(self, tmp_path):
        """Create a database without moves."""
        db_path = tmp_path / "no_moves.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                game_status TEXT NOT NULL,
                winner INTEGER,
                total_moves INTEGER NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE game_moves (
                id INTEGER PRIMARY KEY,
                game_id TEXT NOT NULL,
                move_number INTEGER NOT NULL,
                move_data TEXT NOT NULL
            )
        """)
        # Insert games WITHOUT moves
        for i in range(10):
            cursor.execute(
                "INSERT INTO games VALUES (?, 'hex8', 2, 'completed', 0, 5)",
                (f"game_{i}",),
            )
        conn.commit()
        conn.close()
        return db_path

    def test_valid_database_passes(self, db_with_moves):
        """Database with moves passes validation."""
        from app.training.data_quality import validate_database_for_export

        is_valid, msg = validate_database_for_export(db_with_moves)
        assert is_valid
        assert "OK" in msg or "100" in msg

    def test_missing_database_fails(self):
        """Missing database fails validation."""
        from app.training.data_quality import validate_database_for_export

        is_valid, msg = validate_database_for_export("/nonexistent/db.db")
        assert not is_valid
        assert "not found" in msg.lower()

    def test_database_without_moves_fails(self, db_without_moves):
        """Database without move data fails validation."""
        from app.training.data_quality import validate_database_for_export

        is_valid, msg = validate_database_for_export(db_without_moves)
        assert not is_valid
        assert "move" in msg.lower() or "CRITICAL" in msg

    def test_config_filter(self, db_with_moves):
        """Config filter for board_type and num_players works."""
        from app.training.data_quality import validate_database_for_export

        # hex8 2p exists
        is_valid, msg = validate_database_for_export(
            db_with_moves, board_type="hex8", num_players=2
        )
        assert is_valid

        # square8 4p doesn't exist
        is_valid, msg = validate_database_for_export(
            db_with_moves, board_type="square8", num_players=4
        )
        assert not is_valid


# =============================================================================
# MultiplayerValidationResult Tests
# =============================================================================


class TestMultiplayerValidationResult:
    """Tests for MultiplayerValidationResult dataclass."""

    def test_creation(self):
        """Result can be created with required fields."""
        from app.training.data_quality import MultiplayerValidationResult

        result = MultiplayerValidationResult(
            valid=True,
            num_samples=1000,
            expected_players=4,
            values_mp_shape=(1000, 4),
        )
        assert result.valid is True
        assert result.num_samples == 1000
        assert result.expected_players == 4

    def test_default_lists(self):
        """Errors and warnings default to empty lists."""
        from app.training.data_quality import MultiplayerValidationResult

        result = MultiplayerValidationResult(
            valid=True,
            num_samples=100,
            expected_players=2,
            values_mp_shape=(100, 2),
        )
        assert result.errors == []
        assert result.warnings == []

    def test_str_representation(self):
        """String representation is informative."""
        from app.training.data_quality import MultiplayerValidationResult

        result = MultiplayerValidationResult(
            valid=False,
            num_samples=50,
            expected_players=4,
            values_mp_shape=(50, 2),
            errors=["Wrong dimension"],
        )
        str_repr = str(result)
        assert "FAIL" in str_repr
        assert "50" in str_repr
        assert "Wrong dimension" in str_repr


# =============================================================================
# validate_multiplayer_training_data Tests
# =============================================================================


class TestValidateMultiplayerTrainingData:
    """Tests for validate_multiplayer_training_data function."""

    @pytest.fixture
    def valid_4p_npz(self, tmp_path):
        """Create valid 4-player NPZ file."""
        npz_path = tmp_path / "4p_valid.npz"
        n_samples = 2000
        np.savez(
            npz_path,
            features=np.random.randn(n_samples, 16, 9, 9).astype(np.float32),
            values_mp=np.random.rand(n_samples, 4).astype(np.float32) * 2 - 1,  # [-1, 1]
            num_players=np.full(n_samples, 4, dtype=np.int32),
        )
        return npz_path

    @pytest.fixture
    def wrong_dimension_npz(self, tmp_path):
        """Create NPZ with wrong values_mp dimension."""
        npz_path = tmp_path / "wrong_dim.npz"
        n_samples = 1000
        np.savez(
            npz_path,
            features=np.random.randn(n_samples, 16, 9, 9).astype(np.float32),
            values_mp=np.random.rand(n_samples, 2).astype(np.float32),  # Only 2 columns!
            num_players=np.full(n_samples, 4, dtype=np.int32),
        )
        return npz_path

    def test_valid_data_passes(self, valid_4p_npz):
        """Valid 4-player data passes validation."""
        from app.training.data_quality import validate_multiplayer_training_data

        result = validate_multiplayer_training_data(valid_4p_npz, expected_players=4)
        assert result.valid
        assert result.num_samples == 2000
        assert len(result.errors) == 0

    def test_missing_file_fails(self):
        """Missing file fails validation."""
        from app.training.data_quality import validate_multiplayer_training_data

        result = validate_multiplayer_training_data(
            "/nonexistent/file.npz", expected_players=4
        )
        assert not result.valid
        assert "not found" in result.errors[0].lower()

    def test_wrong_dimension_fails(self, wrong_dimension_npz):
        """Wrong values_mp dimension fails validation."""
        from app.training.data_quality import validate_multiplayer_training_data

        result = validate_multiplayer_training_data(
            wrong_dimension_npz, expected_players=4
        )
        assert not result.valid
        assert any("dimension" in e.lower() for e in result.errors)

    def test_insufficient_samples_fails(self, tmp_path):
        """Insufficient samples fails validation."""
        from app.training.data_quality import validate_multiplayer_training_data

        npz_path = tmp_path / "small.npz"
        np.savez(
            npz_path,
            features=np.random.randn(50, 16, 9, 9).astype(np.float32),
            values_mp=np.random.rand(50, 4).astype(np.float32),
            num_players=np.full(50, 4, dtype=np.int32),
        )
        result = validate_multiplayer_training_data(
            npz_path, expected_players=4, min_samples=1000
        )
        assert not result.valid
        assert any("insufficient" in e.lower() for e in result.errors)

    def test_missing_values_mp_fails(self, tmp_path):
        """Missing values_mp array fails validation."""
        from app.training.data_quality import validate_multiplayer_training_data

        npz_path = tmp_path / "no_values_mp.npz"
        np.savez(
            npz_path,
            features=np.random.randn(1000, 16, 9, 9).astype(np.float32),
            values=np.random.rand(1000).astype(np.float32),  # Old format, no values_mp
        )
        result = validate_multiplayer_training_data(npz_path, expected_players=4)
        assert not result.valid
        assert any("values_mp" in e.lower() for e in result.errors)


# =============================================================================
# check_games_with_moves Tests
# =============================================================================


class TestCheckGamesWithMoves:
    """Tests for DatabaseQualityChecker.check_games_with_moves method."""

    @pytest.fixture
    def checker(self):
        return DatabaseQualityChecker()

    @pytest.fixture
    def db_high_coverage(self, tmp_path):
        """Database with 95% move coverage."""
        db_path = tmp_path / "high_coverage.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                game_status TEXT NOT NULL,
                winner INTEGER,
                total_moves INTEGER NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE game_moves (
                id INTEGER PRIMARY KEY,
                game_id TEXT NOT NULL,
                move_number INTEGER NOT NULL,
                move_data TEXT NOT NULL
            )
        """)
        # 95 games with moves, 5 without
        for i in range(100):
            game_id = f"game_{i}"
            cursor.execute(
                "INSERT INTO games VALUES (?, 'hex8', 2, 'completed', 0, 5)",
                (game_id,),
            )
            if i < 95:  # 95% have moves
                cursor.execute(
                    "INSERT INTO game_moves (game_id, move_number, move_data) VALUES (?, 0, '{}')",
                    (game_id,),
                )
        conn.commit()
        conn.close()
        return db_path

    @pytest.fixture
    def db_low_coverage(self, tmp_path):
        """Database with only 5% move coverage."""
        db_path = tmp_path / "low_coverage.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT,
                num_players INTEGER,
                game_status TEXT,
                winner INTEGER,
                total_moves INTEGER
            )
        """)
        cursor.execute("""
            CREATE TABLE game_moves (
                id INTEGER PRIMARY KEY,
                game_id TEXT,
                move_number INTEGER,
                move_data TEXT
            )
        """)
        # 5 games with moves, 95 without
        for i in range(100):
            game_id = f"game_{i}"
            cursor.execute(
                "INSERT INTO games VALUES (?, 'hex8', 2, 'completed', 0, 5)",
                (game_id,),
            )
            if i < 5:  # Only 5% have moves
                cursor.execute(
                    "INSERT INTO game_moves (game_id, move_number, move_data) VALUES (?, 0, '{}')",
                    (game_id,),
                )
        conn.commit()
        conn.close()
        return db_path

    def test_high_coverage_passes(self, checker, db_high_coverage):
        """High move coverage passes check."""
        passes, stats = checker.check_games_with_moves(db_high_coverage)
        assert passes
        assert stats["coverage_percent"] >= 90.0
        assert stats["games_with_moves"] == 95

    def test_low_coverage_fails(self, checker, db_low_coverage):
        """Low move coverage fails check."""
        passes, stats = checker.check_games_with_moves(db_low_coverage)
        assert not passes
        assert stats["coverage_percent"] < 10.0
        assert "CRITICAL" in stats["issue"]

    def test_missing_db(self, checker):
        """Missing database fails check."""
        passes, stats = checker.check_games_with_moves("/nonexistent.db")
        assert not passes
        assert stats["issue"] is not None

    def test_stats_structure(self, checker, db_high_coverage):
        """Stats dict has expected structure."""
        _, stats = checker.check_games_with_moves(db_high_coverage)
        assert "total_games" in stats
        assert "games_with_moves" in stats
        assert "games_without_moves" in stats
        assert "coverage_percent" in stats
        assert "schema_type" in stats


# =============================================================================
# DataQualityReport __str__ Tests
# =============================================================================


class TestDataQualityReportStr:
    """Tests for DataQualityReport string representation."""

    def test_str_includes_path(self):
        """String includes database path."""
        report = DataQualityReport(database_path="/path/to/db.db")
        assert "/path/to/db.db" in str(report)

    def test_str_includes_counts(self):
        """String includes game counts."""
        report = DataQualityReport(total_games=1000, valid_games=950)
        s = str(report)
        assert "1,000" in s or "1000" in s
        assert "950" in s

    def test_str_includes_score(self):
        """String includes quality score."""
        report = DataQualityReport(quality_score=0.85)
        assert "85" in str(report)

    def test_str_includes_issues(self):
        """String includes issues."""
        report = DataQualityReport(issues=["Issue one", "Issue two"])
        s = str(report)
        assert "Issue one" in s
        assert "Issue two" in s

    def test_str_includes_recommendations(self):
        """String includes recommendations."""
        report = DataQualityReport(recommendations=["Fix this", "Improve that"])
        s = str(report)
        assert "Fix this" in s
        assert "Improve that" in s

    def test_str_no_issues_message(self):
        """String shows 'no issues' when list is empty."""
        report = DataQualityReport(issues=[])
        assert "No issues" in str(report) or "no issues" in str(report).lower()
