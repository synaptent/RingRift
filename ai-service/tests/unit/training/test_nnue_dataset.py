"""Unit tests for app/training/nnue_dataset.py

Tests cover:
- NNUESample, DataValidationResult, NNUEDatasetConfig dataclasses
- Sample validation functions
- Database integrity validation
- PrioritizedExperienceSampler behavior
- Phase-balanced weighting computations
- Feature extraction helpers
"""

import logging
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from app.models import BoardType
from app.training.nnue_dataset import (
    DataValidationResult,
    NNUEDatasetConfig,
    NNUESample,
    NNUESQLiteDataset,
    NNUEStreamingDataset,
    PrioritizedExperienceSampler,
    _ensure_training_columns,
    count_available_samples,
    validate_database_integrity,
    validate_nnue_dataset,
    validate_nnue_sample,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_features():
    """Create sample features for testing."""
    return np.random.randn(768).astype(np.float32)


@pytest.fixture
def valid_sample(sample_features):
    """Create a valid NNUE sample."""
    return NNUESample(
        features=sample_features,
        value=0.5,
        player_number=1,
        game_id="test-game-001",
        move_number=10,
    )


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create games table with proper schema
    cursor.execute("""
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY,
            board_type TEXT,
            num_players INTEGER,
            game_status TEXT,
            winner INTEGER,
            total_moves INTEGER,
            excluded_from_training INTEGER DEFAULT 0,
            schema_version INTEGER DEFAULT 14
        )
    """)

    # Create game_state_snapshots table
    cursor.execute("""
        CREATE TABLE game_state_snapshots (
            game_id TEXT,
            move_number INTEGER,
            state_json TEXT,
            compressed INTEGER DEFAULT 0,
            PRIMARY KEY (game_id, move_number)
        )
    """)

    # Insert a completed game
    cursor.execute("""
        INSERT INTO games (game_id, board_type, num_players, game_status, winner, total_moves)
        VALUES ('test-game-1', 'square8', 2, 'completed', 1, 50)
    """)

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


# =============================================================================
# NNUESample Tests
# =============================================================================


class TestNNUESample:
    """Tests for NNUESample dataclass."""

    def test_create_sample(self, sample_features):
        """Test creating a sample with all fields."""
        sample = NNUESample(
            features=sample_features,
            value=1.0,
            player_number=2,
            game_id="game-123",
            move_number=25,
        )

        assert np.array_equal(sample.features, sample_features)
        assert sample.value == 1.0
        assert sample.player_number == 2
        assert sample.game_id == "game-123"
        assert sample.move_number == 25

    def test_sample_win_value(self, sample_features):
        """Test sample with win value."""
        sample = NNUESample(
            features=sample_features,
            value=1.0,
            player_number=1,
            game_id="win-game",
            move_number=50,
        )
        assert sample.value == 1.0

    def test_sample_loss_value(self, sample_features):
        """Test sample with loss value."""
        sample = NNUESample(
            features=sample_features,
            value=-1.0,
            player_number=2,
            game_id="loss-game",
            move_number=45,
        )
        assert sample.value == -1.0

    def test_sample_draw_value(self, sample_features):
        """Test sample with draw value."""
        sample = NNUESample(
            features=sample_features,
            value=0.0,
            player_number=1,
            game_id="draw-game",
            move_number=100,
        )
        assert sample.value == 0.0


# =============================================================================
# DataValidationResult Tests
# =============================================================================


class TestDataValidationResult:
    """Tests for DataValidationResult dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        result = DataValidationResult(is_valid=True)

        assert result.is_valid is True
        assert result.total_samples == 0
        assert result.valid_samples == 0
        assert result.invalid_samples == 0
        assert result.nan_count == 0
        assert result.inf_count == 0
        assert result.value_out_of_range == 0
        assert result.zero_feature_count == 0
        assert result.class_balance == {}
        assert result.feature_stats == {}
        assert result.errors == []

    def test_mutable_lists_initialized_per_instance(self):
        """Test that mutable defaults are per-instance."""
        result1 = DataValidationResult(is_valid=True)
        result2 = DataValidationResult(is_valid=False)

        result1.errors.append("Error 1")
        result1.class_balance["wins"] = 10

        assert result1.errors == ["Error 1"]
        assert result2.errors == []
        assert result1.class_balance == {"wins": 10}
        assert result2.class_balance == {}

    def test_post_init_initializes_none_values(self):
        """Test __post_init__ handles None values."""
        result = DataValidationResult(
            is_valid=True,
            class_balance=None,
            feature_stats=None,
            errors=None,
        )

        assert result.class_balance == {}
        assert result.feature_stats == {}
        assert result.errors == []


# =============================================================================
# NNUEDatasetConfig Tests
# =============================================================================


class TestNNUEDatasetConfig:
    """Tests for NNUEDatasetConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = NNUEDatasetConfig()

        assert config.board_type == BoardType.SQUARE8
        assert config.num_players == 2
        assert config.sample_every_n_moves == 1
        assert config.min_game_length == 10
        assert config.include_draws is True
        assert config.late_game_weight == 1.0
        assert config.balance_outcomes is False
        assert config.dual_perspective is True
        assert config.require_canonical_schema is True
        assert config.min_schema_version == 9  # CANONICAL_SCHEMA_VERSION

    def test_custom_values(self):
        """Test custom configuration values."""
        config = NNUEDatasetConfig(
            board_type=BoardType.HEXAGONAL,
            num_players=4,
            sample_every_n_moves=3,
            min_game_length=20,
            include_draws=False,
            dual_perspective=False,
            require_canonical_schema=False,
        )

        assert config.board_type == BoardType.HEXAGONAL
        assert config.num_players == 4
        assert config.sample_every_n_moves == 3
        assert config.min_game_length == 20
        assert config.include_draws is False
        assert config.dual_perspective is False
        assert config.require_canonical_schema is False


# =============================================================================
# Sample Validation Tests
# =============================================================================


class TestValidateNNUESample:
    """Tests for validate_nnue_sample function."""

    def test_valid_sample(self, valid_sample):
        """Test validation of a valid sample."""
        is_valid, error = validate_nnue_sample(valid_sample, feature_dim=768)

        assert is_valid is True
        assert error is None

    def test_wrong_feature_dim(self, sample_features):
        """Test validation fails for wrong feature dimension."""
        sample = NNUESample(
            features=sample_features,
            value=0.5,
            player_number=1,
            game_id="test",
            move_number=10,
        )

        is_valid, error = validate_nnue_sample(sample, feature_dim=1024)

        assert is_valid is False
        assert "Feature dim mismatch" in error

    def test_nan_features(self):
        """Test validation fails for NaN features."""
        features = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32)
        sample = NNUESample(
            features=features,
            value=0.5,
            player_number=1,
            game_id="test",
            move_number=10,
        )

        is_valid, error = validate_nnue_sample(sample, feature_dim=4)

        assert is_valid is False
        assert "NaN" in error

    def test_inf_features(self):
        """Test validation fails for Inf features."""
        features = np.array([1.0, np.inf, 3.0, 4.0], dtype=np.float32)
        sample = NNUESample(
            features=features,
            value=0.5,
            player_number=1,
            game_id="test",
            move_number=10,
        )

        is_valid, error = validate_nnue_sample(sample, feature_dim=4)

        assert is_valid is False
        assert "Inf" in error

    def test_value_out_of_range_high(self):
        """Test validation fails for value > 1."""
        features = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        sample = NNUESample(
            features=features,
            value=1.5,
            player_number=1,
            game_id="test",
            move_number=10,
        )

        is_valid, error = validate_nnue_sample(sample, feature_dim=4)

        assert is_valid is False
        assert "out of range" in error

    def test_value_out_of_range_low(self):
        """Test validation fails for value < -1."""
        features = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        sample = NNUESample(
            features=features,
            value=-1.5,
            player_number=1,
            game_id="test",
            move_number=10,
        )

        is_valid, error = validate_nnue_sample(sample, feature_dim=4)

        assert is_valid is False
        assert "out of range" in error

    def test_invalid_player_number_zero(self):
        """Test validation fails for player number 0."""
        features = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        sample = NNUESample(
            features=features,
            value=0.5,
            player_number=0,
            game_id="test",
            move_number=10,
        )

        is_valid, error = validate_nnue_sample(sample, feature_dim=4)

        assert is_valid is False
        assert "Invalid player number" in error

    def test_invalid_player_number_high(self):
        """Test validation fails for player number > 8."""
        features = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        sample = NNUESample(
            features=features,
            value=0.5,
            player_number=9,
            game_id="test",
            move_number=10,
        )

        is_valid, error = validate_nnue_sample(sample, feature_dim=4)

        assert is_valid is False
        assert "Invalid player number" in error

    def test_all_zero_features(self):
        """Test validation fails for all-zero features."""
        features = np.zeros(768, dtype=np.float32)
        sample = NNUESample(
            features=features,
            value=0.5,
            player_number=1,
            game_id="test",
            move_number=10,
        )

        is_valid, error = validate_nnue_sample(sample, feature_dim=768)

        assert is_valid is False
        assert "zero" in error.lower()


# =============================================================================
# Dataset Validation Tests
# =============================================================================


class TestValidateNNUEDataset:
    """Tests for validate_nnue_dataset function."""

    def test_empty_dataset(self):
        """Test validation of empty dataset."""
        result = validate_nnue_dataset([], feature_dim=768)

        assert result.is_valid is True
        assert result.total_samples == 0
        assert result.valid_samples == 0

    def test_valid_dataset(self):
        """Test validation of valid dataset."""
        samples = []
        for i in range(10):
            features = np.random.randn(768).astype(np.float32)
            value = 1.0 if i < 5 else -1.0
            samples.append(NNUESample(
                features=features,
                value=value,
                player_number=1,
                game_id=f"game-{i}",
                move_number=i * 10,
            ))

        result = validate_nnue_dataset(samples, feature_dim=768, log_errors=False)

        assert result.is_valid is True
        assert result.total_samples == 10
        assert result.valid_samples == 10
        assert result.invalid_samples == 0
        assert result.class_balance["wins"] == 5
        assert result.class_balance["losses"] == 5

    def test_dataset_with_invalid_samples(self):
        """Test validation with some invalid samples."""
        samples = []

        # Add valid samples
        for i in range(8):
            features = np.random.randn(768).astype(np.float32)
            samples.append(NNUESample(
                features=features,
                value=0.5,
                player_number=1,
                game_id=f"game-{i}",
                move_number=i,
            ))

        # Add invalid sample (NaN)
        nan_features = np.array([np.nan] * 768, dtype=np.float32)
        samples.append(NNUESample(
            features=nan_features,
            value=0.5,
            player_number=1,
            game_id="bad-game",
            move_number=100,
        ))

        result = validate_nnue_dataset(samples, feature_dim=768, log_errors=False)

        assert result.total_samples == 9
        assert result.valid_samples == 8
        assert result.invalid_samples == 1
        assert result.nan_count == 1

    def test_dataset_too_many_invalid_samples(self):
        """Test dataset marked invalid when > 10% errors."""
        samples = []

        # 5 valid
        for i in range(5):
            features = np.random.randn(768).astype(np.float32)
            samples.append(NNUESample(
                features=features,
                value=0.5,
                player_number=1,
                game_id=f"game-{i}",
                move_number=i,
            ))

        # 5 invalid (> 10% of total)
        for i in range(5):
            nan_features = np.array([np.nan] * 768, dtype=np.float32)
            samples.append(NNUESample(
                features=nan_features,
                value=0.5,
                player_number=1,
                game_id=f"bad-game-{i}",
                move_number=100 + i,
            ))

        result = validate_nnue_dataset(samples, feature_dim=768, log_errors=False)

        assert result.is_valid is False
        assert "Too many invalid samples" in result.errors[0]

    def test_feature_stats_computed(self):
        """Test feature statistics are computed."""
        samples = []
        for i in range(5):
            features = np.ones(768, dtype=np.float32) * (i + 1)
            samples.append(NNUESample(
                features=features,
                value=0.5,
                player_number=1,
                game_id=f"game-{i}",
                move_number=i,
            ))

        result = validate_nnue_dataset(samples, feature_dim=768, log_errors=False)

        assert "mean" in result.feature_stats
        assert "std" in result.feature_stats
        assert "min" in result.feature_stats
        assert "max" in result.feature_stats
        assert "sparsity" in result.feature_stats


# =============================================================================
# Database Integrity Validation Tests
# =============================================================================


class TestValidateDatabaseIntegrity:
    """Tests for validate_database_integrity function."""

    def test_nonexistent_database(self):
        """Test validation of nonexistent database."""
        is_valid, stats = validate_database_integrity("/nonexistent/path.db")

        assert is_valid is False
        assert stats["exists"] is False
        assert "does not exist" in stats["errors"][0]

    def test_valid_database(self, temp_db):
        """Test validation of valid database."""
        is_valid, stats = validate_database_integrity(temp_db)

        assert is_valid is True
        assert stats["exists"] is True
        assert stats["integrity_ok"] is True
        assert stats["total_games"] == 1
        assert stats["completed_games"] == 1

    def test_empty_database(self):
        """Test validation of database with no games."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                game_status TEXT
            )
        """)
        conn.commit()
        conn.close()

        is_valid, stats = validate_database_integrity(db_path)

        assert is_valid is False
        assert stats["total_games"] == 0
        assert "No completed games" in stats["errors"][0]

        Path(db_path).unlink(missing_ok=True)


# =============================================================================
# PrioritizedExperienceSampler Tests
# =============================================================================


class TestPrioritizedExperienceSampler:
    """Tests for PrioritizedExperienceSampler."""

    def test_initialization(self):
        """Test sampler initialization."""
        sampler = PrioritizedExperienceSampler(
            dataset_size=1000,
            alpha=0.6,
            beta=0.4,
        )

        assert len(sampler) == 1000
        assert sampler.alpha == 0.6
        assert sampler.beta == 0.4
        assert sampler.priorities.shape == (1000,)
        assert np.all(sampler.priorities == 1.0)

    def test_iteration_returns_indices(self):
        """Test that iteration returns valid indices."""
        sampler = PrioritizedExperienceSampler(dataset_size=100)

        indices = list(sampler)

        assert len(indices) == 100
        assert all(0 <= idx < 100 for idx in indices)

    def test_update_priorities(self):
        """Test priority update based on errors."""
        sampler = PrioritizedExperienceSampler(dataset_size=100)

        # Update priorities for some indices
        indices = [0, 1, 2]
        errors = np.array([0.1, 0.5, 0.9])
        sampler.update_priorities(indices, errors)

        assert sampler.priorities[0] == pytest.approx(0.1 + sampler.epsilon)
        assert sampler.priorities[1] == pytest.approx(0.5 + sampler.epsilon)
        assert sampler.priorities[2] == pytest.approx(0.9 + sampler.epsilon)
        assert sampler.seen[0] == True  # noqa: E712 - numpy bool comparison
        assert sampler.seen[3] == False  # noqa: E712 - numpy bool comparison

    def test_importance_weights(self):
        """Test importance weight computation."""
        sampler = PrioritizedExperienceSampler(
            dataset_size=100,
            alpha=0.6,
            beta=0.4,
        )

        # Update some priorities to make them non-uniform
        sampler.update_priorities([0, 1], np.array([0.1, 0.9]))

        weights = sampler.get_importance_weights([0, 1, 2])

        assert len(weights) == 3
        assert isinstance(weights, torch.Tensor)
        # Max weight should be normalized to 1
        assert weights.max().item() == pytest.approx(1.0)

    def test_beta_annealing(self):
        """Test beta annealing schedule."""
        sampler = PrioritizedExperienceSampler(
            dataset_size=100,
            beta=0.4,
            beta_schedule=True,
        )

        sampler.set_epoch(0, total_epochs=100)
        beta_0 = sampler.get_stats()["current_beta"]

        sampler.set_epoch(50, total_epochs=100)
        beta_50 = sampler.get_stats()["current_beta"]

        sampler.set_epoch(100, total_epochs=100)
        beta_100 = sampler.get_stats()["current_beta"]

        # Beta should increase from 0.4 toward 1.0
        assert beta_0 == pytest.approx(0.4)
        assert beta_50 > beta_0
        assert beta_100 == pytest.approx(1.0)

    def test_get_stats(self):
        """Test statistics retrieval."""
        sampler = PrioritizedExperienceSampler(dataset_size=100)

        stats = sampler.get_stats()

        assert "mean_priority" in stats
        assert "max_priority" in stats
        assert "min_priority" in stats
        assert "std_priority" in stats
        assert "seen_ratio" in stats
        assert "current_beta" in stats

        # Initial stats
        assert stats["seen_ratio"] == 0.0
        assert stats["mean_priority"] == 1.0


# =============================================================================
# Schema Migration Tests
# =============================================================================


class TestEnsureTrainingColumns:
    """Tests for _ensure_training_columns function."""

    def test_adds_missing_column(self):
        """Test that missing column is added."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create table without excluded_from_training column
        cursor.execute("""
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                game_status TEXT
            )
        """)
        conn.commit()

        # Run migration
        _ensure_training_columns(conn)

        # Check column was added
        cursor.execute("PRAGMA table_info(games)")
        columns = {row[1] for row in cursor.fetchall()}

        assert "excluded_from_training" in columns

        conn.close()
        Path(db_path).unlink(missing_ok=True)

    def test_no_op_if_column_exists(self, temp_db):
        """Test no-op if column already exists."""
        conn = sqlite3.connect(temp_db)

        # Should not raise
        _ensure_training_columns(conn)
        _ensure_training_columns(conn)  # Second call should also work

        conn.close()


# =============================================================================
# NNUESQLiteDataset Tests
# =============================================================================


class TestNNUESQLiteDataset:
    """Tests for NNUESQLiteDataset class."""

    def test_init_empty_db_list(self):
        """Test initialization with empty database list."""
        dataset = NNUESQLiteDataset(
            db_paths=[],
            config=NNUEDatasetConfig(),
        )

        assert len(dataset) == 0
        assert dataset.samples == []

    def test_init_nonexistent_db(self, caplog):
        """Test initialization with nonexistent database."""
        with caplog.at_level(logging.WARNING):
            dataset = NNUESQLiteDataset(
                db_paths=["/nonexistent/path.db"],
                config=NNUEDatasetConfig(),
            )

        assert len(dataset) == 0
        assert "not found" in caplog.text

    def test_getitem_returns_tensors(self):
        """Test __getitem__ returns proper tensors."""
        # Create a minimal dataset with pre-populated samples
        dataset = NNUESQLiteDataset(db_paths=[], config=NNUEDatasetConfig())

        # Manually add a sample for testing
        features = np.random.randn(768).astype(np.float32)
        dataset.samples.append(NNUESample(
            features=features,
            value=0.5,
            player_number=1,
            game_id="test",
            move_number=10,
        ))

        x, y = dataset[0]

        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32
        assert x.shape == (768,)
        assert y.shape == (1,)

    def test_get_move_numbers(self):
        """Test get_move_numbers method."""
        dataset = NNUESQLiteDataset(db_paths=[], config=NNUEDatasetConfig())

        # Add samples with different move numbers
        for move_num in [5, 10, 15, 20]:
            features = np.random.randn(768).astype(np.float32)
            dataset.samples.append(NNUESample(
                features=features,
                value=0.5,
                player_number=1,
                game_id="test",
                move_number=move_num,
            ))

        move_numbers = dataset.get_move_numbers()

        assert isinstance(move_numbers, np.ndarray)
        assert list(move_numbers) == [5, 10, 15, 20]

    def test_compute_phase_balanced_weights_empty(self):
        """Test phase balancing with empty dataset."""
        dataset = NNUESQLiteDataset(db_paths=[], config=NNUEDatasetConfig())

        weights = dataset.compute_phase_balanced_weights()

        assert len(weights) == 0

    def test_compute_phase_balanced_weights(self):
        """Test phase balancing weights computation."""
        dataset = NNUESQLiteDataset(db_paths=[], config=NNUEDatasetConfig())

        # Add samples from different phases
        # Early game (0-39): 2 samples
        # Mid game (40-79): 3 samples
        # Late game (80+): 5 samples
        move_numbers = [10, 20, 50, 60, 70, 90, 100, 110, 120, 130]

        for move_num in move_numbers:
            features = np.random.randn(768).astype(np.float32)
            dataset.samples.append(NNUESample(
                features=features,
                value=0.5,
                player_number=1,
                game_id="test",
                move_number=move_num,
            ))

        weights = dataset.compute_phase_balanced_weights(
            early_end=40,
            mid_end=80,
            target_balance=(0.25, 0.35, 0.40),
        )

        assert len(weights) == 10
        assert weights.sum() == pytest.approx(1.0)

        # Check that early samples have higher weights (less samples, more weight)
        early_indices = [i for i, m in enumerate(move_numbers) if m < 40]
        late_indices = [i for i, m in enumerate(move_numbers) if m >= 80]

        # Early per-sample weight should be higher than late per-sample weight
        assert weights[early_indices[0]] > weights[late_indices[0]]


# =============================================================================
# NNUEStreamingDataset Tests
# =============================================================================


class TestNNUEStreamingDataset:
    """Tests for NNUEStreamingDataset class."""

    def test_init(self):
        """Test initialization."""
        dataset = NNUEStreamingDataset(
            db_paths=["/path/to/db.db"],
            config=NNUEDatasetConfig(),
            shuffle_games=True,
            seed=42,
            buffer_size=1000,
        )

        assert dataset.db_paths == ["/path/to/db.db"]
        assert dataset.shuffle_games is True
        assert dataset.base_seed == 42
        assert dataset.buffer_size == 1000

    def test_set_epoch(self):
        """Test epoch setting for shuffle variance."""
        dataset = NNUEStreamingDataset(
            db_paths=[],
            config=NNUEDatasetConfig(),
        )

        dataset.set_epoch(5)
        assert dataset.epoch == 5

        dataset.set_epoch(10)
        assert dataset.epoch == 10

    def test_iteration_empty_db_list(self):
        """Test iteration with empty database list."""
        dataset = NNUEStreamingDataset(
            db_paths=[],
            config=NNUEDatasetConfig(),
        )

        samples = list(dataset)
        assert samples == []

    def test_iteration_nonexistent_db(self):
        """Test iteration with nonexistent database."""
        dataset = NNUEStreamingDataset(
            db_paths=["/nonexistent/path.db"],
            config=NNUEDatasetConfig(),
        )

        samples = list(dataset)
        assert samples == []


# =============================================================================
# Count Available Samples Tests
# =============================================================================


class TestCountAvailableSamples:
    """Tests for count_available_samples function."""

    def test_empty_db_list(self):
        """Test counting with empty database list."""
        counts = count_available_samples([])

        assert counts == {"total": 0}

    def test_nonexistent_db(self):
        """Test counting with nonexistent database."""
        counts = count_available_samples(["/nonexistent/path.db"])

        assert counts["/nonexistent/path.db"] == 0
        assert counts["total"] == 0

    def test_valid_db(self, temp_db):
        """Test counting with valid database."""
        # The temp_db fixture creates a database with 1 completed game
        # but no snapshots, so count should be 0
        counts = count_available_samples([temp_db])

        assert temp_db in counts
        assert counts["total"] >= 0


# =============================================================================
# Integration-like Tests (with mocks)
# =============================================================================


class TestDatasetCaching:
    """Tests for dataset caching functionality."""

    def test_save_to_cache(self):
        """Test saving dataset to cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.npz"

            dataset = NNUESQLiteDataset(db_paths=[], config=NNUEDatasetConfig())

            # Add samples
            for i in range(5):
                features = np.random.randn(768).astype(np.float32)
                dataset.samples.append(NNUESample(
                    features=features,
                    value=0.5,
                    player_number=1,
                    game_id=f"game-{i}",
                    move_number=i * 10,
                ))

            # Save to cache
            dataset._save_to_cache(str(cache_path))

            assert cache_path.exists()

    def test_load_from_cache(self):
        """Test loading dataset from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.npz"

            # Create and save a dataset
            dataset1 = NNUESQLiteDataset(db_paths=[], config=NNUEDatasetConfig())

            for i in range(5):
                features = np.random.randn(768).astype(np.float32)
                dataset1.samples.append(NNUESample(
                    features=features,
                    value=0.5,
                    player_number=1,
                    game_id=f"game-{i}",
                    move_number=i * 10,
                ))

            dataset1._save_to_cache(str(cache_path))

            # Load from cache
            dataset2 = NNUESQLiteDataset(
                db_paths=[],
                config=NNUEDatasetConfig(),
                cache_path=str(cache_path),
            )

            assert len(dataset2) == 5
            assert dataset2.samples[0].game_id == "game-0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
