"""Tests for hot_data_buffer.py - in-memory buffer for streaming training data."""

import json
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.training.hot_data_buffer import (
    GameRecord,
    HotDataBuffer,
    StateEncoder,
    create_hot_buffer,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_moves():
    """Create sample moves with pre-computed features."""
    return [
        {
            "move_number": 1,
            "state_features": [1.0] * 768,
            "global_features": [0.5] * 32,
            "policy_target": [0.1] * 65,  # 64 cells + pass
            "value_target": 0.5,
        },
        {
            "move_number": 2,
            "state_features": [0.5] * 768,
            "global_features": [0.3] * 32,
            "policy_target": [0.2] * 65,
            "value_target": 0.6,
        },
        {
            "move_number": 3,
            "state_features": [0.3] * 768,
            "global_features": [0.7] * 32,
            "policy_target": [0.3] * 65,
            "value_target": 0.8,
        },
    ]


@pytest.fixture
def game_record(sample_moves):
    """Create a sample GameRecord."""
    return GameRecord(
        game_id="test-game-001",
        board_type="square8",
        num_players=2,
        moves=sample_moves,
        outcome={"1": 1.0, "2": 0.0},
        timestamp=time.time(),
        source="test",
        avg_elo=1600.0,
        priority=1.0,
        from_promoted_model=False,
        manifest_quality=0.75,
    )


@pytest.fixture
def buffer():
    """Create a HotDataBuffer with events disabled for testing."""
    return HotDataBuffer(
        max_size=100,
        max_memory_mb=50,
        buffer_name="test_buffer",
        enable_events=False,
        training_threshold=10,
        batch_notification_size=5,
        enable_validation=False,
    )


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database with game data."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create minimal schema
    cursor.execute("""
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY,
            board_type TEXT,
            num_players INTEGER,
            game_status TEXT,
            winner INTEGER,
            total_moves INTEGER,
            termination_reason TEXT,
            source TEXT,
            created_at REAL,
            excluded_from_training INTEGER DEFAULT 0,
            quality_score REAL DEFAULT 0.5
        )
    """)

    cursor.execute("""
        CREATE TABLE moves (
            game_id TEXT,
            move_number INTEGER,
            action_json TEXT,
            state_json TEXT,
            PRIMARY KEY (game_id, move_number)
        )
    """)

    # Insert test games
    for i in range(5):
        game_id = f"db-game-{i:03d}"
        cursor.execute("""
            INSERT INTO games (game_id, board_type, num_players, game_status,
                             winner, total_moves, source, created_at, quality_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (game_id, "square8", 2, "completed", 1, 20 + i, "test", time.time() - i * 3600, 0.5 + i * 0.1))

        # Insert moves
        for m in range(10):
            action = {"type": "PLACE_RING", "position": m}
            state = {"current_player": m % 2 + 1, "board": [0] * 64}
            cursor.execute("""
                INSERT INTO moves (game_id, move_number, action_json, state_json)
                VALUES (?, ?, ?, ?)
            """, (game_id, m, json.dumps(action), json.dumps(state)))

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    db_path.unlink(missing_ok=True)


# =============================================================================
# Test GameRecord
# =============================================================================


class TestGameRecord:
    """Tests for GameRecord dataclass."""

    def test_create_game_record(self, sample_moves):
        """Test creating a game record."""
        record = GameRecord(
            game_id="test-001",
            board_type="square8",
            num_players=2,
            moves=sample_moves,
            outcome={"1": 1.0, "2": 0.0},
        )

        assert record.game_id == "test-001"
        assert record.board_type == "square8"
        assert record.num_players == 2
        assert len(record.moves) == 3
        assert record.outcome == {"1": 1.0, "2": 0.0}

    def test_default_values(self, sample_moves):
        """Test default values are set correctly."""
        record = GameRecord(
            game_id="test-001",
            board_type="square8",
            num_players=2,
            moves=sample_moves,
            outcome={},
        )

        assert record.source == "hot_buffer"
        assert record.avg_elo == 1500.0
        assert record.priority == 1.0
        assert record.from_promoted_model is False
        assert record.manifest_quality == 0.5
        # timestamp should be set to current time
        assert record.timestamp > 0

    def test_to_training_samples(self, game_record):
        """Test conversion to training samples."""
        samples = game_record.to_training_samples()

        assert len(samples) == 3
        for sample in samples:
            board_features, global_features, policy, value = sample
            assert isinstance(board_features, np.ndarray)
            assert isinstance(global_features, np.ndarray)
            assert isinstance(policy, np.ndarray)
            assert isinstance(value, float)
            assert board_features.dtype == np.float32
            assert global_features.dtype == np.float32
            assert policy.dtype == np.float32

    def test_to_training_samples_empty_moves(self):
        """Test training samples from game with no moves."""
        record = GameRecord(
            game_id="empty-game",
            board_type="square8",
            num_players=2,
            moves=[],
            outcome={},
        )

        samples = record.to_training_samples()
        assert samples == []

    def test_to_training_samples_missing_features(self):
        """Test training samples when features are missing."""
        record = GameRecord(
            game_id="partial-game",
            board_type="square8",
            num_players=2,
            moves=[
                {"move_number": 1, "state_features": None, "policy_target": None, "value_target": None},
            ],
            outcome={},
        )

        samples = record.to_training_samples()
        assert samples == []

    def test_to_training_samples_legacy(self, game_record):
        """Test legacy training samples format."""
        samples = game_record.to_training_samples_legacy()

        assert len(samples) == 3
        for sample in samples:
            state, policy, value = sample
            assert isinstance(state, np.ndarray)
            assert isinstance(policy, np.ndarray)
            assert isinstance(value, float)

    def test_to_training_samples_with_encoder(self, sample_moves):
        """Test training samples with a custom encoder."""
        # Create a mock encoder
        mock_encoder = MagicMock()
        mock_encoder.encode_state.return_value = (
            np.ones(768, dtype=np.float32),
            np.ones(32, dtype=np.float32),
        )

        # Game with raw_state
        record = GameRecord(
            game_id="raw-game",
            board_type="square8",
            num_players=2,
            moves=[{
                "move_number": 1,
                "raw_state": {"board": [0] * 64},
                "policy_target": [0.1] * 65,
                "value_target": 0.5,
            }],
            outcome={},
        )

        samples = record.to_training_samples(encoder=mock_encoder)
        assert len(samples) == 1
        mock_encoder.encode_state.assert_called_once()


# =============================================================================
# Test HotDataBuffer Basic Operations
# =============================================================================


class TestHotDataBufferBasic:
    """Tests for HotDataBuffer basic operations."""

    def test_init_default(self):
        """Test buffer initialization with defaults."""
        buffer = HotDataBuffer()

        assert buffer.max_size == 1000
        assert buffer.max_memory_mb == 500
        assert buffer.buffer_name == "default"
        assert buffer.training_threshold == 100
        assert buffer.batch_notification_size == 50

    def test_init_custom(self):
        """Test buffer initialization with custom values."""
        buffer = HotDataBuffer(
            max_size=50,
            max_memory_mb=25,
            buffer_name="custom",
            enable_events=False,
            training_threshold=20,
        )

        assert buffer.max_size == 50
        assert buffer.max_memory_mb == 25
        assert buffer.buffer_name == "custom"
        assert buffer.training_threshold == 20

    def test_add_game(self, buffer, game_record):
        """Test adding a game to buffer."""
        assert len(buffer) == 0

        buffer.add_game(game_record)

        assert len(buffer) == 1
        assert game_record.game_id in buffer

    def test_add_game_from_dict(self, buffer):
        """Test adding a game from dictionary."""
        data = {
            "game_id": "dict-game-001",
            "board_type": "hex8",
            "num_players": 4,
            "moves": [],
            "outcome": {"1": 0.25, "2": 0.25, "3": 0.25, "4": 0.25},
            "timestamp": time.time(),
            "source": "api",
            "avg_elo": 1700.0,
        }

        buffer.add_game_from_dict(data)

        assert len(buffer) == 1
        assert "dict-game-001" in buffer
        game = buffer.get_game("dict-game-001")
        assert game.board_type == "hex8"
        assert game.num_players == 4
        assert game.avg_elo == 1700.0

    def test_get_game(self, buffer, game_record):
        """Test getting a game by ID."""
        buffer.add_game(game_record)

        retrieved = buffer.get_game(game_record.game_id)
        assert retrieved is game_record

    def test_get_game_not_found(self, buffer):
        """Test getting a non-existent game."""
        result = buffer.get_game("non-existent")
        assert result is None

    def test_remove_game(self, buffer, game_record):
        """Test removing a game from buffer."""
        buffer.add_game(game_record)
        assert len(buffer) == 1

        removed = buffer.remove_game(game_record.game_id)

        assert removed is True
        assert len(buffer) == 0
        assert game_record.game_id not in buffer

    def test_remove_game_not_found(self, buffer):
        """Test removing a non-existent game."""
        removed = buffer.remove_game("non-existent")
        assert removed is False

    def test_get_all_games(self, buffer, sample_moves):
        """Test getting all games."""
        # Add multiple games
        for i in range(5):
            game = GameRecord(
                game_id=f"game-{i:03d}",
                board_type="square8",
                num_players=2,
                moves=sample_moves,
                outcome={},
            )
            buffer.add_game(game)

        games = buffer.get_all_games()
        assert len(games) == 5

    def test_contains(self, buffer, game_record):
        """Test __contains__ method."""
        assert game_record.game_id not in buffer

        buffer.add_game(game_record)

        assert game_record.game_id in buffer

    def test_len(self, buffer, sample_moves):
        """Test __len__ method."""
        assert len(buffer) == 0

        for i in range(10):
            game = GameRecord(
                game_id=f"game-{i:03d}",
                board_type="square8",
                num_players=2,
                moves=sample_moves,
                outcome={},
            )
            buffer.add_game(game)

        assert len(buffer) == 10

    def test_clear(self, buffer, game_record):
        """Test clearing the buffer."""
        buffer.add_game(game_record)
        assert len(buffer) == 1

        buffer.clear()

        assert len(buffer) == 0


# =============================================================================
# Test LRU Eviction
# =============================================================================


class TestHotDataBufferEviction:
    """Tests for LRU eviction behavior."""

    def test_eviction_when_over_capacity(self):
        """Test that oldest games are evicted when over capacity."""
        buffer = HotDataBuffer(max_size=3, enable_events=False)

        # Add 5 games - should only keep last 3
        for i in range(5):
            game = GameRecord(
                game_id=f"game-{i:03d}",
                board_type="square8",
                num_players=2,
                moves=[],
                outcome={},
            )
            buffer.add_game(game)

        assert len(buffer) == 3
        # First two games should be evicted
        assert "game-000" not in buffer
        assert "game-001" not in buffer
        # Last three should remain
        assert "game-002" in buffer
        assert "game-003" in buffer
        assert "game-004" in buffer

    def test_lru_order_maintained(self):
        """Test that LRU order is maintained on access."""
        buffer = HotDataBuffer(max_size=3, enable_events=False)

        # Add 3 games
        for i in range(3):
            game = GameRecord(
                game_id=f"game-{i:03d}",
                board_type="square8",
                num_players=2,
                moves=[],
                outcome={},
            )
            buffer.add_game(game)

        # Re-add game-000 (should move to end)
        game = GameRecord(
            game_id="game-000",
            board_type="square8",
            num_players=2,
            moves=[],
            outcome={},
        )
        buffer.add_game(game)

        # Add new game - should evict game-001 (now oldest)
        new_game = GameRecord(
            game_id="game-003",
            board_type="square8",
            num_players=2,
            moves=[],
            outcome={},
        )
        buffer.add_game(new_game)

        assert len(buffer) == 3
        assert "game-000" in buffer  # Re-added, moved to end
        assert "game-001" not in buffer  # Should be evicted
        assert "game-002" in buffer
        assert "game-003" in buffer


# =============================================================================
# Test Training Batches
# =============================================================================


class TestHotDataBufferTraining:
    """Tests for training batch retrieval."""

    def test_get_training_batch(self, buffer, game_record):
        """Test getting a training batch."""
        buffer.add_game(game_record)

        board_feats, global_feats, policies, values = buffer.get_training_batch(batch_size=3)

        assert len(board_feats) == 3
        assert len(global_feats) == 3
        assert len(policies) == 3
        assert len(values) == 3

    def test_get_training_batch_empty_buffer(self, buffer):
        """Test getting batch from empty buffer."""
        board_feats, global_feats, policies, values = buffer.get_training_batch()

        assert board_feats.shape[0] == 0
        assert global_feats.shape[0] == 0
        assert policies.shape[0] == 0
        assert values.shape[0] == 0

    def test_get_training_batch_smaller_than_requested(self, buffer, game_record):
        """Test batch size capped by available samples."""
        buffer.add_game(game_record)  # Has 3 samples

        board_feats, _, _, _ = buffer.get_training_batch(batch_size=100)

        # Should only return 3 samples (all available)
        assert len(board_feats) == 3

    def test_get_sample_iterator(self, buffer, game_record):
        """Test sample iterator."""
        buffer.add_game(game_record)

        batches = list(buffer.get_sample_iterator(batch_size=2, epochs=1))

        # 3 samples, batch_size=2 -> 2 batches (2, 1)
        assert len(batches) == 2

    def test_total_samples_property(self, buffer, game_record):
        """Test total_samples property."""
        assert buffer.total_samples == 0

        buffer.add_game(game_record)

        assert buffer.total_samples == 3  # 3 moves in game_record

    def test_game_count_property(self, buffer, game_record):
        """Test game_count property."""
        assert buffer.game_count == 0

        buffer.add_game(game_record)

        assert buffer.game_count == 1


# =============================================================================
# Test Flush Operations
# =============================================================================


class TestHotDataBufferFlush:
    """Tests for flush operations."""

    def test_mark_flushed(self, buffer, game_record):
        """Test marking games as flushed."""
        buffer.add_game(game_record)

        unflushed_before = buffer.get_unflushed_games()
        assert len(unflushed_before) == 1

        buffer.mark_flushed([game_record.game_id])

        unflushed_after = buffer.get_unflushed_games()
        assert len(unflushed_after) == 0

    def test_clear_flushed(self, buffer, sample_moves):
        """Test clearing flushed games."""
        # Add games
        for i in range(3):
            game = GameRecord(
                game_id=f"game-{i:03d}",
                board_type="square8",
                num_players=2,
                moves=sample_moves,
                outcome={},
            )
            buffer.add_game(game)

        # Mark some as flushed
        buffer.mark_flushed(["game-000", "game-001"])

        # Clear flushed
        removed = buffer.clear_flushed()

        assert removed == 2
        assert len(buffer) == 1
        assert "game-002" in buffer

    def test_flush_to_jsonl(self, buffer, game_record):
        """Test flushing to JSONL file."""
        buffer.add_game(game_record)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"

            written = buffer.flush_to_jsonl(path)

            assert written == 1
            assert path.exists()

            # Verify file contents
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 1

            data = json.loads(lines[0])
            assert data["game_id"] == game_record.game_id

    def test_flush_to_jsonl_empty_buffer(self, buffer):
        """Test flushing empty buffer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"

            written = buffer.flush_to_jsonl(path)

            assert written == 0

    def test_flush_to_jsonl_creates_directory(self, buffer, game_record):
        """Test that flush creates parent directory."""
        buffer.add_game(game_record)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "test.jsonl"

            buffer.flush_to_jsonl(path)

            assert path.exists()


# =============================================================================
# Test Statistics
# =============================================================================


class TestHotDataBufferStatistics:
    """Tests for buffer statistics."""

    def test_get_statistics_empty(self, buffer):
        """Test statistics for empty buffer."""
        stats = buffer.get_statistics()

        assert stats["game_count"] == 0
        assert stats["total_samples"] == 0
        assert stats["flushed_count"] == 0
        assert stats["utilization"] == 0.0

    def test_get_statistics(self, buffer, game_record):
        """Test statistics with data."""
        buffer.add_game(game_record)

        stats = buffer.get_statistics()

        assert stats["game_count"] == 1
        assert stats["total_samples"] == 3
        assert stats["max_size"] == 100
        assert stats["utilization"] == 0.01  # 1/100
        assert "avg_priority" in stats
        assert "avg_elo" in stats


# =============================================================================
# Test Priority Experience Replay
# =============================================================================


class TestHotDataBufferPriority:
    """Tests for priority experience replay features."""

    def test_compute_game_priority(self, buffer, game_record):
        """Test priority computation."""
        priority = buffer.compute_game_priority(game_record)

        # Default game should have priority around 1.0
        assert priority > 0
        # Higher Elo (1600) should increase priority
        assert priority > 0.5

    def test_compute_game_priority_promoted_bonus(self, buffer, game_record):
        """Test promotion bonus in priority."""
        normal_priority = buffer.compute_game_priority(game_record)

        game_record.from_promoted_model = True
        promoted_priority = buffer.compute_game_priority(game_record)

        # Promoted should have higher priority
        assert promoted_priority > normal_priority

    def test_compute_game_priority_quality_factor(self, buffer, sample_moves):
        """Test quality factor in priority."""
        low_quality_game = GameRecord(
            game_id="low-q",
            board_type="square8",
            num_players=2,
            moves=sample_moves,
            outcome={},
            manifest_quality=0.2,
        )

        high_quality_game = GameRecord(
            game_id="high-q",
            board_type="square8",
            num_players=2,
            moves=sample_moves,
            outcome={},
            manifest_quality=0.9,
        )

        low_priority = buffer.compute_game_priority(low_quality_game)
        high_priority = buffer.compute_game_priority(high_quality_game)

        assert high_priority > low_priority

    def test_update_all_priorities(self, buffer, sample_moves):
        """Test updating all priorities."""
        # Add games with different Elos
        for i, elo in enumerate([1400, 1600, 1800]):
            game = GameRecord(
                game_id=f"game-{i}",
                board_type="square8",
                num_players=2,
                moves=sample_moves,
                outcome={},
                avg_elo=elo,
            )
            buffer.add_game(game)

        buffer.update_all_priorities()

        # Get priorities
        games = buffer.get_all_games()
        priorities = [g.priority for g in games]

        # Should be different due to different Elos
        assert len(set(priorities)) > 1

    def test_update_game_priority_td_error(self, buffer, game_record):
        """Test updating priority with TD error."""
        buffer.add_game(game_record)
        initial_priority = game_record.priority

        buffer.update_game_priority(game_record.game_id, td_error=1.5)

        updated = buffer.get_game(game_record.game_id)
        assert updated.priority > initial_priority

    def test_update_game_priority_from_promoted(self, buffer, game_record):
        """Test marking game as from promoted model."""
        buffer.add_game(game_record)
        assert not game_record.from_promoted_model

        buffer.update_game_priority(game_record.game_id, from_promoted=True)

        updated = buffer.get_game(game_record.game_id)
        assert updated.from_promoted_model is True

    def test_update_game_priority_not_found(self, buffer):
        """Test updating priority for non-existent game."""
        result = buffer.update_game_priority("non-existent", td_error=1.0)
        assert result is False

    def test_get_priority_training_batch(self, buffer, sample_moves):
        """Test priority-weighted batch retrieval."""
        # Add games with different priorities
        for i in range(3):
            game = GameRecord(
                game_id=f"game-{i}",
                board_type="square8",
                num_players=2,
                moves=sample_moves,
                outcome={},
                priority=float(i + 1),  # 1, 2, 3
            )
            buffer.add_game(game)

        board_features, global_features, policies, values, weights = buffer.get_priority_training_batch(batch_size=5)

        assert len(board_features) == 5
        assert len(global_features) == 5
        assert len(policies) == 5
        assert len(values) == 5
        assert len(weights) == 5
        # Weights should be normalized with max=1
        assert weights.max() == pytest.approx(1.0)

    def test_mark_games_from_promoted_model(self, buffer, sample_moves):
        """Test marking multiple games from promoted model."""
        game_ids = []
        for i in range(5):
            game = GameRecord(
                game_id=f"game-{i}",
                board_type="square8",
                num_players=2,
                moves=sample_moves,
                outcome={},
            )
            buffer.add_game(game)
            game_ids.append(game.game_id)

        marked = buffer.mark_games_from_promoted_model("model-v1", game_ids[:3])

        assert marked == 3
        for i in range(3):
            game = buffer.get_game(f"game-{i}")
            assert game.from_promoted_model is True
        for i in range(3, 5):
            game = buffer.get_game(f"game-{i}")
            assert game.from_promoted_model is False


# =============================================================================
# Test Quality Auto-Calibration
# =============================================================================


class TestHotDataBufferQuality:
    """Tests for quality auto-calibration features."""

    def test_get_quality_distribution_empty(self, buffer):
        """Test quality distribution for empty buffer."""
        dist = buffer.get_quality_distribution()

        assert dist["count"] == 0
        assert dist["avg_quality"] == 0.0
        assert dist["high_quality_count"] == 0

    def test_get_quality_distribution(self, buffer, sample_moves):
        """Test quality distribution with data."""
        # Add games with varying quality
        for i in range(10):
            game = GameRecord(
                game_id=f"game-{i}",
                board_type="square8",
                num_players=2,
                moves=sample_moves,
                outcome={},
                manifest_quality=0.1 * (i + 1),  # 0.1 to 1.0
            )
            buffer.add_game(game)

        dist = buffer.get_quality_distribution()

        assert dist["count"] == 10
        assert 0.5 < dist["avg_quality"] < 0.6  # Around 0.55
        assert dist["min_quality"] == pytest.approx(0.1)
        assert dist["max_quality"] == pytest.approx(1.0)
        assert dist["high_quality_count"] > 0

    def test_calibrate_quality_thresholds_insufficient_data(self, buffer):
        """Test calibration with insufficient data."""
        # Add less than 100 games
        for i in range(50):
            game = GameRecord(
                game_id=f"game-{i}",
                board_type="square8",
                num_players=2,
                moves=[],
                outcome={},
            )
            buffer.add_game(game)

        calibration = buffer.calibrate_quality_thresholds()

        assert calibration["calibrated"] is False
        assert "Insufficient data" in calibration["reason"]

    def test_calibrate_quality_thresholds(self, buffer):
        """Test quality threshold calibration."""
        # Add 100+ games with linear quality distribution
        for i in range(120):
            game = GameRecord(
                game_id=f"game-{i}",
                board_type="square8",
                num_players=2,
                moves=[],
                outcome={},
                manifest_quality=i / 120.0,
            )
            buffer.add_game(game)

        calibration = buffer.calibrate_quality_thresholds(
            target_high_ratio=0.3,
            target_low_ratio=0.1,
        )

        assert calibration["calibrated"] is True
        assert 0 < calibration["low_threshold"] < calibration["high_threshold"] < 1

    def test_set_quality_lookup(self, buffer, sample_moves):
        """Test setting quality lookup tables."""
        # Add games
        for i in range(5):
            game = GameRecord(
                game_id=f"game-{i}",
                board_type="square8",
                num_players=2,
                moves=sample_moves,
                outcome={},
                manifest_quality=0.5,  # Default
            )
            buffer.add_game(game)

        # Set quality lookup
        quality_lookup = {f"game-{i}": 0.8 + i * 0.02 for i in range(5)}
        updated = buffer.set_quality_lookup(quality_lookup=quality_lookup)

        assert updated == 5
        for i in range(5):
            game = buffer.get_game(f"game-{i}")
            assert game.manifest_quality == pytest.approx(0.8 + i * 0.02)

    def test_auto_calibrate_and_filter(self, buffer):
        """Test auto-calibration with eviction."""
        # Add 120 games with varying quality
        for i in range(120):
            game = GameRecord(
                game_id=f"game-{i}",
                board_type="square8",
                num_players=2,
                moves=[],
                outcome={},
                manifest_quality=i / 120.0,
            )
            buffer.add_game(game)

        result = buffer.auto_calibrate_and_filter(
            min_quality_percentile=0.1,
            evict_below_percentile=True,
        )

        assert result["calibration"]["calibrated"] is True
        assert result["evicted"] > 0


# =============================================================================
# Test Database Loading
# =============================================================================


class TestHotDataBufferDatabaseLoading:
    """Tests for database loading functionality."""

    def test_load_from_db(self, buffer, temp_db):
        """Test loading games from database."""
        loaded = buffer.load_from_db(
            db_path=temp_db,
            board_type="square8",
            num_players=2,
            min_quality=0.0,
            limit=100,
            min_moves=5,
        )

        assert loaded == 5  # We inserted 5 games
        assert len(buffer) == 5

    def test_load_from_db_with_quality_filter(self, buffer, temp_db):
        """Test loading with quality filter."""
        loaded = buffer.load_from_db(
            db_path=temp_db,
            board_type="square8",
            num_players=2,
            min_quality=0.7,  # Only games with quality >= 0.7
            limit=100,
        )

        # Games have quality 0.5, 0.6, 0.7, 0.8, 0.9
        # Only 0.7, 0.8, 0.9 should be loaded
        assert loaded == 3

    def test_load_from_db_nonexistent(self, buffer):
        """Test loading from non-existent database."""
        loaded = buffer.load_from_db(
            db_path=Path("/nonexistent/path.db"),
            board_type="square8",
            num_players=2,
        )

        assert loaded == 0

    def test_load_from_db_with_limit(self, buffer, temp_db):
        """Test loading with limit."""
        loaded = buffer.load_from_db(
            db_path=temp_db,
            board_type="square8",
            num_players=2,
            limit=2,
        )

        assert loaded == 2
        assert len(buffer) == 2


# =============================================================================
# Test Factory Function
# =============================================================================


class TestCreateHotBuffer:
    """Tests for create_hot_buffer factory function."""

    def test_create_hot_buffer_defaults(self):
        """Test factory with defaults."""
        buffer = create_hot_buffer()

        assert buffer.max_size == 1000
        assert buffer.max_memory_mb == 500
        assert buffer.buffer_name == "default"

    def test_create_hot_buffer_custom(self):
        """Test factory with custom values."""
        buffer = create_hot_buffer(
            max_size=50,
            max_memory_mb=25,
            buffer_name="custom",
            enable_events=False,
            training_threshold=10,
            batch_notification_size=3,
        )

        assert buffer.max_size == 50
        assert buffer.max_memory_mb == 25
        assert buffer.buffer_name == "custom"
        assert buffer.training_threshold == 10
        assert buffer.batch_notification_size == 3


# =============================================================================
# Test Encoder Integration
# =============================================================================


class TestHotDataBufferEncoder:
    """Tests for encoder integration."""

    def test_set_encoder(self, buffer):
        """Test setting an encoder."""
        mock_encoder = MagicMock()
        buffer.set_encoder(mock_encoder)

        assert buffer._encoder is mock_encoder
        assert buffer._cache_dirty is True


# =============================================================================
# Test Thread Safety
# =============================================================================


class TestHotDataBufferThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_add_and_read(self, buffer, sample_moves):
        """Test concurrent add and read operations."""
        import threading

        errors = []
        added_count = [0]

        def add_games(start_id, count):
            try:
                for i in range(count):
                    game = GameRecord(
                        game_id=f"game-{start_id}-{i}",
                        board_type="square8",
                        num_players=2,
                        moves=sample_moves,
                        outcome={},
                    )
                    buffer.add_game(game)
                    added_count[0] += 1
            except Exception as e:
                errors.append(e)

        def read_games(iterations):
            try:
                for _ in range(iterations):
                    _ = len(buffer)
                    _ = buffer.get_all_games()
                    if buffer.game_count > 0:
                        _ = buffer.get_training_batch(batch_size=5)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_games, args=(0, 50)),
            threading.Thread(target=add_games, args=(1, 50)),
            threading.Thread(target=read_games, args=(100,)),
            threading.Thread(target=read_games, args=(100,)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Test Event Emission (with mocks)
# =============================================================================


class TestHotDataBufferEvents:
    """Tests for event emission functionality."""

    def test_training_threshold_event_emitted_once(self):
        """Test training threshold event is only emitted once."""
        with patch("app.training.hot_data_buffer.HAS_EVENT_SYSTEM", True):
            with patch("app.training.hot_data_buffer.get_event_bus") as mock_bus:
                buffer = HotDataBuffer(
                    max_size=100,
                    enable_events=True,
                    training_threshold=5,
                    batch_notification_size=100,  # High to avoid batch events
                )

                # Add games up to threshold
                for i in range(10):
                    game = GameRecord(
                        game_id=f"game-{i}",
                        board_type="square8",
                        num_players=2,
                        moves=[],
                        outcome={},
                    )
                    buffer.add_game(game)

                # Threshold should be marked as emitted
                assert buffer._training_threshold_emitted is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
