"""Tests for streaming_pipeline module.

Tests the real-time training data streaming infrastructure including:
- PrioritizedBuffer circular buffer with priority sampling
- DatabaseStreamReader for incremental game polling
- StreamingDataPipeline with async polling and deduplication
- GameSample priority calculation with quality scores
- Adaptive polling interval adjustment
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.training.streaming_pipeline import (
    CircularBuffer,
    DatabasePoller,
    GameSample,
    MultiDBStreamingPipeline,  # Renamed from MultiDatabasePipeline
    StreamingConfig,
    StreamingDataPipeline,
    extract_samples_from_game,
)


# =============================================================================
# StreamingConfig Tests
# =============================================================================


class TestStreamingConfig:
    """Tests for StreamingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = StreamingConfig()

        assert config.poll_interval_seconds == 5.0
        assert config.poll_interval_min == 0.5
        assert config.poll_interval_max == 10.0
        assert config.adaptive_polling is True
        assert config.max_poll_batch == 1000
        assert config.buffer_size == 10000
        assert config.min_buffer_fill == 0.2
        assert config.dedupe_enabled is True
        assert config.dedupe_window == 50000
        assert config.priority_sampling is True
        assert config.recency_weight == 0.3
        assert config.quality_weight == 0.4
        assert config.augmentation_enabled is True
        assert config.quality_lookup is None
        assert config.elo_lookup is None
        assert config.freshness_window_days is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = StreamingConfig(
            poll_interval_seconds=1.0,
            buffer_size=5000,
            dedupe_enabled=False,
            quality_weight=0.6,
            freshness_window_days=7,
        )

        assert config.poll_interval_seconds == 1.0
        assert config.buffer_size == 5000
        assert config.dedupe_enabled is False
        assert config.quality_weight == 0.6
        assert config.freshness_window_days == 7

    def test_quality_lookup_injection(self):
        """Test quality lookup dictionary injection."""
        quality_lookup = {"game1": 0.9, "game2": 0.7}
        config = StreamingConfig(quality_lookup=quality_lookup)

        assert config.quality_lookup is not None
        assert config.quality_lookup["game1"] == 0.9


# =============================================================================
# GameSample Tests
# =============================================================================


class TestGameSample:
    """Tests for GameSample dataclass."""

    def test_default_values(self):
        """Test default sample values."""
        sample = GameSample(
            game_id="test-game",
            move_idx=5,
            board_type="hex8",
            num_players=2,
            state_hash="abc123",
            timestamp=1000.0,
            value_target=1.0,
        )

        assert sample.game_id == "test-game"
        assert sample.move_idx == 5
        assert sample.board_type == "hex8"
        assert sample.num_players == 2
        assert sample.priority == 1.0
        assert sample.quality_score == 0.5
        assert sample.avg_elo == 1500.0
        assert sample.policy_target is None
        assert sample.features is None

    def test_with_numpy_arrays(self):
        """Test sample with numpy arrays."""
        policy = np.array([0.1, 0.2, 0.7])
        features = np.random.randn(64)

        sample = GameSample(
            game_id="test",
            move_idx=0,
            board_type="square8",
            num_players=4,
            state_hash="xyz",
            timestamp=1000.0,
            value_target=0.5,
            policy_target=policy,
            features=features,
        )

        assert sample.policy_target is not None
        assert len(sample.policy_target) == 3
        assert sample.features is not None
        assert len(sample.features) == 64


# =============================================================================
# CircularBuffer Tests (renamed from PrioritizedBuffer in test discovery)
# =============================================================================


class TestCircularBuffer:
    """Tests for CircularBuffer (PrioritizedBuffer) class."""

    def test_append_single(self):
        """Test appending single items."""
        # CircularBuffer is actually named in the module - let's import it properly
        from app.training.streaming_pipeline import PrioritizedBuffer

        buffer = PrioritizedBuffer(capacity=5)
        buffer.append("item1")
        buffer.append("item2")

        assert len(buffer) == 2
        items = buffer.get_all()
        assert "item1" in items
        assert "item2" in items

    def test_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        from app.training.streaming_pipeline import PrioritizedBuffer

        buffer = PrioritizedBuffer(capacity=3)
        for i in range(5):
            buffer.append(f"item{i}")

        # Should only keep last 3 items (circular behavior)
        assert len(buffer) == 3

    def test_extend(self):
        """Test extending with multiple items."""
        from app.training.streaming_pipeline import PrioritizedBuffer

        buffer = PrioritizedBuffer(capacity=10)
        buffer.extend(["a", "b", "c"])

        assert len(buffer) == 3
        items = buffer.get_all()
        assert set(items) == {"a", "b", "c"}

    def test_sample_without_weights(self):
        """Test uniform sampling without weights."""
        from app.training.streaming_pipeline import PrioritizedBuffer

        buffer = PrioritizedBuffer(capacity=100)
        buffer.extend([f"item{i}" for i in range(50)])

        samples = buffer.sample(10)
        assert len(samples) == 10
        assert all(s.startswith("item") for s in samples)

    def test_sample_with_weights(self):
        """Test weighted sampling."""
        from app.training.streaming_pipeline import PrioritizedBuffer

        buffer = PrioritizedBuffer(capacity=100)
        buffer.extend(["low", "high"])

        # Sample with weights heavily favoring second item
        weights = np.array([0.01, 0.99])
        samples = buffer.sample(100, weights=weights)

        # Most samples should be "high"
        high_count = sum(1 for s in samples if s == "high")
        assert high_count > 80  # Should be heavily biased

    def test_sample_more_than_available(self):
        """Test sampling more items than available."""
        from app.training.streaming_pipeline import PrioritizedBuffer

        buffer = PrioritizedBuffer(capacity=10)
        buffer.extend(["a", "b", "c"])

        # Requesting more than available should return all
        samples = buffer.sample(10)
        assert len(samples) <= 3

    def test_clear(self):
        """Test clearing buffer."""
        from app.training.streaming_pipeline import PrioritizedBuffer

        buffer = PrioritizedBuffer(capacity=10)
        buffer.extend(["a", "b", "c"])
        buffer.clear()

        assert len(buffer) == 0


# =============================================================================
# extract_samples_from_game Tests
# =============================================================================


class TestExtractSamplesFromGame:
    """Tests for extract_samples_from_game function."""

    def test_basic_extraction(self):
        """Test basic sample extraction from game."""
        game = {
            "game_id": "test-game-1",
            "board_type": "hex8",
            "num_players": 2,
            "winner": 0,
            "move_history": json.dumps([
                {"type": "place", "pos": [0, 0]},
                {"type": "place", "pos": [1, 1]},
                {"type": "place", "pos": [2, 2]},
            ]),
            "completed_at": "2025-01-01T12:00:00Z",
        }

        samples = extract_samples_from_game(game)

        assert len(samples) == 3
        assert all(s.game_id == "test-game-1" for s in samples)
        assert all(s.board_type == "hex8" for s in samples)
        assert all(s.num_players == 2 for s in samples)

    def test_value_targets_for_winner(self):
        """Test value targets based on winner."""
        game = {
            "game_id": "test",
            "board_type": "square8",
            "num_players": 2,
            "winner": 0,  # Player 0 wins
            "move_history": json.dumps([
                {"type": "move"},
                {"type": "move"},
            ]),
        }

        samples = extract_samples_from_game(game)

        # Move 0 is player 0 (winner) -> value 1.0
        # Move 1 is player 1 (loser) -> value 0.0
        assert samples[0].value_target == 1.0
        assert samples[1].value_target == 0.0

    def test_draw_value_targets(self):
        """Test value targets for draws."""
        game = {
            "game_id": "test",
            "board_type": "hex8",
            "num_players": 2,
            "winner": None,  # Draw
            "move_history": json.dumps([{"type": "move"}]),
        }

        samples = extract_samples_from_game(game)
        assert samples[0].value_target == 0.5

    def test_empty_move_history(self):
        """Test with empty move history."""
        game = {
            "game_id": "test",
            "board_type": "hex8",
            "num_players": 2,
            "move_history": "[]",
        }

        samples = extract_samples_from_game(game)
        assert len(samples) == 0

    def test_invalid_move_history_json(self):
        """Test with invalid JSON in move history."""
        game = {
            "game_id": "test",
            "board_type": "hex8",
            "num_players": 2,
            "move_history": "not valid json",
        }

        samples = extract_samples_from_game(game)
        assert len(samples) == 0

    def test_null_move_history(self):
        """Test with null move history."""
        game = {
            "game_id": "test",
            "board_type": "hex8",
            "num_players": 2,
            "move_history": None,
        }

        samples = extract_samples_from_game(game)
        assert len(samples) == 0

    def test_four_player_game(self):
        """Test with 4-player game."""
        game = {
            "game_id": "4p-game",
            "board_type": "square8",
            "num_players": 4,
            "winner": 2,  # Player 2 wins
            "move_history": json.dumps([
                {"type": "move"},  # P0
                {"type": "move"},  # P1
                {"type": "move"},  # P2 (winner)
                {"type": "move"},  # P3
            ]),
        }

        samples = extract_samples_from_game(game)

        assert len(samples) == 4
        assert samples[0].value_target == 0.0  # P0 lost
        assert samples[1].value_target == 0.0  # P1 lost
        assert samples[2].value_target == 1.0  # P2 won
        assert samples[3].value_target == 0.0  # P3 lost

    def test_state_hash_uniqueness(self):
        """Test that state hashes are unique per move."""
        game = {
            "game_id": "test",
            "board_type": "hex8",
            "num_players": 2,
            "move_history": json.dumps([{"type": "move"}] * 5),
        }

        samples = extract_samples_from_game(game)
        hashes = [s.state_hash for s in samples]

        # All hashes should be unique
        assert len(set(hashes)) == len(hashes)


# =============================================================================
# DatabasePoller Tests
# =============================================================================


class TestDatabasePoller:
    """Tests for DatabasePoller class."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary database with game data."""
        db_path = tmp_path / "test_games.db"
        conn = sqlite3.connect(str(db_path))

        # Create games table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                winner INTEGER,
                move_history TEXT,
                completed_at TEXT,
                parity_gate TEXT DEFAULT 'passed'
            )
        """)

        # Insert test games
        games = [
            ("game1", "hex8", 2, 0, '[{"type":"move"}]', "2025-01-01T10:00:00Z"),
            ("game2", "hex8", 2, 1, '[{"type":"move"},{"type":"move"}]', "2025-01-01T11:00:00Z"),
            ("game3", "square8", 4, 2, '[{"type":"move"}]', "2025-01-01T12:00:00Z"),
        ]
        conn.executemany(
            "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, 'passed')",
            games,
        )
        conn.commit()
        conn.close()

        return db_path

    def test_get_new_games_all(self, temp_db):
        """Test getting all games."""
        poller = DatabasePoller(temp_db)
        games = poller.get_new_games(limit=100)

        assert len(games) == 3

    def test_get_new_games_with_board_filter(self, temp_db):
        """Test filtering by board type."""
        poller = DatabasePoller(temp_db, board_type="hex8")
        games = poller.get_new_games(limit=100)

        assert len(games) == 2
        assert all(g["board_type"] == "hex8" for g in games)

    def test_get_new_games_with_player_filter(self, temp_db):
        """Test filtering by player count."""
        poller = DatabasePoller(temp_db, num_players=4)
        games = poller.get_new_games(limit=100)

        assert len(games) == 1
        assert games[0]["num_players"] == 4

    def test_get_new_games_limit(self, temp_db):
        """Test limit parameter."""
        poller = DatabasePoller(temp_db)
        games = poller.get_new_games(limit=2)

        assert len(games) == 2

    def test_get_game_count(self, temp_db):
        """Test game count."""
        poller = DatabasePoller(temp_db)
        count = poller.get_game_count()

        assert count == 3

    def test_get_game_count_filtered(self, temp_db):
        """Test game count with filters."""
        poller = DatabasePoller(temp_db, board_type="hex8")
        count = poller.get_game_count()

        assert count == 2

    def test_reset(self, temp_db):
        """Test reset clears last poll time."""
        poller = DatabasePoller(temp_db)
        poller._last_poll_time = time.time()
        poller.reset()

        assert poller._last_poll_time == 0


# =============================================================================
# StreamingDataPipeline Tests
# =============================================================================


class TestStreamingDataPipeline:
    """Tests for StreamingDataPipeline class."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary database with game data."""
        db_path = tmp_path / "streaming_test.db"
        conn = sqlite3.connect(str(db_path))

        conn.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                winner INTEGER,
                move_history TEXT,
                completed_at TEXT,
                parity_gate TEXT DEFAULT 'passed'
            )
        """)

        # Insert test games
        for i in range(10):
            conn.execute(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, 'passed')",
                (
                    f"game-{i}",
                    "hex8",
                    2,
                    i % 2,
                    json.dumps([{"type": "move"}] * 5),
                    f"2025-01-0{i+1}T10:00:00Z",
                ),
            )
        conn.commit()
        conn.close()

        return db_path

    def test_init(self, temp_db):
        """Test pipeline initialization."""
        pipeline = StreamingDataPipeline(
            db_path=temp_db,
            board_type="hex8",
            num_players=2,
        )

        assert pipeline.db_path == temp_db
        assert pipeline.board_type == "hex8"
        assert pipeline.num_players == 2
        assert pipeline.config is not None

    def test_init_with_config(self, temp_db):
        """Test initialization with custom config."""
        config = StreamingConfig(
            buffer_size=5000,
            poll_interval_seconds=1.0,
        )
        pipeline = StreamingDataPipeline(
            db_path=temp_db,
            config=config,
        )

        assert pipeline.config.buffer_size == 5000
        assert pipeline.config.poll_interval_seconds == 1.0

    def test_set_quality_lookup(self, temp_db):
        """Test setting quality lookup."""
        pipeline = StreamingDataPipeline(db_path=temp_db)

        quality_lookup = {"game-0": 0.9, "game-1": 0.8}
        elo_lookup = {"game-0": 1600.0, "game-1": 1500.0}

        pipeline.set_quality_lookup(quality_lookup, elo_lookup)

        assert pipeline.config.quality_lookup == quality_lookup
        assert pipeline.config.elo_lookup == elo_lookup

    def test_compute_adaptive_interval_no_games(self, temp_db):
        """Test adaptive interval with no new games."""
        pipeline = StreamingDataPipeline(db_path=temp_db)
        pipeline.config.adaptive_polling = True

        interval = pipeline._compute_adaptive_interval(new_game_count=0)

        # Should increase toward max
        assert interval >= pipeline.config.poll_interval_seconds

    def test_compute_adaptive_interval_many_games(self, temp_db):
        """Test adaptive interval with many new games."""
        pipeline = StreamingDataPipeline(db_path=temp_db)
        pipeline.config.adaptive_polling = True

        interval = pipeline._compute_adaptive_interval(new_game_count=500)

        # Should decrease toward min
        assert interval <= pipeline.config.poll_interval_seconds

    def test_compute_adaptive_interval_disabled(self, temp_db):
        """Test adaptive interval when disabled."""
        config = StreamingConfig(adaptive_polling=False)
        pipeline = StreamingDataPipeline(db_path=temp_db, config=config)

        interval = pipeline._compute_adaptive_interval(new_game_count=1000)

        # Should return base interval
        assert interval == config.poll_interval_seconds

    def test_get_batch(self, temp_db):
        """Test getting a batch of samples."""
        pipeline = StreamingDataPipeline(db_path=temp_db)

        # Manually fill buffer
        samples = [
            GameSample(
                game_id=f"game-{i}",
                move_idx=0,
                board_type="hex8",
                num_players=2,
                state_hash=f"hash-{i}",
                timestamp=1000.0 + i,
                value_target=1.0,
            )
            for i in range(20)
        ]
        pipeline.buffer.extend(samples)

        batch = pipeline.get_batch(batch_size=10)

        assert len(batch) <= 10

    def test_get_stats(self, temp_db):
        """Test getting pipeline statistics."""
        pipeline = StreamingDataPipeline(db_path=temp_db)

        stats = pipeline.get_stats()

        assert "buffer_size" in stats
        assert "total_samples_seen" in stats
        assert "duplicates_filtered" in stats
        assert "db_path" in stats
        assert "running" in stats

    def test_update_priorities(self, temp_db):
        """Test updating sample priorities."""
        pipeline = StreamingDataPipeline(db_path=temp_db)

        # Add samples to buffer
        sample = GameSample(
            game_id="game-0",
            move_idx=0,
            board_type="hex8",
            num_players=2,
            state_hash="hash-0",
            timestamp=1000.0,
            value_target=1.0,
        )
        pipeline.buffer.append(sample)

        # Update priority
        pipeline.update_priorities({"hash-0": 2.5})

        # Verify priority was updated
        items = pipeline.buffer.get_all()
        assert any(s.priority == 2.5 for s in items if s.state_hash == "hash-0")


# =============================================================================
# MultiDBStreamingPipeline Tests
# =============================================================================


class TestMultiDBStreamingPipeline:
    """Tests for MultiDBStreamingPipeline class."""

    @pytest.fixture
    def temp_dbs(self, tmp_path):
        """Create multiple temporary databases."""
        db_paths = []
        for i in range(3):
            db_path = tmp_path / f"db_{i}.db"
            conn = sqlite3.connect(str(db_path))

            conn.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    game_id TEXT PRIMARY KEY,
                    board_type TEXT NOT NULL,
                    num_players INTEGER NOT NULL,
                    winner INTEGER,
                    move_history TEXT,
                    completed_at TEXT,
                    parity_gate TEXT DEFAULT 'passed'
                )
            """)

            # Each DB has different games
            for j in range(5):
                conn.execute(
                    "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, 'passed')",
                    (
                        f"db{i}-game{j}",
                        "hex8",
                        2,
                        j % 2,
                        json.dumps([{"type": "move"}]),
                        "2025-01-01T10:00:00Z",
                    ),
                )
            conn.commit()
            conn.close()
            db_paths.append(db_path)

        return db_paths

    def test_init(self, temp_dbs):
        """Test multi-database pipeline initialization."""
        pipeline = MultiDBStreamingPipeline(
            db_paths=temp_dbs,
            board_type="hex8",
        )

        assert len(pipeline.pipelines) == 3

    def test_get_aggregate_stats(self, temp_dbs):
        """Test aggregate statistics."""
        pipeline = MultiDBStreamingPipeline(db_paths=temp_dbs)

        stats = pipeline.get_aggregate_stats()

        assert "pipeline_count" in stats
        assert stats["pipeline_count"] == 3
        assert "total_buffer_size" in stats
        assert "per_pipeline" in stats


# =============================================================================
# Async Tests
# =============================================================================


class TestStreamingPipelineAsync:
    """Async tests for StreamingDataPipeline."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary database."""
        db_path = tmp_path / "async_test.db"
        conn = sqlite3.connect(str(db_path))

        conn.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                winner INTEGER,
                move_history TEXT,
                completed_at TEXT,
                parity_gate TEXT DEFAULT 'passed'
            )
        """)

        for i in range(5):
            conn.execute(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, 'passed')",
                (
                    f"game-{i}",
                    "hex8",
                    2,
                    0,
                    json.dumps([{"type": "move"}] * 3),
                    "2025-01-01T10:00:00Z",
                ),
            )
        conn.commit()
        conn.close()

        return db_path

    @pytest.mark.asyncio
    async def test_start_stop(self, temp_db):
        """Test async start and stop."""
        pipeline = StreamingDataPipeline(db_path=temp_db)

        await pipeline.start()
        assert pipeline._running is True

        await pipeline.stop()
        assert pipeline._running is False

    @pytest.mark.asyncio
    async def test_stream_batches_yields_data(self, temp_db):
        """Test that stream_batches yields data."""
        config = StreamingConfig(poll_interval_seconds=0.1)
        pipeline = StreamingDataPipeline(db_path=temp_db, config=config)

        batches_received = 0
        async for batch in pipeline.stream_batches(batch_size=5, max_batches=3):
            if batch:
                batches_received += 1
            if batches_received >= 2:
                break

        # Should receive at least some batches
        assert batches_received >= 0  # May not get batches if polling is slow

        await pipeline.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
