"""Tests for ephemeral_data_guard module (December 2025).

Tests the EphemeralDataGuard class that protects against data loss
from ephemeral hosts like Vast.ai that can terminate without warning.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest


class TestHostCheckpoint:
    """Tests for HostCheckpoint dataclass."""

    def test_unsynced_games_calculation(self):
        """Test unsynced_games property calculates correctly."""
        from app.coordination.ephemeral_data_guard import HostCheckpoint

        checkpoint = HostCheckpoint(
            host="vast-12345",
            is_ephemeral=True,
            last_checkpoint_time=time.time(),
            last_heartbeat_time=time.time(),
            games_generated=100,
            games_synced=60,
            last_game_id="game-001",
        )
        assert checkpoint.unsynced_games == 40

    def test_unsynced_games_never_negative(self):
        """Test unsynced_games is never negative."""
        from app.coordination.ephemeral_data_guard import HostCheckpoint

        checkpoint = HostCheckpoint(
            host="vast-12345",
            is_ephemeral=True,
            last_checkpoint_time=time.time(),
            last_heartbeat_time=time.time(),
            games_generated=50,
            games_synced=100,  # More synced than generated (edge case)
            last_game_id="game-001",
        )
        assert checkpoint.unsynced_games == 0

    def test_seconds_since_heartbeat(self):
        """Test seconds_since_heartbeat calculation."""
        from app.coordination.ephemeral_data_guard import HostCheckpoint

        now = time.time()
        checkpoint = HostCheckpoint(
            host="vast-12345",
            is_ephemeral=True,
            last_checkpoint_time=now,
            last_heartbeat_time=now - 60,  # 60 seconds ago
            games_generated=10,
            games_synced=5,
            last_game_id="game-001",
        )
        assert 59 <= checkpoint.seconds_since_heartbeat <= 61

    def test_seconds_since_heartbeat_never_heartbeated(self):
        """Test seconds_since_heartbeat when never heartbeated."""
        from app.coordination.ephemeral_data_guard import HostCheckpoint

        checkpoint = HostCheckpoint(
            host="vast-12345",
            is_ephemeral=True,
            last_checkpoint_time=0,
            last_heartbeat_time=0,  # Never heartbeated
            games_generated=0,
            games_synced=0,
            last_game_id="",
        )
        assert checkpoint.seconds_since_heartbeat == float('inf')

    def test_needs_evacuation_non_ephemeral(self):
        """Test non-ephemeral hosts never need evacuation."""
        from app.coordination.ephemeral_data_guard import HostCheckpoint

        checkpoint = HostCheckpoint(
            host="runpod-h100",
            is_ephemeral=False,
            last_checkpoint_time=time.time(),
            last_heartbeat_time=time.time() - 9999,  # Very stale
            games_generated=1000,
            games_synced=0,  # Many unsynced
            last_game_id="game-001",
        )
        assert checkpoint.needs_evacuation is False

    def test_needs_evacuation_fresh_heartbeat(self):
        """Test ephemeral hosts with fresh heartbeat don't need evacuation."""
        from app.coordination.ephemeral_data_guard import HostCheckpoint

        checkpoint = HostCheckpoint(
            host="vast-12345",
            is_ephemeral=True,
            last_checkpoint_time=time.time(),
            last_heartbeat_time=time.time(),  # Fresh
            games_generated=1000,
            games_synced=0,  # Many unsynced
            last_game_id="game-001",
        )
        assert checkpoint.needs_evacuation is False

    def test_to_dict(self):
        """Test to_dict serialization."""
        from app.coordination.ephemeral_data_guard import HostCheckpoint

        now = time.time()
        checkpoint = HostCheckpoint(
            host="vast-12345",
            is_ephemeral=True,
            last_checkpoint_time=now,
            last_heartbeat_time=now,
            games_generated=100,
            games_synced=50,
            last_game_id="game-001",
        )
        d = checkpoint.to_dict()

        assert d["host"] == "vast-12345"
        assert d["is_ephemeral"] is True
        assert d["games_generated"] == 100
        assert d["games_synced"] == 50
        assert d["unsynced_games"] == 50
        assert "last_checkpoint" in d
        assert "last_heartbeat" in d


class TestWriteThrough:
    """Tests for WriteThrough dataclass."""

    def test_write_through_creation(self):
        """Test WriteThrough creation with defaults."""
        from app.coordination.ephemeral_data_guard import WriteThrough

        wt = WriteThrough(
            game_id="game-001",
            host="vast-12345",
            priority=5,
            created_at=time.time(),
            data_path="/data/games/game-001.db",
        )
        assert wt.game_id == "game-001"
        assert wt.host == "vast-12345"
        assert wt.priority == 5
        assert wt.synced is False
        assert wt.synced_at == 0.0


class TestEphemeralDataGuard:
    """Tests for EphemeralDataGuard class."""

    @pytest.fixture
    def temp_db(self, tmp_path: Path):
        """Create a temporary database path."""
        return tmp_path / "ephemeral_guard.db"

    @pytest.fixture
    def guard(self, temp_db: Path):
        """Create a fresh EphemeralDataGuard instance."""
        from app.coordination.ephemeral_data_guard import EphemeralDataGuard

        # Reset singleton to get fresh instance with temp db
        EphemeralDataGuard.reset_instance()
        guard = EphemeralDataGuard(db_path=temp_db)
        yield guard
        EphemeralDataGuard.reset_instance()

    def test_singleton_pattern(self, temp_db: Path):
        """Test singleton pattern works correctly."""
        from app.coordination.ephemeral_data_guard import EphemeralDataGuard

        EphemeralDataGuard.reset_instance()

        guard1 = EphemeralDataGuard.get_instance(temp_db)
        guard2 = EphemeralDataGuard.get_instance()

        assert guard1 is guard2

        EphemeralDataGuard.reset_instance()

    def test_reset_instance(self, temp_db: Path):
        """Test reset_instance clears the singleton."""
        from app.coordination.ephemeral_data_guard import EphemeralDataGuard

        EphemeralDataGuard.reset_instance()

        guard1 = EphemeralDataGuard.get_instance(temp_db)
        EphemeralDataGuard.reset_instance()
        guard2 = EphemeralDataGuard.get_instance(temp_db)

        assert guard1 is not guard2

        EphemeralDataGuard.reset_instance()

    def test_database_initialization(self, guard):
        """Test database is properly initialized with tables."""
        conn = guard._get_connection()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}

        assert "host_checkpoints" in tables
        assert "write_through_queue" in tables
        assert "evacuation_history" in tables

    def test_checkpoint_creates_new_host(self, guard):
        """Test checkpoint creates a new host entry."""
        checkpoint = guard.checkpoint(
            host="vast-12345",
            games_generated=100,
            games_synced=50,
            last_game_id="game-001",
        )

        assert checkpoint.host == "vast-12345"
        assert checkpoint.games_generated == 100
        assert checkpoint.games_synced == 50
        assert checkpoint.last_game_id == "game-001"
        assert checkpoint.is_ephemeral is True  # "vast" pattern

    def test_checkpoint_updates_existing(self, guard):
        """Test checkpoint updates existing host entry."""
        guard.checkpoint(host="vast-12345", games_generated=50)
        checkpoint = guard.checkpoint(host="vast-12345", games_generated=100)

        assert checkpoint.games_generated == 100

    def test_checkpoint_with_extra_data(self, guard):
        """Test checkpoint stores extra data."""
        checkpoint = guard.checkpoint(
            host="vast-12345",
            games_generated=10,
            extra_data={"model_version": "v2.0", "board_type": "hex8"},
        )

        assert checkpoint.checkpoint_data["model_version"] == "v2.0"
        assert checkpoint.checkpoint_data["board_type"] == "hex8"

    def test_checkpoint_uses_hostname_when_not_provided(self, guard):
        """Test checkpoint uses socket.gethostname() when host not provided."""
        import socket

        checkpoint = guard.checkpoint(games_generated=10)
        assert checkpoint.host == socket.gethostname()

    def test_heartbeat_creates_checkpoint(self, guard):
        """Test heartbeat creates checkpoint for new host."""
        guard.heartbeat(host="vast-12345")

        checkpoint = guard.get_checkpoint("vast-12345")
        assert checkpoint is not None
        assert checkpoint.host == "vast-12345"

    def test_heartbeat_updates_time(self, guard):
        """Test heartbeat updates last_heartbeat_time."""
        guard.checkpoint(host="vast-12345", games_generated=10)
        old_time = guard.get_checkpoint("vast-12345").last_heartbeat_time

        time.sleep(0.1)
        guard.heartbeat(host="vast-12345")

        new_time = guard.get_checkpoint("vast-12345").last_heartbeat_time
        assert new_time > old_time

    def test_record_sync_complete(self, guard):
        """Test recording sync completion updates games_synced."""
        guard.checkpoint(host="vast-12345", games_generated=100, games_synced=0)
        guard.record_sync_complete("vast-12345", games_synced=50)

        checkpoint = guard.get_checkpoint("vast-12345")
        assert checkpoint.games_synced == 50

    def test_is_ephemeral_host(self, guard):
        """Test ephemeral host detection patterns."""
        assert guard._is_ephemeral_host("vast-12345") is True
        assert guard._is_ephemeral_host("spot-instance-1") is True
        assert guard._is_ephemeral_host("preemptible-vm") is True
        assert guard._is_ephemeral_host("ephemeral-worker") is True
        assert guard._is_ephemeral_host("runpod-h100") is False
        assert guard._is_ephemeral_host("nebius-backbone-1") is False

    def test_queue_write_through(self, guard):
        """Test queueing a write-through request."""
        guard.queue_write_through(
            game_id="game-001",
            host="vast-12345",
            data_path="/data/games/game-001.db",
            priority=5,
        )

        pending = guard.get_pending_write_throughs()
        assert len(pending) == 1
        assert pending[0].game_id == "game-001"
        assert pending[0].priority == 5

    def test_pending_write_throughs_ordered_by_priority(self, guard):
        """Test pending write-throughs are ordered by priority."""
        guard.queue_write_through("game-001", "vast-1", "/data/1", priority=1)
        guard.queue_write_through("game-002", "vast-1", "/data/2", priority=5)
        guard.queue_write_through("game-003", "vast-1", "/data/3", priority=3)

        pending = guard.get_pending_write_throughs()
        assert len(pending) == 3
        assert pending[0].game_id == "game-002"  # Priority 5
        assert pending[1].game_id == "game-003"  # Priority 3
        assert pending[2].game_id == "game-001"  # Priority 1

    def test_mark_write_through_complete(self, guard):
        """Test marking write-through as complete."""
        guard.queue_write_through("game-001", "vast-1", "/data/1", priority=1)

        guard.mark_write_through_complete("game-001")

        pending = guard.get_pending_write_throughs()
        assert len(pending) == 0

    def test_get_evacuation_candidates_empty(self, guard):
        """Test no evacuation candidates when all healthy."""
        guard.checkpoint(host="vast-12345", games_generated=100, games_synced=100)
        candidates = guard.get_evacuation_candidates()
        assert len(candidates) == 0

    def test_request_evacuation(self, guard):
        """Test requesting evacuation for a host."""
        guard.checkpoint(host="vast-12345", games_generated=100, games_synced=50)

        evacuation_id = guard.request_evacuation("vast-12345")
        assert evacuation_id > 0

    def test_complete_evacuation(self, guard):
        """Test completing an evacuation."""
        guard.checkpoint(host="vast-12345", games_generated=100, games_synced=50)
        evacuation_id = guard.request_evacuation("vast-12345")

        guard.complete_evacuation(evacuation_id, games_recovered=45, success=True)

        # Verify in database
        conn = guard._get_connection()
        cursor = conn.execute(
            "SELECT * FROM evacuation_history WHERE evacuation_id = ?",
            (evacuation_id,)
        )
        row = cursor.fetchone()
        assert row["games_recovered"] == 45
        assert row["success"] == 1

    def test_get_status(self, guard):
        """Test get_status returns comprehensive status."""
        guard.checkpoint(host="vast-12345", games_generated=100, games_synced=50)
        guard.checkpoint(host="runpod-h100", games_generated=50, games_synced=50)

        status = guard.get_status()

        assert status["ephemeral_hosts"] == 1  # Only vast
        assert status["total_unsynced_games"] == 50
        assert "hosts" in status
        assert "vast-12345" in status["hosts"]
        assert "runpod-h100" in status["hosts"]

    def test_get_checkpoint(self, guard):
        """Test get_checkpoint returns correct checkpoint."""
        guard.checkpoint(host="vast-12345", games_generated=100)

        checkpoint = guard.get_checkpoint("vast-12345")
        assert checkpoint is not None
        assert checkpoint.games_generated == 100

    def test_get_checkpoint_nonexistent(self, guard):
        """Test get_checkpoint returns None for unknown host."""
        checkpoint = guard.get_checkpoint("unknown-host")
        assert checkpoint is None

    def test_persistence_across_restarts(self, temp_db: Path):
        """Test checkpoints persist across guard restarts."""
        from app.coordination.ephemeral_data_guard import EphemeralDataGuard

        EphemeralDataGuard.reset_instance()

        # Create guard and add checkpoint
        guard1 = EphemeralDataGuard(db_path=temp_db)
        guard1.checkpoint(host="vast-12345", games_generated=100, games_synced=50)

        # Simulate restart
        EphemeralDataGuard.reset_instance()
        guard2 = EphemeralDataGuard(db_path=temp_db)

        checkpoint = guard2.get_checkpoint("vast-12345")
        assert checkpoint is not None
        assert checkpoint.games_generated == 100
        assert checkpoint.games_synced == 50

        EphemeralDataGuard.reset_instance()


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self, tmp_path: Path):
        """Reset singleton before and after each test."""
        from app.coordination.ephemeral_data_guard import EphemeralDataGuard

        EphemeralDataGuard.reset_instance()
        # Set up with temp db
        EphemeralDataGuard.get_instance(tmp_path / "guard.db")
        yield
        EphemeralDataGuard.reset_instance()

    def test_get_ephemeral_guard(self):
        """Test get_ephemeral_guard returns singleton."""
        from app.coordination.ephemeral_data_guard import (
            get_ephemeral_guard,
            EphemeralDataGuard,
        )

        guard = get_ephemeral_guard()
        assert isinstance(guard, EphemeralDataGuard)

    def test_checkpoint_games(self):
        """Test checkpoint_games convenience function."""
        from app.coordination.ephemeral_data_guard import checkpoint_games

        checkpoint = checkpoint_games(
            host="vast-12345",
            games_generated=100,
            games_synced=50,
            last_game_id="game-001",
        )

        assert checkpoint.host == "vast-12345"
        assert checkpoint.games_generated == 100

    def test_ephemeral_heartbeat(self):
        """Test ephemeral_heartbeat convenience function."""
        from app.coordination.ephemeral_data_guard import (
            ephemeral_heartbeat,
            get_ephemeral_guard,
        )

        ephemeral_heartbeat(host="vast-12345")

        checkpoint = get_ephemeral_guard().get_checkpoint("vast-12345")
        assert checkpoint is not None

    def test_is_host_ephemeral(self):
        """Test is_host_ephemeral convenience function."""
        from app.coordination.ephemeral_data_guard import is_host_ephemeral

        assert is_host_ephemeral("vast-12345") is True
        assert is_host_ephemeral("runpod-h100") is False

    def test_get_evacuation_candidates(self):
        """Test get_evacuation_candidates convenience function."""
        from app.coordination.ephemeral_data_guard import (
            get_evacuation_candidates,
            checkpoint_games,
        )

        checkpoint_games(host="vast-12345", games_generated=100, games_synced=100)
        candidates = get_evacuation_candidates()
        assert isinstance(candidates, list)

    def test_request_evacuation(self):
        """Test request_evacuation convenience function."""
        from app.coordination.ephemeral_data_guard import (
            request_evacuation,
            checkpoint_games,
        )

        checkpoint_games(host="vast-12345", games_generated=100, games_synced=50)
        evacuation_id = request_evacuation("vast-12345")
        assert evacuation_id > 0


class TestEmergencySyncEvents:
    """Tests for emergency sync event emission."""

    def test_record_emergency_sync_uses_publish_sync(self, tmp_path: Path):
        """Should emit completion through the unified router helper."""
        from app.coordination.ephemeral_data_guard import EphemeralDataGuard

        EphemeralDataGuard.reset_instance()
        guard = EphemeralDataGuard(db_path=tmp_path / "ephemeral_guard.db")

        with patch("app.coordination.event_router.publish_sync") as mock_publish_sync:
            guard.record_emergency_sync(
                host="vast-12345",
                files_synced=3,
                files_failed=1,
                files_skipped=2,
            )

        mock_publish_sync.assert_called_once()
        args, kwargs = mock_publish_sync.call_args
        assert args[0] == "EMERGENCY_SYNC_COMPLETED"
        assert args[1]["host"] == "vast-12345"
        assert args[1]["files_synced"] == 3
        assert args[1]["files_failed"] == 1
        assert args[1]["files_skipped"] == 2
        assert args[1]["success"] is False
        assert kwargs["source"] == "ephemeral_data_guard"

        EphemeralDataGuard.reset_instance()
