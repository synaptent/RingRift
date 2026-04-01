"""Tests for TrainingFreshnessChecker - Pre-training data freshness validation.

Tests cover:
- Local data age checking (databases and NPZ files)
- Freshness threshold configuration
- Sync triggering for stale data
- Wait-for-sync behavior
- Game count validation
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.training_freshness import (
    DataSourceInfo,
    FreshnessConfig,
    FreshnessResult,
    TrainingFreshnessChecker,
    check_freshness_sync,
    ensure_fresh_data,
    get_freshness_checker,
    reset_freshness_checker,
    DEFAULT_MAX_AGE_HOURS,
)


@pytest.fixture
def data_dir(tmp_path):
    """Create temporary data directory structure."""
    games_dir = tmp_path / "games"
    training_dir = tmp_path / "training"
    games_dir.mkdir()
    training_dir.mkdir()
    return tmp_path


@pytest.fixture
def config():
    """Create test configuration."""
    return FreshnessConfig(
        max_age_hours=1.0,
        sync_timeout_seconds=10,
        wait_for_sync=True,
        trigger_sync=True,
        check_databases=True,
        check_npz_files=True,
        min_games_required=100,
    )


@pytest.fixture
def checker(config, data_dir):
    """Create TrainingFreshnessChecker for testing."""
    reset_freshness_checker()
    checker = TrainingFreshnessChecker(config=config, data_dir=data_dir)
    yield checker
    reset_freshness_checker()


def create_test_database(db_path: Path, num_games: int = 100):
    """Create a test database with games."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            board_type TEXT,
            num_players INTEGER
        )
    """)
    for i in range(num_games):
        conn.execute(
            "INSERT INTO games (game_id, board_type, num_players) VALUES (?, ?, ?)",
            (f"game-{i}", "hex8", 2)
        )
    conn.commit()
    conn.close()


class TestFreshnessConfig:
    """Tests for FreshnessConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FreshnessConfig()

        assert config.max_age_hours == DEFAULT_MAX_AGE_HOURS
        assert config.sync_timeout_seconds == 300
        assert config.wait_for_sync is True
        assert config.trigger_sync is True
        assert config.check_databases is True
        assert config.check_npz_files is True
        assert config.min_games_required == 50

    def test_custom_config(self):
        """Test custom configuration."""
        config = FreshnessConfig(
            max_age_hours=2.0,
            min_games_required=500,
        )

        assert config.max_age_hours == 2.0
        assert config.min_games_required == 500


class TestFreshnessResult:
    """Tests for FreshnessResult dataclass."""

    def test_result_fields(self):
        """Test result dataclass fields."""
        result = FreshnessResult(
            success=True,
            is_fresh=True,
            data_age_hours=0.5,
            games_available=1000,
            sync_triggered=False,
        )

        assert result.success is True
        assert result.is_fresh is True
        assert result.data_age_hours == 0.5
        assert result.games_available == 1000

    def test_result_with_error(self):
        """Test result with error."""
        result = FreshnessResult(
            success=False,
            is_fresh=False,
            data_age_hours=float("inf"),
            games_available=0,
            error="No data found",
        )

        assert result.success is False
        assert result.error == "No data found"


class TestDataSourceInfo:
    """Tests for DataSourceInfo dataclass."""

    def test_source_info_fields(self):
        """Test source info dataclass fields."""
        info = DataSourceInfo(
            path=Path("/data/games.db"),
            age_hours=0.5,
            size_bytes=1_000_000,
            game_count=500,
            is_stale=False,
            board_type="hex8",
            num_players=2,
        )

        assert info.path == Path("/data/games.db")
        assert info.age_hours == 0.5
        assert info.is_stale is False


class TestFindLocalDatabases:
    """Tests for finding local game databases."""

    def test_find_databases_empty_dir(self, checker, data_dir):
        """Test finding databases in empty directory."""
        sources = checker.find_local_databases()
        assert sources == []

    def test_find_databases_by_board_type(self, checker, data_dir):
        """Test finding databases filtered by board type."""
        # Create test databases
        hex8_db = data_dir / "games" / "canonical_hex8_2p.db"
        square8_db = data_dir / "games" / "canonical_square8_2p.db"
        create_test_database(hex8_db, 100)
        create_test_database(square8_db, 100)

        sources = checker.find_local_databases(board_type="hex8")

        # Should only find hex8 database
        paths = [s.path for s in sources]
        assert hex8_db in paths
        assert square8_db not in paths

    def test_find_databases_by_num_players(self, checker, data_dir):
        """Test finding databases filtered by player count."""
        db_2p = data_dir / "games" / "hex8_2p.db"
        db_4p = data_dir / "games" / "hex8_4p.db"
        create_test_database(db_2p, 100)
        create_test_database(db_4p, 100)

        sources = checker.find_local_databases(num_players=2)

        paths = [s.path for s in sources]
        assert db_2p in paths
        assert db_4p not in paths

    def test_find_databases_calculates_age(self, checker, data_dir):
        """Test database age calculation."""
        db_path = data_dir / "games" / "test.db"
        create_test_database(db_path, 100)

        # Modify file time to make it "old"
        old_time = time.time() - 7200  # 2 hours ago
        import os
        os.utime(db_path, (old_time, old_time))

        sources = checker.find_local_databases()

        assert len(sources) == 1
        assert sources[0].age_hours >= 1.9  # At least ~2 hours old

    def test_find_databases_marks_stale(self, checker, data_dir):
        """Test stale detection based on config threshold."""
        db_path = data_dir / "games" / "test.db"
        create_test_database(db_path, 100)

        # Make it old (2 hours)
        old_time = time.time() - 7200
        import os
        os.utime(db_path, (old_time, old_time))

        # Threshold is 1 hour
        checker.config.max_age_hours = 1.0

        sources = checker.find_local_databases()

        assert sources[0].is_stale is True


class TestFindLocalNPZFiles:
    """Tests for finding local NPZ files."""

    def test_find_npz_empty_dir(self, checker, data_dir):
        """Test finding NPZ files in empty directory."""
        with patch.object(checker, "_get_data_catalog", return_value=None):
            sources = checker.find_local_npz_files()
        assert sources == []

    def test_find_npz_by_board_type(self, checker, data_dir):
        """Test finding NPZ files filtered by board type."""
        # Create test NPZ files (just touch them)
        hex8_npz = data_dir / "training" / "hex8_2p.npz"
        square8_npz = data_dir / "training" / "square8_2p.npz"
        hex8_npz.write_bytes(b"\x00" * 1000)
        square8_npz.write_bytes(b"\x00" * 1000)

        with patch.object(checker, "_get_data_catalog", return_value=None):
            sources = checker.find_local_npz_files(board_type="hex8")

        paths = [s.path for s in sources]
        assert hex8_npz in paths
        assert square8_npz not in paths

    def test_find_npz_estimates_samples(self, checker, data_dir):
        """Test NPZ sample count estimation from file size."""
        npz_path = data_dir / "training" / "hex8_2p.npz"
        # ~200 bytes per sample, so 20000 bytes = ~100 samples
        npz_path.write_bytes(b"\x00" * 20000)

        with patch.object(checker, "_get_data_catalog", return_value=None):
            sources = checker.find_local_npz_files()

        assert len(sources) == 1
        assert sources[0].game_count >= 50  # Estimated samples


class TestCheckFreshness:
    """Tests for freshness checking."""

    def test_check_freshness_no_data(self, checker):
        """Test freshness check with no data."""
        result = checker.check_freshness("hex8", 2)

        assert result.success is False
        assert result.is_fresh is False
        assert result.error == "No training data found"

    def test_check_freshness_fresh_data(self, checker, data_dir):
        """Test freshness check with fresh data."""
        # Create fresh database
        db_path = data_dir / "games" / "canonical_hex8_2p.db"
        create_test_database(db_path, 200)  # > min_games_required (100)

        result = checker.check_freshness("hex8", 2)

        assert result.success is True
        assert result.is_fresh is True
        # May find the same DB via multiple glob patterns, so check >= 200
        assert result.games_available >= 200

    def test_check_freshness_stale_data(self, checker, data_dir):
        """Test freshness check with stale data."""
        db_path = data_dir / "games" / "canonical_hex8_2p.db"
        create_test_database(db_path, 200)

        # Make it old
        old_time = time.time() - 7200  # 2 hours
        import os
        os.utime(db_path, (old_time, old_time))

        result = checker.check_freshness("hex8", 2)

        assert result.success is True
        assert result.is_fresh is False
        assert result.data_age_hours >= 1.9

    def test_check_freshness_insufficient_games(self, checker, data_dir):
        """Test freshness check with insufficient games."""
        db_path = data_dir / "games" / "canonical_hex8_2p.db"
        create_test_database(db_path, 10)  # < min_games_required (100)

        result = checker.check_freshness("hex8", 2)

        assert result.success is True
        assert result.is_fresh is False  # Not enough games

    def test_check_freshness_includes_details(self, checker, data_dir):
        """Test freshness check includes detailed information."""
        db_path = data_dir / "games" / "canonical_hex8_2p.db"
        create_test_database(db_path, 100)

        npz_path = data_dir / "training" / "hex8_2p.npz"
        npz_path.write_bytes(b"\x00" * 1000)

        result = checker.check_freshness("hex8", 2)

        assert "databases" in result.details
        assert "npz_files" in result.details
        assert "config_key" in result.details


class TestTriggerSync:
    """Tests for sync triggering."""

    @pytest.mark.asyncio
    async def test_trigger_sync_no_daemon(self, checker):
        """Test sync trigger when daemon unavailable."""
        with patch.object(checker, "_get_sync_daemon", return_value=None):
            result = await checker.trigger_sync("hex8", 2)

            assert result is False

    @pytest.mark.asyncio
    async def test_trigger_sync_starts_daemon(self, checker):
        """Test sync trigger starts daemon if not running."""
        mock_daemon = MagicMock()
        mock_daemon._running = False
        mock_daemon.start = AsyncMock()
        mock_daemon.trigger_sync = AsyncMock()

        with patch(
            "app.coordination.training_data_sync_daemon.sync_best_fresh_data",
            side_effect=ImportError("fallback to AutoSyncDaemon"),
        ), patch.object(checker, "_get_sync_daemon", return_value=mock_daemon):
            result = await checker.trigger_sync("hex8", 2)

            mock_daemon.start.assert_called_once()
            mock_daemon.trigger_sync.assert_called_once()
            assert result is True

    @pytest.mark.asyncio
    async def test_trigger_sync_handles_error(self, checker):
        """Test sync trigger handles errors gracefully."""
        mock_daemon = MagicMock()
        mock_daemon._running = True
        mock_daemon.trigger_sync = AsyncMock(side_effect=Exception("Sync failed"))

        with patch.object(checker, "_get_sync_daemon", return_value=mock_daemon):
            result = await checker.trigger_sync("hex8", 2)

            assert result is False


class TestWaitForFreshData:
    """Tests for wait-for-fresh-data behavior."""

    @pytest.mark.asyncio
    async def test_wait_returns_immediately_if_fresh(self, checker, data_dir):
        """Test waiting returns immediately if data is fresh."""
        db_path = data_dir / "games" / "canonical_hex8_2p.db"
        create_test_database(db_path, 200)

        result = await checker.wait_for_fresh_data("hex8", 2)

        assert result.is_fresh is True
        assert result.sync_completed is True

    @pytest.mark.asyncio
    async def test_wait_times_out(self, checker, data_dir):
        """Test waiting times out with stale data."""
        db_path = data_dir / "games" / "canonical_hex8_2p.db"
        create_test_database(db_path, 200)

        # Make it old
        old_time = time.time() - 7200
        import os
        os.utime(db_path, (old_time, old_time))

        # Short timeout
        result = await checker.wait_for_fresh_data(
            "hex8", 2,
            timeout_seconds=1,
            poll_interval=0.5,
        )

        assert result.is_fresh is False
        assert "Timeout" in result.error


class TestEnsureFreshData:
    """Tests for the main ensure_fresh_data entry point."""

    @pytest.mark.asyncio
    async def test_ensure_fresh_data_already_fresh(self, checker, data_dir):
        """Test ensure_fresh_data returns immediately for fresh data."""
        db_path = data_dir / "games" / "canonical_hex8_2p.db"
        create_test_database(db_path, 200)

        result = await checker.ensure_fresh_data("hex8", 2)

        assert result.is_fresh is True
        assert result.sync_triggered is False

    @pytest.mark.asyncio
    async def test_ensure_fresh_data_triggers_sync(self, checker, data_dir):
        """Test ensure_fresh_data triggers sync for stale data."""
        db_path = data_dir / "games" / "canonical_hex8_2p.db"
        create_test_database(db_path, 200)

        # Make it old
        old_time = time.time() - 7200
        import os
        os.utime(db_path, (old_time, old_time))

        # Mock both trigger_sync and wait_for_fresh_data
        original_trigger = checker.trigger_sync
        original_wait = checker.wait_for_fresh_data

        trigger_called = []

        async def mock_trigger(board_type, num_players):
            trigger_called.append(True)
            return True

        async def mock_wait(board_type, num_players, timeout_seconds=None, poll_interval=5.0):
            return FreshnessResult(
                success=True, is_fresh=False, data_age_hours=2.0, games_available=200,
                sync_triggered=True
            )

        checker.trigger_sync = mock_trigger
        checker.wait_for_fresh_data = mock_wait

        result = await checker.ensure_fresh_data("hex8", 2)

        # Restore
        checker.trigger_sync = original_trigger
        checker.wait_for_fresh_data = original_wait

        assert len(trigger_called) == 1

    @pytest.mark.asyncio
    async def test_ensure_fresh_data_sync_disabled(self, checker, data_dir):
        """Test ensure_fresh_data doesn't sync when disabled."""
        checker.config.trigger_sync = False

        db_path = data_dir / "games" / "canonical_hex8_2p.db"
        create_test_database(db_path, 200)

        # Make it old
        old_time = time.time() - 7200
        import os
        os.utime(db_path, (old_time, old_time))

        result = await checker.ensure_fresh_data("hex8", 2)

        assert result.sync_triggered is False


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_check_freshness_sync(self, data_dir):
        """Test synchronous freshness check."""
        reset_freshness_checker()

        # Create test data
        games_dir = data_dir / "games"
        games_dir.mkdir(exist_ok=True)
        db_path = games_dir / "canonical_hex8_2p.db"
        create_test_database(db_path, 200)

        # Use checker directly with custom data dir
        config = FreshnessConfig(max_age_hours=10.0, trigger_sync=False, wait_for_sync=False)
        checker = TrainingFreshnessChecker(config=config, data_dir=data_dir)
        result = checker.check_freshness("hex8", 2)

        assert result.success is True
        # May find via multiple patterns
        assert result.games_available >= 200

    @pytest.mark.asyncio
    async def test_ensure_fresh_data_function(self, data_dir):
        """Test convenience function for ensuring fresh data."""
        reset_freshness_checker()

        games_dir = data_dir / "games"
        games_dir.mkdir(exist_ok=True)
        db_path = games_dir / "canonical_hex8_2p.db"
        create_test_database(db_path, 200)

        with patch(
            "app.coordination.training_freshness._DATA_DIR",
            data_dir
        ):
            result = await ensure_fresh_data(
                "hex8", 2,
                max_age_hours=10.0,
                trigger_sync=False,
            )

            assert result.success is True


class TestSingletonAccessor:
    """Tests for singleton accessor."""

    def test_get_freshness_checker(self):
        """Test singleton accessor returns same instance."""
        reset_freshness_checker()

        c1 = get_freshness_checker()
        c2 = get_freshness_checker()

        assert c1 is c2

        reset_freshness_checker()

    def test_get_freshness_checker_with_config(self):
        """Test singleton accessor accepts config."""
        reset_freshness_checker()

        config = FreshnessConfig(max_age_hours=5.0)
        checker = get_freshness_checker(config=config)

        assert checker.config.max_age_hours == 5.0

        reset_freshness_checker()


class TestGetStatus:
    """Tests for status reporting."""

    def test_get_status_contents(self, checker):
        """Test status report contains expected fields."""
        status = checker.get_status()

        assert "node_id" in status
        assert "config" in status
        assert "data_dir" in status
        assert "games_dir_exists" in status
        assert "training_dir_exists" in status

    def test_get_status_config_details(self, checker):
        """Test status includes config details."""
        status = checker.get_status()

        assert "max_age_hours" in status["config"]
        assert "sync_timeout_seconds" in status["config"]
        assert "min_games_required" in status["config"]


class TestEventEmission:
    """Tests for freshness event emission."""

    @pytest.mark.asyncio
    async def test_emit_freshness_event_uses_publish_helper(self, checker):
        """Freshness events should use the unified publish helper."""
        result = FreshnessResult(
            success=True,
            is_fresh=False,
            data_age_hours=2.5,
            games_available=321,
        )

        with patch(
            "app.coordination.event_router.publish",
            new_callable=AsyncMock,
        ) as mock_publish:
            await checker._emit_freshness_event("stale", "hex8", 2, result)

        mock_publish.assert_awaited_once()
        kwargs = mock_publish.await_args.kwargs
        assert kwargs["event_type"].value == "data_stale"
        assert kwargs["source"] == "TrainingFreshnessChecker"
        assert kwargs["payload"]["board_type"] == "hex8"
        assert kwargs["payload"]["num_players"] == 2
        assert kwargs["payload"]["games_available"] == 321
