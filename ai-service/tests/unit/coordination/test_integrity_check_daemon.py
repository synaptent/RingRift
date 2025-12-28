#!/usr/bin/env python3
"""Unit tests for IntegrityCheckDaemon (December 2025 - Phase 6).

Tests the periodic data integrity validation daemon:
- Configuration loading from environment
- Database scanning for orphan games
- Quarantine and cleanup functionality
- Health check reporting
- Event emission

Test fixtures create temporary SQLite databases for testing without
affecting production data.
"""

import asyncio
import os
import sqlite3
import tempfile
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from app.coordination.integrity_check_daemon import (
    IntegrityCheckConfig,
    IntegrityCheckDaemon,
    IntegrityCheckResult,
    OrphanGame,
    get_integrity_check_daemon,
    reset_integrity_check_daemon,
)
from app.coordination.daemon_types import DaemonType


class TestIntegrityCheckConfig:
    """Tests for IntegrityCheckConfig."""

    def test_default_config_values(self):
        """Test that default config has expected values."""
        config = IntegrityCheckConfig()

        assert config.check_interval_seconds == 3600  # 1 hour
        assert config.quarantine_after_days == 7
        assert config.max_orphans_per_scan == 1000
        assert config.emit_events is True

    def test_from_env_loads_data_dir(self, monkeypatch):
        """Test that from_env loads RINGRIFT_INTEGRITY_DATA_DIR."""
        monkeypatch.setenv("RINGRIFT_INTEGRITY_DATA_DIR", "/custom/path")

        config = IntegrityCheckConfig.from_env()

        assert config.data_dir == "/custom/path"

    def test_from_env_loads_quarantine_days(self, monkeypatch):
        """Test that from_env loads RINGRIFT_INTEGRITY_QUARANTINE_DAYS."""
        monkeypatch.setenv("RINGRIFT_INTEGRITY_QUARANTINE_DAYS", "14")

        config = IntegrityCheckConfig.from_env()

        assert config.quarantine_after_days == 14

    def test_from_env_loads_max_orphans(self, monkeypatch):
        """Test that from_env loads RINGRIFT_INTEGRITY_MAX_ORPHANS."""
        monkeypatch.setenv("RINGRIFT_INTEGRITY_MAX_ORPHANS", "500")

        config = IntegrityCheckConfig.from_env()

        assert config.max_orphans_per_scan == 500


class TestOrphanGame:
    """Tests for OrphanGame dataclass."""

    def test_orphan_game_creation(self):
        """Test that OrphanGame can be created with all fields."""
        orphan = OrphanGame(
            game_id="test-game-001",
            db_path="/path/to/db.db",
            board_type="hex8",
            num_players=2,
            total_moves=10,
            created_at="2025-12-28T00:00:00Z",
            game_status="completed",
        )

        assert orphan.game_id == "test-game-001"
        assert orphan.db_path == "/path/to/db.db"
        assert orphan.board_type == "hex8"
        assert orphan.num_players == 2
        assert orphan.total_moves == 10
        assert orphan.game_status == "completed"


class TestIntegrityCheckResult:
    """Tests for IntegrityCheckResult dataclass."""

    def test_default_result_values(self):
        """Test that default result has expected values."""
        result = IntegrityCheckResult()

        assert result.databases_scanned == 0
        assert result.orphan_games_found == 0
        assert result.orphan_games_quarantined == 0
        assert result.orphan_games_cleaned == 0
        assert result.errors == []
        assert result.details_by_db == {}


class TestIntegrityCheckDaemon:
    """Tests for IntegrityCheckDaemon."""

    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create a temporary data directory with test databases."""
        data_dir = tmp_path / "games"
        data_dir.mkdir()
        return data_dir

    @pytest.fixture
    def db_with_orphans(self, temp_data_dir):
        """Create a database with orphan games."""
        db_path = temp_data_dir / "test_orphans.db"
        conn = sqlite3.connect(str(db_path))

        # Create tables
        conn.execute(
            """
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT,
                num_players INTEGER,
                total_moves INTEGER,
                created_at TEXT,
                game_status TEXT
            )
        """
        )
        conn.execute(
            """
            CREATE TABLE game_moves (
                id INTEGER PRIMARY KEY,
                game_id TEXT,
                turn_number INTEGER,
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            )
        """
        )

        # Insert orphan games (no moves)
        for i in range(3):
            conn.execute(
                """
                INSERT INTO games (game_id, board_type, num_players, total_moves, created_at, game_status)
                VALUES (?, 'hex8', 2, 10, datetime('now'), 'completed')
            """,
                (f"orphan-{i}",),
            )

        # Insert valid game (has moves)
        conn.execute(
            """
            INSERT INTO games (game_id, board_type, num_players, total_moves, created_at, game_status)
            VALUES ('valid-game', 'hex8', 2, 5, datetime('now'), 'completed')
        """
        )
        conn.execute(
            "INSERT INTO game_moves (game_id, turn_number) VALUES ('valid-game', 1)"
        )

        conn.commit()
        conn.close()

        return db_path

    @pytest.fixture
    def daemon_config(self, temp_data_dir):
        """Create a config for testing."""
        return IntegrityCheckConfig(
            check_interval_seconds=1,
            data_dir=str(temp_data_dir),
            quarantine_after_days=1,
            max_orphans_per_scan=100,
            emit_events=False,
        )

    def test_daemon_name(self):
        """Test that daemon has correct name."""
        assert IntegrityCheckDaemon.DAEMON_NAME == "integrity_check"

    def test_daemon_initialization(self, daemon_config):
        """Test that daemon initializes with config."""
        daemon = IntegrityCheckDaemon(config=daemon_config)

        assert daemon.config == daemon_config
        assert daemon._last_result is None
        assert daemon._total_orphans_found == 0
        assert daemon._total_orphans_cleaned == 0

    def test_find_databases_discovers_db_files(self, temp_data_dir, daemon_config):
        """Test that _find_databases discovers .db files."""
        # Create test db files
        (temp_data_dir / "test1.db").touch()
        (temp_data_dir / "test2.db").touch()
        (temp_data_dir / "jsonl_test.db").touch()  # Should be excluded

        daemon = IntegrityCheckDaemon(config=daemon_config)
        databases = daemon._find_databases()

        db_names = [db.name for db in databases]
        assert "test1.db" in db_names
        assert "test2.db" in db_names
        assert "jsonl_test.db" not in db_names  # Filtered out

    @pytest.mark.asyncio
    async def test_check_database_finds_orphans(self, db_with_orphans, daemon_config):
        """Test that _check_database finds orphan games."""
        daemon = IntegrityCheckDaemon(config=daemon_config)
        orphans = await daemon._check_database(db_with_orphans)

        assert len(orphans) == 3
        orphan_ids = [o.game_id for o in orphans]
        assert "orphan-0" in orphan_ids
        assert "orphan-1" in orphan_ids
        assert "orphan-2" in orphan_ids
        assert "valid-game" not in orphan_ids  # Has moves, not orphan

    @pytest.mark.asyncio
    async def test_quarantine_orphans_creates_table(self, db_with_orphans, daemon_config):
        """Test that _quarantine_orphans creates orphaned_games table."""
        daemon = IntegrityCheckDaemon(config=daemon_config)
        orphans = await daemon._check_database(db_with_orphans)

        quarantined = await daemon._quarantine_orphans(db_with_orphans, orphans)

        assert quarantined == 3

        # Verify table was created with entries
        conn = sqlite3.connect(str(db_with_orphans))
        cursor = conn.execute("SELECT COUNT(*) FROM orphaned_games")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 3

    @pytest.mark.asyncio
    async def test_health_check_returns_result(self, daemon_config):
        """Test that health_check returns HealthCheckResult."""
        daemon = IntegrityCheckDaemon(config=daemon_config)

        result = daemon.health_check()

        assert result.status in ["healthy", "unhealthy"]
        assert "running" in result.details
        assert "data_dir" in result.details

    def test_singleton_access(self):
        """Test that get_integrity_check_daemon returns singleton."""
        reset_integrity_check_daemon()

        daemon1 = get_integrity_check_daemon()
        daemon2 = get_integrity_check_daemon()

        assert daemon1 is daemon2

        reset_integrity_check_daemon()


class TestDaemonRegistration:
    """Tests for daemon registration in registry."""

    def test_integrity_check_in_daemon_types(self):
        """Test that INTEGRITY_CHECK is in DaemonType enum."""
        assert hasattr(DaemonType, "INTEGRITY_CHECK")
        assert DaemonType.INTEGRITY_CHECK.value == "integrity_check"

    def test_daemon_spec_in_registry(self):
        """Test that IntegrityCheckDaemon is registered in daemon registry."""
        from app.coordination.daemon_registry import DAEMON_REGISTRY

        assert DaemonType.INTEGRITY_CHECK in DAEMON_REGISTRY

        spec = DAEMON_REGISTRY[DaemonType.INTEGRITY_CHECK]
        assert spec.runner_name == "create_integrity_check"
        assert spec.category == "recovery"

    def test_runner_function_exists(self):
        """Test that create_integrity_check runner exists."""
        from app.coordination.daemon_runners import get_runner

        runner = get_runner(DaemonType.INTEGRITY_CHECK)
        assert runner is not None
        assert asyncio.iscoroutinefunction(runner)
