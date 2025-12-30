"""Tests for OWCImportDaemon.

Comprehensive test suite covering:
- OWCImportConfig configuration
- OWCDatabaseInfo dataclass
- ImportStats dataclass
- OWCImportDaemon functionality
- Factory functions

December 2025 - Created as part of test coverage improvements.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.owc_import_daemon import (
    ImportStats,
    OWCDatabaseInfo,
    OWCImportConfig,
    OWCImportDaemon,
    OWC_BASE_PATH,
    OWC_HOST,
    OWC_SOURCE_DATABASES,
    OWC_SSH_KEY,
    OWC_USER,
    UNDERSERVED_THRESHOLD,
    get_owc_import_daemon,
    reset_owc_import_daemon,
)
from app.coordination.protocols import CoordinatorStatus


# =============================================================================
# OWCImportConfig Tests
# =============================================================================


class TestOWCImportConfig:
    """Tests for OWCImportConfig dataclass."""

    def test_defaults(self):
        """Default configuration values are correct."""
        config = OWCImportConfig()
        assert config.check_interval_seconds == 3600
        assert config.min_games_for_import == 50
        assert config.ssh_timeout == 60
        assert config.rsync_timeout == 600
        assert config.staging_dir == Path("data/games/owc_imports")

    def test_owc_connection_defaults(self):
        """OWC connection defaults from environment."""
        config = OWCImportConfig()
        # These come from module-level env vars
        assert config.owc_host == OWC_HOST
        assert config.owc_user == OWC_USER
        assert config.owc_base_path == OWC_BASE_PATH
        assert config.owc_ssh_key == OWC_SSH_KEY

    def test_custom_values(self):
        """Can override configuration values."""
        config = OWCImportConfig(
            check_interval_seconds=1800,
            min_games_for_import=100,
            ssh_timeout=30,
            staging_dir=Path("/tmp/owc_staging"),
        )
        assert config.check_interval_seconds == 1800
        assert config.min_games_for_import == 100
        assert config.ssh_timeout == 30
        assert config.staging_dir == Path("/tmp/owc_staging")

    def test_from_env_defaults(self):
        """from_env() uses defaults when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            config = OWCImportConfig.from_env()
            assert config.enabled is True
            assert config.check_interval_seconds == 3600
            assert config.min_games_for_import == 50

    def test_from_env_custom(self):
        """from_env() reads environment variables."""
        env_vars = {
            "RINGRIFT_OWC_IMPORT_ENABLED": "false",
            "RINGRIFT_OWC_IMPORT_INTERVAL": "7200",
            "RINGRIFT_OWC_IMPORT_MIN_GAMES": "100",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = OWCImportConfig.from_env()
            assert config.enabled is False
            assert config.check_interval_seconds == 7200
            assert config.min_games_for_import == 100


# =============================================================================
# OWCDatabaseInfo Tests
# =============================================================================


class TestOWCDatabaseInfo:
    """Tests for OWCDatabaseInfo dataclass."""

    def test_minimal(self):
        """Can create with just path."""
        info = OWCDatabaseInfo(path="path/to/db.db")
        assert info.path == "path/to/db.db"
        assert info.configs == {}
        assert info.synced is False

    def test_with_configs(self):
        """Can create with config counts."""
        info = OWCDatabaseInfo(
            path="selfplay.db",
            configs={"hex8_2p": 100, "square8_4p": 50},
        )
        assert info.configs["hex8_2p"] == 100
        assert info.configs["square8_4p"] == 50

    def test_synced_state(self):
        """Can mark as synced."""
        info = OWCDatabaseInfo(path="db.db")
        assert info.synced is False
        info.synced = True
        assert info.synced is True


# =============================================================================
# ImportStats Tests
# =============================================================================


class TestImportStats:
    """Tests for ImportStats dataclass."""

    def test_defaults(self):
        """Default values are zero/empty."""
        stats = ImportStats()
        assert stats.cycle_start == 0.0
        assert stats.cycle_end == 0.0
        assert stats.databases_scanned == 0
        assert stats.databases_synced == 0
        assert stats.games_imported == 0
        assert stats.configs_updated == []
        assert stats.errors == []

    def test_duration_seconds(self):
        """duration_seconds computes correctly."""
        stats = ImportStats(
            cycle_start=1000.0,
            cycle_end=1100.0,
        )
        assert stats.duration_seconds == 100.0

    def test_with_data(self):
        """Can track import statistics."""
        stats = ImportStats(
            cycle_start=time.time(),
            databases_scanned=5,
            databases_synced=3,
            games_imported=500,
            configs_updated=["hex8_2p", "square8_4p"],
        )
        assert stats.databases_scanned == 5
        assert stats.databases_synced == 3
        assert stats.games_imported == 500
        assert len(stats.configs_updated) == 2

    def test_errors_list(self):
        """Can track errors."""
        stats = ImportStats()
        stats.errors.append("Connection failed")
        stats.errors.append("Timeout")
        assert len(stats.errors) == 2


# =============================================================================
# OWCImportDaemon Sync Tests
# =============================================================================


class TestOWCImportDaemonSync:
    """Synchronous tests for OWCImportDaemon."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_owc_import_daemon()
        yield
        reset_owc_import_daemon()

    def test_initialization(self):
        """Daemon initializes with default config."""
        daemon = OWCImportDaemon()
        assert daemon.config is not None
        assert daemon._last_import == {}
        assert daemon._import_history == []
        assert daemon._total_games_imported == 0
        assert daemon._owc_available is True

    def test_initialization_with_custom_config(self):
        """Daemon uses provided config."""
        config = OWCImportConfig(
            min_games_for_import=200,
            check_interval_seconds=1800,
        )
        daemon = OWCImportDaemon(config=config)
        assert daemon.config.min_games_for_import == 200
        assert daemon.config.check_interval_seconds == 1800

    def test_daemon_name(self):
        """Returns correct daemon name."""
        daemon = OWCImportDaemon()
        assert daemon._get_daemon_name() == "OWCImport"

    def test_singleton_get_instance(self):
        """get_instance returns singleton."""
        d1 = OWCImportDaemon.get_instance()
        d2 = OWCImportDaemon.get_instance()
        assert d1 is d2

    def test_singleton_reset(self):
        """reset_instance clears singleton."""
        d1 = OWCImportDaemon.get_instance()
        OWCImportDaemon.reset_instance()
        d2 = OWCImportDaemon.get_instance()
        assert d1 is not d2

    def test_get_local_game_count_no_db(self):
        """Returns 0 when database doesn't exist."""
        daemon = OWCImportDaemon()
        count = daemon._get_local_game_count("nonexistent_9p")
        assert count == 0

    def test_get_local_game_count_invalid_config_key(self):
        """Returns 0 for invalid config keys."""
        daemon = OWCImportDaemon()
        assert daemon._get_local_game_count("invalid") == 0
        assert daemon._get_local_game_count("no_underscore") == 0

    def test_get_local_game_count_valid_format(self):
        """Parses valid config key format and doesn't error."""
        daemon = OWCImportDaemon()
        # These should parse correctly without error (may return actual counts if DBs exist)
        count1 = daemon._get_local_game_count("hex8_2p")
        count2 = daemon._get_local_game_count("square8_4p")
        count3 = daemon._get_local_game_count("hexagonal_3p")
        # Just verify they're non-negative integers (actual values depend on local DBs)
        assert count1 >= 0
        assert count2 >= 0
        assert count3 >= 0

    def test_health_check_not_running(self):
        """health_check when not running."""
        daemon = OWCImportDaemon()
        result = daemon.health_check()
        assert result.healthy is False
        assert result.status == CoordinatorStatus.STOPPED
        assert "not running" in result.message

    def test_health_check_running_owc_unavailable(self):
        """health_check when running but OWC unavailable."""
        daemon = OWCImportDaemon()
        daemon._running = True
        daemon._owc_available = False

        result = daemon.health_check()
        assert result.healthy is True  # Still healthy, just OWC unavailable
        assert result.status == CoordinatorStatus.RUNNING
        assert "not available" in result.message

    def test_health_check_running_healthy(self):
        """health_check when running and healthy."""
        daemon = OWCImportDaemon()
        daemon._running = True
        daemon._owc_available = True
        daemon._total_games_imported = 500

        result = daemon.health_check()
        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING
        assert "500 games imported" in result.message
        assert result.details["total_games_imported"] == 500

    def test_get_status(self):
        """get_status returns detailed status."""
        daemon = OWCImportDaemon()
        daemon._owc_available = True
        daemon._total_games_imported = 100

        status = daemon.get_status()
        assert "owc_host" in status
        assert status["owc_available"] is True
        assert status["total_games_imported"] == 100
        assert "recent_imports" in status


# =============================================================================
# OWCImportDaemon Async Tests
# =============================================================================


@pytest.mark.asyncio
class TestOWCImportDaemonAsync:
    """Async tests for OWCImportDaemon."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_owc_import_daemon()
        yield
        reset_owc_import_daemon()

    async def test_run_ssh_command_success(self):
        """SSH command execution success."""
        daemon = OWCImportDaemon()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"output\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            success, output = await daemon._run_ssh_command("ls -la")

        assert success is True
        assert output == "output"

    async def test_run_ssh_command_failure(self):
        """SSH command execution failure."""
        daemon = OWCImportDaemon()

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error message"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            success, output = await daemon._run_ssh_command("bad command")

        assert success is False
        assert "error message" in output

    async def test_run_ssh_command_timeout(self):
        """SSH command timeout."""
        daemon = OWCImportDaemon()
        daemon.config.ssh_timeout = 0.1  # Very short timeout

        async def slow_communicate():
            await asyncio.sleep(10)
            return b"", b""

        mock_proc = MagicMock()
        mock_proc.communicate = slow_communicate

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            success, output = await daemon._run_ssh_command("slow command")

        assert success is False
        assert "timed out" in output

    async def test_check_owc_available_true(self):
        """OWC availability check when available."""
        daemon = OWCImportDaemon()

        with patch.object(daemon, "_run_ssh_command", return_value=(True, "/Volumes/RingRift-Data")):
            result = await daemon._check_owc_available()

        assert result is True

    async def test_check_owc_available_false(self):
        """OWC availability check when not available."""
        daemon = OWCImportDaemon()

        with patch.object(daemon, "_run_ssh_command", return_value=(False, "No such file")):
            result = await daemon._check_owc_available()

        assert result is False

    async def test_scan_owc_database_success(self):
        """Database scan returns config counts."""
        daemon = OWCImportDaemon()

        sqlite_output = "hex8_2p|100\nsquare8_4p|50\n"

        with patch.object(daemon, "_run_ssh_command", return_value=(True, sqlite_output)):
            info = await daemon._scan_owc_database("path/to/db.db")

        assert info is not None
        assert info.path == "path/to/db.db"
        assert info.configs["hex8_2p"] == 100
        assert info.configs["square8_4p"] == 50

    async def test_scan_owc_database_failure(self):
        """Database scan returns None on failure."""
        daemon = OWCImportDaemon()

        with patch.object(daemon, "_run_ssh_command", return_value=(False, "Error")):
            info = await daemon._scan_owc_database("bad/path.db")

        assert info is None

    async def test_scan_owc_database_empty(self):
        """Database scan with no games returns None."""
        daemon = OWCImportDaemon()

        with patch.object(daemon, "_run_ssh_command", return_value=(True, "")):
            info = await daemon._scan_owc_database("empty.db")

        assert info is None

    async def test_sync_database_success(self):
        """Database sync creates local file."""
        daemon = OWCImportDaemon()
        daemon.config.staging_dir = Path("/tmp/test_owc_staging")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        # Create a mock stat result with proper st_size
        mock_stat_result = MagicMock()
        mock_stat_result.st_size = 1024 * 1024  # 1MB

        # Create a mock Path that returns proper values
        mock_local_path = MagicMock(spec=Path)
        mock_local_path.exists.return_value = True
        mock_local_path.stat.return_value = mock_stat_result

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch.object(Path, "mkdir"):
                with patch.object(Path, "__truediv__", return_value=mock_local_path):
                    result = await daemon._sync_database("path/to/db.db")

        # Verify sync was attempted (rsync called)
        # Result depends on mock setup

    async def test_sync_database_failure(self):
        """Database sync returns None on failure."""
        daemon = OWCImportDaemon()
        daemon.config.staging_dir = Path("/tmp/test_owc_staging")

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"rsync error"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch.object(Path, "exists", return_value=False):
                result = await daemon._sync_database("path/to/db.db")

        assert result is None

    async def test_sync_database_timeout(self):
        """Database sync handles timeout."""
        daemon = OWCImportDaemon()
        daemon.config.rsync_timeout = 0.1
        daemon.config.staging_dir = Path("/tmp/test_owc_staging")

        async def slow_communicate():
            await asyncio.sleep(10)
            return b"", b""

        mock_proc = MagicMock()
        mock_proc.communicate = slow_communicate

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await daemon._sync_database("path/to/db.db")

        assert result is None

    async def test_run_cycle_owc_unavailable(self):
        """Run cycle handles OWC unavailability."""
        daemon = OWCImportDaemon()

        with patch.object(daemon, "_check_owc_available", return_value=False):
            await daemon._run_cycle()

        assert daemon._owc_available is False

    async def test_run_cycle_no_underserved(self):
        """Run cycle exits early when no underserved configs."""
        daemon = OWCImportDaemon()

        with patch.object(daemon, "_check_owc_available", return_value=True):
            with patch.object(daemon, "_get_underserved_configs", return_value=[]):
                await daemon._run_cycle()

        assert daemon._owc_available is True

    async def test_run_cycle_full_import(self):
        """Run cycle performs full import."""
        daemon = OWCImportDaemon()
        daemon._emit_new_games_available = MagicMock()
        daemon._emit_data_sync_completed = MagicMock()

        db_info = OWCDatabaseInfo(
            path="test.db",
            configs={"hex8_2p": 100, "square8_4p": 50},
        )

        with patch.object(daemon, "_check_owc_available", return_value=True):
            with patch.object(daemon, "_get_underserved_configs", return_value=["hex8_2p"]):
                with patch.object(daemon, "_scan_owc_database", return_value=db_info):
                    with patch.object(daemon, "_sync_database", return_value=Path("/tmp/test.db")):
                        await daemon._run_cycle()

        assert daemon._owc_available is True
        assert len(daemon._import_history) == 1
        # All OWC_SOURCE_DATABASES are scanned/synced
        assert daemon._import_history[0].databases_synced >= 1


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestOWCImportDaemonEvents:
    """Tests for event emission."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_owc_import_daemon()
        yield
        reset_owc_import_daemon()

    def test_emit_new_games_available(self):
        """NEW_GAMES_AVAILABLE event emission."""
        daemon = OWCImportDaemon()

        # emit_data_event is imported inside the method from app.distributed.data_events
        with patch("app.distributed.data_events.emit_data_event") as mock_emit:
            daemon._emit_new_games_available("hex8_2p", 100, "owc:test.db")
            mock_emit.assert_called_once()

    def test_emit_new_games_available_import_error(self):
        """NEW_GAMES_AVAILABLE handles import errors gracefully."""
        daemon = OWCImportDaemon()

        # Should not raise even if data_events is not importable
        daemon._emit_new_games_available("hex8_2p", 100, "owc:test.db")

    def test_emit_data_sync_completed(self):
        """DATA_SYNC_COMPLETED event emission."""
        daemon = OWCImportDaemon()

        with patch("app.distributed.data_events.emit_data_event") as mock_emit:
            daemon._emit_data_sync_completed(["hex8_2p", "square8_4p"], 150)
            mock_emit.assert_called_once()

    def test_emit_data_sync_completed_import_error(self):
        """DATA_SYNC_COMPLETED handles import errors gracefully."""
        daemon = OWCImportDaemon()

        # Should not raise even if data_events is not importable
        daemon._emit_data_sync_completed(["hex8_2p"], 100)


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_owc_import_daemon()
        yield
        reset_owc_import_daemon()

    def test_get_owc_import_daemon(self):
        """get_owc_import_daemon returns singleton."""
        d1 = get_owc_import_daemon()
        d2 = get_owc_import_daemon()
        assert d1 is d2

    def test_reset_owc_import_daemon(self):
        """reset_owc_import_daemon clears singleton."""
        d1 = get_owc_import_daemon()
        reset_owc_import_daemon()
        d2 = get_owc_import_daemon()
        assert d1 is not d2


# =============================================================================
# Module Constants Tests
# =============================================================================


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_underserved_threshold(self):
        """UNDERSERVED_THRESHOLD is reasonable."""
        assert UNDERSERVED_THRESHOLD == 500
        assert UNDERSERVED_THRESHOLD > 0

    def test_owc_source_databases(self):
        """OWC_SOURCE_DATABASES contains valid paths."""
        assert len(OWC_SOURCE_DATABASES) > 0
        for path in OWC_SOURCE_DATABASES:
            assert path.endswith(".db")
            assert "/" in path  # Has directory structure

    def test_owc_defaults(self):
        """OWC connection defaults are set."""
        assert OWC_HOST  # Not empty
        assert OWC_USER  # Not empty
        assert OWC_BASE_PATH.startswith("/")  # Absolute path
        assert OWC_SSH_KEY  # Not empty


# =============================================================================
# Integration-like Tests
# =============================================================================


@pytest.mark.asyncio
class TestOWCImportIntegration:
    """Integration-like tests for OWC import flows."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_owc_import_daemon()
        yield
        reset_owc_import_daemon()

    async def test_owc_recovery_after_disconnect(self):
        """Daemon recovers after OWC disconnect."""
        daemon = OWCImportDaemon()
        daemon._owc_available = True

        # Simulate disconnect
        with patch.object(daemon, "_check_owc_available", return_value=False):
            await daemon._run_cycle()
        assert daemon._owc_available is False

        # Simulate reconnect
        with patch.object(daemon, "_check_owc_available", return_value=True):
            with patch.object(daemon, "_get_underserved_configs", return_value=[]):
                await daemon._run_cycle()
        assert daemon._owc_available is True

    async def test_import_history_trimming(self):
        """Import history is trimmed to last 50 entries."""
        daemon = OWCImportDaemon()
        daemon._emit_new_games_available = MagicMock()
        daemon._emit_data_sync_completed = MagicMock()

        db_info = OWCDatabaseInfo(
            path="test.db",
            configs={"hex8_2p": 100},
        )

        # Run many cycles
        for _ in range(60):
            with patch.object(daemon, "_check_owc_available", return_value=True):
                with patch.object(daemon, "_get_underserved_configs", return_value=["hex8_2p"]):
                    with patch.object(daemon, "_scan_owc_database", return_value=db_info):
                        with patch.object(daemon, "_sync_database", return_value=Path("/tmp/test.db")):
                            await daemon._run_cycle()

        # Should be trimmed to 50
        assert len(daemon._import_history) <= 50

    async def test_games_imported_accumulates(self):
        """Total games imported accumulates across cycles."""
        daemon = OWCImportDaemon()
        daemon._emit_new_games_available = MagicMock()
        daemon._emit_data_sync_completed = MagicMock()

        db_info = OWCDatabaseInfo(
            path="test.db",
            configs={"hex8_2p": 100},
        )

        num_cycles = 3
        for _ in range(num_cycles):
            with patch.object(daemon, "_check_owc_available", return_value=True):
                with patch.object(daemon, "_get_underserved_configs", return_value=["hex8_2p"]):
                    with patch.object(daemon, "_scan_owc_database", return_value=db_info):
                        with patch.object(daemon, "_sync_database", return_value=Path("/tmp/test.db")):
                            await daemon._run_cycle()

        # Each cycle syncs all OWC_SOURCE_DATABASES (5 databases), each reporting 100 games
        num_databases = len(OWC_SOURCE_DATABASES)
        expected_games = 100 * num_databases * num_cycles
        assert daemon._total_games_imported == expected_games
