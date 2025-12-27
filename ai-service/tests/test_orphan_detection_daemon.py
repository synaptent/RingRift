#!/usr/bin/env python3
"""Unit tests for OrphanDetectionDaemon (December 2025).

Tests the daemon that detects and handles orphaned game databases.
"""

import asyncio
import sqlite3
import tempfile
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

pytest.importorskip("app.coordination.orphan_detection_daemon")


class TestOrphanDetectionConfig:
    """Tests for OrphanDetectionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from app.coordination.orphan_detection_daemon import OrphanDetectionConfig

        config = OrphanDetectionConfig()

        assert config.scan_interval_seconds == 300.0
        assert config.games_dir == "data/games"
        assert config.auto_register_orphans is True
        assert config.min_games_to_register == 1
        assert config.cleanup_enabled is False
        assert config.min_age_before_cleanup_hours == 24.0
        assert config.max_orphan_count_before_alert == 100
        assert config.emit_detection_event is True

    def test_custom_values(self):
        """Test custom configuration values."""
        from app.coordination.orphan_detection_daemon import OrphanDetectionConfig

        config = OrphanDetectionConfig(
            scan_interval_seconds=600.0,
            games_dir="custom/games",
            auto_register_orphans=False,
            min_games_to_register=5,
            cleanup_enabled=True,
            max_orphan_count_before_alert=50,
        )

        assert config.scan_interval_seconds == 600.0
        assert config.games_dir == "custom/games"
        assert config.auto_register_orphans is False
        assert config.min_games_to_register == 5
        assert config.cleanup_enabled is True
        assert config.max_orphan_count_before_alert == 50


class TestOrphanInfo:
    """Tests for OrphanInfo dataclass."""

    def test_creation(self):
        """Test OrphanInfo creation."""
        from app.coordination.orphan_detection_daemon import OrphanInfo

        info = OrphanInfo(
            path=Path("/tmp/test.db"),
            game_count=100,
            file_size_bytes=1024000,
            modified_time=1234567890.0,
            board_type="hex8",
            num_players=2,
        )

        assert info.path == Path("/tmp/test.db")
        assert info.game_count == 100
        assert info.file_size_bytes == 1024000
        assert info.modified_time == 1234567890.0
        assert info.board_type == "hex8"
        assert info.num_players == 2

    def test_optional_fields(self):
        """Test OrphanInfo with optional fields as None."""
        from app.coordination.orphan_detection_daemon import OrphanInfo

        info = OrphanInfo(
            path=Path("/tmp/test.db"),
            game_count=50,
            file_size_bytes=512000,
            modified_time=1234567890.0,
        )

        assert info.board_type is None
        assert info.num_players is None


class TestOrphanDetectionDaemon:
    """Tests for OrphanDetectionDaemon class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        from app.coordination.orphan_detection_daemon import (
            OrphanDetectionDaemon,
            OrphanDetectionConfig,
        )

        daemon = OrphanDetectionDaemon()

        assert daemon.config is not None
        assert daemon._running is False
        assert daemon._last_scan_time == 0.0
        assert daemon._orphan_history == []

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        from app.coordination.orphan_detection_daemon import (
            OrphanDetectionDaemon,
            OrphanDetectionConfig,
        )

        config = OrphanDetectionConfig(scan_interval_seconds=120.0)
        daemon = OrphanDetectionDaemon(config=config)

        assert daemon.config.scan_interval_seconds == 120.0

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test daemon start and stop lifecycle."""
        from app.coordination.orphan_detection_daemon import OrphanDetectionDaemon

        daemon = OrphanDetectionDaemon()

        # Start in background
        task = asyncio.create_task(daemon.start())

        # Let it run briefly
        await asyncio.sleep(0.1)
        assert daemon._running is True

        # Stop it
        await daemon.stop()
        await asyncio.sleep(0.1)
        assert daemon._running is False

        # Clean up task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_analyze_database_valid(self):
        """Test analyzing a valid game database."""
        from app.coordination.orphan_detection_daemon import OrphanDetectionDaemon

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock game database
            db_path = Path(tmpdir) / "canonical_hex8_2p.db"
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute("CREATE TABLE games (game_id TEXT PRIMARY KEY)")
                for i in range(10):
                    conn.execute("INSERT INTO games VALUES (?)", (f"game_{i}",))
                conn.commit()

            daemon = OrphanDetectionDaemon()
            orphan_info = await daemon._analyze_database(db_path)

            assert orphan_info is not None
            assert orphan_info.path == db_path
            assert orphan_info.game_count == 10
            assert orphan_info.file_size_bytes > 0
            assert orphan_info.board_type == "hex8"
            assert orphan_info.num_players == 2

    @pytest.mark.asyncio
    async def test_analyze_database_square(self):
        """Test analyzing a square board database."""
        from app.coordination.orphan_detection_daemon import OrphanDetectionDaemon

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "square19_4p.db"
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute("CREATE TABLE games (game_id TEXT PRIMARY KEY)")
                conn.execute("INSERT INTO games VALUES ('game_1')")
                conn.commit()

            daemon = OrphanDetectionDaemon()
            orphan_info = await daemon._analyze_database(db_path)

            assert orphan_info is not None
            assert orphan_info.board_type == "square19"
            assert orphan_info.num_players == 4

    @pytest.mark.asyncio
    async def test_analyze_database_invalid(self):
        """Test analyzing an invalid database."""
        from app.coordination.orphan_detection_daemon import OrphanDetectionDaemon

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a database without games table
            db_path = Path(tmpdir) / "not_a_game.db"
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute("CREATE TABLE other_data (id TEXT)")
                conn.commit()

            daemon = OrphanDetectionDaemon()
            orphan_info = await daemon._analyze_database(db_path)

            # Should return None for non-game databases
            assert orphan_info is None

    @pytest.mark.asyncio
    async def test_analyze_database_empty(self):
        """Test analyzing a database with 0 games."""
        from app.coordination.orphan_detection_daemon import (
            OrphanDetectionDaemon,
            OrphanDetectionConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "hex8_2p.db"
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute("CREATE TABLE games (game_id TEXT PRIMARY KEY)")
                conn.commit()

            config = OrphanDetectionConfig(min_games_to_register=1)
            daemon = OrphanDetectionDaemon(config=config)
            orphan_info = await daemon._analyze_database(db_path)

            # Should still return info, but with 0 games
            assert orphan_info is not None
            assert orphan_info.game_count == 0

    @pytest.mark.asyncio
    async def test_force_scan(self):
        """Test forcing an immediate scan."""
        from app.coordination.orphan_detection_daemon import (
            OrphanDetectionDaemon,
            OrphanDetectionConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create games directory
            games_dir = Path(tmpdir) / "data" / "games"
            games_dir.mkdir(parents=True)

            # Create a test database
            db_path = games_dir / "hex8_3p.db"
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute("CREATE TABLE games (game_id TEXT PRIMARY KEY)")
                conn.execute("INSERT INTO games VALUES ('test_game')")
                conn.commit()

            # Configure daemon with our temp directory
            config = OrphanDetectionConfig(
                games_dir=str(games_dir.relative_to(Path(tmpdir))),
                auto_register_orphans=False,  # Disable to avoid manifest issues
                emit_detection_event=False,  # Disable to avoid event issues
            )

            # Mock the ROOT path
            with patch(
                "app.coordination.orphan_detection_daemon.ROOT",
                Path(tmpdir),
            ):
                daemon = OrphanDetectionDaemon(config=config)
                orphans = await daemon.force_scan()

            # Should find our orphan
            assert len(orphans) == 1
            assert orphans[0].path.name == "hex8_3p.db"
            assert orphans[0].board_type == "hex8"
            assert orphans[0].num_players == 3

    def test_health_check_not_running(self):
        """Test health check when daemon is not running."""
        from app.coordination.orphan_detection_daemon import OrphanDetectionDaemon
        from app.coordination.protocols import CoordinatorStatus

        daemon = OrphanDetectionDaemon()
        result = daemon.health_check()

        assert result.healthy is False
        assert result.status == CoordinatorStatus.STOPPED
        assert "not running" in result.message

    def test_health_check_running(self):
        """Test health check when daemon is running."""
        from app.coordination.orphan_detection_daemon import OrphanDetectionDaemon
        from app.coordination.protocols import CoordinatorStatus
        import time

        daemon = OrphanDetectionDaemon()
        daemon._running = True
        daemon._last_scan_time = time.time()

        result = daemon.health_check()

        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING

    def test_health_check_degraded(self):
        """Test health check when scan is overdue."""
        from app.coordination.orphan_detection_daemon import (
            OrphanDetectionDaemon,
            OrphanDetectionConfig,
        )
        from app.coordination.protocols import CoordinatorStatus
        import time

        config = OrphanDetectionConfig(scan_interval_seconds=60.0)
        daemon = OrphanDetectionDaemon(config=config)
        daemon._running = True
        # Set last scan to way in the past
        daemon._last_scan_time = time.time() - 500  # 500 seconds ago

        result = daemon.health_check()

        assert result.healthy is False
        assert result.status == CoordinatorStatus.DEGRADED
        assert "overdue" in result.message

    def test_get_status(self):
        """Test status reporting."""
        from app.coordination.orphan_detection_daemon import OrphanDetectionDaemon
        import time

        daemon = OrphanDetectionDaemon()
        daemon._running = True
        daemon._last_scan_time = time.time()

        status = daemon.get_status()

        assert status["daemon"] == "OrphanDetectionDaemon"
        assert status["running"] is True
        assert status["last_scan_time"] > 0
        assert status["seconds_since_scan"] is not None
        assert "config" in status


class TestEventEmission:
    """Tests for event emission functionality."""

    @pytest.mark.asyncio
    async def test_emit_detection_event(self):
        """Test ORPHAN_GAMES_DETECTED event emission."""
        from app.coordination.orphan_detection_daemon import (
            OrphanDetectionDaemon,
            OrphanInfo,
        )

        orphans = [
            OrphanInfo(
                path=Path("/tmp/test1.db"),
                game_count=100,
                file_size_bytes=1024000,
                modified_time=1234567890.0,
                board_type="hex8",
                num_players=2,
            ),
            OrphanInfo(
                path=Path("/tmp/test2.db"),
                game_count=50,
                file_size_bytes=512000,
                modified_time=1234567891.0,
                board_type="square8",
                num_players=4,
            ),
        ]

        daemon = OrphanDetectionDaemon()

        # Mock the router
        mock_router = MagicMock()
        mock_router.publish = AsyncMock()

        with patch(
            "app.coordination.orphan_detection_daemon.get_router",
            return_value=mock_router,
        ):
            await daemon._emit_detection_event(orphans)

        # Verify publish was called
        mock_router.publish.assert_called_once()
        call_args = mock_router.publish.call_args
        payload = call_args[0][1]  # Second positional arg

        assert payload["orphan_count"] == 2
        assert payload["total_games"] == 150
        assert payload["total_bytes"] == 1024000 + 512000
        assert len(payload["orphan_paths"]) == 2
        assert "hex8" in payload["board_types"]
        assert "square8" in payload["board_types"]

    @pytest.mark.asyncio
    async def test_emit_detection_event_no_router(self):
        """Test event emission when router is not available."""
        from app.coordination.orphan_detection_daemon import (
            OrphanDetectionDaemon,
            OrphanInfo,
        )

        orphans = [
            OrphanInfo(
                path=Path("/tmp/test.db"),
                game_count=10,
                file_size_bytes=1024,
                modified_time=1234567890.0,
            ),
        ]

        daemon = OrphanDetectionDaemon()

        with patch(
            "app.coordination.orphan_detection_daemon.get_router",
            return_value=None,
        ):
            # Should not raise
            await daemon._emit_detection_event(orphans)

    @pytest.mark.asyncio
    async def test_emit_registration_event(self):
        """Test ORPHAN_GAMES_REGISTERED event emission."""
        from app.coordination.orphan_detection_daemon import (
            OrphanDetectionDaemon,
            OrphanInfo,
        )

        registered = [
            OrphanInfo(
                path=Path("/tmp/registered.db"),
                game_count=75,
                file_size_bytes=768000,
                modified_time=1234567890.0,
                board_type="hex8",
                num_players=2,
            ),
        ]

        daemon = OrphanDetectionDaemon()

        mock_router = MagicMock()
        mock_router.publish = AsyncMock()

        with patch(
            "app.coordination.orphan_detection_daemon.get_router",
            return_value=mock_router,
        ):
            await daemon._emit_registration_event(registered)

        mock_router.publish.assert_called_once()
        call_args = mock_router.publish.call_args
        payload = call_args[0][1]

        assert payload["registered_count"] == 1
        assert payload["total_games"] == 75


class TestDatabaseEventSubscription:
    """Tests for DATABASE_CREATED event subscription."""

    @pytest.mark.asyncio
    async def test_subscribe_to_database_events(self):
        """Test subscription to DATABASE_CREATED events."""
        from app.coordination.orphan_detection_daemon import OrphanDetectionDaemon

        daemon = OrphanDetectionDaemon()

        # Mock the subscribe function
        mock_subscribe = MagicMock()

        with patch(
            "app.coordination.orphan_detection_daemon.subscribe",
            mock_subscribe,
        ):
            await daemon._subscribe_to_database_events()

        # Verify subscribe was called with correct event type
        assert mock_subscribe.called

    @pytest.mark.asyncio
    async def test_register_database_from_event(self):
        """Test database registration from event."""
        from app.coordination.orphan_detection_daemon import OrphanDetectionDaemon

        daemon = OrphanDetectionDaemon()

        mock_manifest = MagicMock()

        with patch(
            "app.coordination.orphan_detection_daemon.get_cluster_manifest",
            return_value=mock_manifest,
        ):
            await daemon._register_database_from_event(
                db_path="/tmp/test.db",
                node_id="node-1",
                config_key="hex8_2p",
                board_type="hex8",
                num_players=2,
                engine_mode="gumbel",
            )

        mock_manifest.register_database.assert_called_once()


class TestRegistration:
    """Tests for orphan registration functionality."""

    @pytest.mark.asyncio
    async def test_register_orphans(self):
        """Test registering orphans in ClusterManifest."""
        from app.coordination.orphan_detection_daemon import (
            OrphanDetectionDaemon,
            OrphanInfo,
        )

        orphans = [
            OrphanInfo(
                path=Path("/tmp/orphan1.db"),
                game_count=50,
                file_size_bytes=512000,
                modified_time=1234567890.0,
                board_type="hex8",
                num_players=2,
            ),
        ]

        daemon = OrphanDetectionDaemon()

        mock_manifest = MagicMock()
        mock_manifest.register_database = MagicMock()

        with patch(
            "app.coordination.orphan_detection_daemon.get_cluster_manifest",
            return_value=mock_manifest,
        ):
            with patch.object(daemon, "_emit_registration_event", new_callable=AsyncMock):
                registered_count = await daemon._register_orphans(orphans)

        assert registered_count == 1
        mock_manifest.register_database.assert_called_once()


class TestModuleRun:
    """Tests for module-level run function."""

    @pytest.mark.asyncio
    async def test_run_function(self):
        """Test the run() entry point."""
        from app.coordination.orphan_detection_daemon import run

        # Mock the daemon
        mock_daemon = MagicMock()
        mock_daemon.start = AsyncMock()

        with patch(
            "app.coordination.orphan_detection_daemon.OrphanDetectionDaemon",
            return_value=mock_daemon,
        ):
            # Create a task that we'll cancel after brief run
            task = asyncio.create_task(run())
            await asyncio.sleep(0.05)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

        mock_daemon.start.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
