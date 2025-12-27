"""Tests for OrphanDetectionDaemon.

Tests the orphan detection and recovery functionality:
- Database scanning and analysis
- Orphan registration in ClusterManifest
- Event emission (ORPHAN_GAMES_DETECTED, ORPHAN_GAMES_REGISTERED)
- Priority sync triggering
- Health check compliance

December 2025: Added as part of coordination test coverage improvement.
"""

import asyncio
import pytest
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.coordination.orphan_detection_daemon import (
    OrphanDetectionDaemon,
    OrphanDetectionConfig,
    OrphanInfo,
)


@pytest.fixture
def mock_config():
    """Create a test configuration."""
    return OrphanDetectionConfig(
        scan_interval_seconds=60.0,
        games_dir="data/games",
        auto_register_orphans=True,
        min_games_to_register=1,
        cleanup_enabled=False,
        emit_detection_event=True,
    )


@pytest.fixture
def daemon(mock_config):
    """Create a test daemon instance."""
    return OrphanDetectionDaemon(config=mock_config)


@pytest.fixture
def temp_games_dir():
    """Create a temporary games directory with test databases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        games_dir = Path(tmpdir) / "data" / "games"
        games_dir.mkdir(parents=True)
        yield games_dir


def create_test_db(path: Path, game_count: int = 10) -> None:
    """Create a test database with games."""
    with sqlite3.connect(str(path)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT,
                num_players INTEGER
            )
        """)
        for i in range(game_count):
            conn.execute(
                "INSERT INTO games (game_id, board_type, num_players) VALUES (?, ?, ?)",
                (f"game_{i}", "hex8", 2),
            )
        conn.commit()


class TestOrphanDetectionConfig:
    """Test OrphanDetectionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OrphanDetectionConfig()
        assert config.scan_interval_seconds == 300.0
        assert config.auto_register_orphans is True
        assert config.min_games_to_register == 1
        assert config.cleanup_enabled is False
        assert config.emit_detection_event is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = OrphanDetectionConfig(
            scan_interval_seconds=120.0,
            min_games_to_register=10,
            cleanup_enabled=True,
        )
        assert config.scan_interval_seconds == 120.0
        assert config.min_games_to_register == 10
        assert config.cleanup_enabled is True


class TestOrphanInfo:
    """Test OrphanInfo dataclass."""

    def test_orphan_info_creation(self, temp_games_dir):
        """Test creating OrphanInfo."""
        db_path = temp_games_dir / "test.db"
        create_test_db(db_path, game_count=5)

        info = OrphanInfo(
            path=db_path,
            game_count=5,
            file_size_bytes=db_path.stat().st_size,
            modified_time=db_path.stat().st_mtime,
            board_type="hex8",
            num_players=2,
        )

        assert info.game_count == 5
        assert info.board_type == "hex8"
        assert info.num_players == 2


class TestOrphanDetectionDaemon:
    """Test OrphanDetectionDaemon functionality."""

    def test_daemon_initialization(self, daemon):
        """Test daemon initializes correctly."""
        assert daemon.config.scan_interval_seconds == 60.0
        assert daemon._running is False
        assert daemon._last_scan_time == 0.0
        assert len(daemon._orphan_history) == 0

    @pytest.mark.asyncio
    async def test_analyze_database_valid(self, daemon, temp_games_dir):
        """Test analyzing a valid game database."""
        db_path = temp_games_dir / "hex8_2p.db"
        create_test_db(db_path, game_count=15)

        info = await daemon._analyze_database(db_path)

        assert info is not None
        assert info.game_count == 15
        assert info.path == db_path
        assert info.board_type == "hex8"
        assert info.num_players == 2

    @pytest.mark.asyncio
    async def test_analyze_database_empty(self, daemon, temp_games_dir):
        """Test analyzing an empty database."""
        db_path = temp_games_dir / "empty.db"
        create_test_db(db_path, game_count=0)

        info = await daemon._analyze_database(db_path)

        assert info is not None
        assert info.game_count == 0

    @pytest.mark.asyncio
    async def test_analyze_database_invalid(self, daemon, temp_games_dir):
        """Test analyzing an invalid/non-game database."""
        db_path = temp_games_dir / "invalid.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("CREATE TABLE other_table (id INTEGER)")

        info = await daemon._analyze_database(db_path)
        assert info is None

    @pytest.mark.asyncio
    async def test_analyze_database_parses_board_type(self, daemon, temp_games_dir):
        """Test that board type and num_players are parsed from filename."""
        test_cases = [
            ("canonical_hex8_2p.db", "hex8", 2),
            ("square8_4p.db", "square8", 4),
            ("hexagonal_3p.db", "hexagonal", 3),
            ("square19_2p.db", "square19", 2),
        ]

        for filename, expected_board, expected_players in test_cases:
            db_path = temp_games_dir / filename
            create_test_db(db_path, game_count=5)

            info = await daemon._analyze_database(db_path)
            assert info.board_type == expected_board, f"Failed for {filename}"
            assert info.num_players == expected_players, f"Failed for {filename}"

    @pytest.mark.asyncio
    async def test_get_registered_databases(self, daemon):
        """Test getting registered databases from manifest."""
        with patch("app.distributed.cluster_manifest.get_cluster_manifest") as mock_manifest:
            mock_manifest.return_value.get_all_game_locations.return_value = [
                {"path": "/data/games/db1.db"},
                {"path": "/data/games/db2.db"},
            ]

            registered = await daemon._get_registered_databases()

            assert "/data/games/db1.db" in registered
            assert "/data/games/db2.db" in registered
            assert "db1.db" in registered  # Also includes basename
            assert "db2.db" in registered

    @pytest.mark.asyncio
    async def test_get_registered_databases_no_manifest(self, daemon):
        """Test behavior when manifest is unavailable."""
        with patch.dict("sys.modules", {"app.distributed.cluster_manifest": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                registered = await daemon._get_registered_databases()
                assert registered == set()

    @pytest.mark.asyncio
    async def test_emit_detection_event(self, daemon):
        """Test emission of ORPHAN_GAMES_DETECTED event."""
        orphans = [
            OrphanInfo(
                path=Path("/data/games/db1.db"),
                game_count=100,
                file_size_bytes=1000000,
                modified_time=time.time(),
                board_type="hex8",
                num_players=2,
            ),
            OrphanInfo(
                path=Path("/data/games/db2.db"),
                game_count=50,
                file_size_bytes=500000,
                modified_time=time.time(),
                board_type="square8",
                num_players=4,
            ),
        ]

        with patch("app.coordination.event_router.get_router") as mock_router:
            mock_router.return_value = MagicMock()
            mock_router.return_value.publish = AsyncMock()

            await daemon._emit_detection_event(orphans)

            mock_router.return_value.publish.assert_called_once()
            call_args = mock_router.return_value.publish.call_args
            payload = call_args[0][1]

            assert payload["orphan_count"] == 2
            assert payload["total_games"] == 150
            assert payload["total_bytes"] == 1500000

    @pytest.mark.asyncio
    async def test_emit_registration_event(self, daemon):
        """Test emission of ORPHAN_GAMES_REGISTERED event."""
        registered = [
            OrphanInfo(
                path=Path("/data/games/db1.db"),
                game_count=100,
                file_size_bytes=1000000,
                modified_time=time.time(),
                board_type="hex8",
                num_players=2,
            ),
        ]

        with patch("app.coordination.event_router.get_router") as mock_router:
            mock_router.return_value = MagicMock()
            mock_router.return_value.publish = AsyncMock()

            await daemon._emit_registration_event(registered)

            mock_router.return_value.publish.assert_called_once()
            call_args = mock_router.return_value.publish.call_args
            payload = call_args[0][1]

            assert payload["registered_count"] == 1
            assert payload["total_games"] == 100

    @pytest.mark.asyncio
    async def test_register_orphans(self, daemon):
        """Test registering orphans in manifest."""
        orphans = [
            OrphanInfo(
                path=Path("/data/games/db1.db"),
                game_count=100,
                file_size_bytes=1000000,
                modified_time=time.time(),
                board_type="hex8",
                num_players=2,
            ),
        ]

        with patch("app.distributed.cluster_manifest.get_cluster_manifest") as mock_manifest:
            mock_manifest.return_value.register_database = MagicMock()

            with patch.object(daemon, "_emit_registration_event", new_callable=AsyncMock):
                count = await daemon._register_orphans(orphans)

            assert count == 1
            mock_manifest.return_value.register_database.assert_called_once()

    def test_health_check_not_running(self, daemon):
        """Test health check when daemon is not running."""
        result = daemon.health_check()

        assert result.healthy is False
        assert "not running" in result.message.lower()

    def test_health_check_running(self, daemon):
        """Test health check when daemon is running."""
        daemon._running = True
        daemon._last_scan_time = time.time() - 60  # 1 minute ago

        result = daemon.health_check()

        assert result.healthy is True

    def test_health_check_scan_overdue(self, daemon):
        """Test health check when scan is overdue."""
        daemon._running = True
        daemon.config.scan_interval_seconds = 300  # 5 minutes
        daemon._last_scan_time = time.time() - 1200  # 20 minutes ago (>2x interval)

        result = daemon.health_check()

        assert result.healthy is False
        assert "overdue" in result.message.lower()

    def test_get_status(self, daemon):
        """Test status reporting."""
        daemon._running = True
        daemon._last_scan_time = time.time() - 120

        status = daemon.get_status()

        assert status["daemon"] == "OrphanDetectionDaemon"
        assert status["running"] is True
        assert status["last_scan_time"] > 0
        assert "config" in status

    @pytest.mark.asyncio
    async def test_stop_unsubscribes_events(self, daemon):
        """Test that stop() unsubscribes from events."""
        mock_callback = MagicMock()
        daemon._event_subscription = mock_callback
        daemon._running = True

        with patch("app.coordination.event_router.unsubscribe") as mock_unsubscribe:
            await daemon.stop()

        assert daemon._running is False
        mock_unsubscribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_force_scan(self, daemon, temp_games_dir):
        """Test force scan functionality."""
        # Patch ROOT to use temp directory
        with patch.object(daemon.config, "games_dir", str(temp_games_dir)):
            with patch("app.coordination.orphan_detection_daemon.ROOT", temp_games_dir.parent.parent):
                db_path = temp_games_dir / "test_hex8_2p.db"
                create_test_db(db_path, game_count=10)

                with patch.object(daemon, "_get_registered_databases", new_callable=AsyncMock) as mock_reg:
                    mock_reg.return_value = set()  # No databases registered

                    with patch.object(daemon, "_register_orphans", new_callable=AsyncMock) as mock_register:
                        mock_register.return_value = 1

                        with patch.object(daemon, "_emit_detection_event", new_callable=AsyncMock):
                            orphans = await daemon.force_scan()

                # Should find the test database as orphan
                assert len(orphans) == 1
                assert orphans[0].path == db_path


class TestOrphanDetectionEventSubscription:
    """Test DATABASE_CREATED event subscription."""

    @pytest.mark.asyncio
    async def test_subscribe_to_database_events(self, daemon):
        """Test subscription to DATABASE_CREATED events."""
        with patch("app.coordination.event_router.subscribe") as mock_subscribe:
            await daemon._subscribe_to_database_events()

            mock_subscribe.assert_called_once()
            # Verify subscription is stored for cleanup
            assert daemon._event_subscription is not None

    @pytest.mark.asyncio
    async def test_register_database_from_event(self, daemon):
        """Test database registration from DATABASE_CREATED event."""
        with patch("app.distributed.cluster_manifest.get_cluster_manifest") as mock_manifest:
            mock_manifest.return_value.register_database = MagicMock()

            await daemon._register_database_from_event(
                db_path="/data/games/hex8_2p.db",
                node_id="node-1",
                config_key="hex8_2p",
                board_type="hex8",
                num_players=2,
                engine_mode="gumbel",
            )

            mock_manifest.return_value.register_database.assert_called_once_with(
                db_path="/data/games/hex8_2p.db",
                node_id="node-1",
                board_type="hex8",
                num_players=2,
                config_key="hex8_2p",
                engine_mode="gumbel",
            )


class TestOrphanRecoveryIntegration:
    """Integration tests for orphan recovery flow."""

    @pytest.mark.asyncio
    async def test_full_orphan_detection_flow(self, temp_games_dir):
        """Test complete orphan detection and recovery flow."""
        config = OrphanDetectionConfig(
            games_dir=str(temp_games_dir),
            auto_register_orphans=True,
            min_games_to_register=5,
            emit_detection_event=True,
        )
        daemon = OrphanDetectionDaemon(config=config)

        # Create test databases
        db1 = temp_games_dir / "hex8_2p_orphan.db"
        db2 = temp_games_dir / "square8_4p_orphan.db"
        create_test_db(db1, game_count=10)
        create_test_db(db2, game_count=8)

        with patch("app.coordination.orphan_detection_daemon.ROOT", temp_games_dir.parent):
            with patch.object(daemon, "_get_registered_databases", new_callable=AsyncMock) as mock_reg:
                mock_reg.return_value = set()  # No databases registered

                with patch("app.distributed.cluster_manifest.get_cluster_manifest") as mock_manifest:
                    mock_manifest.return_value.register_database = MagicMock()
                    mock_manifest.return_value.get_all_game_locations.return_value = []

                    with patch.object(daemon, "_emit_detection_event", new_callable=AsyncMock) as mock_emit:
                        orphans = await daemon._run_scan()

        assert len(orphans) == 2
        mock_emit.assert_called_once()
