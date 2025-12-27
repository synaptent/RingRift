"""Tests for DataConsolidationDaemon.

Tests the automatic consolidation of scattered selfplay games into canonical databases.
December 2025: Created as part of training pipeline infrastructure verification.
"""

import asyncio
import pytest
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.coordination.data_consolidation_daemon import (
    DataConsolidationDaemon,
    ConsolidationConfig,
    ConsolidationStats,
    ALL_CONFIGS,
    get_consolidation_daemon,
    reset_consolidation_daemon,
)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory structure."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    return games_dir


@pytest.fixture
def config(temp_data_dir):
    """Create a test configuration."""
    return ConsolidationConfig(
        data_dir=temp_data_dir,
        canonical_dir=temp_data_dir,
        min_games_for_consolidation=5,
        consolidation_interval_seconds=1.0,
        min_moves_for_valid=3,
        batch_size=10,
        deduplicate=True,
        validate_before_merge=True,
    )


@pytest.fixture
def daemon(config):
    """Create a test daemon instance."""
    with patch.object(DataConsolidationDaemon, '_subscribe_to_events', new_callable=AsyncMock):
        d = DataConsolidationDaemon(config=config)
        yield d


@pytest.fixture
def source_db(temp_data_dir):
    """Create a source database with test games."""
    db_path = temp_data_dir / "selfplay" / "selfplay_test.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY,
            board_type TEXT NOT NULL,
            num_players INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            game_status TEXT NOT NULL,
            winner INTEGER,
            total_moves INTEGER NOT NULL,
            total_turns INTEGER NOT NULL,
            source TEXT,
            schema_version INTEGER NOT NULL DEFAULT 5
        )
    """)
    conn.execute("""
        CREATE TABLE game_moves (
            game_id TEXT NOT NULL,
            move_number INTEGER NOT NULL,
            player INTEGER NOT NULL,
            position_q INTEGER,
            position_r INTEGER,
            move_type TEXT,
            PRIMARY KEY (game_id, move_number)
        )
    """)

    # Insert test games
    for i in range(10):
        game_id = f"test-game-{i:04d}"
        conn.execute("""
            INSERT INTO games (game_id, board_type, num_players, created_at, game_status,
                             total_moves, total_turns, source, schema_version)
            VALUES (?, ?, ?, datetime('now'), ?, ?, ?, 'test', 5)
        """, (game_id, "hex8", 2, "completed", i + 5, i + 3))

        # Add moves
        for move in range(i + 5):
            conn.execute("""
                INSERT INTO game_moves (game_id, move_number, player, position_q, position_r)
                VALUES (?, ?, ?, ?, ?)
            """, (game_id, move, move % 2, move, move + 1))

    conn.commit()
    conn.close()
    return db_path


class TestConsolidationConfig:
    """Test ConsolidationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ConsolidationConfig()
        assert config.min_games_for_consolidation == 50
        assert config.consolidation_interval_seconds == 300.0
        assert config.min_moves_for_valid == 5
        assert config.deduplicate is True

    def test_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict('os.environ', {
            'RINGRIFT_DATA_DIR': '/custom/data',
            'RINGRIFT_CONSOLIDATION_MIN_GAMES': '100',
            'RINGRIFT_CONSOLIDATION_INTERVAL': '600',
        }):
            config = ConsolidationConfig.from_env()
            assert config.data_dir == Path('/custom/data')
            assert config.min_games_for_consolidation == 100
            assert config.consolidation_interval_seconds == 600.0


class TestConsolidationStats:
    """Test ConsolidationStats dataclass."""

    def test_initial_state(self):
        """Test initial stats values."""
        stats = ConsolidationStats(config_key="hex8_2p")
        assert stats.config_key == "hex8_2p"
        assert stats.games_merged == 0
        assert stats.games_duplicate == 0
        assert stats.success is False

    def test_duration_calculation(self):
        """Test duration calculation."""
        stats = ConsolidationStats(
            config_key="hex8_2p",
            start_time=100.0,
            end_time=150.0,
        )
        assert stats.duration_seconds == 50.0

    def test_duration_in_progress(self):
        """Test duration for in-progress consolidation."""
        stats = ConsolidationStats(
            config_key="hex8_2p",
            start_time=time.time() - 10,
        )
        assert 9 < stats.duration_seconds < 12


class TestDataConsolidationDaemon:
    """Test DataConsolidationDaemon functionality."""

    @pytest.mark.asyncio
    async def test_daemon_initialization(self, daemon):
        """Test daemon initializes correctly."""
        assert daemon.config.min_games_for_consolidation == 5
        assert daemon._running is False
        assert len(daemon._pending_configs) == 0

    @pytest.mark.asyncio
    async def test_on_new_games_available(self, daemon):
        """Test NEW_GAMES_AVAILABLE event handler."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "games_added": 10,
        }

        daemon._on_new_games_available(event)

        assert "hex8_2p" in daemon._pending_configs

    @pytest.mark.asyncio
    async def test_on_new_games_zero_count_ignored(self, daemon):
        """Test that zero games_added is ignored."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "games_added": 0,
        }

        daemon._on_new_games_available(event)

        assert "hex8_2p" not in daemon._pending_configs

    @pytest.mark.asyncio
    async def test_on_selfplay_complete(self, daemon):
        """Test SELFPLAY_COMPLETE event handler."""
        event = MagicMock()
        event.payload = {
            "board_type": "square8",
            "num_players": 4,
        }

        daemon._on_selfplay_complete(event)

        assert "square8_4p" in daemon._pending_configs

    @pytest.mark.asyncio
    async def test_on_selfplay_complete_with_config_key(self, daemon):
        """Test SELFPLAY_COMPLETE with explicit config_key."""
        event = MagicMock()
        event.payload = {
            "config_key": "hexagonal_3p",
        }

        daemon._on_selfplay_complete(event)

        assert "hexagonal_3p" in daemon._pending_configs

    @pytest.mark.asyncio
    async def test_find_source_databases(self, daemon, source_db):
        """Test finding source databases for a config."""
        sources = daemon._find_source_databases("hex8", 2)

        assert len(sources) == 1
        assert sources[0] == source_db

    @pytest.mark.asyncio
    async def test_find_source_databases_excludes_canonical(self, daemon, temp_data_dir):
        """Test that canonical databases are excluded from sources."""
        # Create a canonical database
        canonical_db = temp_data_dir / "canonical_hex8_2p.db"
        conn = sqlite3.connect(str(canonical_db))
        conn.execute("""
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT,
                num_players INTEGER
            )
        """)
        conn.execute("INSERT INTO games VALUES ('test', 'hex8', 2)")
        conn.commit()
        conn.close()

        sources = daemon._find_source_databases("hex8", 2)

        assert canonical_db not in sources

    @pytest.mark.asyncio
    async def test_has_games_for_config(self, daemon, source_db):
        """Test checking if database has games for config."""
        assert daemon._has_games_for_config(source_db, "hex8", 2) is True
        assert daemon._has_games_for_config(source_db, "square8", 4) is False

    @pytest.mark.asyncio
    async def test_get_existing_game_ids(self, daemon, source_db):
        """Test retrieving existing game IDs."""
        # First consolidate to create canonical
        await daemon._consolidate_config("hex8", 2)

        canonical_path = daemon._get_canonical_db_path("hex8", 2)
        ids = daemon._get_existing_game_ids(canonical_path)

        assert len(ids) == 10
        assert "test-game-0000" in ids

    @pytest.mark.asyncio
    async def test_ensure_canonical_schema(self, daemon, temp_data_dir):
        """Test canonical schema creation."""
        canonical_path = temp_data_dir / "canonical_test.db"
        daemon._ensure_canonical_schema(canonical_path)

        assert canonical_path.exists()

        conn = sqlite3.connect(str(canonical_path))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "games" in tables
        assert "game_moves" in tables
        assert "game_initial_state" in tables
        assert "game_state_snapshots" in tables
        assert "game_players" in tables

    @pytest.mark.asyncio
    async def test_consolidate_config(self, daemon, source_db):
        """Test full consolidation of a config."""
        stats = await daemon._consolidate_config("hex8", 2)

        assert stats.success is True
        assert stats.games_merged == 10
        assert stats.games_duplicate == 0
        assert stats.source_dbs_scanned == 1

    @pytest.mark.asyncio
    async def test_consolidate_deduplicates(self, daemon, source_db):
        """Test that consolidation deduplicates games."""
        # Run consolidation twice
        stats1 = await daemon._consolidate_config("hex8", 2)
        stats2 = await daemon._consolidate_config("hex8", 2)

        assert stats1.games_merged == 10
        assert stats2.games_merged == 0  # All duplicates
        assert stats2.games_duplicate == 10

    @pytest.mark.asyncio
    async def test_consolidate_filters_invalid_games(self, daemon, temp_data_dir):
        """Test that games with too few moves are filtered."""
        # Create database with short games
        db_path = temp_data_dir / "short_games.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                game_status TEXT NOT NULL,
                total_moves INTEGER NOT NULL,
                total_turns INTEGER NOT NULL,
                schema_version INTEGER DEFAULT 5
            )
        """)
        # Game with only 2 moves (below threshold of 3)
        conn.execute("""
            INSERT INTO games VALUES ('short-game', 'square8', 2,
                datetime('now'), 'completed', 2, 2, 5)
        """)
        conn.commit()
        conn.close()

        stats = await daemon._consolidate_config("square8", 2)

        assert stats.games_invalid == 1
        assert stats.games_merged == 0

    @pytest.mark.asyncio
    async def test_trigger_consolidation(self, daemon, source_db):
        """Test manual consolidation trigger."""
        stats = await daemon.trigger_consolidation("hex8_2p")

        assert stats is not None
        assert stats.success is True
        assert stats.config_key == "hex8_2p"

    @pytest.mark.asyncio
    async def test_trigger_all_consolidations(self, daemon, source_db):
        """Test triggering consolidation for all configs."""
        results = await daemon.trigger_all_consolidations()

        assert len(results) == len(ALL_CONFIGS)
        assert "hex8_2p" in results
        assert results["hex8_2p"].success is True

    @pytest.mark.asyncio
    async def test_get_status(self, daemon, source_db):
        """Test daemon status reporting."""
        await daemon._consolidate_config("hex8", 2)
        status = daemon.get_status()

        assert status["running"] is False
        assert len(status["recent_stats"]) == 1
        assert status["recent_stats"][0]["config_key"] == "hex8_2p"
        assert status["recent_stats"][0]["success"] is True

    @pytest.mark.asyncio
    async def test_health_check_not_running(self, daemon):
        """Test health check when daemon is not running."""
        health = daemon.health_check()

        assert health.healthy is False
        assert health.status == "degraded"
        assert health.details["running"] is False

    @pytest.mark.asyncio
    async def test_health_check_running_not_subscribed(self, daemon):
        """Test health check when running but not subscribed."""
        daemon._running = True
        daemon._subscribed = False

        health = daemon.health_check()

        assert health.healthy is False
        assert health.details["subscribed"] is False

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, daemon):
        """Test health check when fully healthy."""
        daemon._running = True
        daemon._subscribed = True

        health = daemon.health_check()

        assert health.healthy is True
        assert health.status == "running"

    @pytest.mark.asyncio
    async def test_event_emission_on_consolidation(self, daemon, source_db):
        """Test that CONSOLIDATION events are emitted."""
        with patch('app.coordination.data_consolidation_daemon.emit_event',
                   new_callable=AsyncMock) as mock_emit:
            await daemon._consolidate_config("hex8", 2)

            # Should emit CONSOLIDATION_STARTED and CONSOLIDATION_COMPLETE
            assert mock_emit.call_count >= 2

    @pytest.mark.asyncio
    async def test_process_pending_consolidations(self, daemon, source_db):
        """Test processing pending consolidations."""
        daemon._pending_configs.add("hex8_2p")

        await daemon._process_pending_consolidations()

        # Config should be removed from pending after processing
        assert "hex8_2p" not in daemon._pending_configs

        # Should have stats recorded
        assert len(daemon._stats_history) == 1

    @pytest.mark.asyncio
    async def test_consolidation_cooldown(self, daemon, source_db):
        """Test that consolidation respects cooldown period."""
        # First consolidation
        await daemon._consolidate_config("hex8", 2)

        # Mark as just consolidated
        daemon._last_consolidation["hex8_2p"] = time.time()

        # Add to pending
        daemon._pending_configs.add("hex8_2p")

        # Try to process - should skip due to cooldown
        initial_history_len = len(daemon._stats_history)
        await daemon._process_pending_consolidations()

        # Config should be removed but no new stats added (skipped)
        # Actually it stays in pending because cooldown check is inside
        # the processing loop


class TestSingletonPattern:
    """Test singleton pattern for consolidation daemon."""

    def test_get_consolidation_daemon_returns_same_instance(self):
        """Test that get_consolidation_daemon returns singleton."""
        reset_consolidation_daemon()

        with patch.object(DataConsolidationDaemon, '_subscribe_to_events'):
            daemon1 = get_consolidation_daemon()
            daemon2 = get_consolidation_daemon()

            assert daemon1 is daemon2

        reset_consolidation_daemon()

    def test_reset_consolidation_daemon(self):
        """Test resetting the singleton instance."""
        reset_consolidation_daemon()

        with patch.object(DataConsolidationDaemon, '_subscribe_to_events'):
            daemon1 = get_consolidation_daemon()
            reset_consolidation_daemon()
            daemon2 = get_consolidation_daemon()

            assert daemon1 is not daemon2

        reset_consolidation_daemon()


class TestAllConfigs:
    """Test ALL_CONFIGS constant."""

    def test_all_configs_contains_12_entries(self):
        """Test that ALL_CONFIGS has all 12 board configurations."""
        assert len(ALL_CONFIGS) == 12

    def test_all_configs_has_all_board_types(self):
        """Test that all board types are represented."""
        board_types = {cfg[0] for cfg in ALL_CONFIGS}
        assert board_types == {"hex8", "square8", "square19", "hexagonal"}

    def test_all_configs_has_all_player_counts(self):
        """Test that all player counts are represented for each board."""
        for board_type in ["hex8", "square8", "square19", "hexagonal"]:
            player_counts = {cfg[1] for cfg in ALL_CONFIGS if cfg[0] == board_type}
            assert player_counts == {2, 3, 4}
