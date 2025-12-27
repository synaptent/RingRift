"""Tests for Data Cleanup Daemon.

Tests the automatic data cleanup functionality that:
- Quarantines databases with quality < 30%
- Deletes databases with quality < 10%
- Logs all cleanup actions to audit file

December 27, 2025: Created as part of coordination test coverage improvement.
"""

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.data_cleanup_daemon import (
    CleanupConfig,
    CleanupStats,
    DataCleanupDaemon,
    DatabaseAssessment,
    get_cleanup_daemon,
    reset_cleanup_daemon,
)
from app.coordination.protocols import CoordinatorStatus


@pytest.fixture(autouse=True)
def cleanup_singleton():
    """Reset singleton after each test."""
    reset_cleanup_daemon()
    yield
    reset_cleanup_daemon()


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory with test databases."""
    data_dir = tmp_path / "data" / "games"
    data_dir.mkdir(parents=True)
    return data_dir


@pytest.fixture
def mock_config(temp_data_dir):
    """Create a test configuration."""
    return CleanupConfig(
        enabled=True,
        scan_interval_seconds=60,
        quality_threshold_delete=0.1,
        quality_threshold_quarantine=0.3,
        move_coverage_threshold=0.1,
        quality_sample_size=5,
        data_dir=temp_data_dir,
        min_games_before_delete=10,
        require_canonical_pattern=False,  # Allow cleanup of all DBs for testing
    )


@pytest.fixture
def daemon(mock_config):
    """Create a test daemon instance."""
    return DataCleanupDaemon(config=mock_config)


def create_test_database(db_path: Path, num_games: int = 50, quality: float = 0.5) -> None:
    """Create a minimal test database with games."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            game_status TEXT DEFAULT 'completed',
            winner INTEGER,
            termination_reason TEXT DEFAULT 'score',
            total_moves INTEGER DEFAULT 20,
            board_type TEXT DEFAULT 'square8',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_moves (
            game_id TEXT,
            move_number INTEGER,
            move_type TEXT,
            PRIMARY KEY (game_id, move_number)
        )
    """)

    # Insert test games with quality-appropriate settings
    for i in range(num_games):
        game_id = f"game_{i}"
        winner = 0 if quality < 0.2 else (i % 4)
        term_reason = "unknown" if quality < 0.2 else "score"
        moves = 5 if quality < 0.2 else 20

        conn.execute(
            "INSERT INTO games (game_id, winner, termination_reason, total_moves) VALUES (?, ?, ?, ?)",
            (game_id, winner, term_reason, moves)
        )

        # Add moves for good quality games
        if quality >= 0.3:
            for m in range(moves):
                conn.execute(
                    "INSERT INTO game_moves (game_id, move_number, move_type) VALUES (?, ?, ?)",
                    (game_id, m, "place")
                )

    conn.commit()
    conn.close()


# =============================================================================
# Config Tests
# =============================================================================


class TestCleanupConfig:
    """Tests for CleanupConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CleanupConfig()
        assert config.enabled is True
        assert config.quality_threshold_delete == 0.1
        assert config.quality_threshold_quarantine == 0.3
        assert config.move_coverage_threshold == 0.1
        assert config.require_canonical_pattern is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CleanupConfig(
            enabled=False,
            quality_threshold_delete=0.05,
            quality_threshold_quarantine=0.2,
            min_games_before_delete=500,
        )
        assert config.enabled is False
        assert config.quality_threshold_delete == 0.05
        assert config.quality_threshold_quarantine == 0.2
        assert config.min_games_before_delete == 500


# =============================================================================
# DatabaseAssessment Tests
# =============================================================================


class TestDatabaseAssessment:
    """Tests for DatabaseAssessment dataclass."""

    def test_basic_creation(self):
        """Test basic assessment creation."""
        assessment = DatabaseAssessment(
            path="/data/test.db",
            total_games=100,
            quality_score=0.7,
            move_coverage=0.9,
            valid_victory_types=0.85,
        )
        assert assessment.path == "/data/test.db"
        assert assessment.total_games == 100
        assert assessment.quality_score == 0.7
        assert assessment.issues == []

    def test_path_conversion(self):
        """Test that Path objects are converted to strings."""
        assessment = DatabaseAssessment(
            path=Path("/data/test.db"),
            total_games=50,
            quality_score=0.5,
            move_coverage=0.8,
            valid_victory_types=0.9,
        )
        assert isinstance(assessment.path, str)
        assert assessment.path == "/data/test.db"

    def test_issues_list(self):
        """Test issues list tracking."""
        assessment = DatabaseAssessment(
            path="/data/test.db",
            total_games=10,
            quality_score=0.1,
            move_coverage=0.05,
            valid_victory_types=0.3,
            issues=["Low quality", "Low move coverage"],
        )
        assert len(assessment.issues) == 2
        assert "Low quality" in assessment.issues


# =============================================================================
# CleanupStats Tests
# =============================================================================


class TestCleanupStats:
    """Tests for CleanupStats dataclass."""

    def test_default_values(self):
        """Test stats initialize to zero."""
        stats = CleanupStats()
        assert stats.items_scanned == 0
        assert stats.items_quarantined == 0
        assert stats.databases_deleted == 0
        assert stats.games_quarantined == 0

    def test_backward_compat_aliases(self):
        """Test backward compatibility aliases."""
        stats = CleanupStats()
        stats.items_scanned = 10
        stats.items_quarantined = 3

        assert stats.databases_scanned == 10
        assert stats.databases_quarantined == 3

    def test_record_database_scan(self):
        """Test recording database scans."""
        stats = CleanupStats()
        stats.record_database_scan(5)
        assert stats.items_scanned == 5
        assert stats.last_scan_time > 0

    def test_record_database_quarantine(self):
        """Test recording quarantine operations."""
        stats = CleanupStats()
        stats.record_database_quarantine(databases=2, games=100)
        assert stats.items_quarantined == 2
        assert stats.games_quarantined == 100

    def test_record_database_delete(self):
        """Test recording delete operations."""
        stats = CleanupStats()
        stats.record_database_delete(databases=1, games=50)
        assert stats.databases_deleted == 1
        assert stats.items_cleaned == 1
        assert stats.games_deleted == 50


# =============================================================================
# Daemon Initialization Tests
# =============================================================================


class TestDataCleanupDaemonInit:
    """Tests for daemon initialization."""

    def test_default_initialization(self):
        """Test daemon initializes with defaults."""
        daemon = DataCleanupDaemon()
        assert daemon.config is not None
        assert daemon._running is False
        assert daemon._coordinator_status == CoordinatorStatus.INITIALIZING

    def test_custom_config(self, mock_config):
        """Test daemon with custom config."""
        daemon = DataCleanupDaemon(config=mock_config)
        assert daemon.config.enabled is True
        assert daemon.config.scan_interval_seconds == 60

    def test_name_property(self, daemon):
        """Test name property."""
        assert daemon.name == "DataCleanupDaemon"

    def test_uptime_before_start(self, daemon):
        """Test uptime is zero before start."""
        assert daemon.uptime_seconds == 0.0


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestDaemonLifecycle:
    """Tests for daemon lifecycle."""

    @pytest.mark.asyncio
    async def test_start_creates_quarantine_dir(self, daemon, temp_data_dir):
        """Test that start creates quarantine directory."""
        quarantine_dir = temp_data_dir / "quarantine"
        assert not quarantine_dir.exists()

        with patch.object(daemon, "_subscribe_to_events", new_callable=AsyncMock):
            with patch.object(daemon, "_scan_loop", new_callable=AsyncMock):
                await daemon.start()

        assert quarantine_dir.exists()
        await daemon.stop()

    @pytest.mark.asyncio
    async def test_start_disabled(self, mock_config):
        """Test that disabled daemon doesn't start."""
        mock_config.enabled = False
        daemon = DataCleanupDaemon(config=mock_config)

        await daemon.start()

        assert daemon._coordinator_status == CoordinatorStatus.STOPPED
        assert not daemon._running

    @pytest.mark.asyncio
    async def test_start_idempotent(self, daemon):
        """Test that starting twice is safe."""
        with patch.object(daemon, "_subscribe_to_events", new_callable=AsyncMock):
            with patch.object(daemon, "_scan_loop", new_callable=AsyncMock):
                await daemon.start()
                await daemon.start()  # Should not error

        assert daemon._running is True
        await daemon.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self, daemon):
        """Test that stop cancels scan task."""
        with patch.object(daemon, "_subscribe_to_events", new_callable=AsyncMock):
            with patch.object(daemon, "_scan_loop", new_callable=AsyncMock):
                await daemon.start()

        assert daemon._scan_task is not None

        await daemon.stop()

        assert daemon._coordinator_status == CoordinatorStatus.STOPPED
        assert not daemon._running

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, daemon):
        """Test that stopping twice is safe."""
        await daemon.stop()
        await daemon.stop()  # Should not error
        assert daemon._coordinator_status == CoordinatorStatus.STOPPED


# =============================================================================
# Database Assessment Tests
# =============================================================================


class TestDatabaseAssessment:
    """Tests for database assessment."""

    def test_assess_empty_database(self, daemon, temp_data_dir):
        """Test assessment of empty database."""
        db_path = temp_data_dir / "empty.db"
        create_test_database(db_path, num_games=0)

        assessment = daemon._assess_database(db_path)

        assert assessment.total_games == 0
        assert assessment.quality_score == 0.0

    def test_assess_database_with_games(self, daemon, temp_data_dir):
        """Test assessment of database with games."""
        db_path = temp_data_dir / "test.db"
        create_test_database(db_path, num_games=50, quality=0.8)

        with patch("app.quality.unified_quality.compute_game_quality_from_params") as mock_quality:
            mock_quality.return_value = MagicMock(quality_score=0.8)
            assessment = daemon._assess_database(db_path)

        assert assessment.total_games == 50
        assert assessment.sampled_games > 0

    def test_assess_database_quality_module_missing(self, daemon, temp_data_dir):
        """Test fallback when quality module not available."""
        db_path = temp_data_dir / "test.db"
        create_test_database(db_path, num_games=20)

        with patch.dict("sys.modules", {"app.quality.unified_quality": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                assessment = daemon._assess_database(db_path)

        assert assessment.quality_score == 0.5  # Default fallback
        assert "Quality module not available" in assessment.issues


# =============================================================================
# Scan and Cleanup Tests
# =============================================================================


class TestScanAndCleanup:
    """Tests for scan and cleanup logic."""

    @pytest.mark.asyncio
    async def test_scan_no_databases(self, daemon):
        """Test scan with no databases."""
        await daemon._scan_and_cleanup()
        assert daemon._stats.databases_scanned == 0

    @pytest.mark.asyncio
    async def test_scan_skips_canonical(self, daemon, temp_data_dir, mock_config):
        """Test that canonical databases are skipped when configured."""
        mock_config.require_canonical_pattern = True
        daemon = DataCleanupDaemon(config=mock_config)

        db_path = temp_data_dir / "canonical_hex8_2p.db"
        create_test_database(db_path, num_games=50)

        with patch.object(daemon, "_assess_database") as mock_assess:
            mock_assess.return_value = DatabaseAssessment(
                path=str(db_path),
                total_games=50,
                quality_score=0.05,  # Very low - would normally delete
                move_coverage=0.9,
                valid_victory_types=0.9,
            )
            await daemon._scan_and_cleanup()

        # Should not quarantine or delete canonical
        assert daemon._stats.databases_quarantined == 0
        assert daemon._stats.databases_deleted == 0

    @pytest.mark.asyncio
    async def test_scan_quarantines_low_quality(self, daemon, temp_data_dir):
        """Test that low quality databases are quarantined."""
        db_path = temp_data_dir / "bad.db"
        create_test_database(db_path, num_games=50)

        # Ensure quarantine directory exists (normally created by _on_start)
        (temp_data_dir / "quarantine").mkdir(parents=True, exist_ok=True)

        with patch.object(daemon, "_assess_database") as mock_assess:
            mock_assess.return_value = DatabaseAssessment(
                path=str(db_path),
                total_games=50,
                quality_score=0.15,  # Below quarantine threshold
                move_coverage=0.9,
                valid_victory_types=0.9,
            )
            await daemon._scan_and_cleanup()

        assert daemon._stats.databases_quarantined == 1
        assert not db_path.exists()
        assert (temp_data_dir / "quarantine" / "bad.db").exists()

    @pytest.mark.asyncio
    async def test_scan_deletes_very_low_quality(self, daemon, temp_data_dir):
        """Test that very low quality databases are deleted."""
        db_path = temp_data_dir / "terrible.db"
        create_test_database(db_path, num_games=100)  # Over min_games_before_delete

        with patch.object(daemon, "_assess_database") as mock_assess:
            mock_assess.return_value = DatabaseAssessment(
                path=str(db_path),
                total_games=100,
                quality_score=0.05,  # Below delete threshold
                move_coverage=0.9,
                valid_victory_types=0.9,
            )
            await daemon._scan_and_cleanup()

        assert daemon._stats.databases_deleted == 1
        assert not db_path.exists()

    @pytest.mark.asyncio
    async def test_scan_skips_delete_under_min_games(self, daemon, temp_data_dir):
        """Test that deletion is skipped for small databases."""
        db_path = temp_data_dir / "small.db"
        create_test_database(db_path, num_games=5)  # Under min_games_before_delete

        with patch.object(daemon, "_assess_database") as mock_assess:
            mock_assess.return_value = DatabaseAssessment(
                path=str(db_path),
                total_games=5,
                quality_score=0.05,  # Very low quality
                move_coverage=0.9,
                valid_victory_types=0.9,
            )
            await daemon._scan_and_cleanup()

        assert daemon._stats.databases_deleted == 0
        assert db_path.exists()  # Not deleted due to low game count


# =============================================================================
# Event Handler Tests
# =============================================================================


class TestEventHandlers:
    """Tests for event handlers."""

    @pytest.mark.asyncio
    async def test_on_quality_alert_triggers_scan(self, daemon):
        """Test that quality alert triggers immediate scan."""
        daemon._running = True

        with patch.object(daemon, "_scan_and_cleanup", new_callable=AsyncMock) as mock_scan:
            event = SimpleNamespace(
                payload={
                    "database": "/path/to/bad.db",
                    "quality_score": 0.1,
                }
            )
            await daemon._on_quality_alert(event)

            mock_scan.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_sync_completed_triggers_scan(self, daemon):
        """Test that sync completed triggers scan for significant sync."""
        daemon._running = True

        with patch.object(daemon, "_scan_and_cleanup", new_callable=AsyncMock) as mock_scan:
            event = SimpleNamespace(
                payload={
                    "games_synced": 500,  # > 100 threshold
                }
            )
            await daemon._on_sync_completed(event)

            mock_scan.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_sync_completed_skips_small_sync(self, daemon):
        """Test that small sync doesn't trigger scan."""
        daemon._running = True

        with patch.object(daemon, "_scan_and_cleanup", new_callable=AsyncMock) as mock_scan:
            event = SimpleNamespace(
                payload={
                    "games_synced": 50,  # < 100 threshold
                }
            )
            await daemon._on_sync_completed(event)

            mock_scan.assert_not_called()


# =============================================================================
# Status and Metrics Tests
# =============================================================================


class TestStatusAndMetrics:
    """Tests for status and metrics methods."""

    def test_get_status(self, daemon):
        """Test get_status returns expected fields."""
        daemon._running = True
        daemon._stats.items_scanned = 100
        daemon._stats.items_quarantined = 5

        status = daemon.get_status()

        assert status["running"] is True
        assert status["stats"]["databases_scanned"] == 100
        assert status["stats"]["databases_quarantined"] == 5
        assert "config" in status
        assert "directories" in status

    def test_get_metrics(self, daemon):
        """Test get_metrics returns protocol-compliant format."""
        daemon._coordinator_status = CoordinatorStatus.RUNNING
        daemon._stats.items_scanned = 50

        metrics = daemon.get_metrics()

        assert metrics["name"] == "DataCleanupDaemon"
        assert metrics["status"] == "running"
        assert metrics["databases_scanned"] == 50


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health check."""

    def test_health_check_stopped(self, daemon):
        """Test health check when stopped."""
        daemon._coordinator_status = CoordinatorStatus.STOPPED

        result = daemon.health_check()

        assert result.healthy is True
        assert result.status == CoordinatorStatus.STOPPED

    def test_health_check_disabled(self, mock_config):
        """Test health check when disabled."""
        mock_config.enabled = False
        daemon = DataCleanupDaemon(config=mock_config)

        result = daemon.health_check()

        assert result.healthy is True
        assert "disabled" in result.message.lower()

    def test_health_check_error_state(self, daemon):
        """Test health check in error state."""
        daemon._coordinator_status = CoordinatorStatus.ERROR
        daemon._last_error = "Test error"

        result = daemon.health_check()

        assert result.healthy is False
        assert "error" in result.message.lower()

    def test_health_check_stale_scan(self, daemon):
        """Test health check with stale scan returns degraded status."""
        daemon._coordinator_status = CoordinatorStatus.RUNNING
        daemon._stats.last_scan_time = time.time() - 3600  # 1 hour ago
        daemon.config.scan_interval_seconds = 60  # 1 minute interval

        result = daemon.health_check()

        # Degraded is still healthy=True but with degraded status
        assert result.status == CoordinatorStatus.DEGRADED
        assert "No scan" in result.message

    def test_health_check_running_ok(self, daemon):
        """Test health check when running normally."""
        daemon._coordinator_status = CoordinatorStatus.RUNNING
        daemon._stats.last_scan_time = time.time()

        result = daemon.health_check()

        assert result.healthy is True


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_cleanup_daemon_returns_singleton(self):
        """Test singleton access."""
        daemon1 = get_cleanup_daemon()
        daemon2 = get_cleanup_daemon()
        assert daemon1 is daemon2

    def test_reset_cleanup_daemon(self):
        """Test singleton reset."""
        daemon1 = get_cleanup_daemon()
        reset_cleanup_daemon()
        daemon2 = get_cleanup_daemon()
        assert daemon1 is not daemon2


# =============================================================================
# Audit Logging Tests
# =============================================================================


class TestAuditLogging:
    """Tests for audit log functionality."""

    def test_log_audit_action(self, daemon, temp_data_dir):
        """Test audit log writing."""
        assessment = DatabaseAssessment(
            path=str(temp_data_dir / "test.db"),
            total_games=100,
            quality_score=0.1,
            move_coverage=0.8,
            valid_victory_types=0.7,
            issues=["Low quality"],
        )

        daemon._log_audit_action("delete", temp_data_dir / "test.db", assessment, "Test reason")

        # Check audit file was written
        with open(daemon._audit_path) as f:
            lines = f.readlines()

        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["action"] == "delete"
        assert entry["reason"] == "Test reason"
        assert entry["quality_score"] == 0.1


# =============================================================================
# Scan Now Tests
# =============================================================================


class TestScanNow:
    """Tests for immediate scan trigger."""

    @pytest.mark.asyncio
    async def test_scan_now_returns_counts(self, daemon):
        """Test scan_now returns scan counts."""
        with patch.object(daemon, "_scan_and_cleanup", new_callable=AsyncMock):
            result = await daemon.scan_now()

        assert "scanned" in result
        assert "quarantined" in result
        assert "deleted" in result

    @pytest.mark.asyncio
    async def test_scan_now_tracks_deltas(self, daemon, temp_data_dir):
        """Test scan_now tracks changes from scan."""
        # Pre-populate some stats
        daemon._stats.items_scanned = 10

        with patch.object(daemon, "_scan_and_cleanup", new_callable=AsyncMock) as mock:
            async def increment_stats():
                daemon._stats.items_scanned += 5
            mock.side_effect = increment_stats

            result = await daemon.scan_now()

        assert result["scanned"] == 5
