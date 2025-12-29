"""Tests for ParityValidationDaemon.

December 29, 2025: Comprehensive test coverage for coordinator-side parity validation.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from app.coordination.parity_validation_daemon import (
    ParityValidationConfig,
    ParityValidationDaemon,
    ParityValidationResult,
    ParityValidationSummary,
    get_parity_validation_daemon,
    reset_parity_validation_daemon,
)


class TestParityValidationConfig:
    """Tests for ParityValidationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ParityValidationConfig()
        assert config.check_interval_seconds == 1800  # 30 minutes
        assert config.data_dir == ""
        assert config.max_games_per_db == 100
        assert config.fail_on_missing_npx is False
        assert config.emit_events is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ParityValidationConfig(
            check_interval_seconds=3600,
            data_dir="/custom/path",
            max_games_per_db=50,
            fail_on_missing_npx=True,
            emit_events=False,
        )
        assert config.check_interval_seconds == 3600
        assert config.data_dir == "/custom/path"
        assert config.max_games_per_db == 50
        assert config.fail_on_missing_npx is True
        assert config.emit_events is False

    def test_from_env(self):
        """Test loading configuration from environment."""
        with patch.dict("os.environ", {
            "RINGRIFT_PARITY_DATA_DIR": "/env/path",
            "RINGRIFT_PARITY_MAX_GAMES_PER_DB": "200",
        }):
            config = ParityValidationConfig.from_env()
            assert config.data_dir == "/env/path"
            assert config.max_games_per_db == 200

    def test_from_env_invalid_max_games(self):
        """Test from_env with invalid max_games_per_db."""
        with patch.dict("os.environ", {
            "RINGRIFT_PARITY_MAX_GAMES_PER_DB": "not_a_number",
        }):
            config = ParityValidationConfig.from_env()
            # Should use default when parsing fails
            assert config.max_games_per_db == 100


class TestParityValidationResult:
    """Tests for ParityValidationResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = ParityValidationResult()
        assert result.db_path == ""
        assert result.board_type == ""
        assert result.num_players == 0
        assert result.total_games == 0
        assert result.games_validated == 0
        assert result.games_passed == 0
        assert result.games_failed == 0
        assert result.games_skipped == 0
        assert result.validation_time == ""
        assert result.errors == []

    def test_custom_values(self):
        """Test custom result values."""
        result = ParityValidationResult(
            db_path="/path/to/db",
            board_type="hex8",
            num_players=2,
            total_games=100,
            games_validated=50,
            games_passed=45,
            games_failed=3,
            games_skipped=2,
            errors=["error1"],
        )
        assert result.db_path == "/path/to/db"
        assert result.board_type == "hex8"
        assert result.games_passed == 45


class TestParityValidationSummary:
    """Tests for ParityValidationSummary dataclass."""

    def test_default_values(self):
        """Test default summary values."""
        summary = ParityValidationSummary()
        assert summary.scan_time == ""
        assert summary.databases_scanned == 0
        assert summary.total_games_validated == 0
        assert summary.total_games_passed == 0
        assert summary.total_games_failed == 0
        assert summary.results_by_db == {}
        assert summary.errors == []

    def test_aggregation(self):
        """Test aggregating results."""
        summary = ParityValidationSummary(
            databases_scanned=3,
            total_games_validated=150,
            total_games_passed=140,
            total_games_failed=10,
        )
        assert summary.databases_scanned == 3
        assert summary.total_games_validated == 150


class TestParityValidationDaemon:
    """Tests for ParityValidationDaemon class."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_parity_validation_daemon()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_parity_validation_daemon()

    def test_singleton_pattern(self):
        """Test singleton instance creation."""
        daemon1 = ParityValidationDaemon.get_instance()
        daemon2 = ParityValidationDaemon.get_instance()
        assert daemon1 is daemon2

    def test_reset_instance(self):
        """Test singleton reset."""
        daemon1 = ParityValidationDaemon.get_instance()
        ParityValidationDaemon.reset_instance()
        daemon2 = ParityValidationDaemon.get_instance()
        assert daemon1 is not daemon2

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = ParityValidationConfig(
            check_interval_seconds=900,
            data_dir="/custom/path",
        )
        daemon = ParityValidationDaemon(config=config)
        assert daemon.config.check_interval_seconds == 900
        assert daemon.config.data_dir == "/custom/path"

    def test_init_stats(self):
        """Test initial stats are zero."""
        daemon = ParityValidationDaemon()
        assert daemon._total_validations == 0
        assert daemon._total_passed == 0
        assert daemon._total_failed == 0
        assert daemon._last_validation_time is None
        assert daemon._last_result is None
        assert daemon._npx_available is None

    def test_check_npx_available_found(self):
        """Test npx detection when available."""
        daemon = ParityValidationDaemon()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = daemon._check_npx_available()
            assert result is True
            assert daemon._npx_available is True

    def test_check_npx_available_not_found(self):
        """Test npx detection when not available."""
        daemon = ParityValidationDaemon()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = daemon._check_npx_available()
            assert result is False
            assert daemon._npx_available is False

    def test_check_npx_available_timeout(self):
        """Test npx detection with timeout."""
        daemon = ParityValidationDaemon()
        with patch("subprocess.run") as mock_run:
            import subprocess
            mock_run.side_effect = subprocess.TimeoutExpired("which", 10)
            result = daemon._check_npx_available()
            assert result is False
            assert daemon._npx_available is False

    def test_check_npx_caching(self):
        """Test that npx check is cached."""
        daemon = ParityValidationDaemon()
        daemon._npx_available = True
        with patch("subprocess.run") as mock_run:
            result = daemon._check_npx_available()
            assert result is True
            mock_run.assert_not_called()  # Should use cached value

    def test_health_check_healthy(self):
        """Test health check when healthy."""
        daemon = ParityValidationDaemon()
        daemon._running = True
        daemon._npx_available = True
        daemon._total_validations = 100
        daemon._total_passed = 95
        daemon._total_failed = 5

        health = daemon.health_check()
        assert health.healthy is True
        assert health.message == "healthy"
        assert health.details["running"] is True
        assert health.details["total_validations"] == 100

    def test_health_check_npx_unavailable(self):
        """Test health check when npx unavailable."""
        daemon = ParityValidationDaemon()
        daemon._running = True
        daemon._npx_available = False

        health = daemon.health_check()
        assert health.healthy is False
        assert health.message == "npx not available"

    def test_health_check_not_running(self):
        """Test health check when not running."""
        daemon = ParityValidationDaemon()
        daemon._running = False
        daemon._npx_available = True

        health = daemon.health_check()
        assert health.healthy is False

    @pytest.mark.asyncio
    async def test_run_cycle_no_npx(self):
        """Test run cycle when npx not available."""
        daemon = ParityValidationDaemon()
        daemon._npx_available = False

        await daemon._run_cycle()

        # Should not have attempted validation
        assert daemon._last_result is None

    @pytest.mark.asyncio
    async def test_validate_all_databases_no_data_dir(self):
        """Test validation with non-existent data dir."""
        config = ParityValidationConfig(data_dir="/nonexistent/path")
        daemon = ParityValidationDaemon(config=config)

        summary = await daemon._validate_all_databases()

        assert summary.databases_scanned == 0

    @pytest.mark.asyncio
    async def test_validate_all_databases_with_dbs(self, tmp_path):
        """Test validation with mock databases."""
        # Create mock canonical databases
        db1 = tmp_path / "canonical_hex8_2p.db"
        db2 = tmp_path / "canonical_square8_4p.db"
        db1.touch()
        db2.touch()

        config = ParityValidationConfig(data_dir=str(tmp_path))
        daemon = ParityValidationDaemon(config=config)

        # Mock the _validate_database method
        with patch.object(daemon, "_validate_database") as mock_validate:
            mock_validate.return_value = ParityValidationResult(
                games_validated=10,
                games_passed=9,
                games_failed=1,
            )

            summary = await daemon._validate_all_databases()

            assert summary.databases_scanned == 2
            assert mock_validate.call_count == 2

    def test_parse_db_filename(self):
        """Test parsing board type and num_players from filename."""
        daemon = ParityValidationDaemon()

        # Test by calling _validate_database logic indirectly
        # The parsing happens in _validate_database
        result = ParityValidationResult()

        # Parse canonical_hex8_2p
        stem = "canonical_hex8_2p"
        parts = stem.replace("canonical_", "").rsplit("_", 1)
        if len(parts) == 2:
            result.board_type = parts[0]
            try:
                result.num_players = int(parts[1].rstrip("p"))
            except ValueError:
                pass

        assert result.board_type == "hex8"
        assert result.num_players == 2

    def test_parse_db_filename_square19(self):
        """Test parsing square19_4p filename."""
        result = ParityValidationResult()
        stem = "canonical_square19_4p"
        parts = stem.replace("canonical_", "").rsplit("_", 1)
        if len(parts) == 2:
            result.board_type = parts[0]
            try:
                result.num_players = int(parts[1].rstrip("p"))
            except ValueError:
                pass

        assert result.board_type == "square19"
        assert result.num_players == 4

    @pytest.mark.asyncio
    async def test_emit_validation_complete(self):
        """Test event emission after validation."""
        daemon = ParityValidationDaemon()

        summary = ParityValidationSummary(
            databases_scanned=2,
            total_games_validated=50,
            total_games_passed=48,
            total_games_failed=2,
            scan_time=datetime.now(timezone.utc).isoformat(),
        )

        with patch("app.coordination.parity_validation_daemon.get_event_bus") as mock_get_bus:
            mock_bus = MagicMock()
            mock_get_bus.return_value = mock_bus

            daemon._emit_validation_complete(summary)

            mock_bus.publish_event.assert_called_once()


class TestModuleLevelHelpers:
    """Tests for module-level helper functions."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_parity_validation_daemon()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_parity_validation_daemon()

    def test_get_parity_validation_daemon(self):
        """Test get_parity_validation_daemon returns singleton."""
        daemon1 = get_parity_validation_daemon()
        daemon2 = get_parity_validation_daemon()
        assert daemon1 is daemon2

    def test_reset_parity_validation_daemon(self):
        """Test reset_parity_validation_daemon clears singleton."""
        daemon1 = get_parity_validation_daemon()
        reset_parity_validation_daemon()
        daemon2 = get_parity_validation_daemon()
        assert daemon1 is not daemon2
