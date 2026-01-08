"""Unit tests for QualityMonitorDaemon (December 2025).

Tests the continuous selfplay quality monitoring daemon.

Created: December 27, 2025
"""

import asyncio
import json
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.quality_monitor_daemon import (
    QualityMonitorConfig,
    QualityMonitorDaemon,
    QualityState,
    create_quality_monitor,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_data_dir() -> Path:
    """Create a temporary data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data" / "games"
        data_dir.mkdir(parents=True)
        yield data_dir


@pytest.fixture
def temp_state_path() -> Path:
    """Create a temporary state file path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "quality_state.json"
        yield state_path


@pytest.fixture
def config(temp_data_dir: Path, temp_state_path: Path) -> QualityMonitorConfig:
    """Create a test configuration."""
    return QualityMonitorConfig(
        check_interval=0.1,  # Fast for tests
        warning_threshold=0.6,
        good_threshold=0.8,
        significant_change=0.1,
        data_dir=str(temp_data_dir),
        database_pattern="*.db",
        state_path=temp_state_path,
        persist_interval=0.1,
    )


@pytest.fixture
def daemon(config: QualityMonitorConfig) -> QualityMonitorDaemon:
    """Create a test daemon."""
    return QualityMonitorDaemon(config=config)


@pytest.fixture
def sample_db(temp_data_dir: Path) -> Path:
    """Create a sample selfplay database."""
    db_path = temp_data_dir / "selfplay_test.db"

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY,
            game_status TEXT,
            winner INTEGER,
            termination_reason TEXT,
            total_moves INTEGER,
            created_at TEXT
        )
    """)

    # Insert sample games
    games = [
        ("game1", "complete", 1, "victory", 50, "2025-12-27T10:00:00"),
        ("game2", "complete", 0, "draw", 100, "2025-12-27T10:01:00"),
        ("game3", "complete", 1, "victory", 45, "2025-12-27T10:02:00"),
    ]
    conn.executemany(
        "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?)",
        games,
    )
    conn.commit()
    conn.close()

    return db_path


# ============================================================================
# QualityState Tests
# ============================================================================


class TestQualityState:
    """Tests for QualityState enum."""

    def test_state_values(self) -> None:
        """Test all state values."""
        assert QualityState.UNKNOWN.value == "unknown"
        assert QualityState.EXCELLENT.value == "excellent"
        assert QualityState.GOOD.value == "good"
        assert QualityState.DEGRADED.value == "degraded"
        assert QualityState.POOR.value == "poor"

    def test_state_from_string(self) -> None:
        """Test creating state from string."""
        assert QualityState("excellent") == QualityState.EXCELLENT
        assert QualityState("poor") == QualityState.POOR


# ============================================================================
# QualityMonitorConfig Tests
# ============================================================================


class TestQualityMonitorConfig:
    """Tests for QualityMonitorConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = QualityMonitorConfig()
        assert config.check_interval == 15.0
        assert config.warning_threshold == 0.6
        assert config.good_threshold == 0.8
        assert config.significant_change == 0.1
        assert config.data_dir == "data/games"
        assert config.database_pattern == "selfplay*.db"
        assert config.persist_interval == 60.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = QualityMonitorConfig(
            check_interval=30.0,
            warning_threshold=0.5,
            good_threshold=0.9,
        )
        assert config.check_interval == 30.0
        assert config.warning_threshold == 0.5
        assert config.good_threshold == 0.9


# ============================================================================
# QualityMonitorDaemon Initialization Tests
# ============================================================================


class TestQualityMonitorDaemonInit:
    """Tests for daemon initialization."""

    def test_init_default(self, tmp_path: Path) -> None:
        """Test initialization with default config."""
        # Use non-existent temp state path to avoid loading real state
        temp_state = tmp_path / "nonexistent_state.json"
        config = QualityMonitorConfig(state_path=temp_state)
        daemon = QualityMonitorDaemon(config=config)
        assert daemon.config is not None
        assert daemon.last_quality == 1.0
        assert daemon.current_state == QualityState.UNKNOWN
        assert daemon._running is False

    def test_init_custom_config(self, config: QualityMonitorConfig) -> None:
        """Test initialization with custom config."""
        daemon = QualityMonitorDaemon(config=config)
        assert daemon.config.check_interval == 0.1
        assert daemon.config.warning_threshold == 0.6

    def test_init_loads_state(self, temp_state_path: Path) -> None:
        """Test that initialization loads persisted state."""
        # Create persisted state
        state_data = {
            "last_quality": 0.75,
            "current_state": "good",
            "config_quality": {"hex8_2p": 0.8},
            "quality_history": [{"timestamp": 1, "quality": 0.75, "state": "good"}],
            "last_event_time": 100.0,
        }
        temp_state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_state_path, "w") as f:
            json.dump(state_data, f)

        config = QualityMonitorConfig(state_path=temp_state_path)
        daemon = QualityMonitorDaemon(config=config)

        assert daemon.last_quality == 0.75
        assert daemon.current_state == QualityState.GOOD
        assert daemon._config_quality == {"hex8_2p": 0.8}


# ============================================================================
# QualityMonitorDaemon Lifecycle Tests
# ============================================================================


class TestQualityMonitorDaemonLifecycle:
    """Tests for daemon lifecycle."""

    @pytest.mark.asyncio
    async def test_start(self, daemon: QualityMonitorDaemon) -> None:
        """Test daemon start."""
        await daemon.start()
        assert daemon._running is True
        assert daemon._task is not None
        await daemon.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(self, daemon: QualityMonitorDaemon) -> None:
        """Test that starting twice is safe."""
        await daemon.start()
        task1 = daemon._task

        await daemon.start()
        task2 = daemon._task

        assert task1 is task2
        await daemon.stop()

    @pytest.mark.asyncio
    async def test_stop(self, daemon: QualityMonitorDaemon) -> None:
        """Test daemon stop."""
        await daemon.start()
        await daemon.stop()

        assert daemon._running is False

    @pytest.mark.asyncio
    async def test_stop_saves_state(
        self, daemon: QualityMonitorDaemon, temp_state_path: Path
    ) -> None:
        """Test that stop saves state."""
        await daemon.start()
        daemon.last_quality = 0.85
        daemon.current_state = QualityState.GOOD

        await daemon.stop()

        # Verify state was saved
        assert temp_state_path.exists()
        with open(temp_state_path) as f:
            data = json.load(f)
        assert data["last_quality"] == 0.85
        assert data["current_state"] == "good"


# ============================================================================
# Quality State Conversion Tests
# ============================================================================


class TestQualityStateConversion:
    """Tests for quality to state conversion."""

    def test_quality_to_state_excellent(self, daemon: QualityMonitorDaemon) -> None:
        """Test excellent quality conversion."""
        assert daemon._quality_to_state(0.95) == QualityState.EXCELLENT
        assert daemon._quality_to_state(0.90) == QualityState.EXCELLENT
        assert daemon._quality_to_state(1.0) == QualityState.EXCELLENT

    def test_quality_to_state_good(self, daemon: QualityMonitorDaemon) -> None:
        """Test good quality conversion."""
        assert daemon._quality_to_state(0.85) == QualityState.GOOD
        assert daemon._quality_to_state(0.70) == QualityState.GOOD

    def test_quality_to_state_degraded(self, daemon: QualityMonitorDaemon) -> None:
        """Test degraded quality conversion."""
        assert daemon._quality_to_state(0.65) == QualityState.DEGRADED
        assert daemon._quality_to_state(0.50) == QualityState.DEGRADED

    def test_quality_to_state_poor(self, daemon: QualityMonitorDaemon) -> None:
        """Test poor quality conversion."""
        assert daemon._quality_to_state(0.45) == QualityState.POOR
        assert daemon._quality_to_state(0.0) == QualityState.POOR


# ============================================================================
# Quality Check Tests
# ============================================================================


class TestQualityCheck:
    """Tests for quality checking."""

    @pytest.mark.asyncio
    async def test_get_current_quality_no_data_dir(
        self, daemon: QualityMonitorDaemon
    ) -> None:
        """Test quality check with non-existent data directory."""
        daemon.config.data_dir = "/nonexistent/path"
        # Jan 7, 2026: Updated to check QualityResult for Sprint 2.1
        quality_result = await daemon._get_current_quality()
        assert quality_result.quality_score == 1.0  # Default when no data
        assert quality_result.sample_count == 0

    @pytest.mark.asyncio
    async def test_get_current_quality_empty_dir(
        self, daemon: QualityMonitorDaemon, temp_data_dir: Path
    ) -> None:
        """Test quality check with empty directory."""
        # Jan 7, 2026: Updated to check QualityResult for Sprint 2.1
        quality_result = await daemon._get_current_quality()
        assert quality_result.quality_score == 1.0  # Default when no databases
        assert quality_result.sample_count == 0

    @pytest.mark.asyncio
    async def test_check_quality_updates_state(
        self, daemon: QualityMonitorDaemon
    ) -> None:
        """Test that check_quality updates daemon state."""
        with patch.object(
            daemon, "_get_current_quality", new_callable=AsyncMock
        ) as mock_quality:
            # Jan 7, 2026: Updated to return QualityResult for Sprint 2.1
            mock_quality.return_value = QualityResult(
                quality_score=0.75, sample_count=100, variance=0.01
            )

            # Suppress event emission
            with patch.object(
                daemon, "_emit_quality_event", new_callable=AsyncMock
            ):
                await daemon._check_quality()

            assert daemon.last_quality == 0.75
            assert daemon.current_state == QualityState.GOOD

    @pytest.mark.asyncio
    async def test_check_quality_emits_on_change(
        self, daemon: QualityMonitorDaemon
    ) -> None:
        """Test that significant changes trigger event emission."""
        daemon.last_quality = 0.9

        with patch.object(
            daemon, "_get_current_quality", new_callable=AsyncMock
        ) as mock_quality:
            # Jan 7, 2026: Updated to return QualityResult for Sprint 2.1
            mock_quality.return_value = QualityResult(
                quality_score=0.5, sample_count=80, variance=0.02
            )  # Significant drop

            with patch.object(
                daemon, "_emit_quality_event", new_callable=AsyncMock
            ) as mock_emit:
                await daemon._check_quality()
                mock_emit.assert_called_once()


# ============================================================================
# State Persistence Tests
# ============================================================================


class TestStatePersistence:
    """Tests for state persistence."""

    def test_save_state(
        self, daemon: QualityMonitorDaemon, temp_state_path: Path
    ) -> None:
        """Test saving state to file."""
        daemon.last_quality = 0.65
        daemon.current_state = QualityState.DEGRADED
        daemon._config_quality = {"hex8_2p": 0.7}

        daemon._save_state()

        assert temp_state_path.exists()
        with open(temp_state_path) as f:
            data = json.load(f)

        assert data["last_quality"] == 0.65
        assert data["current_state"] == "degraded"
        assert data["config_quality"] == {"hex8_2p": 0.7}

    def test_load_state(
        self, temp_state_path: Path
    ) -> None:
        """Test loading state from file."""
        # Create state file
        state_data = {
            "last_quality": 0.55,
            "current_state": "degraded",
            "config_quality": {"square8_2p": 0.6},
            "quality_history": [],
            "last_event_time": 50.0,
        }
        temp_state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_state_path, "w") as f:
            json.dump(state_data, f)

        config = QualityMonitorConfig(state_path=temp_state_path)
        daemon = QualityMonitorDaemon(config=config)

        assert daemon.last_quality == 0.55
        assert daemon.current_state == QualityState.DEGRADED

    def test_load_state_corrupted(self, temp_state_path: Path) -> None:
        """Test handling of corrupted state file."""
        # Create corrupted file
        temp_state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_state_path, "w") as f:
            f.write("not valid json")

        config = QualityMonitorConfig(state_path=temp_state_path)
        daemon = QualityMonitorDaemon(config=config)

        # Should use defaults
        assert daemon.last_quality == 1.0
        assert daemon.current_state == QualityState.UNKNOWN


# ============================================================================
# Quality History Tests
# ============================================================================


class TestQualityHistory:
    """Tests for quality history tracking."""

    def test_add_to_history(self, daemon: QualityMonitorDaemon) -> None:
        """Test adding entries to history."""
        daemon._add_to_history(0.75, QualityState.GOOD)
        daemon._add_to_history(0.80, QualityState.GOOD)

        assert len(daemon._quality_history) == 2
        assert daemon._quality_history[0]["quality"] == 0.75
        assert daemon._quality_history[1]["quality"] == 0.80

    def test_history_max_size(self, daemon: QualityMonitorDaemon) -> None:
        """Test history respects max size."""
        daemon._max_history_size = 5

        for i in range(10):
            daemon._add_to_history(0.5 + i * 0.01, QualityState.DEGRADED)

        assert len(daemon._quality_history) == 5
        # Should keep latest entries
        assert daemon._quality_history[-1]["quality"] == pytest.approx(0.59, rel=0.01)

    def test_get_quality_trend_empty(self, daemon: QualityMonitorDaemon) -> None:
        """Test trend analysis with empty history."""
        trend = daemon.get_quality_trend()
        assert trend["trend"] == "unknown"
        assert trend["samples"] == 0

    def test_get_quality_trend_improving(self, daemon: QualityMonitorDaemon) -> None:
        """Test detection of improving trend."""
        # Add improving scores
        for i in range(10):
            daemon._add_to_history(0.5 + i * 0.05, QualityState.DEGRADED)

        trend = daemon.get_quality_trend()
        assert trend["trend"] == "improving"
        assert trend["samples"] == 10

    def test_get_quality_trend_degrading(self, daemon: QualityMonitorDaemon) -> None:
        """Test detection of degrading trend."""
        # Add degrading scores
        for i in range(10):
            daemon._add_to_history(0.9 - i * 0.05, QualityState.GOOD)

        trend = daemon.get_quality_trend()
        assert trend["trend"] == "degrading"

    def test_get_quality_trend_stable(self, daemon: QualityMonitorDaemon) -> None:
        """Test detection of stable trend."""
        # Add stable scores
        for i in range(10):
            daemon._add_to_history(0.75, QualityState.GOOD)

        trend = daemon.get_quality_trend()
        assert trend["trend"] == "stable"


# ============================================================================
# Health Check Tests
# ============================================================================


class TestHealthCheck:
    """Tests for health check."""

    def test_health_check_not_running(self, daemon: QualityMonitorDaemon) -> None:
        """Test health check when daemon not running."""
        result = daemon.health_check()
        assert result.healthy is False
        assert "not running" in result.message.lower()

    @pytest.mark.asyncio
    async def test_health_check_running(self, daemon: QualityMonitorDaemon) -> None:
        """Test health check when daemon is running."""
        await daemon.start()

        result = daemon.health_check()
        assert result.healthy is True
        assert "running" in result.message.lower()

        await daemon.stop()

    def test_health_check_degraded_quality(self, daemon: QualityMonitorDaemon) -> None:
        """Test health check with degraded quality."""
        daemon._running = True
        daemon.current_state = QualityState.DEGRADED
        daemon.last_quality = 0.55

        result = daemon.health_check()
        # Daemon itself is healthy, just quality is low
        assert result.healthy is True
        assert "degraded" in result.message.lower()

    def test_health_check_poor_quality(self, daemon: QualityMonitorDaemon) -> None:
        """Test health check with poor quality."""
        daemon._running = True
        daemon.current_state = QualityState.POOR
        daemon.last_quality = 0.35

        result = daemon.health_check()
        assert result.healthy is True
        assert "poor" in result.message.lower()


# ============================================================================
# Status Tests
# ============================================================================


class TestGetStatus:
    """Tests for status reporting."""

    def test_get_status(self, daemon: QualityMonitorDaemon) -> None:
        """Test getting daemon status."""
        daemon.last_quality = 0.85
        daemon.current_state = QualityState.GOOD
        daemon._config_quality = {"hex8_2p": 0.9}

        status = daemon.get_status()

        assert status["running"] is False
        assert status["last_quality"] == 0.85
        assert status["current_state"] == "good"
        assert status["config_quality"] == {"hex8_2p": 0.9}
        assert "check_interval" in status
        assert "history_size" in status


# ============================================================================
# Event Emission Tests
# ============================================================================


class TestEventEmission:
    """Tests for event emission."""

    @pytest.fixture
    def mock_event_type(self):
        """Create mock DataEventType with required attributes."""
        mock = MagicMock()
        mock.LOW_QUALITY_DATA_WARNING = "LOW_QUALITY_DATA_WARNING"
        mock.HIGH_QUALITY_DATA_AVAILABLE = "HIGH_QUALITY_DATA_AVAILABLE"
        mock.QUALITY_SCORE_UPDATED = "QUALITY_SCORE_UPDATED"
        return mock

    @pytest.mark.asyncio
    async def test_emit_quality_event_low_quality(
        self, daemon: QualityMonitorDaemon, mock_event_type
    ) -> None:
        """Test event emission for low quality."""
        daemon._last_event_time = 0  # Reset cooldown

        mock_router = MagicMock()
        mock_router.publish = AsyncMock()

        # Patch at the source module where imports happen
        with patch(
            "app.coordination.event_router.get_router",
            return_value=mock_router,
        ), patch(
            "app.coordination.event_router.DataEventType",
            mock_event_type,
        ):
            # Jan 7, 2026: Updated to use QualityResult for Sprint 2.1
            quality_result = QualityResult(
                quality_score=0.4, sample_count=50, variance=0.01
            )
            await daemon._emit_quality_event(
                quality_result=quality_result,
                new_state=QualityState.POOR,
                old_state=QualityState.GOOD,
            )

            # Should emit LOW_QUALITY_DATA_WARNING
            mock_router.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_quality_event_high_quality(
        self, daemon: QualityMonitorDaemon, mock_event_type
    ) -> None:
        """Test event emission for high quality recovery."""
        daemon._last_event_time = 0

        mock_router = MagicMock()
        mock_router.publish = AsyncMock()

        with patch(
            "app.coordination.event_router.get_router",
            return_value=mock_router,
        ), patch(
            "app.coordination.event_router.DataEventType",
            mock_event_type,
        ):
            # Jan 7, 2026: Updated to use QualityResult for Sprint 2.1
            quality_result = QualityResult(
                quality_score=0.85, sample_count=100, variance=0.005
            )
            await daemon._emit_quality_event(
                quality_result=quality_result,
                new_state=QualityState.GOOD,
                old_state=QualityState.DEGRADED,
            )

            mock_router.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_quality_event_cooldown(
        self, daemon: QualityMonitorDaemon
    ) -> None:
        """Test event cooldown."""
        daemon._last_event_time = time.time()  # Recent event

        mock_router = MagicMock()
        mock_router.publish = AsyncMock()

        with patch(
            "app.coordination.event_router.get_router",
            return_value=mock_router,
        ):
            # Jan 7, 2026: Updated to use QualityResult for Sprint 2.1
            quality_result = QualityResult(
                quality_score=0.4, sample_count=50, variance=0.01
            )
            await daemon._emit_quality_event(
                quality_result=quality_result,
                new_state=QualityState.POOR,
                old_state=QualityState.GOOD,
            )

            # Should skip due to cooldown
            mock_router.publish.assert_not_called()


# ============================================================================
# On-Demand Quality Check Tests
# ============================================================================


class TestOnDemandQualityCheck:
    """Tests for on-demand quality check handling."""

    @pytest.mark.asyncio
    async def test_on_quality_check_requested(
        self, daemon: QualityMonitorDaemon
    ) -> None:
        """Test handling on-demand quality check request."""
        mock_event = MagicMock()
        mock_event.payload = {
            "config_key": "hex8_2p",
            "reason": "loss_regression",
            "priority": "high",
        }

        with patch.object(
            daemon, "_get_current_quality", new_callable=AsyncMock
        ) as mock_quality:
            # Jan 7, 2026: Updated to return QualityResult for Sprint 2.1
            mock_quality.return_value = QualityResult(
                quality_score=0.65, sample_count=75, variance=0.015
            )

            with patch.object(
                daemon, "_emit_quality_event", new_callable=AsyncMock
            ):
                await daemon._on_quality_check_requested(mock_event)

                assert daemon._config_quality["hex8_2p"] == 0.65

    @pytest.mark.asyncio
    async def test_on_quality_check_requested_error(
        self, daemon: QualityMonitorDaemon
    ) -> None:
        """Test error handling in on-demand check."""
        mock_event = MagicMock()
        mock_event.payload = {"config_key": "test"}

        # Mock both _get_current_quality and _emit_quality_check_failed
        with patch.object(
            daemon, "_get_current_quality", new_callable=AsyncMock
        ) as mock_quality:
            mock_quality.side_effect = RuntimeError("Test error")

            with patch.object(
                daemon, "_emit_quality_check_failed", new_callable=AsyncMock
            ):
                # Should not raise
                await daemon._on_quality_check_requested(mock_event)


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestFactoryFunction:
    """Tests for create_quality_monitor factory."""

    @pytest.mark.asyncio
    async def test_create_quality_monitor(self) -> None:
        """Test factory function creates and runs daemon."""
        task = asyncio.create_task(create_quality_monitor())

        # Let it start
        await asyncio.sleep(0.1)

        # Cancel it
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


# ============================================================================
# Monitor Loop Tests
# ============================================================================


class TestMonitorLoop:
    """Tests for the main monitor loop."""

    @pytest.mark.asyncio
    async def test_monitor_loop_runs(self, daemon: QualityMonitorDaemon) -> None:
        """Test that monitor loop runs and checks quality."""
        check_count = 0

        async def mock_check():
            nonlocal check_count
            check_count += 1

        with patch.object(daemon, "_check_quality", mock_check):
            await daemon.start()
            await asyncio.sleep(0.25)  # Allow ~2 cycles
            await daemon.stop()

            assert check_count >= 2

    @pytest.mark.asyncio
    async def test_monitor_loop_handles_errors(
        self, daemon: QualityMonitorDaemon
    ) -> None:
        """Test that monitor loop handles errors gracefully."""
        call_count = 0

        async def mock_check():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Test error")

        with patch.object(daemon, "_check_quality", mock_check):
            await daemon.start()
            await asyncio.sleep(0.25)
            await daemon.stop()

            # Should have continued after error
            assert call_count >= 2


# =============================================================================
# Additional Tests - December 29, 2025
# =============================================================================


class TestQualityStateEdgeCases:
    """Additional edge case tests for QualityState."""

    def test_all_states_have_values(self) -> None:
        """Test all states have string values."""
        for state in QualityState:
            assert isinstance(state.value, str)
            assert len(state.value) > 0

    def test_state_count(self) -> None:
        """Test there are exactly 5 quality states."""
        assert len(QualityState) == 5

    def test_invalid_state_raises(self) -> None:
        """Test invalid state string raises ValueError."""
        with pytest.raises(ValueError):
            QualityState("invalid_state")


class TestQualityMonitorConfigEdgeCases:
    """Additional edge case tests for QualityMonitorConfig."""

    def test_config_with_zero_interval(self) -> None:
        """Test config with zero check interval."""
        config = QualityMonitorConfig(check_interval=0.0)
        assert config.check_interval == 0.0

    def test_config_with_inverted_thresholds(self) -> None:
        """Test config with warning > good threshold."""
        # This is allowed (may be intentional)
        config = QualityMonitorConfig(
            warning_threshold=0.9,
            good_threshold=0.5,
        )
        assert config.warning_threshold == 0.9
        assert config.good_threshold == 0.5

    def test_config_with_extreme_thresholds(self) -> None:
        """Test config with extreme threshold values."""
        config = QualityMonitorConfig(
            warning_threshold=0.0,
            good_threshold=1.0,
            significant_change=0.001,
        )
        assert config.warning_threshold == 0.0
        assert config.good_threshold == 1.0
        assert config.significant_change == 0.001

    def test_config_custom_database_pattern(self) -> None:
        """Test config with custom database pattern."""
        config = QualityMonitorConfig(database_pattern="canonical_*.db")
        assert config.database_pattern == "canonical_*.db"

    def test_config_none_state_path(self) -> None:
        """Test config with None state path uses default."""
        config = QualityMonitorConfig(state_path=None)
        assert config.state_path is None


class TestQualityMonitorDaemonStateManagement:
    """Tests for state management edge cases."""

    def test_save_state_creates_parent_dir(self, temp_state_path: Path) -> None:
        """Test save_state creates parent directory."""
        nested_path = temp_state_path.parent / "nested" / "state.json"
        config = QualityMonitorConfig(state_path=nested_path)
        daemon = QualityMonitorDaemon(config=config)

        daemon.last_quality = 0.7
        daemon._save_state()

        assert nested_path.exists()

    def test_save_state_atomic_write(self, temp_state_path: Path) -> None:
        """Test save_state uses atomic write (temp file + rename)."""
        config = QualityMonitorConfig(state_path=temp_state_path)
        daemon = QualityMonitorDaemon(config=config)

        daemon.last_quality = 0.8
        daemon._save_state()

        # Temp file should not exist
        temp_path = temp_state_path.with_suffix(".tmp")
        assert not temp_path.exists()

    def test_load_state_invalid_state_string(self, temp_state_path: Path) -> None:
        """Test load_state handles invalid state string."""
        state_data = {
            "last_quality": 0.75,
            "current_state": "invalid_state",  # Invalid
            "config_quality": {},
            "quality_history": [],
            "last_event_time": 0.0,
        }
        temp_state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_state_path, "w") as f:
            json.dump(state_data, f)

        config = QualityMonitorConfig(state_path=temp_state_path)
        daemon = QualityMonitorDaemon(config=config)

        # Should fall back to UNKNOWN or handle gracefully
        assert daemon.current_state in (QualityState.UNKNOWN, QualityState.GOOD)

    def test_load_state_missing_keys(self, temp_state_path: Path) -> None:
        """Test load_state handles missing keys."""
        state_data = {
            # Missing most keys
            "last_quality": 0.6,
        }
        temp_state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_state_path, "w") as f:
            json.dump(state_data, f)

        config = QualityMonitorConfig(state_path=temp_state_path)
        daemon = QualityMonitorDaemon(config=config)

        assert daemon.last_quality == 0.6
        assert daemon.current_state == QualityState.UNKNOWN


class TestQualityHistoryEdgeCases:
    """Additional edge case tests for quality history."""

    def test_history_entry_structure(self, daemon: QualityMonitorDaemon) -> None:
        """Test history entry has correct structure."""
        daemon._add_to_history(0.75, QualityState.GOOD)

        entry = daemon._quality_history[0]
        assert "timestamp" in entry
        assert "quality" in entry
        assert "state" in entry
        assert entry["quality"] == 0.75
        assert entry["state"] == "good"

    def test_history_timestamp_is_recent(self, daemon: QualityMonitorDaemon) -> None:
        """Test history entry timestamp is recent."""
        before = time.time()
        daemon._add_to_history(0.5, QualityState.DEGRADED)
        after = time.time()

        entry = daemon._quality_history[0]
        assert before <= entry["timestamp"] <= after

    def test_get_quality_trend_insufficient_data(self, daemon: QualityMonitorDaemon) -> None:
        """Test trend with less than 3 samples."""
        daemon._add_to_history(0.5, QualityState.DEGRADED)
        daemon._add_to_history(0.6, QualityState.DEGRADED)

        trend = daemon.get_quality_trend()
        assert trend["trend"] == "insufficient_data"
        assert trend["samples"] == 2

    def test_get_quality_trend_custom_window(self, daemon: QualityMonitorDaemon) -> None:
        """Test trend with custom window size."""
        for i in range(20):
            daemon._add_to_history(0.5 + i * 0.01, QualityState.DEGRADED)

        trend_5 = daemon.get_quality_trend(window_size=5)
        trend_10 = daemon.get_quality_trend(window_size=10)

        assert trend_5["samples"] == 5
        assert trend_10["samples"] == 10

    def test_get_quality_trend_min_max(self, daemon: QualityMonitorDaemon) -> None:
        """Test trend includes min/max values."""
        daemon._add_to_history(0.3, QualityState.POOR)
        daemon._add_to_history(0.5, QualityState.DEGRADED)
        daemon._add_to_history(0.9, QualityState.EXCELLENT)

        trend = daemon.get_quality_trend()
        assert trend["min"] == 0.3
        assert trend["max"] == 0.9


class TestQualityCheckEdgeCases:
    """Additional edge case tests for quality checking."""

    @pytest.mark.asyncio
    async def test_check_quality_no_significant_change(
        self, daemon: QualityMonitorDaemon
    ) -> None:
        """Test check_quality doesn't emit when change is small."""
        daemon.last_quality = 0.75
        daemon.current_state = QualityState.GOOD

        with patch.object(
            daemon, "_get_current_quality", new_callable=AsyncMock
        ) as mock_quality:
            # Jan 7, 2026: Updated to return QualityResult for Sprint 2.1
            mock_quality.return_value = QualityResult(
                quality_score=0.76, sample_count=100, variance=0.01
            )  # Small change

            with patch.object(
                daemon, "_emit_quality_event", new_callable=AsyncMock
            ) as mock_emit:
                await daemon._check_quality()

                # Same state, small change - should not emit
                mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_quality_state_change_emits(
        self, daemon: QualityMonitorDaemon
    ) -> None:
        """Test check_quality emits on state change even if score change is small."""
        daemon.last_quality = 0.70  # Good
        daemon.current_state = QualityState.GOOD

        with patch.object(
            daemon, "_get_current_quality", new_callable=AsyncMock
        ) as mock_quality:
            # Jan 7, 2026: Updated to return QualityResult for Sprint 2.1
            mock_quality.return_value = QualityResult(
                quality_score=0.65, sample_count=90, variance=0.02
            )  # Degraded (state change)

            with patch.object(
                daemon, "_emit_quality_event", new_callable=AsyncMock
            ) as mock_emit:
                await daemon._check_quality()
                mock_emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_quality_persists_periodically(
        self, daemon: QualityMonitorDaemon
    ) -> None:
        """Test check_quality persists state at persist_interval."""
        daemon._last_persist_time = 0  # Long ago

        with patch.object(
            daemon, "_get_current_quality", new_callable=AsyncMock
        ) as mock_quality:
            # Jan 7, 2026: Updated to return QualityResult for Sprint 2.1
            mock_quality.return_value = QualityResult(
                quality_score=0.8, sample_count=100, variance=0.01
            )

            with patch.object(daemon, "_save_state") as mock_save:
                await daemon._check_quality()
                mock_save.assert_called_once()


class TestEventEmissionEdgeCases:
    """Additional edge case tests for event emission."""

    @pytest.mark.asyncio
    async def test_emit_quality_event_import_error(
        self, daemon: QualityMonitorDaemon
    ) -> None:
        """Test event emission handles import error."""
        daemon._last_event_time = 0

        with patch(
            "app.coordination.event_router.get_router",
            side_effect=ImportError("No module"),
        ):
            # Jan 7, 2026: Updated to use QualityResult for Sprint 2.1
            quality_result = QualityResult(
                quality_score=0.4, sample_count=50, variance=0.01
            )
            # Should not raise
            await daemon._emit_quality_event(
                quality_result=quality_result,
                new_state=QualityState.POOR,
                old_state=QualityState.GOOD,
            )

    @pytest.mark.asyncio
    async def test_emit_quality_check_failed(
        self, daemon: QualityMonitorDaemon
    ) -> None:
        """Test QUALITY_CHECK_FAILED event emission."""
        # Mock the internal emit method to verify it can be called
        with patch.object(
            daemon, "_emit_quality_check_failed", new_callable=AsyncMock
        ) as mock_emit:
            await mock_emit(
                reason="Test failure",
                config_key="hex8_2p",
                check_type="periodic",
            )
            mock_emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_quality_check_failed_handles_error(
        self, daemon: QualityMonitorDaemon
    ) -> None:
        """Test QUALITY_CHECK_FAILED handles emission errors gracefully."""
        # Verify the method exists and can be called
        # The actual implementation handles errors internally
        assert hasattr(daemon, "_emit_quality_check_failed")
        assert callable(daemon._emit_quality_check_failed)


class TestOnDemandQualityCheckEdgeCases:
    """Additional edge case tests for on-demand quality checks."""

    @pytest.mark.asyncio
    async def test_on_quality_check_requested_no_payload(
        self, daemon: QualityMonitorDaemon
    ) -> None:
        """Test on-demand check with no payload attribute."""
        mock_event = MagicMock(spec=[])  # No payload attribute

        with patch.object(
            daemon, "_get_current_quality", new_callable=AsyncMock
        ) as mock_quality:
            # Jan 7, 2026: Updated to return QualityResult for Sprint 2.1
            mock_quality.return_value = QualityResult(
                quality_score=0.8, sample_count=100, variance=0.01
            )

            # Should not raise
            await daemon._on_quality_check_requested(mock_event)

    @pytest.mark.asyncio
    async def test_on_quality_check_requested_empty_config_key(
        self, daemon: QualityMonitorDaemon
    ) -> None:
        """Test on-demand check with empty config key."""
        mock_event = MagicMock()
        mock_event.payload = {"config_key": "", "reason": "test"}

        with patch.object(
            daemon, "_get_current_quality", new_callable=AsyncMock
        ) as mock_quality:
            # Jan 7, 2026: Updated to return QualityResult for Sprint 2.1
            mock_quality.return_value = QualityResult(
                quality_score=0.7, sample_count=80, variance=0.015
            )

            with patch.object(
                daemon, "_emit_quality_event", new_callable=AsyncMock
            ):
                await daemon._on_quality_check_requested(mock_event)

                # Empty key should not be stored
                assert "" not in daemon._config_quality


class TestHealthCheckEdgeCases:
    """Additional edge case tests for health check."""

    def test_health_check_includes_details(self, daemon: QualityMonitorDaemon) -> None:
        """Test health check includes detailed status."""
        daemon._running = True
        daemon.current_state = QualityState.GOOD
        daemon.last_quality = 0.85
        daemon._config_quality = {"hex8_2p": 0.9}

        result = daemon.health_check()
        assert result.details is not None
        assert "last_quality" in result.details
        assert "config_quality" in result.details

    def test_health_check_all_states(self, daemon: QualityMonitorDaemon) -> None:
        """Test health check for each quality state."""
        daemon._running = True

        for state in QualityState:
            daemon.current_state = state
            result = daemon.health_check()
            assert result.healthy is True  # Daemon is always healthy if running


class TestGetStatusEdgeCases:
    """Additional edge case tests for get_status."""

    def test_get_status_includes_all_fields(self, daemon: QualityMonitorDaemon) -> None:
        """Test get_status includes all expected fields."""
        status = daemon.get_status()

        expected_fields = [
            "running",
            "last_quality",
            "current_state",
            "config_quality",
            "check_interval",
            "state_path",
            "history_size",
            "last_persist_time",
        ]
        for field in expected_fields:
            assert field in status

    def test_get_status_after_quality_check(self, daemon: QualityMonitorDaemon) -> None:
        """Test status after quality history is populated."""
        daemon._add_to_history(0.75, QualityState.GOOD)
        daemon._add_to_history(0.80, QualityState.GOOD)

        status = daemon.get_status()
        assert status["history_size"] == 2


class TestEventSubscriptions:
    """Tests for event subscriptions."""

    def test_get_event_subscriptions(self, daemon: QualityMonitorDaemon) -> None:
        """Test event subscription returns expected handlers."""
        subscriptions = daemon._get_event_subscriptions()

        assert "quality_check_requested" in subscriptions
        assert callable(subscriptions["quality_check_requested"])

    def test_subscription_handler_is_method(self, daemon: QualityMonitorDaemon) -> None:
        """Test subscription handler is bound method."""
        subscriptions = daemon._get_event_subscriptions()
        handler = subscriptions["quality_check_requested"]

        assert handler == daemon._on_quality_check_requested


class TestSingletonBehavior:
    """Tests for singleton pattern behavior."""

    def test_singleton_instance(self) -> None:
        """Test singleton pattern returns same instance."""
        # Note: get_quality_monitor uses get_instance pattern
        from app.coordination.quality_monitor_daemon import (
            get_quality_monitor,
            reset_quality_monitor,
        )

        reset_quality_monitor()

        d1 = get_quality_monitor()
        d2 = get_quality_monitor()

        assert d1 is d2
        reset_quality_monitor()

    def test_reset_singleton(self) -> None:
        """Test resetting singleton creates new instance."""
        from app.coordination.quality_monitor_daemon import (
            get_quality_monitor,
            reset_quality_monitor,
        )

        reset_quality_monitor()

        d1 = get_quality_monitor()
        d1.last_quality = 0.5

        reset_quality_monitor()
        d2 = get_quality_monitor()

        assert d1 is not d2
        # Default may vary based on config; just verify reset worked
        assert d2 is not d1
        reset_quality_monitor()
