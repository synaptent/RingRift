"""Tests for QualityMonitorDaemon.

Covers:
- QualityState enum
- QualityMonitorConfig dataclass
- QualityMonitorDaemon lifecycle (start/stop)
- State persistence (load/save)
- Quality checking and state transitions
- Event emission
- Quality trend analysis
- Health check implementation
- On-demand quality check requests

December 2025: Added comprehensive tests for quality monitoring daemon.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.quality_monitor_daemon import (
    QualityMonitorConfig,
    QualityMonitorDaemon,
    QualityState,
    create_quality_monitor,
)


# =============================================================================
# QualityState Enum Tests
# =============================================================================


class TestQualityState:
    """Tests for QualityState enum."""

    def test_state_values(self):
        """Test that all states have expected string values."""
        assert QualityState.UNKNOWN.value == "unknown"
        assert QualityState.EXCELLENT.value == "excellent"
        assert QualityState.GOOD.value == "good"
        assert QualityState.DEGRADED.value == "degraded"
        assert QualityState.POOR.value == "poor"

    def test_state_creation_from_string(self):
        """Test creating state from string value."""
        assert QualityState("excellent") == QualityState.EXCELLENT
        assert QualityState("poor") == QualityState.POOR

    def test_all_states_defined(self):
        """Test that exactly 5 states are defined."""
        assert len(QualityState) == 5


# =============================================================================
# QualityMonitorConfig Tests
# =============================================================================


class TestQualityMonitorConfig:
    """Tests for QualityMonitorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = QualityMonitorConfig()

        assert config.check_interval == 15.0
        assert config.warning_threshold == 0.6
        assert config.good_threshold == 0.8
        assert config.significant_change == 0.1
        assert config.data_dir == "data/games"
        assert config.database_pattern == "selfplay*.db"
        assert config.state_path is None
        assert config.persist_interval == 60.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = QualityMonitorConfig(
            check_interval=30.0,
            warning_threshold=0.5,
            good_threshold=0.9,
            significant_change=0.05,
            data_dir="/custom/path",
            database_pattern="*.db",
            state_path=Path("/tmp/test_state.json"),
            persist_interval=120.0,
        )

        assert config.check_interval == 30.0
        assert config.warning_threshold == 0.5
        assert config.good_threshold == 0.9
        assert config.data_dir == "/custom/path"
        assert config.state_path == Path("/tmp/test_state.json")


# =============================================================================
# QualityMonitorDaemon Initialization Tests
# =============================================================================


class TestQualityMonitorDaemonInit:
    """Tests for QualityMonitorDaemon initialization."""

    def test_default_initialization(self):
        """Test daemon initializes with default config."""
        daemon = QualityMonitorDaemon()

        assert daemon.config is not None
        assert daemon._running is False
        assert daemon._task is None
        assert daemon.last_quality == 1.0
        assert daemon.current_state == QualityState.UNKNOWN
        assert daemon._subscribed is False

    def test_custom_config(self):
        """Test daemon initializes with custom config."""
        config = QualityMonitorConfig(check_interval=45.0)
        daemon = QualityMonitorDaemon(config=config)

        assert daemon.config.check_interval == 45.0

    def test_state_path_default(self):
        """Test default state path is set."""
        daemon = QualityMonitorDaemon()

        assert daemon._state_path.name == "quality_monitor_state.json"


# =============================================================================
# Quality State Conversion Tests
# =============================================================================


class TestQualityToState:
    """Tests for _quality_to_state method."""

    def test_excellent_threshold(self):
        """Test excellent state threshold (>= 0.9)."""
        daemon = QualityMonitorDaemon()

        assert daemon._quality_to_state(0.9) == QualityState.EXCELLENT
        assert daemon._quality_to_state(0.95) == QualityState.EXCELLENT
        assert daemon._quality_to_state(1.0) == QualityState.EXCELLENT

    def test_good_threshold(self):
        """Test good state threshold (>= 0.7, < 0.9)."""
        daemon = QualityMonitorDaemon()

        assert daemon._quality_to_state(0.7) == QualityState.GOOD
        assert daemon._quality_to_state(0.8) == QualityState.GOOD
        assert daemon._quality_to_state(0.89) == QualityState.GOOD

    def test_degraded_threshold(self):
        """Test degraded state threshold (>= 0.5, < 0.7)."""
        daemon = QualityMonitorDaemon()

        assert daemon._quality_to_state(0.5) == QualityState.DEGRADED
        assert daemon._quality_to_state(0.6) == QualityState.DEGRADED
        assert daemon._quality_to_state(0.69) == QualityState.DEGRADED

    def test_poor_threshold(self):
        """Test poor state threshold (< 0.5)."""
        daemon = QualityMonitorDaemon()

        assert daemon._quality_to_state(0.0) == QualityState.POOR
        assert daemon._quality_to_state(0.3) == QualityState.POOR
        assert daemon._quality_to_state(0.49) == QualityState.POOR


# =============================================================================
# State Persistence Tests
# =============================================================================


class TestStatePersistence:
    """Tests for state save/load functionality."""

    def test_save_state(self):
        """Test saving state to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            config = QualityMonitorConfig(state_path=state_path)
            daemon = QualityMonitorDaemon(config=config)

            daemon.last_quality = 0.75
            daemon.current_state = QualityState.GOOD
            daemon._config_quality = {"hex8_2p": 0.8}
            daemon._quality_history = [
                {"timestamp": 100, "quality": 0.7, "state": "good"},
            ]

            daemon._save_state()

            assert state_path.exists()

            with open(state_path) as f:
                data = json.load(f)

            assert data["last_quality"] == 0.75
            assert data["current_state"] == "good"
            assert data["config_quality"]["hex8_2p"] == 0.8
            assert len(data["quality_history"]) == 1

    def test_load_state(self):
        """Test loading state from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"

            # Create state file
            state_data = {
                "last_quality": 0.65,
                "current_state": "degraded",
                "config_quality": {"square8_4p": 0.6},
                "quality_history": [
                    {"timestamp": 200, "quality": 0.65, "state": "degraded"},
                ],
                "last_event_time": 12345.0,
            }
            with open(state_path, "w") as f:
                json.dump(state_data, f)

            config = QualityMonitorConfig(state_path=state_path)
            daemon = QualityMonitorDaemon(config=config)

            assert daemon.last_quality == 0.65
            assert daemon.current_state == QualityState.DEGRADED
            assert daemon._config_quality == {"square8_4p": 0.6}
            assert len(daemon._quality_history) == 1

    def test_load_nonexistent_state(self):
        """Test loading when state file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "nonexistent.json"
            config = QualityMonitorConfig(state_path=state_path)
            daemon = QualityMonitorDaemon(config=config)

            # Should use defaults
            assert daemon.last_quality == 1.0
            assert daemon.current_state == QualityState.UNKNOWN

    def test_load_corrupted_state(self):
        """Test loading corrupted state file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "corrupt.json"

            with open(state_path, "w") as f:
                f.write("not valid json{")

            config = QualityMonitorConfig(state_path=state_path)
            daemon = QualityMonitorDaemon(config=config)

            # Should use defaults after failed load
            assert daemon.last_quality == 1.0


# =============================================================================
# Quality History Tests
# =============================================================================


class TestQualityHistory:
    """Tests for quality history tracking."""

    def test_add_to_history(self):
        """Test adding entries to quality history."""
        daemon = QualityMonitorDaemon()

        daemon._add_to_history(0.8, QualityState.GOOD)
        daemon._add_to_history(0.7, QualityState.GOOD)

        assert len(daemon._quality_history) == 2
        assert daemon._quality_history[0]["quality"] == 0.8
        assert daemon._quality_history[1]["quality"] == 0.7

    def test_history_trimming(self):
        """Test that history is trimmed to max size."""
        daemon = QualityMonitorDaemon()
        daemon._max_history_size = 5

        # Add more than max
        for i in range(10):
            daemon._add_to_history(0.5 + i * 0.01, QualityState.DEGRADED)

        assert len(daemon._quality_history) == 5
        # Should have the last 5 entries
        assert daemon._quality_history[0]["quality"] == 0.55

    def test_get_quality_trend_empty(self):
        """Test trend analysis with empty history."""
        daemon = QualityMonitorDaemon()

        trend = daemon.get_quality_trend()

        assert trend["trend"] == "unknown"
        assert trend["samples"] == 0

    def test_get_quality_trend_improving(self):
        """Test trend analysis detecting improvement."""
        daemon = QualityMonitorDaemon()

        # Add improving quality scores
        scores = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
        for score in scores:
            daemon._add_to_history(score, daemon._quality_to_state(score))

        trend = daemon.get_quality_trend()

        assert trend["trend"] == "improving"
        assert trend["samples"] == 8

    def test_get_quality_trend_degrading(self):
        """Test trend analysis detecting degradation."""
        daemon = QualityMonitorDaemon()

        # Add degrading quality scores
        scores = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]
        for score in scores:
            daemon._add_to_history(score, daemon._quality_to_state(score))

        trend = daemon.get_quality_trend()

        assert trend["trend"] == "degrading"
        assert trend["samples"] == 8

    def test_get_quality_trend_stable(self):
        """Test trend analysis detecting stable quality."""
        daemon = QualityMonitorDaemon()

        # Add stable quality scores (small variation)
        scores = [0.75, 0.76, 0.74, 0.75, 0.76, 0.75, 0.74, 0.75]
        for score in scores:
            daemon._add_to_history(score, daemon._quality_to_state(score))

        trend = daemon.get_quality_trend()

        assert trend["trend"] == "stable"

    def test_get_quality_trend_insufficient_data(self):
        """Test trend analysis with insufficient data."""
        daemon = QualityMonitorDaemon()

        daemon._add_to_history(0.8, QualityState.GOOD)
        daemon._add_to_history(0.75, QualityState.GOOD)

        trend = daemon.get_quality_trend()

        assert trend["trend"] == "insufficient_data"
        assert trend["samples"] == 2


# =============================================================================
# Daemon Lifecycle Tests
# =============================================================================


class TestDaemonLifecycle:
    """Tests for daemon start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_creates_task(self):
        """Test that start creates monitoring task."""
        daemon = QualityMonitorDaemon()

        with patch.object(daemon, "_subscribe_to_events"):
            await daemon.start()

        assert daemon._running is True
        assert daemon._task is not None

        await daemon.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        """Test that starting twice is safe."""
        daemon = QualityMonitorDaemon()

        with patch.object(daemon, "_subscribe_to_events"):
            await daemon.start()
            await daemon.start()  # Should not create second task

        assert daemon._running is True

        await daemon.stop()

    @pytest.mark.asyncio
    async def test_stop_saves_state(self):
        """Test that stop saves state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            config = QualityMonitorConfig(state_path=state_path)
            daemon = QualityMonitorDaemon(config=config)

            daemon.last_quality = 0.85

            with patch.object(daemon, "_subscribe_to_events"):
                await daemon.start()
                await daemon.stop()

            assert state_path.exists()

            with open(state_path) as f:
                data = json.load(f)
            assert data["last_quality"] == 0.85


# =============================================================================
# Status and Health Check Tests
# =============================================================================


class TestStatusAndHealthCheck:
    """Tests for get_status and health_check methods."""

    def test_get_status(self):
        """Test status returns expected fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            config = QualityMonitorConfig(state_path=state_path)
            daemon = QualityMonitorDaemon(config=config)

            daemon._running = True
            daemon.last_quality = 0.75
            daemon.current_state = QualityState.GOOD

            status = daemon.get_status()

            assert status["running"] is True
            assert status["last_quality"] == 0.75
            assert status["current_state"] == "good"
            assert "check_interval" in status
            assert "history_size" in status

    def test_health_check_stopped(self):
        """Test health check when daemon is stopped."""
        daemon = QualityMonitorDaemon()
        daemon._running = False

        result = daemon.health_check()

        assert result.healthy is False
        assert "not running" in result.message.lower()

    def test_health_check_running_good(self):
        """Test health check when running with good quality."""
        daemon = QualityMonitorDaemon()
        daemon._running = True
        daemon.last_quality = 0.85
        daemon.current_state = QualityState.GOOD

        result = daemon.health_check()

        assert result.healthy is True
        assert "0.85" in result.message

    def test_health_check_running_poor(self):
        """Test health check when running with poor quality."""
        daemon = QualityMonitorDaemon()
        daemon._running = True
        daemon.last_quality = 0.35
        daemon.current_state = QualityState.POOR

        result = daemon.health_check()

        # Daemon itself is healthy, just quality is poor
        assert result.healthy is True
        assert "poor" in result.message.lower()


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventEmission:
    """Tests for event emission functionality."""

    @pytest.mark.asyncio
    async def test_emit_low_quality_warning(self):
        """Test emitting low quality warning event."""
        daemon = QualityMonitorDaemon()
        daemon._event_cooldown = 0  # Disable cooldown

        mock_router = MagicMock()
        mock_router.publish = AsyncMock()

        with patch.dict(
            "sys.modules",
            {
                "app.coordination.event_router": MagicMock(
                    get_router=MagicMock(return_value=mock_router),
                    DataEventType=MagicMock(
                        LOW_QUALITY_DATA_WARNING="LOW_QUALITY_DATA_WARNING"
                    ),
                ),
            },
        ):
            await daemon._emit_quality_event(
                quality=0.4,
                new_state=QualityState.POOR,
                old_state=QualityState.GOOD,
            )

    @pytest.mark.asyncio
    async def test_emit_quality_check_failed(self):
        """Test emitting quality check failed event."""
        daemon = QualityMonitorDaemon()

        mock_router = MagicMock()
        mock_router.publish = AsyncMock()

        with patch.dict(
            "sys.modules",
            {
                "app.coordination.event_router": MagicMock(
                    get_router=MagicMock(return_value=mock_router),
                    DataEventType=MagicMock(
                        QUALITY_CHECK_FAILED="QUALITY_CHECK_FAILED"
                    ),
                ),
            },
        ):
            await daemon._emit_quality_check_failed(
                reason="test error",
                config_key="hex8_2p",
                check_type="periodic",
            )


# =============================================================================
# Quality Check Tests
# =============================================================================


class TestQualityCheck:
    """Tests for quality checking functionality."""

    @pytest.mark.asyncio
    async def test_get_current_quality_no_data_dir(self):
        """Test quality check when data directory doesn't exist."""
        config = QualityMonitorConfig(data_dir="/nonexistent/path")
        daemon = QualityMonitorDaemon(config=config)

        quality = await daemon._get_current_quality()

        assert quality == 1.0  # Default when no data

    @pytest.mark.asyncio
    async def test_get_current_quality_no_databases(self):
        """Test quality check when no databases exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = QualityMonitorConfig(data_dir=tmpdir)
            daemon = QualityMonitorDaemon(config=config)

            quality = await daemon._get_current_quality()

            assert quality == 1.0  # Default when no databases


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunction:
    """Tests for create_quality_monitor factory function."""

    @pytest.mark.asyncio
    async def test_create_quality_monitor_starts_daemon(self):
        """Test that factory function creates and starts daemon."""
        # Create a task that we can cancel
        task = asyncio.create_task(create_quality_monitor())

        # Give it a moment to start
        await asyncio.sleep(0.1)

        # Cancel the task
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


# =============================================================================
# On-Demand Quality Check Tests
# =============================================================================


class TestOnDemandQualityCheck:
    """Tests for on-demand quality check handling."""

    @pytest.mark.asyncio
    async def test_on_quality_check_requested(self):
        """Test handling on-demand quality check request."""
        daemon = QualityMonitorDaemon()

        # Mock event payload
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "reason": "training_anomaly",
            "priority": "high",
        }

        with patch.object(
            daemon, "_get_current_quality", new_callable=AsyncMock, return_value=0.75
        ):
            with patch.object(daemon, "_emit_quality_event", new_callable=AsyncMock):
                await daemon._on_quality_check_requested(event)

        assert daemon._config_quality.get("hex8_2p") == 0.75
        assert daemon.last_quality == 0.75
        assert daemon.current_state == QualityState.GOOD

    @pytest.mark.asyncio
    async def test_on_quality_check_requested_error(self):
        """Test handling error during on-demand quality check."""
        daemon = QualityMonitorDaemon()

        event = MagicMock()
        event.payload = {"config_key": "hex8_2p"}

        with patch.object(
            daemon,
            "_get_current_quality",
            new_callable=AsyncMock,
            side_effect=Exception("DB error"),
        ):
            with patch.object(
                daemon, "_emit_quality_check_failed", new_callable=AsyncMock
            ) as mock_emit:
                await daemon._on_quality_check_requested(event)

                mock_emit.assert_called_once()


# =============================================================================
# Event Subscription Tests
# =============================================================================


class TestEventSubscription:
    """Tests for event subscription functionality."""

    def test_subscribe_to_events_idempotent(self):
        """Test that subscribing twice is safe."""
        daemon = QualityMonitorDaemon()
        daemon._subscribed = True

        # Should return early without doing anything
        daemon._subscribe_to_events()

        assert daemon._subscribed is True

    def test_subscribe_handles_missing_module(self):
        """Test subscription handles missing event module."""
        daemon = QualityMonitorDaemon()

        with patch.dict("sys.modules", {"app.coordination.event_router": None}):
            daemon._subscribe_to_events()

        # Should not crash, subscription flag depends on actual import


# =============================================================================
# Task Error Handling Tests
# =============================================================================


class TestTaskErrorHandling:
    """Tests for task error handling."""

    def test_handle_task_error_cancelled(self):
        """Test handling cancelled task."""
        daemon = QualityMonitorDaemon()

        task = MagicMock()
        task.cancelled.return_value = True

        # Should not raise
        daemon._handle_task_error(task)

    def test_handle_task_error_with_exception(self):
        """Test handling task with exception."""
        daemon = QualityMonitorDaemon()

        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = ValueError("test error")

        # Should not raise
        daemon._handle_task_error(task)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
