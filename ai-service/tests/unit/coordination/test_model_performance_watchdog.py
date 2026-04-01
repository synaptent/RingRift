"""Unit tests for ModelPerformanceWatchdog.

December 2025: Tests for model performance monitoring and degradation detection.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from app.coordination.model_performance_watchdog import (
    ModelPerformance,
    ModelPerformanceWatchdog,
    get_watchdog,
)


# =============================================================================
# Test ModelPerformance
# =============================================================================


class TestModelPerformance:
    """Tests for ModelPerformance dataclass."""

    def test_default_values(self):
        """ModelPerformance has expected defaults."""
        perf = ModelPerformance(
            model_id="test_model",
            board_type="hex8",
            num_players=2,
        )
        assert perf.model_id == "test_model"
        assert perf.board_type == "hex8"
        assert perf.num_players == 2
        assert perf.win_rate_vs_random == 0.0
        assert perf.win_rate_vs_heuristic == 0.0
        assert perf.win_rate_vs_previous == 0.0
        assert perf.recent_vs_random == []
        assert perf.recent_vs_heuristic == []
        assert perf.evaluation_count == 0
        assert perf.is_degraded is False
        assert perf.degraded_since is None

    def test_mutable_defaults_isolated(self):
        """Each instance has isolated mutable defaults."""
        p1 = ModelPerformance(model_id="m1", board_type="hex8", num_players=2)
        p2 = ModelPerformance(model_id="m2", board_type="hex8", num_players=2)

        p1.recent_vs_random.append(0.9)
        assert 0.9 not in p2.recent_vs_random


# =============================================================================
# Test ModelPerformanceWatchdog Initialization
# =============================================================================


class TestWatchdogInit:
    """Tests for watchdog initialization."""

    def test_default_initialization(self):
        """Watchdog initializes with defaults."""
        watchdog = ModelPerformanceWatchdog()
        assert watchdog._running is False
        assert watchdog.models == {}
        assert watchdog._degradation_threshold == 0.55
        assert watchdog._rolling_window_size == 5
        assert watchdog._alert_cooldown == 300.0


# =============================================================================
# Test Watchdog Lifecycle
# =============================================================================


class TestWatchdogLifecycle:
    """Tests for watchdog start/stop lifecycle."""

    @pytest.fixture
    def watchdog(self):
        """Create fresh watchdog for each test."""
        return ModelPerformanceWatchdog()

    @pytest.mark.asyncio
    async def test_start_sets_running(self, watchdog):
        """Start sets running flag."""
        with patch.object(watchdog, "_subscribe_all_events", return_value=True):
            await watchdog.start()
            assert watchdog._running is True
            assert watchdog.is_running is True  # is_running is a property

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, watchdog):
        """Multiple starts don't cause issues."""
        call_count = 0

        def mock_subscribe():
            nonlocal call_count
            call_count += 1
            return True

        with patch.object(watchdog, "_subscribe_all_events", side_effect=mock_subscribe):
            await watchdog.start()
            await watchdog.start()
            assert call_count == 1

    @pytest.mark.asyncio
    async def test_stop_clears_running(self, watchdog):
        """Stop clears running flag."""
        watchdog._running = True
        await watchdog.stop()
        assert watchdog._running is False
        assert watchdog.is_running is False  # is_running is a property


# =============================================================================
# Test Performance Tracking
# =============================================================================


class TestPerformanceTracking:
    """Tests for model performance tracking."""

    @pytest.fixture
    def watchdog(self):
        """Create watchdog with small window for testing."""
        w = ModelPerformanceWatchdog()
        w._rolling_window_size = 3
        return w

    @pytest.mark.asyncio
    async def test_creates_new_model_record(self, watchdog):
        """Creates new ModelPerformance for new models."""
        await watchdog._update_model_performance(
            model_id="model_1",
            board_type="hex8",
            num_players=2,
            win_rate_vs_random=0.90,
            win_rate_vs_heuristic=0.70,
        )

        assert "model_1" in watchdog.models
        perf = watchdog.models["model_1"]
        assert perf.board_type == "hex8"
        assert perf.num_players == 2
        assert perf.win_rate_vs_random == 0.90
        assert perf.win_rate_vs_heuristic == 0.70
        assert perf.evaluation_count == 1

    @pytest.mark.asyncio
    async def test_updates_existing_model(self, watchdog):
        """Updates existing model record."""
        # First evaluation
        await watchdog._update_model_performance(
            model_id="model_1",
            board_type="hex8",
            num_players=2,
            win_rate_vs_random=0.90,
            win_rate_vs_heuristic=0.70,
        )

        # Second evaluation (different rates)
        await watchdog._update_model_performance(
            model_id="model_1",
            board_type="hex8",
            num_players=2,
            win_rate_vs_random=0.85,
            win_rate_vs_heuristic=0.65,
        )

        perf = watchdog.models["model_1"]
        assert perf.evaluation_count == 2
        assert perf.win_rate_vs_random == 0.85
        assert perf.win_rate_vs_heuristic == 0.65

    @pytest.mark.asyncio
    async def test_rolling_history_maintained(self, watchdog):
        """Rolling history is maintained correctly."""
        for i in range(5):
            await watchdog._update_model_performance(
                model_id="model_1",
                board_type="hex8",
                num_players=2,
                win_rate_vs_random=0.80 + i * 0.02,
                win_rate_vs_heuristic=0.60 + i * 0.02,
            )

        perf = watchdog.models["model_1"]
        # Window size is 3, so only last 3 should be kept
        assert len(perf.recent_vs_random) == 3
        assert len(perf.recent_vs_heuristic) == 3
        # Check last values are correct
        assert perf.recent_vs_random[-1] == pytest.approx(0.88)
        assert perf.recent_vs_heuristic[-1] == pytest.approx(0.68)


# =============================================================================
# Test Degradation Detection
# =============================================================================


class TestDegradationDetection:
    """Tests for performance degradation detection."""

    @pytest.fixture
    def watchdog(self):
        """Create watchdog with specific thresholds."""
        w = ModelPerformanceWatchdog()
        w._degradation_threshold = 0.55
        w._alert_cooldown = 0.0  # No cooldown for testing
        return w

    @pytest.mark.asyncio
    async def test_detects_degradation(self, watchdog):
        """Detects when model performance drops below threshold."""
        with patch.object(watchdog, "_emit_degradation_alert", new_callable=AsyncMock) as mock_alert:
            await watchdog._update_model_performance(
                model_id="model_1",
                board_type="hex8",
                num_players=2,
                win_rate_vs_random=0.90,
                win_rate_vs_heuristic=0.50,  # Below 0.55 threshold
            )

            perf = watchdog.models["model_1"]
            assert perf.is_degraded is True
            assert perf.degraded_since is not None
            mock_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_alert_for_healthy_model(self, watchdog):
        """No alert for models above threshold."""
        with patch.object(watchdog, "_emit_degradation_alert", new_callable=AsyncMock) as mock_alert:
            await watchdog._update_model_performance(
                model_id="model_1",
                board_type="hex8",
                num_players=2,
                win_rate_vs_random=0.90,
                win_rate_vs_heuristic=0.70,  # Above 0.55 threshold
            )

            perf = watchdog.models["model_1"]
            assert perf.is_degraded is False
            mock_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_recovery_from_degradation(self, watchdog):
        """Detects recovery from degradation."""
        with patch.object(watchdog, "_emit_degradation_alert", new_callable=AsyncMock):
            # First: degraded
            await watchdog._update_model_performance(
                model_id="model_1",
                board_type="hex8",
                num_players=2,
                win_rate_vs_random=0.90,
                win_rate_vs_heuristic=0.50,  # Below threshold
            )

            assert watchdog.models["model_1"].is_degraded is True

            # Second: recovered
            await watchdog._update_model_performance(
                model_id="model_1",
                board_type="hex8",
                num_players=2,
                win_rate_vs_random=0.90,
                win_rate_vs_heuristic=0.65,  # Above threshold
            )

            perf = watchdog.models["model_1"]
            assert perf.is_degraded is False
            assert perf.degraded_since is None


# =============================================================================
# Test Alert Cooldown
# =============================================================================


class TestAlertCooldown:
    """Tests for alert cooldown mechanism."""

    @pytest.mark.asyncio
    async def test_respects_cooldown(self):
        """Doesn't alert during cooldown period."""
        from app.coordination.event_router import DataEventType

        watchdog = ModelPerformanceWatchdog()
        watchdog._degradation_threshold = 0.55
        watchdog._alert_cooldown = 300.0  # 5 minute cooldown

        with patch("app.coordination.event_router.publish", new_callable=AsyncMock) as mock_publish:
            # First alert
            perf = ModelPerformance(
                model_id="model_1",
                board_type="hex8",
                num_players=2,
                win_rate_vs_heuristic=0.50,
            )
            await watchdog._emit_degradation_alert(perf)

            # Second alert (should be blocked by cooldown)
            await watchdog._emit_degradation_alert(perf)

            # Only one publish call
            assert mock_publish.await_count == 1
            assert mock_publish.await_args.kwargs["event_type"] == DataEventType.REGRESSION_DETECTED


# =============================================================================
# Test Event Handling
# =============================================================================


class TestEventHandling:
    """Tests for event handling."""

    @pytest.fixture
    def watchdog(self):
        """Create fresh watchdog."""
        return ModelPerformanceWatchdog()

    @pytest.mark.asyncio
    async def test_handles_evaluation_event(self, watchdog):
        """Correctly processes EVALUATION_COMPLETED event."""
        event = MagicMock()
        event.payload = {
            "model_id": "test_model",
            "board_type": "hex8",
            "num_players": 2,
            "win_rate_vs_random": 0.90,
            "win_rate_vs_heuristic": 0.70,
        }

        await watchdog._on_evaluation_completed(event)

        assert "test_model" in watchdog.models
        assert watchdog.models["test_model"].win_rate_vs_random == 0.90

    @pytest.mark.asyncio
    async def test_handles_results_dict_format(self, watchdog):
        """Handles alternative 'results' dict format."""
        event = MagicMock()
        event.payload = {
            "model_id": "test_model",
            "board_type": "hex8",
            "num_players": 2,
            "results": {
                "vs_random": {"win_rate": 0.88},
                "vs_heuristic": {"win_rate": 0.72},
            },
        }

        await watchdog._on_evaluation_completed(event)

        assert "test_model" in watchdog.models
        assert watchdog.models["test_model"].win_rate_vs_random == 0.88
        assert watchdog.models["test_model"].win_rate_vs_heuristic == 0.72

    @pytest.mark.asyncio
    async def test_handles_dict_event(self, watchdog):
        """Handles plain dict event (no payload attribute)."""
        event = {
            "model_id": "test_model",
            "board_type": "hex8",
            "num_players": 2,
            "win_rate_vs_random": 0.85,
            "win_rate_vs_heuristic": 0.65,
        }

        await watchdog._on_evaluation_completed(event)

        assert "test_model" in watchdog.models


# =============================================================================
# Test Model Summary
# =============================================================================


class TestModelSummary:
    """Tests for model summary reporting."""

    @pytest.mark.asyncio
    async def test_get_model_summary(self):
        """Summary includes all tracked models."""
        watchdog = ModelPerformanceWatchdog()

        await watchdog._update_model_performance(
            model_id="model_1",
            board_type="hex8",
            num_players=2,
            win_rate_vs_random=0.90,
            win_rate_vs_heuristic=0.70,
        )

        summary = watchdog.get_model_summary()

        assert "model_1" in summary
        assert summary["model_1"]["board_type"] == "hex8"
        assert summary["model_1"]["win_rate_vs_random"] == 0.90
        assert summary["model_1"]["evaluation_count"] == 1
        assert "rolling_avg_vs_heuristic" in summary["model_1"]

    def test_empty_summary(self):
        """Summary is empty for fresh watchdog."""
        watchdog = ModelPerformanceWatchdog()
        assert watchdog.get_model_summary() == {}


# =============================================================================
# Test Health Check
# =============================================================================


class TestHealthCheck:
    """Tests for health check reporting."""

    def test_health_check_not_running(self):
        """Health check returns unhealthy when not running."""
        watchdog = ModelPerformanceWatchdog()
        result = watchdog.health_check()

        assert result.healthy is False
        assert "not running" in result.message.lower()

    def test_get_status(self):
        """get_status returns expected structure."""
        watchdog = ModelPerformanceWatchdog()
        status = watchdog.get_status()

        assert "name" in status
        assert "running" in status


# =============================================================================
# Test Singleton
# =============================================================================


class TestSingleton:
    """Tests for singleton behavior."""

    def test_get_watchdog_returns_singleton(self):
        """get_watchdog returns same instance."""
        # Note: Can't easily reset singleton without modifying module
        w1 = get_watchdog()
        w2 = get_watchdog()
        assert w1 is w2
