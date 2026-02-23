"""Tests for PipelineCompletenessMonitor daemon.

February 2026: Tests pipeline stage tracking, overdue detection,
health check RED/GREEN transitions, and large board threshold override.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.contracts import CoordinatorStatus
from app.coordination.pipeline_completeness_monitor import (
    DEFAULT_THRESHOLDS_HOURS,
    LARGE_BOARD_THRESHOLDS_HOURS,
    LARGE_BOARDS,
    PIPELINE_STAGES,
    PipelineCompletenessMonitor,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton between tests."""
    PipelineCompletenessMonitor.reset_instance()
    yield
    PipelineCompletenessMonitor.reset_instance()


@pytest.fixture
def monitor() -> PipelineCompletenessMonitor:
    """Create a fresh monitor instance."""
    m = PipelineCompletenessMonitor()
    m._all_configs = ["hex8_2p", "square8_2p", "square19_2p", "hexagonal_2p"]
    # Seed timestamps so stages are tracked
    now = time.time()
    for config_key in m._all_configs:
        m._stage_timestamps[config_key] = {stage: now for stage in PIPELINE_STAGES}
    return m


class TestStageTracking:
    """Test stage tracking on event receipt."""

    @pytest.mark.asyncio
    async def test_on_stage_complete_records_timestamp(self, monitor: PipelineCompletenessMonitor):
        """Calling the event handler should update the stage timestamp."""
        config_key = "hex8_2p"
        old_ts = monitor._stage_timestamps[config_key]["selfplay"]

        # Simulate a small delay
        event = MagicMock()
        event.event_type = "selfplay_complete"
        event.payload = {"config_key": config_key}

        await monitor._on_stage_complete(event)

        new_ts = monitor._stage_timestamps[config_key]["selfplay"]
        assert new_ts >= old_ts

    @pytest.mark.asyncio
    async def test_on_stage_complete_handles_dict_event(self, monitor: PipelineCompletenessMonitor):
        """Should handle dict-style events."""
        event = {"type": "training_completed", "config_key": "square8_2p"}

        await monitor._on_stage_complete(event)

        assert monitor._stage_timestamps["square8_2p"]["training"] > 0

    @pytest.mark.asyncio
    async def test_unknown_event_type_ignored(self, monitor: PipelineCompletenessMonitor):
        """Unknown event types should be silently ignored."""
        event = MagicMock()
        event.event_type = "unknown_event"
        event.payload = {"config_key": "hex8_2p"}

        # Should not raise
        await monitor._on_stage_complete(event)

    @pytest.mark.asyncio
    async def test_unknown_config_key_ignored(self, monitor: PipelineCompletenessMonitor):
        """Events with unknown config should still be recorded."""
        event = MagicMock()
        event.event_type = "selfplay_complete"
        event.payload = {}

        # Should not raise; config_key will be "unknown" and ignored
        await monitor._on_stage_complete(event)

    @pytest.mark.asyncio
    async def test_new_config_added_dynamically(self, monitor: PipelineCompletenessMonitor):
        """A new config seen via events should be recorded."""
        event = MagicMock()
        event.event_type = "selfplay_complete"
        event.payload = {"config_key": "hex8_3p"}

        await monitor._on_stage_complete(event)

        assert "hex8_3p" in monitor._stage_timestamps
        assert "selfplay" in monitor._stage_timestamps["hex8_3p"]


class TestOverdueDetection:
    """Test overdue detection at threshold boundaries."""

    @pytest.mark.asyncio
    async def test_no_overdue_when_recent(self, monitor: PipelineCompletenessMonitor):
        """No overdue events when all stages completed recently."""
        with patch.object(monitor, "_safe_emit_event_async", new_callable=AsyncMock) as mock_emit:
            await monitor._run_cycle()
            mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_overdue_detected_when_past_threshold(self, monitor: PipelineCompletenessMonitor):
        """Should emit PIPELINE_STAGE_OVERDUE when stage exceeds threshold."""
        # Set selfplay timestamp to 7 hours ago (threshold is 6h for small boards)
        config_key = "hex8_2p"
        monitor._stage_timestamps[config_key]["selfplay"] = time.time() - (7 * 3600)

        with patch.object(monitor, "_safe_emit_event_async", new_callable=AsyncMock) as mock_emit:
            await monitor._run_cycle()

            mock_emit.assert_called()
            call_args = mock_emit.call_args_list[0]
            assert call_args[0][0] == "PIPELINE_STAGE_OVERDUE"
            payload = call_args[0][1]
            assert payload["config_key"] == config_key
            assert payload["stage"] == "selfplay"
            assert payload["hours_since"] > 6.0
            assert payload["threshold"] == 6.0

    @pytest.mark.asyncio
    async def test_not_overdue_at_boundary(self, monitor: PipelineCompletenessMonitor):
        """Stage exactly at threshold should not trigger overdue."""
        config_key = "hex8_2p"
        # Set to exactly 5.9 hours ago (threshold is 6h)
        monitor._stage_timestamps[config_key]["selfplay"] = time.time() - (5.9 * 3600)

        with patch.object(monitor, "_safe_emit_event_async", new_callable=AsyncMock) as mock_emit:
            await monitor._run_cycle()

            # Should not emit for selfplay since 5.9h < 6h
            for call in mock_emit.call_args_list:
                payload = call[0][1]
                if payload.get("config_key") == config_key and payload.get("stage") == "selfplay":
                    pytest.fail("Should not emit overdue for stage within threshold")

    @pytest.mark.asyncio
    async def test_multiple_stages_overdue(self, monitor: PipelineCompletenessMonitor):
        """Multiple overdue stages should each emit separate events."""
        config_key = "hex8_2p"
        # Set selfplay (6h threshold) and training (24h threshold) as overdue
        monitor._stage_timestamps[config_key]["selfplay"] = time.time() - (10 * 3600)
        monitor._stage_timestamps[config_key]["training"] = time.time() - (30 * 3600)

        with patch.object(monitor, "_safe_emit_event_async", new_callable=AsyncMock) as mock_emit:
            await monitor._run_cycle()

            overdue_stages = set()
            for call in mock_emit.call_args_list:
                payload = call[0][1]
                if payload.get("config_key") == config_key:
                    overdue_stages.add(payload["stage"])

            assert "selfplay" in overdue_stages
            assert "training" in overdue_stages


class TestLargeBoardThresholds:
    """Test large board threshold override."""

    def test_large_boards_defined(self):
        """Verify large boards constant includes expected boards."""
        assert "square19" in LARGE_BOARDS
        assert "hexagonal" in LARGE_BOARDS
        assert "hex8" not in LARGE_BOARDS
        assert "square8" not in LARGE_BOARDS

    def test_large_board_selfplay_threshold(self, monitor: PipelineCompletenessMonitor):
        """Large boards should use 12h selfplay threshold."""
        threshold = monitor._get_threshold_hours("selfplay", is_large_board=True)
        assert threshold == 12.0

    def test_small_board_selfplay_threshold(self, monitor: PipelineCompletenessMonitor):
        """Small boards should use 6h selfplay threshold."""
        threshold = monitor._get_threshold_hours("selfplay", is_large_board=False)
        assert threshold == 6.0

    def test_training_threshold_same_for_all(self, monitor: PipelineCompletenessMonitor):
        """Training threshold should be same regardless of board size."""
        small = monitor._get_threshold_hours("training", is_large_board=False)
        large = monitor._get_threshold_hours("training", is_large_board=True)
        assert small == large == 24.0

    @pytest.mark.asyncio
    async def test_large_board_not_overdue_at_small_threshold(
        self, monitor: PipelineCompletenessMonitor
    ):
        """square19 selfplay at 8h should NOT be overdue (threshold is 12h)."""
        config_key = "square19_2p"
        monitor._stage_timestamps[config_key]["selfplay"] = time.time() - (8 * 3600)

        with patch.object(monitor, "_safe_emit_event_async", new_callable=AsyncMock) as mock_emit:
            await monitor._run_cycle()

            for call in mock_emit.call_args_list:
                payload = call[0][1]
                if (
                    payload.get("config_key") == config_key
                    and payload.get("stage") == "selfplay"
                ):
                    pytest.fail(
                        "square19 selfplay at 8h should not be overdue "
                        "(large board threshold is 12h)"
                    )

    @pytest.mark.asyncio
    async def test_large_board_overdue_past_large_threshold(
        self, monitor: PipelineCompletenessMonitor
    ):
        """square19 selfplay at 14h SHOULD be overdue (threshold is 12h)."""
        config_key = "square19_2p"
        monitor._stage_timestamps[config_key]["selfplay"] = time.time() - (14 * 3600)

        with patch.object(monitor, "_safe_emit_event_async", new_callable=AsyncMock) as mock_emit:
            await monitor._run_cycle()

            found = False
            for call in mock_emit.call_args_list:
                payload = call[0][1]
                if (
                    payload.get("config_key") == config_key
                    and payload.get("stage") == "selfplay"
                ):
                    found = True
                    assert payload["threshold"] == 12.0
            assert found, "Expected PIPELINE_STAGE_OVERDUE for square19 selfplay at 14h"


class TestHealthCheck:
    """Test health_check RED/GREEN transitions."""

    def test_healthy_when_no_overdue(self, monitor: PipelineCompletenessMonitor):
        """Health check should be GREEN when no stages are overdue."""
        monitor._running = True
        monitor._overdue_counts = {"hex8_2p": 0, "square8_2p": 0}

        result = monitor.health_check()
        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING

    def test_healthy_with_single_overdue_stage(self, monitor: PipelineCompletenessMonitor):
        """Health check should be GREEN with only 1 overdue stage per config."""
        monitor._running = True
        monitor._overdue_counts = {"hex8_2p": 1, "square8_2p": 0}

        result = monitor.health_check()
        assert result.healthy is True

    def test_unhealthy_with_two_overdue_stages(self, monitor: PipelineCompletenessMonitor):
        """Health check should be RED when any config has 2+ overdue stages."""
        monitor._running = True
        monitor._overdue_counts = {"hex8_2p": 2, "square8_2p": 0}

        result = monitor.health_check()
        assert result.healthy is False
        assert result.status == CoordinatorStatus.ERROR
        assert "hex8_2p" in result.message

    def test_unhealthy_with_multiple_configs_red(self, monitor: PipelineCompletenessMonitor):
        """Multiple configs with 2+ overdue should all be reported."""
        monitor._running = True
        monitor._overdue_counts = {"hex8_2p": 3, "square8_2p": 2, "square19_2p": 0}

        result = monitor.health_check()
        assert result.healthy is False
        configs_red = result.details.get("configs_red", [])
        assert "hex8_2p" in configs_red
        assert "square8_2p" in configs_red
        assert "square19_2p" not in configs_red

    def test_not_running_returns_unhealthy(self, monitor: PipelineCompletenessMonitor):
        """Health check should return unhealthy when not running."""
        monitor._running = False

        result = monitor.health_check()
        assert result.healthy is False


class TestEventSubscriptions:
    """Test event subscription setup."""

    def test_subscribes_to_all_pipeline_events(self, monitor: PipelineCompletenessMonitor):
        """Should subscribe to all 6 pipeline stage events."""
        subs = monitor._get_event_subscriptions()
        expected_events = {
            "selfplay_complete",
            "data_sync_completed",
            "npz_export_complete",
            "training_completed",
            "evaluation_completed",
            "model_promoted",
        }
        assert set(subs.keys()) == expected_events
