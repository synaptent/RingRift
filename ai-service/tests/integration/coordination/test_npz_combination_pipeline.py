"""Integration tests for NPZ combination pipeline event flow.

December 2025 - Verifies the complete event chain:
    NPZ_EXPORT_COMPLETE -> NPZ_COMBINATION_COMPLETE -> TRAINING_THRESHOLD_REACHED

Tests ensure that:
1. Export completion triggers combination daemon
2. Combination completion triggers training
3. Combination failure falls back to single-file training
4. Event deduplication works correctly
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.npz_combination_daemon import (
    NPZCombinationConfig,
    NPZCombinationDaemon,
    get_npz_combination_daemon,
)
from app.distributed.data_events import DataEventType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def combination_config():
    """Create test configuration."""
    return NPZCombinationConfig(
        freshness_weight=1.5,
        min_quality_score=0.2,
        min_interval_seconds=0.0,  # No throttling for tests
    )


@pytest.fixture
def daemon(combination_config):
    """Create a fresh daemon instance for each test."""
    NPZCombinationDaemon.reset_instance()
    daemon = NPZCombinationDaemon(config=combination_config)
    yield daemon
    # Cleanup
    NPZCombinationDaemon.reset_instance()


@pytest.fixture
def mock_combine_result():
    """Create a successful combine result."""
    from app.training.npz_combiner import CombineResult

    return CombineResult(
        success=True,
        output_path=Path("/tmp/test_combined.npz"),
        total_samples=10000,
        samples_by_source={"fresh": 6000, "historical": 4000},
        error=None,
    )


# =============================================================================
# Test Event Subscription
# =============================================================================


class TestEventSubscription:
    """Tests for event subscription wiring."""

    def test_subscribes_to_npz_export_complete(self, daemon):
        """Daemon should subscribe to NPZ_EXPORT_COMPLETE events."""
        subscriptions = daemon._get_event_subscriptions()

        assert DataEventType.NPZ_EXPORT_COMPLETE.value in subscriptions
        assert subscriptions[DataEventType.NPZ_EXPORT_COMPLETE.value] == daemon._on_npz_export_complete

    def test_subscribes_to_npz_combination_complete(self, daemon):
        """Daemon should also subscribe to combination complete for manual triggers."""
        subscriptions = daemon._get_event_subscriptions()

        assert DataEventType.NPZ_COMBINATION_COMPLETE.value in subscriptions


# =============================================================================
# Test Export -> Combination Flow
# =============================================================================


class TestExportToCombinationFlow:
    """Tests for NPZ_EXPORT_COMPLETE -> combination trigger."""

    @pytest.mark.asyncio
    async def test_export_complete_triggers_combination(self, daemon, mock_combine_result):
        """NPZ_EXPORT_COMPLETE should trigger _combine_for_config."""
        event = {
            "config_key": "hex8_2p",
            "board_type": "hex8",
            "num_players": 2,
            "output_path": "/tmp/test.npz",
            "samples_exported": 5000,
        }

        with patch.object(
            daemon, "_combine_for_config", new_callable=AsyncMock
        ) as mock_combine:
            mock_combine.return_value = mock_combine_result

            await daemon._on_npz_export_complete(event)

            mock_combine.assert_called_once_with("hex8_2p")

    @pytest.mark.asyncio
    async def test_export_complete_updates_stats(self, daemon, mock_combine_result):
        """Successful combination should update statistics."""
        event = {"config_key": "hex8_2p"}

        with patch.object(
            daemon, "_combine_for_config", new_callable=AsyncMock
        ) as mock_combine:
            mock_combine.return_value = mock_combine_result

            await daemon._on_npz_export_complete(event)

            assert daemon.combination_stats.combinations_triggered == 1
            assert daemon.combination_stats.combinations_succeeded == 1
            assert daemon.combination_stats.total_samples_combined == 10000

    @pytest.mark.asyncio
    async def test_missing_config_key_logs_warning(self, daemon):
        """Event without config_key should log warning and return early."""
        event = {}  # Missing config_key

        with patch.object(
            daemon, "_combine_for_config", new_callable=AsyncMock
        ) as mock_combine:
            await daemon._on_npz_export_complete(event)

            mock_combine.assert_not_called()


# =============================================================================
# Test Combination -> Training Flow
# =============================================================================


class TestCombinationToTrainingFlow:
    """Tests for NPZ_COMBINATION_COMPLETE -> training trigger."""

    @pytest.mark.asyncio
    async def test_combination_emits_complete_event(self, daemon, mock_combine_result):
        """Successful combination should emit NPZ_COMBINATION_COMPLETE."""
        # Patch where emit_data_event is imported (inside the method)
        with patch(
            "app.distributed.data_events.emit_data_event"
        ) as mock_emit:
            daemon._emit_combination_complete("hex8_2p", mock_combine_result)

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0] == DataEventType.NPZ_COMBINATION_COMPLETE

    @pytest.mark.asyncio
    async def test_combination_failure_emits_failed_event(self, daemon):
        """Failed combination should emit NPZ_COMBINATION_FAILED."""
        # Patch where emit_data_event is imported (inside the method)
        with patch(
            "app.distributed.data_events.emit_data_event"
        ) as mock_emit:
            daemon._emit_combination_failed("hex8_2p", "Test error")

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0] == DataEventType.NPZ_COMBINATION_FAILED


# =============================================================================
# Test Throttling
# =============================================================================


class TestThrottling:
    """Tests for combination throttling."""

    @pytest.mark.asyncio
    async def test_throttles_rapid_events(self):
        """Rapid events for same config should be throttled."""
        config = NPZCombinationConfig(min_interval_seconds=60.0)
        daemon = NPZCombinationDaemon(config=config)

        # Simulate first combination just completed
        daemon.combination_stats.last_combination_by_config["hex8_2p"] = time.time()

        event = {"config_key": "hex8_2p"}

        with patch.object(
            daemon, "_combine_for_config", new_callable=AsyncMock
        ) as mock_combine:
            await daemon._on_npz_export_complete(event)

            # Should be skipped due to throttling
            mock_combine.assert_not_called()
            assert daemon.combination_stats.combinations_skipped == 1

    @pytest.mark.asyncio
    async def test_different_configs_not_throttled(self, daemon, mock_combine_result):
        """Different configs should not throttle each other."""
        # Simulate hex8_2p just completed
        daemon.combination_stats.last_combination_by_config["hex8_2p"] = time.time()

        event = {"config_key": "square8_2p"}  # Different config

        with patch.object(
            daemon, "_combine_for_config", new_callable=AsyncMock
        ) as mock_combine:
            mock_combine.return_value = mock_combine_result

            await daemon._on_npz_export_complete(event)

            # Should NOT be throttled
            mock_combine.assert_called_once_with("square8_2p")


# =============================================================================
# Test Event Deduplication
# =============================================================================


class TestEventDeduplication:
    """Tests for event deduplication."""

    @pytest.mark.asyncio
    async def test_duplicate_events_are_skipped(self, daemon, mock_combine_result):
        """Duplicate events should be detected and skipped."""
        event = {
            "config_key": "hex8_2p",
            "event_id": "test-event-123",
        }

        with patch.object(
            daemon, "_combine_for_config", new_callable=AsyncMock
        ) as mock_combine:
            mock_combine.return_value = mock_combine_result

            # First call should process
            await daemon._on_npz_export_complete(event)
            assert mock_combine.call_count == 1

            # Second call with same event should be deduplicated by HandlerBase
            # (if _is_duplicate_event returns True)
            with patch.object(daemon, "_is_duplicate_event", return_value=True):
                await daemon._on_npz_export_complete(event)
                # Should still be 1 call
                assert mock_combine.call_count == 1


# =============================================================================
# Test Health Check
# =============================================================================


class TestHealthCheck:
    """Tests for daemon health check."""

    def test_health_check_when_running(self, daemon):
        """Health check should report healthy when running."""
        daemon._running = True
        daemon.combination_stats.combinations_succeeded = 10
        daemon.combination_stats.combinations_failed = 1

        result = daemon.health_check()

        assert result.healthy is True
        assert result.status == "healthy"
        assert "11" in result.message or "10" in result.message  # Total attempts

    def test_health_check_when_stopped(self, daemon):
        """Health check should report unhealthy when stopped."""
        daemon._running = False

        result = daemon.health_check()

        assert result.healthy is False
        assert result.status == "stopped"

    def test_health_check_degraded_on_high_error_rate(self, daemon):
        """Health check should report degraded on high error rate."""
        daemon._running = True
        daemon.combination_stats.combinations_succeeded = 1
        daemon.combination_stats.combinations_failed = 10  # 90% error rate

        result = daemon.health_check()

        assert result.healthy is False
        assert result.status == "degraded"


# =============================================================================
# Test End-to-End Pipeline (Mocked)
# =============================================================================


class TestEndToEndPipeline:
    """End-to-end tests for the complete pipeline flow."""

    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self, daemon, mock_combine_result):
        """Test complete flow: export -> combine -> emit training trigger."""
        emit_calls = []

        def capture_complete(config_key, result):
            emit_calls.append(("complete", config_key, result))

        event = {
            "config_key": "hex8_2p",
            "output_path": "/tmp/exported.npz",
            "samples_exported": 5000,
        }

        with patch.object(
            daemon, "_combine_for_config", new_callable=AsyncMock
        ) as mock_combine, patch.object(
            daemon, "_emit_combination_complete", side_effect=capture_complete
        ):
            mock_combine.return_value = mock_combine_result

            # Trigger the pipeline
            await daemon._on_npz_export_complete(event)

            # Verify combination was called
            mock_combine.assert_called_once_with("hex8_2p")

            # Verify _emit_combination_complete was called
            assert len(emit_calls) == 1
            call_type, config_key, result = emit_calls[0]
            assert call_type == "complete"
            assert config_key == "hex8_2p"
            assert result.total_samples == 10000

            # Stats should still be updated
            assert daemon.combination_stats.combinations_succeeded == 1

    @pytest.mark.asyncio
    async def test_pipeline_fallback_on_failure(self, daemon):
        """Test fallback flow when combination fails."""
        from app.training.npz_combiner import CombineResult

        failed_result = CombineResult(
            success=False,
            error="Test failure",
        )

        emit_calls = []

        def capture_failed(config_key, error):
            emit_calls.append(("failed", config_key, error))

        event = {"config_key": "hex8_2p"}

        with patch.object(
            daemon, "_combine_for_config", new_callable=AsyncMock
        ) as mock_combine, patch.object(
            daemon, "_emit_combination_failed", side_effect=capture_failed
        ):
            mock_combine.return_value = failed_result

            await daemon._on_npz_export_complete(event)

            # Verify _emit_combination_failed was called
            assert len(emit_calls) == 1
            call_type, config_key, error = emit_calls[0]
            assert call_type == "failed"
            assert config_key == "hex8_2p"
            assert "Test failure" in error

            # Verify stats updated
            assert daemon.combination_stats.combinations_failed == 1
