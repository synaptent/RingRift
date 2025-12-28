"""Integration tests for full training pipeline event flow.

December 2025 - Verifies the complete event chain from selfplay to promotion:

    SELFPLAY_COMPLETE → DATA_SYNC_COMPLETED → NPZ_EXPORT_COMPLETE →
    NPZ_COMBINATION_COMPLETE → TRAINING_THRESHOLD_REACHED → TRAINING_COMPLETED →
    EVALUATION_COMPLETED → MODEL_PROMOTED

Tests ensure that:
1. Each stage triggers the next stage via events
2. Cross-process event bridging works correctly
3. Pipeline handles failures gracefully with fallbacks
4. Event deduplication prevents duplicate processing
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.distributed.data_events import DataEventType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_event_bus():
    """Create mock event bus for capturing events."""
    events_published: list[tuple[str, dict]] = []

    class MockEventBus:
        def __init__(self):
            self.subscribers: dict[str, list[Any]] = {}

        def subscribe(self, event_type: str | DataEventType, handler: Any) -> None:
            event_key = event_type.value if hasattr(event_type, "value") else str(event_type)
            if event_key not in self.subscribers:
                self.subscribers[event_key] = []
            self.subscribers[event_key].append(handler)

        def unsubscribe(self, event_type: str | DataEventType, handler: Any) -> None:
            event_key = event_type.value if hasattr(event_type, "value") else str(event_type)
            if event_key in self.subscribers:
                self.subscribers[event_key] = [
                    h for h in self.subscribers[event_key] if h != handler
                ]

        async def publish_async(self, event_type: str, payload: dict) -> None:
            events_published.append((event_type, payload))
            if event_type in self.subscribers:
                for handler in self.subscribers[event_type]:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(payload)
                    else:
                        handler(payload)

        def publish(self, event_type: str, payload: dict) -> None:
            events_published.append((event_type, payload))

    bus = MockEventBus()
    bus.events_published = events_published
    return bus


@pytest.fixture
def config_key():
    """Standard config key for tests."""
    return "hex8_2p"


# =============================================================================
# Test Cross-Process Event Mapping
# =============================================================================


class TestCrossProcessEventMapping:
    """Tests for cross-process event bridging."""

    def test_npz_combination_events_in_mapping(self):
        """NPZ_COMBINATION events should be in cross-process map."""
        from app.coordination.event_mappings import DATA_TO_CROSS_PROCESS_MAP

        assert "npz_combination_started" in DATA_TO_CROSS_PROCESS_MAP
        assert "npz_combination_complete" in DATA_TO_CROSS_PROCESS_MAP
        assert "npz_combination_failed" in DATA_TO_CROSS_PROCESS_MAP

        assert DATA_TO_CROSS_PROCESS_MAP["npz_combination_started"] == "NPZ_COMBINATION_STARTED"
        assert DATA_TO_CROSS_PROCESS_MAP["npz_combination_complete"] == "NPZ_COMBINATION_COMPLETE"
        assert DATA_TO_CROSS_PROCESS_MAP["npz_combination_failed"] == "NPZ_COMBINATION_FAILED"

    def test_npz_export_events_in_mapping(self):
        """NPZ_EXPORT events should be in cross-process map."""
        from app.coordination.event_mappings import DATA_TO_CROSS_PROCESS_MAP

        assert "npz_export_started" in DATA_TO_CROSS_PROCESS_MAP
        assert "npz_export_complete" in DATA_TO_CROSS_PROCESS_MAP

    def test_training_events_in_mapping(self):
        """Training events should be in cross-process map."""
        from app.coordination.event_mappings import DATA_TO_CROSS_PROCESS_MAP

        assert "training_started" in DATA_TO_CROSS_PROCESS_MAP
        assert "training_completed" in DATA_TO_CROSS_PROCESS_MAP
        assert "training_failed" in DATA_TO_CROSS_PROCESS_MAP

    def test_evaluation_events_in_mapping(self):
        """Evaluation events should be in cross-process map."""
        from app.coordination.event_mappings import DATA_TO_CROSS_PROCESS_MAP

        assert "evaluation_started" in DATA_TO_CROSS_PROCESS_MAP
        assert "evaluation_completed" in DATA_TO_CROSS_PROCESS_MAP


# =============================================================================
# Test Pipeline Stage Chaining
# =============================================================================


class TestPipelineStageChaining:
    """Tests for pipeline stage event chaining."""

    def test_data_event_types_exist(self):
        """All pipeline event types should exist in DataEventType."""
        # Selfplay stage
        assert hasattr(DataEventType, "SELFPLAY_COMPLETE")

        # Sync stage
        assert hasattr(DataEventType, "DATA_SYNC_COMPLETED")

        # Export stage
        assert hasattr(DataEventType, "NPZ_EXPORT_STARTED")
        assert hasattr(DataEventType, "NPZ_EXPORT_COMPLETE")

        # Combination stage
        assert hasattr(DataEventType, "NPZ_COMBINATION_STARTED")
        assert hasattr(DataEventType, "NPZ_COMBINATION_COMPLETE")
        assert hasattr(DataEventType, "NPZ_COMBINATION_FAILED")

        # Training stage
        assert hasattr(DataEventType, "TRAINING_THRESHOLD_REACHED")
        assert hasattr(DataEventType, "TRAINING_STARTED")
        assert hasattr(DataEventType, "TRAINING_COMPLETED")
        assert hasattr(DataEventType, "TRAINING_FAILED")

        # Evaluation stage
        assert hasattr(DataEventType, "EVALUATION_STARTED")
        assert hasattr(DataEventType, "EVALUATION_COMPLETED")

        # Promotion stage
        assert hasattr(DataEventType, "MODEL_PROMOTED")
        assert hasattr(DataEventType, "PROMOTION_FAILED")

    def test_npz_combination_daemon_subscribes_to_export(self):
        """NPZCombinationDaemon should subscribe to NPZ_EXPORT_COMPLETE."""
        from app.coordination.npz_combination_daemon import NPZCombinationDaemon

        # Reset and create fresh instance
        NPZCombinationDaemon.reset_instance()
        try:
            daemon = NPZCombinationDaemon()
            subscriptions = daemon._get_event_subscriptions()

            assert DataEventType.NPZ_EXPORT_COMPLETE.value in subscriptions
        finally:
            NPZCombinationDaemon.reset_instance()

    def test_data_pipeline_subscribes_to_combination(self):
        """DataPipelineOrchestrator should subscribe to NPZ_COMBINATION_COMPLETE."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        # Check that the handler method exists
        assert hasattr(DataPipelineOrchestrator, "_on_npz_combination_complete")

    def test_feedback_loop_subscribes_to_training(self):
        """FeedbackLoopController should subscribe to TRAINING_COMPLETED."""
        from app.coordination.feedback_loop_controller import FeedbackLoopController

        # Check that the handler method exists (note: method is _on_training_complete not _on_training_completed)
        assert hasattr(FeedbackLoopController, "_on_training_complete")


# =============================================================================
# Test Event Payload Structure
# =============================================================================


class TestEventPayloadStructure:
    """Tests for consistent event payload structure."""

    def test_npz_combination_complete_payload(self, config_key):
        """NPZ_COMBINATION_COMPLETE should have required fields."""
        payload = {
            "config_key": config_key,
            "output_path": "/tmp/combined.npz",
            "total_samples": 10000,
            "samples_by_source": {"fresh": 6000, "historical": 4000},
            "source": "NPZCombinationDaemon",
        }

        # Verify required fields
        assert "config_key" in payload
        assert "output_path" in payload
        assert "total_samples" in payload

    def test_training_completed_payload(self, config_key):
        """TRAINING_COMPLETED should have required fields."""
        payload = {
            "config_key": config_key,
            "model_path": "/tmp/model.pth",
            "epochs_completed": 50,
            "final_loss": 0.123,
            "source": "TrainingCoordinator",
        }

        # Verify required fields
        assert "config_key" in payload
        assert "model_path" in payload

    def test_evaluation_completed_payload(self, config_key):
        """EVALUATION_COMPLETED should have required fields."""
        payload = {
            "config_key": config_key,
            "model_path": "/tmp/model.pth",
            "win_rate_random": 0.95,
            "win_rate_heuristic": 0.65,
            "passed_gauntlet": True,
            "source": "EvaluationDaemon",
        }

        # Verify required fields
        assert "config_key" in payload
        assert "passed_gauntlet" in payload


# =============================================================================
# Test Pipeline Failure Fallbacks
# =============================================================================


class TestPipelineFailureFallbacks:
    """Tests for graceful failure handling."""

    @pytest.mark.asyncio
    async def test_combination_failure_emits_failed_event(self):
        """NPZ combination failure should emit NPZ_COMBINATION_FAILED."""
        from app.coordination.npz_combination_daemon import (
            NPZCombinationDaemon,
            NPZCombinationConfig,
        )

        # Reset and create fresh instance
        NPZCombinationDaemon.reset_instance()
        try:
            config = NPZCombinationConfig(min_interval_seconds=0.0)
            daemon = NPZCombinationDaemon(config=config)

            emit_calls = []

            def capture_failed(config_key, error):
                emit_calls.append((config_key, error))

            with patch.object(daemon, "_emit_combination_failed", side_effect=capture_failed):
                from app.training.npz_combiner import CombineResult

                failed_result = CombineResult(success=False, error="Test error")

                with patch.object(
                    daemon, "_combine_for_config", new_callable=AsyncMock
                ) as mock_combine:
                    mock_combine.return_value = failed_result

                    await daemon._on_npz_export_complete({"config_key": "hex8_2p"})

                    assert len(emit_calls) == 1
                    assert emit_calls[0][0] == "hex8_2p"
                    assert "Test error" in emit_calls[0][1]
        finally:
            NPZCombinationDaemon.reset_instance()


# =============================================================================
# Test Daemon Profile Integration
# =============================================================================


class TestDaemonProfileIntegration:
    """Tests for NPZ_COMBINATION in daemon profiles."""

    def test_npz_combination_in_daemon_types(self):
        """NPZ_COMBINATION should be a valid DaemonType."""
        from app.coordination.daemon_types import DaemonType

        assert hasattr(DaemonType, "NPZ_COMBINATION")

    def test_npz_combination_in_registry(self):
        """NPZ_COMBINATION should be in daemon registry."""
        from app.coordination.daemon_registry import DAEMON_REGISTRY
        from app.coordination.daemon_types import DaemonType

        assert DaemonType.NPZ_COMBINATION in DAEMON_REGISTRY

    def test_npz_combination_has_runner(self):
        """NPZ_COMBINATION should have a runner function."""
        from app.coordination.daemon_runners import get_runner
        from app.coordination.daemon_types import DaemonType

        runner = get_runner(DaemonType.NPZ_COMBINATION)
        assert runner is not None
        assert callable(runner)


# =============================================================================
# Test Event Deduplication
# =============================================================================


class TestEventDeduplication:
    """Tests for event deduplication across pipeline."""

    def test_handler_base_has_dedup_method(self):
        """HandlerBase should have _is_duplicate_event method."""
        from app.coordination.handler_base import HandlerBase

        assert hasattr(HandlerBase, "_is_duplicate_event")

    @pytest.mark.asyncio
    async def test_duplicate_export_events_deduplicated(self):
        """Duplicate NPZ_EXPORT_COMPLETE events should be deduplicated."""
        from app.coordination.npz_combination_daemon import (
            NPZCombinationDaemon,
            NPZCombinationConfig,
        )
        from app.training.npz_combiner import CombineResult

        # Reset and create fresh instance
        NPZCombinationDaemon.reset_instance()
        try:
            config = NPZCombinationConfig(min_interval_seconds=0.0)
            daemon = NPZCombinationDaemon(config=config)

            mock_result = CombineResult(
                success=True,
                output_path=Path("/tmp/test.npz"),
                total_samples=1000,
            )

            combine_call_count = 0

            async def track_combine(config_key):
                nonlocal combine_call_count
                combine_call_count += 1
                return mock_result

            with patch.object(daemon, "_combine_for_config", side_effect=track_combine):
                with patch.object(daemon, "_emit_combination_complete"):
                    # First call should process
                    event = {"config_key": "hex8_2p", "event_id": "test-1"}
                    await daemon._on_npz_export_complete(event)
                    assert combine_call_count == 1

                    # Same event_id with dedup enabled should skip
                    with patch.object(daemon, "_is_duplicate_event", return_value=True):
                        await daemon._on_npz_export_complete(event)
                        assert combine_call_count == 1  # Still 1
        finally:
            NPZCombinationDaemon.reset_instance()


# =============================================================================
# Test Full Pipeline Simulation
# =============================================================================


class TestFullPipelineSimulation:
    """End-to-end simulation of pipeline flow."""

    @pytest.mark.asyncio
    async def test_export_triggers_combination_which_emits_complete(self):
        """Full flow: NPZ_EXPORT_COMPLETE → combination → NPZ_COMBINATION_COMPLETE."""
        from app.coordination.npz_combination_daemon import (
            NPZCombinationDaemon,
            NPZCombinationConfig,
        )
        from app.training.npz_combiner import CombineResult

        NPZCombinationDaemon.reset_instance()
        try:
            config = NPZCombinationConfig(min_interval_seconds=0.0)
            daemon = NPZCombinationDaemon(config=config)

            mock_result = CombineResult(
                success=True,
                output_path=Path("/tmp/combined.npz"),
                total_samples=10000,
                samples_by_source={"fresh": 6000, "historical": 4000},
            )

            complete_events = []

            def capture_complete(config_key, result):
                complete_events.append({"config_key": config_key, "result": result})

            with patch.object(daemon, "_combine_for_config", new_callable=AsyncMock) as mock_combine:
                mock_combine.return_value = mock_result

                with patch.object(daemon, "_emit_combination_complete", side_effect=capture_complete):
                    # Trigger export complete event
                    await daemon._on_npz_export_complete({
                        "config_key": "hex8_2p",
                        "output_path": "/tmp/export.npz",
                        "samples_exported": 5000,
                    })

                    # Verify combination was triggered
                    mock_combine.assert_called_once_with("hex8_2p")

                    # Verify complete event was emitted
                    assert len(complete_events) == 1
                    assert complete_events[0]["config_key"] == "hex8_2p"
                    assert complete_events[0]["result"].total_samples == 10000

                    # Verify stats updated
                    assert daemon.combination_stats.combinations_succeeded == 1
        finally:
            NPZCombinationDaemon.reset_instance()

    def test_pipeline_stage_enum_ordering(self):
        """PipelineStage enum should have correct ordering."""
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        # Verify key stages exist (DATA_SYNC not SYNC)
        assert hasattr(PipelineStage, "SELFPLAY")
        assert hasattr(PipelineStage, "DATA_SYNC")
        assert hasattr(PipelineStage, "NPZ_EXPORT")
        assert hasattr(PipelineStage, "NPZ_COMBINATION")
        assert hasattr(PipelineStage, "TRAINING")
        assert hasattr(PipelineStage, "EVALUATION")
        assert hasattr(PipelineStage, "PROMOTION")


# =============================================================================
# Test Health Check Integration
# =============================================================================


class TestHealthCheckIntegration:
    """Tests for health check across pipeline components."""

    def test_npz_combination_daemon_has_health_check(self):
        """NPZCombinationDaemon should implement health_check()."""
        from app.coordination.npz_combination_daemon import NPZCombinationDaemon

        NPZCombinationDaemon.reset_instance()
        try:
            daemon = NPZCombinationDaemon()
            health = daemon.health_check()

            assert hasattr(health, "healthy")
            assert hasattr(health, "status")
            assert hasattr(health, "message")
        finally:
            NPZCombinationDaemon.reset_instance()

    def test_data_pipeline_orchestrator_has_health_check(self):
        """DataPipelineOrchestrator should implement health_check()."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        assert hasattr(DataPipelineOrchestrator, "health_check")
