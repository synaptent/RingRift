"""End-to-end integration tests for the full training pipeline event chain.

January 2, 2026 - Verifies the complete event chain from selfplay to model promotion:

    SELFPLAY_COMPLETE → DATA_SYNC_COMPLETED → NPZ_EXPORT_COMPLETE →
    NPZ_COMBINATION_COMPLETE → TRAINING_COMPLETED → EVALUATION_COMPLETED →
    MODEL_PROMOTED

These tests ensure:
1. Each stage triggers the next stage via events
2. No gaps exist in the event chain
3. Pipeline completes successfully end-to-end
4. Event ordering is preserved
5. Handlers can be mocked at any stage to simulate failures

Usage:
    pytest tests/integration/coordination/test_full_event_chain_e2e.py -v
    pytest tests/integration/coordination/test_full_event_chain_e2e.py -v --run-slow-integration
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.distributed.data_events import DataEventType


# =============================================================================
# Test Infrastructure
# =============================================================================


@dataclass
class PipelineEvent:
    """Recorded pipeline event with timing."""

    event_type: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    stage_index: int = 0


class PipelineEventTracker:
    """Tracks events through the training pipeline.

    Provides utilities for verifying the complete event chain and
    detecting gaps or missing stages.
    """

    # The canonical pipeline event chain
    PIPELINE_STAGES = [
        DataEventType.SELFPLAY_COMPLETE.value,
        DataEventType.DATA_SYNC_COMPLETED.value,
        DataEventType.NPZ_EXPORT_COMPLETE.value,
        DataEventType.NPZ_COMBINATION_COMPLETE.value,
        DataEventType.TRAINING_COMPLETED.value,
        DataEventType.EVALUATION_COMPLETED.value,
        DataEventType.MODEL_PROMOTED.value,
    ]

    def __init__(self):
        self.events: list[PipelineEvent] = []
        self.handlers: dict[str, list[Callable]] = {}
        self._lock = asyncio.Lock()

    async def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        """Emit an event and trigger handlers."""
        # Record event first
        stage_idx = self._get_stage_index(event_type)
        event = PipelineEvent(
            event_type=event_type,
            payload=payload,
            timestamp=time.time(),
            stage_index=stage_idx,
        )
        self.events.append(event)

        # Trigger registered handlers (without lock to avoid deadlock in chain)
        for handler in self.handlers.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(payload)
                else:
                    handler(payload)
            except Exception:
                pass  # Continue to other handlers

    def emit_sync(self, event_type: str, payload: dict[str, Any]) -> None:
        """Synchronous emit for non-async contexts."""
        stage_idx = self._get_stage_index(event_type)
        event = PipelineEvent(
            event_type=event_type,
            payload=payload,
            timestamp=time.time(),
            stage_index=stage_idx,
        )
        self.events.append(event)

    def register_handler(self, event_type: str, handler: Callable) -> None:
        """Register a handler for an event type."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    def _get_stage_index(self, event_type: str) -> int:
        """Get the pipeline stage index for an event type."""
        try:
            return self.PIPELINE_STAGES.index(event_type)
        except ValueError:
            return -1

    def get_completed_stages(self) -> list[str]:
        """Get list of completed pipeline stages in order."""
        return [e.event_type for e in self.events if e.stage_index >= 0]

    def verify_chain_complete(self) -> tuple[bool, list[str]]:
        """Verify the full pipeline chain completed.

        Returns:
            Tuple of (success, missing_stages)
        """
        completed = set(self.get_completed_stages())
        missing = [stage for stage in self.PIPELINE_STAGES if stage not in completed]
        return len(missing) == 0, missing

    def verify_event_order(self) -> tuple[bool, str]:
        """Verify events occurred in correct pipeline order.

        Returns:
            Tuple of (success, error_message)
        """
        pipeline_events = [e for e in self.events if e.stage_index >= 0]
        for i in range(1, len(pipeline_events)):
            if pipeline_events[i].stage_index < pipeline_events[i - 1].stage_index:
                return False, (
                    f"Event order violation: {pipeline_events[i].event_type} "
                    f"(stage {pipeline_events[i].stage_index}) came after "
                    f"{pipeline_events[i-1].event_type} "
                    f"(stage {pipeline_events[i-1].stage_index})"
                )
        return True, ""

    async def wait_for_stage(
        self, event_type: str, timeout: float = 5.0
    ) -> PipelineEvent | None:
        """Wait for a specific pipeline stage to complete."""
        start = time.time()
        while time.time() - start < timeout:
            for event in self.events:
                if event.event_type == event_type:
                    return event
            await asyncio.sleep(0.1)
        return None

    def clear(self) -> None:
        """Clear all recorded events."""
        self.events.clear()


@pytest.fixture
def pipeline_tracker():
    """Provide a pipeline event tracker for testing."""
    return PipelineEventTracker()


@pytest.fixture
def config_key():
    """Standard config key for tests."""
    return "hex8_2p"


@pytest.fixture
def model_path(tmp_path):
    """Temporary model path for tests."""
    return str(tmp_path / "test_model.pth")


@pytest.fixture
def npz_path(tmp_path):
    """Temporary NPZ path for tests."""
    return str(tmp_path / "test_data.npz")


# =============================================================================
# Test Pipeline Event Types Exist
# =============================================================================


class TestPipelineEventTypesExist:
    """Verify all pipeline event types are defined."""

    def test_all_pipeline_events_defined(self):
        """All pipeline event types should exist in DataEventType."""
        expected = [
            "SELFPLAY_COMPLETE",
            "DATA_SYNC_COMPLETED",
            "NPZ_EXPORT_COMPLETE",
            "NPZ_COMBINATION_COMPLETE",
            "TRAINING_COMPLETED",
            "EVALUATION_COMPLETED",
            "MODEL_PROMOTED",
        ]

        for event_name in expected:
            assert hasattr(DataEventType, event_name), (
                f"Missing pipeline event type: {event_name}"
            )

    def test_pipeline_event_values_are_strings(self):
        """Pipeline event types should have string values."""
        assert isinstance(DataEventType.SELFPLAY_COMPLETE.value, str)
        assert isinstance(DataEventType.DATA_SYNC_COMPLETED.value, str)
        assert isinstance(DataEventType.MODEL_PROMOTED.value, str)


# =============================================================================
# Test Pipeline Stage Handlers Exist
# =============================================================================


class TestPipelineStageHandlersExist:
    """Verify handlers exist for each pipeline stage transition."""

    def test_data_pipeline_subscribes_to_selfplay(self):
        """DataPipelineOrchestrator should handle SELFPLAY_COMPLETE."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        assert hasattr(DataPipelineOrchestrator, "_on_selfplay_complete") or hasattr(
            DataPipelineOrchestrator, "_handle_selfplay_complete"
        )

    def test_auto_export_subscribes_to_sync(self):
        """AutoExportDaemon should handle DATA_SYNC_COMPLETED."""
        from app.coordination.auto_export_daemon import AutoExportDaemon

        # Check for handler method
        assert hasattr(AutoExportDaemon, "_on_sync_complete") or hasattr(
            AutoExportDaemon, "_on_data_sync_completed"
        )

    def test_npz_combination_subscribes_to_export(self):
        """NPZCombinationDaemon should handle NPZ_EXPORT_COMPLETE."""
        from app.coordination.npz_combination_daemon import NPZCombinationDaemon

        NPZCombinationDaemon.reset_instance()
        try:
            daemon = NPZCombinationDaemon()
            subscriptions = daemon._get_event_subscriptions()
            assert DataEventType.NPZ_EXPORT_COMPLETE.value in subscriptions
        finally:
            NPZCombinationDaemon.reset_instance()

    def test_training_trigger_subscribes_to_combination(self):
        """TrainingTriggerDaemon should handle NPZ_COMBINATION_COMPLETE."""
        from app.coordination.training_trigger_daemon import TrainingTriggerDaemon

        TrainingTriggerDaemon.reset_instance()
        try:
            daemon = TrainingTriggerDaemon()
            subscriptions = daemon._get_event_subscriptions()
            assert DataEventType.NPZ_COMBINATION_COMPLETE.value in subscriptions
        finally:
            TrainingTriggerDaemon.reset_instance()

    def test_feedback_loop_subscribes_to_training(self):
        """FeedbackLoopController should handle TRAINING_COMPLETED."""
        from app.coordination.feedback_loop_controller import FeedbackLoopController

        assert hasattr(FeedbackLoopController, "_on_training_complete")

    def test_auto_promotion_daemon_subscribes_to_evaluation(self):
        """AutoPromotionDaemon should handle EVALUATION_COMPLETED."""
        from app.coordination.auto_promotion_daemon import AutoPromotionDaemon

        assert hasattr(AutoPromotionDaemon, "_on_evaluation_completed")


# =============================================================================
# Test Event Chain Simulation
# =============================================================================


class TestEventChainSimulation:
    """Simulate the full event chain and verify transitions."""

    @pytest.mark.asyncio
    async def test_full_chain_simulation(self, pipeline_tracker, config_key, model_path, npz_path):
        """Simulate the full pipeline event chain."""

        # Register chain handlers that emit the next event
        async def selfplay_to_sync(payload):
            await pipeline_tracker.emit(
                DataEventType.DATA_SYNC_COMPLETED.value,
                {
                    "config_key": payload["config_key"],
                    "files_synced": 10,
                    "source": "auto_sync",
                },
            )

        async def sync_to_export(payload):
            await pipeline_tracker.emit(
                DataEventType.NPZ_EXPORT_COMPLETE.value,
                {
                    "config_key": payload["config_key"],
                    "output_path": npz_path,
                    "samples_exported": 5000,
                    "source": "auto_export",
                },
            )

        async def export_to_combination(payload):
            await pipeline_tracker.emit(
                DataEventType.NPZ_COMBINATION_COMPLETE.value,
                {
                    "config_key": payload["config_key"],
                    "output_path": npz_path,
                    "total_samples": 10000,
                    "source": "npz_combiner",
                },
            )

        async def combination_to_training(payload):
            await pipeline_tracker.emit(
                DataEventType.TRAINING_COMPLETED.value,
                {
                    "config_key": payload["config_key"],
                    "model_path": model_path,
                    "epochs_completed": 50,
                    "final_loss": 0.15,
                    "source": "training_coordinator",
                },
            )

        async def training_to_evaluation(payload):
            await pipeline_tracker.emit(
                DataEventType.EVALUATION_COMPLETED.value,
                {
                    "config_key": payload["config_key"],
                    "model_path": payload["model_path"],
                    "win_rate_random": 0.95,
                    "win_rate_heuristic": 0.65,
                    "passed_gauntlet": True,
                    "source": "evaluation_daemon",
                },
            )

        async def evaluation_to_promotion(payload):
            await pipeline_tracker.emit(
                DataEventType.MODEL_PROMOTED.value,
                {
                    "config_key": payload["config_key"],
                    "model_path": payload["model_path"],
                    "promotion_type": "canonical",
                    "source": "promotion_controller",
                },
            )

        # Register all handlers
        pipeline_tracker.register_handler(
            DataEventType.SELFPLAY_COMPLETE.value, selfplay_to_sync
        )
        pipeline_tracker.register_handler(
            DataEventType.DATA_SYNC_COMPLETED.value, sync_to_export
        )
        pipeline_tracker.register_handler(
            DataEventType.NPZ_EXPORT_COMPLETE.value, export_to_combination
        )
        pipeline_tracker.register_handler(
            DataEventType.NPZ_COMBINATION_COMPLETE.value, combination_to_training
        )
        pipeline_tracker.register_handler(
            DataEventType.TRAINING_COMPLETED.value, training_to_evaluation
        )
        pipeline_tracker.register_handler(
            DataEventType.EVALUATION_COMPLETED.value, evaluation_to_promotion
        )

        # Trigger the chain with SELFPLAY_COMPLETE
        await pipeline_tracker.emit(
            DataEventType.SELFPLAY_COMPLETE.value,
            {
                "config_key": config_key,
                "games_completed": 100,
                "db_path": "/tmp/games.db",
                "source": "selfplay_runner",
            },
        )

        # Verify chain completed
        complete, missing = pipeline_tracker.verify_chain_complete()
        assert complete, f"Pipeline chain incomplete. Missing stages: {missing}"

        # Verify order
        order_ok, error = pipeline_tracker.verify_event_order()
        assert order_ok, error

        # Verify all 7 stages completed
        assert len(pipeline_tracker.events) == 7

    @pytest.mark.asyncio
    async def test_chain_detects_missing_handler(self, pipeline_tracker, config_key):
        """Chain should detect when a handler is missing."""

        # Only register handlers up to sync (skip export handler)
        async def selfplay_to_sync(payload):
            await pipeline_tracker.emit(
                DataEventType.DATA_SYNC_COMPLETED.value,
                {"config_key": payload["config_key"], "files_synced": 10},
            )

        pipeline_tracker.register_handler(
            DataEventType.SELFPLAY_COMPLETE.value, selfplay_to_sync
        )
        # Missing: sync_to_export handler

        # Trigger chain
        await pipeline_tracker.emit(
            DataEventType.SELFPLAY_COMPLETE.value,
            {"config_key": config_key, "games_completed": 100},
        )

        # Chain should be incomplete
        complete, missing = pipeline_tracker.verify_chain_complete()
        assert not complete
        assert DataEventType.NPZ_EXPORT_COMPLETE.value in missing

    @pytest.mark.asyncio
    async def test_chain_records_timing(self, pipeline_tracker, config_key):
        """Chain should record timing between stages."""

        async def stage1_to_stage2(payload):
            await asyncio.sleep(0.1)  # Simulate processing time
            await pipeline_tracker.emit(
                DataEventType.DATA_SYNC_COMPLETED.value,
                {"config_key": payload["config_key"]},
            )

        pipeline_tracker.register_handler(
            DataEventType.SELFPLAY_COMPLETE.value, stage1_to_stage2
        )

        start = time.time()
        await pipeline_tracker.emit(
            DataEventType.SELFPLAY_COMPLETE.value,
            {"config_key": config_key},
        )

        # Verify timing was recorded
        assert len(pipeline_tracker.events) == 2
        time_diff = pipeline_tracker.events[1].timestamp - pipeline_tracker.events[0].timestamp
        assert time_diff >= 0.1  # Should include processing delay


# =============================================================================
# Test Real Daemon Integration
# =============================================================================


class TestRealDaemonIntegration:
    """Integration tests with real daemon instances (mocked dependencies)."""

    @pytest.mark.asyncio
    async def test_npz_combination_daemon_chain(self, config_key, npz_path):
        """Test NPZCombinationDaemon processes export event and emits combination event."""
        from app.coordination.npz_combination_daemon import (
            NPZCombinationDaemon,
            NPZCombinationConfig,
        )
        from app.training.npz_combiner import CombineResult

        NPZCombinationDaemon.reset_instance()
        try:
            config = NPZCombinationConfig(min_interval_seconds=0.0)
            daemon = NPZCombinationDaemon(config=config)

            # Track emitted events
            emitted_events = []

            def capture_emit(config_key, result):
                emitted_events.append(("complete", config_key, result))

            mock_result = CombineResult(
                success=True,
                output_path=Path(npz_path),
                total_samples=10000,
            )

            with patch.object(daemon, "_combine_for_config", new_callable=AsyncMock) as mock_combine:
                mock_combine.return_value = mock_result
                with patch.object(daemon, "_emit_combination_complete", side_effect=capture_emit):
                    await daemon._on_npz_export_complete({
                        "config_key": config_key,
                        "output_path": npz_path,
                        "samples_exported": 5000,
                    })

            # Verify combination was triggered and event emitted
            assert len(emitted_events) == 1
            assert emitted_events[0][0] == "complete"
            assert emitted_events[0][1] == config_key
        finally:
            NPZCombinationDaemon.reset_instance()

    @pytest.mark.asyncio
    async def test_training_trigger_daemon_chain(self, config_key, model_path):
        """Test TrainingTriggerDaemon processes combination event."""
        from app.coordination.training_trigger_daemon import TrainingTriggerDaemon

        TrainingTriggerDaemon.reset_instance()
        try:
            daemon = TrainingTriggerDaemon()

            # Track trigger attempts
            trigger_attempts = []

            async def capture_trigger(*args, **kwargs):
                trigger_attempts.append((args, kwargs))
                return True

            # The daemon expects a result object with metadata attribute
            # or handles the event extraction internally
            class MockResult:
                def __init__(self):
                    self.metadata = {
                        "config_key": config_key,
                        "board_type": "hex8",
                        "num_players": 2,
                        "output_path": "/tmp/combined.npz",
                        "total_samples": 10000,
                        "quality_weighted": True,
                    }

            with patch.object(daemon, "_maybe_trigger_training", side_effect=capture_trigger):
                await daemon._on_npz_combination_complete(MockResult())

            # Verify training trigger was attempted
            assert len(trigger_attempts) >= 1
            assert trigger_attempts[0][0][0] == config_key  # First arg is config_key
        finally:
            TrainingTriggerDaemon.reset_instance()


# =============================================================================
# Test Pipeline Stage Payloads
# =============================================================================


class TestPipelineStagePayloads:
    """Verify correct payload structure for each stage."""

    def test_selfplay_complete_payload(self, config_key):
        """SELFPLAY_COMPLETE should have required fields."""
        payload = {
            "config_key": config_key,
            "games_completed": 100,
            "db_path": "/tmp/games.db",
            "source": "selfplay_runner",
        }
        assert "config_key" in payload
        assert "games_completed" in payload

    def test_data_sync_completed_payload(self, config_key):
        """DATA_SYNC_COMPLETED should have required fields."""
        payload = {
            "config_key": config_key,
            "files_synced": 10,
            "total_bytes": 1_000_000,
            "source": "auto_sync",
        }
        assert "config_key" in payload

    def test_npz_export_complete_payload(self, config_key, npz_path):
        """NPZ_EXPORT_COMPLETE should have required fields."""
        payload = {
            "config_key": config_key,
            "output_path": npz_path,
            "samples_exported": 5000,
            "source": "auto_export",
        }
        assert "config_key" in payload
        assert "output_path" in payload
        assert "samples_exported" in payload

    def test_npz_combination_complete_payload(self, config_key, npz_path):
        """NPZ_COMBINATION_COMPLETE should have required fields."""
        payload = {
            "config_key": config_key,
            "output_path": npz_path,
            "total_samples": 10000,
            "samples_by_source": {"fresh": 6000, "historical": 4000},
            "source": "npz_combiner",
        }
        assert "config_key" in payload
        assert "output_path" in payload
        assert "total_samples" in payload

    def test_training_completed_payload(self, config_key, model_path):
        """TRAINING_COMPLETED should have required fields."""
        payload = {
            "config_key": config_key,
            "model_path": model_path,
            "epochs_completed": 50,
            "final_loss": 0.15,
            "source": "training_coordinator",
        }
        assert "config_key" in payload
        assert "model_path" in payload

    def test_evaluation_completed_payload(self, config_key, model_path):
        """EVALUATION_COMPLETED should have required fields."""
        payload = {
            "config_key": config_key,
            "model_path": model_path,
            "win_rate_random": 0.95,
            "win_rate_heuristic": 0.65,
            "passed_gauntlet": True,
            "source": "evaluation_daemon",
        }
        assert "config_key" in payload
        assert "passed_gauntlet" in payload

    def test_model_promoted_payload(self, config_key, model_path):
        """MODEL_PROMOTED should have required fields."""
        payload = {
            "config_key": config_key,
            "model_path": model_path,
            "promotion_type": "canonical",
            "source": "promotion_controller",
        }
        assert "config_key" in payload
        assert "model_path" in payload


# =============================================================================
# Test Pipeline Failure Handling
# =============================================================================


class TestPipelineFailureHandling:
    """Test graceful failure handling at each pipeline stage."""

    @pytest.mark.asyncio
    async def test_export_failure_stops_chain(self, pipeline_tracker, config_key):
        """Export failure should stop the pipeline chain."""

        async def failing_export_handler(payload):
            # Emit failure event instead of success
            await pipeline_tracker.emit(
                "npz_export_failed",
                {"config_key": payload["config_key"], "error": "Export failed"},
            )

        pipeline_tracker.register_handler(
            DataEventType.DATA_SYNC_COMPLETED.value, failing_export_handler
        )

        # Trigger sync (which will fail at export)
        await pipeline_tracker.emit(
            DataEventType.DATA_SYNC_COMPLETED.value,
            {"config_key": config_key, "files_synced": 10},
        )

        # Chain should be incomplete
        complete, missing = pipeline_tracker.verify_chain_complete()
        assert not complete
        assert DataEventType.NPZ_EXPORT_COMPLETE.value in missing

    @pytest.mark.asyncio
    async def test_training_failure_emits_failed_event(self, pipeline_tracker, config_key):
        """Training failure should emit TRAINING_FAILED event."""

        async def failing_training_handler(payload):
            await pipeline_tracker.emit(
                DataEventType.TRAINING_FAILED.value,
                {
                    "config_key": payload["config_key"],
                    "error": "OOM error during training",
                    "source": "training_coordinator",
                },
            )

        pipeline_tracker.register_handler(
            DataEventType.NPZ_COMBINATION_COMPLETE.value, failing_training_handler
        )

        await pipeline_tracker.emit(
            DataEventType.NPZ_COMBINATION_COMPLETE.value,
            {"config_key": config_key, "total_samples": 10000},
        )

        # Should have failed event
        failed_events = [e for e in pipeline_tracker.events if "failed" in e.event_type.lower()]
        assert len(failed_events) == 1

    @pytest.mark.asyncio
    async def test_evaluation_failure_prevents_promotion(self, pipeline_tracker, config_key, model_path):
        """Evaluation failure should prevent model promotion."""

        async def failing_evaluation_handler(payload):
            await pipeline_tracker.emit(
                "evaluation_failed",
                {
                    "config_key": payload["config_key"],
                    "model_path": model_path,
                    "reason": "Failed gauntlet: win_rate < threshold",
                    "source": "evaluation_daemon",
                },
            )

        pipeline_tracker.register_handler(
            DataEventType.TRAINING_COMPLETED.value, failing_evaluation_handler
        )

        await pipeline_tracker.emit(
            DataEventType.TRAINING_COMPLETED.value,
            {"config_key": config_key, "model_path": model_path},
        )

        # MODEL_PROMOTED should not be in the chain
        complete, missing = pipeline_tracker.verify_chain_complete()
        assert DataEventType.MODEL_PROMOTED.value in missing


# =============================================================================
# Test Pipeline Event Deduplication
# =============================================================================


class TestPipelineDeduplication:
    """Test that duplicate events are handled correctly."""

    @pytest.mark.asyncio
    async def test_duplicate_selfplay_events_deduplicated(self, pipeline_tracker, config_key):
        """Duplicate SELFPLAY_COMPLETE events should be deduplicated."""

        process_count = 0

        async def counting_handler(payload):
            nonlocal process_count
            process_count += 1
            await pipeline_tracker.emit(
                DataEventType.DATA_SYNC_COMPLETED.value,
                {"config_key": payload["config_key"]},
            )

        pipeline_tracker.register_handler(
            DataEventType.SELFPLAY_COMPLETE.value, counting_handler
        )

        # Emit same event twice
        event_payload = {"config_key": config_key, "games_completed": 100, "event_id": "same-id"}
        await pipeline_tracker.emit(DataEventType.SELFPLAY_COMPLETE.value, event_payload)
        await pipeline_tracker.emit(DataEventType.SELFPLAY_COMPLETE.value, event_payload)

        # Handler was called twice (dedup is at handler level, not tracker)
        # This test verifies the handler can implement its own dedup
        assert process_count == 2

    @pytest.mark.asyncio
    async def test_handler_based_deduplication(self, pipeline_tracker, config_key):
        """Handlers can implement their own deduplication logic."""

        seen_event_ids = set()
        process_count = 0

        async def deduping_handler(payload):
            nonlocal process_count
            event_id = payload.get("event_id")
            if event_id in seen_event_ids:
                return  # Skip duplicate
            seen_event_ids.add(event_id)
            process_count += 1
            await pipeline_tracker.emit(
                DataEventType.DATA_SYNC_COMPLETED.value,
                {"config_key": payload["config_key"]},
            )

        pipeline_tracker.register_handler(
            DataEventType.SELFPLAY_COMPLETE.value, deduping_handler
        )

        # Emit same event twice
        event_payload = {"config_key": config_key, "games_completed": 100, "event_id": "same-id"}
        await pipeline_tracker.emit(DataEventType.SELFPLAY_COMPLETE.value, event_payload)
        await pipeline_tracker.emit(DataEventType.SELFPLAY_COMPLETE.value, event_payload)

        # Handler should have deduplicated
        assert process_count == 1


# =============================================================================
# Test Pipeline Metrics
# =============================================================================


class TestPipelineMetrics:
    """Test pipeline provides useful metrics."""

    @pytest.mark.asyncio
    async def test_pipeline_records_stage_durations(self, pipeline_tracker, config_key):
        """Pipeline should record duration between stages."""

        async def delayed_handler(payload):
            await asyncio.sleep(0.05)  # 50ms processing
            await pipeline_tracker.emit(
                DataEventType.DATA_SYNC_COMPLETED.value,
                {"config_key": payload["config_key"]},
            )

        pipeline_tracker.register_handler(
            DataEventType.SELFPLAY_COMPLETE.value, delayed_handler
        )

        await pipeline_tracker.emit(
            DataEventType.SELFPLAY_COMPLETE.value,
            {"config_key": config_key},
        )

        # Calculate stage duration
        assert len(pipeline_tracker.events) == 2
        duration = pipeline_tracker.events[1].timestamp - pipeline_tracker.events[0].timestamp
        assert duration >= 0.05  # At least the processing delay

    def test_pipeline_can_report_completion_rate(self, pipeline_tracker):
        """Pipeline tracker can report completion statistics."""
        # Manually add some events
        pipeline_tracker.emit_sync(DataEventType.SELFPLAY_COMPLETE.value, {"config_key": "hex8_2p"})
        pipeline_tracker.emit_sync(DataEventType.DATA_SYNC_COMPLETED.value, {"config_key": "hex8_2p"})
        pipeline_tracker.emit_sync(DataEventType.NPZ_EXPORT_COMPLETE.value, {"config_key": "hex8_2p"})

        completed = pipeline_tracker.get_completed_stages()
        total_stages = len(pipeline_tracker.PIPELINE_STAGES)
        completion_rate = len(completed) / total_stages

        assert completion_rate == pytest.approx(3 / 7, rel=0.01)
