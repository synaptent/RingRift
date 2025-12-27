"""Tests for stage_events module (December 2025).

Tests the StageEventBus class and related pipeline event infrastructure.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


class TestStageEvent:
    """Tests for StageEvent enum."""

    def test_all_events_exist(self):
        """Test all expected events are defined."""
        from app.coordination.stage_events import StageEvent

        # Selfplay stages
        assert StageEvent.SELFPLAY_COMPLETE.value == "selfplay_complete"
        assert StageEvent.CANONICAL_SELFPLAY_COMPLETE.value == "canonical_selfplay_complete"
        assert StageEvent.GPU_SELFPLAY_COMPLETE.value == "gpu_selfplay_complete"

        # Data processing stages
        assert StageEvent.SYNC_COMPLETE.value == "sync_complete"
        assert StageEvent.PARITY_VALIDATION_COMPLETE.value == "parity_validation_complete"
        assert StageEvent.NPZ_EXPORT_STARTED.value == "npz_export_started"
        assert StageEvent.NPZ_EXPORT_COMPLETE.value == "npz_export_complete"

        # Training stages
        assert StageEvent.TRAINING_COMPLETE.value == "training_complete"
        assert StageEvent.TRAINING_STARTED.value == "training_started"
        assert StageEvent.TRAINING_FAILED.value == "training_failed"

        # Evaluation stages
        assert StageEvent.EVALUATION_COMPLETE.value == "evaluation_complete"

        # Promotion stages
        assert StageEvent.PROMOTION_COMPLETE.value == "promotion_complete"

    def test_event_count(self):
        """Test all events are enumerated."""
        from app.coordination.stage_events import StageEvent

        # Should have at least 15 event types
        assert len(StageEvent) >= 15


class TestStageCompletionResult:
    """Tests for StageCompletionResult dataclass."""

    def test_basic_creation(self):
        """Test creating a result with minimal fields."""
        from app.coordination.stage_events import StageCompletionResult, StageEvent

        result = StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
        )
        assert result.event == StageEvent.SELFPLAY_COMPLETE
        assert result.success is True
        assert result.iteration == 1
        assert result.board_type == "square8"  # Default
        assert result.num_players == 2  # Default

    def test_full_creation(self):
        """Test creating a result with all fields."""
        from app.coordination.stage_events import StageCompletionResult, StageEvent

        result = StageCompletionResult(
            event=StageEvent.TRAINING_COMPLETE,
            success=True,
            iteration=5,
            timestamp="2025-12-27T12:00:00",
            board_type="hex8",
            num_players=4,
            games_generated=500,
            model_path="/models/test.pth",
            model_id="model-123",
            train_loss=0.5,
            val_loss=0.6,
            win_rate=0.75,
            elo_delta=150.0,
            promoted=True,
            promotion_reason="win_rate_threshold",
            metadata={"epochs": 20},
        )
        assert result.board_type == "hex8"
        assert result.num_players == 4
        assert result.model_path == "/models/test.pth"
        assert result.win_rate == 0.75
        assert result.promoted is True
        assert result.metadata["epochs"] == 20

    def test_error_fields(self):
        """Test error-related fields."""
        from app.coordination.stage_events import StageCompletionResult, StageEvent

        result = StageCompletionResult(
            event=StageEvent.TRAINING_FAILED,
            success=False,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
            error="CUDA out of memory",
            error_details="GPU memory exhausted at batch 512",
        )
        assert result.success is False
        assert result.error == "CUDA out of memory"
        assert result.error_details is not None

    def test_to_dict(self):
        """Test to_dict serialization."""
        from app.coordination.stage_events import StageCompletionResult, StageEvent

        result = StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
            games_generated=100,
        )
        d = result.to_dict()

        assert d["event"] == "selfplay_complete"
        assert d["success"] is True
        assert d["iteration"] == 1
        assert d["games_generated"] == 100
        assert d["board_type"] == "square8"
        assert "metadata" in d

    def test_from_dict(self):
        """Test from_dict deserialization."""
        from app.coordination.stage_events import StageCompletionResult, StageEvent

        data = {
            "event": "training_complete",
            "success": True,
            "iteration": 3,
            "timestamp": "2025-12-27T10:00:00",
            "board_type": "hex8",
            "num_players": 4,
            "model_path": "/models/test.pth",
        }
        result = StageCompletionResult.from_dict(data)

        assert result.event == StageEvent.TRAINING_COMPLETE
        assert result.success is True
        assert result.iteration == 3
        assert result.board_type == "hex8"
        assert result.model_path == "/models/test.pth"

    def test_from_dict_with_defaults(self):
        """Test from_dict with missing fields uses defaults."""
        from app.coordination.stage_events import StageCompletionResult, StageEvent

        data = {"event": "sync_complete"}
        result = StageCompletionResult.from_dict(data)

        assert result.event == StageEvent.SYNC_COMPLETE
        assert result.success is False  # Default
        assert result.iteration == 0  # Default
        assert result.board_type == "square8"  # Default

    def test_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        from app.coordination.stage_events import StageCompletionResult, StageEvent

        original = StageCompletionResult(
            event=StageEvent.EVALUATION_COMPLETE,
            success=True,
            iteration=10,
            timestamp="2025-12-27T10:00:00",
            win_rate=0.82,
            elo_delta=200.0,
            metadata={"games_played": 50},
        )
        d = original.to_dict()
        restored = StageCompletionResult.from_dict(d)

        assert restored.event == original.event
        assert restored.success == original.success
        assert restored.win_rate == original.win_rate
        assert restored.elo_delta == original.elo_delta


class TestStageEventBus:
    """Tests for StageEventBus class."""

    @pytest.fixture
    def bus(self):
        """Create a fresh event bus."""
        from app.coordination.stage_events import StageEventBus

        return StageEventBus(max_history=50)

    def test_init(self, bus):
        """Test event bus initialization."""
        assert bus._subscribers == {}
        assert bus._history == []
        assert bus._max_history == 50

    def test_subscribe(self, bus):
        """Test subscribing to an event."""
        from app.coordination.stage_events import StageEvent

        async def my_callback(result):
            pass

        bus.subscribe(StageEvent.SELFPLAY_COMPLETE, my_callback)

        assert bus.subscriber_count(StageEvent.SELFPLAY_COMPLETE) == 1

    def test_subscribe_multiple(self, bus):
        """Test subscribing multiple callbacks."""
        from app.coordination.stage_events import StageEvent

        async def callback1(result):
            pass

        async def callback2(result):
            pass

        bus.subscribe(StageEvent.SELFPLAY_COMPLETE, callback1)
        bus.subscribe(StageEvent.SELFPLAY_COMPLETE, callback2)

        assert bus.subscriber_count(StageEvent.SELFPLAY_COMPLETE) == 2

    def test_subscribe_same_callback_once(self, bus):
        """Test that subscribing same callback twice doesn't duplicate."""
        from app.coordination.stage_events import StageEvent

        async def my_callback(result):
            pass

        bus.subscribe(StageEvent.SELFPLAY_COMPLETE, my_callback)
        bus.subscribe(StageEvent.SELFPLAY_COMPLETE, my_callback)

        assert bus.subscriber_count(StageEvent.SELFPLAY_COMPLETE) == 1

    def test_unsubscribe(self, bus):
        """Test unsubscribing from an event."""
        from app.coordination.stage_events import StageEvent

        async def my_callback(result):
            pass

        bus.subscribe(StageEvent.SELFPLAY_COMPLETE, my_callback)
        assert bus.subscriber_count(StageEvent.SELFPLAY_COMPLETE) == 1

        result = bus.unsubscribe(StageEvent.SELFPLAY_COMPLETE, my_callback)
        assert result is True
        assert bus.subscriber_count(StageEvent.SELFPLAY_COMPLETE) == 0

    def test_unsubscribe_not_found(self, bus):
        """Test unsubscribing when callback not subscribed."""
        from app.coordination.stage_events import StageEvent

        async def my_callback(result):
            pass

        result = bus.unsubscribe(StageEvent.SELFPLAY_COMPLETE, my_callback)
        assert result is False

    def test_clear_subscribers_specific(self, bus):
        """Test clearing subscribers for a specific event."""
        from app.coordination.stage_events import StageEvent

        async def callback1(result):
            pass

        async def callback2(result):
            pass

        bus.subscribe(StageEvent.SELFPLAY_COMPLETE, callback1)
        bus.subscribe(StageEvent.TRAINING_COMPLETE, callback2)

        count = bus.clear_subscribers(StageEvent.SELFPLAY_COMPLETE)
        assert count == 1
        assert bus.subscriber_count(StageEvent.SELFPLAY_COMPLETE) == 0
        assert bus.subscriber_count(StageEvent.TRAINING_COMPLETE) == 1

    def test_clear_subscribers_all(self, bus):
        """Test clearing all subscribers."""
        from app.coordination.stage_events import StageEvent

        async def callback1(result):
            pass

        async def callback2(result):
            pass

        bus.subscribe(StageEvent.SELFPLAY_COMPLETE, callback1)
        bus.subscribe(StageEvent.TRAINING_COMPLETE, callback2)

        count = bus.clear_subscribers()
        assert count == 2
        assert bus.subscriber_count(StageEvent.SELFPLAY_COMPLETE) == 0
        assert bus.subscriber_count(StageEvent.TRAINING_COMPLETE) == 0

    @pytest.mark.asyncio
    async def test_emit(self, bus):
        """Test emitting an event invokes callbacks."""
        from app.coordination.stage_events import StageEvent, StageCompletionResult

        callback = AsyncMock()
        bus.subscribe(StageEvent.SELFPLAY_COMPLETE, callback)

        result = StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
            games_generated=100,
        )
        invoked = await bus.emit(result)

        assert invoked == 1
        callback.assert_called_once_with(result)

    @pytest.mark.asyncio
    async def test_emit_no_subscribers(self, bus):
        """Test emitting event with no subscribers."""
        from app.coordination.stage_events import StageEvent, StageCompletionResult

        result = StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
        )
        invoked = await bus.emit(result)

        assert invoked == 0

    @pytest.mark.asyncio
    async def test_emit_callback_error(self, bus):
        """Test callback errors don't affect other callbacks."""
        from app.coordination.stage_events import StageEvent, StageCompletionResult

        callback1 = AsyncMock(side_effect=RuntimeError("Test error"))
        callback2 = AsyncMock()

        bus.subscribe(StageEvent.SELFPLAY_COMPLETE, callback1)
        bus.subscribe(StageEvent.SELFPLAY_COMPLETE, callback2)

        result = StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
        )
        invoked = await bus.emit(result)

        # First callback failed, second succeeded
        assert invoked == 1
        callback2.assert_called_once()

        # Error should be recorded
        errors = bus.get_callback_errors()
        assert len(errors) == 1
        assert "Test error" in errors[0]["error"]

    @pytest.mark.asyncio
    async def test_emit_adds_to_history(self, bus):
        """Test emitting events adds them to history."""
        from app.coordination.stage_events import StageEvent, StageCompletionResult

        result = StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
        )
        await bus.emit(result)

        history = bus.get_history()
        assert len(history) == 1
        assert history[0].event == StageEvent.SELFPLAY_COMPLETE

    @pytest.mark.asyncio
    async def test_emit_history_limit(self):
        """Test history doesn't exceed max_history."""
        from app.coordination.stage_events import StageEventBus, StageEvent, StageCompletionResult

        bus = StageEventBus(max_history=3)

        for i in range(5):
            result = StageCompletionResult(
                event=StageEvent.SELFPLAY_COMPLETE,
                success=True,
                iteration=i,
                timestamp=f"2025-12-27T10:0{i}:00",
            )
            await bus.emit(result)

        history = bus.get_history()
        # Should only have last 3 events
        assert len(history) == 3
        # Newest first, so iterations should be 4, 3, 2
        assert history[0].iteration == 4
        assert history[1].iteration == 3
        assert history[2].iteration == 2

    @pytest.mark.asyncio
    async def test_emit_and_wait(self, bus):
        """Test emit_and_wait waits for all callbacks."""
        from app.coordination.stage_events import StageEvent, StageCompletionResult

        results = []

        async def slow_callback(result):
            await asyncio.sleep(0.01)
            results.append("slow")

        async def fast_callback(result):
            results.append("fast")

        bus.subscribe(StageEvent.SELFPLAY_COMPLETE, slow_callback)
        bus.subscribe(StageEvent.SELFPLAY_COMPLETE, fast_callback)

        result = StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
        )
        await bus.emit_and_wait(result)

        assert len(results) == 2
        assert "slow" in results
        assert "fast" in results

    @pytest.mark.asyncio
    async def test_emit_and_wait_timeout(self, bus):
        """Test emit_and_wait respects timeout."""
        from app.coordination.stage_events import StageEvent, StageCompletionResult

        async def very_slow_callback(result):
            await asyncio.sleep(10)  # Very slow

        bus.subscribe(StageEvent.SELFPLAY_COMPLETE, very_slow_callback)

        result = StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
        )
        results = await bus.emit_and_wait(result, timeout=0.01)

        assert results == []  # Timed out

    def test_get_history_filter(self, bus):
        """Test get_history with event filter."""
        from app.coordination.stage_events import StageEvent, StageCompletionResult

        # Manually add to history
        result1 = StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True, iteration=1, timestamp="2025-12-27T10:00:00"
        )
        result2 = StageCompletionResult(
            event=StageEvent.TRAINING_COMPLETE,
            success=True, iteration=1, timestamp="2025-12-27T11:00:00"
        )
        bus._history = [result1, result2]

        selfplay_history = bus.get_history(event=StageEvent.SELFPLAY_COMPLETE)
        assert len(selfplay_history) == 1
        assert selfplay_history[0].event == StageEvent.SELFPLAY_COMPLETE

    def test_get_stats(self, bus):
        """Test get_stats returns correct stats."""
        from app.coordination.stage_events import StageEvent

        async def callback(result):
            pass

        bus.subscribe(StageEvent.SELFPLAY_COMPLETE, callback)
        bus.subscribe(StageEvent.TRAINING_COMPLETE, callback)

        stats = bus.get_stats()

        assert stats["total_subscribers"] == 2
        assert "selfplay_complete" in stats["subscribers_by_event"]
        assert "training_complete" in stats["subscribers_by_event"]
        assert "supported_events" in stats
        assert len(stats["supported_events"]) > 0

    def test_set_logger(self, bus):
        """Test set_logger sets custom logging function."""
        log_messages = []

        def my_logger(msg):
            log_messages.append(msg)

        bus.set_logger(my_logger)
        assert bus._log_callback is my_logger


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_reset_event_bus(self):
        """Test reset_event_bus clears the global bus."""
        from app.coordination import stage_events

        # Reset first
        stage_events.reset_event_bus()

        # Create bus (suppress deprecation warning)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            bus1 = stage_events.get_event_bus()

        # Reset
        stage_events.reset_event_bus()

        # New bus should be different
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            bus2 = stage_events.get_event_bus()

        assert bus1 is not bus2

        # Cleanup
        stage_events.reset_event_bus()

    def test_get_event_bus_emits_deprecation(self):
        """Test get_event_bus emits deprecation warning."""
        from app.coordination import stage_events
        import warnings

        stage_events.reset_event_bus()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            stage_events.get_event_bus()

            assert len(w) >= 1
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

        stage_events.reset_event_bus()

    def test_create_pipeline_callbacks(self):
        """Test create_pipeline_callbacks returns expected callbacks."""
        from app.coordination.stage_events import (
            StageEvent,
            create_pipeline_callbacks,
        )

        callbacks = create_pipeline_callbacks()

        assert StageEvent.SELFPLAY_COMPLETE in callbacks
        assert StageEvent.SYNC_COMPLETE in callbacks
        assert StageEvent.TRAINING_COMPLETE in callbacks
        assert StageEvent.EVALUATION_COMPLETE in callbacks

        # All callbacks should be callable
        for cb in callbacks.values():
            assert callable(cb)

    @pytest.mark.asyncio
    async def test_pipeline_callbacks_execute(self):
        """Test pipeline callbacks execute without error."""
        from app.coordination.stage_events import (
            StageEvent,
            StageCompletionResult,
            create_pipeline_callbacks,
        )

        callbacks = create_pipeline_callbacks()

        # Test selfplay callback
        result = StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
            games_generated=100,
        )
        await callbacks[StageEvent.SELFPLAY_COMPLETE](result)

        # Test training callback
        result = StageCompletionResult(
            event=StageEvent.TRAINING_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
            model_path="/models/test.pth",
        )
        await callbacks[StageEvent.TRAINING_COMPLETE](result)

    def test_register_standard_callbacks(self):
        """Test register_standard_callbacks subscribes to bus."""
        from app.coordination.stage_events import (
            StageEvent,
            StageEventBus,
            register_standard_callbacks,
        )

        bus = StageEventBus()
        register_standard_callbacks(bus)

        # Should have subscribers for the standard events
        assert bus.subscriber_count(StageEvent.SELFPLAY_COMPLETE) >= 1
        assert bus.subscriber_count(StageEvent.SYNC_COMPLETE) >= 1
        assert bus.subscriber_count(StageEvent.TRAINING_COMPLETE) >= 1
        assert bus.subscriber_count(StageEvent.EVALUATION_COMPLETE) >= 1

    def test_module_exports(self):
        """Test __all__ contains expected exports."""
        from app.coordination import stage_events

        assert "StageEvent" in stage_events.__all__
        assert "StageCompletionResult" in stage_events.__all__
        assert "StageEventBus" in stage_events.__all__
        assert "get_event_bus" in stage_events.__all__
        assert "reset_event_bus" in stage_events.__all__
        assert "create_pipeline_callbacks" in stage_events.__all__


class TestDeadLetterQueue:
    """Tests for DLQ integration in StageEventBus."""

    @pytest.mark.asyncio
    async def test_emit_captures_to_dlq_on_error(self):
        """Test emit captures to DLQ when callback fails and DLQ is attached."""
        from app.coordination.stage_events import (
            StageEvent,
            StageEventBus,
            StageCompletionResult,
        )

        bus = StageEventBus()

        # Mock DLQ
        mock_dlq = MagicMock()
        bus._dlq = mock_dlq

        async def failing_callback(result):
            raise RuntimeError("Callback failed")

        bus.subscribe(StageEvent.SELFPLAY_COMPLETE, failing_callback)

        result = StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
        )
        await bus.emit(result)

        # DLQ capture should have been called
        mock_dlq.capture.assert_called_once()
        call_kwargs = mock_dlq.capture.call_args[1]
        assert call_kwargs["event_type"] == "selfplay_complete"
        assert "Callback failed" in call_kwargs["error"]

    @pytest.mark.asyncio
    async def test_emit_handles_dlq_error_gracefully(self):
        """Test emit handles DLQ errors gracefully."""
        from app.coordination.stage_events import (
            StageEvent,
            StageEventBus,
            StageCompletionResult,
        )

        bus = StageEventBus()

        # Mock DLQ that throws
        mock_dlq = MagicMock()
        mock_dlq.capture.side_effect = RuntimeError("DLQ failed")
        bus._dlq = mock_dlq

        async def failing_callback(result):
            raise RuntimeError("Callback failed")

        bus.subscribe(StageEvent.SELFPLAY_COMPLETE, failing_callback)

        result = StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
        )

        # Should not raise even though DLQ fails
        invoked = await bus.emit(result)
        assert invoked == 0  # Callback failed
