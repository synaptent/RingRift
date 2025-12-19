"""
Tests for app.coordination.unified_event_coordinator module.

Tests the unified event coordination system that bridges events between:
- DataEventBus (in-memory async events)
- StageEventBus (pipeline stage events)
- CrossProcessEventQueue (SQLite-backed IPC)
"""

import asyncio
import pytest
from datetime import datetime
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

from app.coordination.unified_event_coordinator import (
    # Data classes
    CoordinatorStats,
    # Main class
    UnifiedEventCoordinator,
    # Functions
    get_event_coordinator,
    start_coordinator,
    stop_coordinator,
    get_coordinator_stats,
    # Event type mappings
    DATA_TO_CROSS_PROCESS_MAP,
    STAGE_TO_CROSS_PROCESS_MAP,
    CROSS_PROCESS_TO_DATA_MAP,
    # Async event emitters
    emit_training_started,
    emit_training_completed,
    emit_training_failed,
    emit_evaluation_completed,
    emit_sync_completed,
    emit_model_promoted,
    emit_selfplay_batch_completed,
)


# ============================================
# Test Fixtures
# ============================================

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton coordinator between tests."""
    UnifiedEventCoordinator._instance = None
    yield
    UnifiedEventCoordinator._instance = None


@pytest.fixture
def coordinator():
    """Create a fresh coordinator instance."""
    return UnifiedEventCoordinator()


@pytest.fixture
def mock_data_bus():
    """Create a mock DataEventBus."""
    bus = AsyncMock()
    bus.subscribe = MagicMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def mock_stage_bus():
    """Create a mock StageEventBus."""
    bus = AsyncMock()
    bus.subscribe = MagicMock()
    return bus


@pytest.fixture
def mock_cross_queue():
    """Create a mock CrossProcessEventQueue."""
    queue = MagicMock()
    queue.subscribe = MagicMock(return_value="subscriber-123")
    queue.unsubscribe = MagicMock()
    queue.publish = MagicMock()
    queue.poll = MagicMock(return_value=[])
    queue.ack = MagicMock()
    return queue


# ============================================
# Test Event Type Mappings
# ============================================

class TestEventTypeMappings:
    """Tests for event type mapping constants."""

    def test_data_to_cross_process_map_has_entries(self):
        """Test DATA_TO_CROSS_PROCESS_MAP has entries."""
        assert len(DATA_TO_CROSS_PROCESS_MAP) > 0

    def test_data_to_cross_process_map_training_events(self):
        """Test training events are mapped."""
        assert "training_started" in DATA_TO_CROSS_PROCESS_MAP
        assert "training_completed" in DATA_TO_CROSS_PROCESS_MAP
        assert "training_failed" in DATA_TO_CROSS_PROCESS_MAP

    def test_stage_to_cross_process_map_has_entries(self):
        """Test STAGE_TO_CROSS_PROCESS_MAP has entries."""
        assert len(STAGE_TO_CROSS_PROCESS_MAP) > 0

    def test_stage_to_cross_process_map_training(self):
        """Test stage training events are mapped."""
        assert "training_complete" in STAGE_TO_CROSS_PROCESS_MAP
        assert "training_started" in STAGE_TO_CROSS_PROCESS_MAP

    def test_cross_process_to_data_map_has_entries(self):
        """Test CROSS_PROCESS_TO_DATA_MAP has entries."""
        assert len(CROSS_PROCESS_TO_DATA_MAP) > 0

    def test_cross_process_to_data_map_reverse_mapping(self):
        """Test reverse mapping exists for key events."""
        assert "TRAINING_COMPLETED" in CROSS_PROCESS_TO_DATA_MAP
        assert "EVALUATION_COMPLETED" in CROSS_PROCESS_TO_DATA_MAP
        assert "MODEL_PROMOTED" in CROSS_PROCESS_TO_DATA_MAP


# ============================================
# Test CoordinatorStats
# ============================================

class TestCoordinatorStats:
    """Tests for CoordinatorStats dataclass."""

    def test_default_values(self):
        """Test default values are initialized."""
        stats = CoordinatorStats()

        assert stats.events_bridged_data_to_cross == 0
        assert stats.events_bridged_stage_to_cross == 0
        assert stats.events_bridged_cross_to_data == 0
        assert stats.events_dropped == 0
        assert stats.last_bridge_time is None
        assert stats.errors == []
        assert stats.start_time is None
        assert stats.is_running is False

    def test_custom_values(self):
        """Test setting custom values."""
        stats = CoordinatorStats(
            events_bridged_data_to_cross=10,
            events_bridged_stage_to_cross=5,
            is_running=True,
            start_time="2025-12-19T10:00:00",
        )

        assert stats.events_bridged_data_to_cross == 10
        assert stats.events_bridged_stage_to_cross == 5
        assert stats.is_running is True
        assert stats.start_time == "2025-12-19T10:00:00"


# ============================================
# Test UnifiedEventCoordinator
# ============================================

class TestUnifiedEventCoordinator:
    """Tests for UnifiedEventCoordinator class."""

    def test_singleton_pattern(self):
        """Test get_instance returns singleton."""
        instance1 = UnifiedEventCoordinator.get_instance()
        instance2 = UnifiedEventCoordinator.get_instance()
        assert instance1 is instance2

    def test_initialization(self, coordinator):
        """Test coordinator initialization."""
        assert coordinator._running is False
        assert coordinator._data_bus is None
        assert coordinator._stage_bus is None
        assert coordinator._cross_queue is None
        assert coordinator._stats.is_running is False

    @pytest.mark.asyncio
    async def test_start_sets_running(self, coordinator):
        """Test start sets running state."""
        with patch("app.coordination.unified_event_coordinator.HAS_DATA_EVENTS", False), \
             patch("app.coordination.unified_event_coordinator.HAS_STAGE_EVENTS", False), \
             patch("app.coordination.unified_event_coordinator.HAS_CROSS_PROCESS", False):
            result = await coordinator.start()

        assert result is True
        assert coordinator._running is True
        assert coordinator._stats.is_running is True
        assert coordinator._stats.start_time is not None

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_start_already_running(self, coordinator):
        """Test start when already running."""
        coordinator._running = True

        result = await coordinator.start()

        assert result is True  # Should return True even if already running

    @pytest.mark.asyncio
    async def test_stop_clears_running(self, coordinator):
        """Test stop clears running state."""
        coordinator._running = True
        coordinator._stats.is_running = True

        await coordinator.stop()

        assert coordinator._running is False
        assert coordinator._stats.is_running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_poll_task(self, coordinator):
        """Test stop cancels the poll task."""
        # Create a real cancelled task
        async def dummy_coro():
            await asyncio.sleep(100)

        task = asyncio.create_task(dummy_coro())
        coordinator._poll_task = task
        coordinator._running = True

        await coordinator.stop()

        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_start_with_data_bus(self, coordinator, mock_data_bus):
        """Test start subscribes to data bus."""
        with patch("app.coordination.unified_event_coordinator.HAS_DATA_EVENTS", True), \
             patch("app.coordination.unified_event_coordinator.get_data_event_bus", return_value=mock_data_bus), \
             patch("app.coordination.unified_event_coordinator.DataEventType", MagicMock()), \
             patch("app.coordination.unified_event_coordinator.HAS_STAGE_EVENTS", False), \
             patch("app.coordination.unified_event_coordinator.HAS_CROSS_PROCESS", False):
            await coordinator.start()

        assert coordinator._data_bus is mock_data_bus
        assert mock_data_bus.subscribe.called

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_start_with_cross_process(self, coordinator, mock_cross_queue):
        """Test start subscribes to cross process queue."""
        with patch("app.coordination.unified_event_coordinator.HAS_DATA_EVENTS", False), \
             patch("app.coordination.unified_event_coordinator.HAS_STAGE_EVENTS", False), \
             patch("app.coordination.unified_event_coordinator.HAS_CROSS_PROCESS", True), \
             patch("app.coordination.unified_event_coordinator.get_cross_process_queue", return_value=mock_cross_queue):
            await coordinator.start()

        assert coordinator._cross_queue is mock_cross_queue
        assert coordinator._subscriber_id == "subscriber-123"
        mock_cross_queue.subscribe.assert_called_once()

        await coordinator.stop()

    def test_get_stats(self, coordinator):
        """Test get_stats returns stats."""
        coordinator._stats.events_bridged_data_to_cross = 10

        stats = coordinator.get_stats()

        assert stats.events_bridged_data_to_cross == 10

    def test_register_handler(self, coordinator):
        """Test registering a custom handler."""
        handler = MagicMock()

        coordinator.register_handler("TRAINING_COMPLETED", handler)

        assert "TRAINING_COMPLETED" in coordinator._event_handlers
        assert handler in coordinator._event_handlers["TRAINING_COMPLETED"]

    def test_register_multiple_handlers(self, coordinator):
        """Test registering multiple handlers for same event."""
        handler1 = MagicMock()
        handler2 = MagicMock()

        coordinator.register_handler("TRAINING_COMPLETED", handler1)
        coordinator.register_handler("TRAINING_COMPLETED", handler2)

        assert len(coordinator._event_handlers["TRAINING_COMPLETED"]) == 2

    @pytest.mark.asyncio
    async def test_dispatch_handlers(self, coordinator):
        """Test dispatching handlers."""
        called_with = []

        def handler(payload):
            called_with.append(payload)

        coordinator.register_handler("TEST_EVENT", handler)

        await coordinator._dispatch_handlers(
            event_type="TEST_EVENT",
            payload={"key": "value"},
            source="test",
            origin="unit_test",
        )

        assert len(called_with) == 1
        assert called_with[0]["key"] == "value"
        assert called_with[0]["event_type"] == "TEST_EVENT"

    @pytest.mark.asyncio
    async def test_dispatch_async_handlers(self, coordinator):
        """Test dispatching async handlers."""
        called = []

        async def async_handler(payload):
            called.append(payload)

        coordinator.register_handler("TEST_EVENT", async_handler)

        await coordinator._dispatch_handlers(
            event_type="TEST_EVENT",
            payload={"key": "value"},
            source="test",
            origin="unit_test",
        )

        assert len(called) == 1

    @pytest.mark.asyncio
    async def test_dispatch_handlers_error_handling(self, coordinator):
        """Test handler errors are caught."""
        def bad_handler(payload):
            raise RuntimeError("Handler error")

        coordinator.register_handler("TEST_EVENT", bad_handler)

        # Should not raise
        await coordinator._dispatch_handlers(
            event_type="TEST_EVENT",
            payload={},
            source="test",
            origin="unit_test",
        )

        assert len(coordinator._stats.errors) == 1
        assert "Handler error" in coordinator._stats.errors[0]

    @pytest.mark.asyncio
    async def test_emit_to_all_cross_process(self, coordinator, mock_cross_queue):
        """Test emit_to_all publishes to cross process."""
        coordinator._cross_queue = mock_cross_queue
        coordinator._running = True

        await coordinator.emit_to_all(
            event_type="TRAINING_COMPLETED",
            payload={"model_id": "test-model"},
            source="test",
        )

        mock_cross_queue.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_to_all_data_bus(self, coordinator, mock_data_bus):
        """Test emit_to_all publishes to data bus."""
        coordinator._data_bus = mock_data_bus
        coordinator._running = True

        mock_event_type = MagicMock()

        with patch("app.coordination.unified_event_coordinator.DataEventType", mock_event_type), \
             patch("app.coordination.unified_event_coordinator.DataEvent", MagicMock()):
            await coordinator.emit_to_all(
                event_type="TRAINING_COMPLETED",
                payload={"model_id": "test-model"},
                source="test",
            )

        mock_data_bus.publish.assert_called_once()


# ============================================
# Test Data Event Handling
# ============================================

class TestDataEventHandling:
    """Tests for data event bridging."""

    @pytest.mark.asyncio
    async def test_handle_data_event_not_running(self, coordinator):
        """Test data event ignored when not running."""
        coordinator._running = False

        mock_event = MagicMock()
        mock_event.event_type.value = "training_completed"
        mock_event.timestamp = 12345

        await coordinator._handle_data_event(mock_event)

        # Should not increment stats
        assert coordinator._stats.events_bridged_data_to_cross == 0

    @pytest.mark.asyncio
    async def test_handle_data_event_success(self, coordinator, mock_cross_queue):
        """Test successful data event bridging."""
        coordinator._running = True
        coordinator._cross_queue = mock_cross_queue

        mock_event = MagicMock()
        mock_event.event_type.value = "training_completed"
        mock_event.timestamp = 12345
        mock_event.payload = {"model_id": "test"}
        mock_event.source = "test"

        await coordinator._handle_data_event(mock_event)

        mock_cross_queue.publish.assert_called_once()
        assert coordinator._stats.events_bridged_data_to_cross == 1
        assert coordinator._stats.last_bridge_time is not None

    @pytest.mark.asyncio
    async def test_handle_data_event_dedup(self, coordinator, mock_cross_queue):
        """Test data events are deduplicated."""
        coordinator._running = True
        coordinator._cross_queue = mock_cross_queue

        mock_event = MagicMock()
        mock_event.event_type.value = "training_completed"
        mock_event.timestamp = 12345
        mock_event.payload = {"model_id": "test"}
        mock_event.source = "test"

        # First call
        await coordinator._handle_data_event(mock_event)
        # Second call (should be deduplicated)
        await coordinator._handle_data_event(mock_event)

        assert mock_cross_queue.publish.call_count == 1

    @pytest.mark.asyncio
    async def test_handle_data_event_unmapped(self, coordinator, mock_cross_queue):
        """Test unmapped data events are ignored."""
        coordinator._running = True
        coordinator._cross_queue = mock_cross_queue

        mock_event = MagicMock()
        mock_event.event_type.value = "unmapped_event"
        mock_event.timestamp = 12345

        await coordinator._handle_data_event(mock_event)

        mock_cross_queue.publish.assert_not_called()


# ============================================
# Test Stage Event Handling
# ============================================

class TestStageEventHandling:
    """Tests for stage event bridging."""

    @pytest.mark.asyncio
    async def test_handle_stage_event_not_running(self, coordinator):
        """Test stage event ignored when not running."""
        coordinator._running = False

        mock_result = MagicMock()
        mock_result.event.value = "training_complete"
        mock_result.timestamp = 12345

        await coordinator._handle_stage_event(mock_result)

        assert coordinator._stats.events_bridged_stage_to_cross == 0

    @pytest.mark.asyncio
    async def test_handle_stage_event_success(self, coordinator, mock_cross_queue):
        """Test successful stage event bridging."""
        coordinator._running = True
        coordinator._cross_queue = mock_cross_queue

        mock_result = MagicMock()
        mock_result.event.value = "training_complete"
        mock_result.timestamp = 12345
        mock_result.board_type = "square8"
        mock_result.num_players = 2
        mock_result.to_dict = MagicMock(return_value={"key": "value"})

        await coordinator._handle_stage_event(mock_result)

        mock_cross_queue.publish.assert_called_once()
        assert coordinator._stats.events_bridged_stage_to_cross == 1


# ============================================
# Test Module Functions
# ============================================

class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_event_coordinator_returns_singleton(self):
        """Test get_event_coordinator returns singleton."""
        coord1 = get_event_coordinator()
        coord2 = get_event_coordinator()
        assert coord1 is coord2

    @pytest.mark.asyncio
    async def test_start_coordinator(self):
        """Test start_coordinator function."""
        with patch("app.coordination.unified_event_coordinator.HAS_DATA_EVENTS", False), \
             patch("app.coordination.unified_event_coordinator.HAS_STAGE_EVENTS", False), \
             patch("app.coordination.unified_event_coordinator.HAS_CROSS_PROCESS", False):
            result = await start_coordinator()

        assert result is True

        await stop_coordinator()

    @pytest.mark.asyncio
    async def test_stop_coordinator(self):
        """Test stop_coordinator function."""
        coordinator = get_event_coordinator()
        coordinator._running = True

        await stop_coordinator()

        assert coordinator._running is False

    def test_get_coordinator_stats(self):
        """Test get_coordinator_stats function."""
        stats = get_coordinator_stats()
        assert isinstance(stats, CoordinatorStats)


# ============================================
# Test Event Emitters
# ============================================

class TestEventEmitters:
    """Tests for event emitter functions."""

    @pytest.mark.asyncio
    async def test_emit_training_started(self):
        """Test emit_training_started function."""
        coordinator = get_event_coordinator()
        coordinator.emit_to_all = AsyncMock()

        await emit_training_started(
            config_key="square8_2p",
            node_name="test-node",
        )

        coordinator.emit_to_all.assert_called_once()
        call_args = coordinator.emit_to_all.call_args
        assert call_args[1]["event_type"] == "TRAINING_STARTED"
        assert call_args[1]["payload"]["config_key"] == "square8_2p"
        assert call_args[1]["payload"]["node_name"] == "test-node"

    @pytest.mark.asyncio
    async def test_emit_training_completed(self):
        """Test emit_training_completed function."""
        coordinator = get_event_coordinator()
        coordinator.emit_to_all = AsyncMock()

        await emit_training_completed(
            config_key="square8_2p",
            model_id="model-v42",
            val_loss=0.123,
            epochs=50,
        )

        coordinator.emit_to_all.assert_called_once()
        call_args = coordinator.emit_to_all.call_args
        assert call_args[1]["event_type"] == "TRAINING_COMPLETED"
        assert call_args[1]["payload"]["model_id"] == "model-v42"
        assert call_args[1]["payload"]["val_loss"] == 0.123
        assert call_args[1]["payload"]["epochs"] == 50

    @pytest.mark.asyncio
    async def test_emit_training_failed(self):
        """Test emit_training_failed function."""
        coordinator = get_event_coordinator()
        coordinator.emit_to_all = AsyncMock()

        await emit_training_failed(
            config_key="square8_2p",
            error="OOM error",
        )

        coordinator.emit_to_all.assert_called_once()
        call_args = coordinator.emit_to_all.call_args
        assert call_args[1]["event_type"] == "TRAINING_FAILED"
        assert call_args[1]["payload"]["error"] == "OOM error"

    @pytest.mark.asyncio
    async def test_emit_evaluation_completed(self):
        """Test emit_evaluation_completed function."""
        coordinator = get_event_coordinator()
        coordinator.emit_to_all = AsyncMock()

        await emit_evaluation_completed(
            model_id="model-v42",
            elo=1650.0,
            win_rate=0.58,
            games_played=100,
        )

        coordinator.emit_to_all.assert_called_once()
        call_args = coordinator.emit_to_all.call_args
        assert call_args[1]["event_type"] == "EVALUATION_COMPLETED"
        assert call_args[1]["payload"]["elo"] == 1650.0
        assert call_args[1]["payload"]["win_rate"] == 0.58

    @pytest.mark.asyncio
    async def test_emit_sync_completed(self):
        """Test emit_sync_completed function."""
        coordinator = get_event_coordinator()
        coordinator.emit_to_all = AsyncMock()

        await emit_sync_completed(
            sync_type="games",
            files_synced=150,
            bytes_transferred=1000000,
        )

        coordinator.emit_to_all.assert_called_once()
        call_args = coordinator.emit_to_all.call_args
        assert call_args[1]["event_type"] == "DATA_SYNC_COMPLETED"
        assert call_args[1]["payload"]["files_synced"] == 150

    @pytest.mark.asyncio
    async def test_emit_model_promoted(self):
        """Test emit_model_promoted function."""
        coordinator = get_event_coordinator()
        coordinator.emit_to_all = AsyncMock()

        await emit_model_promoted(
            model_id="model-v42",
            tier="production",
            elo=1650.0,
        )

        coordinator.emit_to_all.assert_called_once()
        call_args = coordinator.emit_to_all.call_args
        assert call_args[1]["event_type"] == "MODEL_PROMOTED"
        assert call_args[1]["payload"]["tier"] == "production"

    @pytest.mark.asyncio
    async def test_emit_selfplay_batch_completed(self):
        """Test emit_selfplay_batch_completed function."""
        coordinator = get_event_coordinator()
        coordinator.emit_to_all = AsyncMock()

        await emit_selfplay_batch_completed(
            config_key="square8_2p",
            games_generated=500,
            duration_seconds=3600.0,
        )

        coordinator.emit_to_all.assert_called_once()
        call_args = coordinator.emit_to_all.call_args
        assert call_args[1]["event_type"] == "SELFPLAY_BATCH_COMPLETE"
        assert call_args[1]["payload"]["games_generated"] == 500

    @pytest.mark.asyncio
    async def test_emit_with_extra_payload(self):
        """Test emitters accept extra payload."""
        coordinator = get_event_coordinator()
        coordinator.emit_to_all = AsyncMock()

        await emit_training_started(
            config_key="square8_2p",
            custom_field="custom_value",
        )

        call_args = coordinator.emit_to_all.call_args
        assert call_args[1]["payload"]["custom_field"] == "custom_value"


# ============================================
# Integration Tests
# ============================================

class TestCoordinatorIntegration:
    """Integration tests for the event coordinator."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Test full coordinator lifecycle."""
        with patch("app.coordination.unified_event_coordinator.HAS_DATA_EVENTS", False), \
             patch("app.coordination.unified_event_coordinator.HAS_STAGE_EVENTS", False), \
             patch("app.coordination.unified_event_coordinator.HAS_CROSS_PROCESS", False):
            coordinator = get_event_coordinator()

            # Start
            await coordinator.start()
            assert coordinator._running is True

            # Get stats
            stats = coordinator.get_stats()
            assert stats.is_running is True

            # Stop
            await coordinator.stop()
            assert coordinator._running is False

    @pytest.mark.asyncio
    async def test_handler_registration_and_dispatch(self):
        """Test handler registration and dispatch in integration."""
        coordinator = get_event_coordinator()
        coordinator._running = True

        received_events = []

        def handler(payload):
            received_events.append(payload)

        coordinator.register_handler("TEST_EVENT", handler)

        await coordinator._dispatch_handlers(
            event_type="TEST_EVENT",
            payload={"test": True},
            source="integration_test",
            origin="test",
        )

        assert len(received_events) == 1
        assert received_events[0]["test"] is True

    @pytest.mark.asyncio
    async def test_event_bridging_with_handlers(self, mock_cross_queue):
        """Test event bridging triggers handlers."""
        coordinator = get_event_coordinator()
        coordinator._running = True
        coordinator._cross_queue = mock_cross_queue

        received = []

        def handler(payload):
            received.append(payload)

        coordinator.register_handler("TRAINING_COMPLETED", handler)

        # Create mock data event
        mock_event = MagicMock()
        mock_event.event_type.value = "training_completed"
        mock_event.timestamp = 12345
        mock_event.payload = {"model_id": "test-model"}
        mock_event.source = "test"

        await coordinator._handle_data_event(mock_event)

        # Handler should have been called
        assert len(received) == 1
        assert received[0]["model_id"] == "test-model"
