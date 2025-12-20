"""Tests for the data pipeline event bus.

Tests the event bus functionality including:
- Event subscription and publishing
- Async and sync callback handling
- Event type filtering
- Error handling in callbacks
- Health/recovery event integration (Phase 10)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.distributed.data_events import (
    DataEvent,
    DataEventType,
    EventBus,
    get_event_bus,
    reset_event_bus,
)


@pytest.fixture
def event_bus():
    """Create a fresh event bus for each test."""
    reset_event_bus()
    return EventBus()


@pytest.fixture
def global_event_bus():
    """Get the global event bus singleton."""
    reset_event_bus()
    return get_event_bus()


class TestDataEvent:
    """Tests for DataEvent dataclass."""

    def test_create_event(self):
        """Test basic event creation."""
        event = DataEvent(
            event_type=DataEventType.TRAINING_STARTED,
            payload={"config": "square8_2p"},
            source="test",
        )
        assert event.event_type == DataEventType.TRAINING_STARTED
        assert event.payload["config"] == "square8_2p"
        assert event.source == "test"
        assert event.timestamp > 0

    def test_event_to_dict(self):
        """Test event serialization."""
        event = DataEvent(
            event_type=DataEventType.ELO_UPDATED,
            payload={"model_id": "model_v1", "elo": 1550},
        )
        data = event.to_dict()
        assert data["event_type"] == "elo_updated"
        assert data["payload"]["elo"] == 1550

    def test_event_from_dict(self):
        """Test event deserialization."""
        data = {
            "event_type": "training_completed",
            "payload": {"val_loss": 0.05},
            "timestamp": 1234567890.0,
            "source": "trainer",
        }
        event = DataEvent.from_dict(data)
        assert event.event_type == DataEventType.TRAINING_COMPLETED
        assert event.payload["val_loss"] == 0.05
        assert event.source == "trainer"


class TestEventBus:
    """Tests for EventBus functionality."""

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, event_bus):
        """Test basic subscribe and publish flow."""
        received = []

        def handler(event):
            received.append(event)

        event_bus.subscribe(DataEventType.NEW_GAMES_AVAILABLE, handler)

        event = DataEvent(
            event_type=DataEventType.NEW_GAMES_AVAILABLE,
            payload={"new_games": 100},
        )
        await event_bus.publish(event, bridge_cross_process=False)

        assert len(received) == 1
        assert received[0].payload["new_games"] == 100

    @pytest.mark.asyncio
    async def test_async_callback(self, event_bus):
        """Test async callback handling."""
        received = []

        async def async_handler(event):
            await asyncio.sleep(0.01)
            received.append(event)

        event_bus.subscribe(DataEventType.TRAINING_STARTED, async_handler)

        event = DataEvent(event_type=DataEventType.TRAINING_STARTED)
        await event_bus.publish(event, bridge_cross_process=False)

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_event_filtering(self, event_bus):
        """Test that subscribers only receive matching events."""
        training_events = []
        eval_events = []

        event_bus.subscribe(
            DataEventType.TRAINING_STARTED,
            lambda e: training_events.append(e)
        )
        event_bus.subscribe(
            DataEventType.EVALUATION_STARTED,
            lambda e: eval_events.append(e)
        )

        await event_bus.publish(
            DataEvent(event_type=DataEventType.TRAINING_STARTED),
            bridge_cross_process=False
        )
        await event_bus.publish(
            DataEvent(event_type=DataEventType.EVALUATION_STARTED),
            bridge_cross_process=False
        )
        await event_bus.publish(
            DataEvent(event_type=DataEventType.TRAINING_STARTED),
            bridge_cross_process=False
        )

        assert len(training_events) == 2
        assert len(eval_events) == 1

    @pytest.mark.asyncio
    async def test_global_subscriber(self, event_bus):
        """Test global subscriber receives all events."""
        all_events = []

        event_bus.subscribe(None, lambda e: all_events.append(e))

        await event_bus.publish(
            DataEvent(event_type=DataEventType.TRAINING_STARTED),
            bridge_cross_process=False
        )
        await event_bus.publish(
            DataEvent(event_type=DataEventType.EVALUATION_COMPLETED),
            bridge_cross_process=False
        )

        assert len(all_events) == 2

    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus):
        """Test unsubscribe functionality."""
        received = []
        def handler(e):
            return received.append(e)

        event_bus.subscribe(DataEventType.MODEL_PROMOTED, handler)

        await event_bus.publish(
            DataEvent(event_type=DataEventType.MODEL_PROMOTED),
            bridge_cross_process=False
        )
        assert len(received) == 1

        result = event_bus.unsubscribe(DataEventType.MODEL_PROMOTED, handler)
        assert result is True

        await event_bus.publish(
            DataEvent(event_type=DataEventType.MODEL_PROMOTED),
            bridge_cross_process=False
        )
        assert len(received) == 1  # No new events

    @pytest.mark.asyncio
    async def test_callback_error_handling(self, event_bus):
        """Test that callback errors don't prevent other callbacks."""
        received = []

        def failing_handler(event):
            raise ValueError("Intentional test error")

        def working_handler(event):
            received.append(event)

        event_bus.subscribe(DataEventType.TRAINING_FAILED, failing_handler)
        event_bus.subscribe(DataEventType.TRAINING_FAILED, working_handler)

        # Should not raise, error should be logged
        await event_bus.publish(
            DataEvent(event_type=DataEventType.TRAINING_FAILED),
            bridge_cross_process=False
        )

        assert len(received) == 1


class TestHealthRecoveryEvents:
    """Tests for health and recovery event types (Phase 10)."""

    def test_health_event_types_exist(self):
        """Verify new health/recovery event types are defined."""
        assert hasattr(DataEventType, 'HEALTH_CHECK_PASSED')
        assert hasattr(DataEventType, 'HEALTH_CHECK_FAILED')
        assert hasattr(DataEventType, 'HEALTH_ALERT')
        assert hasattr(DataEventType, 'RESOURCE_CONSTRAINT')
        assert hasattr(DataEventType, 'RECOVERY_INITIATED')
        assert hasattr(DataEventType, 'RECOVERY_COMPLETED')
        assert hasattr(DataEventType, 'RECOVERY_FAILED')

    def test_cluster_event_types_exist(self):
        """Verify cluster status event types are defined."""
        assert hasattr(DataEventType, 'CLUSTER_STATUS_CHANGED')
        assert hasattr(DataEventType, 'NODE_UNHEALTHY')
        assert hasattr(DataEventType, 'NODE_RECOVERED')

    @pytest.mark.asyncio
    async def test_recovery_event_flow(self, event_bus):
        """Test the recovery event flow."""
        events = []

        for event_type in [
            DataEventType.HEALTH_CHECK_FAILED,
            DataEventType.RECOVERY_INITIATED,
            DataEventType.RECOVERY_COMPLETED,
        ]:
            event_bus.subscribe(event_type, lambda e: events.append(e))

        # Simulate recovery flow
        await event_bus.publish(
            DataEvent(
                event_type=DataEventType.HEALTH_CHECK_FAILED,
                payload={"component": "data_sync", "failure_count": 3},
                source="health_checker",
            ),
            bridge_cross_process=False
        )

        await event_bus.publish(
            DataEvent(
                event_type=DataEventType.RECOVERY_INITIATED,
                payload={"component": "data_sync", "action": "restart"},
                source="recovery_manager",
            ),
            bridge_cross_process=False
        )

        await event_bus.publish(
            DataEvent(
                event_type=DataEventType.RECOVERY_COMPLETED,
                payload={"component": "data_sync", "success": True},
                source="recovery_manager",
            ),
            bridge_cross_process=False
        )

        assert len(events) == 3
        assert events[0].event_type == DataEventType.HEALTH_CHECK_FAILED
        assert events[1].event_type == DataEventType.RECOVERY_INITIATED
        assert events[2].event_type == DataEventType.RECOVERY_COMPLETED


class TestEventBusSingleton:
    """Tests for global event bus singleton."""

    def test_singleton_returns_same_instance(self):
        """Test that get_event_bus returns the same instance."""
        reset_event_bus()
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2

    def test_reset_creates_new_instance(self):
        """Test that reset creates a new event bus."""
        reset_event_bus()
        bus1 = get_event_bus()
        reset_event_bus()
        bus2 = get_event_bus()
        assert bus1 is not bus2
