"""Unit tests for BaseEventHandler.

Tests the base class for event-driven coordinators.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Any, Callable, Dict

from app.coordination.base_event_handler import (
    BaseEventHandler,
    EventHandlerConfig,
)
from app.coordination.contracts import CoordinatorStatus, HealthCheckResult


class ConcreteEventHandler(BaseEventHandler):
    """Concrete implementation for testing."""

    def __init__(self, name: str = "TestHandler"):
        super().__init__(name)
        self.handled_events: list = []

    def _get_subscriptions(self) -> Dict[Any, Callable]:
        return {
            "TEST_EVENT_1": self._on_test_event_1,
            "TEST_EVENT_2": self._on_test_event_2,
        }

    async def _on_test_event_1(self, event: Any) -> None:
        self.handled_events.append(("TEST_EVENT_1", event))

    async def _on_test_event_2(self, event: Any) -> None:
        self.handled_events.append(("TEST_EVENT_2", event))


class ErrorHandler(BaseEventHandler):
    """Handler that raises errors for testing."""

    def __init__(self, name: str = "ErrorHandler"):
        super().__init__(name)

    def _get_subscriptions(self) -> Dict[Any, Callable]:
        return {
            "ERROR_EVENT": self._on_error_event,
        }

    async def _on_error_event(self, event: Any) -> None:
        raise ValueError("Test error")


class EmptyHandler(BaseEventHandler):
    """Handler with no subscriptions."""

    def __init__(self, name: str = "EmptyHandler"):
        super().__init__(name)

    def _get_subscriptions(self) -> Dict[Any, Callable]:
        return {}


class TestBaseEventHandlerInit:
    """Test BaseEventHandler initialization."""

    def test_initialization(self):
        """Test default initialization."""
        handler = ConcreteEventHandler("MyHandler")
        assert handler.name == "MyHandler"
        assert handler.status == CoordinatorStatus.INITIALIZING
        assert not handler.is_running
        assert handler._events_processed == 0
        assert handler._errors_count == 0

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = EventHandlerConfig()
        config.register_with_registry = False
        handler = ConcreteEventHandler("MyHandler")
        handler._config = config
        assert handler._config.register_with_registry is False

    def test_name_property(self):
        """Test name property."""
        handler = ConcreteEventHandler("CustomName")
        assert handler.name == "CustomName"

    def test_status_property(self):
        """Test status property."""
        handler = ConcreteEventHandler()
        assert handler.status == CoordinatorStatus.INITIALIZING


class TestBaseEventHandlerLifecycle:
    """Test BaseEventHandler lifecycle methods."""

    @pytest.mark.asyncio
    async def test_start_success(self):
        """Test successful start."""
        handler = ConcreteEventHandler()

        mock_bus = Mock()
        mock_bus.subscribe = Mock()

        with patch("app.coordination.event_router.get_event_bus", return_value=mock_bus):
            with patch.object(handler, "_config") as mock_config:
                mock_config.register_with_registry = False
                result = await handler.start()

        assert result is True
        assert handler._running is True
        assert handler._subscribed is True
        assert handler.status == CoordinatorStatus.RUNNING
        assert handler._start_time > 0

    @pytest.mark.asyncio
    async def test_start_no_event_bus(self):
        """Test start when event bus not available."""
        handler = ConcreteEventHandler()

        with patch("app.coordination.event_router.get_event_bus", return_value=None):
            result = await handler.start()

        assert result is False
        assert handler._running is False
        assert handler.status == CoordinatorStatus.ERROR

    @pytest.mark.asyncio
    async def test_start_already_running(self):
        """Test start when already running."""
        handler = ConcreteEventHandler()
        handler._running = True

        result = await handler.start()

        assert result is True  # Returns True without doing anything

    @pytest.mark.asyncio
    async def test_stop(self):
        """Test stop lifecycle."""
        handler = ConcreteEventHandler()
        handler._config.register_with_registry = False

        mock_bus = Mock()
        mock_bus.subscribe = Mock()
        mock_bus.unsubscribe = Mock()

        with patch("app.coordination.event_router.get_event_bus", return_value=mock_bus):
            await handler.start()
            await handler.stop()

        assert handler._running is False
        assert handler._subscribed is False
        assert handler.status == CoordinatorStatus.STOPPED

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self):
        """Test stop when not running."""
        handler = ConcreteEventHandler()

        await handler.stop()  # Should not raise

        assert handler._running is False

    @pytest.mark.asyncio
    async def test_uptime_seconds(self):
        """Test uptime calculation."""
        handler = ConcreteEventHandler()
        assert handler.uptime_seconds == 0.0

        handler._start_time = 100.0
        with patch("time.time", return_value=150.0):
            assert handler.uptime_seconds == 50.0


class TestBaseEventHandlerSubscriptions:
    """Test subscription management."""

    @pytest.mark.asyncio
    async def test_subscribe_to_events(self):
        """Test event subscription."""
        handler = ConcreteEventHandler()

        mock_bus = Mock()
        mock_bus.subscribe = Mock()

        with patch("app.coordination.event_router.get_event_bus", return_value=mock_bus):
            result = await handler._subscribe_to_events()

        assert result is True
        assert handler._subscribed is True
        assert mock_bus.subscribe.call_count == 2  # Two events

    @pytest.mark.asyncio
    async def test_subscribe_empty_subscriptions(self):
        """Test subscription with no events."""
        handler = EmptyHandler()

        mock_bus = Mock()

        with patch("app.coordination.event_router.get_event_bus", return_value=mock_bus):
            result = await handler._subscribe_to_events()

        assert result is True  # Not an error
        assert mock_bus.subscribe.call_count == 0

    @pytest.mark.asyncio
    async def test_unsubscribe_from_events(self):
        """Test event unsubscription."""
        handler = ConcreteEventHandler()
        handler._subscribed = True
        handler._wrapped_handlers = {"TEST_EVENT": Mock()}

        mock_bus = Mock()
        mock_bus.unsubscribe = Mock()

        with patch("app.coordination.event_router.get_event_bus", return_value=mock_bus):
            await handler._unsubscribe_from_events()

        assert handler._subscribed is False
        assert len(handler._wrapped_handlers) == 0


class TestBaseEventHandlerEventHandling:
    """Test event handling."""

    @pytest.mark.asyncio
    async def test_handle_event_async(self):
        """Test async event handling."""
        handler = ConcreteEventHandler()
        event = {"key": "value"}

        await handler._handle_event_async(
            handler._on_test_event_1,
            event,
            "TEST_EVENT_1",
        )

        assert handler._events_processed == 1
        assert handler._errors_count == 0
        assert len(handler.handled_events) == 1
        assert handler.handled_events[0] == ("TEST_EVENT_1", event)

    @pytest.mark.asyncio
    async def test_handle_event_error(self):
        """Test error handling in event processing."""
        handler = ErrorHandler()
        event = {"key": "value"}

        await handler._handle_event_async(
            handler._on_error_event,
            event,
            "ERROR_EVENT",
        )

        assert handler._events_processed == 1
        assert handler._errors_count == 1
        assert "Test error" in handler._last_error

    def test_handle_event_sync(self):
        """Test sync event handling."""
        handler = ConcreteEventHandler()
        handler._config.async_handlers = False

        # Test sync handler
        def sync_handler(event):
            handler.handled_events.append(("SYNC", event))

        event = {"sync": True}
        handler._handle_event_sync(sync_handler, event, "SYNC_EVENT")

        assert handler._events_processed == 1
        assert ("SYNC", event) in handler.handled_events


class TestBaseEventHandlerHealthCheck:
    """Test health check functionality."""

    def test_health_check_running(self):
        """Test health check when running."""
        handler = ConcreteEventHandler()
        handler._running = True
        handler._subscribed = True
        handler._status = CoordinatorStatus.RUNNING
        handler._events_processed = 10
        handler._start_time = 100.0

        result = handler.health_check()

        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING
        assert "10 events" in result.message

    def test_health_check_stopped(self):
        """Test health check when stopped."""
        handler = ConcreteEventHandler()
        handler._status = CoordinatorStatus.STOPPED

        result = handler.health_check()

        assert result.healthy is True
        assert result.status == CoordinatorStatus.STOPPED

    def test_health_check_error(self):
        """Test health check in error state."""
        handler = ConcreteEventHandler()
        handler._status = CoordinatorStatus.ERROR
        handler._last_error = "Something went wrong"

        result = handler.health_check()

        assert result.healthy is False
        assert result.status == CoordinatorStatus.ERROR
        assert "Something went wrong" in result.message

    def test_health_check_not_subscribed(self):
        """Test health check when not subscribed."""
        handler = ConcreteEventHandler()
        handler._running = True
        handler._subscribed = False
        handler._status = CoordinatorStatus.RUNNING

        result = handler.health_check()

        assert result.healthy is False
        assert result.status == CoordinatorStatus.DEGRADED


class TestBaseEventHandlerMetrics:
    """Test metrics functionality."""

    def test_get_metrics(self):
        """Test get_metrics returns correct data."""
        handler = ConcreteEventHandler("MetricsHandler")
        handler._status = CoordinatorStatus.RUNNING
        handler._start_time = 100.0
        handler._events_processed = 50
        handler._errors_count = 2
        handler._subscribed = True
        handler._subscriptions = {"A": None, "B": None}

        metrics = handler.get_metrics()

        assert metrics["name"] == "MetricsHandler"
        assert metrics["status"] == "running"
        assert metrics["events_processed"] == 50
        assert metrics["errors_count"] == 2
        assert metrics["subscribed"] is True
        assert metrics["subscription_count"] == 2

    def test_get_status(self):
        """Test get_status for DaemonManager."""
        handler = ConcreteEventHandler("StatusHandler")
        handler._running = True
        handler._subscribed = True
        handler._status = CoordinatorStatus.RUNNING
        handler._events_processed = 25

        status = handler.get_status()

        assert status["daemon"] == "StatusHandler"
        assert status["running"] is True
        assert status["subscribed"] is True
        assert status["events_processed"] == 25


class TestEventHandlerConfig:
    """Test EventHandlerConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EventHandlerConfig()

        assert config.register_with_registry is True
        assert config.async_handlers is True
        assert config.use_fire_and_forget is True
        assert config.handler_timeout_seconds == 0.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = EventHandlerConfig()
        config.register_with_registry = False
        config.async_handlers = False

        assert config.register_with_registry is False
        assert config.async_handlers is False
