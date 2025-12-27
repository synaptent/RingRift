"""Tests for app.coordination.handler_base module.

Tests the unified HandlerBase class that provides common patterns:
- Singleton management
- Event subscription
- Event deduplication
- Health check
- Error tracking
- Lifecycle management

December 2025 - Phase 2 handler consolidation tests.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.handler_base import (
    CoordinatorStatus,
    HandlerBase,
    HandlerStats,
    HealthCheckResult,
)


class ConcreteHandler(HandlerBase):
    """Concrete implementation for testing."""

    def __init__(self, config=None, **kwargs):
        super().__init__(name="test_handler", config=config, **kwargs)
        self.cycle_count = 0
        self.on_start_called = False
        self.on_stop_called = False

    async def _run_cycle(self) -> None:
        self.cycle_count += 1

    async def _on_start(self) -> None:
        self.on_start_called = True

    async def _on_stop(self) -> None:
        self.on_stop_called = True


class EventSubscribingHandler(HandlerBase):
    """Handler with event subscriptions for testing."""

    def __init__(self):
        super().__init__(name="event_handler")
        self.events_received = []

    async def _run_cycle(self) -> None:
        pass

    def _get_event_subscriptions(self) -> dict:
        return {
            "test_event": self._on_test_event,
        }

    async def _on_test_event(self, event: dict) -> None:
        self.events_received.append(event)


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        result = HealthCheckResult(healthy=True)
        assert result.healthy is True
        assert result.status == CoordinatorStatus.STOPPED
        assert result.message == ""
        assert result.details == {}
        assert result.timestamp > 0

    def test_custom_values(self):
        """Should accept custom values."""
        result = HealthCheckResult(
            healthy=False,
            status=CoordinatorStatus.ERROR,
            message="Something went wrong",
            details={"error_rate": 0.5},
        )
        assert result.healthy is False
        assert result.status == CoordinatorStatus.ERROR
        assert result.message == "Something went wrong"
        assert result.details["error_rate"] == 0.5

    def test_to_dict(self):
        """Should convert to dictionary."""
        result = HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="OK",
        )
        d = result.to_dict()
        assert d["healthy"] is True
        assert d["status"] == "running"
        assert d["message"] == "OK"
        assert "timestamp" in d


class TestHandlerStats:
    """Tests for HandlerStats dataclass."""

    def test_default_values(self):
        """Should have zero/empty defaults."""
        stats = HandlerStats()
        assert stats.events_processed == 0
        assert stats.events_deduplicated == 0
        assert stats.cycles_completed == 0
        assert stats.errors_count == 0
        assert stats.last_error == ""

    def test_mutable_fields(self):
        """Should allow field updates."""
        stats = HandlerStats()
        stats.events_processed = 10
        stats.errors_count = 2
        assert stats.events_processed == 10
        assert stats.errors_count == 2


class TestHandlerBaseInit:
    """Tests for HandlerBase initialization."""

    def test_init_with_name(self):
        """Should initialize with name."""
        handler = ConcreteHandler()
        assert handler.name == "test_handler"
        assert handler.is_running is False
        assert handler._status == CoordinatorStatus.STOPPED

    def test_init_with_config(self):
        """Should store config."""
        config = {"key": "value"}
        handler = ConcreteHandler(config=config)
        assert handler._config == config

    def test_init_with_custom_interval(self):
        """Should accept custom cycle interval."""
        handler = ConcreteHandler(cycle_interval=30.0)
        assert handler._cycle_interval == 30.0

    def test_init_dedup_enabled_by_default(self):
        """Should enable deduplication by default."""
        handler = ConcreteHandler()
        assert handler._dedup_enabled is True

    def test_init_dedup_can_be_disabled(self):
        """Should allow disabling deduplication."""
        handler = ConcreteHandler(dedup_enabled=False)
        assert handler._dedup_enabled is False


class TestHandlerBaseSingleton:
    """Tests for singleton pattern."""

    def teardown_method(self):
        """Clean up singleton after each test."""
        ConcreteHandler.reset_instance()

    def test_get_instance_creates_singleton(self):
        """Should create singleton on first call."""
        instance1 = ConcreteHandler.get_instance()
        instance2 = ConcreteHandler.get_instance()
        assert instance1 is instance2

    def test_reset_instance(self):
        """Should reset singleton."""
        instance1 = ConcreteHandler.get_instance()
        ConcreteHandler.reset_instance()
        assert not ConcreteHandler.has_instance()
        instance2 = ConcreteHandler.get_instance()
        assert instance1 is not instance2

    def test_has_instance(self):
        """Should report instance existence."""
        assert not ConcreteHandler.has_instance()
        ConcreteHandler.get_instance()
        assert ConcreteHandler.has_instance()


class TestHandlerBaseLifecycle:
    """Tests for lifecycle management."""

    def teardown_method(self):
        """Clean up singleton after each test."""
        ConcreteHandler.reset_instance()

    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        """Should set running state on start."""
        handler = ConcreteHandler(cycle_interval=0.1)
        await handler.start()
        try:
            assert handler.is_running is True
            assert handler._status == CoordinatorStatus.RUNNING
        finally:
            await handler.stop()

    @pytest.mark.asyncio
    async def test_start_calls_on_start(self):
        """Should call _on_start hook."""
        handler = ConcreteHandler(cycle_interval=0.1)
        await handler.start()
        try:
            assert handler.on_start_called is True
        finally:
            await handler.stop()

    @pytest.mark.asyncio
    async def test_start_when_already_running(self):
        """Should warn when already running."""
        handler = ConcreteHandler(cycle_interval=0.1)
        await handler.start()
        try:
            # Second start should just return
            await handler.start()
            assert handler.is_running is True
        finally:
            await handler.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_running(self):
        """Should clear running state on stop."""
        handler = ConcreteHandler(cycle_interval=0.1)
        await handler.start()
        await handler.stop()
        assert handler.is_running is False
        assert handler._status == CoordinatorStatus.STOPPED

    @pytest.mark.asyncio
    async def test_stop_calls_on_stop(self):
        """Should call _on_stop hook."""
        handler = ConcreteHandler(cycle_interval=0.1)
        await handler.start()
        await handler.stop()
        assert handler.on_stop_called is True

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self):
        """Should handle stop when not running."""
        handler = ConcreteHandler()
        await handler.stop()  # Should not raise
        assert handler.is_running is False

    @pytest.mark.asyncio
    async def test_shutdown_calls_stop(self):
        """Shutdown should be alias for stop."""
        handler = ConcreteHandler(cycle_interval=0.1)
        await handler.start()
        await handler.shutdown()
        assert handler.is_running is False

    @pytest.mark.asyncio
    async def test_main_loop_runs_cycle(self):
        """Should run _run_cycle periodically."""
        handler = ConcreteHandler(cycle_interval=0.01)
        await handler.start()
        await asyncio.sleep(0.05)  # Wait for a few cycles
        await handler.stop()
        assert handler.cycle_count > 0
        assert handler.stats.cycles_completed > 0


class TestHandlerBaseHealthCheck:
    """Tests for health check functionality."""

    def test_health_check_when_stopped(self):
        """Should report unhealthy when stopped."""
        handler = ConcreteHandler()
        result = handler.health_check()
        assert result.healthy is False
        assert result.status == CoordinatorStatus.STOPPED

    @pytest.mark.asyncio
    async def test_health_check_when_running(self):
        """Should report healthy when running normally."""
        handler = ConcreteHandler(cycle_interval=0.1)
        await handler.start()
        try:
            result = handler.health_check()
            assert result.healthy is True
            assert result.status == CoordinatorStatus.RUNNING
        finally:
            await handler.stop()

    @pytest.mark.asyncio
    async def test_health_check_degraded_on_errors(self):
        """Should report degraded on elevated error rate."""
        handler = ConcreteHandler(cycle_interval=0.1)
        await handler.start()
        try:
            # Simulate some operations and errors
            handler._stats.cycles_completed = 10
            handler._stats.errors_count = 3  # 30% error rate
            result = handler.health_check()
            assert result.status == CoordinatorStatus.DEGRADED
        finally:
            await handler.stop()

    @pytest.mark.asyncio
    async def test_health_check_error_on_high_errors(self):
        """Should report error on high error rate."""
        handler = ConcreteHandler(cycle_interval=0.1)
        await handler.start()
        try:
            handler._stats.cycles_completed = 10
            handler._stats.errors_count = 6  # 60% error rate
            result = handler.health_check()
            assert result.healthy is False
            assert result.status == CoordinatorStatus.ERROR
        finally:
            await handler.stop()

    def test_get_health_details(self):
        """Should return detailed health info."""
        handler = ConcreteHandler()
        handler._stats.cycles_completed = 5
        handler._stats.events_processed = 10
        details = handler._get_health_details()
        assert details["name"] == "test_handler"
        assert details["cycles_completed"] == 5
        assert details["events_processed"] == 10


class TestHandlerBaseGetStatus:
    """Tests for get_status functionality."""

    def test_get_status_returns_dict(self):
        """Should return status dictionary."""
        handler = ConcreteHandler()
        status = handler.get_status()
        assert isinstance(status, dict)
        assert status["name"] == "test_handler"
        assert status["running"] is False
        assert "health" in status
        assert "stats" in status


class TestHandlerBaseEventDeduplication:
    """Tests for event deduplication."""

    def test_is_duplicate_event_first_occurrence(self):
        """Should return False for first occurrence."""
        handler = ConcreteHandler()
        event = {"id": "123", "data": "test"}
        assert handler._is_duplicate_event(event) is False

    def test_is_duplicate_event_second_occurrence(self):
        """Should return True for duplicate."""
        handler = ConcreteHandler()
        event = {"id": "123", "data": "test"}
        handler._is_duplicate_event(event)  # First
        assert handler._is_duplicate_event(event) is True  # Duplicate

    def test_is_duplicate_event_different_events(self):
        """Should return False for different events."""
        handler = ConcreteHandler()
        event1 = {"id": "123"}
        event2 = {"id": "456"}
        handler._is_duplicate_event(event1)
        assert handler._is_duplicate_event(event2) is False

    def test_is_duplicate_event_disabled(self):
        """Should always return False when disabled."""
        handler = ConcreteHandler(dedup_enabled=False)
        event = {"id": "123"}
        handler._is_duplicate_event(event)
        assert handler._is_duplicate_event(event) is False

    def test_is_duplicate_event_with_key_fields(self):
        """Should only check specified fields."""
        handler = ConcreteHandler()
        event1 = {"id": "123", "timestamp": 1}
        event2 = {"id": "123", "timestamp": 2}
        handler._is_duplicate_event(event1, key_fields=["id"])
        # Same id, different timestamp - still duplicate by key_fields
        assert handler._is_duplicate_event(event2, key_fields=["id"]) is True

    def test_is_duplicate_event_increments_stat(self):
        """Should increment deduplicated counter."""
        handler = ConcreteHandler()
        event = {"id": "123"}
        handler._is_duplicate_event(event)
        assert handler._stats.events_deduplicated == 0
        handler._is_duplicate_event(event)
        assert handler._stats.events_deduplicated == 1

    def test_mark_event_processed(self):
        """Should increment processed counter."""
        handler = ConcreteHandler()
        handler._mark_event_processed({"id": "123"})
        assert handler._stats.events_processed == 1


class TestHandlerBaseErrorTracking:
    """Tests for error tracking."""

    def test_record_error(self):
        """Should record error in stats."""
        handler = ConcreteHandler()
        handler._record_error("Test error")
        assert handler._stats.errors_count == 1
        assert handler._stats.last_error == "Test error"
        assert handler._stats.last_error_time > 0

    def test_record_error_with_exception(self):
        """Should record exception details."""
        handler = ConcreteHandler()
        exc = ValueError("test")
        handler._record_error("Error occurred", exc=exc)
        errors = handler.get_recent_errors()
        assert len(errors) == 1
        assert errors[0]["exception"] == "test"

    def test_error_log_bounded(self):
        """Should keep error log bounded."""
        handler = ConcreteHandler()
        handler._max_error_log = 5
        for i in range(10):
            handler._record_error(f"Error {i}")
        errors = handler.get_recent_errors(limit=10)
        assert len(errors) == 5

    def test_get_recent_errors_with_limit(self):
        """Should return limited errors."""
        handler = ConcreteHandler()
        for i in range(5):
            handler._record_error(f"Error {i}")
        errors = handler.get_recent_errors(limit=3)
        assert len(errors) == 3


class TestHandlerBaseEventSubscription:
    """Tests for event subscription."""

    def test_get_event_subscriptions_default_empty(self):
        """Should return empty dict by default."""
        handler = ConcreteHandler()
        subs = handler._get_event_subscriptions()
        assert subs == {}

    def test_get_event_subscriptions_override(self):
        """Should allow override in subclass."""
        handler = EventSubscribingHandler()
        subs = handler._get_event_subscriptions()
        assert "test_event" in subs

    @patch("app.coordination.event_router.get_router")
    def test_subscribe_all_events_success(self, mock_get_router):
        """Should subscribe to all events."""
        mock_router = MagicMock()
        mock_router.subscribe.return_value = MagicMock()  # unsub callable
        mock_get_router.return_value = mock_router

        handler = EventSubscribingHandler()
        result = handler._subscribe_all_events()

        assert result is True
        assert handler._event_subscribed is True
        mock_router.subscribe.assert_called_once()

    def test_subscribe_all_events_already_subscribed(self):
        """Should return True if already subscribed."""
        handler = EventSubscribingHandler()
        handler._event_subscribed = True

        result = handler._subscribe_all_events()
        assert result is True

    def test_unsubscribe_all_events(self):
        """Should clear subscriptions."""
        handler = EventSubscribingHandler()
        mock_unsub = MagicMock()
        handler._event_subscriptions = {"test": mock_unsub}
        handler._event_subscribed = True

        handler._unsubscribe_all_events()

        mock_unsub.assert_called_once()
        assert handler._event_subscriptions == {}
        assert handler._event_subscribed is False


class TestCoordinatorStatus:
    """Tests for CoordinatorStatus enum."""

    def test_status_values(self):
        """Should have expected status values."""
        assert CoordinatorStatus.STOPPED.value == "stopped"
        assert CoordinatorStatus.STARTING.value == "starting"
        assert CoordinatorStatus.RUNNING.value == "running"
        assert CoordinatorStatus.STOPPING.value == "stopping"
        assert CoordinatorStatus.DEGRADED.value == "degraded"
        assert CoordinatorStatus.ERROR.value == "error"


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Should export expected classes."""
        from app.coordination import handler_base

        assert hasattr(handler_base, "HandlerBase")
        assert hasattr(handler_base, "HandlerStats")
        assert hasattr(handler_base, "HealthCheckResult")
        assert hasattr(handler_base, "CoordinatorStatus")
        assert len(handler_base.__all__) == 4
