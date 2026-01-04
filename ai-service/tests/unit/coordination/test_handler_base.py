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
        """Should have sensible defaults.

        Note: HealthCheckResult now imports from contracts.py which defaults
        status to RUNNING (not STOPPED) - this is the canonical behavior.
        """
        result = HealthCheckResult(healthy=True)
        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING  # contracts.py default
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


class TestHandlerStatsAdvanced:
    """Additional tests for HandlerStats dataclass."""

    def test_error_count_alias(self):
        """Should have backward-compat error_count alias."""
        stats = HandlerStats(errors_count=5)
        assert stats.error_count == 5

    def test_success_rate_zero_events(self):
        """Should return 1.0 when no events processed."""
        stats = HandlerStats()
        assert stats.success_rate == 1.0

    def test_success_rate_all_success(self):
        """Should return 1.0 when all successful."""
        stats = HandlerStats(events_processed=10, success_count=10)
        assert stats.success_rate == 1.0

    def test_success_rate_partial(self):
        """Should calculate correct rate."""
        stats = HandlerStats(events_processed=10, success_count=7)
        assert stats.success_rate == 0.7

    def test_to_dict_complete(self):
        """Should include all fields in to_dict."""
        stats = HandlerStats(
            subscribed=True,
            events_processed=100,
            events_deduplicated=5,
            success_count=90,
            errors_count=10,
            cycles_completed=50,
            last_error="Test error",
            started_at=1000.0,
            last_activity=2000.0,
        )
        stats.custom_stats = {"queue_depth": 42}

        d = stats.to_dict()

        assert d["subscribed"] is True
        assert d["events_processed"] == 100
        assert d["events_deduplicated"] == 5
        assert d["success_count"] == 90
        assert d["errors_count"] == 10
        assert d["cycles_completed"] == 50
        assert d["last_error"] == "Test error"
        assert d["success_rate"] == 0.9
        # Custom stats included
        assert d["queue_depth"] == 42


class TestHandlerBaseBackwardCompat:
    """Tests for backward-compatible methods from base_handler.py."""

    def test_handler_name_alias(self):
        """Should have handler_name alias for name."""
        handler = ConcreteHandler()
        assert handler.handler_name == handler.name

    def test_is_subscribed_alias(self):
        """Should have is_subscribed alias."""
        handler = ConcreteHandler()
        handler._event_subscribed = True
        assert handler.is_subscribed is True

    def test_emit_metrics_property(self):
        """Should have emit_metrics property (always True)."""
        handler = ConcreteHandler()
        assert handler.emit_metrics is True

    def test_record_success(self):
        """Should increment success counters."""
        handler = ConcreteHandler()
        handler._record_success()
        assert handler._stats.events_processed == 1
        assert handler._stats.success_count == 1
        assert handler._stats.last_activity > 0

    def test_get_stats_returns_dict(self):
        """Should return stats as dictionary."""
        handler = ConcreteHandler()
        handler._event_subscribed = True
        handler._stats.events_processed = 10
        handler._stats.success_count = 8
        handler._stats.errors_count = 2

        stats = handler.get_stats()

        assert stats["subscribed"] is True
        assert stats["events_processed"] == 10
        assert stats["success_count"] == 8
        assert stats["errors_count"] == 2
        assert stats["success_rate"] == 0.8

    def test_subscribe_marks_subscribed(self):
        """Should mark as subscribed."""
        handler = ConcreteHandler()
        result = handler.subscribe()

        assert result is True
        assert handler._event_subscribed is True
        assert handler._stats.subscribed is True

    def test_subscribe_already_subscribed(self):
        """Should return True if already subscribed."""
        handler = ConcreteHandler()
        handler._event_subscribed = True

        result = handler.subscribe()
        assert result is True

    def test_unsubscribe_clears_state(self):
        """Should clear subscription state."""
        handler = ConcreteHandler()
        handler._event_subscribed = True
        handler._stats.subscribed = True

        handler.unsubscribe()

        assert handler._event_subscribed is False
        assert handler._stats.subscribed is False

    def test_get_payload_from_event_object(self):
        """Should extract payload from event object."""
        handler = ConcreteHandler()

        class MockEvent:
            payload = {"key": "value"}

        payload = handler._get_payload(MockEvent())
        assert payload == {"key": "value"}

    def test_get_payload_from_dict(self):
        """Should return dict as-is."""
        handler = ConcreteHandler()
        payload = handler._get_payload({"key": "value"})
        assert payload == {"key": "value"}

    def test_get_payload_from_other(self):
        """Should return empty dict for unknown types."""
        handler = ConcreteHandler()
        payload = handler._get_payload("string")
        assert payload == {}

    def test_add_custom_stat(self):
        """Should add custom stats."""
        handler = ConcreteHandler()
        handler.add_custom_stat("queue_depth", 100)
        handler.add_custom_stat("active_jobs", 5)

        assert handler._stats.custom_stats["queue_depth"] == 100
        assert handler._stats.custom_stats["active_jobs"] == 5

    def test_reset_clears_counters(self):
        """Should reset counters but preserve subscription."""
        handler = ConcreteHandler()
        handler._event_subscribed = True
        handler._stats.events_processed = 100
        handler._stats.success_count = 90
        handler._stats.errors_count = 10
        handler._stats.last_error = "Error"
        handler._stats.custom_stats = {"x": 1}

        handler.reset()

        assert handler._stats.events_processed == 0
        assert handler._stats.success_count == 0
        assert handler._stats.errors_count == 0
        assert handler._stats.last_error == ""
        assert handler._stats.custom_stats == {}
        # Subscription preserved
        assert handler._event_subscribed is True

    @pytest.mark.asyncio
    async def test_handle_event_routes_async(self):
        """Should route to async handler."""
        handler = ConcreteHandler()
        handled = []

        async def mock_handler(event):
            handled.append(event)

        handler._event_handlers["test"] = mock_handler

        await handler._handle_event({"type": "test", "data": "value"})

        assert len(handled) == 1
        assert handled[0]["data"] == "value"

    @pytest.mark.asyncio
    async def test_handle_event_routes_sync(self):
        """Should route to sync handler."""
        handler = ConcreteHandler()
        handled = []

        def sync_handler(event):
            handled.append(event)

        handler._event_handlers["test"] = sync_handler

        await handler._handle_event({"type": "test"})

        assert len(handled) == 1

    def test_uptime_seconds_zero_when_not_started(self):
        """Should return 0 when not started."""
        handler = ConcreteHandler()
        assert handler.uptime_seconds == 0.0

    @pytest.mark.asyncio
    async def test_uptime_seconds_when_started(self):
        """Should return uptime when started."""
        import time

        handler = ConcreteHandler(cycle_interval=0.1)
        await handler.start()
        await asyncio.sleep(0.1)
        try:
            uptime = handler.uptime_seconds
            assert uptime >= 0.1
        finally:
            await handler.stop()


class TestEventHandlerConfig:
    """Tests for EventHandlerConfig class."""

    def test_default_values(self):
        """Should have expected defaults."""
        from app.coordination.handler_base import EventHandlerConfig

        config = EventHandlerConfig()
        assert config.register_with_registry is True
        assert config.async_handlers is True
        assert config.use_fire_and_forget is True
        assert config.handler_timeout_seconds == 0.0


class TestHelperFunctions:
    """Tests for module helper functions."""

    def test_create_handler_stats_basic(self):
        """Should create empty stats."""
        from app.coordination.handler_base import create_handler_stats

        stats = create_handler_stats()
        assert isinstance(stats, HandlerStats)
        assert stats.events_processed == 0

    def test_create_handler_stats_with_custom(self):
        """Should create stats with custom values."""
        from app.coordination.handler_base import create_handler_stats

        stats = create_handler_stats(queue_depth=100, active_jobs=5)

        assert stats.custom_stats["queue_depth"] == 100
        assert stats.custom_stats["active_jobs"] == 5

    def test_safe_subscribe_success(self):
        """Should return True on success."""
        from app.coordination.handler_base import safe_subscribe

        handler = ConcreteHandler()
        result = safe_subscribe(handler)
        assert result is True

    def test_safe_subscribe_failure(self):
        """Should return fallback on failure."""
        from app.coordination.handler_base import safe_subscribe

        handler = ConcreteHandler()
        handler.subscribe = MagicMock(side_effect=RuntimeError("Failed"))

        result = safe_subscribe(handler, fallback=False)
        assert result is False

    def test_safe_subscribe_custom_fallback(self):
        """Should use custom fallback value."""
        from app.coordination.handler_base import safe_subscribe

        handler = ConcreteHandler()
        handler.subscribe = MagicMock(side_effect=RuntimeError("Failed"))

        result = safe_subscribe(handler, fallback=True)
        assert result is True


class TestBackwardCompatAliases:
    """Tests for backward-compatible class aliases."""

    def test_base_event_handler_is_handler_base(self):
        """BaseEventHandler should be HandlerBase."""
        from app.coordination.handler_base import BaseEventHandler

        assert BaseEventHandler is HandlerBase

    def test_base_singleton_handler_is_handler_base(self):
        """BaseSingletonHandler should be HandlerBase."""
        from app.coordination.handler_base import BaseSingletonHandler

        assert BaseSingletonHandler is HandlerBase

    def test_multi_event_handler_is_handler_base(self):
        """MultiEventHandler should be HandlerBase."""
        from app.coordination.handler_base import MultiEventHandler

        assert MultiEventHandler is HandlerBase


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Should export expected classes."""
        from app.coordination import handler_base

        # Canonical exports
        assert hasattr(handler_base, "HandlerBase")
        assert hasattr(handler_base, "HandlerStats")
        assert hasattr(handler_base, "HealthCheckResult")
        assert hasattr(handler_base, "CoordinatorStatus")
        assert hasattr(handler_base, "EventHandlerConfig")
        # Backward-compatible aliases
        assert hasattr(handler_base, "BaseEventHandler")
        assert hasattr(handler_base, "BaseSingletonHandler")
        assert hasattr(handler_base, "MultiEventHandler")
        # Helper functions (backward-compat for base_handler.py)
        assert hasattr(handler_base, "create_handler_stats")
        assert hasattr(handler_base, "safe_subscribe")
        # SafeEventEmitterMixin added Dec 30, 2025
        assert hasattr(handler_base, "SafeEventEmitterMixin")
        assert len(handler_base.__all__) == 11


class TestDeduplicationTTL:
    """Tests for event deduplication TTL behavior."""

    def test_dedup_expires_after_ttl(self):
        """Events should expire after TTL."""
        import time

        handler = ConcreteHandler()
        handler.DEDUP_TTL_SECONDS = 0.05  # Very short for testing
        event = {"id": "123"}

        # First occurrence
        assert handler._is_duplicate_event(event) is False
        # Immediate duplicate
        assert handler._is_duplicate_event(event) is True

        # Wait for TTL
        time.sleep(0.1)

        # Should no longer be duplicate
        assert handler._is_duplicate_event(event) is False

    def test_dedup_prunes_on_overflow(self):
        """Should prune old entries when max size reached."""
        import time

        handler = ConcreteHandler()
        handler.DEDUP_MAX_SIZE = 5
        handler.DEDUP_TTL_SECONDS = 0.01

        # Add many events
        for i in range(10):
            handler._is_duplicate_event({"id": i})

        # Wait for TTL
        time.sleep(0.02)

        # Trigger pruning
        handler._is_duplicate_event({"id": 100})

        # Old entries should be pruned
        assert len(handler._seen_events) <= handler.DEDUP_MAX_SIZE + 1


class TestMainLoopErrorHandling:
    """Tests for error handling in main loop."""

    @pytest.mark.asyncio
    async def test_main_loop_continues_after_error(self):
        """Main loop should continue after _run_cycle error."""

        class FailingHandler(HandlerBase):
            def __init__(self):
                super().__init__(name="failing", cycle_interval=0.01)
                self.calls = 0

            async def _run_cycle(self) -> None:
                self.calls += 1
                if self.calls == 1:
                    raise ValueError("Test error")

        handler = FailingHandler()
        await handler.start()
        await asyncio.sleep(0.05)
        await handler.stop()

        assert handler.calls >= 2  # Continued after error
        assert handler._stats.errors_count >= 1

    @pytest.mark.asyncio
    async def test_main_loop_stops_on_cancelled(self):
        """Main loop should stop on CancelledError."""
        handler = ConcreteHandler(cycle_interval=10.0)
        await handler.start()
        await handler.stop()
        assert handler.is_running is False


class TestFireAndForgetTaskHelpers:
    """Tests for fire-and-forget task helper methods (January 2026)."""

    @pytest.mark.asyncio
    async def test_safe_create_task_creates_task(self):
        """_safe_create_task should create and return a task."""
        handler = ConcreteHandler()
        handler._running = True  # Simulate started state

        completed = False

        async def test_coro():
            nonlocal completed
            completed = True

        task = handler._safe_create_task(test_coro(), context="test")
        assert task is not None

        # Wait for task to complete
        await asyncio.sleep(0.01)
        assert completed is True

    @pytest.mark.asyncio
    async def test_safe_create_task_handles_errors(self):
        """_safe_create_task should log errors without crashing."""
        handler = ConcreteHandler()
        handler._running = True

        async def failing_coro():
            raise ValueError("Test error")

        task = handler._safe_create_task(failing_coro(), context="test")
        assert task is not None

        # Wait for task to complete (error should be caught)
        await asyncio.sleep(0.01)

        # Error should be recorded
        assert handler._stats.errors_count >= 1
        assert "Test error" in handler._stats.last_error

    @pytest.mark.asyncio
    async def test_safe_create_task_with_name(self):
        """_safe_create_task should accept optional task name."""
        handler = ConcreteHandler()
        handler._running = True

        async def test_coro():
            await asyncio.sleep(0)

        task = handler._safe_create_task(
            test_coro(),
            context="test",
            name="my_named_task"
        )
        assert task is not None
        assert task.get_name() == "my_named_task"
        await task

    def test_safe_create_task_returns_none_on_runtime_error(self):
        """_safe_create_task should return None if event loop is closed."""
        handler = ConcreteHandler()

        async def test_coro():
            pass

        # Simulate closed event loop scenario (mocking)
        with patch("asyncio.create_task", side_effect=RuntimeError("No loop")):
            result = handler._safe_create_task(test_coro(), context="test")
            assert result is None

    def test_handle_task_error_logs_exception(self):
        """_handle_task_error should record exception in stats."""
        handler = ConcreteHandler()
        initial_errors = handler._stats.errors_count

        # Create a mock task with an exception
        mock_task = MagicMock()
        mock_task.exception.return_value = ValueError("Test exception")

        handler._handle_task_error(mock_task, context="test_context")

        assert handler._stats.errors_count == initial_errors + 1
        assert "Test exception" in handler._stats.last_error
        assert "test_context" in handler._stats.last_error

    def test_handle_task_error_ignores_cancelled(self):
        """_handle_task_error should ignore CancelledError."""
        handler = ConcreteHandler()
        initial_errors = handler._stats.errors_count

        mock_task = MagicMock()
        mock_task.exception.side_effect = asyncio.CancelledError()

        handler._handle_task_error(mock_task, context="test")

        # Should not record as error
        assert handler._stats.errors_count == initial_errors

    def test_handle_task_error_ignores_invalid_state(self):
        """_handle_task_error should ignore InvalidStateError."""
        handler = ConcreteHandler()
        initial_errors = handler._stats.errors_count

        mock_task = MagicMock()
        mock_task.exception.side_effect = asyncio.InvalidStateError()

        handler._handle_task_error(mock_task, context="test")

        # Should not record as error
        assert handler._stats.errors_count == initial_errors

    def test_try_emit_event_returns_false_if_no_emitter(self):
        """_try_emit_event should return False if emitter is None."""
        handler = ConcreteHandler()

        result = handler._try_emit_event(
            "MY_EVENT",
            {"key": "value"},
            None,
            context="test"
        )
        assert result is False

    def test_try_emit_event_calls_sync_emitter(self):
        """_try_emit_event should call synchronous emitter function."""
        handler = ConcreteHandler()
        mock_emitter = MagicMock(return_value=None)

        result = handler._try_emit_event(
            "MY_EVENT",
            {"key": "value"},
            mock_emitter,
            context="test"
        )

        assert result is True
        mock_emitter.assert_called_once_with(key="value")

    @pytest.mark.asyncio
    async def test_try_emit_event_handles_async_emitter(self):
        """_try_emit_event should handle async emitter with _safe_create_task."""
        handler = ConcreteHandler()
        handler._running = True

        async_emitter = AsyncMock(return_value=None)

        result = handler._try_emit_event(
            "MY_EVENT",
            {"key": "value"},
            async_emitter,
            context="test"
        )

        assert result is True
        # Let the background task complete
        await asyncio.sleep(0.01)
        async_emitter.assert_called_once_with(key="value")

    def test_try_emit_event_handles_runtime_error(self):
        """_try_emit_event should handle RuntimeError gracefully."""
        handler = ConcreteHandler()
        mock_emitter = MagicMock(side_effect=RuntimeError("Loop closed"))

        result = handler._try_emit_event(
            "MY_EVENT",
            {"key": "value"},
            mock_emitter,
            context="test"
        )

        assert result is False

    def test_try_emit_event_handles_type_error(self):
        """_try_emit_event should handle TypeError from bad payload."""
        handler = ConcreteHandler()
        mock_emitter = MagicMock(side_effect=TypeError("Bad argument"))

        result = handler._try_emit_event(
            "MY_EVENT",
            {"key": "value"},
            mock_emitter,
            context="test"
        )

        assert result is False
