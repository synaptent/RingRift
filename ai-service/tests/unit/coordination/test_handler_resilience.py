"""Unit tests for HandlerResilience - exception boundaries and timeout management.

December 2025: Tests for resilient handler wrappers and metrics tracking.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.handler_resilience import (
    HandlerMetrics,
    ResilientCoordinatorMixin,
    ResilientHandlerConfig,
    get_all_handler_metrics,
    get_handler_metrics,
    make_handlers_resilient,
    reset_handler_metrics,
    resilient_handler,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_metrics_before_each():
    """Reset global metrics registry before each test."""
    reset_handler_metrics()
    yield
    reset_handler_metrics()


# =============================================================================
# Test ResilientHandlerConfig
# =============================================================================


class TestResilientHandlerConfig:
    """Tests for ResilientHandlerConfig dataclass."""

    def test_default_values(self):
        """Config has expected defaults."""
        config = ResilientHandlerConfig()
        assert config.timeout_seconds == 30.0
        assert config.emit_failure_events is True
        assert config.emit_timeout_events is True
        assert config.log_exceptions is True
        assert config.log_timeouts is True
        assert config.retry_on_timeout is False
        assert config.max_consecutive_failures == 5

    def test_custom_values(self):
        """Config accepts custom values."""
        config = ResilientHandlerConfig(
            timeout_seconds=10.0,
            emit_failure_events=False,
            emit_timeout_events=False,
            log_exceptions=False,
            log_timeouts=False,
            retry_on_timeout=True,
            max_consecutive_failures=3,
        )
        assert config.timeout_seconds == 10.0
        assert config.emit_failure_events is False
        assert config.emit_timeout_events is False
        assert config.log_exceptions is False
        assert config.log_timeouts is False
        assert config.retry_on_timeout is True
        assert config.max_consecutive_failures == 3


# =============================================================================
# Test HandlerMetrics
# =============================================================================


class TestHandlerMetrics:
    """Tests for HandlerMetrics dataclass."""

    def test_default_values(self):
        """Metrics has expected defaults."""
        metrics = HandlerMetrics(handler_name="test", coordinator="TestCoord")
        assert metrics.handler_name == "test"
        assert metrics.coordinator == "TestCoord"
        assert metrics.invocation_count == 0
        assert metrics.success_count == 0
        assert metrics.failure_count == 0
        assert metrics.timeout_count == 0
        assert metrics.total_duration_ms == 0.0
        assert metrics.last_failure_time == 0.0
        assert metrics.last_failure_error == ""
        assert metrics.consecutive_failures == 0

    def test_success_rate_no_invocations(self):
        """Success rate is 1.0 when no invocations."""
        metrics = HandlerMetrics(handler_name="test", coordinator="TestCoord")
        assert metrics.success_rate == 1.0

    def test_success_rate_with_invocations(self):
        """Success rate calculated correctly."""
        metrics = HandlerMetrics(
            handler_name="test",
            coordinator="TestCoord",
            invocation_count=10,
            success_count=8,
        )
        assert metrics.success_rate == 0.8

    def test_avg_duration_no_successes(self):
        """Average duration is 0.0 when no successes."""
        metrics = HandlerMetrics(handler_name="test", coordinator="TestCoord")
        assert metrics.avg_duration_ms == 0.0

    def test_avg_duration_with_successes(self):
        """Average duration calculated correctly."""
        metrics = HandlerMetrics(
            handler_name="test",
            coordinator="TestCoord",
            success_count=5,
            total_duration_ms=100.0,
        )
        assert metrics.avg_duration_ms == 20.0

    def test_record_success(self):
        """record_success updates metrics correctly."""
        metrics = HandlerMetrics(handler_name="test", coordinator="TestCoord")
        metrics.consecutive_failures = 3  # Simulate previous failures

        metrics.record_success(50.0)

        assert metrics.invocation_count == 1
        assert metrics.success_count == 1
        assert metrics.total_duration_ms == 50.0
        assert metrics.consecutive_failures == 0  # Reset on success

    def test_record_failure(self):
        """record_failure updates metrics correctly."""
        metrics = HandlerMetrics(handler_name="test", coordinator="TestCoord")
        before_time = time.time()

        metrics.record_failure("ValueError: test error")

        assert metrics.invocation_count == 1
        assert metrics.failure_count == 1
        assert metrics.last_failure_time >= before_time
        assert metrics.last_failure_error == "ValueError: test error"
        assert metrics.consecutive_failures == 1

    def test_record_timeout(self):
        """record_timeout updates metrics correctly."""
        metrics = HandlerMetrics(handler_name="test", coordinator="TestCoord")

        metrics.record_timeout()

        assert metrics.invocation_count == 1
        assert metrics.timeout_count == 1
        assert metrics.consecutive_failures == 1

    def test_consecutive_failures_accumulate(self):
        """Consecutive failures increment on each failure/timeout."""
        metrics = HandlerMetrics(handler_name="test", coordinator="TestCoord")

        metrics.record_failure("error 1")
        assert metrics.consecutive_failures == 1

        metrics.record_timeout()
        assert metrics.consecutive_failures == 2

        metrics.record_failure("error 2")
        assert metrics.consecutive_failures == 3

        metrics.record_success(10.0)
        assert metrics.consecutive_failures == 0  # Reset


# =============================================================================
# Test Metrics Registry
# =============================================================================


class TestMetricsRegistry:
    """Tests for global metrics registry functions."""

    def test_get_handler_metrics_creates_new(self):
        """get_handler_metrics creates new metrics if not exists."""
        metrics = get_handler_metrics("new_handler", "NewCoord")

        assert metrics.handler_name == "new_handler"
        assert metrics.coordinator == "NewCoord"
        assert metrics.invocation_count == 0

    def test_get_handler_metrics_returns_existing(self):
        """get_handler_metrics returns same instance for same key."""
        m1 = get_handler_metrics("handler", "Coord")
        m1.invocation_count = 5

        m2 = get_handler_metrics("handler", "Coord")

        assert m1 is m2
        assert m2.invocation_count == 5

    def test_get_all_handler_metrics(self):
        """get_all_handler_metrics returns copy of registry."""
        get_handler_metrics("h1", "C1")
        get_handler_metrics("h2", "C2")

        all_metrics = get_all_handler_metrics()

        assert len(all_metrics) == 2
        assert "C1:h1" in all_metrics
        assert "C2:h2" in all_metrics

    def test_reset_handler_metrics(self):
        """reset_handler_metrics clears registry."""
        get_handler_metrics("handler", "Coord")
        assert len(get_all_handler_metrics()) == 1

        reset_handler_metrics()

        assert len(get_all_handler_metrics()) == 0


# =============================================================================
# Test resilient_handler Wrapper
# =============================================================================


class TestResilientHandler:
    """Tests for resilient_handler wrapper function."""

    @pytest.mark.asyncio
    async def test_wraps_successful_handler(self):
        """Wrapper executes handler and records success."""
        async def handler(event):
            await asyncio.sleep(0.01)
            return "result"

        wrapped = resilient_handler(handler, coordinator="TestCoord")
        event = MagicMock()
        event.event_type = MagicMock(value="TEST_EVENT")

        await wrapped(event)

        metrics = get_handler_metrics("handler", "TestCoord")
        assert metrics.invocation_count == 1
        assert metrics.success_count == 1
        assert metrics.failure_count == 0
        assert metrics.total_duration_ms > 0

    @pytest.mark.asyncio
    async def test_catches_exception(self):
        """Wrapper catches exceptions and records failure."""
        async def handler(event):
            raise ValueError("test error")

        config = ResilientHandlerConfig(emit_failure_events=False, log_exceptions=False)
        wrapped = resilient_handler(handler, coordinator="TestCoord", config=config)
        event = MagicMock()
        event.event_type = MagicMock(value="TEST_EVENT")

        # Should not raise
        await wrapped(event)

        metrics = get_handler_metrics("handler", "TestCoord")
        assert metrics.invocation_count == 1
        assert metrics.success_count == 0
        assert metrics.failure_count == 1
        assert "ValueError: test error" in metrics.last_failure_error

    @pytest.mark.asyncio
    async def test_enforces_timeout(self):
        """Wrapper enforces timeout and records timeout."""
        async def slow_handler(event):
            await asyncio.sleep(1.0)  # Much longer than timeout

        config = ResilientHandlerConfig(
            timeout_seconds=0.05,
            emit_timeout_events=False,
            log_timeouts=False,
        )
        wrapped = resilient_handler(slow_handler, coordinator="TestCoord", config=config)
        event = MagicMock()
        event.event_type = MagicMock(value="TEST_EVENT")

        await wrapped(event)

        metrics = get_handler_metrics("slow_handler", "TestCoord")
        assert metrics.invocation_count == 1
        assert metrics.timeout_count == 1
        assert metrics.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_emits_failure_event(self):
        """Wrapper emits HANDLER_FAILED event on exception."""
        async def handler(event):
            raise RuntimeError("crash")

        config = ResilientHandlerConfig(emit_failure_events=True, log_exceptions=False)
        wrapped = resilient_handler(handler, coordinator="TestCoord", config=config)
        event = MagicMock()
        event.event_type = MagicMock(value="TEST_EVENT")

        with patch(
            "app.coordination.handler_resilience._emit_failure_event",
            new_callable=AsyncMock,
        ) as mock_emit:
            await wrapped(event)

            mock_emit.assert_called_once()
            # _emit_failure_event is called with positional args
            call_args = mock_emit.call_args.args
            assert call_args[0] == "handler"  # handler_name
            assert call_args[1] == "TEST_EVENT"  # event_type
            assert "RuntimeError: crash" in call_args[2]  # error

    @pytest.mark.asyncio
    async def test_emits_timeout_event(self):
        """Wrapper emits HANDLER_TIMEOUT event on timeout."""
        async def slow_handler(event):
            await asyncio.sleep(1.0)

        config = ResilientHandlerConfig(
            timeout_seconds=0.05,
            emit_timeout_events=True,
            log_timeouts=False,
        )
        wrapped = resilient_handler(slow_handler, coordinator="TestCoord", config=config)
        event = MagicMock()
        event.event_type = MagicMock(value="SLOW_EVENT")

        with patch(
            "app.coordination.handler_resilience._emit_timeout_event",
            new_callable=AsyncMock,
        ) as mock_emit:
            await wrapped(event)

            mock_emit.assert_called_once()
            # _emit_timeout_event is called with positional args
            call_args = mock_emit.call_args.args
            assert call_args[0] == "slow_handler"  # handler_name
            assert call_args[2] == 0.05  # timeout_seconds

    @pytest.mark.asyncio
    async def test_emits_health_degraded_on_threshold(self):
        """Wrapper emits health degraded when consecutive failures reach threshold."""
        async def failing_handler(event):
            raise ValueError("always fails")

        config = ResilientHandlerConfig(
            max_consecutive_failures=2,
            emit_failure_events=False,
            log_exceptions=False,
        )
        wrapped = resilient_handler(failing_handler, coordinator="TestCoord", config=config)
        event = MagicMock()
        event.event_type = MagicMock(value="TEST_EVENT")

        with patch(
            "app.coordination.handler_resilience._emit_health_degraded",
            new_callable=AsyncMock,
        ) as mock_emit:
            # First failure - no health degraded
            await wrapped(event)
            mock_emit.assert_not_called()

            # Second failure - triggers health degraded
            await wrapped(event)
            mock_emit.assert_called_once()
            # _emit_health_degraded is called with positional args
            call_args = mock_emit.call_args.args
            assert call_args[2] == 2  # consecutive_failures

    @pytest.mark.asyncio
    async def test_preserves_function_name(self):
        """Wrapper preserves original function name."""
        async def my_special_handler(event):
            pass

        wrapped = resilient_handler(my_special_handler)

        assert wrapped.__name__ == "my_special_handler"

    @pytest.mark.asyncio
    async def test_handles_event_without_event_type(self):
        """Wrapper handles events without event_type attribute."""
        async def handler(event):
            pass

        wrapped = resilient_handler(handler, coordinator="TestCoord")
        event = {}  # Plain dict, no event_type

        # Should not raise
        await wrapped(event)

        metrics = get_handler_metrics("handler", "TestCoord")
        assert metrics.success_count == 1


# =============================================================================
# Test make_handlers_resilient
# =============================================================================


class TestMakeHandlersResilient:
    """Tests for make_handlers_resilient function."""

    def test_wraps_on_methods(self):
        """Wraps methods starting with _on_ prefix."""

        class MyCoordinator:
            async def _on_event_one(self, event):
                pass

            async def _on_event_two(self, event):
                pass

            async def _helper_method(self, event):
                pass

            def sync_method(self):
                pass

        coord = MyCoordinator()
        wrapped = make_handlers_resilient(coord, "MyCoordinator")

        assert "_on_event_one" in wrapped
        assert "_on_event_two" in wrapped
        assert "_helper_method" not in wrapped  # Different prefix
        assert "sync_method" not in wrapped  # Not async

    def test_replaces_methods_on_instance(self):
        """Replaces methods on the instance itself."""

        class MyCoordinator:
            async def _on_event(self, event):
                return "original"

        coord = MyCoordinator()
        original_method = coord._on_event

        make_handlers_resilient(coord, "MyCoordinator")

        assert coord._on_event is not original_method

    @pytest.mark.asyncio
    async def test_wrapped_handlers_record_metrics(self):
        """Wrapped handlers record metrics."""

        class MyCoordinator:
            async def _on_test_event(self, event):
                await asyncio.sleep(0.01)

        coord = MyCoordinator()
        make_handlers_resilient(coord, "MyCoordinator")

        event = MagicMock()
        event.event_type = MagicMock(value="TEST")
        await coord._on_test_event(event)

        metrics = get_handler_metrics("_on_test_event", "MyCoordinator")
        assert metrics.invocation_count == 1
        assert metrics.success_count == 1

    def test_custom_prefix(self):
        """Uses custom handler prefix."""

        class MyCoordinator:
            async def handle_event(self, event):
                pass

            async def _on_event(self, event):
                pass

        coord = MyCoordinator()
        wrapped = make_handlers_resilient(
            coord,
            "MyCoordinator",
            handler_prefix="handle_",
        )

        assert "handle_event" in wrapped
        assert "_on_event" not in wrapped

    def test_applies_config(self):
        """Applies provided config to wrapped handlers."""

        class MyCoordinator:
            async def _on_slow(self, event):
                await asyncio.sleep(1.0)

        config = ResilientHandlerConfig(
            timeout_seconds=0.05,
            emit_timeout_events=False,
            log_timeouts=False,
        )
        coord = MyCoordinator()
        make_handlers_resilient(coord, "MyCoordinator", config=config)

        # Verify config is applied by checking timeout behavior
        event = MagicMock()
        event.event_type = MagicMock(value="TEST")

        # Run the handler (should timeout)
        import asyncio
        asyncio.get_event_loop().run_until_complete(coord._on_slow(event))

        metrics = get_handler_metrics("_on_slow", "MyCoordinator")
        assert metrics.timeout_count == 1


# =============================================================================
# Test ResilientCoordinatorMixin
# =============================================================================


class TestResilientCoordinatorMixin:
    """Tests for ResilientCoordinatorMixin class."""

    def test_has_expected_class_attributes(self):
        """Mixin has expected class attributes."""
        assert ResilientCoordinatorMixin._coordinator_name == "unknown"
        assert ResilientCoordinatorMixin._handler_config is None
        assert ResilientCoordinatorMixin._events_processed == 0
        assert ResilientCoordinatorMixin._handler_failures == 0

    def test_wrap_handlers_method(self):
        """_wrap_handlers wraps all _on_* methods."""

        class MyCoordinator(ResilientCoordinatorMixin):
            _coordinator_name = "MyCoordinator"

            async def _on_event_a(self, event):
                pass

            async def _on_event_b(self, event):
                pass

        coord = MyCoordinator()
        original_a = coord._on_event_a
        original_b = coord._on_event_b

        coord._wrap_handlers()

        # Methods should be replaced with wrapped versions
        assert coord._on_event_a is not original_a
        assert coord._on_event_b is not original_b

    @pytest.mark.asyncio
    async def test_get_handler_health_empty(self):
        """_get_handler_health returns empty for fresh coordinator."""

        class MyCoordinator(ResilientCoordinatorMixin):
            _coordinator_name = "MyCoordinator"

        coord = MyCoordinator()
        health = coord._get_handler_health()

        assert health["coordinator"] == "MyCoordinator"
        assert health["handler_count"] == 0
        assert health["total_invocations"] == 0
        assert health["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_get_handler_health_with_metrics(self):
        """_get_handler_health includes handler metrics."""

        class MyCoordinator(ResilientCoordinatorMixin):
            _coordinator_name = "MyCoordinator"

            async def _on_test(self, event):
                pass

        coord = MyCoordinator()
        coord._wrap_handlers()

        # Invoke handler
        event = MagicMock()
        event.event_type = MagicMock(value="TEST")
        await coord._on_test(event)

        health = coord._get_handler_health()

        assert health["handler_count"] == 1
        assert health["total_invocations"] == 1
        assert health["total_failures"] == 0
        assert health["success_rate"] == 1.0
        assert "_on_test" in health["handlers"]
        assert health["handlers"]["_on_test"]["invocations"] == 1

    @pytest.mark.asyncio
    async def test_get_handler_health_calculates_success_rate(self):
        """Success rate calculated correctly with failures."""

        class MyCoordinator(ResilientCoordinatorMixin):
            _coordinator_name = "TestCoord"
            _handler_config = ResilientHandlerConfig(
                emit_failure_events=False,
                log_exceptions=False,
            )

            async def _on_failing(self, event):
                raise ValueError("fail")

        coord = MyCoordinator()
        coord._wrap_handlers()

        event = MagicMock()
        event.event_type = MagicMock(value="TEST")

        # Invoke twice to get failures
        await coord._on_failing(event)
        await coord._on_failing(event)

        health = coord._get_handler_health()

        assert health["total_invocations"] == 2
        assert health["total_failures"] == 2
        assert health["success_rate"] == 0.0


# =============================================================================
# Test Event Emission Helpers
# =============================================================================


class TestEventEmissionHelpers:
    """Tests for event emission helper functions."""

    @pytest.mark.asyncio
    async def test_emit_failure_event_success(self):
        """_emit_failure_event calls event emitter correctly."""
        with patch(
            "app.coordination.event_emitters.emit_handler_failed",
            new_callable=AsyncMock,
        ) as mock_emit:
            from app.coordination.handler_resilience import _emit_failure_event

            await _emit_failure_event(
                handler_name="test_handler",
                event_type="TEST_EVENT",
                error="ValueError: oops",
                coordinator="TestCoord",
            )

            mock_emit.assert_called_once_with(
                handler_name="test_handler",
                event_type="TEST_EVENT",
                error="ValueError: oops",
                coordinator="TestCoord",
            )

    @pytest.mark.asyncio
    async def test_emit_failure_event_handles_error(self):
        """_emit_failure_event handles import/emit errors gracefully."""
        with patch(
            "app.coordination.event_emitters.emit_handler_failed",
            side_effect=RuntimeError("emit failed"),
        ):
            from app.coordination.handler_resilience import _emit_failure_event

            # Should not raise
            await _emit_failure_event(
                handler_name="test",
                event_type="TEST",
                error="error",
                coordinator="Coord",
            )

    @pytest.mark.asyncio
    async def test_emit_timeout_event_success(self):
        """_emit_timeout_event calls event emitter correctly."""
        with patch(
            "app.coordination.event_emitters.emit_handler_timeout",
            new_callable=AsyncMock,
        ) as mock_emit:
            from app.coordination.handler_resilience import _emit_timeout_event

            await _emit_timeout_event(
                handler_name="slow_handler",
                event_type="SLOW_EVENT",
                timeout_seconds=30.0,
                coordinator="TestCoord",
            )

            mock_emit.assert_called_once_with(
                handler_name="slow_handler",
                event_type="SLOW_EVENT",
                timeout_seconds=30.0,
                coordinator="TestCoord",
            )

    @pytest.mark.asyncio
    async def test_emit_health_degraded_success(self):
        """_emit_health_degraded calls event emitter correctly."""
        with patch(
            "app.coordination.event_emitters.emit_coordinator_health_degraded",
            new_callable=AsyncMock,
        ) as mock_emit:
            from app.coordination.handler_resilience import _emit_health_degraded

            await _emit_health_degraded(
                coordinator="TestCoord",
                handler_name="bad_handler",
                consecutive_failures=5,
            )

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args[1]
            assert call_args["coordinator_name"] == "TestCoord"
            assert "bad_handler" in call_args["reason"]
            assert call_args["health_score"] == 0.5


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_handler_with_event_attribute(self):
        """Handles events with 'event' attribute instead of 'event_type'."""
        async def handler(event):
            pass

        wrapped = resilient_handler(handler, coordinator="TestCoord")
        event = MagicMock()
        del event.event_type
        event.event = MagicMock(value="ALTERNATE_EVENT")

        await wrapped(event)

        metrics = get_handler_metrics("handler", "TestCoord")
        assert metrics.success_count == 1

    @pytest.mark.asyncio
    async def test_multiple_handlers_same_coordinator(self):
        """Multiple handlers under same coordinator tracked separately."""
        async def handler_a(event):
            pass

        async def handler_b(event):
            raise ValueError("fail")

        config = ResilientHandlerConfig(emit_failure_events=False, log_exceptions=False)
        wrapped_a = resilient_handler(handler_a, coordinator="Coord")
        wrapped_b = resilient_handler(handler_b, coordinator="Coord", config=config)

        event = MagicMock()
        event.event_type = MagicMock(value="TEST")

        await wrapped_a(event)
        await wrapped_b(event)

        metrics_a = get_handler_metrics("handler_a", "Coord")
        metrics_b = get_handler_metrics("handler_b", "Coord")

        assert metrics_a.success_count == 1
        assert metrics_a.failure_count == 0
        assert metrics_b.success_count == 0
        assert metrics_b.failure_count == 1

    @pytest.mark.asyncio
    async def test_handler_without_coordinator(self):
        """Handler works without coordinator name."""
        async def handler(event):
            pass

        wrapped = resilient_handler(handler)  # No coordinator
        event = MagicMock()
        event.event_type = MagicMock(value="TEST")

        await wrapped(event)

        metrics = get_handler_metrics("handler", "")
        assert metrics.success_count == 1

    @pytest.mark.asyncio
    async def test_zero_timeout_behavior(self):
        """Very short timeout still allows fast handlers."""
        async def fast_handler(event):
            pass  # No await, nearly instant

        config = ResilientHandlerConfig(
            timeout_seconds=0.5,  # 500ms should be plenty
            emit_timeout_events=False,
        )
        wrapped = resilient_handler(fast_handler, config=config)
        event = MagicMock()
        event.event_type = MagicMock(value="TEST")

        await wrapped(event)

        metrics = get_handler_metrics("fast_handler", "")
        assert metrics.success_count == 1
        assert metrics.timeout_count == 0

    def test_metrics_key_format(self):
        """Metrics key uses coordinator:handler_name format."""
        get_handler_metrics("my_handler", "MyCoordinator")

        all_metrics = get_all_handler_metrics()
        assert "MyCoordinator:my_handler" in all_metrics

    @pytest.mark.asyncio
    async def test_consecutive_success_after_failure(self):
        """Success after failure resets consecutive failures."""
        call_count = 0

        async def flaky_handler(event):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("first call fails")
            # Subsequent calls succeed

        config = ResilientHandlerConfig(emit_failure_events=False, log_exceptions=False)
        wrapped = resilient_handler(flaky_handler, coordinator="Coord", config=config)
        event = MagicMock()
        event.event_type = MagicMock(value="TEST")

        # First call fails
        await wrapped(event)
        metrics = get_handler_metrics("flaky_handler", "Coord")
        assert metrics.consecutive_failures == 1

        # Second call succeeds
        await wrapped(event)
        assert metrics.consecutive_failures == 0
        assert metrics.success_count == 1
        assert metrics.failure_count == 1
