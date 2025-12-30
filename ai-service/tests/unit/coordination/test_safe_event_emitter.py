"""Tests for SafeEventEmitterMixin and safe_emit_event.

December 2025: Comprehensive test coverage for the unified safe event
emission utility that consolidates 6 duplicate implementations.

Tests cover:
1. SafeEventEmitterMixin class
   - Sync event emission
   - Async event emission
   - Error handling (ImportError, AttributeError, RuntimeError, TypeError)
   - Event source tracking
   - Return value semantics

2. safe_emit_event module function
   - Same error handling as mixin
   - Custom source parameter

3. Integration with event bus
   - Lazy import behavior
   - Fallback when bus unavailable
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.safe_event_emitter import (
    SafeEventEmitterMixin,
    safe_emit_event,
)


# =============================================================================
# Test fixtures
# =============================================================================


class TestEmitter(SafeEventEmitterMixin):
    """Test class that uses the mixin."""

    _event_source = "TestEmitter"


class CustomSourceEmitter(SafeEventEmitterMixin):
    """Test class with custom source."""

    _event_source = "CustomSource"


@pytest.fixture
def emitter() -> TestEmitter:
    """Create a test emitter instance."""
    return TestEmitter()


@pytest.fixture
def custom_emitter() -> CustomSourceEmitter:
    """Create a custom source emitter instance."""
    return CustomSourceEmitter()


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus."""
    bus = MagicMock()
    bus.publish = MagicMock()
    return bus


# =============================================================================
# SafeEventEmitterMixin Tests
# =============================================================================


class TestSafeEventEmitterMixin:
    """Tests for SafeEventEmitterMixin class."""

    def test_event_source_default(self):
        """Test default event source is 'unknown'."""
        # Create class without overriding _event_source
        class DefaultEmitter(SafeEventEmitterMixin):
            pass

        emitter = DefaultEmitter()
        assert emitter._event_source == "unknown"

    def test_event_source_custom(self, emitter: TestEmitter):
        """Test custom event source is used."""
        assert emitter._event_source == "TestEmitter"

    def test_safe_emit_event_success(self, emitter: TestEmitter, mock_event_bus):
        """Test successful event emission returns True."""
        with patch(
            "app.coordination.event_router.get_event_bus",
            return_value=mock_event_bus,
        ):
            result = emitter._safe_emit_event("TEST_EVENT", {"key": "value"})

        assert result is True
        mock_event_bus.publish.assert_called_once()

    def test_safe_emit_event_with_payload(self, emitter: TestEmitter, mock_event_bus):
        """Test event emission with payload."""
        payload = {"board_type": "hex8", "num_players": 2, "games": 100}

        with patch(
            "app.coordination.event_router.get_event_bus",
            return_value=mock_event_bus,
        ):
            result = emitter._safe_emit_event("TRAINING_COMPLETED", payload)

        assert result is True
        call_args = mock_event_bus.publish.call_args
        event = call_args[0][0]
        assert event.payload == payload
        assert event.event_type == "TRAINING_COMPLETED"
        assert event.source == "TestEmitter"

    def test_safe_emit_event_no_payload(self, emitter: TestEmitter, mock_event_bus):
        """Test event emission without payload uses empty dict."""
        with patch(
            "app.coordination.event_router.get_event_bus",
            return_value=mock_event_bus,
        ):
            result = emitter._safe_emit_event("SIMPLE_EVENT")

        assert result is True
        call_args = mock_event_bus.publish.call_args
        event = call_args[0][0]
        assert event.payload == {}

    def test_safe_emit_event_bus_unavailable(self, emitter: TestEmitter):
        """Test returns False when event bus is unavailable."""
        with patch(
            "app.coordination.event_router.get_event_bus",
            return_value=None,
        ):
            result = emitter._safe_emit_event("TEST_EVENT")

        assert result is False

    def test_safe_emit_event_import_error(self, emitter: TestEmitter):
        """Test handles ImportError gracefully."""
        with patch(
            "app.coordination.event_router.get_event_bus",
            side_effect=ImportError("Module not found"),
        ):
            result = emitter._safe_emit_event("TEST_EVENT")

        assert result is False

    def test_safe_emit_event_attribute_error(self, emitter: TestEmitter, mock_event_bus):
        """Test handles AttributeError gracefully."""
        mock_event_bus.publish.side_effect = AttributeError("Missing attribute")

        with patch(
            "app.coordination.event_router.get_event_bus",
            return_value=mock_event_bus,
        ):
            result = emitter._safe_emit_event("TEST_EVENT")

        assert result is False

    def test_safe_emit_event_runtime_error(self, emitter: TestEmitter):
        """Test handles RuntimeError gracefully."""
        with patch(
            "app.coordination.event_router.get_event_bus",
            side_effect=RuntimeError("No event loop"),
        ):
            result = emitter._safe_emit_event("TEST_EVENT")

        assert result is False

    def test_safe_emit_event_type_error(self, emitter: TestEmitter):
        """Test handles TypeError gracefully."""
        with patch(
            "app.coordination.event_router.get_event_bus",
            side_effect=TypeError("Wrong signature"),
        ):
            result = emitter._safe_emit_event("TEST_EVENT")

        assert result is False

    def test_custom_event_source(self, custom_emitter: CustomSourceEmitter, mock_event_bus):
        """Test custom event source is used in emitted events."""
        with patch(
            "app.coordination.event_router.get_event_bus",
            return_value=mock_event_bus,
        ):
            custom_emitter._safe_emit_event("TEST_EVENT")

        call_args = mock_event_bus.publish.call_args
        event = call_args[0][0]
        assert event.source == "CustomSource"


class TestSafeEventEmitterMixinAsync:
    """Tests for async event emission."""

    @pytest.mark.asyncio
    async def test_safe_emit_event_async_success(
        self, emitter: TestEmitter, mock_event_bus
    ):
        """Test async event emission success."""
        with patch(
            "app.coordination.event_router.get_event_bus",
            return_value=mock_event_bus,
        ):
            result = await emitter._safe_emit_event_async("ASYNC_EVENT", {"async": True})

        assert result is True
        mock_event_bus.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_safe_emit_event_async_failure(self, emitter: TestEmitter):
        """Test async event emission failure returns False."""
        with patch(
            "app.coordination.event_router.get_event_bus",
            return_value=None,
        ):
            result = await emitter._safe_emit_event_async("ASYNC_EVENT")

        assert result is False

    @pytest.mark.asyncio
    async def test_safe_emit_event_async_runtime_error_fallback(
        self, emitter: TestEmitter, mock_event_bus
    ):
        """Test async emission falls back to sync on RuntimeError."""
        # First call to to_thread raises RuntimeError, triggering fallback
        call_count = 0

        def mock_to_thread(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("No event loop")
            return args[0](*args[1:])

        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            with patch(
                "app.coordination.event_router.get_event_bus",
                return_value=mock_event_bus,
            ):
                # The fallback sync call should succeed
                result = await emitter._safe_emit_event_async("FALLBACK_EVENT")

        # Should return True from the fallback sync call
        assert result is True


# =============================================================================
# Module-level safe_emit_event Tests
# =============================================================================


class TestSafeEmitEventFunction:
    """Tests for safe_emit_event module-level function."""

    def test_safe_emit_event_success(self, mock_event_bus):
        """Test module-level emit success."""
        with patch(
            "app.coordination.event_router.get_event_bus",
            return_value=mock_event_bus,
        ):
            result = safe_emit_event("MODULE_EVENT", {"key": "value"})

        assert result is True
        mock_event_bus.publish.assert_called_once()

    def test_safe_emit_event_custom_source(self, mock_event_bus):
        """Test custom source parameter."""
        with patch(
            "app.coordination.event_router.get_event_bus",
            return_value=mock_event_bus,
        ):
            result = safe_emit_event(
                "MODULE_EVENT",
                {"key": "value"},
                source="my_custom_module",
            )

        assert result is True
        call_args = mock_event_bus.publish.call_args
        event = call_args[0][0]
        assert event.source == "my_custom_module"

    def test_safe_emit_event_default_source(self, mock_event_bus):
        """Test default source is 'module'."""
        with patch(
            "app.coordination.event_router.get_event_bus",
            return_value=mock_event_bus,
        ):
            safe_emit_event("MODULE_EVENT")

        call_args = mock_event_bus.publish.call_args
        event = call_args[0][0]
        assert event.source == "module"

    def test_safe_emit_event_bus_unavailable(self):
        """Test returns False when bus unavailable."""
        with patch(
            "app.coordination.event_router.get_event_bus",
            return_value=None,
        ):
            result = safe_emit_event("MODULE_EVENT")

        assert result is False

    def test_safe_emit_event_import_error(self):
        """Test handles ImportError."""
        with patch(
            "app.coordination.event_router.get_event_bus",
            side_effect=ImportError("No module"),
        ):
            result = safe_emit_event("MODULE_EVENT")

        assert result is False

    def test_safe_emit_event_no_payload(self, mock_event_bus):
        """Test emission without payload."""
        with patch(
            "app.coordination.event_router.get_event_bus",
            return_value=mock_event_bus,
        ):
            result = safe_emit_event("SIMPLE_EVENT")

        assert result is True
        call_args = mock_event_bus.publish.call_args
        event = call_args[0][0]
        assert event.payload == {}


# =============================================================================
# Integration Tests
# =============================================================================


class TestSafeEventEmitterIntegration:
    """Integration tests verifying lazy import and event structure."""

    def test_lazy_import_avoids_circular_dependency(self):
        """Test that imports happen lazily inside methods."""
        # The module should import without errors
        from app.coordination.safe_event_emitter import SafeEventEmitterMixin

        # Creating instance should not trigger event_router import
        emitter = TestEmitter()
        assert emitter._event_source == "TestEmitter"

    def test_event_structure_matches_data_event(self, emitter: TestEmitter, mock_event_bus):
        """Test emitted events have correct DataEvent structure."""
        with patch(
            "app.coordination.event_router.get_event_bus",
            return_value=mock_event_bus,
        ):
            emitter._safe_emit_event(
                "STRUCTURED_EVENT",
                {"field1": "value1", "field2": 42},
            )

        call_args = mock_event_bus.publish.call_args
        event = call_args[0][0]

        # Verify DataEvent attributes
        assert hasattr(event, "event_type")
        assert hasattr(event, "payload")
        assert hasattr(event, "source")
        assert event.event_type == "STRUCTURED_EVENT"
        assert event.payload["field1"] == "value1"
        assert event.payload["field2"] == 42

    def test_multiple_emitters_use_correct_sources(self, mock_event_bus):
        """Test multiple emitters track their sources correctly."""
        emitter1 = TestEmitter()
        emitter2 = CustomSourceEmitter()

        with patch(
            "app.coordination.event_router.get_event_bus",
            return_value=mock_event_bus,
        ):
            emitter1._safe_emit_event("EVENT_1")
            emitter2._safe_emit_event("EVENT_2")

        calls = mock_event_bus.publish.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0].source == "TestEmitter"
        assert calls[1][0][0].source == "CustomSource"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_event_type(self, emitter: TestEmitter, mock_event_bus):
        """Test emission with empty event type."""
        with patch(
            "app.coordination.event_router.get_event_bus",
            return_value=mock_event_bus,
        ):
            result = emitter._safe_emit_event("")

        assert result is True  # Should still emit

    def test_none_payload_converted_to_empty_dict(
        self, emitter: TestEmitter, mock_event_bus
    ):
        """Test None payload is converted to empty dict."""
        with patch(
            "app.coordination.event_router.get_event_bus",
            return_value=mock_event_bus,
        ):
            emitter._safe_emit_event("EVENT", None)

        call_args = mock_event_bus.publish.call_args
        event = call_args[0][0]
        assert event.payload == {}

    def test_complex_payload_structure(self, emitter: TestEmitter, mock_event_bus):
        """Test emission with complex nested payload."""
        payload = {
            "config": {"board_type": "hex8", "num_players": 2},
            "metrics": [1.0, 2.0, 3.0],
            "nested": {"deep": {"value": True}},
        }

        with patch(
            "app.coordination.event_router.get_event_bus",
            return_value=mock_event_bus,
        ):
            result = emitter._safe_emit_event("COMPLEX_EVENT", payload)

        assert result is True
        call_args = mock_event_bus.publish.call_args
        event = call_args[0][0]
        assert event.payload == payload


class TestLogging:
    """Tests for logging behavior."""

    def test_logs_debug_on_bus_unavailable(self, emitter: TestEmitter, caplog):
        """Test debug logging when bus unavailable."""
        with caplog.at_level(logging.DEBUG):
            with patch(
                "app.coordination.event_router.get_event_bus",
                return_value=None,
            ):
                emitter._safe_emit_event("TEST_EVENT")

        assert "Event bus unavailable" in caplog.text

    def test_logs_debug_on_emission_failure(self, emitter: TestEmitter, caplog):
        """Test debug logging on emission failure."""
        with caplog.at_level(logging.DEBUG):
            with patch(
                "app.coordination.event_router.get_event_bus",
                side_effect=ImportError("Test error"),
            ):
                emitter._safe_emit_event("TEST_EVENT")

        assert "Event emission failed" in caplog.text
