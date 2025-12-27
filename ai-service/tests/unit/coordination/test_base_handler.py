"""Tests for base_handler module (December 2025).

Tests the BaseEventHandler, BaseSingletonHandler, MultiEventHandler, and HandlerStats
classes that provide common patterns for event-driven handlers.
"""

from __future__ import annotations

import asyncio
import pytest
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch


class TestHandlerStats:
    """Tests for HandlerStats dataclass."""

    def test_default_values(self):
        """Test HandlerStats initializes with correct defaults."""
        from app.coordination.base_handler import HandlerStats

        stats = HandlerStats()
        assert stats.subscribed is False
        assert stats.events_processed == 0
        assert stats.success_count == 0
        assert stats.error_count == 0
        assert stats.last_event_time == 0.0
        assert stats.last_error == ""
        assert stats.custom_stats == {}

    def test_success_rate_zero_events(self):
        """Test success_rate is 1.0 when no events processed."""
        from app.coordination.base_handler import HandlerStats

        stats = HandlerStats()
        assert stats.success_rate == 1.0

    def test_success_rate_calculation(self):
        """Test success_rate calculates correctly."""
        from app.coordination.base_handler import HandlerStats

        # Use canonical field name (errors_count, not error_count)
        # error_count is a read-only property alias for backward compatibility
        stats = HandlerStats(events_processed=10, success_count=8, errors_count=2)
        assert stats.success_rate == 0.8
        # Verify backward-compat alias works for reading
        assert stats.error_count == 2

    def test_to_dict(self):
        """Test HandlerStats serializes to dict."""
        from app.coordination.base_handler import HandlerStats

        # Use canonical field name (errors_count, not error_count)
        stats = HandlerStats(
            subscribed=True,
            events_processed=5,
            success_count=4,
            errors_count=1,
        )
        result = stats.to_dict()

        assert result["subscribed"] is True
        assert result["events_processed"] == 5
        assert result["success_count"] == 4
        # to_dict uses canonical key (errors_count, not error_count)
        assert result["errors_count"] == 1
        assert result["success_rate"] == 0.8

    def test_custom_stats_in_to_dict(self):
        """Test custom_stats are included in to_dict output."""
        from app.coordination.base_handler import HandlerStats

        stats = HandlerStats()
        stats.custom_stats["my_metric"] = 42

        result = stats.to_dict()
        assert result["my_metric"] == 42


class TestBaseEventHandler:
    """Tests for BaseEventHandler abstract class."""

    def test_initialization(self):
        """Test BaseEventHandler initializes correctly."""
        from app.coordination.base_handler import BaseEventHandler

        class TestHandler(BaseEventHandler):
            def _do_subscribe(self) -> bool:
                self._subscribed = True
                return True

            async def _handle_event(self, event: Any) -> None:
                pass

        handler = TestHandler("TestHandler")
        assert handler.handler_name == "TestHandler"
        assert handler.is_subscribed is False
        assert handler.emit_metrics is True
        assert handler.uptime_seconds >= 0

    def test_record_success(self):
        """Test _record_success updates stats correctly."""
        from app.coordination.base_handler import BaseEventHandler

        class TestHandler(BaseEventHandler):
            def _do_subscribe(self) -> bool:
                return True

            async def _handle_event(self, event: Any) -> None:
                pass

        handler = TestHandler("TestHandler")
        handler._record_success()

        stats = handler.get_stats()
        assert stats["events_processed"] == 1
        assert stats["success_count"] == 1
        # Use canonical key (errors_count, not error_count)
        assert stats["errors_count"] == 0

    def test_record_error(self):
        """Test _record_error updates stats correctly."""
        from app.coordination.base_handler import BaseEventHandler

        class TestHandler(BaseEventHandler):
            def _do_subscribe(self) -> bool:
                return True

            async def _handle_event(self, event: Any) -> None:
                pass

        handler = TestHandler("TestHandler")
        handler._record_error("Test error")

        stats = handler.get_stats()
        assert stats["events_processed"] == 1
        assert stats["success_count"] == 0
        # Use canonical key (errors_count, not error_count)
        assert stats["errors_count"] == 1
        assert stats["last_error"] == "Test error"

    def test_subscribe_success(self):
        """Test subscribe() calls _do_subscribe() and updates state."""
        from app.coordination.base_handler import BaseEventHandler

        class TestHandler(BaseEventHandler):
            def _do_subscribe(self) -> bool:
                self._subscribed = True
                return True

            async def _handle_event(self, event: Any) -> None:
                pass

        handler = TestHandler("TestHandler")
        result = handler.subscribe()

        assert result is True
        assert handler.is_subscribed is True
        assert handler.get_stats()["subscribed"] is True

    def test_subscribe_already_subscribed(self):
        """Test subscribe() returns True if already subscribed."""
        from app.coordination.base_handler import BaseEventHandler

        class TestHandler(BaseEventHandler):
            def _do_subscribe(self) -> bool:
                self._subscribed = True
                return True

            async def _handle_event(self, event: Any) -> None:
                pass

        handler = TestHandler("TestHandler")
        handler._subscribed = True

        result = handler.subscribe()
        assert result is True

    def test_unsubscribe(self):
        """Test unsubscribe() updates state."""
        from app.coordination.base_handler import BaseEventHandler

        class TestHandler(BaseEventHandler):
            def _do_subscribe(self) -> bool:
                self._subscribed = True
                return True

            async def _handle_event(self, event: Any) -> None:
                pass

        handler = TestHandler("TestHandler")
        handler._subscribed = True
        handler._stats.subscribed = True

        handler.unsubscribe()

        assert handler.is_subscribed is False
        assert handler.get_stats()["subscribed"] is False

    def test_get_payload_from_event_object(self):
        """Test _get_payload extracts payload from Event object."""
        from app.coordination.base_handler import BaseEventHandler

        class TestHandler(BaseEventHandler):
            def _do_subscribe(self) -> bool:
                return True

            async def _handle_event(self, event: Any) -> None:
                pass

        class MockEvent:
            payload = {"key": "value"}

        handler = TestHandler("TestHandler")
        payload = handler._get_payload(MockEvent())
        assert payload == {"key": "value"}

    def test_get_payload_from_dict(self):
        """Test _get_payload returns dict events as-is."""
        from app.coordination.base_handler import BaseEventHandler

        class TestHandler(BaseEventHandler):
            def _do_subscribe(self) -> bool:
                return True

            async def _handle_event(self, event: Any) -> None:
                pass

        handler = TestHandler("TestHandler")
        payload = handler._get_payload({"key": "value"})
        assert payload == {"key": "value"}

    def test_add_custom_stat(self):
        """Test add_custom_stat adds to stats."""
        from app.coordination.base_handler import BaseEventHandler

        class TestHandler(BaseEventHandler):
            def _do_subscribe(self) -> bool:
                return True

            async def _handle_event(self, event: Any) -> None:
                pass

        handler = TestHandler("TestHandler")
        handler.add_custom_stat("my_metric", 100)

        stats = handler.get_stats()
        assert stats["my_metric"] == 100

    def test_reset(self):
        """Test reset() clears stats."""
        from app.coordination.base_handler import BaseEventHandler

        class TestHandler(BaseEventHandler):
            def _do_subscribe(self) -> bool:
                return True

            async def _handle_event(self, event: Any) -> None:
                pass

        handler = TestHandler("TestHandler")
        handler._record_success()
        handler._record_success()
        handler._subscribed = True

        handler.reset()

        stats = handler.get_stats()
        assert stats["events_processed"] == 0
        assert stats["success_count"] == 0
        # Subscription state should be preserved
        assert stats["subscribed"] is True


class TestBaseSingletonHandler:
    """Tests for BaseSingletonHandler singleton pattern."""

    def test_get_instance_creates_singleton(self):
        """Test get_instance creates a singleton."""
        from app.coordination.base_handler import BaseSingletonHandler

        class TestSingleton(BaseSingletonHandler):
            _instance: ClassVar["TestSingleton | None"] = None

            def __init__(self) -> None:
                super().__init__("TestSingleton")

            def _do_subscribe(self) -> bool:
                return True

            async def _handle_event(self, event: Any) -> None:
                pass

        TestSingleton._instance = None  # Reset

        instance1 = TestSingleton.get_instance()
        instance2 = TestSingleton.get_instance()

        assert instance1 is instance2

        TestSingleton.reset_singleton()

    def test_has_instance(self):
        """Test has_instance returns correct state."""
        from app.coordination.base_handler import BaseSingletonHandler

        class TestSingleton(BaseSingletonHandler):
            _instance: ClassVar["TestSingleton | None"] = None

            def __init__(self) -> None:
                super().__init__("TestSingleton")

            def _do_subscribe(self) -> bool:
                return True

            async def _handle_event(self, event: Any) -> None:
                pass

        TestSingleton._instance = None

        assert TestSingleton.has_instance() is False

        TestSingleton.get_instance()
        assert TestSingleton.has_instance() is True

        TestSingleton.reset_singleton()
        assert TestSingleton.has_instance() is False

    def test_reset_singleton(self):
        """Test reset_singleton clears the instance."""
        from app.coordination.base_handler import BaseSingletonHandler

        class TestSingleton(BaseSingletonHandler):
            _instance: ClassVar["TestSingleton | None"] = None

            def __init__(self) -> None:
                super().__init__("TestSingleton")

            def _do_subscribe(self) -> bool:
                self._subscribed = True
                return True

            async def _handle_event(self, event: Any) -> None:
                pass

        TestSingleton._instance = None

        instance = TestSingleton.get_instance()
        instance._subscribed = True

        TestSingleton.reset_singleton()

        assert TestSingleton._instance is None


class TestMultiEventHandler:
    """Tests for MultiEventHandler multi-event routing."""

    def test_event_handlers_registry(self):
        """Test _event_handlers dict is initialized."""
        from app.coordination.base_handler import MultiEventHandler

        class TestMulti(MultiEventHandler):
            def _do_subscribe(self) -> bool:
                return True

        handler = TestMulti("TestMulti")
        assert handler._event_handlers == {}

    @pytest.mark.asyncio
    async def test_handle_event_routes_to_handler(self):
        """Test _handle_event routes to correct handler."""
        from app.coordination.base_handler import MultiEventHandler

        handled_events = []

        class TestMulti(MultiEventHandler):
            def __init__(self):
                super().__init__("TestMulti")
                self._event_handlers = {
                    "TYPE_A": self._on_type_a,
                    "TYPE_B": self._on_type_b,
                }

            def _do_subscribe(self) -> bool:
                return True

            async def _on_type_a(self, event):
                handled_events.append(("A", event))

            async def _on_type_b(self, event):
                handled_events.append(("B", event))

        handler = TestMulti()

        await handler._handle_event({"type": "TYPE_A", "data": 1})
        await handler._handle_event({"type": "TYPE_B", "data": 2})

        assert len(handled_events) == 2
        assert handled_events[0] == ("A", {"type": "TYPE_A", "data": 1})
        assert handled_events[1] == ("B", {"type": "TYPE_B", "data": 2})


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_handler_stats(self):
        """Test create_handler_stats factory function."""
        from app.coordination.base_handler import create_handler_stats

        stats = create_handler_stats(my_custom="value", count=42)

        assert stats.custom_stats["my_custom"] == "value"
        assert stats.custom_stats["count"] == 42

    def test_safe_subscribe_success(self):
        """Test safe_subscribe returns True on success."""
        from app.coordination.base_handler import BaseEventHandler, safe_subscribe

        class TestHandler(BaseEventHandler):
            def _do_subscribe(self) -> bool:
                self._subscribed = True
                return True

            async def _handle_event(self, event: Any) -> None:
                pass

        handler = TestHandler("TestHandler")
        result = safe_subscribe(handler)

        assert result is True

    def test_safe_subscribe_fallback_on_error(self):
        """Test safe_subscribe returns fallback on error."""
        from app.coordination.base_handler import BaseEventHandler, safe_subscribe

        class TestHandler(BaseEventHandler):
            def _do_subscribe(self) -> bool:
                raise RuntimeError("Test error")

            async def _handle_event(self, event: Any) -> None:
                pass

        handler = TestHandler("TestHandler")
        result = safe_subscribe(handler, fallback=False)

        assert result is False
