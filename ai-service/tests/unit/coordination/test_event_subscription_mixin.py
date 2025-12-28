"""Tests for EventSubscribingDaemonMixin (December 2025).

NOTE: This test file references a module that was consolidated into handler_base.py.
Tests are skipped - use test_handler_base.py instead for the unified implementation.
"""

import pytest

# Skip all tests - module was consolidated
pytestmark = pytest.mark.skip(
    reason="app.coordination.event_subscription_mixin was consolidated into "
    "handler_base.py. Use test_handler_base.py for event subscription tests."
)

from unittest.mock import MagicMock, patch

# These imports will fail - kept for documentation
try:
    from app.coordination.event_subscription_mixin import (
        EventSubscribingDaemonMixin,
        create_event_subscribing_daemon,
    )
except ImportError:
    pass  # Expected - consolidated into handler_base.py


class MockDaemon:
    """Mock base daemon for testing."""

    def __init__(self, name: str = "mock_daemon"):
        self.name = name
        self._running = False

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False


class TestEventSubscribingDaemonMixin:
    """Test cases for EventSubscribingDaemonMixin."""

    def test_init_event_subscriptions(self):
        """Should initialize subscription tracking attributes."""

        class TestDaemon(MockDaemon, EventSubscribingDaemonMixin):
            def __init__(self):
                super().__init__("test")
                self._init_event_subscriptions()

        daemon = TestDaemon()
        assert daemon._event_subscriptions == {}
        assert daemon._event_subscribed is False
        assert daemon._event_router is None

    def test_get_daemon_name_from_name_attr(self):
        """Should get daemon name from name attribute."""

        class TestDaemon(MockDaemon, EventSubscribingDaemonMixin):
            def __init__(self):
                super().__init__("my_daemon")
                self._init_event_subscriptions()

        daemon = TestDaemon()
        assert daemon._get_daemon_name() == "my_daemon"

    def test_get_daemon_name_fallback_to_class_name(self):
        """Should fall back to class name if no name attribute."""

        class NoNameDaemon(EventSubscribingDaemonMixin):
            def __init__(self):
                self._init_event_subscriptions()

        daemon = NoNameDaemon()
        assert daemon._get_daemon_name() == "NoNameDaemon"

    def test_is_event_subscribed_property(self):
        """Should return subscription status via property."""

        class TestDaemon(MockDaemon, EventSubscribingDaemonMixin):
            def __init__(self):
                super().__init__("test")
                self._init_event_subscriptions()

        daemon = TestDaemon()
        assert daemon.is_event_subscribed is False

        daemon._event_subscribed = True
        assert daemon.is_event_subscribed is True

    def test_active_subscription_count_property(self):
        """Should return count of active subscriptions."""

        class TestDaemon(MockDaemon, EventSubscribingDaemonMixin):
            def __init__(self):
                super().__init__("test")
                self._init_event_subscriptions()

        daemon = TestDaemon()
        assert daemon.active_subscription_count == 0

        daemon._event_subscriptions["event1"] = lambda: None
        daemon._event_subscriptions["event2"] = lambda: None
        assert daemon.active_subscription_count == 2

    def test_get_subscription_status(self):
        """Should return detailed subscription status dict."""

        class TestDaemon(MockDaemon, EventSubscribingDaemonMixin):
            def __init__(self):
                super().__init__("test")
                self._init_event_subscriptions()

        daemon = TestDaemon()
        status = daemon.get_subscription_status()

        assert status["subscribed"] is False
        assert status["subscription_count"] == 0
        assert status["subscribed_events"] == []
        assert status["router_available"] is False

    def test_get_event_subscriptions_default_empty(self):
        """Should return empty dict by default."""

        class TestDaemon(MockDaemon, EventSubscribingDaemonMixin):
            def __init__(self):
                super().__init__("test")
                self._init_event_subscriptions()

        daemon = TestDaemon()
        assert daemon._get_event_subscriptions() == {}

    def test_get_event_subscriptions_override(self):
        """Should allow override to specify subscriptions."""

        class TestDaemon(MockDaemon, EventSubscribingDaemonMixin):
            def __init__(self):
                super().__init__("test")
                self._init_event_subscriptions()

            def _get_event_subscriptions(self):
                return {
                    "training_completed": self._on_training,
                    "sync_completed": self._on_sync,
                }

            def _on_training(self, event):
                pass

            def _on_sync(self, event):
                pass

        daemon = TestDaemon()
        subs = daemon._get_event_subscriptions()
        assert "training_completed" in subs
        assert "sync_completed" in subs

    def test_get_conditional_subscriptions_default_empty(self):
        """Should return empty dict for conditional subscriptions by default."""

        class TestDaemon(MockDaemon, EventSubscribingDaemonMixin):
            def __init__(self):
                super().__init__("test")
                self._init_event_subscriptions()

        daemon = TestDaemon()
        assert daemon._get_conditional_subscriptions() == {}

    def test_subscribe_all_events_success(self):
        """Should subscribe to all events successfully."""
        mock_router = MagicMock()
        mock_unsub = MagicMock()
        mock_router.subscribe.return_value = mock_unsub

        class TestDaemon(MockDaemon, EventSubscribingDaemonMixin):
            def __init__(self):
                super().__init__("test")
                self._init_event_subscriptions()

            def _get_event_subscriptions(self):
                return {"event1": self._handler}

            def _handler(self, event):
                pass

        daemon = TestDaemon()

        with patch.object(
            daemon.__class__,
            "_subscribe_all_events",
            wraps=daemon._subscribe_all_events,
        ):
            with patch(
                "app.coordination.event_router.get_router",
                return_value=mock_router,
            ):
                result = daemon._subscribe_all_events()

        assert result is True
        assert daemon._event_subscribed is True
        assert daemon._event_router is mock_router
        assert "event1" in daemon._event_subscriptions
        mock_router.subscribe.assert_called_once()

    def test_subscribe_all_events_already_subscribed(self):
        """Should return True immediately if already subscribed."""

        class TestDaemon(MockDaemon, EventSubscribingDaemonMixin):
            def __init__(self):
                super().__init__("test")
                self._init_event_subscriptions()
                self._event_subscribed = True

        daemon = TestDaemon()
        result = daemon._subscribe_all_events()

        assert result is True
        # Should not attempt to get router since already subscribed

    def test_subscribe_all_events_import_error(self):
        """Should return False on ImportError."""

        class TestDaemon(MockDaemon, EventSubscribingDaemonMixin):
            def __init__(self):
                super().__init__("test")
                self._init_event_subscriptions()

            def _get_event_subscriptions(self):
                return {"test": lambda e: None}

        daemon = TestDaemon()

        # We need to test when the internal import fails
        # Since the import happens inside the method, we can't easily patch it
        # Instead, test that when router is not available it returns False
        with patch(
            "app.coordination.event_router.get_router",
            side_effect=ImportError("No module"),
        ):
            result = daemon._subscribe_all_events()

        assert result is False
        assert daemon._event_subscribed is False

    def test_subscribe_single_event(self):
        """Should subscribe to a single event."""
        mock_router = MagicMock()
        mock_unsub = MagicMock()
        mock_router.subscribe.return_value = mock_unsub

        class TestDaemon(MockDaemon, EventSubscribingDaemonMixin):
            def __init__(self):
                super().__init__("test")
                self._init_event_subscriptions()

        daemon = TestDaemon()

        with patch(
            "app.coordination.event_router.get_router",
            return_value=mock_router,
        ):
            result = daemon._subscribe_single_event("my_event", lambda e: None)

        assert result is True
        assert "my_event" in daemon._event_subscriptions
        mock_router.subscribe.assert_called_once()

    def test_unsubscribe_single_event_not_subscribed(self):
        """Should return True if not subscribed to event."""

        class TestDaemon(MockDaemon, EventSubscribingDaemonMixin):
            def __init__(self):
                super().__init__("test")
                self._init_event_subscriptions()

        daemon = TestDaemon()
        result = daemon._unsubscribe_single_event("nonexistent")
        assert result is True

    def test_unsubscribe_single_event_success(self):
        """Should unsubscribe from event and call unsub callback."""
        unsub_called = []

        class TestDaemon(MockDaemon, EventSubscribingDaemonMixin):
            def __init__(self):
                super().__init__("test")
                self._init_event_subscriptions()

        daemon = TestDaemon()
        daemon._event_subscriptions["my_event"] = lambda: unsub_called.append(True)

        result = daemon._unsubscribe_single_event("my_event")

        assert result is True
        assert len(unsub_called) == 1
        assert "my_event" not in daemon._event_subscriptions

    def test_unsubscribe_all_events(self):
        """Should unsubscribe from all events."""
        unsub_calls = []

        class TestDaemon(MockDaemon, EventSubscribingDaemonMixin):
            def __init__(self):
                super().__init__("test")
                self._init_event_subscriptions()

        daemon = TestDaemon()
        daemon._event_subscriptions["event1"] = lambda: unsub_calls.append("event1")
        daemon._event_subscriptions["event2"] = lambda: unsub_calls.append("event2")
        daemon._event_subscribed = True

        daemon._unsubscribe_all_events()

        assert len(unsub_calls) == 2
        assert daemon._event_subscriptions == {}
        assert daemon._event_subscribed is False
        assert daemon._event_router is None

    def test_unsubscribe_all_events_handles_errors(self):
        """Should continue unsubscribing even if some fail."""

        def failing_unsub():
            raise RuntimeError("Unsub failed")

        class TestDaemon(MockDaemon, EventSubscribingDaemonMixin):
            def __init__(self):
                super().__init__("test")
                self._init_event_subscriptions()

        daemon = TestDaemon()
        daemon._event_subscriptions["event1"] = failing_unsub
        daemon._event_subscriptions["event2"] = lambda: None
        daemon._event_subscribed = True

        # Should not raise
        daemon._unsubscribe_all_events()

        assert daemon._event_subscriptions == {}
        assert daemon._event_subscribed is False


class TestCreateEventSubscribingDaemon:
    """Test cases for create_event_subscribing_daemon factory."""

    def test_creates_enhanced_class(self):
        """Should create a class with mixin functionality."""
        handler = MagicMock()
        EnhancedDaemon = create_event_subscribing_daemon(
            MockDaemon,
            {"test_event": handler},
            name="enhanced",
        )

        daemon = EnhancedDaemon("base_name")
        assert hasattr(daemon, "_event_subscriptions")
        assert hasattr(daemon, "_event_subscribed")
        assert daemon._get_event_subscriptions() == {"test_event": handler}

    def test_enhanced_class_inherits_base(self):
        """Should inherit from base class."""
        EnhancedDaemon = create_event_subscribing_daemon(
            MockDaemon,
            {},
        )

        daemon = EnhancedDaemon("test")
        assert isinstance(daemon, MockDaemon)
        assert hasattr(daemon, "start")
        assert hasattr(daemon, "stop")

    def test_enhanced_class_uses_custom_name(self):
        """Should use custom name if provided."""
        EnhancedDaemon = create_event_subscribing_daemon(
            MockDaemon,
            {},
            name="custom_name",
        )

        daemon = EnhancedDaemon("base")
        # _name is set, which _get_daemon_name checks via hasattr
        assert hasattr(daemon, "_name")
        assert daemon._name == "custom_name"


class TestEventSubscribingDaemonMixinIntegration:
    """Integration tests with real event router."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Test complete subscribe/unsubscribe lifecycle."""
        events_received = []

        class TestDaemon(MockDaemon, EventSubscribingDaemonMixin):
            def __init__(self):
                super().__init__("integration_test")
                self._init_event_subscriptions()

            def _get_event_subscriptions(self):
                return {"test_event": self._on_test}

            def _on_test(self, event):
                events_received.append(event)

            async def start(self):
                await super().start()
                self._subscribe_all_events()

            async def stop(self):
                self._unsubscribe_all_events()
                await super().stop()

        daemon = TestDaemon()

        # Start may fail if event router not available - that's OK
        await daemon.start()
        assert daemon._running

        # Stop should always work
        await daemon.stop()
        assert not daemon._running
        assert daemon._event_subscriptions == {}
        assert daemon._event_subscribed is False
