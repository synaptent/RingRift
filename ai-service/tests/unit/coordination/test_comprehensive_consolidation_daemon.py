"""Focused tests for ComprehensiveConsolidationDaemon router subscriptions."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.coordination.comprehensive_consolidation_daemon import (
    ComprehensiveConsolidationConfig,
    ComprehensiveConsolidationDaemon,
)


@pytest.fixture
def daemon(tmp_path):
    """Create a comprehensive consolidation daemon without starting it."""
    config = ComprehensiveConsolidationConfig(
        data_dir=tmp_path / "games",
        canonical_dir=tmp_path / "games",
        tracking_db_path=tmp_path / "tracking.db",
    )
    return ComprehensiveConsolidationDaemon(config=config)


class TestComprehensiveConsolidationEventSubscriptions:
    """Test router helper usage for subscription lifecycle."""

    @pytest.mark.asyncio
    async def test_subscribe_to_events_uses_router_helper(self, daemon):
        """Subscription should go through unified router helpers."""
        fake_event_type = SimpleNamespace(CONSOLIDATION_REQUESTED=SimpleNamespace(value="consolidation_requested"))

        with patch("app.distributed.data_events.DataEventType", fake_event_type), \
             patch("app.coordination.event_router.subscribe") as mock_subscribe:
            await daemon._subscribe_to_events()

        mock_subscribe.assert_called_once()
        event_type, callback = mock_subscribe.call_args.args
        assert event_type.value == "consolidation_requested"
        assert callback == daemon._on_consolidation_requested
        assert daemon._subscribed is True

    @pytest.mark.asyncio
    async def test_unsubscribe_from_events_uses_router_helper(self, daemon):
        """Unsubscription should go through unified router helpers."""
        daemon._subscribed = True
        fake_event_type = SimpleNamespace(CONSOLIDATION_REQUESTED=SimpleNamespace(value="consolidation_requested"))

        with patch("app.distributed.data_events.DataEventType", fake_event_type), \
             patch("app.coordination.event_router.unsubscribe") as mock_unsubscribe:
            await daemon._unsubscribe_from_events()

        mock_unsubscribe.assert_called_once()
        event_type, callback = mock_unsubscribe.call_args.args
        assert event_type.value == "consolidation_requested"
        assert callback == daemon._on_consolidation_requested
        assert daemon._subscribed is False
