"""Focused tests for NodeDataAgent router subscriptions."""

from unittest.mock import patch

import pytest

from app.coordination.node_data_agent import NodeDataAgent, NodeDataAgentConfig


@pytest.fixture
def agent(tmp_path):
    """Create a node data agent without starting it."""
    config = NodeDataAgentConfig(
        cache_dir=tmp_path / "cache",
        auto_report_inventory=False,
    )
    return NodeDataAgent(config=config)


class TestNodeDataAgentEventSubscriptions:
    """Test router helper usage for agent subscription lifecycle."""

    @pytest.mark.asyncio
    async def test_subscribe_to_events_uses_router_helper(self, agent):
        """Subscription should go through unified router helpers."""
        with patch("app.coordination.event_router.subscribe") as mock_subscribe:
            await agent._subscribe_to_events()

        assert mock_subscribe.call_count == 2
        subscribed = [call.args[0] for call in mock_subscribe.call_args_list]
        assert "DATA_CATALOG_UPDATED" in subscribed
        assert "DATA_FETCH_REQUESTED" in subscribed

    @pytest.mark.asyncio
    async def test_unsubscribe_from_events_uses_router_helper(self, agent):
        """Unsubscription should go through unified router helpers."""
        with patch("app.coordination.event_router.unsubscribe") as mock_unsubscribe:
            await agent._unsubscribe_from_events()

        assert mock_unsubscribe.call_count == 2
        unsubscribed = [call.args[0] for call in mock_unsubscribe.call_args_list]
        assert "DATA_CATALOG_UPDATED" in unsubscribed
        assert "DATA_FETCH_REQUESTED" in unsubscribed
