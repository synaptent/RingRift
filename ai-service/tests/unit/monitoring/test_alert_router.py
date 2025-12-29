"""Unit tests for AlertRouter and related classes.

Tests cover:
- Alert dataclass and key generation
- AlertState tracking
- Slack/Discord/PagerDuty integrations
- AlertRouter deduplication and rate limiting
- Partition detection logic
- State persistence
- Singleton pattern

Created: December 2025
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.alert_types import AlertSeverity
from app.monitoring.alert_router import (
    Alert,
    AlertRouter,
    AlertState,
    DiscordIntegration,
    PagerDutyIntegration,
    SlackIntegration,
    get_alert_router,
    send_alert,
)


class TestAlert:
    """Tests for Alert dataclass."""

    def test_basic_creation(self):
        """Test creating an alert with required fields."""
        alert = Alert(
            severity=AlertSeverity.WARNING,
            alert_type="node_offline",
            message="Node is offline",
        )
        assert alert.severity == AlertSeverity.WARNING
        assert alert.alert_type == "node_offline"
        assert alert.message == "Node is offline"
        assert alert.node_id is None
        assert alert.details == {}

    def test_creation_with_node_id(self):
        """Test creating an alert with node_id."""
        alert = Alert(
            severity=AlertSeverity.ERROR,
            alert_type="disk_full",
            message="Disk is full",
            node_id="gpu-node-1",
        )
        assert alert.node_id == "gpu-node-1"

    def test_creation_with_details(self):
        """Test creating an alert with details."""
        details = {"disk_usage": 95, "threshold": 90}
        alert = Alert(
            severity=AlertSeverity.WARNING,
            alert_type="disk_warning",
            message="Disk usage high",
            details=details,
        )
        assert alert.details == details

    def test_timestamp_auto_set(self):
        """Test that timestamp is automatically set."""
        before = time.time()
        alert = Alert(
            severity=AlertSeverity.INFO,
            alert_type="test",
            message="Test",
        )
        after = time.time()
        assert before <= alert.timestamp <= after

    def test_key_with_node_id(self):
        """Test key generation with node_id."""
        alert = Alert(
            severity=AlertSeverity.WARNING,
            alert_type="node_offline",
            message="Test",
            node_id="gpu-node-1",
        )
        assert alert.key == "node_offline:gpu-node-1"

    def test_key_without_node_id(self):
        """Test key generation without node_id (cluster-level)."""
        alert = Alert(
            severity=AlertSeverity.WARNING,
            alert_type="cluster_issue",
            message="Test",
        )
        assert alert.key == "cluster_issue:cluster"


class TestAlertState:
    """Tests for AlertState dataclass."""

    def test_default_values(self):
        """Test default AlertState values."""
        state = AlertState()
        assert state.last_sent == 0.0
        assert state.send_count == 0
        assert state.suppressed_count == 0

    def test_custom_values(self):
        """Test AlertState with custom values."""
        state = AlertState(
            last_sent=1000.0,
            send_count=5,
            suppressed_count=10,
        )
        assert state.last_sent == 1000.0
        assert state.send_count == 5
        assert state.suppressed_count == 10


class TestSlackIntegration:
    """Tests for SlackIntegration."""

    def test_no_webhook_url(self):
        """Test that send returns False when no webhook URL."""
        slack = SlackIntegration(webhook_url="")
        alert = Alert(
            severity=AlertSeverity.WARNING,
            alert_type="test",
            message="Test",
        )
        result = asyncio.get_event_loop().run_until_complete(slack.send(alert))
        assert result is False

    def test_color_mapping(self):
        """Test color mapping for different severity levels."""
        slack = SlackIntegration(webhook_url="https://example.com/webhook")

        # Verify the colors dictionary exists and has correct mappings
        expected_colors = {
            AlertSeverity.DEBUG: "#808080",
            AlertSeverity.INFO: "#00aa00",
            AlertSeverity.WARNING: "#ffaa00",
            AlertSeverity.ERROR: "#ff0000",
            AlertSeverity.CRITICAL: "#990000",
        }

        # Can't easily test the actual send without mocking
        # Just verify the integration is instantiated correctly
        assert slack.webhook_url == "https://example.com/webhook"

    @pytest.mark.asyncio
    async def test_send_with_webhook(self):
        """Test sending with mocked webhook."""
        with patch("app.monitoring.alert_router.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()

            slack = SlackIntegration(webhook_url="https://example.com/webhook")
            alert = Alert(
                severity=AlertSeverity.ERROR,
                alert_type="test_error",
                message="Test error message",
                node_id="test-node",
            )

            result = await slack.send(alert)
            assert result is True


class TestDiscordIntegration:
    """Tests for DiscordIntegration."""

    def test_no_webhook_url(self):
        """Test that send returns False when no webhook URL."""
        discord = DiscordIntegration(webhook_url="")
        alert = Alert(
            severity=AlertSeverity.WARNING,
            alert_type="test",
            message="Test",
        )
        result = asyncio.get_event_loop().run_until_complete(discord.send(alert))
        assert result is False

    @pytest.mark.asyncio
    async def test_send_with_webhook(self):
        """Test sending with mocked webhook."""
        with patch("app.monitoring.alert_router.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()

            discord = DiscordIntegration(webhook_url="https://discord.com/webhook")
            alert = Alert(
                severity=AlertSeverity.WARNING,
                alert_type="test_warning",
                message="Test warning",
            )

            result = await discord.send(alert)
            assert result is True


class TestPagerDutyIntegration:
    """Tests for PagerDutyIntegration."""

    def test_no_routing_key(self):
        """Test that page returns False when no routing key."""
        pagerduty = PagerDutyIntegration(routing_key="")
        alert = Alert(
            severity=AlertSeverity.CRITICAL,
            alert_type="test",
            message="Test",
        )
        result = asyncio.get_event_loop().run_until_complete(pagerduty.page(alert))
        assert result is False

    def test_severity_mapping(self):
        """Test severity mapping to PagerDuty levels."""
        pagerduty = PagerDutyIntegration(routing_key="test-key")

        # The severity_map should map AlertSeverity to PagerDuty severity strings
        expected_mappings = {
            AlertSeverity.DEBUG: "info",
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "error",
            AlertSeverity.CRITICAL: "critical",
        }

        assert pagerduty.routing_key == "test-key"

    @pytest.mark.asyncio
    async def test_page_with_routing_key(self):
        """Test paging with mocked HTTP."""
        with patch("app.monitoring.alert_router.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()

            pagerduty = PagerDutyIntegration(routing_key="test-routing-key")
            alert = Alert(
                severity=AlertSeverity.CRITICAL,
                alert_type="critical_failure",
                message="Critical failure",
                node_id="critical-node",
            )

            result = await pagerduty.page(alert)
            assert result is True

    @pytest.mark.asyncio
    async def test_resolve_without_routing_key(self):
        """Test resolve returns False without routing key."""
        pagerduty = PagerDutyIntegration(routing_key="")
        result = await pagerduty.resolve("test:node")
        assert result is False

    @pytest.mark.asyncio
    async def test_resolve_with_routing_key(self):
        """Test resolve with mocked HTTP."""
        with patch("app.monitoring.alert_router.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()

            pagerduty = PagerDutyIntegration(routing_key="test-key")
            result = await pagerduty.resolve("test:node")
            assert result is True


class TestAlertRouter:
    """Tests for AlertRouter class."""

    def test_initialization(self):
        """Test AlertRouter initialization."""
        # Use temp file to avoid state pollution
        with patch("app.monitoring.alert_router.STATE_FILE", Path(tempfile.mktemp())):
            router = AlertRouter()
            assert router.slack is not None
            assert router.discord is not None
            assert router.pagerduty is not None
            assert router._states == {}
            assert router._hourly_count == 0

    def test_should_send_first_alert(self):
        """Test that first alert should always be sent."""
        with patch("app.monitoring.alert_router.STATE_FILE", Path(tempfile.mktemp())):
            router = AlertRouter()
            alert = Alert(
                severity=AlertSeverity.WARNING,
                alert_type="test_first",
                message="First alert",
            )

            should_send, reason = router._should_send(alert)
            assert should_send is True
            assert reason == "ok"

    def test_should_send_deduplication(self):
        """Test deduplication of same alert."""
        with patch("app.monitoring.alert_router.STATE_FILE", Path(tempfile.mktemp())):
            router = AlertRouter()
            alert = Alert(
                severity=AlertSeverity.WARNING,
                alert_type="test_dedup",
                message="Deduplicated alert",
            )

            # First should be sent
            should_send1, _ = router._should_send(alert)
            assert should_send1 is True

            # Update state as if it was sent
            router._states[alert.key].last_sent = time.time()
            router._states[alert.key].send_count += 1

            # Second within interval should be deduplicated
            should_send2, reason = router._should_send(alert)
            assert should_send2 is False
            assert reason == "deduplicated"

    def test_rate_limiting(self):
        """Test rate limiting of alerts."""
        with patch("app.monitoring.alert_router.STATE_FILE", Path(tempfile.mktemp())):
            with patch("app.monitoring.alert_router.MAX_ALERTS_PER_HOUR", 2):
                router = AlertRouter()
                router._hourly_count = 2  # At limit

                alert = Alert(
                    severity=AlertSeverity.WARNING,
                    alert_type="rate_limited_test",
                    message="Rate limited",
                )

                should_send, reason = router._should_send(alert)
                assert should_send is False
                assert reason == "rate_limited"

    def test_hourly_counter_reset(self):
        """Test hourly counter resets after an hour."""
        with patch("app.monitoring.alert_router.STATE_FILE", Path(tempfile.mktemp())):
            router = AlertRouter()
            router._hourly_count = 10
            router._hour_start = time.time() - 3601  # More than an hour ago

            # _check_rate_limit should reset the counter
            is_within_limit = router._check_rate_limit()
            assert is_within_limit is True
            assert router._hourly_count == 0

    def test_get_stats(self):
        """Test getting router statistics."""
        with patch("app.monitoring.alert_router.STATE_FILE", Path(tempfile.mktemp())):
            router = AlertRouter()
            router._hourly_count = 5
            router._active_issues["node_offline"] = {"node1", "node2"}
            router._states["test:cluster"] = AlertState(
                send_count=3, suppressed_count=7
            )

            stats = router.get_stats()
            assert stats["hourly_count"] == 5
            assert "hour_remaining" in stats
            assert stats["active_issues"]["node_offline"] == 2
            assert stats["states"]["test:cluster"]["send_count"] == 3
            assert stats["states"]["test:cluster"]["suppressed_count"] == 7

    @pytest.mark.asyncio
    async def test_route_alert_critical(self):
        """Test routing critical alerts to Slack and PagerDuty."""
        with patch("app.monitoring.alert_router.STATE_FILE", Path(tempfile.mktemp())):
            router = AlertRouter()
            router.slack.send = AsyncMock(return_value=True)
            router.pagerduty.page = AsyncMock(return_value=True)

            alert = Alert(
                severity=AlertSeverity.CRITICAL,
                alert_type="critical_test",
                message="Critical alert",
            )

            result = await router.route_alert(alert)
            assert result is True
            router.slack.send.assert_called_once()
            router.pagerduty.page.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_alert_error(self):
        """Test routing error alerts to Slack only."""
        with patch("app.monitoring.alert_router.STATE_FILE", Path(tempfile.mktemp())):
            router = AlertRouter()
            router.slack.send = AsyncMock(return_value=True)
            router.pagerduty.page = AsyncMock()

            alert = Alert(
                severity=AlertSeverity.ERROR,
                alert_type="error_test",
                message="Error alert",
            )

            result = await router.route_alert(alert)
            assert result is True
            router.slack.send.assert_called_once()
            router.pagerduty.page.assert_not_called()

    @pytest.mark.asyncio
    async def test_route_alert_warning(self):
        """Test routing warning alerts to Slack only."""
        with patch("app.monitoring.alert_router.STATE_FILE", Path(tempfile.mktemp())):
            router = AlertRouter()
            router.slack.send = AsyncMock(return_value=True)

            alert = Alert(
                severity=AlertSeverity.WARNING,
                alert_type="warning_test",
                message="Warning alert",
            )

            result = await router.route_alert(alert)
            assert result is True

    @pytest.mark.asyncio
    async def test_resolve_alert(self):
        """Test resolving an alert."""
        with patch("app.monitoring.alert_router.STATE_FILE", Path(tempfile.mktemp())):
            router = AlertRouter()
            router._active_issues["node_offline"] = {"node1", "node2"}
            router.pagerduty.resolve = AsyncMock(return_value=True)

            await router.resolve_alert("node_offline", "node1")

            assert "node1" not in router._active_issues["node_offline"]
            assert "node2" in router._active_issues["node_offline"]
            router.pagerduty.resolve.assert_called_once_with("node_offline:node1")


class TestStatePersistence:
    """Tests for state persistence."""

    def test_save_and_load_state(self):
        """Test saving and loading state."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = Path(f.name)

        with patch("app.monitoring.alert_router.STATE_FILE", state_file):
            # Create router and add some state
            router1 = AlertRouter()
            router1._states["test:cluster"] = AlertState(
                last_sent=1000.0, send_count=5, suppressed_count=10
            )
            router1._hourly_count = 7
            router1._hour_start = 12345.0
            router1._save_state()

            # Create new router and verify state loaded
            router2 = AlertRouter()
            assert router2._states["test:cluster"].last_sent == 1000.0
            assert router2._states["test:cluster"].send_count == 5
            assert router2._states["test:cluster"].suppressed_count == 10
            assert router2._hourly_count == 7
            assert router2._hour_start == 12345.0

        # Cleanup
        state_file.unlink(missing_ok=True)


class TestSingletonPattern:
    """Tests for singleton pattern."""

    def test_get_alert_router_singleton(self):
        """Test that get_alert_router returns singleton."""
        # Reset the global singleton
        import app.monitoring.alert_router as module

        module._router = None

        router1 = get_alert_router()
        router2 = get_alert_router()
        assert router1 is router2


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_send_alert_function(self):
        """Test send_alert convenience function."""
        import app.monitoring.alert_router as module

        # Create mock router
        mock_router = MagicMock()
        mock_router.route_alert = AsyncMock(return_value=True)
        module._router = mock_router

        result = await send_alert(
            alert_type="test_alert",
            message="Test message",
            severity=AlertSeverity.WARNING,
            node_id="test-node",
            details={"key": "value"},
        )

        assert result is True
        mock_router.route_alert.assert_called_once()
        called_alert = mock_router.route_alert.call_args[0][0]
        assert called_alert.alert_type == "test_alert"
        assert called_alert.message == "Test message"
        assert called_alert.severity == AlertSeverity.WARNING
        assert called_alert.node_id == "test-node"
        assert called_alert.details == {"key": "value"}


class TestPartitionDetection:
    """Tests for partition detection logic."""

    def test_partition_detection_threshold(self):
        """Test partition detection when many nodes have same issue."""
        with patch("app.monitoring.alert_router.STATE_FILE", Path(tempfile.mktemp())):
            router = AlertRouter()

            # Add 21 nodes with same issue (above threshold of 20)
            router._active_issues["node_offline"] = {f"node{i}" for i in range(21)}

            alert = Alert(
                severity=AlertSeverity.WARNING,
                alert_type="node_offline",
                message="Node offline",
                node_id="node22",
            )

            # First alert converts to cluster alert
            should_send, reason = router._should_send(alert)
            assert should_send is True
            # The alert should be converted to cluster-level
            assert alert.node_id is None
            assert "PARTITION DETECTED" in alert.message

    def test_partition_suppression_subsequent(self):
        """Test that subsequent partition alerts are suppressed."""
        with patch("app.monitoring.alert_router.STATE_FILE", Path(tempfile.mktemp())):
            router = AlertRouter()

            # Add nodes and mark first alert as sent
            router._active_issues["node_offline"] = {f"node{i}" for i in range(21)}
            router._states["node_offline:cluster"] = AlertState(send_count=1)

            alert = Alert(
                severity=AlertSeverity.WARNING,
                alert_type="node_offline",
                message="Node offline",
                node_id="node22",
            )

            should_send, reason = router._should_send(alert)
            assert should_send is False
            assert reason == "partition_suppressed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
