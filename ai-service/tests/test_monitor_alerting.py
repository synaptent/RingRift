"""Tests for scripts.monitor.alerting module.

Tests the unified alerting module.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from scripts.monitor.alerting import (
    LEVEL_COLORS,
    LEVEL_EMOJI,
    AlertLevel,
    AlertSeverity,
    send_alert,
    send_discord_alert,
    send_slack_alert,
)


class TestAlertLevel:
    """Tests for AlertLevel/AlertSeverity enum."""

    def test_alert_level_is_alert_severity(self):
        """Test that AlertLevel is an alias for AlertSeverity."""
        # AlertLevel should be the same as AlertSeverity for compatibility
        assert AlertLevel.INFO == AlertSeverity.INFO
        assert AlertLevel.WARNING == AlertSeverity.WARNING
        assert AlertLevel.CRITICAL == AlertSeverity.CRITICAL

    def test_all_levels_exist(self):
        """Test that all expected levels exist."""
        assert hasattr(AlertSeverity, 'DEBUG')
        assert hasattr(AlertSeverity, 'INFO')
        assert hasattr(AlertSeverity, 'WARNING')
        assert hasattr(AlertSeverity, 'ERROR')
        assert hasattr(AlertSeverity, 'CRITICAL')

    def test_level_values(self):
        """Test level value strings."""
        assert AlertSeverity.DEBUG.value == "debug"
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestLevelMappings:
    """Tests for level color and emoji mappings."""

    def test_all_levels_have_colors(self):
        """Test that all levels have color mappings."""
        for level in AlertSeverity:
            assert level in LEVEL_COLORS
            assert LEVEL_COLORS[level].startswith("#")

    def test_all_levels_have_emojis(self):
        """Test that all levels have emoji mappings."""
        for level in AlertSeverity:
            assert level in LEVEL_EMOJI
            assert LEVEL_EMOJI[level].startswith(":")

    def test_color_format(self):
        """Test color format is valid hex."""
        for _level, color in LEVEL_COLORS.items():
            assert len(color) == 7  # #RRGGBB
            assert color[0] == "#"
            # Should be valid hex
            int(color[1:], 16)


class TestSendSlackAlert:
    """Tests for send_slack_alert function."""

    @patch('urllib.request.urlopen')
    def test_successful_slack_alert(self, mock_urlopen):
        """Test successful Slack alert."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = send_slack_alert(
            webhook_url="https://hooks.slack.com/test",
            title="Test Alert",
            message="This is a test",
            level=AlertSeverity.INFO,
            node_id="test-node",
        )

        assert result is True
        mock_urlopen.assert_called_once()

    @patch('urllib.request.urlopen')
    def test_failed_slack_alert(self, mock_urlopen):
        """Test failed Slack alert."""
        mock_urlopen.side_effect = Exception("Connection failed")

        result = send_slack_alert(
            webhook_url="https://hooks.slack.com/test",
            title="Test",
            message="Test",
        )

        assert result is False

    @patch('urllib.request.urlopen')
    def test_slack_payload_format(self, mock_urlopen):
        """Test Slack payload is correctly formatted."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        send_slack_alert(
            webhook_url="https://hooks.slack.com/test",
            title="Test Title",
            message="Test Message",
            level=AlertSeverity.WARNING,
            node_id="my-node",
        )

        # Check the request was made with correct payload
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        payload = json.loads(request.data.decode())

        assert "attachments" in payload
        assert len(payload["attachments"]) == 1
        attachment = payload["attachments"][0]
        assert attachment["title"] == "Test Title"
        assert attachment["text"] == "Test Message"
        assert attachment["color"] == LEVEL_COLORS[AlertSeverity.WARNING]
        assert "my-node" in attachment["footer"]


class TestSendDiscordAlert:
    """Tests for send_discord_alert function."""

    @patch('urllib.request.urlopen')
    def test_successful_discord_alert(self, mock_urlopen):
        """Test successful Discord alert."""
        mock_response = MagicMock()
        mock_response.status = 204  # Discord returns 204 on success
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = send_discord_alert(
            webhook_url="https://discord.com/api/webhooks/test",
            title="Test Alert",
            message="This is a test",
            level=AlertSeverity.ERROR,
        )

        assert result is True

    @patch('urllib.request.urlopen')
    def test_discord_payload_format(self, mock_urlopen):
        """Test Discord payload is correctly formatted."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        send_discord_alert(
            webhook_url="https://discord.com/api/webhooks/test",
            title="Test Title",
            message="Test Message",
            level=AlertSeverity.CRITICAL,
        )

        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        payload = json.loads(request.data.decode())

        assert "embeds" in payload
        assert len(payload["embeds"]) == 1
        embed = payload["embeds"][0]
        assert "Test Title" in embed["title"]
        assert embed["description"] == "Test Message"
        # Color should be integer for Discord
        assert isinstance(embed["color"], int)


class TestSendAlert:
    """Tests for send_alert unified function."""

    @patch('scripts.monitor.alerting.get_webhook_urls')
    @patch('scripts.monitor.alerting.send_slack_alert')
    @patch('scripts.monitor.alerting.send_discord_alert')
    def test_send_to_both_webhooks(self, mock_discord, mock_slack, mock_get_urls):
        """Test sending to both Slack and Discord."""
        mock_get_urls.return_value = {
            "slack": "https://hooks.slack.com/test",
            "discord": "https://discord.com/api/webhooks/test",
        }
        mock_slack.return_value = True
        mock_discord.return_value = True

        result = send_alert(
            title="Test",
            message="Test message",
            level=AlertSeverity.INFO,
        )

        assert result is True
        mock_slack.assert_called_once()
        mock_discord.assert_called_once()

    @patch('scripts.monitor.alerting.get_webhook_urls')
    @patch('scripts.monitor.alerting.send_slack_alert')
    def test_send_slack_only(self, mock_slack, mock_get_urls):
        """Test sending to Slack only when Discord not configured."""
        mock_get_urls.return_value = {"slack": "https://hooks.slack.com/test"}
        mock_slack.return_value = True

        result = send_alert(
            title="Test",
            message="Test message",
        )

        assert result is True
        mock_slack.assert_called_once()

    @patch('scripts.monitor.alerting.get_webhook_urls')
    def test_no_webhooks_configured(self, mock_get_urls):
        """Test behavior when no webhooks configured."""
        mock_get_urls.return_value = {}

        result = send_alert(
            title="Test",
            message="Test message",
        )

        assert result is False

    @patch('scripts.monitor.alerting.get_webhook_urls')
    @patch('scripts.monitor.alerting.send_slack_alert')
    def test_override_webhook_url(self, mock_slack, mock_get_urls):
        """Test overriding webhook URL."""
        mock_get_urls.return_value = {}
        mock_slack.return_value = True

        result = send_alert(
            title="Test",
            message="Test",
            slack_url="https://custom.slack.webhook",
        )

        assert result is True
        mock_slack.assert_called_once()
        call_args = mock_slack.call_args
        assert call_args[0][0] == "https://custom.slack.webhook"

    @patch('scripts.monitor.alerting.get_webhook_urls')
    @patch('scripts.monitor.alerting.send_slack_alert')
    @patch('scripts.monitor.alerting.send_discord_alert')
    def test_partial_success(self, mock_discord, mock_slack, mock_get_urls):
        """Test partial success when one webhook fails."""
        mock_get_urls.return_value = {
            "slack": "https://hooks.slack.com/test",
            "discord": "https://discord.com/api/webhooks/test",
        }
        mock_slack.return_value = True
        mock_discord.return_value = False

        result = send_alert(title="Test", message="Test")

        # Should still return True if at least one succeeded
        assert result is True
