"""Webhook Notifier for Slack/Discord alerts.

January 2026: Extracted from p2p_orchestrator.py (~175 LOC saved).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import TYPE_CHECKING

from aiohttp import ClientSession, ClientTimeout

from scripts.p2p.cluster_config import get_webhook_urls

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class WebhookNotifier:
    """Sends alerts to Slack/Discord webhooks for important events.

    Configure via environment variables:
    - RINGRIFT_SLACK_WEBHOOK: Slack incoming webhook URL
    - RINGRIFT_DISCORD_WEBHOOK: Discord webhook URL
    - RINGRIFT_ALERT_LEVEL: Minimum level to alert (debug/info/warning/error) default: warning
    """

    LEVELS = {"debug": 0, "info": 1, "warning": 2, "error": 3}

    def __init__(self):
        # Try environment variables first, then fall back to cluster.yaml
        self.slack_webhook = os.environ.get("RINGRIFT_SLACK_WEBHOOK", "")
        self.discord_webhook = os.environ.get("RINGRIFT_DISCORD_WEBHOOK", "")

        # Fall back to cluster.yaml config if env vars not set
        if not self.slack_webhook or not self.discord_webhook:
            try:
                yaml_webhooks = get_webhook_urls()
                if not self.slack_webhook and "slack" in yaml_webhooks:
                    self.slack_webhook = yaml_webhooks["slack"]
                if not self.discord_webhook and "discord" in yaml_webhooks:
                    self.discord_webhook = yaml_webhooks["discord"]
            except (KeyError, IndexError, AttributeError):
                pass  # Ignore config loading errors

        self.min_level = self.LEVELS.get(
            os.environ.get("RINGRIFT_ALERT_LEVEL", "warning").lower(), 2
        )
        self._session: ClientSession | None = None
        self._last_alert: dict[str, float] = {}  # Throttle repeated alerts
        self._throttle_seconds = 300  # 5 minutes between duplicate alerts

    async def _get_session(self) -> ClientSession:
        if self._session is None or self._session.closed:
            self._session = ClientSession(timeout=ClientTimeout(total=10))
        return self._session

    async def close(self) -> None:
        """Close the HTTP session to prevent memory leaks.

        December 2025: Added to fix memory leak from unclosed sessions.
        """
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    def close_sync(self) -> None:
        """Synchronously close the HTTP session (for finally blocks)."""
        if self._session is not None and not self._session.closed:
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self._session.close())
                loop.close()
            except (RuntimeError, OSError, asyncio.CancelledError) as e:
                # Dec 2025: Narrowed from bare Exception; best effort cleanup
                logger.debug(f"HTTP session close failed (best effort): {e}")
            self._session = None

    def _should_throttle(self, alert_key: str) -> bool:
        """Check if this alert should be throttled (duplicate within window)."""
        now = time.time()
        if alert_key in self._last_alert and now - self._last_alert[alert_key] < self._throttle_seconds:
            return True
        self._last_alert[alert_key] = now
        return False

    async def send(
        self,
        title: str,
        message: str = "",
        level: str = "warning",
        fields: dict[str, str] | None = None,
        node_id: str = "",
        # Aliases for backward compatibility (December 28, 2025)
        severity: str | None = None,
        context: dict[str, str] | None = None,
    ):
        """Send an alert to configured webhooks.

        Args:
            title: Alert title/subject (or message if message not provided)
            message: Alert body text
            level: debug/info/warning/error
            fields: Additional key-value pairs to include
            node_id: Node ID for deduplication
            severity: Alias for level (backward compatibility)
            context: Alias for fields (backward compatibility)

        December 28, 2025: Added severity and context aliases to fix API mismatch
        with callers using the alternative parameter names.
        """
        # Handle aliases - severity takes precedence if provided
        if severity is not None:
            level = severity
        if context is not None:
            fields = context
        # If message is empty, use title as message (for single-arg callers)
        if not message:
            message = title
            title = "RingRift Alert"

        if self.LEVELS.get(level, 2) < self.min_level:
            return

        if not self.slack_webhook and not self.discord_webhook:
            return

        # Throttle duplicate alerts
        alert_key = f"{title}:{node_id}"
        if self._should_throttle(alert_key):
            return

        try:
            session = await self._get_session()

            # Color based on level
            colors = {"debug": "#808080", "info": "#36a64f", "warning": "#ff9800", "error": "#ff0000"}
            color = colors.get(level, "#808080")

            # Send to Slack
            if self.slack_webhook:
                slack_fields = []
                if fields:
                    for k, v in fields.items():
                        slack_fields.append({"title": k, "value": str(v), "short": True})

                slack_payload = {
                    "attachments": [{
                        "color": color,
                        "title": f"[{level.upper()}] {title}",
                        "text": message,
                        "fields": slack_fields,
                        "footer": f"RingRift AI | {node_id}" if node_id else "RingRift AI",
                        "ts": int(time.time()),
                    }]
                }
                try:
                    async with session.post(self.slack_webhook, json=slack_payload) as resp:
                        if resp.status != 200:
                            logger.warning(f"[Webhook] Slack alert failed: {resp.status}")
                except Exception as e:  # noqa: BLE001
                    logger.error(f"[Webhook] Slack error: {e}")

            # Send to Discord
            if self.discord_webhook:
                discord_fields = []
                if fields:
                    for k, v in fields.items():
                        discord_fields.append({"name": k, "value": str(v), "inline": True})

                discord_payload = {
                    "embeds": [{
                        "title": f"[{level.upper()}] {title}",
                        "description": message,
                        "color": int(color.lstrip("#"), 16),
                        "fields": discord_fields,
                        "footer": {"text": f"RingRift AI | {node_id}" if node_id else "RingRift AI"},
                        "timestamp": datetime.utcnow().isoformat(),
                    }]
                }
                try:
                    async with session.post(self.discord_webhook, json=discord_payload) as resp:
                        if resp.status not in (200, 204):
                            logger.warning(f"[Webhook] Discord alert failed: {resp.status}")
                except Exception as e:  # noqa: BLE001
                    logger.error(f"[Webhook] Discord error: {e}")

        except Exception as e:  # noqa: BLE001
            logger.error(f"[Webhook] Alert send error: {e}")

    # Dec 28, 2025: Removed duplicate close() method (was lines 2031-2033)
    # The proper close() is defined at line 1898 with docstring and session=None cleanup
