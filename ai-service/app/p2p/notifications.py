"""P2P Webhook and Notification Utilities.

Provides utilities for sending notifications about cluster events.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


@dataclass
class WebhookConfig:
    """Configuration for webhook notifications."""

    # Slack webhook URL
    slack_url: str = field(
        default_factory=lambda: os.environ.get("RINGRIFT_SLACK_WEBHOOK_URL", "")
    )

    # Discord webhook URL
    discord_url: str = field(
        default_factory=lambda: os.environ.get("RINGRIFT_DISCORD_WEBHOOK_URL", "")
    )

    # Generic webhook URL (receives JSON)
    generic_url: str = field(
        default_factory=lambda: os.environ.get("RINGRIFT_WEBHOOK_URL", "")
    )

    # Rate limiting
    min_interval_seconds: float = 60.0  # Minimum time between notifications
    max_per_hour: int = 30  # Maximum notifications per hour

    # Filtering
    min_severity: str = "warning"  # "info", "warning", "error", "critical"

    def has_any_webhook(self) -> bool:
        """Check if any webhook is configured."""
        return bool(self.slack_url or self.discord_url or self.generic_url)


@dataclass
class NotificationEvent:
    """An event to be notified."""
    event_type: str  # e.g., "training_started", "node_offline", "elo_improvement"
    severity: str  # "info", "warning", "error", "critical"
    title: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_slack_payload(self) -> dict[str, Any]:
        """Convert to Slack message format."""
        # Severity colors
        colors = {
            "info": "#2196F3",
            "warning": "#FFC107",
            "error": "#F44336",
            "critical": "#9C27B0",
        }

        # Build attachment
        attachment = {
            "color": colors.get(self.severity, "#808080"),
            "title": self.title,
            "text": self.message,
            "ts": time.time(),
            "fields": [
                {"title": k, "value": str(v), "short": True}
                for k, v in list(self.details.items())[:6]
            ],
        }

        return {
            "attachments": [attachment],
        }

    def to_discord_payload(self) -> dict[str, Any]:
        """Convert to Discord embed format."""
        # Severity colors (Discord uses decimal)
        colors = {
            "info": 2201331,
            "warning": 16761095,
            "error": 16007990,
            "critical": 10233904,
        }

        embed = {
            "title": self.title,
            "description": self.message,
            "color": colors.get(self.severity, 8421504),
            "timestamp": self.timestamp,
            "fields": [
                {"name": k, "value": str(v), "inline": True}
                for k, v in list(self.details.items())[:6]
            ],
        }

        return {"embeds": [embed]}

    def to_generic_payload(self) -> dict[str, Any]:
        """Convert to generic JSON payload."""
        return {
            "event_type": self.event_type,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class NotificationManager:
    """Manages sending notifications with rate limiting."""

    def __init__(self, config: WebhookConfig | None = None):
        self.config = config or WebhookConfig()
        self._last_notification_time: float = 0
        self._notifications_this_hour: list[float] = []

    def _can_send(self, severity: str) -> bool:
        """Check if we can send a notification."""
        # Check severity filter
        severity_levels = {"info": 0, "warning": 1, "error": 2, "critical": 3}
        if severity_levels.get(severity, 0) < severity_levels.get(self.config.min_severity, 1):
            return False

        # Check rate limit (per-hour)
        now = time.time()
        hour_ago = now - 3600
        self._notifications_this_hour = [
            t for t in self._notifications_this_hour if t > hour_ago
        ]
        if len(self._notifications_this_hour) >= self.config.max_per_hour:
            return False

        # Check interval (allow critical notifications to bypass)
        if (now - self._last_notification_time < self.config.min_interval_seconds
                and severity != "critical"):
            return False

        return True

    async def send_async(self, event: NotificationEvent) -> bool:
        """Send notification asynchronously."""
        if not HAS_AIOHTTP:
            return False

        if not self.config.has_any_webhook():
            return False

        if not self._can_send(event.severity):
            return False

        success = False
        async with aiohttp.ClientSession() as session:
            tasks = []

            if self.config.slack_url:
                tasks.append(self._send_to_url(
                    session, self.config.slack_url, event.to_slack_payload()
                ))

            if self.config.discord_url:
                tasks.append(self._send_to_url(
                    session, self.config.discord_url, event.to_discord_payload()
                ))

            if self.config.generic_url:
                tasks.append(self._send_to_url(
                    session, self.config.generic_url, event.to_generic_payload()
                ))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                success = any(r is True for r in results)

        if success:
            self._last_notification_time = time.time()
            self._notifications_this_hour.append(time.time())

        return success

    async def _send_to_url(
        self,
        session: aiohttp.ClientSession,
        url: str,
        payload: dict[str, Any],
    ) -> bool:
        """Send payload to a URL."""
        try:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                return response.status < 400
        except Exception:
            return False

    def send_sync(self, event: NotificationEvent) -> bool:
        """Send notification synchronously."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule as task if already in async context
                asyncio.create_task(self.send_async(event))
                return True
            else:
                return loop.run_until_complete(self.send_async(event))
        except RuntimeError:
            # No event loop - create one
            return asyncio.run(self.send_async(event))


# Global notification manager
_notification_manager: NotificationManager | None = None


def get_notification_manager() -> NotificationManager:
    """Get global notification manager."""
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager()
    return _notification_manager


def send_webhook_notification(
    event_type: str,
    title: str,
    message: str,
    severity: str = "info",
    details: dict[str, Any] | None = None,
) -> bool:
    """Convenience function to send a webhook notification.

    Args:
        event_type: Type of event (e.g., "training_complete")
        title: Notification title
        message: Notification message
        severity: "info", "warning", "error", or "critical"
        details: Additional details dict

    Returns:
        True if notification was sent successfully
    """
    event = NotificationEvent(
        event_type=event_type,
        severity=severity,
        title=title,
        message=message,
        details=details or {},
    )

    manager = get_notification_manager()
    return manager.send_sync(event)


# Convenience functions for common events
def notify_training_started(
    board_type: str,
    num_players: int,
    games: int,
    node_id: str,
) -> bool:
    """Notify that training has started."""
    return send_webhook_notification(
        event_type="training_started",
        title=f"Training Started: {board_type}_{num_players}p",
        message=f"Training job started on {node_id} with {games:,} games",
        severity="info",
        details={
            "board_type": board_type,
            "num_players": num_players,
            "games": games,
            "node_id": node_id,
        },
    )


def notify_training_complete(
    board_type: str,
    num_players: int,
    elo_change: float,
    duration_hours: float,
) -> bool:
    """Notify that training completed."""
    severity = "info" if elo_change >= 0 else "warning"
    return send_webhook_notification(
        event_type="training_complete",
        title=f"Training Complete: {board_type}_{num_players}p",
        message=f"Elo change: {elo_change:+.1f} after {duration_hours:.1f}h training",
        severity=severity,
        details={
            "board_type": board_type,
            "num_players": num_players,
            "elo_change": elo_change,
            "duration_hours": duration_hours,
        },
    )


def notify_node_offline(node_id: str, last_seen: str) -> bool:
    """Notify that a node went offline."""
    return send_webhook_notification(
        event_type="node_offline",
        title=f"Node Offline: {node_id}",
        message=f"Node {node_id} is offline (last seen: {last_seen})",
        severity="warning",
        details={"node_id": node_id, "last_seen": last_seen},
    )


def notify_cluster_health(health: str, online_nodes: int, total_nodes: int) -> bool:
    """Notify about cluster health status."""
    severity = {
        "healthy": "info",
        "degraded": "warning",
        "unhealthy": "error",
        "critical": "critical",
    }.get(health, "warning")

    return send_webhook_notification(
        event_type="cluster_health",
        title=f"Cluster Health: {health.upper()}",
        message=f"{online_nodes}/{total_nodes} nodes online",
        severity=severity,
        details={
            "health": health,
            "online_nodes": online_nodes,
            "total_nodes": total_nodes,
        },
    )
