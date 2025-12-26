"""Alert Router - Multi-channel alerting with severity-based routing.

Routes alerts to appropriate channels (Slack, Discord, PagerDuty) based on
severity and provides deduplication and rate limiting.

Usage:
    from app.monitoring.alert_router import AlertRouter, Alert, AlertSeverity

    router = AlertRouter()
    await router.route_alert(Alert(
        severity=AlertSeverity.ERROR,
        alert_type="node_offline",
        node_id="gpu-node-1",
        message="Node offline for 30 minutes",
    ))
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class AlertSeverity(IntEnum):
    """Alert severity levels."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


@dataclass
class Alert:
    """An alert to be routed."""
    severity: AlertSeverity
    alert_type: str
    message: str
    node_id: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def key(self) -> str:
        """Unique key for deduplication."""
        return f"{self.alert_type}:{self.node_id or 'cluster'}"


@dataclass
class AlertState:
    """State tracking for an alert type."""
    last_sent: float = 0.0
    send_count: int = 0
    suppressed_count: int = 0


# Configuration
SLACK_WEBHOOK_URL = os.environ.get("RINGRIFT_SLACK_WEBHOOK", "")
DISCORD_WEBHOOK_URL = os.environ.get("RINGRIFT_DISCORD_WEBHOOK", "")
PAGERDUTY_ROUTING_KEY = os.environ.get("PAGERDUTY_ROUTING_KEY", "")

# Rate limiting (December 2025: imported from centralized thresholds)
try:
    from app.config.thresholds import (
        ALERT_PARTITION_THRESHOLD,
        MAX_ALERTS_PER_HOUR,
        MIN_ALERT_INTERVAL_SECONDS,
    )
    MIN_ALERT_INTERVAL = MIN_ALERT_INTERVAL_SECONDS  # Legacy alias
    PARTITION_THRESHOLD = ALERT_PARTITION_THRESHOLD  # Legacy alias
except ImportError:
    # Fallback if thresholds not available
    MIN_ALERT_INTERVAL = 1800  # 30 minutes between same alert
    MAX_ALERTS_PER_HOUR = 20
    PARTITION_THRESHOLD = 0.5  # Suppress if >50% nodes have same issue

# State persistence
STATE_FILE = Path("/tmp/ringrift_alert_router_state.json")


class SlackIntegration:
    """Slack webhook integration."""

    def __init__(self, webhook_url: str = SLACK_WEBHOOK_URL):
        self.webhook_url = webhook_url

    async def send(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        if not self.webhook_url:
            return False

        colors = {
            AlertSeverity.DEBUG: "#808080",
            AlertSeverity.INFO: "#00aa00",
            AlertSeverity.WARNING: "#ffaa00",
            AlertSeverity.ERROR: "#ff0000",
            AlertSeverity.CRITICAL: "#990000",
        }

        payload = {
            "attachments": [{
                "color": colors.get(alert.severity, "#808080"),
                "title": f"RingRift Alert: {alert.alert_type}",
                "text": alert.message,
                "fields": [
                    {"title": "Severity", "value": alert.severity.name, "short": True},
                    {"title": "Node", "value": alert.node_id or "cluster", "short": True},
                    {"title": "Time", "value": datetime.fromtimestamp(alert.timestamp).strftime("%Y-%m-%d %H:%M:%S"), "short": True},
                ],
                "footer": "RingRift Alert Router"
            }]
        }

        if alert.details:
            payload["attachments"][0]["fields"].append({
                "title": "Details",
                "value": json.dumps(alert.details, indent=2)[:500],
                "short": False
            })

        return await self._send_webhook(payload)

    async def _send_webhook(self, payload: dict) -> bool:
        """Send webhook request."""
        try:
            data = json.dumps(payload).encode('utf-8')
            req = Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            # Run in thread pool to not block
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: urlopen(req, timeout=10))
            return True
        except Exception as e:
            logger.error(f"Slack webhook failed: {e}")
            return False


class DiscordIntegration:
    """Discord webhook integration."""

    def __init__(self, webhook_url: str = DISCORD_WEBHOOK_URL):
        self.webhook_url = webhook_url

    async def send(self, alert: Alert) -> bool:
        """Send alert to Discord."""
        if not self.webhook_url:
            return False

        colors = {
            AlertSeverity.DEBUG: 0x808080,
            AlertSeverity.INFO: 0x00aa00,
            AlertSeverity.WARNING: 0xffaa00,
            AlertSeverity.ERROR: 0xff0000,
            AlertSeverity.CRITICAL: 0x990000,
        }

        payload = {
            "embeds": [{
                "title": f"RingRift Alert: {alert.alert_type}",
                "description": alert.message,
                "color": colors.get(alert.severity, 0x808080),
                "fields": [
                    {"name": "Severity", "value": alert.severity.name, "inline": True},
                    {"name": "Node", "value": alert.node_id or "cluster", "inline": True},
                ],
                "timestamp": datetime.fromtimestamp(alert.timestamp).isoformat(),
                "footer": {"text": "RingRift Alert Router"}
            }]
        }

        try:
            data = json.dumps(payload).encode('utf-8')
            req = Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: urlopen(req, timeout=10))
            return True
        except Exception as e:
            logger.error(f"Discord webhook failed: {e}")
            return False


class PagerDutyIntegration:
    """PagerDuty Events API v2 integration."""

    EVENTS_URL = "https://events.pagerduty.com/v2/enqueue"

    def __init__(self, routing_key: str = PAGERDUTY_ROUTING_KEY):
        self.routing_key = routing_key

    async def page(self, alert: Alert) -> bool:
        """Send alert to PagerDuty."""
        if not self.routing_key:
            return False

        severity_map = {
            AlertSeverity.DEBUG: "info",
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "error",
            AlertSeverity.CRITICAL: "critical",
        }

        payload = {
            "routing_key": self.routing_key,
            "event_action": "trigger",
            "dedup_key": alert.key,
            "payload": {
                "summary": f"[{alert.alert_type}] {alert.message}",
                "severity": severity_map.get(alert.severity, "warning"),
                "source": alert.node_id or "ringrift-cluster",
                "component": "keepalive",
                "custom_details": alert.details,
            }
        }

        try:
            data = json.dumps(payload).encode('utf-8')
            req = Request(
                self.EVENTS_URL,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: urlopen(req, timeout=10))
            logger.info(f"Sent PagerDuty alert: {alert.alert_type}")
            return True
        except Exception as e:
            logger.error(f"PagerDuty alert failed: {e}")
            return False

    async def resolve(self, alert_key: str) -> bool:
        """Resolve a PagerDuty incident."""
        if not self.routing_key:
            return False

        payload = {
            "routing_key": self.routing_key,
            "event_action": "resolve",
            "dedup_key": alert_key,
        }

        try:
            data = json.dumps(payload).encode('utf-8')
            req = Request(
                self.EVENTS_URL,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: urlopen(req, timeout=10))
            return True
        except Exception as e:
            logger.error(f"PagerDuty resolve failed: {e}")
            return False


class AlertRouter:
    """Routes alerts to appropriate channels based on severity."""

    def __init__(self):
        self.slack = SlackIntegration()
        self.discord = DiscordIntegration()
        self.pagerduty = PagerDutyIntegration()

        # Alert state tracking
        self._states: dict[str, AlertState] = {}
        self._hourly_count = 0
        self._hour_start = time.time()

        # Track active issues per type for partition detection
        self._active_issues: dict[str, set[str]] = {}  # alert_type -> set of node_ids

        self._load_state()

    def _load_state(self):
        """Load state from disk."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    data = json.load(f)
                    for key, state_data in data.get("states", {}).items():
                        self._states[key] = AlertState(**state_data)
                    self._hourly_count = data.get("hourly_count", 0)
                    self._hour_start = data.get("hour_start", time.time())
            except Exception as e:
                logger.warning(f"Failed to load alert state: {e}")

    def _save_state(self):
        """Save state to disk."""
        try:
            data = {
                "states": {k: {"last_sent": v.last_sent, "send_count": v.send_count, "suppressed_count": v.suppressed_count}
                           for k, v in self._states.items()},
                "hourly_count": self._hourly_count,
                "hour_start": self._hour_start,
            }
            with open(STATE_FILE, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save alert state: {e}")

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = time.time()

        # Reset hourly counter if needed
        if now - self._hour_start > 3600:
            self._hourly_count = 0
            self._hour_start = now

        return self._hourly_count < MAX_ALERTS_PER_HOUR

    def _should_send(self, alert: Alert) -> tuple[bool, str]:
        """Check if alert should be sent.

        Returns:
            (should_send, reason)
        """
        key = alert.key
        now = time.time()

        # Get or create state
        if key not in self._states:
            self._states[key] = AlertState()
        state = self._states[key]

        # Check deduplication
        if now - state.last_sent < MIN_ALERT_INTERVAL:
            state.suppressed_count += 1
            return False, "deduplicated"

        # Check rate limit
        if not self._check_rate_limit():
            state.suppressed_count += 1
            return False, "rate_limited"

        # Check for network partition (many nodes with same issue)
        if alert.node_id:
            if alert.alert_type not in self._active_issues:
                self._active_issues[alert.alert_type] = set()
            self._active_issues[alert.alert_type].add(alert.node_id)

            # If >50% of nodes have same issue, it's likely a partition
            # Send a single cluster-level alert instead
            if len(self._active_issues[alert.alert_type]) > 20:  # Rough threshold
                if state.send_count == 0:
                    # First alert of partition - convert to cluster alert
                    alert.node_id = None
                    alert.message = f"PARTITION DETECTED: {len(self._active_issues[alert.alert_type])} nodes affected by {alert.alert_type}"
                else:
                    return False, "partition_suppressed"

        return True, "ok"

    async def route_alert(self, alert: Alert) -> bool:
        """Route alert to appropriate channels based on severity.

        Returns:
            True if alert was sent successfully
        """
        should_send, reason = self._should_send(alert)
        if not should_send:
            logger.debug(f"Alert suppressed ({reason}): {alert.alert_type}")
            return False

        # Update state
        state = self._states[alert.key]
        state.last_sent = time.time()
        state.send_count += 1
        self._hourly_count += 1
        self._save_state()

        # Route based on severity
        tasks = []

        if alert.severity >= AlertSeverity.CRITICAL:
            # Critical: Slack + PagerDuty
            tasks.append(self.slack.send(alert))
            tasks.append(self.pagerduty.page(alert))
            logger.warning(f"CRITICAL alert: {alert.alert_type} - {alert.message}")

        elif alert.severity >= AlertSeverity.ERROR:
            # Error: Slack only
            tasks.append(self.slack.send(alert))
            logger.warning(f"ERROR alert: {alert.alert_type} - {alert.message}")

        elif alert.severity >= AlertSeverity.WARNING:
            # Warning: Slack only
            tasks.append(self.slack.send(alert))
            logger.info(f"WARNING alert: {alert.alert_type} - {alert.message}")

        else:
            # Info/Debug: Log only, maybe Discord
            if DISCORD_WEBHOOK_URL:
                tasks.append(self.discord.send(alert))
            logger.debug(f"INFO alert: {alert.alert_type}")

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return any(r is True for r in results)

        return True

    async def resolve_alert(self, alert_type: str, node_id: Optional[str] = None):
        """Mark an alert as resolved."""
        key = f"{alert_type}:{node_id or 'cluster'}"

        # Remove from active issues
        if alert_type in self._active_issues and node_id:
            self._active_issues[alert_type].discard(node_id)

        # Resolve PagerDuty if critical
        await self.pagerduty.resolve(key)

        logger.info(f"Alert resolved: {key}")

    def get_stats(self) -> dict[str, Any]:
        """Get alerting statistics."""
        return {
            "hourly_count": self._hourly_count,
            "hour_remaining": max(0, 3600 - (time.time() - self._hour_start)),
            "active_issues": {k: len(v) for k, v in self._active_issues.items()},
            "states": {
                k: {"send_count": v.send_count, "suppressed_count": v.suppressed_count}
                for k, v in self._states.items()
            }
        }


# Singleton instance
_router: Optional[AlertRouter] = None


def get_alert_router() -> AlertRouter:
    """Get singleton AlertRouter instance."""
    global _router
    if _router is None:
        _router = AlertRouter()
    return _router


async def send_alert(
    alert_type: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.WARNING,
    node_id: Optional[str] = None,
    details: Optional[dict] = None,
) -> bool:
    """Convenience function to send an alert."""
    router = get_alert_router()
    alert = Alert(
        severity=severity,
        alert_type=alert_type,
        message=message,
        node_id=node_id,
        details=details or {},
    )
    return await router.route_alert(alert)
