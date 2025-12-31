"""
Notification-based transport implementations.

Tier 5 (EXTERNAL): Slack, Discord, Telegram, Email
Tier 6 (MANUAL): SMS, PagerDuty

These are "transports of last resort" - used when all network transports fail.
They don't actually deliver the payload to the target node, but instead
alert operators so they can manually intervene.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any

try:
    import aiohttp
    from aiohttp import ClientTimeout
except ImportError:
    aiohttp = None  # type: ignore
    ClientTimeout = None  # type: ignore

from ..transport_cascade import BaseTransport, TransportResult, TransportTier

logger = logging.getLogger(__name__)


class SlackWebhookTransport(BaseTransport):
    """
    Slack webhook transport for emergency notifications.

    Tier 5 (EXTERNAL): Alerts operators when network transports fail.
    """

    name = "slack_webhook"
    tier = TransportTier.TIER_5_EXTERNAL

    def __init__(self, webhook_url: str | None = None):
        self._webhook_url = webhook_url or os.environ.get("RINGRIFT_SLACK_WEBHOOK_URL")
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=30.0)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def send(self, target: str, payload: bytes) -> TransportResult:
        """Send alert to Slack."""
        if not self._webhook_url:
            return self._make_result(
                success=False,
                latency_ms=0,
                error="Slack webhook URL not configured",
            )

        if aiohttp is None:
            return self._make_result(
                success=False, latency_ms=0, error="aiohttp not installed"
            )

        # Format payload as Slack message
        message = {
            "text": f":warning: P2P Transport Alert",
            "attachments": [
                {
                    "color": "warning",
                    "title": f"Failed to reach node: {target}",
                    "text": f"All network transports exhausted. Payload size: {len(payload)} bytes",
                    "fields": [
                        {"title": "Target", "value": target, "short": True},
                        {"title": "Payload Size", "value": f"{len(payload)} bytes", "short": True},
                    ],
                    "footer": "RingRift P2P Transport Cascade",
                    "ts": int(time.time()),
                }
            ],
        }

        start_time = time.time()
        try:
            session = await self._get_session()
            async with session.post(
                self._webhook_url,
                json=message,
                headers={"Content-Type": "application/json"},
            ) as resp:
                latency_ms = (time.time() - start_time) * 1000
                if resp.status == 200:
                    return self._make_result(
                        success=True,
                        latency_ms=latency_ms,
                        response=b"alert_sent",
                        channel="slack",
                    )
                else:
                    return self._make_result(
                        success=False,
                        latency_ms=latency_ms,
                        error=f"Slack HTTP {resp.status}",
                    )

        except Exception as e:
            return self._make_result(
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=f"{type(e).__name__}: {e}",
            )

    async def is_available(self, target: str) -> bool:
        """Check if Slack webhook is configured."""
        return bool(self._webhook_url)


class DiscordWebhookTransport(BaseTransport):
    """
    Discord webhook transport for emergency notifications.

    Tier 5 (EXTERNAL): Alerts operators when network transports fail.
    """

    name = "discord_webhook"
    tier = TransportTier.TIER_5_EXTERNAL

    def __init__(self, webhook_url: str | None = None):
        self._webhook_url = webhook_url or os.environ.get("RINGRIFT_DISCORD_WEBHOOK_URL")
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=30.0)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def send(self, target: str, payload: bytes) -> TransportResult:
        """Send alert to Discord."""
        if not self._webhook_url:
            return self._make_result(
                success=False,
                latency_ms=0,
                error="Discord webhook URL not configured",
            )

        if aiohttp is None:
            return self._make_result(
                success=False, latency_ms=0, error="aiohttp not installed"
            )

        # Format as Discord embed
        message = {
            "embeds": [
                {
                    "title": ":warning: P2P Transport Alert",
                    "description": f"Failed to reach node: **{target}**",
                    "color": 16776960,  # Yellow
                    "fields": [
                        {"name": "Target", "value": target, "inline": True},
                        {"name": "Payload Size", "value": f"{len(payload)} bytes", "inline": True},
                    ],
                    "footer": {"text": "RingRift P2P Transport Cascade"},
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            ]
        }

        start_time = time.time()
        try:
            session = await self._get_session()
            async with session.post(
                self._webhook_url,
                json=message,
                headers={"Content-Type": "application/json"},
            ) as resp:
                latency_ms = (time.time() - start_time) * 1000
                if resp.status in (200, 204):
                    return self._make_result(
                        success=True,
                        latency_ms=latency_ms,
                        response=b"alert_sent",
                        channel="discord",
                    )
                else:
                    return self._make_result(
                        success=False,
                        latency_ms=latency_ms,
                        error=f"Discord HTTP {resp.status}",
                    )

        except Exception as e:
            return self._make_result(
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=f"{type(e).__name__}: {e}",
            )

    async def is_available(self, target: str) -> bool:
        """Check if Discord webhook is configured."""
        return bool(self._webhook_url)


class TelegramTransport(BaseTransport):
    """
    Telegram bot transport for emergency notifications.

    Tier 5 (EXTERNAL): Alerts operators via Telegram.
    """

    name = "telegram_bot"
    tier = TransportTier.TIER_5_EXTERNAL

    def __init__(self, bot_token: str | None = None, chat_id: str | None = None):
        self._bot_token = bot_token or os.environ.get("RINGRIFT_TELEGRAM_BOT_TOKEN")
        self._chat_id = chat_id or os.environ.get("RINGRIFT_TELEGRAM_CHAT_ID")
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=30.0)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def send(self, target: str, payload: bytes) -> TransportResult:
        """Send alert to Telegram."""
        if not self._bot_token or not self._chat_id:
            return self._make_result(
                success=False,
                latency_ms=0,
                error="Telegram bot token or chat ID not configured",
            )

        if aiohttp is None:
            return self._make_result(
                success=False, latency_ms=0, error="aiohttp not installed"
            )

        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        message = {
            "chat_id": self._chat_id,
            "text": (
                f"⚠️ *P2P Transport Alert*\n\n"
                f"Failed to reach node: `{target}`\n"
                f"Payload size: {len(payload)} bytes\n"
                f"All network transports exhausted."
            ),
            "parse_mode": "Markdown",
        }

        start_time = time.time()
        try:
            session = await self._get_session()
            async with session.post(url, json=message) as resp:
                latency_ms = (time.time() - start_time) * 1000
                data = await resp.json()

                if data.get("ok"):
                    return self._make_result(
                        success=True,
                        latency_ms=latency_ms,
                        response=b"alert_sent",
                        channel="telegram",
                    )
                else:
                    return self._make_result(
                        success=False,
                        latency_ms=latency_ms,
                        error=f"Telegram API error: {data.get('description', 'unknown')}",
                    )

        except Exception as e:
            return self._make_result(
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=f"{type(e).__name__}: {e}",
            )

    async def is_available(self, target: str) -> bool:
        """Check if Telegram is configured."""
        return bool(self._bot_token) and bool(self._chat_id)


class EmailTransport(BaseTransport):
    """
    Email (SMTP) transport for emergency notifications.

    Tier 5 (EXTERNAL): Alerts operators via email.
    """

    name = "email_smtp"
    tier = TransportTier.TIER_5_EXTERNAL

    def __init__(
        self,
        smtp_host: str | None = None,
        smtp_port: int | None = None,
        smtp_user: str | None = None,
        smtp_password: str | None = None,
        to_emails: list[str] | None = None,
    ):
        self._smtp_host = smtp_host or os.environ.get("RINGRIFT_SMTP_HOST")
        self._smtp_port = smtp_port or int(os.environ.get("RINGRIFT_SMTP_PORT", "587"))
        self._smtp_user = smtp_user or os.environ.get("RINGRIFT_SMTP_USER")
        self._smtp_password = smtp_password or os.environ.get("RINGRIFT_SMTP_PASSWORD")
        to_emails_env = os.environ.get("RINGRIFT_ALERT_EMAILS", "")
        self._to_emails = to_emails or [e.strip() for e in to_emails_env.split(",") if e.strip()]

    async def send(self, target: str, payload: bytes) -> TransportResult:
        """Send alert via email."""
        if not self._smtp_host or not self._smtp_user or not self._to_emails:
            return self._make_result(
                success=False,
                latency_ms=0,
                error="Email SMTP not configured",
            )

        start_time = time.time()
        try:
            # Run SMTP in thread to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._send_email_sync,
                target,
                len(payload),
            )
            latency_ms = (time.time() - start_time) * 1000

            if result:
                return self._make_result(
                    success=True,
                    latency_ms=latency_ms,
                    response=b"alert_sent",
                    channel="email",
                )
            else:
                return self._make_result(
                    success=False,
                    latency_ms=latency_ms,
                    error="Email send failed",
                )

        except Exception as e:
            return self._make_result(
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=f"{type(e).__name__}: {e}",
            )

    def _send_email_sync(self, target: str, payload_size: int) -> bool:
        """Synchronous email send."""
        msg = MIMEMultipart()
        msg["From"] = self._smtp_user
        msg["To"] = ", ".join(self._to_emails)
        msg["Subject"] = f"[RingRift] P2P Transport Alert: {target}"

        body = (
            f"P2P Transport Alert\n\n"
            f"Failed to reach node: {target}\n"
            f"Payload size: {payload_size} bytes\n"
            f"All network transports exhausted.\n\n"
            f"Please investigate immediately."
        )
        msg.attach(MIMEText(body, "plain"))

        try:
            with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
                server.starttls()
                if self._smtp_password:
                    server.login(self._smtp_user, self._smtp_password)
                server.sendmail(self._smtp_user, self._to_emails, msg.as_string())
            return True
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    async def is_available(self, target: str) -> bool:
        """Check if email is configured."""
        return bool(self._smtp_host) and bool(self._smtp_user) and bool(self._to_emails)


class SMSTransport(BaseTransport):
    """
    Twilio SMS transport for critical alerts.

    Tier 6 (MANUAL): Last resort for critical failures.
    """

    name = "sms_twilio"
    tier = TransportTier.TIER_6_MANUAL

    def __init__(
        self,
        account_sid: str | None = None,
        auth_token: str | None = None,
        from_number: str | None = None,
        to_numbers: list[str] | None = None,
    ):
        self._account_sid = account_sid or os.environ.get("RINGRIFT_TWILIO_ACCOUNT_SID")
        self._auth_token = auth_token or os.environ.get("RINGRIFT_TWILIO_AUTH_TOKEN")
        self._from_number = from_number or os.environ.get("RINGRIFT_TWILIO_FROM_NUMBER")
        to_numbers_env = os.environ.get("RINGRIFT_TWILIO_TO_NUMBERS", "")
        self._to_numbers = to_numbers or [n.strip() for n in to_numbers_env.split(",") if n.strip()]
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=30.0)
            auth = aiohttp.BasicAuth(self._account_sid or "", self._auth_token or "")
            self._session = aiohttp.ClientSession(timeout=timeout, auth=auth)
        return self._session

    async def send(self, target: str, payload: bytes) -> TransportResult:
        """Send SMS via Twilio."""
        if not self._account_sid or not self._auth_token or not self._from_number or not self._to_numbers:
            return self._make_result(
                success=False,
                latency_ms=0,
                error="Twilio SMS not configured",
            )

        if aiohttp is None:
            return self._make_result(
                success=False, latency_ms=0, error="aiohttp not installed"
            )

        url = f"https://api.twilio.com/2010-04-01/Accounts/{self._account_sid}/Messages.json"
        message_body = f"[RingRift] CRITICAL: Failed to reach {target}. All transports exhausted."

        start_time = time.time()
        errors = []

        session = await self._get_session()
        for to_number in self._to_numbers:
            try:
                async with session.post(
                    url,
                    data={
                        "From": self._from_number,
                        "To": to_number,
                        "Body": message_body,
                    },
                ) as resp:
                    if resp.status not in (200, 201):
                        errors.append(f"{to_number}: HTTP {resp.status}")
                    else:
                        logger.info(f"SMS sent to {to_number}")

            except Exception as e:
                errors.append(f"{to_number}: {e}")

        latency_ms = (time.time() - start_time) * 1000

        if errors:
            return self._make_result(
                success=False,
                latency_ms=latency_ms,
                error=f"SMS errors: {'; '.join(errors)}",
            )
        else:
            return self._make_result(
                success=True,
                latency_ms=latency_ms,
                response=b"alert_sent",
                channel="sms",
                recipients=len(self._to_numbers),
            )

    async def is_available(self, target: str) -> bool:
        """Check if Twilio SMS is configured."""
        return (
            bool(self._account_sid)
            and bool(self._auth_token)
            and bool(self._from_number)
            and bool(self._to_numbers)
        )


class PagerDutyTransport(BaseTransport):
    """
    PagerDuty incident creation transport.

    Tier 6 (MANUAL): Creates incidents for on-call response.
    """

    name = "pagerduty"
    tier = TransportTier.TIER_6_MANUAL

    def __init__(self, routing_key: str | None = None):
        self._routing_key = routing_key or os.environ.get("RINGRIFT_PAGERDUTY_ROUTING_KEY")
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=30.0)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def send(self, target: str, payload: bytes) -> TransportResult:
        """Create PagerDuty incident."""
        if not self._routing_key:
            return self._make_result(
                success=False,
                latency_ms=0,
                error="PagerDuty routing key not configured",
            )

        if aiohttp is None:
            return self._make_result(
                success=False, latency_ms=0, error="aiohttp not installed"
            )

        url = "https://events.pagerduty.com/v2/enqueue"
        event = {
            "routing_key": self._routing_key,
            "event_action": "trigger",
            "dedup_key": f"ringrift-transport-{target}",
            "payload": {
                "summary": f"RingRift P2P: Failed to reach node {target}",
                "severity": "critical",
                "source": "ringrift-p2p",
                "custom_details": {
                    "target_node": target,
                    "payload_size": len(payload),
                    "message": "All network transports exhausted",
                },
            },
        }

        start_time = time.time()
        try:
            session = await self._get_session()
            async with session.post(
                url,
                json=event,
                headers={"Content-Type": "application/json"},
            ) as resp:
                latency_ms = (time.time() - start_time) * 1000
                data = await resp.json()

                if data.get("status") == "success":
                    return self._make_result(
                        success=True,
                        latency_ms=latency_ms,
                        response=b"incident_created",
                        channel="pagerduty",
                        dedup_key=event["dedup_key"],
                    )
                else:
                    return self._make_result(
                        success=False,
                        latency_ms=latency_ms,
                        error=f"PagerDuty error: {data.get('message', 'unknown')}",
                    )

        except Exception as e:
            return self._make_result(
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=f"{type(e).__name__}: {e}",
            )

    async def is_available(self, target: str) -> bool:
        """Check if PagerDuty is configured."""
        return bool(self._routing_key)
