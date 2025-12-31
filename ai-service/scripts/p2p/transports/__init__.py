"""
Transport implementations for the transport cascade.

Dec 30, 2025: Comprehensive transport layer with tiered failover.
"""

from .http_transports import (
    DirectHTTPTransport,
    TailscaleHTTPTransport,
    CloudflareHTTPTransport,
)
from .ssh_transport import SSHTunnelTransport
from .relay_transport import P2PRelayTransport
from .notification_transports import (
    SlackWebhookTransport,
    DiscordWebhookTransport,
    TelegramTransport,
    EmailTransport,
    SMSTransport,
    PagerDutyTransport,
)

__all__ = [
    # Tier 1-2: Fast/Reliable
    "DirectHTTPTransport",
    "TailscaleHTTPTransport",
    # Tier 3: Tunneled
    "CloudflareHTTPTransport",
    "SSHTunnelTransport",
    # Tier 4: Relay
    "P2PRelayTransport",
    # Tier 5: External
    "SlackWebhookTransport",
    "DiscordWebhookTransport",
    "TelegramTransport",
    "EmailTransport",
    # Tier 6: Manual
    "SMSTransport",
    "PagerDutyTransport",
]
