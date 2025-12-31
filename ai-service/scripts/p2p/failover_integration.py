"""
Failover Integration Mixin for P2POrchestrator.

Dec 30, 2025: Part of Phase 9 - Multi-Layer Failover Architecture.

Integrates the transport cascade, union discovery, and protocol union
into the P2P orchestrator for maximum connectivity.

Usage:
    class P2POrchestrator(FailoverIntegrationMixin, ...):
        pass
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

from scripts.p2p.p2p_mixin_base import P2PMixinBase

if TYPE_CHECKING:
    from scripts.p2p.models import NodeInfo
    from scripts.p2p.transport_cascade import TransportCascade, TransportResult
    from scripts.p2p.union_discovery import UnionDiscovery
    from scripts.p2p.protocol_union import ProtocolUnion

logger = logging.getLogger(__name__)

# Feature flag for enabling the failover system
FAILOVER_ENABLED = os.environ.get("RINGRIFT_FAILOVER_ENABLED", "true").lower() == "true"


class FailoverIntegrationMixin(P2PMixinBase):
    """
    Mixin that integrates advanced failover capabilities into P2POrchestrator.

    Provides:
    - Transport cascade for multi-tier message delivery
    - Union discovery for comprehensive peer discovery
    - Protocol union for combined membership views

    Features:
    - Automatic initialization on first use
    - Graceful fallback when components unavailable
    - Event emission for failover tracking
    - Stats collection for monitoring

    Requires the implementing class to have:
    - node_id: str
    - peers: dict[str, NodeInfo]
    - peers_lock: RLock
    - voter_node_ids: list[str]
    """

    MIXIN_TYPE = "failover_integration"

    # Type hints
    node_id: str
    peers: dict[str, Any]
    peers_lock: Any
    voter_node_ids: list[str]

    # Internal state
    _transport_cascade: "TransportCascade | None" = None
    _union_discovery: "UnionDiscovery | None" = None
    _protocol_union: "ProtocolUnion | None" = None
    _failover_initialized: bool = False

    def _init_failover_system(self) -> bool:
        """Initialize the failover system components.

        Returns:
            True if initialization successful
        """
        if not FAILOVER_ENABLED:
            self._log_debug("Failover system disabled via RINGRIFT_FAILOVER_ENABLED")
            return False

        if self._failover_initialized:
            return True

        try:
            # Initialize transport cascade
            self._init_transport_cascade()

            # Initialize union discovery
            self._init_union_discovery()

            # Initialize protocol union
            self._init_protocol_union()

            self._failover_initialized = True
            self._log_info("Failover system initialized successfully")
            return True

        except Exception as e:
            self._log_error(f"Failed to initialize failover system: {e}")
            return False

    def _init_transport_cascade(self) -> None:
        """Initialize the transport cascade with all transport tiers."""
        try:
            from scripts.p2p.transport_cascade import (
                TransportCascade,
                get_transport_cascade,
            )
            from scripts.p2p.transports import (
                DirectHTTPTransport,
                TailscaleHTTPTransport,
                CloudflareHTTPTransport,
                SSHTunnelTransport,
                P2PRelayTransport,
            )

            # Get or create cascade
            self._transport_cascade = get_transport_cascade()

            # Register Tier 1: Direct HTTP
            direct_http = DirectHTTPTransport(port=getattr(self, "port", 8770))
            self._transport_cascade.register_transport(direct_http)

            # Register Tier 2: Tailscale
            tailscale = TailscaleHTTPTransport(port=getattr(self, "port", 8770))
            self._transport_cascade.register_transport(tailscale)

            # Register Tier 3: SSH Tunnel
            ssh = SSHTunnelTransport()
            self._transport_cascade.register_transport(ssh)

            # Register Tier 3: Cloudflare (if configured)
            cloudflare = CloudflareHTTPTransport()
            self._transport_cascade.register_transport(cloudflare)

            # Register Tier 4: P2P Relay
            relay = P2PRelayTransport(port=getattr(self, "port", 8770))
            relay.set_leader_node(getattr(self, "leader_id", None))
            self._transport_cascade.register_transport(relay)

            # Try to add notification transports if credentials available
            self._init_notification_transports()

            self._log_debug(
                f"Transport cascade initialized with "
                f"{len(self._transport_cascade._transports)} transports"
            )

        except ImportError as e:
            self._log_debug(f"Transport cascade not available: {e}")
        except Exception as e:
            self._log_error(f"Transport cascade init failed: {e}")

    def _init_notification_transports(self) -> None:
        """Initialize notification transports (Slack, Telegram, etc.)."""
        if not self._transport_cascade:
            return

        try:
            from scripts.p2p.transports.notification_transports import (
                SlackWebhookTransport,
                DiscordWebhookTransport,
                TelegramTransport,
                EmailTransport,
                SMSTransport,
                PagerDutyTransport,
            )

            # Slack
            slack_webhook = os.environ.get("RINGRIFT_SLACK_WEBHOOK", "")
            if slack_webhook:
                self._transport_cascade.register_transport(
                    SlackWebhookTransport(webhook_url=slack_webhook)
                )

            # Discord
            discord_webhook = os.environ.get("RINGRIFT_DISCORD_WEBHOOK", "")
            if discord_webhook:
                self._transport_cascade.register_transport(
                    DiscordWebhookTransport(webhook_url=discord_webhook)
                )

            # Telegram
            telegram_token = os.environ.get("RINGRIFT_TELEGRAM_BOT_TOKEN", "")
            telegram_chat = os.environ.get("RINGRIFT_TELEGRAM_CHAT_ID", "")
            if telegram_token and telegram_chat:
                self._transport_cascade.register_transport(
                    TelegramTransport(bot_token=telegram_token, chat_id=telegram_chat)
                )

            # Email (SMTP)
            smtp_host = os.environ.get("RINGRIFT_SMTP_HOST", "")
            if smtp_host:
                self._transport_cascade.register_transport(
                    EmailTransport(
                        smtp_host=smtp_host,
                        smtp_port=int(os.environ.get("RINGRIFT_SMTP_PORT", "587")),
                        smtp_user=os.environ.get("RINGRIFT_SMTP_USER", ""),
                        smtp_password=os.environ.get("RINGRIFT_SMTP_PASSWORD", ""),
                        from_email=os.environ.get("RINGRIFT_SMTP_FROM", ""),
                        to_emails=os.environ.get("RINGRIFT_ALERT_EMAILS", "").split(","),
                    )
                )

            # SMS (Twilio)
            twilio_sid = os.environ.get("RINGRIFT_TWILIO_ACCOUNT_SID", "")
            if twilio_sid:
                self._transport_cascade.register_transport(
                    SMSTransport(
                        account_sid=twilio_sid,
                        auth_token=os.environ.get("RINGRIFT_TWILIO_AUTH_TOKEN", ""),
                        from_number=os.environ.get("RINGRIFT_TWILIO_FROM_NUMBER", ""),
                        to_numbers=os.environ.get("RINGRIFT_TWILIO_TO_NUMBERS", "").split(","),
                    )
                )

            # PagerDuty
            pagerduty_key = os.environ.get("RINGRIFT_PAGERDUTY_ROUTING_KEY", "")
            if pagerduty_key:
                self._transport_cascade.register_transport(
                    PagerDutyTransport(routing_key=pagerduty_key)
                )

        except ImportError:
            self._log_debug("Notification transports not available")
        except Exception as e:
            self._log_debug(f"Notification transports init failed: {e}")

    def _init_union_discovery(self) -> None:
        """Initialize union discovery for comprehensive peer finding."""
        try:
            from scripts.p2p.union_discovery import UnionDiscovery, get_union_discovery

            self._union_discovery = get_union_discovery()
            self._log_debug("Union discovery initialized")

        except ImportError as e:
            self._log_debug(f"Union discovery not available: {e}")
        except Exception as e:
            self._log_error(f"Union discovery init failed: {e}")

    def _init_protocol_union(self) -> None:
        """Initialize protocol union for combined membership."""
        try:
            from scripts.p2p.protocol_union import ProtocolUnion, get_protocol_union

            self._protocol_union = get_protocol_union(orchestrator=self)
            self._log_debug("Protocol union initialized")

        except ImportError as e:
            self._log_debug(f"Protocol union not available: {e}")
        except Exception as e:
            self._log_error(f"Protocol union init failed: {e}")

    async def send_with_failover(
        self,
        target: str,
        payload: bytes,
        timeout_per_transport: float = 10.0,
    ) -> "TransportResult | None":
        """Send payload using transport cascade with full failover.

        Args:
            target: Target node ID or address
            payload: Message bytes to send
            timeout_per_transport: Timeout per transport attempt

        Returns:
            TransportResult if any transport succeeded, None otherwise
        """
        if not self._failover_initialized:
            self._init_failover_system()

        if not self._transport_cascade:
            self._log_debug("Transport cascade not available, using direct HTTP")
            return None

        try:
            result = await self._transport_cascade.send_with_cascade(
                target=target,
                payload=payload,
                timeout_per_transport=timeout_per_transport,
            )

            if result.success:
                self._log_debug(
                    f"Failover send to {target} succeeded via {result.transport_name} "
                    f"({result.latency_ms:.0f}ms)"
                )
            else:
                self._log_debug(f"Failover send to {target} failed: {result.error}")

            return result

        except Exception as e:
            self._log_error(f"Failover send to {target} error: {e}")
            return None

    async def discover_all_peers_union(self) -> dict[str, Any]:
        """Discover peers using union of all discovery sources.

        Returns:
            Dict mapping node_id to DiscoveredPeer
        """
        if not self._failover_initialized:
            self._init_failover_system()

        if not self._union_discovery:
            return {}

        try:
            return await self._union_discovery.discover_all_peers()
        except Exception as e:
            self._log_error(f"Union discovery failed: {e}")
            return {}

    async def get_all_alive_peers_union(self) -> set[str]:
        """Get alive peers from union of all membership protocols.

        Returns:
            Set of node IDs that any protocol considers alive
        """
        if not self._failover_initialized:
            self._init_failover_system()

        if not self._protocol_union:
            # Fallback to basic peer list
            from scripts.p2p.handlers.handlers_utils import get_alive_peers
            return set(get_alive_peers(self.peers, self.peers_lock))

        try:
            return await self._protocol_union.get_all_alive_peers()
        except Exception as e:
            self._log_error(f"Protocol union get_alive failed: {e}")
            return set()

    async def refresh_peer_discovery(self) -> int:
        """Force refresh of peer discovery from all sources.

        Returns:
            Number of new peers discovered
        """
        if not self._failover_initialized:
            self._init_failover_system()

        discovered = await self.discover_all_peers_union()
        if not discovered:
            return 0

        # Add newly discovered peers to our peers dict
        new_count = 0
        for node_id, peer_info in discovered.items():
            if node_id == self.node_id:
                continue
            if node_id not in self.peers:
                # Create minimal NodeInfo for newly discovered peer
                try:
                    from scripts.p2p.models import NodeInfo, NodeRole

                    with self.peers_lock:
                        if node_id not in self.peers:
                            self.peers[node_id] = NodeInfo(
                                node_id=node_id,
                                host=peer_info.addresses[0] if peer_info.addresses else "",
                                port=getattr(self, "port", 8770),
                                role=NodeRole.FOLLOWER,
                            )
                            new_count += 1
                            self._log_debug(
                                f"Discovered new peer: {node_id} via union "
                                f"(confidence={peer_info.confidence:.2f})"
                            )
                except Exception as e:
                    self._log_debug(f"Failed to add discovered peer {node_id}: {e}")

        if new_count > 0:
            self._safe_emit_event(
                "PEERS_DISCOVERED",
                {
                    "source": "union_discovery",
                    "count": new_count,
                    "node_id": self.node_id,
                },
            )

        return new_count

    def update_relay_leader(self, leader_id: str | None) -> None:
        """Update the relay transport's leader node.

        Called when leadership changes.
        """
        if self._transport_cascade:
            for transport in self._transport_cascade._transports:
                if hasattr(transport, "set_leader_node"):
                    transport.set_leader_node(leader_id)

    def get_failover_stats(self) -> dict[str, Any]:
        """Get comprehensive failover system statistics."""
        stats: dict[str, Any] = {
            "enabled": FAILOVER_ENABLED,
            "initialized": self._failover_initialized,
        }

        if self._transport_cascade:
            stats["transport_cascade"] = {
                "transport_count": len(self._transport_cascade._transports),
                "transports": [t.name for t in self._transport_cascade._transports],
                "health": self._transport_cascade.get_all_health(),
            }

        if self._protocol_union:
            stats["protocol_union"] = self._protocol_union.get_stats()

        if self._union_discovery:
            stats["union_discovery"] = {
                "peer_count": len(self._union_discovery._known_peers),
            }

        return stats


# Helper function to check if failover is available
def is_failover_available() -> bool:
    """Check if the failover system components are available."""
    try:
        from scripts.p2p.transport_cascade import TransportCascade
        from scripts.p2p.union_discovery import UnionDiscovery
        from scripts.p2p.protocol_union import ProtocolUnion
        return True
    except ImportError:
        return False
