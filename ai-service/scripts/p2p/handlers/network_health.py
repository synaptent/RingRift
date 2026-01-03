"""Network Health Handler for P2P Orchestrator.

December 2025: Provides HTTP endpoints for cross-verifying P2P mesh health
against Tailscale connectivity.

Problem: P2P orchestrator shows 5-7 peers while Tailscale shows 40 online.
This handler exposes the discrepancy and provides manual reconnection.

Endpoints:
    GET /network/health - Cross-verification health report
    POST /network/reconnect - Force reconnection to missing peers

Usage:
    from scripts.p2p.handlers.network_health import NetworkHealthMixin

    class P2POrchestrator(NetworkHealthMixin, ...):
        pass

    # Routes are registered in setup_network_health_routes()
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from aiohttp import web

from .base import BaseP2PHandler
from .timeout_decorator import handler_timeout, HANDLER_TIMEOUT_GOSSIP

if TYPE_CHECKING:
    from scripts.p2p.loops.tailscale_discovery_loop import TailscalePeerDiscoveryLoop

logger = logging.getLogger(__name__)


class NetworkHealthMixin(BaseP2PHandler):
    """Mixin providing network health HTTP endpoints.

    Provides cross-verification between P2P mesh and Tailscale connectivity,
    allowing operators to identify and remediate peer discovery failures.

    Required attributes on implementing class:
        - peers: dict[str, PeerInfo] - Current P2P peers
        - _tailscale_discovery_loop: Optional TailscalePeerDiscoveryLoop
        - _get_tailscale_status() -> dict[str, bool]
        - _load_distributed_hosts() -> dict
        - reconnect_missing_peers() -> list[str]
    """

    # These will be set by the implementing class
    peers: dict
    _tailscale_discovery_loop: TailscalePeerDiscoveryLoop | None

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_network_health(self, request: web.Request) -> web.Response:
        """GET /network/health - Cross-verification health report.

        Returns:
            JSON response with:
            - tailscale_online: Number of peers online in Tailscale
            - p2p_connected: Number of peers connected in P2P
            - missing_from_p2p: Number missing from P2P
            - missing_peers: List of missing peer node IDs
            - health_score: P2P connected / Tailscale online (0.0 - 1.0)
            - status: "healthy" if score > 0.8, else "degraded"
        """
        try:
            # Get Tailscale status
            ts_status = await self._get_tailscale_status()
            ts_online_count = sum(1 for online in ts_status.values() if online)

            # Get P2P connected peers
            p2p_connected = 0
            p2p_peer_ids = set()
            for peer_id, peer_info in self.peers.items():
                is_alive = getattr(peer_info, "is_alive", lambda: True)
                if callable(is_alive) and is_alive():
                    p2p_connected += 1
                    p2p_peer_ids.add(peer_id)
                elif not callable(is_alive) and is_alive:
                    p2p_connected += 1
                    p2p_peer_ids.add(peer_id)

            # Get configured hosts to find missing peers
            config_hosts = self._load_distributed_hosts().get("hosts", {})
            ip_to_node = {
                h.get("tailscale_ip"): name
                for name, h in config_hosts.items()
                if h.get("tailscale_ip") and h.get("p2p_enabled", True)
            }

            # Find missing peers
            missing_peers = []
            for ts_ip, is_online in ts_status.items():
                if is_online and ts_ip in ip_to_node:
                    node_name = ip_to_node[ts_ip]
                    if node_name not in p2p_peer_ids:
                        missing_peers.append(node_name)

            # Calculate health score
            health_score = p2p_connected / max(ts_online_count, 1)

            # Get discovery loop stats if available
            discovery_stats = None
            if hasattr(self, "_tailscale_discovery_loop") and self._tailscale_discovery_loop:
                discovery_stats = self._tailscale_discovery_loop.stats.to_dict()

            return self.json_response({
                "timestamp": time.time(),
                "tailscale_online": ts_online_count,
                "p2p_connected": p2p_connected,
                "missing_from_p2p": len(missing_peers),
                "missing_peers": sorted(missing_peers),
                "health_score": round(health_score, 3),
                "status": "healthy" if health_score > 0.8 else "degraded",
                "node_id": self.node_id,
                "discovery_loop": discovery_stats,
            })

        except Exception as e:
            logger.exception("Error generating network health report")
            return self.error_response(f"Health check failed: {e}", status=500)

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_network_reconnect(self, request: web.Request) -> web.Response:
        """POST /network/reconnect - Force reconnection to missing peers.

        Triggers immediate reconnection attempt to all peers that are online
        in Tailscale but missing from the P2P mesh.

        Returns:
            JSON response with:
            - attempted: Number of reconnection attempts
            - reconnected: List of successfully reconnected peers
            - failed: List of peers that failed to reconnect
        """
        try:
            # Parse optional body for selective reconnect
            node_filter = None
            try:
                body = await self.parse_json_body(request)
                if body and "nodes" in body:
                    node_filter = set(body["nodes"])
            except Exception:
                pass  # No body is fine

            # Get missing peers
            ts_status = await self._get_tailscale_status()
            config_hosts = self._load_distributed_hosts().get("hosts", {})

            ip_to_node = {
                h.get("tailscale_ip"): (name, h)
                for name, h in config_hosts.items()
                if h.get("tailscale_ip") and h.get("p2p_enabled", True)
            }

            # Get current P2P peer IDs
            p2p_peer_ids = set()
            for peer_id, peer_info in self.peers.items():
                is_alive = getattr(peer_info, "is_alive", lambda: True)
                if (callable(is_alive) and is_alive()) or (not callable(is_alive) and is_alive):
                    p2p_peer_ids.add(peer_id)

            # Find and reconnect missing peers
            attempted = []
            reconnected = []
            failed = []

            for ts_ip, is_online in ts_status.items():
                if not is_online:
                    continue

                if ts_ip not in ip_to_node:
                    continue

                node_name, node_config = ip_to_node[ts_ip]

                # Skip if already connected
                if node_name in p2p_peer_ids:
                    continue

                # Skip if filter is set and node not in filter
                if node_filter and node_name not in node_filter:
                    continue

                attempted.append(node_name)
                port = node_config.get("p2p_port", 8770)

                try:
                    success = await self._reconnect_discovered_peer(node_name, ts_ip, port)
                    if success:
                        reconnected.append(node_name)
                    else:
                        failed.append(node_name)
                except Exception as e:
                    logger.warning(f"Failed to reconnect {node_name}: {e}")
                    failed.append(node_name)

            return self.json_response({
                "timestamp": time.time(),
                "attempted": len(attempted),
                "attempted_peers": attempted,
                "reconnected": reconnected,
                "failed": failed,
                "success_rate": len(reconnected) / max(len(attempted), 1),
            })

        except Exception as e:
            logger.exception("Error during network reconnect")
            return self.error_response(f"Reconnect failed: {e}", status=500)

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_network_status(self, request: web.Request) -> web.Response:
        """GET /network/status - Detailed network status for debugging.

        Returns comprehensive status including:
        - All Tailscale peers and their online status
        - All P2P peers and their state
        - Configuration mapping
        """
        try:
            # Get Tailscale status
            ts_status = await self._get_tailscale_status()

            # Get config hosts
            config_hosts = self._load_distributed_hosts().get("hosts", {})

            # Build detailed peer info
            peer_details = []
            for node_name, node_config in config_hosts.items():
                ts_ip = node_config.get("tailscale_ip")
                if not ts_ip:
                    continue

                ts_online = ts_status.get(ts_ip, False)

                # Check P2P status
                p2p_status = "unknown"
                p2p_last_seen = None
                if node_name in self.peers:
                    peer_info = self.peers[node_name]
                    is_alive = getattr(peer_info, "is_alive", lambda: False)
                    if callable(is_alive):
                        p2p_status = "alive" if is_alive() else "dead"
                    else:
                        p2p_status = "alive" if is_alive else "dead"
                    p2p_last_seen = getattr(peer_info, "last_heartbeat", None)
                else:
                    p2p_status = "not_connected"

                peer_details.append({
                    "node_id": node_name,
                    "tailscale_ip": ts_ip,
                    "tailscale_online": ts_online,
                    "p2p_status": p2p_status,
                    "p2p_last_seen": p2p_last_seen,
                    "provider": node_config.get("provider"),
                    "gpu": node_config.get("gpu"),
                })

            # Summary stats
            ts_online_count = sum(1 for p in peer_details if p["tailscale_online"])
            p2p_alive_count = sum(1 for p in peer_details if p["p2p_status"] == "alive")
            missing_count = sum(
                1 for p in peer_details
                if p["tailscale_online"] and p["p2p_status"] != "alive"
            )

            return self.json_response({
                "timestamp": time.time(),
                "summary": {
                    "total_configured": len(peer_details),
                    "tailscale_online": ts_online_count,
                    "p2p_alive": p2p_alive_count,
                    "missing_from_p2p": missing_count,
                    "health_score": round(p2p_alive_count / max(ts_online_count, 1), 3),
                },
                "peers": sorted(peer_details, key=lambda p: p["node_id"]),
            })

        except Exception as e:
            logger.exception("Error generating network status")
            return self.error_response(f"Status failed: {e}", status=500)


def setup_network_health_routes(app: web.Application, handler: NetworkHealthMixin) -> None:
    """Register network health routes with the application.

    Args:
        app: aiohttp web application
        handler: Handler instance with network health methods
    """
    app.router.add_get("/network/health", handler.handle_network_health)
    app.router.add_post("/network/reconnect", handler.handle_network_reconnect)
    app.router.add_get("/network/status", handler.handle_network_status)
    logger.info("Registered network health routes: /network/health, /network/reconnect, /network/status")
