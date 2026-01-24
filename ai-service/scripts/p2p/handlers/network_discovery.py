"""Network Discovery Handler Mixin (December 2025).

Provides network discovery and IP management methods for the P2P orchestrator.
Extracted to reduce p2p_orchestrator.py complexity and improve testability.
Inherits from BaseP2PHandler for consistent response formatting.

This mixin handles:
- Local IP and Tailscale IP detection
- Network partition detection and recovery
- Tailscale priority mode for partition recovery
- HTTP connectivity diagnostics endpoints

Usage:
    class P2POrchestrator(NetworkDiscoveryMixin, ...):
        pass

Requires implementing class to provide:
    - peers: dict[str, NodeInfo]
    - peers_lock: threading.RLock
    - node_id: str
    - self_info: NodeInfo
    - _is_tailscale_host(host: str) -> bool
    - hybrid_transport: HybridTransport | None
"""

from __future__ import annotations

import logging
import socket
import subprocess
import time
from typing import TYPE_CHECKING, Any

from aiohttp import web

from scripts.p2p.handlers.base import BaseP2PHandler
from scripts.p2p.handlers.timeout_decorator import handler_timeout, HANDLER_TIMEOUT_GOSSIP

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# January 2026: Use centralized timeouts from loop_constants
try:
    from scripts.p2p.loops.loop_constants import LoopTimeouts
    PEER_TIMEOUT = LoopTimeouts.PEER_DEAD_TIMEOUT  # 90.0 seconds (matches app/p2p/constants.py)
except ImportError:
    PEER_TIMEOUT = 90.0  # Fallback - peer considered dead after this (matches PEER_TIMEOUT)


class NetworkDiscoveryMixin(BaseP2PHandler):
    """Mixin providing network discovery and IP management.

    Inherits from BaseP2PHandler for consistent response formatting.
    This mixin contains methods for detecting network topology,
    managing Tailscale connectivity, and diagnosing connectivity issues.
    """

    # ===========================================================================
    # Local IP Discovery
    # ===========================================================================

    def _get_local_ip(self) -> str:
        """Get local IP address via socket probe.

        Returns:
            Local IPv4 address, or "127.0.0.1" if detection fails.
        """
        try:
            # Connect to external address to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except OSError:
            return "127.0.0.1"

    def _get_tailscale_ip(self) -> str:
        """Return this node's Tailscale IPv4 (100.x) when available.

        Returns:
            Tailscale IP address, or empty string if unavailable.
        """
        try:
            result = subprocess.run(
                ["tailscale", "ip", "-4"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return ""
            ip = (result.stdout or "").strip().splitlines()[0].strip()
            return ip
        except FileNotFoundError:
            return ""
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, OSError, KeyError, IndexError, AttributeError):
            return ""

    def _local_has_tailscale(self) -> bool:
        """Check if this node has a Tailscale address.

        Returns:
            True if node appears to have Tailscale connectivity.
        """
        try:
            info = getattr(self, "self_info", None)
            if not info:
                return False
            host = str(getattr(info, "host", "") or "").strip()
            reported_host = str(getattr(info, "reported_host", "") or "").strip()
            return self._is_tailscale_host(host) or self._is_tailscale_host(reported_host)
        except AttributeError:
            return False

    # ===========================================================================
    # Network Partition Detection
    # ===========================================================================

    def _detect_network_partition(self) -> bool:
        """Detect if we're in a network partition (>50% peers unreachable).

        Used to trigger Tailscale-first connectivity mode when the public
        network is fragmented but mesh connectivity remains intact.

        Returns:
            True if partition detected (majority of peers unreachable).
        """
        with self.peers_lock:
            peers_snapshot = [p for p in self.peers.values() if p.node_id != self.node_id]

        if len(peers_snapshot) < 2:
            return False

        # Count peers with recent heartbeat failures
        now = time.time()
        unreachable = 0
        for peer in peers_snapshot:
            if peer.consecutive_failures >= 3 or (now - peer.last_heartbeat > PEER_TIMEOUT):
                unreachable += 1

        partition_ratio = unreachable / len(peers_snapshot)
        if partition_ratio > 0.5:
            logger.info(
                f"Network partition detected: {unreachable}/{len(peers_snapshot)} "
                f"peers unreachable ({partition_ratio:.0%})"
            )
            return True
        return False

    # ===========================================================================
    # Tailscale Priority Mode
    # ===========================================================================

    def _get_tailscale_priority_mode(self) -> bool:
        """Check if Tailscale-first mode is enabled (partition recovery).

        Returns:
            True if Tailscale should be preferred over direct connectivity.
        """
        return getattr(self, "_tailscale_priority", False)

    def _enable_tailscale_priority(self) -> None:
        """Enable Tailscale-first mode for heartbeats during partition recovery.

        Mode expires after 5 minutes automatically.
        """
        if not getattr(self, "_tailscale_priority", False):
            logger.info("Enabling Tailscale-priority mode for partition recovery")
            self._tailscale_priority = True
            self._tailscale_priority_until = time.time() + 300  # 5 minutes

    def _disable_tailscale_priority(self) -> None:
        """Disable Tailscale-first mode when connectivity recovers."""
        if getattr(self, "_tailscale_priority", False):
            logger.info("Disabling Tailscale-priority mode (connectivity recovered)")
            self._tailscale_priority = False

    # ===========================================================================
    # HTTP Handlers
    # ===========================================================================

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_connectivity_diagnose(self, request: web.Request) -> web.Response:
        """Diagnose connectivity to a specific node.

        POST /connectivity/diagnose
        {
            "node_id": "target-node-id"
        }

        Returns detailed connectivity diagnostics including:
        - Direct HTTP reachability
        - Tailscale reachability
        - SSH availability
        - Cloudflare tunnel status
        """
        try:
            data = await request.json()
            target_node = data.get("node_id", "")
            if not target_node:
                return self.error_response("node_id required", status=400)

            # Check if hybrid transport is available
            hybrid = getattr(self, "hybrid_transport", None)
            if not hybrid:
                return self.json_response({
                    "node_id": target_node,
                    "error": "Hybrid transport not available",
                    "available_transports": [],
                })

            # Run diagnostics
            results = await hybrid.diagnose_connectivity(target_node)
            return self.json_response({
                "node_id": target_node,
                "diagnostics": results,
            })

        except Exception as e:
            logger.warning(f"Connectivity diagnosis failed: {e}")
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_network_status(self, request: web.Request) -> web.Response:
        """Get network topology status.

        GET /network/status

        Returns:
        - Partition detection status
        - Tailscale priority mode
        - Peer reachability summary
        """
        try:
            partition_detected = self._detect_network_partition()
            tailscale_priority = self._get_tailscale_priority_mode()
            local_tailscale = self._local_has_tailscale()
            local_ip = self._get_local_ip()
            tailscale_ip = self._get_tailscale_ip()

            # Count peer reachability
            with self.peers_lock:
                total_peers = len([p for p in self.peers.values() if p.node_id != self.node_id])
                now = time.time()
                unreachable = sum(
                    1 for p in self.peers.values()
                    if p.node_id != self.node_id and (
                        p.consecutive_failures >= 3 or (now - p.last_heartbeat > PEER_TIMEOUT)
                    )
                )

            return self.json_response({
                "local_ip": local_ip,
                "tailscale_ip": tailscale_ip,
                "has_tailscale": local_tailscale,
                "partition_detected": partition_detected,
                "tailscale_priority_mode": tailscale_priority,
                "total_peers": total_peers,
                "unreachable_peers": unreachable,
                "reachable_peers": total_peers - unreachable,
            })

        except Exception as e:
            logger.warning(f"Network status check failed: {e}")
            return self.error_response(str(e), status=500)
