"""Peer discovery, Tailscale reconnect, voter prepopulation, and partition mode helpers.

April 2026: Extracted from p2p_orchestrator.py (Part 3 Phase 2).
"""
from __future__ import annotations

from scripts.p2p.p2p_mixin_base import P2PMixinBase
from scripts.p2p.startup_infrastructure import *  # noqa: F401,F403


class PeerDiscoveryMixin(P2PMixinBase):
    """Mixin for P2POrchestrator peer discovery, tailscale reconnect, voter prepopulation, and partition mode helpers."""

    MIXIN_TYPE = "peer_discovery"

    def _prepopulate_voter_peers(self) -> None:
        """Pre-populate voter nodes into peers dict for immediate gossip reachability.

        Jan 28, 2026: Fixes bootstrap chicken-and-egg where voters are invisible
        to gossip until discovered via heartbeat. Without this, new nodes have an
        empty peers dict → gossip can't reach voters → voters never get added.
        """
        if not self.voter_node_ids:
            return

        if os.environ.get("RINGRIFT_SKIP_VOTER_PREPOPULATION", "").lower() in ("1", "true"):
            logger.info("[P2P] Voter pre-population disabled via env var")
            return

        try:
            from app.config.cluster_config import get_cluster_nodes
            cluster_nodes = get_cluster_nodes()
        except ImportError:
            logger.warning("[P2P] Cannot pre-populate voters: cluster_config unavailable")
            return
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[P2P] Cannot pre-populate voters: {e}")
            return

        prepopulated = 0
        for voter_id in self.voter_node_ids:
            if voter_id == self.node_id:
                continue  # Skip self

            if voter_id in self.peers:
                continue  # Already known

            node_cfg = cluster_nodes.get(voter_id)
            if not node_cfg:
                logger.debug(f"[P2P] Voter {voter_id} not in cluster_config, skipping prepopulation")
                continue

            host = getattr(node_cfg, 'best_ip', None) or getattr(node_cfg, 'tailscale_ip', None)
            if not host:
                logger.debug(f"[P2P] Voter {voter_id} has no IP in cluster_config, skipping")
                continue

            voter_info = NodeInfo(
                node_id=voter_id,
                host=host,
                port=DEFAULT_PORT,
                tailscale_ip=getattr(node_cfg, 'tailscale_ip', '') or '',
                role=NodeRole.FOLLOWER,
                last_heartbeat=0,  # Will update on first heartbeat
            )
            self.peers[voter_id] = voter_info
            prepopulated += 1
            logger.debug(f"[P2P] Pre-populated voter {voter_id} at {host}:{DEFAULT_PORT}")

        if prepopulated:
            self._publish_peers_snapshot()
            logger.info(f"[P2P] Pre-populated {prepopulated} voter peers for gossip reachability")

    def _cache_local_ips(self) -> set[str]:
        """Cache all local IPs at startup to avoid DNS blocking in health endpoints.

        Jan 29, 2026: Delegate to PeerNetworkOrchestrator if available,
        otherwise inline basic IP detection (called during early __init__
        before self.network is set).

        Jan 26, 2026: Called once at initialization and cached.
        """
        if hasattr(self, "network") and self.network is not None:
            return self.network.cache_local_ips()

        # Fallback: inline basic IP detection for early __init__ call
        import socket
        import subprocess

        local_ips: set[str] = set()
        try:
            hostname = socket.gethostname()
            for addr in socket.getaddrinfo(hostname, None):
                local_ips.add(addr[4][0])
        except (socket.gaierror, socket.herror, OSError, UnicodeError):
            pass
        try:
            for addr in socket.getaddrinfo("localhost", None):
                local_ips.add(addr[4][0])
        except (socket.gaierror, socket.herror, OSError):
            pass
        try:
            result = subprocess.run(
                ["hostname", "-I"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for ip in result.stdout.strip().split():
                    local_ips.add(ip.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        # Always include loopback
        local_ips.add("127.0.0.1")
        local_ips.add("::1")
        if hasattr(self, "advertise_host") and self.advertise_host:
            local_ips.add(self.advertise_host)
        if hasattr(self, "tailscale_ip") and self.tailscale_ip:
            local_ips.add(self.tailscale_ip)
        return local_ips

    def _sync_peer_snapshot(self) -> None:
        """Synchronize PeerSnapshot with current peers dictionary.

        January 12, 2026: Added for lock-free reads in handle_status.
        Call this after any operation that modifies self.peers.

        This uses bulk_update for efficiency when there are many peers.
        The PeerSnapshot will be atomically updated with the current state.
        """
        try:
            # Use bulk update for efficiency - single lock acquisition, single snapshot refresh
            with self._peer_snapshot.bulk_update():
                # Clear and repopulate (handles removes and updates)
                self._peer_snapshot.clear()
                for node_id, info in self.peers.items():
                    self._peer_snapshot.update_peer(node_id, info)
        except Exception as e:  # noqa: BLE001
            # Log but don't fail - reads will use stale snapshot
            logger.warning(f"[PeerSnapshot] Sync failed: {e}")

    def _local_has_tailscale(self) -> bool:
        """Best-effort: True when this node appears to have a Tailscale address."""
        try:
            info = getattr(self, "self_info", None)
            if not info:
                return False
            host = str(getattr(info, "host", "") or "").strip()
            reported_host = str(getattr(info, "reported_host", "") or "").strip()
            return self._is_tailscale_host(host) or self._is_tailscale_host(reported_host)
        except (AttributeError):
            return False

    async def _get_tailscale_status(self) -> dict[str, bool]:
        """Jan 29, 2026: Delegated to PeerNetworkOrchestrator.get_tailscale_status()."""
        return await self.network.get_tailscale_status()

    async def _reconnect_discovered_peer(
        self, node_id: str, host: str, port: int
    ) -> bool:
        """Attempt to reconnect to a peer discovered via Tailscale.

        Probes the peer's health endpoint and sends a heartbeat to establish
        P2P connection.

        Args:
            node_id: Peer node identifier
            host: Tailscale IP address
            port: P2P port (usually 8770)

        Returns:
            True if reconnection successful, False otherwise
        """
        try:
            # Probe health endpoint
            url = f"http://{host}:{port}/health"
            timeout = ClientTimeout(total=5)
            async with get_client_session(timeout) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return False
                    data, error = await safe_json_response(resp, default={}, log_errors=False)
                    if error:
                        return False

            # Extract node_id from response if available
            actual_node_id = data.get("node_id", node_id)

            # Send heartbeat to establish connection
            await self._send_heartbeat_to_peer(host, port)

            # Check if peer is now in our peers dict
            async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
                if actual_node_id not in self.peers or not self.peers[actual_node_id].is_alive():
                    # Register the peer
                    self.peers[actual_node_id] = NodeInfo(
                        node_id=actual_node_id,
                        host=host,
                        port=port,
                        last_heartbeat=time.time(),
                        state="alive",
                    )
                    # C2 fix: Sync peer snapshot after adding new peer
                    self._sync_peer_snapshot()
                    self._publish_peers_snapshot()
                    logger.info(f"Reconnected peer via network health: {actual_node_id} ({host}:{port})")
                    await self._emit_host_online(actual_node_id)
                    return True

            return True  # Already connected

        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to reconnect {node_id}: {e}")
            return False

    async def reconnect_missing_peers(self) -> list[str]:
        """Reconnect to all peers that are online in Tailscale but not in P2P.

        Returns:
            List of node IDs that were successfully reconnected
        """
        ts_peers = await self._get_tailscale_status()
        config_hosts = self._load_distributed_hosts().get("hosts", {})

        # Build IP to node mapping
        ip_to_node: dict[str, tuple[str, dict]] = {}
        for name, h in config_hosts.items():
            ts_ip = h.get("tailscale_ip")
            if ts_ip and h.get("p2p_enabled", True):
                ip_to_node[ts_ip] = (name, h)

        # Get current alive peer IDs
        # Jan 2026: Use lock-free PeerSnapshot for read-only access
        current_ids: set[str] = set()
        for peer in self._peer_snapshot.get_snapshot().values():
            if peer.is_alive():
                current_ids.add(peer.node_id)

        # Find and reconnect missing peers
        reconnected: list[str] = []
        for ts_ip, is_online in ts_peers.items():
            if not is_online:
                continue

            if ts_ip not in ip_to_node:
                continue

            node_id, node_config = ip_to_node[ts_ip]

            # Skip if already connected
            if node_id in current_ids:
                continue

            # Skip self
            if node_id == self.node_id:
                continue

            # Attempt reconnection
            port = node_config.get("p2p_port", DEFAULT_PORT)
            if await self._reconnect_discovered_peer(node_id, ts_ip, port):
                reconnected.append(node_id)

        if reconnected:
            logger.info(f"Reconnected {len(reconnected)} missing peers: {reconnected}")

        return reconnected

    def _check_partition_mode(self) -> None:
        """Check partition status and enable/disable read-only mode.

        December 2025 (Phase 2.4): Prevent data divergence during network partitions.

        When this node is in a minority partition (<50% of peers alive):
        - Pause training job dispatch
        - Pause selfplay job dispatch
        - Continue serving existing data (read-only)
        - Allow sync operations to help recovery

        This prevents split-brain scenarios where both partitions continue
        generating training data that later conflicts during merge.
        """
        now = time.time()

        # Rate limit partition checks
        if now - self._last_partition_check < self._partition_check_interval:
            return
        self._last_partition_check = now

        # Use gossip protocol's partition detection
        status, ratio = self.detect_partition_status()

        if status in ("minority", "isolated"):
            if not self._partition_readonly_mode:
                logger.warning(
                    f"[P2P] Entering partition read-only mode: "
                    f"status={status}, health_ratio={ratio:.2%}"
                )
                self._partition_readonly_mode = True
                self._partition_readonly_since = now

                # Emit event for monitoring
                self._safe_emit_event("PARTITION_READONLY_ENTERED", {
                    "node_id": self.node_id,
                    "status": status,
                    "health_ratio": ratio,
                    "timestamp": now,
                })
        else:
            if self._partition_readonly_mode:
                readonly_duration = now - self._partition_readonly_since
                logger.info(
                    f"[P2P] Exiting partition read-only mode: "
                    f"status={status}, health_ratio={ratio:.2%}, "
                    f"was_readonly_for={readonly_duration:.0f}s"
                )
                self._partition_readonly_mode = False
                self._partition_readonly_since = 0.0

                # Emit event for monitoring
                self._safe_emit_event("PARTITION_READONLY_EXITED", {
                    "node_id": self.node_id,
                    "status": status,
                    "health_ratio": ratio,
                    "readonly_duration_seconds": readonly_duration,
                    "timestamp": now,
                })

    def is_partition_readonly(self) -> bool:
        """Check if this node is in partition read-only mode.

        December 2025 (Phase 2.4): Query method for dispatch gates.

        Returns:
            True if job dispatch should be paused due to partition status.
        """
        # Do a fresh check if it's been a while
        self._check_partition_mode()
        return self._partition_readonly_mode

    def get_partition_status(self) -> dict[str, Any]:
        """Get current partition status details.

        December 2025 (Phase 2.4): Status API for monitoring/debugging.

        Returns:
            Dict with partition status, mode, and duration.
        """
        status, ratio = self.detect_partition_status()
        now = time.time()

        result = {
            "partition_status": status,
            "health_ratio": round(ratio, 3),
            "readonly_mode": self._partition_readonly_mode,
            "readonly_since": self._partition_readonly_since,
            "readonly_duration_seconds": (
                now - self._partition_readonly_since
                if self._partition_readonly_mode else 0.0
            ),
            "last_check": self._last_partition_check,
        }

        # Add detailed peer info if available
        if hasattr(self, "get_partition_details"):
            result["details"] = self.get_partition_details()

        return result
