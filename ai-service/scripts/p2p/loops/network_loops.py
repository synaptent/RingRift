"""Network Management Loops for P2P Orchestrator.

December 2025: Background loops for network health and recovery.

Loops:
- IpDiscoveryLoop: Updates node IP addresses as they change
- TailscaleRecoveryLoop: Recovers Tailscale connections when they fail
- TailscalePeerDiscoveryLoop: Discovers and connects to Tailscale peers not in P2P network
- NATManagementLoop: Manages NAT traversal and relay selection

Usage:
    from scripts.p2p.loops import IpDiscoveryLoop, TailscalePeerDiscoveryLoop

    ip_loop = IpDiscoveryLoop(
        get_nodes=lambda: orchestrator.peer_status,
        update_node_ip=orchestrator.update_node_ip,
    )
    await ip_loop.run_forever()

    # Peer discovery for cross-cloud connectivity
    discovery_loop = TailscalePeerDiscoveryLoop(
        is_leader=lambda: orchestrator.role == NodeRole.LEADER,
        get_current_peers=lambda: {p.node_id for p in orchestrator.peers.values()},
        get_alive_peer_count=lambda: sum(1 for p in orchestrator.peers.values() if p.is_alive()),
        probe_and_connect=orchestrator.probe_and_register_peer,
    )
    await discovery_loop.run_forever()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

from .base import BaseLoop

logger = logging.getLogger(__name__)


@dataclass
class IpDiscoveryConfig:
    """Configuration for IP discovery loop."""

    check_interval_seconds: float = 300.0  # 5 minutes
    dns_timeout_seconds: float = 10.0
    max_nodes_per_cycle: int = 20

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        if self.dns_timeout_seconds <= 0:
            raise ValueError("dns_timeout_seconds must be > 0")
        if self.max_nodes_per_cycle <= 0:
            raise ValueError("max_nodes_per_cycle must be > 0")


class IpDiscoveryLoop(BaseLoop):
    """Background loop that discovers and updates node IP addresses.

    Handles:
    - Nodes with dynamic IPs (cloud instances)
    - Tailscale IP changes
    - Fallback to alternative IPs when primary fails
    """

    def __init__(
        self,
        get_nodes: Callable[[], dict[str, dict[str, Any]]],
        update_node_ip: Callable[[str, str], Coroutine[Any, Any, None]],
        config: IpDiscoveryConfig | None = None,
    ):
        """Initialize IP discovery loop.

        Args:
            get_nodes: Callback returning node_id -> node_info dict
            update_node_ip: Async callback to update a node's IP
            config: Discovery configuration
        """
        self.config = config or IpDiscoveryConfig()
        super().__init__(
            name="ip_discovery",
            interval=self.config.check_interval_seconds,
        )
        self._get_nodes = get_nodes
        self._update_node_ip = update_node_ip
        self._updates_count = 0

    async def _run_once(self) -> None:
        """Check for IP changes on all nodes."""
        nodes = self._get_nodes()
        if not nodes:
            return

        # Process nodes in batches
        node_list = list(nodes.items())[:self.config.max_nodes_per_cycle]

        for node_id, node_info in node_list:
            try:
                current_ip = node_info.get("ip") or node_info.get("host")
                tailscale_ip = node_info.get("tailscale_ip")
                public_ip = node_info.get("public_ip")

                # Try to resolve hostname if present
                hostname = node_info.get("hostname")
                if hostname and hostname != current_ip:
                    new_ip = await self._resolve_hostname(hostname)
                    if new_ip and new_ip != current_ip:
                        await self._update_node_ip(node_id, new_ip)
                        self._updates_count += 1
                        logger.info(f"[IpDiscovery] Updated {node_id}: {current_ip} -> {new_ip}")
                        continue

                # Check if current IP is reachable, try alternatives if not
                if current_ip and not await self._is_reachable(current_ip):
                    # Try tailscale first
                    if tailscale_ip and await self._is_reachable(tailscale_ip):
                        await self._update_node_ip(node_id, tailscale_ip)
                        self._updates_count += 1
                        logger.info(f"[IpDiscovery] Switched {node_id} to Tailscale: {tailscale_ip}")
                    # Then try public IP
                    elif public_ip and await self._is_reachable(public_ip):
                        await self._update_node_ip(node_id, public_ip)
                        self._updates_count += 1
                        logger.info(f"[IpDiscovery] Switched {node_id} to public IP: {public_ip}")

            except Exception as e:
                logger.debug(f"[IpDiscovery] Failed to check {node_id}: {e}")

    async def _resolve_hostname(self, hostname: str) -> str | None:
        """Resolve hostname to IP address."""
        try:
            import socket
            # Dec 2025: Use get_running_loop() in async context
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, socket.gethostbyname, hostname),
                timeout=self.config.dns_timeout_seconds,
            )
            return result
        except (OSError, asyncio.TimeoutError) as e:
            logger.debug(f"[NetworkDiscovery] DNS resolution failed for {hostname}: {e}")
            return None

    async def _is_reachable(self, ip: str, port: int = 22) -> bool:
        """Quick check if IP is reachable via TCP."""
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, port),
                timeout=5.0,
            )
            writer.close()
            await writer.wait_closed()
            return True
        except (ConnectionError, asyncio.TimeoutError, OSError) as e:
            logger.debug(f"[NetworkDiscovery] Reachability check failed for {ip}:{port}: {e}")
            return False

    def get_discovery_stats(self) -> dict[str, Any]:
        """Get discovery statistics."""
        return {
            "total_updates": self._updates_count,
            **self.stats.to_dict(),
        }


@dataclass
class TailscaleRecoveryConfig:
    """Configuration for Tailscale recovery loop."""

    check_interval_seconds: float = 120.0  # 2 minutes
    recovery_timeout_seconds: float = 60.0
    max_recovery_attempts: int = 3
    cooldown_after_recovery_seconds: float = 300.0  # 5 minutes

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        if self.recovery_timeout_seconds <= 0:
            raise ValueError("recovery_timeout_seconds must be > 0")
        if self.max_recovery_attempts <= 0:
            raise ValueError("max_recovery_attempts must be > 0")
        if self.cooldown_after_recovery_seconds < 0:
            raise ValueError("cooldown_after_recovery_seconds must be >= 0")


class TailscaleRecoveryLoop(BaseLoop):
    """Background loop that monitors and recovers Tailscale connections.

    Detects when Tailscale connectivity fails and attempts recovery via:
    1. Restarting tailscaled service
    2. Re-authenticating with auth key
    3. Falling back to direct IPs
    """

    def __init__(
        self,
        get_tailscale_status: Callable[[], dict[str, Any]],
        run_ssh_command: Callable[[str, str], Coroutine[Any, Any, Any]],
        on_recovery_failed: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        config: TailscaleRecoveryConfig | None = None,
    ):
        """Initialize Tailscale recovery loop.

        Args:
            get_tailscale_status: Callback returning Tailscale status dict
            run_ssh_command: Async callback to run SSH command on node
            on_recovery_failed: Optional callback when recovery fails
            config: Recovery configuration
        """
        self.config = config or TailscaleRecoveryConfig()
        super().__init__(
            name="tailscale_recovery",
            interval=self.config.check_interval_seconds,
        )
        self._get_tailscale_status = get_tailscale_status
        self._run_ssh_command = run_ssh_command
        self._on_recovery_failed = on_recovery_failed
        self._recovery_attempts: dict[str, int] = {}
        self._last_recovery: dict[str, float] = {}
        self._recoveries_count = 0

    async def _run_once(self) -> None:
        """Check Tailscale health and recover if needed."""
        status = self._get_tailscale_status()
        if not status:
            return

        now = time.time()
        nodes_needing_recovery = []

        # Check each node's Tailscale status
        for node_id, node_status in status.items():
            ts_state = node_status.get("tailscale_state", "unknown")
            ts_online = node_status.get("tailscale_online", True)

            if ts_state in ("stopped", "offline") or not ts_online:
                # Check cooldown
                last_attempt = self._last_recovery.get(node_id, 0)
                if now - last_attempt < self.config.cooldown_after_recovery_seconds:
                    continue

                # Check attempt limit
                attempts = self._recovery_attempts.get(node_id, 0)
                if attempts >= self.config.max_recovery_attempts:
                    continue

                nodes_needing_recovery.append(node_id)

        # Attempt recovery on each node
        for node_id in nodes_needing_recovery:
            success = await self._attempt_recovery(node_id)
            self._last_recovery[node_id] = now

            if success:
                self._recovery_attempts[node_id] = 0
                self._recoveries_count += 1
                logger.info(f"[TailscaleRecovery] Successfully recovered {node_id}")
            else:
                self._recovery_attempts[node_id] = self._recovery_attempts.get(node_id, 0) + 1
                if self._recovery_attempts[node_id] >= self.config.max_recovery_attempts:
                    logger.warning(f"[TailscaleRecovery] Max attempts reached for {node_id}")
                    if self._on_recovery_failed:
                        try:
                            await self._on_recovery_failed(node_id)
                        except Exception as e:
                            logger.error(f"[TailscaleRecovery] Failed callback error: {e}")

    async def _attempt_recovery(self, node_id: str) -> bool:
        """Attempt to recover Tailscale on a node."""
        logger.info(f"[TailscaleRecovery] Attempting recovery on {node_id}")

        try:
            # Step 1: Try restarting tailscaled
            result = await self._run_ssh_command(
                node_id,
                "sudo systemctl restart tailscaled && sleep 5 && tailscale status --json",
            )

            if result.returncode == 0:
                # Check if it came back online
                import json
                try:
                    status = json.loads(result.stdout)
                    if status.get("Self", {}).get("Online", False):
                        return True
                except json.JSONDecodeError:
                    pass

            # Step 2: Try tailscale up with reset
            result = await self._run_ssh_command(
                node_id,
                "tailscale up --reset --accept-routes",
            )

            if result.returncode == 0:
                await asyncio.sleep(5)  # Wait for connection
                # Verify
                result = await self._run_ssh_command(node_id, "tailscale status --json")
                if result.returncode == 0:
                    return True

            return False

        except Exception as e:
            logger.warning(f"[TailscaleRecovery] Recovery failed on {node_id}: {e}")
            return False

    def get_recovery_stats(self) -> dict[str, Any]:
        """Get recovery statistics."""
        return {
            "total_recoveries": self._recoveries_count,
            "nodes_at_max_attempts": len([
                n for n, a in self._recovery_attempts.items()
                if a >= self.config.max_recovery_attempts
            ]),
            "pending_recoveries": dict(self._recovery_attempts),
            **self.stats.to_dict(),
        }


@dataclass
class NATManagementConfig:
    """Configuration for NAT management loop."""

    check_interval_seconds: float = 60.0  # NAT_BLOCKED_PROBE_INTERVAL
    stun_probe_interval_seconds: float = 300.0  # NAT_STUN_LIKE_PROBE_INTERVAL
    symmetric_detection_enabled: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        if self.stun_probe_interval_seconds <= 0:
            raise ValueError("stun_probe_interval_seconds must be > 0")


class NATManagementLoop(BaseLoop):
    """Advanced NAT management loop.

    Handles:
    - STUN-like probing to detect NAT type
    - Symmetric NAT detection (which breaks direct connectivity)
    - Intelligent relay selection
    - Hole-punch coordination

    December 2025: Extracted from p2p_orchestrator._nat_management_loop
    """

    def __init__(
        self,
        detect_nat_type: Callable[[], Coroutine[Any, Any, None]],
        probe_nat_blocked_peers: Callable[[], Coroutine[Any, Any, None]],
        update_relay_preferences: Callable[[], Coroutine[Any, Any, None]],
        config: NATManagementConfig | None = None,
    ):
        """Initialize NAT management loop.

        Args:
            detect_nat_type: Async callback to detect NAT type via STUN-like probing
            probe_nat_blocked_peers: Async callback to probe NAT-blocked peers for recovery
            update_relay_preferences: Async callback to update relay preferences
            config: NAT management configuration
        """
        self.config = config or NATManagementConfig()
        super().__init__(
            name="nat_management",
            interval=self.config.check_interval_seconds,
        )
        self._detect_nat_type = detect_nat_type
        self._probe_nat_blocked_peers = probe_nat_blocked_peers
        self._update_relay_preferences = update_relay_preferences
        self._last_stun_probe = 0.0

        # Statistics
        self._stun_probes_count = 0
        self._nat_recovery_attempts = 0
        self._relay_updates_count = 0

    async def _on_start(self) -> None:
        """Log startup."""
        logger.info("Starting advanced NAT management loop")

    async def _run_once(self) -> None:
        """Perform NAT management cycle."""
        now = time.time()

        # Periodic STUN-like probe to detect external IP and NAT type
        if (
            self.config.symmetric_detection_enabled
            and now - self._last_stun_probe > self.config.stun_probe_interval_seconds
        ):
            self._last_stun_probe = now
            self._stun_probes_count += 1
            await self._detect_nat_type()

        # Probe NAT-blocked peers for recovery
        self._nat_recovery_attempts += 1
        await self._probe_nat_blocked_peers()

        # Update relay preferences based on connectivity
        self._relay_updates_count += 1
        await self._update_relay_preferences()

    def get_nat_stats(self) -> dict[str, Any]:
        """Get NAT management statistics."""
        return {
            "stun_probes": self._stun_probes_count,
            "nat_recovery_attempts": self._nat_recovery_attempts,
            "relay_updates": self._relay_updates_count,
            "last_stun_probe": self._last_stun_probe,
            **self.stats.to_dict(),
        }


@dataclass
class TailscalePeerDiscoveryConfig:
    """Configuration for Tailscale peer discovery loop.

    December 2025: Extracted from p2p_orchestrator._tailscale_peer_recovery_loop
    """

    discovery_interval_seconds: float = 120.0  # 2 minutes for leaders
    non_leader_skip_count: int = 3  # Non-leaders run every 3rd iteration (6 min)
    min_connected_peers: int = 3  # If fewer peers, always run discovery
    connect_timeout_seconds: float = 10.0
    max_nodes_per_cycle: int = 10
    p2p_port: int = 8770
    # Hostname patterns for compute nodes we want in the P2P network
    compute_patterns: tuple[str, ...] = (
        "lambda-", "vast-", "gh200", "h100", "a100", "a10",
        "192-222-", "aws-", "nebius-", "runpod-", "vultr-", "hetzner-",
    )

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.discovery_interval_seconds <= 0:
            raise ValueError("discovery_interval_seconds must be > 0")
        if self.non_leader_skip_count <= 0:
            raise ValueError("non_leader_skip_count must be > 0")
        if self.min_connected_peers < 0:
            raise ValueError("min_connected_peers must be >= 0")
        if self.connect_timeout_seconds <= 0:
            raise ValueError("connect_timeout_seconds must be > 0")
        if self.max_nodes_per_cycle <= 0:
            raise ValueError("max_nodes_per_cycle must be > 0")
        if self.p2p_port < 1 or self.p2p_port > 65535:
            raise ValueError("p2p_port must be between 1 and 65535")


class TailscalePeerDiscoveryLoop(BaseLoop):
    """Background loop that discovers and connects to Tailscale peers.

    Proactively finds compute nodes in the Tailscale mesh that are not yet
    in the P2P network and attempts to establish connections.

    Key behaviors:
    - Leaders run discovery every 2 minutes
    - Non-leaders run every 6 minutes (unless isolated)
    - Isolated nodes (few peers) run at leader frequency
    - Connects via HTTP health check + heartbeat

    December 2025: Extracted from p2p_orchestrator._tailscale_peer_recovery_loop
    """

    def __init__(
        self,
        is_leader: Callable[[], bool],
        get_current_peers: Callable[[], set[str]],
        get_alive_peer_count: Callable[[], int],
        probe_and_connect: Callable[[str, str], Coroutine[Any, Any, bool]],
        config: TailscalePeerDiscoveryConfig | None = None,
    ):
        """Initialize Tailscale peer discovery loop.

        Args:
            is_leader: Callback returning True if this node is the cluster leader
            get_current_peers: Callback returning set of current peer node_ids
            get_alive_peer_count: Callback returning count of alive peers
            probe_and_connect: Async callback (ip, hostname) -> success to probe
                               a node and establish P2P connection
            config: Discovery configuration
        """
        self.config = config or TailscalePeerDiscoveryConfig()
        super().__init__(
            name="tailscale_peer_discovery",
            interval=self.config.discovery_interval_seconds,
        )
        self._is_leader = is_leader
        self._get_current_peers = get_current_peers
        self._get_alive_peer_count = get_alive_peer_count
        self._probe_and_connect = probe_and_connect

        # Statistics
        self._loop_count = 0
        self._nodes_discovered = 0
        self._connections_attempted = 0
        self._connections_succeeded = 0

    async def _on_start(self) -> None:
        """Log startup."""
        logger.info(
            f"[TailscalePeerDiscovery] Starting peer discovery loop "
            f"(interval={self.config.discovery_interval_seconds}s)"
        )

    async def _run_once(self) -> None:
        """Discover and connect to Tailscale peers not in P2P network."""
        self._loop_count += 1

        # Non-leaders run less frequently (unless isolated)
        if not self._is_leader():
            if self._loop_count % self.config.non_leader_skip_count != 0:
                # Check if we're isolated (few peers)
                alive_count = self._get_alive_peer_count()
                if alive_count >= self.config.min_connected_peers:
                    return  # Skip this iteration

        # Get current peers
        current_peers = self._get_current_peers()

        # Get Tailscale peers via `tailscale status --json`
        ts_peers = await self._get_tailscale_peers()
        if not ts_peers:
            return

        # Find compute nodes not in P2P network
        missing_nodes = self._find_missing_compute_nodes(ts_peers, current_peers)
        if not missing_nodes:
            return

        self._nodes_discovered += len(missing_nodes)
        logger.info(
            f"[TailscalePeerDiscovery] Found {len(missing_nodes)} compute nodes "
            f"not in P2P network"
        )

        # Attempt to connect to missing nodes
        for hostname, ip in missing_nodes[:self.config.max_nodes_per_cycle]:
            self._connections_attempted += 1
            try:
                success = await asyncio.wait_for(
                    self._probe_and_connect(ip, hostname),
                    timeout=self.config.connect_timeout_seconds,
                )
                if success:
                    self._connections_succeeded += 1
                    logger.info(f"[TailscalePeerDiscovery] Connected to {hostname} ({ip})")
                else:
                    logger.debug(f"[TailscalePeerDiscovery] Failed to connect to {hostname}")
            except asyncio.TimeoutError:
                logger.debug(f"[TailscalePeerDiscovery] Timeout connecting to {hostname}")
            except Exception as e:
                logger.debug(f"[TailscalePeerDiscovery] Error connecting to {hostname}: {e}")

    async def _get_tailscale_peers(self) -> dict[str, Any] | None:
        """Get Tailscale peer information via CLI.

        Returns:
            Dict of peer_id -> peer_info, or None if command fails
        """
        import json
        import subprocess

        try:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["tailscale", "status", "--json"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    ),
                ),
                timeout=15.0,
            )
            if result.returncode != 0:
                logger.debug(f"[TailscalePeerDiscovery] tailscale status failed: {result.stderr}")
                return None
            ts_data = json.loads(result.stdout)
            return ts_data.get("Peer", {})
        except json.JSONDecodeError as e:
            logger.debug(f"[TailscalePeerDiscovery] JSON parse error: {e}")
            return None
        except asyncio.TimeoutError:
            logger.debug("[TailscalePeerDiscovery] tailscale status timed out")
            return None
        except FileNotFoundError:
            logger.debug("[TailscalePeerDiscovery] tailscale not installed")
            return None
        except Exception as e:
            logger.debug(f"[TailscalePeerDiscovery] Error getting tailscale status: {e}")
            return None

    def _find_missing_compute_nodes(
        self,
        ts_peers: dict[str, Any],
        current_peers: set[str],
    ) -> list[tuple[str, str]]:
        """Find compute nodes in Tailscale but not in P2P network.

        Args:
            ts_peers: Tailscale peer data (from `tailscale status --json`)
            current_peers: Set of current P2P peer node_ids

        Returns:
            List of (hostname, ip) tuples for missing nodes
        """
        missing: list[tuple[str, str]] = []

        for _peer_id, peer_info in ts_peers.items():
            hostname = peer_info.get("HostName", "").lower()
            online = peer_info.get("Online", False)
            ts_ips = peer_info.get("TailscaleIPs", [])

            if not online or not ts_ips:
                continue

            # Check if this is a compute node
            is_compute = any(pat in hostname for pat in self.config.compute_patterns)
            if not is_compute:
                continue

            # Check if already in P2P network (by hostname or variants)
            if hostname in current_peers:
                continue
            if hostname.replace("-", "_") in current_peers:
                continue

            # Also check by IP prefix
            ip = ts_ips[0] if ts_ips else ""
            ip_based_id = ip.replace(".", "-")
            if ip_based_id in current_peers:
                continue

            missing.append((hostname, ip))

        return missing

    def get_discovery_stats(self) -> dict[str, Any]:
        """Get discovery statistics."""
        return {
            "loop_count": self._loop_count,
            "nodes_discovered": self._nodes_discovered,
            "connections_attempted": self._connections_attempted,
            "connections_succeeded": self._connections_succeeded,
            "connection_success_rate": (
                self._connections_succeeded / self._connections_attempted * 100
                if self._connections_attempted > 0
                else 0.0
            ),
            **self.stats.to_dict(),
        }


__all__ = [
    "IpDiscoveryConfig",
    "IpDiscoveryLoop",
    "NATManagementConfig",
    "NATManagementLoop",
    "TailscalePeerDiscoveryConfig",
    "TailscalePeerDiscoveryLoop",
    "TailscaleRecoveryConfig",
    "TailscaleRecoveryLoop",
]
