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
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

from .base import BaseLoop

# Import centralized port constant
try:
    from ..constants import DEFAULT_PORT
except ImportError:
    DEFAULT_PORT = 8770  # Fallback if constants unavailable

# Import Tailscale discovery constants (Dec 30, 2025)
try:
    from app.p2p.constants import (
        TAILSCALE_DISCOVERY_BOOTSTRAP_INTERVAL,
        TAILSCALE_DISCOVERY_MAINTENANCE_INTERVAL,
        TAILSCALE_DISCOVERY_MIN_PEERS_FOR_MAINTENANCE,
        TAILSCALE_DISCOVERY_JITTER,
    )
except ImportError:
    # Fallback defaults if constants unavailable
    TAILSCALE_DISCOVERY_BOOTSTRAP_INTERVAL = 60
    TAILSCALE_DISCOVERY_MAINTENANCE_INTERVAL = 120
    TAILSCALE_DISCOVERY_MIN_PEERS_FOR_MAINTENANCE = 5
    TAILSCALE_DISCOVERY_JITTER = 0.1

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
        validate_relay_assignments: Callable[[], Coroutine[Any, Any, None]] | None = None,
        config: NATManagementConfig | None = None,
    ):
        """Initialize NAT management loop.

        Args:
            detect_nat_type: Async callback to detect NAT type via STUN-like probing
            probe_nat_blocked_peers: Async callback to probe NAT-blocked peers for recovery
            update_relay_preferences: Async callback to update relay preferences
            validate_relay_assignments: Async callback to validate relay assignments (Dec 30, 2025)
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
        self._validate_relay_assignments = validate_relay_assignments
        self._last_stun_probe = 0.0

        # Statistics
        self._stun_probes_count = 0
        self._nat_recovery_attempts = 0
        self._relay_updates_count = 0
        self._relay_validations_count = 0

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

        # Dec 30, 2025: Validate existing relay assignments are healthy
        if self._validate_relay_assignments:
            self._relay_validations_count += 1
            await self._validate_relay_assignments()

    def get_nat_stats(self) -> dict[str, Any]:
        """Get NAT management statistics."""
        return {
            "stun_probes": self._stun_probes_count,
            "nat_recovery_attempts": self._nat_recovery_attempts,
            "relay_updates": self._relay_updates_count,
            "relay_validations": self._relay_validations_count,
            "last_stun_probe": self._last_stun_probe,
            **self.stats.to_dict(),
        }


@dataclass
class TailscalePeerDiscoveryConfig:
    """Configuration for Tailscale peer discovery loop.

    December 2025: Extracted from p2p_orchestrator._tailscale_peer_recovery_loop
    December 30, 2025: MAJOR UPDATE - Enable discovery on ALL nodes with adaptive intervals.

    Previous behavior (REMOVED):
    - Leaders ran discovery every 2 minutes
    - Non-leaders ran every 6 minutes (skip_count=3)
    - Isolated nodes (<3 peers) ran at leader frequency

    New behavior:
    - ALL nodes run discovery unconditionally
    - Bootstrap mode: 60s interval when < min_peers_for_maintenance
    - Maintenance mode: 120s interval when >= min_peers_for_maintenance
    - ±10% jitter prevents simultaneous discovery storms
    """

    # Adaptive intervals (Dec 30, 2025)
    bootstrap_interval_seconds: float = float(TAILSCALE_DISCOVERY_BOOTSTRAP_INTERVAL)
    maintenance_interval_seconds: float = float(TAILSCALE_DISCOVERY_MAINTENANCE_INTERVAL)
    min_peers_for_maintenance: int = TAILSCALE_DISCOVERY_MIN_PEERS_FOR_MAINTENANCE
    interval_jitter: float = TAILSCALE_DISCOVERY_JITTER

    # Legacy field for backward compat (now ignored)
    discovery_interval_seconds: float = 120.0  # DEPRECATED: Use bootstrap/maintenance intervals
    non_leader_skip_count: int = 3  # DEPRECATED: No longer used (all nodes run discovery)

    # Connection settings
    connect_timeout_seconds: float = 10.0
    max_nodes_per_cycle: int = 10
    p2p_port: int = DEFAULT_PORT

    # Hostname patterns for compute nodes we want in the P2P network
    compute_patterns: tuple[str, ...] = (
        "lambda-", "vast-", "gh200", "h100", "a100", "a10",
        "192-222-", "aws-", "nebius-", "runpod-", "vultr-", "hetzner-",
    )

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.bootstrap_interval_seconds <= 0:
            raise ValueError("bootstrap_interval_seconds must be > 0")
        if self.maintenance_interval_seconds <= 0:
            raise ValueError("maintenance_interval_seconds must be > 0")
        if self.min_peers_for_maintenance < 0:
            raise ValueError("min_peers_for_maintenance must be >= 0")
        if self.interval_jitter < 0 or self.interval_jitter > 1:
            raise ValueError("interval_jitter must be between 0 and 1")
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

    December 30, 2025 - MAJOR CHANGE: All-Node Discovery

    Previous behavior (REMOVED):
    - Leaders ran discovery every 2 minutes
    - Non-leaders ran every 6 minutes (unless isolated)
    - Bootstrap problem: isolated nodes couldn't discover peers

    New behavior:
    - ALL nodes run discovery unconditionally (no leader check)
    - Adaptive intervals based on peer count:
        - Bootstrap mode (<5 peers): 60s interval (aggressive)
        - Maintenance mode (>=5 peers): 120s interval (conservative)
    - ±10% jitter prevents simultaneous discovery storms

    This enables isolated nodes to bootstrap independently even when:
    - Leader is down or electing
    - Gossip chain is broken
    - Node restarted and lost peer list
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
            is_leader: Callback returning True if this node is cluster leader
                       (DEPRECATED: No longer used for gating, kept for compatibility)
            get_current_peers: Callback returning set of current peer node_ids
            get_alive_peer_count: Callback returning count of alive peers
            probe_and_connect: Async callback (ip, hostname) -> success to probe
                               a node and establish P2P connection
            config: Discovery configuration
        """
        self.config = config or TailscalePeerDiscoveryConfig()
        super().__init__(
            name="tailscale_peer_discovery",
            interval=self.config.bootstrap_interval_seconds,  # Start with bootstrap interval
        )
        self._is_leader = is_leader  # Kept for backward compat but no longer used
        self._get_current_peers = get_current_peers
        self._get_alive_peer_count = get_alive_peer_count
        self._probe_and_connect = probe_and_connect

        # Statistics
        self._loop_count = 0
        self._nodes_discovered = 0
        self._connections_attempted = 0
        self._connections_succeeded = 0

        # Adaptive interval tracking (Dec 30, 2025)
        self._current_mode: str = "bootstrap"  # "bootstrap" or "maintenance"
        self._last_interval_adjustment = 0.0

    async def _on_start(self) -> None:
        """Log startup."""
        logger.info(
            f"[TailscalePeerDiscovery] Starting ALL-NODE peer discovery loop "
            f"(bootstrap={self.config.bootstrap_interval_seconds}s, "
            f"maintenance={self.config.maintenance_interval_seconds}s, "
            f"min_peers={self.config.min_peers_for_maintenance})"
        )

    def _get_jittered_interval(self, base_interval: float) -> float:
        """Apply jitter to prevent discovery storms.

        Returns interval ± jitter% (e.g., 120s ± 10% = 108-132s).
        """
        jitter_range = base_interval * self.config.interval_jitter
        return base_interval + random.uniform(-jitter_range, jitter_range)

    def _update_interval_mode(self, alive_count: int) -> None:
        """Update discovery interval based on current peer count.

        Bootstrap mode (<5 peers): Aggressive 60s interval
        Maintenance mode (>=5 peers): Conservative 120s interval
        """
        min_peers = self.config.min_peers_for_maintenance

        if alive_count < min_peers:
            new_mode = "bootstrap"
            base_interval = self.config.bootstrap_interval_seconds
        else:
            new_mode = "maintenance"
            base_interval = self.config.maintenance_interval_seconds

        # Log mode transition
        if new_mode != self._current_mode:
            logger.info(
                f"[TailscalePeerDiscovery] Mode transition: {self._current_mode} -> {new_mode} "
                f"(peers={alive_count}, threshold={min_peers})"
            )
            self._current_mode = new_mode

        # Update loop interval with jitter
        self._interval = self._get_jittered_interval(base_interval)

    async def _run_once(self) -> None:
        """Discover and connect to Tailscale peers not in P2P network.

        December 30, 2025: Runs on ALL nodes (leader check removed).
        """
        self._loop_count += 1

        # Get alive peer count and update interval mode
        alive_count = self._get_alive_peer_count()
        self._update_interval_mode(alive_count)

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
            # Dec 30, 2025: Added mode tracking for adaptive intervals
            "current_mode": self._current_mode,
            "current_interval": self._interval,
            "min_peers_for_maintenance": self.config.min_peers_for_maintenance,
            **self.stats.to_dict(),
        }


@dataclass
class ProviderIpUpdateConfig:
    """Configuration for provider IP update loops.

    December 2025: Extracted from p2p_orchestrator inline loops.
    """

    update_interval_seconds: float = 300.0  # 5 minutes
    error_retry_seconds: float = 60.0  # Retry faster on error

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.update_interval_seconds <= 0:
            raise ValueError("update_interval_seconds must be > 0")
        if self.error_retry_seconds <= 0:
            raise ValueError("error_retry_seconds must be > 0")


class VastIpUpdateLoop(BaseLoop):
    """Background loop to periodically refresh Vast.ai instance connection info.

    Uses VAST_API_KEY when available, otherwise falls back to the `vastai`
    CLI if installed (see DynamicHostRegistry.update_vast_ips).

    December 2025: Extracted from p2p_orchestrator._vast_ip_update_loop
    """

    def __init__(
        self,
        get_registry: Callable[[], Any],
        has_registry: bool = True,
        config: ProviderIpUpdateConfig | None = None,
    ):
        """Initialize Vast IP update loop.

        Args:
            get_registry: Callback returning DynamicHostRegistry instance
            has_registry: Whether the registry module is available
            config: Update configuration
        """
        self.config = config or ProviderIpUpdateConfig()
        super().__init__(
            name="vast_ip_update",
            interval=self.config.update_interval_seconds,
        )
        self._get_registry = get_registry
        self._has_registry = has_registry
        self._updates_count = 0

    async def _run_once(self) -> None:
        """Refresh Vast instance IPs from API."""
        if not self._has_registry:
            return

        try:
            registry = self._get_registry()
            updated = await registry.update_vast_ips()
            if updated > 0:
                self._updates_count += updated
                logger.info(f"[VastIpUpdate] Updated {updated} Vast instance IPs from API")
        except Exception as e:
            logger.debug(f"[VastIpUpdate] Error: {e}")
            # Sleep briefly before next attempt on error
            await asyncio.sleep(self.config.error_retry_seconds)

    def get_update_stats(self) -> dict[str, Any]:
        """Get update statistics."""
        return {
            "total_updates": self._updates_count,
            **self.stats.to_dict(),
        }


class AwsIpUpdateLoop(BaseLoop):
    """Background loop to periodically refresh AWS instance connection info.

    Uses the `aws` CLI (see DynamicHostRegistry.update_aws_ips). No-op when
    no AWS instances are configured in distributed_hosts.yaml properties.

    December 2025: Extracted from p2p_orchestrator._aws_ip_update_loop
    """

    def __init__(
        self,
        get_registry: Callable[[], Any],
        has_registry: bool = True,
        config: ProviderIpUpdateConfig | None = None,
    ):
        """Initialize AWS IP update loop.

        Args:
            get_registry: Callback returning DynamicHostRegistry instance
            has_registry: Whether the registry module is available
            config: Update configuration
        """
        self.config = config or ProviderIpUpdateConfig()
        super().__init__(
            name="aws_ip_update",
            interval=self.config.update_interval_seconds,
        )
        self._get_registry = get_registry
        self._has_registry = has_registry
        self._updates_count = 0

    async def _run_once(self) -> None:
        """Refresh AWS instance IPs via CLI."""
        if not self._has_registry:
            return

        try:
            registry = self._get_registry()
            updated = await registry.update_aws_ips()
            if updated > 0:
                self._updates_count += updated
                logger.info(f"[AwsIpUpdate] Updated {updated} AWS instance IPs via CLI")
        except Exception as e:
            logger.debug(f"[AwsIpUpdate] Error: {e}")
            await asyncio.sleep(self.config.error_retry_seconds)

    def get_update_stats(self) -> dict[str, Any]:
        """Get update statistics."""
        return {
            "total_updates": self._updates_count,
            **self.stats.to_dict(),
        }


class TailscaleIpUpdateLoop(BaseLoop):
    """Background loop to discover and update Tailscale IPs for cluster nodes.

    Uses `tailscale status --json` to discover mesh network peers.
    Tailscale provides reliable connectivity even when public IPs change.

    December 2025: Extracted from p2p_orchestrator._tailscale_ip_update_loop
    """

    def __init__(
        self,
        get_registry: Callable[[], Any],
        has_registry: bool = True,
        config: ProviderIpUpdateConfig | None = None,
    ):
        """Initialize Tailscale IP update loop.

        Args:
            get_registry: Callback returning DynamicHostRegistry instance
            has_registry: Whether the registry module is available
            config: Update configuration (default 2 min interval for Tailscale)
        """
        self.config = config or ProviderIpUpdateConfig(update_interval_seconds=120.0)
        super().__init__(
            name="tailscale_ip_update",
            interval=self.config.update_interval_seconds,
        )
        self._get_registry = get_registry
        self._has_registry = has_registry
        self._updates_count = 0

    async def _run_once(self) -> None:
        """Refresh Tailscale IPs from mesh status."""
        if not self._has_registry:
            return

        try:
            registry = self._get_registry()
            updated = await registry.update_tailscale_ips()
            if updated > 0:
                self._updates_count += updated
                logger.info(f"[TailscaleIpUpdate] Updated {updated} node Tailscale IPs")
        except Exception as e:
            logger.debug(f"[TailscaleIpUpdate] Error: {e}")
            await asyncio.sleep(self.config.error_retry_seconds)

    def get_update_stats(self) -> dict[str, Any]:
        """Get update statistics."""
        return {
            "total_updates": self._updates_count,
            **self.stats.to_dict(),
        }


# ============================================
# Heartbeat Loops
# ============================================


@dataclass
class HeartbeatConfig:
    """Configuration for heartbeat loop.

    December 2025: Extracted from p2p_orchestrator._heartbeat_loop
    """

    interval_seconds: float = 15.0  # HEARTBEAT_INTERVAL
    relay_heartbeat_interval: float = 30.0  # RELAY_HEARTBEAT_INTERVAL
    leader_lease_duration: float = 60.0  # LEADER_LEASE_DURATION
    bootstrap_on_low_peers: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        if self.relay_heartbeat_interval <= 0:
            raise ValueError("relay_heartbeat_interval must be > 0")
        if self.leader_lease_duration <= 0:
            raise ValueError("leader_lease_duration must be > 0")


class HeartbeatLoop(BaseLoop):
    """Background loop that sends heartbeats to all known peers.

    Responsible for:
    - Sending heartbeats to configured known peers
    - Discovering and following leaders via heartbeat responses
    - Multi-path retry (public IP, reported IP, Tailscale fallback)
    - Handling relay heartbeats for HTTPS endpoints
    - Bootstrapping from known peers when isolated

    December 2025: Extracted from p2p_orchestrator._heartbeat_loop
    This loop is the core P2P network maintenance mechanism.
    """

    def __init__(
        self,
        get_known_peers: Callable[[], list[str]],
        get_relay_peers: Callable[[], set[str]],
        get_peers_snapshot: Callable[[], list[Any]],
        send_heartbeat_to_peer: Callable[[str, int, str], Coroutine[Any, Any, Any | None]],
        send_relay_heartbeat: Callable[[str], Coroutine[Any, Any, dict[str, Any]]],
        update_peer: Callable[[Any], Coroutine[Any, Any, None]],
        parse_peer_address: Callable[[str], tuple[str, str, int]],
        get_node_id: Callable[[], str],
        get_role: Callable[[], Any],
        get_leader_id: Callable[[], str | None],
        set_leader: Callable[[str], Coroutine[Any, Any, None]],
        is_leader_eligible: Callable[[Any, set[str]], bool],
        is_leader_lease_valid: Callable[[], bool],
        endpoint_conflict_keys: Callable[[list[Any]], set[str]],
        bootstrap_from_known_peers: Callable[[], Coroutine[Any, Any, None]],
        emit_host_online: Callable[[str, list[str]], Coroutine[Any, Any, None]],
        get_tailscale_ip_for_peer: Callable[[str], str | None],
        get_self_info: Callable[[], Any],
        config: HeartbeatConfig | None = None,
    ):
        """Initialize heartbeat loop.

        Args:
            get_known_peers: Returns list of peer addresses from config
            get_relay_peers: Returns set of relay peer addresses
            get_peers_snapshot: Returns list of current peer NodeInfo objects
            send_heartbeat_to_peer: Async callback to send heartbeat (host, port, scheme) -> NodeInfo
            send_relay_heartbeat: Async callback for relay heartbeats (url) -> result dict
            update_peer: Async callback to update peer info
            parse_peer_address: Parses peer address string to (scheme, host, port)
            get_node_id: Returns this node's ID
            get_role: Returns this node's role (LEADER/FOLLOWER)
            get_leader_id: Returns current leader's node ID
            set_leader: Async callback to set new leader
            is_leader_eligible: Checks if peer is eligible to be leader
            is_leader_lease_valid: Checks if current leader lease is valid
            endpoint_conflict_keys: Returns conflicting endpoint keys
            bootstrap_from_known_peers: Async callback to bootstrap from known peers
            emit_host_online: Async callback to emit HOST_ONLINE event
            get_tailscale_ip_for_peer: Returns Tailscale IP for peer if available
            get_self_info: Returns this node's NodeInfo
            config: Heartbeat configuration
        """
        self.config = config or HeartbeatConfig()
        super().__init__(
            name="heartbeat",
            interval=self.config.interval_seconds,
        )

        # Store callbacks
        self._get_known_peers = get_known_peers
        self._get_relay_peers = get_relay_peers
        self._get_peers_snapshot = get_peers_snapshot
        self._send_heartbeat_to_peer = send_heartbeat_to_peer
        self._send_relay_heartbeat = send_relay_heartbeat
        self._update_peer = update_peer
        self._parse_peer_address = parse_peer_address
        self._get_node_id = get_node_id
        self._get_role = get_role
        self._get_leader_id = get_leader_id
        self._set_leader = set_leader
        self._is_leader_eligible = is_leader_eligible
        self._is_leader_lease_valid = is_leader_lease_valid
        self._endpoint_conflict_keys = endpoint_conflict_keys
        self._bootstrap_from_known_peers = bootstrap_from_known_peers
        self._emit_host_online = emit_host_online
        self._get_tailscale_ip_for_peer = get_tailscale_ip_for_peer
        self._get_self_info = get_self_info

        # Statistics
        self._heartbeats_sent = 0
        self._heartbeats_succeeded = 0
        self._heartbeats_failed = 0
        self._peers_discovered = 0
        self._leaders_discovered = 0
        self._last_relay_heartbeat = 0.0

    async def _on_start(self) -> None:
        """Log startup."""
        logger.info(
            f"[Heartbeat] Starting heartbeat loop (interval={self.config.interval_seconds}s)"
        )

    async def _run_once(self) -> None:
        """Send heartbeats to all known and discovered peers."""
        node_id = self._get_node_id()
        relay_peers = self._get_relay_peers()

        # Send to known peers from config
        for peer_addr in self._get_known_peers():
            try:
                scheme, host, port = self._parse_peer_address(peer_addr)
            except (AttributeError, ValueError):
                continue

            # Use relay heartbeat for HTTPS endpoints or configured relay peers
            use_relay = scheme == "https" or peer_addr in relay_peers
            if use_relay:
                relay_url = f"{scheme}://{host}" if port in (80, 443) else f"{scheme}://{host}:{port}"
                result = await self._send_relay_heartbeat(relay_url)
                if result.get("success"):
                    continue

            self._heartbeats_sent += 1
            info = await self._send_heartbeat_to_peer(host, port, scheme)

            if info:
                if info.node_id == node_id:
                    continue

                self._heartbeats_succeeded += 1

                # Check if this is first contact (new peer)
                peers_snapshot = self._get_peers_snapshot()
                peer_ids = {p.node_id for p in peers_snapshot}
                is_first_contact = info.node_id not in peer_ids

                info.last_heartbeat = time.time()
                await self._update_peer(info)

                # Emit HOST_ONLINE for newly discovered peers
                if is_first_contact:
                    self._peers_discovered += 1
                    # Dec 30, 2025: All nodes get selfplay capability as base
                    # This ensures nodes can receive work from the scheduler
                    capabilities = ["selfplay"]
                    if getattr(info, "has_gpu", False):
                        gpu_type = getattr(info, "gpu_type", "") or "gpu"
                        # GPU nodes can also do training and cmaes
                        capabilities.extend(["training", "cmaes", gpu_type])
                    else:
                        capabilities.append("cpu")
                    await self._emit_host_online(info.node_id, capabilities)
                    logger.info(f"[Heartbeat] First-contact peer: {info.node_id}")

                # Handle leader discovery
                await self._handle_leader_discovery(info)
            else:
                self._heartbeats_failed += 1

        # Send to discovered peers
        await self._send_to_discovered_peers()

        # Bootstrap from known peers if isolated
        if self.config.bootstrap_on_low_peers:
            await self._bootstrap_from_known_peers()

    async def _send_to_discovered_peers(self) -> None:
        """Send heartbeats to discovered peers with multi-path retry."""
        node_id = self._get_node_id()
        peers_snapshot = self._get_peers_snapshot()
        self_info = self._get_self_info()

        conflict_keys = self._endpoint_conflict_keys([self_info, *peers_snapshot])

        # Filter to non-NAT-blocked peers without endpoint conflicts
        peer_list = [
            p for p in peers_snapshot
            if (
                not getattr(p, "nat_blocked", False)
                and self._endpoint_key(p) not in conflict_keys
            )
        ]

        for peer in peer_list:
            if peer.node_id == node_id:
                continue

            if not peer.should_retry():
                continue

            peer_scheme = getattr(peer, "scheme", "http") or "http"
            self._heartbeats_sent += 1

            # Try primary endpoint
            info = await self._send_heartbeat_to_peer(peer.host, peer.port, peer_scheme)

            # Multi-path retry: fall back to reported endpoint
            if not info:
                rh = str(getattr(peer, "reported_host", "") or "").strip()
                rp = int(getattr(peer, "reported_port", 0) or 0)
                if rh and rp and (rh != peer.host or rp != peer.port):
                    info = await self._send_heartbeat_to_peer(rh, rp, peer_scheme)

            # Self-healing: Tailscale IP fallback
            if not info:
                ts_ip = self._get_tailscale_ip_for_peer(peer.node_id)
                if ts_ip and ts_ip != peer.host:
                    info = await self._send_heartbeat_to_peer(ts_ip, peer.port, peer_scheme)
                    if info:
                        logger.info(f"[Heartbeat] Reached {peer.node_id} via Tailscale ({ts_ip})")

            if info:
                self._heartbeats_succeeded += 1
                info.consecutive_failures = 0
                info.last_failure_time = 0.0
                info.last_heartbeat = time.time()
                await self._update_peer(info)

                # Handle leader discovery
                await self._handle_leader_discovery(info)
            else:
                self._heartbeats_failed += 1
                # Increment failure count on peer
                peer.consecutive_failures = int(getattr(peer, "consecutive_failures", 0) or 0) + 1
                peer.last_failure_time = time.time()

    async def _handle_leader_discovery(self, info: Any) -> None:
        """Handle potential leader discovery from heartbeat response."""
        # Skip if this is ourself or we're already the leader
        node_role = self._get_role()
        if info.role != "LEADER" or node_role == "LEADER":
            return

        current_leader = self._get_leader_id()
        peers_snapshot = self._get_peers_snapshot()
        self_info = self._get_self_info()
        conflict_keys = self._endpoint_conflict_keys([self_info, *peers_snapshot])

        # Check if peer is eligible to be leader
        if not self._is_leader_eligible(info, conflict_keys):
            return

        # Don't override existing valid leader with lower-priority node
        if (
            current_leader
            and current_leader != info.node_id
            and self._is_leader_lease_valid()
            and info.node_id <= current_leader
        ):
            return

        if current_leader != info.node_id:
            self._leaders_discovered += 1
            logger.info(f"[Heartbeat] Discovered leader: {info.node_id}")

        await self._set_leader(info.node_id)

    def _endpoint_key(self, peer: Any) -> str:
        """Create endpoint key for conflict detection."""
        return f"{peer.host}:{peer.port}"

    def get_heartbeat_stats(self) -> dict[str, Any]:
        """Get heartbeat statistics."""
        return {
            "heartbeats_sent": self._heartbeats_sent,
            "heartbeats_succeeded": self._heartbeats_succeeded,
            "heartbeats_failed": self._heartbeats_failed,
            "success_rate": (
                self._heartbeats_succeeded / self._heartbeats_sent * 100
                if self._heartbeats_sent > 0
                else 0.0
            ),
            "peers_discovered": self._peers_discovered,
            "leaders_discovered": self._leaders_discovered,
            **self.stats.to_dict(),
        }


@dataclass
class VoterHeartbeatConfig:
    """Configuration for voter heartbeat loop.

    December 2025: Extracted from p2p_orchestrator._voter_heartbeat_loop
    """

    interval_seconds: float = 10.0  # VOTER_HEARTBEAT_INTERVAL (faster than regular)
    heartbeat_timeout_seconds: float = 5.0  # VOTER_HEARTBEAT_TIMEOUT
    mesh_refresh_interval_seconds: float = 60.0  # VOTER_MESH_REFRESH_INTERVAL
    nat_recovery_aggressive: bool = True  # VOTER_NAT_RECOVERY_AGGRESSIVE

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        if self.heartbeat_timeout_seconds <= 0:
            raise ValueError("heartbeat_timeout_seconds must be > 0")
        if self.mesh_refresh_interval_seconds <= 0:
            raise ValueError("mesh_refresh_interval_seconds must be > 0")


class VoterHeartbeatLoop(BaseLoop):
    """Dedicated high-frequency heartbeat loop for voter nodes.

    Provides:
    - Faster heartbeat interval (10s vs 15s) for voter nodes
    - Aggressive NAT-blocked status clearing on successful heartbeats
    - Full mesh connectivity between all voters
    - Voter list propagation for consistent quorum

    Only runs if this node is a voter (checked via callback).

    December 2025: Extracted from p2p_orchestrator._voter_heartbeat_loop
    """

    def __init__(
        self,
        get_voter_node_ids: Callable[[], list[str]],
        get_node_id: Callable[[], str],
        get_peer: Callable[[str], Any | None],
        send_voter_heartbeat: Callable[[Any], Coroutine[Any, Any, bool]],
        try_alternative_endpoints: Callable[[Any], Coroutine[Any, Any, bool]],
        discover_voter_peer: Callable[[str], Coroutine[Any, Any, None]],
        refresh_voter_mesh: Callable[[], Coroutine[Any, Any, None]],
        clear_nat_blocked: Callable[[str], Coroutine[Any, Any, None]],
        increment_failures: Callable[[str], None],
        config: VoterHeartbeatConfig | None = None,
    ):
        """Initialize voter heartbeat loop.

        Args:
            get_voter_node_ids: Returns list of voter node IDs
            get_node_id: Returns this node's ID
            get_peer: Returns peer info for given node ID
            send_voter_heartbeat: Async callback to send voter heartbeat
            try_alternative_endpoints: Async callback to try alternative endpoints
            discover_voter_peer: Async callback to discover voter peer
            refresh_voter_mesh: Async callback to refresh voter mesh
            clear_nat_blocked: Async callback to clear NAT-blocked status
            increment_failures: Callback to increment failure count for peer
            config: Voter heartbeat configuration
        """
        self.config = config or VoterHeartbeatConfig()
        super().__init__(
            name="voter_heartbeat",
            interval=self.config.interval_seconds,
        )

        self._get_voter_node_ids = get_voter_node_ids
        self._get_node_id = get_node_id
        self._get_peer = get_peer
        self._send_voter_heartbeat = send_voter_heartbeat
        self._try_alternative_endpoints = try_alternative_endpoints
        self._discover_voter_peer = discover_voter_peer
        self._refresh_voter_mesh = refresh_voter_mesh
        self._clear_nat_blocked = clear_nat_blocked
        self._increment_failures = increment_failures

        # State
        self._last_mesh_refresh = 0.0
        self._is_voter = False

        # Statistics
        self._heartbeats_sent = 0
        self._heartbeats_succeeded = 0
        self._nat_recoveries = 0
        self._mesh_refreshes = 0

    async def _on_start(self) -> None:
        """Check if this node is a voter."""
        node_id = self._get_node_id()
        voter_ids = self._get_voter_node_ids()
        self._is_voter = node_id in voter_ids

        if not self._is_voter:
            logger.info("[VoterHeartbeat] This node is not a voter, loop will skip")
        else:
            logger.info(
                f"[VoterHeartbeat] Starting voter heartbeat loop "
                f"(interval={self.config.interval_seconds}s)"
            )

    async def _run_once(self) -> None:
        """Send heartbeats to all other voter nodes."""
        if not self._is_voter:
            return

        node_id = self._get_node_id()
        voter_ids = self._get_voter_node_ids()
        other_voters = [v for v in voter_ids if v != node_id]
        now = time.time()

        for voter_id in other_voters:
            voter_peer = self._get_peer(voter_id)

            if not voter_peer:
                # Try to discover voter from known peers
                await self._discover_voter_peer(voter_id)
                continue

            self._heartbeats_sent += 1
            success = await self._send_voter_heartbeat(voter_peer)

            if success:
                self._heartbeats_succeeded += 1

                # Aggressive NAT recovery: clear NAT-blocked immediately on success
                if (
                    self.config.nat_recovery_aggressive
                    and getattr(voter_peer, "nat_blocked", False)
                ):
                    await self._clear_nat_blocked(voter_id)
                    self._nat_recoveries += 1
                    logger.info(
                        f"[VoterHeartbeat] Voter {voter_id} NAT-blocked status cleared"
                    )
            else:
                # Try alternative endpoints
                success = await self._try_alternative_endpoints(voter_peer)

                if not success:
                    self._increment_failures(voter_id)

        # Periodic voter mesh refresh
        if now - self._last_mesh_refresh > self.config.mesh_refresh_interval_seconds:
            self._last_mesh_refresh = now
            self._mesh_refreshes += 1
            await self._refresh_voter_mesh()

    def get_voter_stats(self) -> dict[str, Any]:
        """Get voter heartbeat statistics."""
        return {
            "is_voter": self._is_voter,
            "heartbeats_sent": self._heartbeats_sent,
            "heartbeats_succeeded": self._heartbeats_succeeded,
            "success_rate": (
                self._heartbeats_succeeded / self._heartbeats_sent * 100
                if self._heartbeats_sent > 0
                else 0.0
            ),
            "nat_recoveries": self._nat_recoveries,
            "mesh_refreshes": self._mesh_refreshes,
            **self.stats.to_dict(),
        }


__all__ = [
    # IP Discovery
    "IpDiscoveryConfig",
    "IpDiscoveryLoop",
    # NAT Management
    "NATManagementConfig",
    "NATManagementLoop",
    # Tailscale Peer Discovery
    "TailscalePeerDiscoveryConfig",
    "TailscalePeerDiscoveryLoop",
    # Tailscale Recovery
    "TailscaleRecoveryConfig",
    "TailscaleRecoveryLoop",
    # Provider IP Updates (December 2025)
    "ProviderIpUpdateConfig",
    "VastIpUpdateLoop",
    "AwsIpUpdateLoop",
    "TailscaleIpUpdateLoop",
    # Heartbeat Loops (December 2025)
    "HeartbeatConfig",
    "HeartbeatLoop",
    "VoterHeartbeatConfig",
    "VoterHeartbeatLoop",
]
