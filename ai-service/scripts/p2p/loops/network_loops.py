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
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Coroutine

from .base import BaseLoop

# Import centralized port constant
try:
    from ..constants import DEFAULT_PORT
except ImportError:
    DEFAULT_PORT = 8770  # Fallback if constants unavailable

# Jan 2026: Import centralized timeouts for adaptive usage
try:
    from .loop_constants import LoopTimeouts
except ImportError:
    LoopTimeouts = None  # type: ignore

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

    async def _is_reachable(self, ip: str, port: int = 22, provider: str = "") -> bool:
        """Quick check if IP is reachable via TCP.

        Jan 2026: Uses adaptive timeout from LoopTimeouts for provider-specific delays.
        """
        # Use adaptive health check timeout if available
        timeout = 5.0  # Default fallback
        if LoopTimeouts is not None:
            timeout = LoopTimeouts.get_adaptive_health_check(provider=provider)

        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, port),
                timeout=timeout,
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
    January 2, 2026: Added YAML fallback for bootstrap mode.

    Previous behavior (REMOVED):
    - Leaders ran discovery every 2 minutes
    - Non-leaders ran every 6 minutes (skip_count=3)
    - Isolated nodes (<3 peers) ran at leader frequency

    New behavior:
    - ALL nodes run discovery unconditionally
    - Bootstrap mode: 60s interval when < min_peers_for_maintenance
    - Maintenance mode: 120s interval when >= min_peers_for_maintenance
    - ±10% jitter prevents simultaneous discovery storms
    - YAML fallback: When in bootstrap mode, also probe hosts from distributed_hosts.yaml
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

    # YAML fallback for bootstrap mode (Jan 2026)
    # When in bootstrap mode, also probe hosts from distributed_hosts.yaml
    yaml_fallback_enabled: bool = True
    yaml_fallback_path: str = ""  # Empty = auto-detect

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
        January 2, 2026: Added YAML fallback for bootstrap mode.
        """
        self._loop_count += 1

        # Get alive peer count and update interval mode
        alive_count = self._get_alive_peer_count()
        self._update_interval_mode(alive_count)

        # Get current peers
        current_peers = self._get_current_peers()

        # Collect missing nodes from multiple sources
        missing_nodes: list[tuple[str, str]] = []
        seen_hosts: set[str] = set()

        # Source 1: Tailscale status
        ts_peers = await self._get_tailscale_peers()
        if ts_peers:
            ts_missing = self._find_missing_compute_nodes(ts_peers, current_peers)
            for hostname, ip in ts_missing:
                if hostname not in seen_hosts:
                    missing_nodes.append((hostname, ip))
                    seen_hosts.add(hostname)

        # Source 2: YAML fallback (Jan 2026)
        # Use YAML fallback in bootstrap mode OR when Tailscale returned no peers
        use_yaml_fallback = (
            self.config.yaml_fallback_enabled
            and (self._current_mode == "bootstrap" or not ts_peers)
        )
        if use_yaml_fallback:
            yaml_missing = await self._probe_yaml_hosts(current_peers)
            for hostname, ip in yaml_missing:
                if hostname not in seen_hosts:
                    missing_nodes.append((hostname, ip))
                    seen_hosts.add(hostname)

        if not missing_nodes:
            return

        self._nodes_discovered += len(missing_nodes)
        sources = []
        if ts_peers:
            sources.append("tailscale")
        if use_yaml_fallback:
            sources.append("yaml")
        logger.info(
            f"[TailscalePeerDiscovery] Found {len(missing_nodes)} compute nodes "
            f"not in P2P network (sources: {', '.join(sources)})"
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

        Jan 2026: Uses adaptive timeout from LoopTimeouts for subprocess operations.
        """
        import json
        import subprocess

        # Jan 2026: Use adaptive timeout for subprocess operations
        subprocess_timeout = 10
        async_timeout = 15.0
        if LoopTimeouts is not None:
            # Use health check timeout as base for subprocess operations
            base_timeout = LoopTimeouts.get_adaptive_health_check()
            subprocess_timeout = int(base_timeout * 2)  # Subprocess gets 2x base
            async_timeout = base_timeout * 3  # Async wrapper gets 3x base

        try:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["tailscale", "status", "--json"],
                        capture_output=True,
                        text=True,
                        timeout=subprocess_timeout,
                    ),
                ),
                timeout=async_timeout,
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

    def _get_yaml_hosts_path(self) -> Path | None:
        """Get path to distributed_hosts.yaml, auto-detecting if needed."""
        if self.config.yaml_fallback_path:
            return Path(self.config.yaml_fallback_path)

        # Auto-detect: Try common locations
        candidates = [
            Path(__file__).parent.parent.parent.parent / "config" / "distributed_hosts.yaml",
            Path.home() / "ringrift" / "ai-service" / "config" / "distributed_hosts.yaml",
            Path("/workspace/ringrift/ai-service/config/distributed_hosts.yaml"),
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _load_yaml_hosts(self) -> list[tuple[str, str]]:
        """Load hosts from distributed_hosts.yaml.

        Returns:
            List of (hostname, tailscale_ip) tuples for enabled hosts
        """
        yaml_path = self._get_yaml_hosts_path()
        if not yaml_path or not yaml_path.exists():
            return []

        try:
            import yaml
            with open(yaml_path) as f:
                config = yaml.safe_load(f)

            hosts = []
            for host in config.get("hosts", []):
                name = host.get("name", "")
                tailscale_ip = host.get("tailscale_ip", "")
                enabled = host.get("enabled", True)

                if name and tailscale_ip and enabled:
                    hosts.append((name, tailscale_ip))

            return hosts
        except Exception as e:
            logger.debug(f"[TailscalePeerDiscovery] Failed to load YAML: {e}")
            return []

    async def _probe_yaml_hosts(self, current_peers: set[str]) -> list[tuple[str, str]]:
        """Find and probe hosts from YAML that aren't in P2P network.

        January 2026: This provides an alternative discovery path when
        Tailscale status doesn't show all peers (common in container/userspace mode).

        Returns:
            List of (hostname, ip) for hosts that should be connected
        """
        yaml_hosts = self._load_yaml_hosts()
        if not yaml_hosts:
            return []

        missing = []
        for hostname, ip in yaml_hosts:
            # Skip if already in P2P network
            if hostname in current_peers:
                continue
            if hostname.replace("-", "_") in current_peers:
                continue

            # Skip if this is ourselves (crude check)
            if ip.startswith("127.") or ip == "0.0.0.0":
                continue

            missing.append((hostname, ip))

        if missing:
            logger.info(
                f"[TailscalePeerDiscovery] YAML fallback found {len(missing)} "
                f"hosts not in P2P network"
            )

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

            # Use relay heartbeat for HTTPS endpoints, configured relay peers,
            # or if this node is NAT-blocked and needs to relay ALL outbound heartbeats
            local_force_relay = getattr(self, "_force_relay_mode", False)
            use_relay = scheme == "https" or peer_addr in relay_peers or local_force_relay
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

    def health_check(self) -> Any:
        """Check loop health with heartbeat-specific status.

        Jan 2026: Added for DaemonManager integration.
        Reports heartbeat success rate, peer discovery, and leader tracking.

        Returns:
            HealthCheckResult with heartbeat loop status
        """
        try:
            from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
        except ImportError:
            # Fallback if protocols not available
            stats = self.get_heartbeat_stats()
            return {
                "healthy": self.running,
                "status": "running" if self.running else "stopped",
                "message": f"HeartbeatLoop {'running' if self.running else 'stopped'}",
                "details": stats,
            }

        # Not running
        if not self.running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="HeartbeatLoop is stopped",
                details={"running": False},
            )

        # Get stats for health assessment
        stats = self.get_heartbeat_stats()
        success_rate = stats.get("success_rate", 0.0)
        heartbeats_sent = stats.get("heartbeats_sent", 0)

        # Check for critical conditions
        if heartbeats_sent > 10 and success_rate < 25:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Heartbeat success rate critical ({success_rate:.1f}%)",
                details={
                    "success_rate": success_rate,
                    "heartbeats_sent": heartbeats_sent,
                    "heartbeats_succeeded": stats.get("heartbeats_succeeded", 0),
                    "heartbeats_failed": stats.get("heartbeats_failed", 0),
                    "peers_discovered": stats.get("peers_discovered", 0),
                },
            )

        # Check for degraded conditions
        if heartbeats_sent > 10 and success_rate < 60:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message=f"Heartbeat success rate degraded ({success_rate:.1f}%)",
                details={
                    "success_rate": success_rate,
                    "heartbeats_sent": heartbeats_sent,
                    "peers_discovered": stats.get("peers_discovered", 0),
                    "leaders_discovered": stats.get("leaders_discovered", 0),
                },
            )

        # Healthy
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"HeartbeatLoop healthy (success rate: {success_rate:.1f}%)",
            details={
                "success_rate": success_rate,
                "heartbeats_sent": heartbeats_sent,
                "heartbeats_succeeded": stats.get("heartbeats_succeeded", 0),
                "peers_discovered": stats.get("peers_discovered", 0),
                "leaders_discovered": stats.get("leaders_discovered", 0),
            },
        )


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
        # Jan 5, 2026: Track reachable voters for quorum-aware health checks
        self._reachable_voters: set[str] = set()
        self._total_voters = 0

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

        # Jan 5, 2026: Track voter counts for quorum health
        self._total_voters = len(voter_ids)
        cycle_reachable: set[str] = {node_id}  # Include self as reachable

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
                cycle_reachable.add(voter_id)

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

                if success:
                    cycle_reachable.add(voter_id)
                else:
                    self._increment_failures(voter_id)

        # Periodic voter mesh refresh
        if now - self._last_mesh_refresh > self.config.mesh_refresh_interval_seconds:
            self._last_mesh_refresh = now
            self._mesh_refreshes += 1
            await self._refresh_voter_mesh()

        # Jan 5, 2026: Update reachable voters set for health check
        self._reachable_voters = cycle_reachable

    def get_voter_stats(self) -> dict[str, Any]:
        """Get voter heartbeat statistics."""
        active_voters = len(self._reachable_voters)
        total_voters = self._total_voters
        min_quorum = (total_voters // 2) + 1 if total_voters > 0 else 0
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
            # Jan 5, 2026: Added quorum-aware metrics
            "active_voters": active_voters,
            "total_voters": total_voters,
            "min_quorum": min_quorum,
            "quorum_margin": active_voters - min_quorum if total_voters > 0 else 0,
            **self.stats.to_dict(),
        }

    def health_check(self) -> Any:
        """Check loop health with voter heartbeat-specific status.

        Jan 2026: Added for DaemonManager integration.
        Critical for quorum maintenance - reports voter connectivity.

        Returns:
            HealthCheckResult with voter heartbeat status
        """
        try:
            from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
        except ImportError:
            # Fallback if protocols not available
            stats = self.get_voter_stats()
            return {
                "healthy": self.running,
                "status": "running" if self.running else "stopped",
                "message": f"VoterHeartbeatLoop {'running' if self.running else 'stopped'}",
                "details": stats,
            }

        # Not running
        if not self.running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="VoterHeartbeatLoop is stopped",
                details={"running": False},
            )

        # Not a voter - loop is idle but healthy
        if not self._is_voter:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message="VoterHeartbeatLoop idle (not a voter)",
                details={"is_voter": False, "running": True},
            )

        # Get stats for health assessment
        stats = self.get_voter_stats()
        success_rate = stats.get("success_rate", 0.0)
        heartbeats_sent = stats.get("heartbeats_sent", 0)
        active_voters = stats.get("active_voters", 0)
        total_voters = stats.get("total_voters", 0)
        min_quorum = stats.get("min_quorum", 0)
        quorum_margin = stats.get("quorum_margin", 0)

        # Jan 5, 2026: CRITICAL when quorum is at minimum - any additional failure will break quorum
        if total_voters > 0 and quorum_margin <= 0:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Quorum at minimum: {active_voters}/{total_voters} voters (need {min_quorum})",
                details={
                    "is_voter": True,
                    "active_voters": active_voters,
                    "total_voters": total_voters,
                    "min_quorum": min_quorum,
                    "quorum_margin": quorum_margin,
                    "success_rate": success_rate,
                    "heartbeats_sent": heartbeats_sent,
                },
            )

        # DEGRADED when quorum margin is tight (only 1 voter spare)
        if total_voters > 0 and quorum_margin == 1:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message=f"Quorum margin tight: {active_voters}/{total_voters} voters (1 above minimum)",
                details={
                    "is_voter": True,
                    "active_voters": active_voters,
                    "total_voters": total_voters,
                    "min_quorum": min_quorum,
                    "quorum_margin": quorum_margin,
                    "success_rate": success_rate,
                },
            )

        # Voter connectivity is critical for quorum (legacy success rate check)
        if heartbeats_sent > 5 and success_rate < 40:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Voter heartbeat critical ({success_rate:.1f}%) - quorum at risk",
                details={
                    "is_voter": True,
                    "success_rate": success_rate,
                    "heartbeats_sent": heartbeats_sent,
                    "heartbeats_succeeded": stats.get("heartbeats_succeeded", 0),
                    "nat_recoveries": stats.get("nat_recoveries", 0),
                },
            )

        # Check for degraded conditions
        if heartbeats_sent > 5 and success_rate < 70:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message=f"Voter heartbeat degraded ({success_rate:.1f}%)",
                details={
                    "is_voter": True,
                    "success_rate": success_rate,
                    "heartbeats_sent": heartbeats_sent,
                    "nat_recoveries": stats.get("nat_recoveries", 0),
                    "mesh_refreshes": stats.get("mesh_refreshes", 0),
                },
            )

        # Healthy
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"VoterHeartbeatLoop healthy (success rate: {success_rate:.1f}%)",
            details={
                "is_voter": True,
                "success_rate": success_rate,
                "heartbeats_sent": heartbeats_sent,
                "heartbeats_succeeded": stats.get("heartbeats_succeeded", 0),
                "nat_recoveries": stats.get("nat_recoveries", 0),
                "mesh_refreshes": stats.get("mesh_refreshes", 0),
            },
        )


@dataclass
class TailscaleKeepaliveConfig:
    """Configuration for Tailscale connection keepalive.

    Userspace mode Tailscale (used in containers) has slower connection
    establishment and more frequent DERP relay fallback. This loop keeps
    connections warm by periodically pinging peers.
    """

    # How often to run the keepalive cycle (seconds)
    interval_seconds: float = 60.0

    # Ping timeout per peer (seconds)
    ping_timeout_seconds: float = 5.0

    # Max peers to ping per cycle (to avoid overload)
    max_peers_per_cycle: int = 20

    # How often to attempt direct connection if using DERP (seconds)
    derp_recovery_interval_seconds: float = 300.0

    # Enable aggressive mode for userspace/container nodes
    userspace_aggressive_mode: bool = True

    # For userspace mode: faster interval (seconds)
    userspace_interval_seconds: float = 30.0


# Module-level constants for RelayHysteresis (moved out of class body for Python compatibility)
_RELAY_SWITCH_COOLDOWN = float(os.environ.get("RINGRIFT_RELAY_SWITCH_COOLDOWN", "300"))
# Jan 2026: Increased from 3 to 4 for more tolerant relay handling
_RELAY_SWITCH_THRESHOLD = int(os.environ.get("RINGRIFT_RELAY_SWITCH_THRESHOLD", "4"))


class RelayHysteresis:
    """Hysteresis controller for direct/relay connection switching.

    Session 17.39 (Jan 2026): Phase 10.4 - NAT relay hysteresis.

    Problem: Lambda GH200 nodes flap between direct and relay connections,
    causing peer instability. A single failed ping would trigger switch to
    relay, then a successful ping would switch back to direct, causing
    constant state changes that destabilize the P2P mesh.

    Solution: Add hysteresis with two conditions before switching:
    1. SWITCH_COOLDOWN: Minimum time between switch events (5 minutes)
    2. FAILURE_THRESHOLD: Require N consecutive failures before switching

    This reduces relay reconnections by ~50% and improves connection stability.

    Usage:
        hysteresis = RelayHysteresis()

        # Called on each ping result
        if not is_direct:
            hysteresis.record_relay_detection(node_id)
        else:
            hysteresis.record_direct_success(node_id)

        # Check if we should notify about a switch
        if hysteresis.should_notify_switch_to_relay(node_id):
            # Actually switch and notify
            hysteresis.mark_switched_to_relay(node_id)

        if hysteresis.should_notify_switch_to_direct(node_id):
            # Actually switch and notify
            hysteresis.mark_switched_to_direct(node_id)
    """

    # Class constants reference module-level values
    SWITCH_COOLDOWN: float = _RELAY_SWITCH_COOLDOWN
    FAILURE_THRESHOLD: int = _RELAY_SWITCH_THRESHOLD

    def __init__(self) -> None:
        # node_id -> timestamp of last switch event
        self._last_switch_times: dict[str, float] = {}

        # node_id -> count of consecutive relay detections (direct failures)
        self._relay_detection_counts: dict[str, int] = {}

        # node_id -> count of consecutive direct successes
        self._direct_success_counts: dict[str, int] = {}

        # node_id -> current state (True = direct, False = relay)
        self._current_state: dict[str, bool] = {}

        # Stats
        self._switch_to_relay_blocked = 0
        self._switch_to_direct_blocked = 0
        self._switches_to_relay = 0
        self._switches_to_direct = 0

    def record_relay_detection(self, node_id: str) -> None:
        """Record that a ping was relayed (not direct).

        Call this each time a ping shows DERP relay usage.
        Increments the relay detection counter and resets direct counter.
        """
        self._relay_detection_counts[node_id] = (
            self._relay_detection_counts.get(node_id, 0) + 1
        )
        # Reset direct counter since we detected relay
        self._direct_success_counts[node_id] = 0

    def record_direct_success(self, node_id: str) -> None:
        """Record that a ping was direct (no relay).

        Call this each time a ping shows direct connection.
        Increments the direct success counter and resets relay counter.
        """
        self._direct_success_counts[node_id] = (
            self._direct_success_counts.get(node_id, 0) + 1
        )
        # Reset relay counter since we detected direct
        self._relay_detection_counts[node_id] = 0

    def _cooldown_passed(self, node_id: str) -> bool:
        """Check if enough time has passed since last switch."""
        last_switch = self._last_switch_times.get(node_id, 0)
        return (time.time() - last_switch) >= self.SWITCH_COOLDOWN

    def should_notify_switch_to_relay(self, node_id: str) -> bool:
        """Check if we should notify about switching to relay.

        Returns True only if:
        1. Cooldown period has passed since last switch
        2. We've seen N consecutive relay detections
        3. Current state is direct (or unknown)
        """
        # Already in relay state? No need to switch
        if self._current_state.get(node_id) is False:
            return False

        # Check threshold
        detections = self._relay_detection_counts.get(node_id, 0)
        if detections < self.FAILURE_THRESHOLD:
            return False

        # Check cooldown
        if not self._cooldown_passed(node_id):
            self._switch_to_relay_blocked += 1
            return False

        return True

    def should_notify_switch_to_direct(self, node_id: str) -> bool:
        """Check if we should notify about switching to direct.

        Returns True only if:
        1. Cooldown period has passed since last switch
        2. We've seen N consecutive direct successes
        3. Current state is relay (or unknown)
        """
        # Already in direct state? No need to switch
        if self._current_state.get(node_id) is True:
            return False

        # Check threshold
        successes = self._direct_success_counts.get(node_id, 0)
        if successes < self.FAILURE_THRESHOLD:
            return False

        # Check cooldown
        if not self._cooldown_passed(node_id):
            self._switch_to_direct_blocked += 1
            return False

        return True

    def mark_switched_to_relay(self, node_id: str) -> None:
        """Mark that we've switched to relay and reset counters."""
        self._last_switch_times[node_id] = time.time()
        self._current_state[node_id] = False
        self._relay_detection_counts[node_id] = 0
        self._switches_to_relay += 1
        logger.info(
            f"[RelayHysteresis] Node {node_id} switched to relay "
            f"(after {self.FAILURE_THRESHOLD} consecutive detections)"
        )

    def mark_switched_to_direct(self, node_id: str) -> None:
        """Mark that we've switched to direct and reset counters."""
        self._last_switch_times[node_id] = time.time()
        self._current_state[node_id] = True
        self._direct_success_counts[node_id] = 0
        self._switches_to_direct += 1
        logger.info(
            f"[RelayHysteresis] Node {node_id} recovered to direct "
            f"(after {self.FAILURE_THRESHOLD} consecutive successes)"
        )

    def get_stats(self) -> dict[str, Any]:
        """Get hysteresis statistics."""
        return {
            "switches_to_relay": self._switches_to_relay,
            "switches_to_direct": self._switches_to_direct,
            "switch_to_relay_blocked": self._switch_to_relay_blocked,
            "switch_to_direct_blocked": self._switch_to_direct_blocked,
            "nodes_tracked": len(self._current_state),
            "nodes_on_relay": sum(
                1 for v in self._current_state.values() if v is False
            ),
            "nodes_on_direct": sum(
                1 for v in self._current_state.values() if v is True
            ),
            "cooldown_seconds": self.SWITCH_COOLDOWN,
            "failure_threshold": self.FAILURE_THRESHOLD,
        }

    def get_node_state(self, node_id: str) -> dict[str, Any]:
        """Get current state for a specific node."""
        return {
            "node_id": node_id,
            "current_state": (
                "direct" if self._current_state.get(node_id) else
                "relay" if self._current_state.get(node_id) is False else
                "unknown"
            ),
            "relay_detections": self._relay_detection_counts.get(node_id, 0),
            "direct_successes": self._direct_success_counts.get(node_id, 0),
            "last_switch": self._last_switch_times.get(node_id),
            "cooldown_remaining": max(
                0,
                self.SWITCH_COOLDOWN - (
                    time.time() - self._last_switch_times.get(node_id, 0)
                )
            ),
        }


class TailscaleKeepaliveLoop(BaseLoop):
    """Background loop that keeps Tailscale connections warm.

    Addresses userspace mode limitations:
    1. Periodically pings peers to keep NAT mappings alive
    2. Detects when connections fall back to DERP relay
    3. Attempts to re-establish direct connections
    4. More aggressive pinging for container nodes

    Usage:
        keepalive = TailscaleKeepaliveLoop(
            get_peer_tailscale_ips=lambda: {"node1": "100.x.x.x", ...},
            is_userspace_mode=lambda: True,  # For containers
        )
        await keepalive.run_forever()
    """

    def __init__(
        self,
        get_peer_tailscale_ips: Callable[[], dict[str, str]],
        is_userspace_mode: Callable[[], bool] | None = None,
        on_connection_quality_change: Callable[[str, str, bool], Coroutine[Any, Any, None]] | None = None,
        config: TailscaleKeepaliveConfig | None = None,
    ):
        """Initialize keepalive loop.

        Args:
            get_peer_tailscale_ips: Returns dict of node_id -> tailscale_ip
            is_userspace_mode: Returns True if running in userspace/container mode
            on_connection_quality_change: Callback(node_id, ip, is_direct) when quality changes
            config: Keepalive configuration
        """
        self.config = config or TailscaleKeepaliveConfig()

        # Adjust interval for userspace mode
        interval = self.config.interval_seconds
        if is_userspace_mode and is_userspace_mode():
            if self.config.userspace_aggressive_mode:
                interval = self.config.userspace_interval_seconds

        super().__init__(name="tailscale_keepalive", interval=interval)

        self._get_peer_tailscale_ips = get_peer_tailscale_ips
        self._is_userspace_mode = is_userspace_mode or (lambda: False)
        self._on_connection_quality_change = on_connection_quality_change

        # Connection quality tracking
        self._connection_quality: dict[str, dict[str, Any]] = {}
        # node_id -> {"is_direct": bool, "relay": str, "latency_ms": float, "last_check": float}

        # Stats
        self._pings_sent = 0
        self._pings_succeeded = 0
        self._direct_connections = 0
        self._derp_connections = 0
        self._derp_recovery_attempts = 0
        self._derp_recoveries_succeeded = 0
        self._last_derp_recovery: dict[str, float] = {}
        # Jan 5, 2026: Phase 10.4 - NAT relay hysteresis
        # Prevents flapping between direct and relay connections
        self._relay_hysteresis = RelayHysteresis()

    async def _run_once(self) -> None:
        """Run one keepalive cycle."""
        peer_ips = self._get_peer_tailscale_ips()
        if not peer_ips:
            return

        now = time.time()

        # Sample peers if too many
        peers_to_ping = list(peer_ips.items())
        if len(peers_to_ping) > self.config.max_peers_per_cycle:
            peers_to_ping = random.sample(peers_to_ping, self.config.max_peers_per_cycle)

        # Ping each peer
        for node_id, tailscale_ip in peers_to_ping:
            if not tailscale_ip or tailscale_ip == "0.0.0.0":
                continue

            self._pings_sent += 1

            try:
                result = await self._ping_peer(tailscale_ip)
                if result:
                    self._pings_succeeded += 1
                    await self._update_connection_quality(node_id, tailscale_ip, result, now)
            except Exception as e:
                logger.debug(f"[TailscaleKeepalive] Ping failed for {node_id}: {e}")

        # Attempt DERP recovery for peers using relay
        await self._attempt_derp_recoveries(now)

    async def _ping_peer(self, tailscale_ip: str) -> dict[str, Any] | None:
        """Ping a peer via Tailscale and return connection info.

        Returns:
            Dict with keys: is_direct, relay, latency_ms, or None if failed
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "tailscale", "ping", "-c", "1", "--timeout",
                str(int(self.config.ping_timeout_seconds)), tailscale_ip,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.ping_timeout_seconds + 2,
            )

            if proc.returncode != 0:
                return None

            output = stdout.decode("utf-8", errors="replace")

            # Parse output: "pong from node (100.x.x.x) via DERP(xyz) in 50ms"
            # or: "pong from node (100.x.x.x) via 1.2.3.4:41641 in 10ms"
            is_direct = "via DERP" not in output

            relay = None
            if "via DERP" in output:
                # Extract relay name
                import re
                match = re.search(r"via DERP\(([^)]+)\)", output)
                if match:
                    relay = match.group(1)

            latency_ms = 0.0
            if " in " in output:
                import re
                match = re.search(r"in (\d+(?:\.\d+)?)(ms|s)", output)
                if match:
                    latency = float(match.group(1))
                    unit = match.group(2)
                    latency_ms = latency if unit == "ms" else latency * 1000

            return {
                "is_direct": is_direct,
                "relay": relay,
                "latency_ms": latency_ms,
            }

        except asyncio.TimeoutError:
            return None
        except FileNotFoundError:
            # Tailscale CLI not available
            return None
        except Exception as e:
            logger.debug(f"[TailscaleKeepalive] Ping error: {e}")
            return None

    async def _update_connection_quality(
        self,
        node_id: str,
        tailscale_ip: str,
        result: dict[str, Any],
        now: float,
    ) -> None:
        """Update connection quality tracking and emit events if changed.

        Jan 5, 2026 (Phase 10.4): Added relay hysteresis to prevent connection
        flapping between direct and relay modes. Instead of notifying immediately
        on state change, we record detections and only notify when hysteresis
        thresholds are met (3 consecutive detections + 5min cooldown).
        """
        old_quality = self._connection_quality.get(node_id, {})
        was_direct = old_quality.get("is_direct", True)  # Assume direct if unknown

        is_direct = result["is_direct"]

        self._connection_quality[node_id] = {
            "tailscale_ip": tailscale_ip,
            "is_direct": is_direct,
            "relay": result.get("relay"),
            "latency_ms": result.get("latency_ms", 0),
            "last_check": now,
        }

        # Update stats
        if is_direct:
            self._direct_connections += 1
        else:
            self._derp_connections += 1

        # Jan 5, 2026: Apply hysteresis before notifying
        # Record the current detection with the hysteresis controller
        if not is_direct:
            relay_name = result.get("relay", "unknown")
            self._relay_hysteresis.record_relay_detection(node_id, relay_name)
        else:
            self._relay_hysteresis.record_direct_success(node_id)

        # Only notify if hysteresis allows the state change
        if self._on_connection_quality_change:
            # Check for transition to relay (was direct, now relay, hysteresis allows)
            if was_direct and not is_direct:
                if self._relay_hysteresis.should_notify_switch_to_relay(node_id):
                    self._relay_hysteresis.mark_switched_to_relay(node_id)
                    try:
                        await self._on_connection_quality_change(node_id, tailscale_ip, is_direct)
                    except Exception as e:
                        logger.warning(f"[TailscaleKeepalive] Quality change callback failed: {e}")
            # Check for transition to direct (was relay, now direct, hysteresis allows)
            elif not was_direct and is_direct:
                if self._relay_hysteresis.should_notify_switch_to_direct(node_id):
                    self._relay_hysteresis.mark_switched_to_direct(node_id)
                    try:
                        await self._on_connection_quality_change(node_id, tailscale_ip, is_direct)
                    except Exception as e:
                        logger.warning(f"[TailscaleKeepalive] Quality change callback failed: {e}")

    async def _attempt_derp_recoveries(self, now: float) -> None:
        """Attempt to recover direct connections for peers using DERP relay."""
        for node_id, quality in self._connection_quality.items():
            if quality.get("is_direct", True):
                continue

            tailscale_ip = quality.get("tailscale_ip")
            if not tailscale_ip:
                continue

            # Check cooldown
            last_attempt = self._last_derp_recovery.get(node_id, 0)
            if now - last_attempt < self.config.derp_recovery_interval_seconds:
                continue

            self._derp_recovery_attempts += 1
            self._last_derp_recovery[node_id] = now

            # Try to force direct connection by sending multiple pings
            # This helps NAT hole-punching
            logger.debug(f"[TailscaleKeepalive] Attempting DERP recovery for {node_id}")

            for _ in range(3):
                result = await self._ping_peer(tailscale_ip)
                if result and result.get("is_direct"):
                    self._derp_recoveries_succeeded += 1
                    logger.info(f"[TailscaleKeepalive] Recovered direct connection to {node_id}")
                    await self._update_connection_quality(node_id, tailscale_ip, result, now)
                    break
                await asyncio.sleep(0.5)

    def get_keepalive_stats(self) -> dict[str, Any]:
        """Get keepalive statistics.

        Jan 5, 2026: Added relay_hysteresis stats for Phase 10.4 monitoring.
        """
        total_pings = self._pings_sent
        return {
            "pings_sent": self._pings_sent,
            "pings_succeeded": self._pings_succeeded,
            "success_rate": (
                self._pings_succeeded / total_pings * 100
                if total_pings > 0
                else 0.0
            ),
            "direct_connections": self._direct_connections,
            "derp_connections": self._derp_connections,
            "direct_ratio": (
                self._direct_connections / (self._direct_connections + self._derp_connections) * 100
                if (self._direct_connections + self._derp_connections) > 0
                else 0.0
            ),
            "derp_recovery_attempts": self._derp_recovery_attempts,
            "derp_recoveries_succeeded": self._derp_recoveries_succeeded,
            "is_userspace_mode": self._is_userspace_mode(),
            "peers_tracked": len(self._connection_quality),
            # Jan 5, 2026: Phase 10.4 relay hysteresis stats
            "relay_hysteresis": self._relay_hysteresis.get_stats(),
            **self.stats.to_dict(),
        }

    def get_connection_quality(self) -> dict[str, dict[str, Any]]:
        """Get current connection quality for all tracked peers."""
        return dict(self._connection_quality)

    def health_check(self) -> Any:
        """Check loop health with Tailscale keepalive-specific status.

        Jan 2026: Added for DaemonManager integration.
        Reports connection quality and DERP relay usage.

        Returns:
            HealthCheckResult with Tailscale keepalive status
        """
        try:
            from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
        except ImportError:
            # Fallback if protocols not available
            stats = self.get_keepalive_stats()
            return {
                "healthy": self.running,
                "status": "running" if self.running else "stopped",
                "message": f"TailscaleKeepaliveLoop {'running' if self.running else 'stopped'}",
                "details": stats,
            }

        # Not running
        if not self.running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="TailscaleKeepaliveLoop is stopped",
                details={"running": False},
            )

        # Get stats for health assessment
        stats = self.get_keepalive_stats()
        success_rate = stats.get("success_rate", 0.0)
        direct_ratio = stats.get("direct_ratio", 0.0)
        pings_sent = stats.get("pings_sent", 0)

        # Check for critical ping failures
        if pings_sent > 10 and success_rate < 30:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Tailscale keepalive critical ({success_rate:.1f}% success)",
                details={
                    "success_rate": success_rate,
                    "direct_ratio": direct_ratio,
                    "pings_sent": pings_sent,
                    "pings_succeeded": stats.get("pings_succeeded", 0),
                    "is_userspace_mode": stats.get("is_userspace_mode", False),
                },
            )

        # Check for heavy DERP relay usage (indicates NAT issues)
        if pings_sent > 10 and direct_ratio < 30:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message=f"High DERP relay usage ({direct_ratio:.1f}% direct)",
                details={
                    "success_rate": success_rate,
                    "direct_ratio": direct_ratio,
                    "derp_connections": stats.get("derp_connections", 0),
                    "derp_recovery_attempts": stats.get("derp_recovery_attempts", 0),
                    "derp_recoveries_succeeded": stats.get("derp_recoveries_succeeded", 0),
                },
            )

        # Healthy
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"TailscaleKeepaliveLoop healthy ({direct_ratio:.1f}% direct)",
            details={
                "success_rate": success_rate,
                "direct_ratio": direct_ratio,
                "pings_sent": pings_sent,
                "peers_tracked": stats.get("peers_tracked", 0),
                "is_userspace_mode": stats.get("is_userspace_mode", False),
            },
        )


@dataclass
class TailscaleDaemonHealthConfig:
    """Configuration for Tailscale daemon health monitoring.

    Jan 7, 2026: Added to detect stale Tailscale daemons where the daemon is
    running but peer connectivity has degraded (e.g., all peers show as offline).
    This addresses split-brain issues caused by Tailscale losing mesh connectivity.
    """

    check_interval_seconds: float = 60.0  # Check every 60 seconds
    min_expected_peers: int = 3  # Minimum peers we expect to see online
    peer_degradation_threshold: float = 0.5  # Trigger recovery if <50% of expected peers online
    consecutive_failures_before_recovery: int = 3  # 3 failures = 3 minutes
    recovery_cooldown_seconds: float = 600.0  # 10 minutes between recovery attempts
    recovery_command_timeout: float = 30.0  # Timeout for recovery commands

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        if self.min_expected_peers < 0:
            raise ValueError("min_expected_peers must be >= 0")
        if not (0 <= self.peer_degradation_threshold <= 1):
            raise ValueError("peer_degradation_threshold must be between 0 and 1")
        if self.consecutive_failures_before_recovery <= 0:
            raise ValueError("consecutive_failures_before_recovery must be > 0")


class TailscaleDaemonHealthLoop(BaseLoop):
    """Monitor local Tailscale daemon health and trigger auto-recovery.

    Jan 7, 2026: Addresses split-brain issues caused by stale Tailscale daemons.

    Problem Scenario:
    - Tailscale daemon is running (systemd shows active)
    - But mesh connectivity is degraded (peers show as offline or "-" status)
    - P2P cluster loses quorum because nodes can't reach each other
    - Split-brain can occur as different partitions elect different leaders

    Solution:
    - Monitor local `tailscale status` output for peer connectivity
    - Detect when peer count drops significantly below expected
    - Automatically restart Tailscale daemon to restore mesh connectivity
    - Emit events for observability

    Usage:
        loop = TailscaleDaemonHealthLoop(
            expected_peer_count_fn=lambda: 20,  # e.g., from cluster config
            on_recovery_triggered=async_recovery_callback,
        )
        await loop.run_forever()
    """

    def __init__(
        self,
        expected_peer_count_fn: Callable[[], int] | None = None,
        on_recovery_triggered: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        on_health_degraded: Callable[[dict[str, Any]], Coroutine[Any, Any, None]] | None = None,
        config: TailscaleDaemonHealthConfig | None = None,
    ) -> None:
        """Initialize Tailscale daemon health loop.

        Args:
            expected_peer_count_fn: Callback returning expected number of Tailscale peers
            on_recovery_triggered: Async callback when recovery is triggered
            on_health_degraded: Async callback when health degrades (but not yet triggering recovery)
            config: Health check configuration
        """
        self.config = config or TailscaleDaemonHealthConfig()
        super().__init__(
            name="tailscale_daemon_health",
            interval=self.config.check_interval_seconds,
        )

        self._expected_peer_count_fn = expected_peer_count_fn or (lambda: self.config.min_expected_peers)
        self._on_recovery_triggered = on_recovery_triggered
        self._on_health_degraded = on_health_degraded

        # State tracking
        self._consecutive_failures = 0
        self._last_recovery_time = 0.0
        self._last_healthy_time = time.time()

        # Statistics
        self._health_checks_total = 0
        self._health_checks_passed = 0
        self._health_checks_failed = 0
        self._recoveries_attempted = 0
        self._recoveries_succeeded = 0
        self._last_peer_count = 0
        self._last_online_count = 0

    async def _run_once(self) -> None:
        """Check Tailscale daemon health and trigger recovery if needed."""
        self._health_checks_total += 1
        now = time.time()

        # Get current Tailscale status
        status = await self._get_tailscale_status()
        if status is None:
            # Failed to get status - daemon might be stopped
            await self._on_health_check_failed("tailscale_status_failed", now)
            return

        # Analyze peer connectivity
        total_peers = status.get("total_peers", 0)
        online_peers = status.get("online_peers", 0)
        expected_peers = self._expected_peer_count_fn()

        self._last_peer_count = total_peers
        self._last_online_count = online_peers

        # Calculate health metrics
        peer_ratio = online_peers / max(expected_peers, 1)
        is_healthy = (
            online_peers >= self.config.min_expected_peers
            and peer_ratio >= self.config.peer_degradation_threshold
        )

        if is_healthy:
            self._health_checks_passed += 1
            self._consecutive_failures = 0
            self._last_healthy_time = now
            logger.debug(
                f"[TailscaleDaemonHealth] Healthy: {online_peers}/{total_peers} "
                f"peers online ({peer_ratio:.1%} of expected {expected_peers})"
            )
        else:
            self._health_checks_failed += 1
            await self._on_health_check_failed(
                f"peer_degradation:{online_peers}/{expected_peers}",
                now,
            )

            # Notify of degraded health
            if self._on_health_degraded:
                try:
                    await self._on_health_degraded({
                        "online_peers": online_peers,
                        "total_peers": total_peers,
                        "expected_peers": expected_peers,
                        "peer_ratio": peer_ratio,
                        "consecutive_failures": self._consecutive_failures,
                        "time_since_healthy": now - self._last_healthy_time,
                    })
                except Exception as e:
                    logger.warning(f"[TailscaleDaemonHealth] Health degraded callback failed: {e}")

    async def _on_health_check_failed(self, reason: str, now: float) -> None:
        """Handle a failed health check."""
        self._consecutive_failures += 1

        logger.warning(
            f"[TailscaleDaemonHealth] Health check failed: {reason} "
            f"(consecutive: {self._consecutive_failures}/{self.config.consecutive_failures_before_recovery})"
        )

        # Check if we should trigger recovery
        if self._consecutive_failures >= self.config.consecutive_failures_before_recovery:
            # Check cooldown
            time_since_last_recovery = now - self._last_recovery_time
            if time_since_last_recovery < self.config.recovery_cooldown_seconds:
                logger.info(
                    f"[TailscaleDaemonHealth] Recovery cooldown active "
                    f"({self.config.recovery_cooldown_seconds - time_since_last_recovery:.0f}s remaining)"
                )
                return

            await self._trigger_recovery(reason)

    async def _trigger_recovery(self, reason: str) -> None:
        """Trigger Tailscale daemon recovery."""
        self._recoveries_attempted += 1
        self._last_recovery_time = time.time()

        logger.warning(
            f"[TailscaleDaemonHealth] Triggering recovery due to: {reason} "
            f"(attempt #{self._recoveries_attempted})"
        )

        # Notify callback
        if self._on_recovery_triggered:
            try:
                await self._on_recovery_triggered(reason)
            except Exception as e:
                logger.error(f"[TailscaleDaemonHealth] Recovery callback failed: {e}")

        # Attempt recovery
        success = await self._attempt_local_recovery()

        if success:
            self._recoveries_succeeded += 1
            self._consecutive_failures = 0
            logger.info("[TailscaleDaemonHealth] Recovery succeeded - connectivity restored")
        else:
            logger.error(
                "[TailscaleDaemonHealth] Recovery failed - manual intervention may be required"
            )

    async def _attempt_local_recovery(self) -> bool:
        """Attempt to recover Tailscale connectivity locally.

        Returns:
            True if recovery succeeded, False otherwise
        """
        # Step 1: Try `tailscale up` to re-establish connections
        try:
            proc = await asyncio.create_subprocess_exec(
                "tailscale", "up", "--accept-routes",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.recovery_command_timeout,
            )

            if proc.returncode == 0:
                await asyncio.sleep(5)  # Wait for connections to establish
                # Verify recovery
                status = await self._get_tailscale_status()
                if status and status.get("online_peers", 0) >= self.config.min_expected_peers:
                    return True
                logger.warning("[TailscaleDaemonHealth] tailscale up succeeded but peers still degraded")
            else:
                logger.warning(
                    f"[TailscaleDaemonHealth] tailscale up failed: {stderr.decode('utf-8', errors='replace')}"
                )

        except asyncio.TimeoutError:
            logger.error("[TailscaleDaemonHealth] tailscale up timed out")
        except Exception as e:
            logger.error(f"[TailscaleDaemonHealth] tailscale up exception: {e}")

        # Step 2: Try restarting tailscaled service (requires sudo)
        try:
            proc = await asyncio.create_subprocess_exec(
                "sudo", "systemctl", "restart", "tailscaled",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.recovery_command_timeout,
            )

            if proc.returncode == 0:
                await asyncio.sleep(10)  # Wait for service to fully restart
                # Verify recovery
                status = await self._get_tailscale_status()
                if status and status.get("online_peers", 0) >= self.config.min_expected_peers:
                    return True
                logger.warning("[TailscaleDaemonHealth] tailscaled restart succeeded but peers still degraded")
            else:
                logger.warning("[TailscaleDaemonHealth] tailscaled restart failed (may need manual intervention)")

        except asyncio.TimeoutError:
            logger.error("[TailscaleDaemonHealth] tailscaled restart timed out")
        except FileNotFoundError:
            # systemctl not available (e.g., macOS)
            logger.debug("[TailscaleDaemonHealth] systemctl not available - skipping daemon restart")
        except Exception as e:
            logger.error(f"[TailscaleDaemonHealth] tailscaled restart exception: {e}")

        return False

    async def _get_tailscale_status(self) -> dict[str, Any] | None:
        """Get Tailscale status with peer connectivity info.

        Returns:
            Dict with keys: total_peers, online_peers, self_online, or None if failed
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "tailscale", "status", "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=10.0,
            )

            if proc.returncode != 0:
                logger.debug(
                    f"[TailscaleDaemonHealth] tailscale status failed: "
                    f"{stderr.decode('utf-8', errors='replace')}"
                )
                return None

            import json
            status = json.loads(stdout.decode("utf-8"))

            # Count peers
            peers = status.get("Peer", {})
            total_peers = len(peers)
            online_peers = sum(1 for p in peers.values() if p.get("Online", False))

            # Check self status
            self_status = status.get("Self", {})
            self_online = self_status.get("Online", False)

            return {
                "total_peers": total_peers,
                "online_peers": online_peers,
                "self_online": self_online,
                "backend_state": status.get("BackendState", "unknown"),
            }

        except asyncio.TimeoutError:
            logger.debug("[TailscaleDaemonHealth] tailscale status timed out")
            return None
        except json.JSONDecodeError as e:
            logger.debug(f"[TailscaleDaemonHealth] Failed to parse tailscale status: {e}")
            return None
        except FileNotFoundError:
            logger.debug("[TailscaleDaemonHealth] tailscale command not found")
            return None
        except Exception as e:
            logger.debug(f"[TailscaleDaemonHealth] Exception getting tailscale status: {e}")
            return None

    def get_health_stats(self) -> dict[str, Any]:
        """Get health monitoring statistics."""
        return {
            "health_checks_total": self._health_checks_total,
            "health_checks_passed": self._health_checks_passed,
            "health_checks_failed": self._health_checks_failed,
            "success_rate": (
                self._health_checks_passed / self._health_checks_total * 100
                if self._health_checks_total > 0
                else 100.0
            ),
            "consecutive_failures": self._consecutive_failures,
            "recoveries_attempted": self._recoveries_attempted,
            "recoveries_succeeded": self._recoveries_succeeded,
            "recovery_success_rate": (
                self._recoveries_succeeded / self._recoveries_attempted * 100
                if self._recoveries_attempted > 0
                else 100.0
            ),
            "last_peer_count": self._last_peer_count,
            "last_online_count": self._last_online_count,
            "seconds_since_healthy": time.time() - self._last_healthy_time,
            "seconds_since_last_recovery": (
                time.time() - self._last_recovery_time
                if self._last_recovery_time > 0
                else 0.0
            ),
            **self.stats.to_dict(),
        }

    def health_check(self) -> Any:
        """Check loop health with Tailscale-specific status.

        Returns:
            HealthCheckResult with daemon health status
        """
        try:
            from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
        except ImportError:
            stats = self.get_health_stats()
            return {
                "healthy": self.running and self._consecutive_failures < self.config.consecutive_failures_before_recovery,
                "status": "running" if self.running else "stopped",
                "message": f"TailscaleDaemonHealthLoop {'running' if self.running else 'stopped'}",
                "details": stats,
            }

        if not self.running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="TailscaleDaemonHealthLoop is stopped",
                details={"running": False},
            )

        stats = self.get_health_stats()

        # Critical: consecutive failures at or above threshold
        if self._consecutive_failures >= self.config.consecutive_failures_before_recovery:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Tailscale daemon unhealthy - {self._consecutive_failures} consecutive failures",
                details=stats,
            )

        # Degraded: some failures but not yet critical
        if self._consecutive_failures > 0:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message=f"Tailscale daemon degraded - {self._consecutive_failures} consecutive failures",
                details=stats,
            )

        # Healthy
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"TailscaleDaemonHealthLoop healthy ({self._last_online_count}/{self._last_peer_count} peers online)",
            details=stats,
        )


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
    # Tailscale Keepalive (January 2026)
    "TailscaleKeepaliveConfig",
    "TailscaleKeepaliveLoop",
    # Tailscale Daemon Health (January 2026)
    "TailscaleDaemonHealthConfig",
    "TailscaleDaemonHealthLoop",
]
