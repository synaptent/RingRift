"""Network Management Loops for P2P Orchestrator.

December 2025: Background loops for network health and recovery.

Loops:
- IpDiscoveryLoop: Updates node IP addresses as they change
- TailscaleRecoveryLoop: Recovers Tailscale connections when they fail

Usage:
    from scripts.p2p.loops import IpDiscoveryLoop, TailscaleRecoveryLoop

    ip_loop = IpDiscoveryLoop(
        get_nodes=lambda: orchestrator.peer_status,
        update_node_ip=orchestrator.update_node_ip,
    )
    await ip_loop.run_forever()
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


__all__ = [
    "IpDiscoveryConfig",
    "IpDiscoveryLoop",
    "NATManagementConfig",
    "NATManagementLoop",
    "TailscaleRecoveryConfig",
    "TailscaleRecoveryLoop",
]
