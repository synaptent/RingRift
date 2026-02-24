"""Network Configuration Manager for P2P Orchestrator.

January 2026: Phase 3 P2P Orchestrator Deep Decomposition

This module extracts IP discovery, validation, and network configuration
from p2p_orchestrator.py for better modularity and testability.

Responsibilities:
- IP address discovery (IPv4, IPv6, Tailscale)
- Advertise host validation and auto-fixing
- Alternate IP collection for multi-path connectivity
- Periodic IP revalidation for late Tailscale availability

Usage:
    from scripts.p2p.managers.network_config_manager import (
        NetworkConfigManager,
        NetworkConfig,
        get_network_config_manager,
    )

    # Create with config
    manager = NetworkConfigManager(
        node_id="my-node",
        initial_host="100.64.0.1",
        initial_port=8770,
        config=NetworkConfig(),
    )

    # Validate and fix advertise host
    manager.validate_and_fix_advertise_host()

    # Get all discovered IPs
    all_ips = manager.discover_all_ips()

    # Access current network state
    print(f"Primary host: {manager.advertise_host}")
    print(f"Alternate IPs: {manager.alternate_ips}")
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
import threading

from app.core.async_context import safe_create_task
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable

logger = logging.getLogger(__name__)

# Module-level singleton
_network_config_manager: NetworkConfigManager | None = None
_manager_lock = threading.Lock()


@dataclass
class NetworkConfig:
    """Configuration for network config manager.

    Attributes:
        prefer_ipv4: Prefer IPv4 over IPv6 for primary host
        include_private_ips: Include private IPs in discovery
        tailscale_ipv4_prefix: Tailscale CGNAT IPv4 prefix
        tailscale_ipv6_prefix: Tailscale IPv6 prefix
        revalidation_interval: Seconds between periodic IP revalidation
        initial_revalidation_delay: Delay before first revalidation
    """

    prefer_ipv4: bool = True
    include_private_ips: bool = False
    tailscale_ipv4_prefix: str = "100."
    tailscale_ipv6_prefix: str = "fd7a:115c:a1e0:"
    revalidation_interval: float = 300.0
    initial_revalidation_delay: float = 30.0


@dataclass
class NetworkState:
    """Current network state for this node.

    Attributes:
        advertise_host: Primary IP to advertise to peers
        advertise_port: Port to advertise to peers
        alternate_ips: Set of alternate IPs for multi-path connectivity
        has_tailscale: Whether Tailscale is available
        last_validation_time: Timestamp of last validation
    """

    advertise_host: str = ""
    advertise_port: int = 8770
    alternate_ips: set[str] = field(default_factory=set)
    has_tailscale: bool = False
    last_validation_time: float = 0.0


class NetworkConfigManager:
    """Manages network configuration for P2P node.

    Handles IP discovery, validation, and alternate IP collection
    for robust mesh connectivity.

    Thread-safe: All public methods are protected by lock.
    """

    def __init__(
        self,
        node_id: str,
        initial_host: str = "",
        initial_port: int = 8770,
        config: NetworkConfig | None = None,
        on_host_changed: Callable[[str, str], None] | None = None,
    ):
        """Initialize network config manager.

        Args:
            node_id: This node's identifier
            initial_host: Initial advertise host (may be auto-fixed)
            initial_port: Port for P2P communication
            config: Network configuration
            on_host_changed: Callback when advertise_host changes (old, new)
        """
        self._node_id = node_id
        self._config = config or NetworkConfig()
        self._on_host_changed = on_host_changed
        self._lock = threading.RLock()

        self._state = NetworkState(
            advertise_host=initial_host,
            advertise_port=initial_port,
        )

        # Revalidation loop control
        self._revalidation_task: asyncio.Task | None = None
        self._running = False

    @property
    def node_id(self) -> str:
        """Get node identifier."""
        return self._node_id

    @property
    def advertise_host(self) -> str:
        """Get current advertise host."""
        with self._lock:
            return self._state.advertise_host

    @advertise_host.setter
    def advertise_host(self, value: str) -> None:
        """Set advertise host."""
        with self._lock:
            self._state.advertise_host = value

    @property
    def advertise_port(self) -> int:
        """Get current advertise port."""
        with self._lock:
            return self._state.advertise_port

    @property
    def alternate_ips(self) -> set[str]:
        """Get alternate IPs (copy for thread safety)."""
        with self._lock:
            return self._state.alternate_ips.copy()

    @alternate_ips.setter
    def alternate_ips(self, value: set[str]) -> None:
        """Set alternate IPs."""
        with self._lock:
            self._state.alternate_ips = value.copy() if value else set()

    @property
    def has_tailscale(self) -> bool:
        """Check if Tailscale is available."""
        with self._lock:
            return self._state.has_tailscale

    def validate_and_fix_advertise_host(self) -> bool:
        """Validate advertise_host, fix private IP issues, and populate alternate_ips.

        This method:
        1. Discovers all reachable IPs (both address families)
        2. Selects best primary (prefer Tailscale IPv4, then any IPv4, then IPv6)
        3. Populates alternate_ips with remaining addresses
        4. Emits warnings/errors for operator awareness if using private IP

        Returns:
            True if validation succeeded and host is valid, False otherwise
        """
        with self._lock:
            return self._validate_and_fix_internal()

    def _validate_and_fix_internal(self) -> bool:
        """Internal validation (must hold lock)."""
        import time

        all_ips = self._discover_all_ips_internal()

        if not all_ips:
            logger.warning(f"[NetworkConfig] No IPs discovered for {self._node_id}")
            return False

        # Check if current host is valid
        current_host = self._state.advertise_host
        if current_host and current_host in all_ips:
            # Check if current is IPv6 but we have IPv4 available
            is_current_ipv6 = ":" in current_host
            ipv4_ips = {ip for ip in all_ips if ":" not in ip}

            if is_current_ipv6 and ipv4_ips and self._config.prefer_ipv4:
                # Prefer IPv4 for primary
                primary, alternates = self._select_primary_host(all_ips)
                if primary and primary != current_host:
                    old_host = current_host
                    self._state.advertise_host = primary
                    self._state.alternate_ips = alternates
                    logger.info(
                        f"[NetworkConfig] Switched from IPv6 {old_host} to IPv4 {primary}"
                    )
                    self._notify_host_changed(old_host, primary)
                    return True

            # Current host is valid, just update alternates
            self._state.alternate_ips = all_ips - {current_host}
            logger.debug(
                f"[NetworkConfig] Updated alternate_ips: {len(self._state.alternate_ips)} addresses"
            )
            return True

        # Current host is not in discovered IPs - check if private/loopback
        if current_host:
            try:
                ip = ipaddress.ip_address(current_host)
                if ip.is_private or ip.is_loopback:
                    ip_type = "loopback" if ip.is_loopback else "private"
                    logger.warning(
                        f"[NetworkConfig] Current {current_host} is {ip_type}, selecting new primary"
                    )
            except ValueError:
                pass

        # Select best primary
        primary, alternates = self._select_primary_host(all_ips)

        if primary:
            old_host = current_host
            self._state.advertise_host = primary
            self._state.alternate_ips = alternates
            self._state.last_validation_time = time.time()

            if old_host and old_host != primary:
                logger.warning(
                    f"[NetworkConfig] Auto-fixed: {old_host} -> {primary} "
                    f"({len(alternates)} alternates)"
                )
                self._notify_host_changed(old_host, primary)
            else:
                logger.info(
                    f"[NetworkConfig] Host set to {primary} with {len(alternates)} alternates"
                )
            return True
        else:
            logger.error(
                f"[NetworkConfig] No valid IPs found for {self._node_id} - mesh will fail!"
            )
            return False

    def _notify_host_changed(self, old_host: str, new_host: str) -> None:
        """Notify callback if registered."""
        if self._on_host_changed:
            try:
                self._on_host_changed(old_host, new_host)
            except Exception as e:
                logger.debug(f"[NetworkConfig] Host change callback error: {e}")

    def discover_all_ips(self, exclude_primary: str | None = None) -> set[str]:
        """Discover all IP addresses this node can be reached at (IPv4 AND IPv6).

        Collects IPs from:
        1. Tailscale IPs (100.x.x.x IPv4, fd7a:115c:a1e0:: IPv6)
        2. Hostname resolution (both address families)
        3. Local network interfaces (both address families)
        4. YAML config (tailscale_ip, ssh_host if resolvable)

        Args:
            exclude_primary: IP to exclude from results

        Returns:
            Set of discovered IP addresses
        """
        with self._lock:
            return self._discover_all_ips_internal(exclude_primary)

    def _discover_all_ips_internal(self, exclude_primary: str | None = None) -> set[str]:
        """Internal IP discovery (must hold lock)."""
        ips: set[str] = set()

        # 1. Tailscale IPs (both IPv4 and IPv6)
        ips.update(self._discover_tailscale_ips())

        # 2. Hostname resolution
        ips.update(self._discover_hostname_ips())

        # 3. Network interfaces
        ips.update(self._discover_interface_ips())

        # 4. YAML config
        ips.update(self._discover_config_ips())

        # Remove excluded and loopback
        if exclude_primary and exclude_primary in ips:
            ips.discard(exclude_primary)

        ips = {
            ip for ip in ips
            if not ip.startswith("127.") and ip not in ("0.0.0.0", "::1")
        }

        # Update Tailscale availability flag
        self._state.has_tailscale = any(
            ip.startswith(self._config.tailscale_ipv4_prefix) or
            ip.startswith(self._config.tailscale_ipv6_prefix)
            for ip in ips
        )

        logger.debug(f"[NetworkConfig] Discovered IPs: {ips}")
        return ips

    def _discover_tailscale_ips(self) -> set[str]:
        """Discover Tailscale IPs."""
        ips: set[str] = set()
        try:
            from scripts.p2p.resource_detector import ResourceDetector
            detector = ResourceDetector()

            ts_ipv4 = detector.get_tailscale_ipv4()
            if ts_ipv4:
                ips.add(ts_ipv4)
                logger.debug(f"[NetworkConfig] Tailscale IPv4: {ts_ipv4}")

            ts_ipv6 = detector.get_tailscale_ipv6()
            if ts_ipv6:
                ips.add(ts_ipv6)
                logger.debug(f"[NetworkConfig] Tailscale IPv6: {ts_ipv6}")
        except Exception as e:
            logger.debug(f"[NetworkConfig] Tailscale discovery failed: {e}")
        return ips

    def _discover_hostname_ips(self) -> set[str]:
        """Discover IPs via hostname resolution."""
        ips: set[str] = set()
        try:
            hostname = socket.gethostname()

            # IPv4
            for addr_info in socket.getaddrinfo(hostname, None, socket.AF_INET):
                ip = addr_info[4][0]
                if ip and ip != "127.0.0.1":
                    ips.add(ip)

            # IPv6
            for addr_info in socket.getaddrinfo(hostname, None, socket.AF_INET6):
                ip = addr_info[4][0]
                if ip and not ip.startswith("fe80:") and ip != "::1":
                    ips.add(ip)
        except (socket.gaierror, socket.herror, socket.timeout, OSError):
            # DNS lookup or socket errors
            pass
        return ips

    def _discover_interface_ips(self) -> set[str]:
        """Discover IPs from network interfaces."""
        ips: set[str] = set()
        try:
            import netifaces
            for iface in netifaces.interfaces():
                if iface.startswith("lo"):
                    continue

                addrs = netifaces.ifaddresses(iface)

                # IPv4
                for addr_info in addrs.get(netifaces.AF_INET, []):
                    ip = addr_info.get("addr")
                    if ip and ip != "127.0.0.1":
                        try:
                            ip_obj = ipaddress.ip_address(ip)
                            # Include Tailscale and public IPs
                            if (not ip_obj.is_private or
                                ip.startswith(self._config.tailscale_ipv4_prefix) or
                                self._config.include_private_ips):
                                ips.add(ip)
                        except ValueError:
                            pass

                # IPv6
                for addr_info in addrs.get(netifaces.AF_INET6, []):
                    ip = addr_info.get("addr", "")
                    if ip:
                        ip = ip.split("%")[0]  # Strip zone ID
                        if not ip.startswith("fe80:") and ip != "::1":
                            ips.add(ip)
        except ImportError:
            # netifaces not available, try socket approach
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
                if local_ip and local_ip != "127.0.0.1":
                    ips.add(local_ip)
            except (socket.error, OSError):
                # Socket connection error
                pass
        except (OSError, ValueError, KeyError, AttributeError):
            # Interface enumeration errors
            pass
        return ips

    def _discover_config_ips(self) -> set[str]:
        """Discover IPs from YAML config."""
        ips: set[str] = set()
        try:
            from app.config.cluster_config import load_cluster_config
            config = load_cluster_config()
            nodes = getattr(config, "hosts_raw", {}) or {}
            node_cfg = nodes.get(self._node_id, {})

            # Tailscale IP from config
            cfg_ts_ip = node_cfg.get("tailscale_ip")
            if cfg_ts_ip:
                ips.add(cfg_ts_ip)

            # Try to resolve ssh_host
            ssh_host = node_cfg.get("ssh_host")
            if ssh_host and not ssh_host.startswith("ssh"):
                try:
                    for addr_info in socket.getaddrinfo(ssh_host, None, socket.AF_INET):
                        ip = addr_info[4][0]
                        if ip and ip != "127.0.0.1":
                            ips.add(ip)
                    for addr_info in socket.getaddrinfo(ssh_host, None, socket.AF_INET6):
                        ip = addr_info[4][0]
                        if ip and not ip.startswith("fe80:") and ip != "::1":
                            ips.add(ip)
                except (socket.gaierror, socket.herror, socket.timeout, OSError):
                    # DNS lookup or socket errors
                    pass
        except ImportError:
            # cluster_config module not available
            pass
        except (OSError, IOError):
            # File I/O errors during config loading
            pass
        except (AttributeError, KeyError, TypeError):
            # Unexpected config structure
            pass
        return ips

    def _select_primary_host(self, all_ips: set[str]) -> tuple[str, set[str]]:
        """Select best primary address and return alternates.

        Preference order:
        1. Tailscale CGNAT IPv4 (100.x.x.x)
        2. Other IPv4 addresses
        3. Tailscale IPv6 (fd7a:115c:a1e0::)
        4. Other IPv6 addresses

        Args:
            all_ips: Set of all discovered IP addresses

        Returns:
            Tuple of (primary_host, alternate_ips)
        """
        if not all_ips:
            return "", set()

        ipv4_ips: set[str] = set()
        ipv6_ips: set[str] = set()

        for ip in all_ips:
            if ":" in ip:
                ipv6_ips.add(ip)
            else:
                ipv4_ips.add(ip)

        # Preference 1: Tailscale CGNAT IPv4
        tailscale_v4 = [
            ip for ip in ipv4_ips
            if ip.startswith(self._config.tailscale_ipv4_prefix)
        ]
        if tailscale_v4:
            primary = tailscale_v4[0]
            return primary, all_ips - {primary}

        # Preference 2: Any other IPv4
        if ipv4_ips:
            primary = next(iter(ipv4_ips))
            return primary, all_ips - {primary}

        # Preference 3: Tailscale IPv6
        tailscale_v6 = [
            ip for ip in ipv6_ips
            if ip.startswith(self._config.tailscale_ipv6_prefix)
        ]
        if tailscale_v6:
            primary = tailscale_v6[0]
            return primary, all_ips - {primary}

        # Preference 4: Any other IPv6
        if ipv6_ips:
            primary = next(iter(ipv6_ips))
            return primary, all_ips - {primary}

        return "", set()

    @staticmethod
    def format_ip_for_url(ip: str) -> str:
        """Format IP for URL (bracket IPv6 addresses).

        Args:
            ip: IP address string (IPv4 or IPv6)

        Returns:
            IP formatted for URL: IPv4 unchanged, IPv6 wrapped in brackets
        """
        if ":" in ip and not ip.startswith("["):
            return f"[{ip}]"
        return ip

    async def start_revalidation_loop(self) -> None:
        """Start periodic IP revalidation loop.

        Useful for detecting late Tailscale availability after startup.
        """
        if self._running:
            return

        self._running = True
        self._revalidation_task = safe_create_task(
            self._revalidation_loop(),
            name="network-config-revalidation",
        )
        logger.info("[NetworkConfig] Started revalidation loop")

    async def stop_revalidation_loop(self) -> None:
        """Stop periodic IP revalidation loop."""
        self._running = False
        if self._revalidation_task:
            self._revalidation_task.cancel()
            try:
                await self._revalidation_task
            except asyncio.CancelledError:
                pass
            self._revalidation_task = None
        logger.info("[NetworkConfig] Stopped revalidation loop")

    async def _revalidation_loop(self) -> None:
        """Periodic IP revalidation loop."""
        await asyncio.sleep(self._config.initial_revalidation_delay)

        while self._running:
            try:
                old_host = self.advertise_host
                self.validate_and_fix_advertise_host()

                if old_host != self.advertise_host:
                    logger.info(
                        f"[NetworkConfig] Revalidation: {old_host} -> {self.advertise_host}"
                    )
            except Exception as e:
                logger.debug(f"[NetworkConfig] Revalidation error: {e}")

            await asyncio.sleep(self._config.revalidation_interval)

    def get_network_state(self) -> dict:
        """Get current network state for status reporting.

        Returns:
            Dictionary with network state info
        """
        with self._lock:
            return {
                "advertise_host": self._state.advertise_host,
                "advertise_port": self._state.advertise_port,
                "alternate_ips": list(self._state.alternate_ips),
                "has_tailscale": self._state.has_tailscale,
                "last_validation_time": self._state.last_validation_time,
            }

    def health_check(self) -> dict:
        """Check network configuration health.

        Returns:
            Health check result dictionary
        """
        with self._lock:
            has_host = bool(self._state.advertise_host)
            has_tailscale = self._state.has_tailscale
            alternate_count = len(self._state.alternate_ips)

            if not has_host:
                status = "error"
                message = "No advertise host configured"
            elif not has_tailscale and alternate_count == 0:
                status = "degraded"
                message = "No Tailscale and no alternate IPs"
            elif not has_tailscale:
                status = "warning"
                message = f"No Tailscale, {alternate_count} alternate IPs available"
            else:
                status = "healthy"
                message = f"Tailscale available, {alternate_count} alternate IPs"

            return {
                "status": status,
                "message": message,
                "details": {
                    "advertise_host": self._state.advertise_host,
                    "has_tailscale": has_tailscale,
                    "alternate_ips_count": alternate_count,
                    "revalidation_active": self._running,
                },
            }


def get_network_config_manager() -> NetworkConfigManager | None:
    """Get singleton network config manager instance.

    Returns:
        NetworkConfigManager instance or None if not initialized
    """
    return _network_config_manager


def set_network_config_manager(manager: NetworkConfigManager | None) -> None:
    """Set singleton network config manager instance.

    Args:
        manager: NetworkConfigManager instance or None to clear
    """
    global _network_config_manager
    with _manager_lock:
        _network_config_manager = manager


def create_network_config_manager(
    node_id: str,
    initial_host: str = "",
    initial_port: int = 8770,
    config: NetworkConfig | None = None,
    on_host_changed: Callable[[str, str], None] | None = None,
) -> NetworkConfigManager:
    """Create and register singleton network config manager.

    Args:
        node_id: This node's identifier
        initial_host: Initial advertise host
        initial_port: Port for P2P communication
        config: Network configuration
        on_host_changed: Callback when advertise_host changes

    Returns:
        NetworkConfigManager instance
    """
    global _network_config_manager
    with _manager_lock:
        _network_config_manager = NetworkConfigManager(
            node_id=node_id,
            initial_host=initial_host,
            initial_port=initial_port,
            config=config,
            on_host_changed=on_host_changed,
        )
        return _network_config_manager
