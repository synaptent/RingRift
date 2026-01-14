"""
HTTP-based transport implementations.

Tier 1: DirectHTTPTransport - Direct HTTP to public IP
Tier 2: TailscaleHTTPTransport - HTTP over Tailscale mesh
Tier 3: CloudflareHTTPTransport - HTTPS via Cloudflare tunnel

Jan 13, 2026: Updated to use session_factory.py for proper connection lifecycle
management. This fixes CLOSE_WAIT connection leaks caused by bare ClientSession
instances without TCPConnector configuration.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

try:
    import aiohttp
    from aiohttp import ClientTimeout
except ImportError:
    aiohttp = None  # type: ignore
    ClientTimeout = None  # type: ignore

from ..transport_cascade import BaseTransport, TransportResult, TransportTier
from .session_factory import ReusableSessionManager, close_session

logger = logging.getLogger(__name__)

# Default P2P port
DEFAULT_P2P_PORT = int(os.environ.get("RINGRIFT_P2P_PORT", "8770"))


class DirectHTTPTransport(BaseTransport):
    """
    Direct HTTP transport to public IP addresses.

    Tier 1 (FAST): Lowest latency when public connectivity available.

    Jan 13, 2026: Now uses ReusableSessionManager for proper connection
    lifecycle management (keepalive, cleanup, connection pooling).
    """

    name = "http_direct"
    tier = TransportTier.TIER_1_FAST

    def __init__(self, port: int = DEFAULT_P2P_PORT, timeout: float = 5.0):
        self._port = port
        self._timeout = timeout
        # Use centralized session manager for proper lifecycle management
        self._session_manager = ReusableSessionManager(
            timeout=timeout,
            connect_timeout=min(timeout, 5.0),
            keepalive_timeout=30,
            limit=50,
            enable_keepalive=True,
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with proper lifecycle management."""
        return await self._session_manager.get_session()

    async def send(self, target: str, payload: bytes) -> TransportResult:
        """Send payload via direct HTTP POST."""
        if aiohttp is None:
            return self._make_result(
                success=False, latency_ms=0, error="aiohttp not installed"
            )

        # Parse target - could be "node_id" or "host:port" or "http://..."
        url = self._target_to_url(target)
        if not url:
            return self._make_result(
                success=False, latency_ms=0, error=f"Cannot resolve target: {target}"
            )

        start_time = time.time()
        try:
            session = await self._get_session()
            async with session.post(
                url,
                data=payload,
                headers={"Content-Type": "application/octet-stream"},
            ) as resp:
                latency_ms = (time.time() - start_time) * 1000
                response_data = await resp.read()

                if resp.status == 200:
                    return self._make_result(
                        success=True,
                        latency_ms=latency_ms,
                        response=response_data,
                        status_code=resp.status,
                    )
                else:
                    return self._make_result(
                        success=False,
                        latency_ms=latency_ms,
                        error=f"HTTP {resp.status}",
                        status_code=resp.status,
                    )

        except asyncio.TimeoutError:
            return self._make_result(
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=f"timeout ({self._timeout}s)",
            )
        except aiohttp.ClientError as e:
            return self._make_result(
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=f"ClientError: {e}",
            )
        except Exception as e:
            return self._make_result(
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=f"{type(e).__name__}: {e}",
            )

    async def is_available(self, target: str) -> bool:
        """Check if direct HTTP is available to target."""
        url = self._target_to_url(target)
        if not url:
            return False

        # Try health endpoint
        health_url = url.replace("/cascade", "/health")
        try:
            session = await self._get_session()
            async with session.get(health_url, timeout=ClientTimeout(total=2.0)) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _target_to_url(self, target: str) -> str | None:
        """Convert target to HTTP URL."""
        if target.startswith("http://") or target.startswith("https://"):
            return target

        # Assume it's a hostname or IP
        if ":" in target:
            host, port = target.rsplit(":", 1)
        else:
            host = target
            port = str(self._port)

        return f"http://{host}:{port}/cascade"

    async def close(self) -> None:
        """Close the HTTP session."""
        await self._session_manager.close()


class TailscaleHTTPTransport(BaseTransport):
    """
    HTTP transport over Tailscale mesh network.

    Tier 2 (RELIABLE): Works through NAT via Tailscale overlay.

    Jan 13, 2026: Now uses ReusableSessionManager for proper connection
    lifecycle management (keepalive, cleanup, connection pooling).
    """

    name = "tailscale"
    tier = TransportTier.TIER_2_RELIABLE

    def __init__(self, port: int = DEFAULT_P2P_PORT, timeout: float = 10.0):
        self._port = port
        self._timeout = timeout
        # Use centralized session manager for proper lifecycle management
        self._session_manager = ReusableSessionManager(
            timeout=timeout,
            connect_timeout=min(timeout, 5.0),
            keepalive_timeout=30,
            limit=50,
            enable_keepalive=True,
        )
        # Cache of node_id -> tailscale_ip
        self._tailscale_ips: dict[str, str] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with proper lifecycle management."""
        return await self._session_manager.get_session()

    async def send(self, target: str, payload: bytes) -> TransportResult:
        """Send payload via Tailscale HTTP."""
        if aiohttp is None:
            return self._make_result(
                success=False, latency_ms=0, error="aiohttp not installed"
            )

        tailscale_ip = await self._get_tailscale_ip(target)
        if not tailscale_ip:
            return self._make_result(
                success=False,
                latency_ms=0,
                error=f"No Tailscale IP for target: {target}",
            )

        url = f"http://{tailscale_ip}:{self._port}/cascade"
        start_time = time.time()

        try:
            session = await self._get_session()
            async with session.post(
                url,
                data=payload,
                headers={"Content-Type": "application/octet-stream"},
            ) as resp:
                latency_ms = (time.time() - start_time) * 1000
                response_data = await resp.read()

                if resp.status == 200:
                    return self._make_result(
                        success=True,
                        latency_ms=latency_ms,
                        response=response_data,
                        tailscale_ip=tailscale_ip,
                    )
                else:
                    return self._make_result(
                        success=False,
                        latency_ms=latency_ms,
                        error=f"HTTP {resp.status}",
                    )

        except Exception as e:
            return self._make_result(
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=f"{type(e).__name__}: {e}",
            )

    async def is_available(self, target: str) -> bool:
        """Check if Tailscale transport is available."""
        tailscale_ip = await self._get_tailscale_ip(target)
        return tailscale_ip is not None

    async def _get_tailscale_ip(self, target: str, prefer_ipv6: bool = True) -> str | None:
        """Get Tailscale IP for a target node.

        Jan 2026: Added IPv6 support. Prefers IPv6 (fd7a:...) as it bypasses NAT.

        Args:
            target: Target node ID or IP
            prefer_ipv6: If True, prefer IPv6 over IPv4. Default True.

        Returns:
            Tailscale IP or None
        """
        # Check cache first
        if target in self._tailscale_ips:
            return self._tailscale_ips[target]

        # If target is already a Tailscale IPv4 (100.x.x.x) or IPv6 (fd7a:...)
        if target.startswith("100."):
            return target.split(":")[0]  # Remove port if present
        if target.startswith("fd7a:"):
            # IPv6 - extract address without port (brackets notation)
            if target.startswith("["):
                return target.split("]")[0][1:]
            return target.split("]:")[0] if "]:" in target else target

        # Try to get from tailscale status
        try:
            proc = await asyncio.create_subprocess_exec(
                "tailscale", "status", "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)

            import json
            data = json.loads(stdout.decode())

            for peer_key, peer_info in data.get("Peer", {}).items():
                hostname = peer_info.get("HostName", "")
                if hostname == target or hostname.startswith(target):
                    ips = peer_info.get("TailscaleIPs", [])
                    if ips:
                        # Prefer IPv6 if available and requested
                        ipv4 = None
                        ipv6 = None
                        for ip in ips:
                            if ":" in ip:
                                ipv6 = ip
                            else:
                                ipv4 = ip

                        if prefer_ipv6 and ipv6:
                            self._tailscale_ips[target] = ipv6
                            return ipv6
                        elif ipv4:
                            self._tailscale_ips[target] = ipv4
                            return ipv4
                        elif ips:
                            self._tailscale_ips[target] = ips[0]
                            return ips[0]

        except Exception as e:
            logger.debug(f"Failed to get Tailscale IP for {target}: {e}")

        return None

    async def close(self) -> None:
        """Close the HTTP session."""
        await self._session_manager.close()


class CloudflareHTTPTransport(BaseTransport):
    """
    HTTPS transport via Cloudflare Zero Trust tunnel.

    Tier 3 (TUNNELED): Works through any firewall via Cloudflare.

    Jan 13, 2026: Now uses ReusableSessionManager for proper connection
    lifecycle management (keepalive, cleanup, connection pooling).
    """

    name = "cloudflare_tunnel"
    tier = TransportTier.TIER_3_TUNNELED

    def __init__(self, timeout: float = 15.0):
        self._timeout = timeout
        # Use centralized session manager for proper lifecycle management
        self._session_manager = ReusableSessionManager(
            timeout=timeout,
            connect_timeout=min(timeout, 10.0),
            keepalive_timeout=30,
            limit=50,
            enable_keepalive=True,
        )
        # node_id -> tunnel hostname
        self._tunnel_hostnames: dict[str, str] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with proper lifecycle management."""
        return await self._session_manager.get_session()

    def configure_tunnel(self, node_id: str, tunnel_hostname: str) -> None:
        """Configure Cloudflare tunnel for a node."""
        self._tunnel_hostnames[node_id] = tunnel_hostname

    async def send(self, target: str, payload: bytes) -> TransportResult:
        """Send payload via Cloudflare tunnel."""
        if aiohttp is None:
            return self._make_result(
                success=False, latency_ms=0, error="aiohttp not installed"
            )

        tunnel_hostname = self._tunnel_hostnames.get(target)
        if not tunnel_hostname:
            return self._make_result(
                success=False,
                latency_ms=0,
                error=f"No Cloudflare tunnel configured for: {target}",
            )

        url = f"https://{tunnel_hostname}/cascade"
        start_time = time.time()

        try:
            session = await self._get_session()
            async with session.post(
                url,
                data=payload,
                headers={"Content-Type": "application/octet-stream"},
            ) as resp:
                latency_ms = (time.time() - start_time) * 1000
                response_data = await resp.read()

                if resp.status == 200:
                    return self._make_result(
                        success=True,
                        latency_ms=latency_ms,
                        response=response_data,
                        tunnel_hostname=tunnel_hostname,
                    )
                else:
                    return self._make_result(
                        success=False,
                        latency_ms=latency_ms,
                        error=f"HTTP {resp.status}",
                    )

        except Exception as e:
            return self._make_result(
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=f"{type(e).__name__}: {e}",
            )

    async def is_available(self, target: str) -> bool:
        """Check if Cloudflare tunnel is configured for target."""
        return target in self._tunnel_hostnames

    async def close(self) -> None:
        """Close the HTTP session."""
        await self._session_manager.close()
