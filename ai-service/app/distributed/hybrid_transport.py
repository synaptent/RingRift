"""Hybrid Transport Layer - Multi-Protocol Fallback.

Provides seamless communication with cluster nodes by automatically
falling back through multiple transport protocols. This makes the P2P
network self-healing for hard-to-reach nodes like Vast.ai instances.

Architecture (API requests):
    HTTP Request (direct)
        ↓ (fail)
    Tailscale IP (mesh network)
        ↓ (fail)
    Cloudflare Zero Trust (tunnel)
        ↓ (fail)
    SSH Transport (port forward)
        ↓
    Success or Final Failure

Architecture (large file transfers):
    aria2 (multi-source parallel download)
        ↓ (fail)
    rsync over SSH
        ↓
    Success or Final Failure

Usage:
    from app.distributed.hybrid_transport import HybridTransport

    transport = HybridTransport()

    # Automatically tries HTTP → Tailscale → Cloudflare → SSH
    result = await transport.send_heartbeat(node_id, self_info)

    # Configure cloudflare tunnel for a node
    transport.configure_cloudflare(node_id, "ringrift-node.tunnel.example.com")

    # Large file transfer via aria2
    success, path = await transport.download_file(
        node_id,
        urls=["http://node1/model.pth", "http://node2/model.pth"],
        local_path="/tmp/model.pth"
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Thresholds for transport switching
HTTP_FAILURES_BEFORE_SSH = 3  # Switch to SSH after 3 HTTP failures
SSH_SUCCESS_BEFORE_HTTP_RETRY = 5  # Retry HTTP after 5 SSH successes
TRANSPORT_HEALTH_CHECK_INTERVAL = 300  # Re-evaluate transport choice every 5 min


class TransportType(str, Enum):
    """Transport method for P2P communication."""
    HTTP = "http"
    TAILSCALE = "tailscale"
    CLOUDFLARE = "cloudflare"  # Cloudflare Zero Trust tunnel
    ARIA2 = "aria2"  # aria2 parallel download for large files
    SSH = "ssh"


@dataclass
class NodeTransportState:
    """Tracks transport health for a single node."""
    node_id: str
    preferred_transport: TransportType = TransportType.HTTP

    # HTTP tracking
    http_consecutive_failures: int = 0
    http_last_success: float = 0.0
    http_last_failure: float = 0.0

    # Tailscale tracking
    tailscale_consecutive_failures: int = 0
    tailscale_last_success: float = 0.0

    # Cloudflare Zero Trust tracking
    cloudflare_consecutive_failures: int = 0
    cloudflare_last_success: float = 0.0
    cloudflare_tunnel: Optional[str] = None

    # Aria2 tracking (for large file transfers)
    aria2_available: bool = False
    aria2_last_success: float = 0.0

    # SSH tracking
    ssh_consecutive_successes: int = 0
    ssh_last_success: float = 0.0
    ssh_available: bool = False

    # Last health check
    last_health_check: float = 0.0

    def record_http_success(self) -> None:
        """Record successful HTTP communication."""
        self.http_consecutive_failures = 0
        self.http_last_success = time.time()
        # HTTP success means we can prefer it again
        if self.preferred_transport != TransportType.HTTP:
            logger.info(f"[Transport] {self.node_id}: Switching back to HTTP")
            self.preferred_transport = TransportType.HTTP

    def record_http_failure(self) -> None:
        """Record HTTP failure and potentially switch transport."""
        self.http_consecutive_failures += 1
        self.http_last_failure = time.time()

        if self.http_consecutive_failures >= HTTP_FAILURES_BEFORE_SSH:
            if self.ssh_available and self.preferred_transport != TransportType.SSH:
                logger.info(
                    f"[Transport] {self.node_id}: Switching to SSH after "
                    f"{self.http_consecutive_failures} HTTP failures"
                )
                self.preferred_transport = TransportType.SSH

    def record_tailscale_success(self) -> None:
        """Record successful Tailscale communication."""
        self.tailscale_consecutive_failures = 0
        self.tailscale_last_success = time.time()

    def record_tailscale_failure(self) -> None:
        """Record Tailscale failure."""
        self.tailscale_consecutive_failures += 1

    def record_cloudflare_success(self) -> None:
        """Record successful Cloudflare Zero Trust tunnel communication."""
        self.cloudflare_consecutive_failures = 0
        self.cloudflare_last_success = time.time()

    def record_cloudflare_failure(self) -> None:
        """Record Cloudflare tunnel failure."""
        self.cloudflare_consecutive_failures += 1

    def record_aria2_success(self) -> None:
        """Record successful aria2 file transfer."""
        self.aria2_available = True
        self.aria2_last_success = time.time()

    def should_try_cloudflare(self) -> bool:
        """Check if Cloudflare tunnel is configured and working."""
        return (
            self.cloudflare_tunnel is not None
            and self.cloudflare_consecutive_failures < 3
        )

    def should_try_aria2(self) -> bool:
        """Check if aria2 is available for large file transfers."""
        return self.aria2_available

    def record_ssh_success(self) -> None:
        """Record successful SSH communication."""
        self.ssh_consecutive_successes += 1
        self.ssh_last_success = time.time()
        self.ssh_available = True

        # After enough SSH successes, try HTTP again
        if self.ssh_consecutive_successes >= SSH_SUCCESS_BEFORE_HTTP_RETRY:
            self.http_consecutive_failures = 0
            self.ssh_consecutive_successes = 0

    def record_ssh_failure(self) -> None:
        """Record SSH failure."""
        self.ssh_consecutive_successes = 0

    def should_try_http(self) -> bool:
        """Check if we should try HTTP first."""
        return (
            self.preferred_transport == TransportType.HTTP
            or self.http_consecutive_failures < HTTP_FAILURES_BEFORE_SSH
        )

    def should_try_ssh(self) -> bool:
        """Check if SSH is available and should be tried."""
        return self.ssh_available


class HybridTransport:
    """Hybrid transport with automatic HTTP/SSH switching.

    Provides reliable communication by:
    1. Trying HTTP first (fastest)
    2. Trying Tailscale IP as fallback
    3. Falling back to SSH for Vast instances

    Automatically tracks success/failure per node and adapts
    transport selection for optimal performance.
    """

    def __init__(self):
        self._states: Dict[str, NodeTransportState] = {}
        self._lock = asyncio.Lock()
        self._ssh_transport = None
        self._http_session = None

    def _get_state(self, node_id: str) -> NodeTransportState:
        """Get or create transport state for a node."""
        if node_id not in self._states:
            self._states[node_id] = NodeTransportState(node_id=node_id)
        return self._states[node_id]

    async def _get_ssh_transport(self):
        """Lazy-load SSH transport."""
        if self._ssh_transport is None:
            try:
                from app.distributed.ssh_transport import get_ssh_transport
                self._ssh_transport = get_ssh_transport()
            except ImportError:
                logger.warning("SSH transport not available")
        return self._ssh_transport

    async def _try_http(
        self,
        node_id: str,
        url: str,
        method: str = "POST",
        payload: Optional[Dict[str, Any]] = None,
        timeout: float = 15.0,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Try HTTP communication.

        Returns:
            Tuple of (success, response_dict or None)
        """
        try:
            from aiohttp import ClientSession, ClientTimeout

            timeout_obj = ClientTimeout(total=timeout, connect=5)
            async with ClientSession(timeout=timeout_obj) as session:
                kwargs = {}
                if payload:
                    kwargs["json"] = payload

                async with session.request(method, url, **kwargs) as resp:
                    if resp.status == 200:
                        return True, await resp.json()
                    else:
                        return False, {"status": resp.status, "error": await resp.text()}

        except Exception as e:
            logger.debug(f"HTTP request to {node_id} failed: {e}")
            return False, None

    async def _try_tailscale(
        self,
        node_id: str,
        port: int,
        path: str,
        method: str = "POST",
        payload: Optional[Dict[str, Any]] = None,
        timeout: float = 15.0,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Try communication via Tailscale IP.

        Returns:
            Tuple of (success, response_dict or None)
        """
        try:
            from app.distributed.dynamic_registry import get_registry
            registry = get_registry()

            node_info = registry._nodes.get(node_id)
            if not node_info or not node_info.tailscale_ip:
                return False, None

            ts_ip = node_info.tailscale_ip
            url = f"http://{ts_ip}:{port}{path}"

            return await self._try_http(node_id, url, method, payload, timeout)

        except ImportError:
            return False, None
        except Exception as e:
            logger.debug(f"Tailscale request to {node_id} failed: {e}")
            return False, None

    async def _try_cloudflare(
        self,
        node_id: str,
        path: str,
        method: str = "POST",
        payload: Optional[Dict[str, Any]] = None,
        timeout: float = 15.0,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Try communication via Cloudflare Zero Trust tunnel.

        Returns:
            Tuple of (success, response_dict or None)
        """
        state = self._get_state(node_id)
        if not state.cloudflare_tunnel:
            return False, None

        try:
            tunnel_url = f"https://{state.cloudflare_tunnel}{path}"
            return await self._try_http(node_id, tunnel_url, method, payload, timeout)
        except Exception as e:
            logger.debug(f"Cloudflare request to {node_id} failed: {e}")
            return False, None

    async def _try_aria2(
        self,
        node_id: str,
        urls: List[str],
        local_path: str,
        max_connections: int = 16,
    ) -> Tuple[bool, Optional[str]]:
        """Download file using aria2 with parallel connections.

        Best for large files like model checkpoints.

        Returns:
            Tuple of (success, local_path or None)
        """
        try:
            import tempfile

            # Create aria2 input file with all source URLs
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                for url in urls:
                    f.write(f"{url}\n")
                input_file = f.name

            cmd = [
                "aria2c",
                "--input-file", input_file,
                "--dir", str(Path(local_path).parent),
                "--out", Path(local_path).name,
                "--max-connection-per-server", str(max_connections),
                "--split", str(len(urls)),
                "--min-split-size", "1M",
                "--file-allocation", "none",
                "--auto-file-renaming", "false",
                "--allow-overwrite", "true",
                "--console-log-level", "warn",
                "--summary-interval", "0",
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)

            # Clean up input file
            try:
                Path(input_file).unlink()
            except Exception:
                pass

            if proc.returncode == 0:
                self._get_state(node_id).record_aria2_success()
                return True, local_path
            else:
                logger.debug(f"aria2 download failed: {stderr.decode()}")
                return False, None

        except asyncio.TimeoutError:
            logger.warning(f"aria2 download timed out for {node_id}")
            return False, None
        except FileNotFoundError:
            logger.debug("aria2c not installed")
            return False, None
        except Exception as e:
            logger.debug(f"aria2 download failed: {e}")
            return False, None

    async def _try_ssh(
        self,
        node_id: str,
        command_type: str,
        payload: Dict[str, Any],
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Try SSH communication.

        Returns:
            Tuple of (success, response_dict or None)
        """
        ssh = await self._get_ssh_transport()
        if not ssh:
            return False, None

        return await ssh.send_command(node_id, command_type, payload)

    async def send_request(
        self,
        node_id: str,
        host: str,
        port: int,
        path: str,
        method: str = "POST",
        payload: Optional[Dict[str, Any]] = None,
        command_type: Optional[str] = None,
        timeout: float = 15.0,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Send a request using the best available transport.

        Tries transports in order of preference, tracking success/failure
        to optimize future requests.

        Args:
            node_id: Target node identifier
            host: HTTP host
            port: HTTP port
            path: API path
            method: HTTP method
            payload: Request payload
            command_type: Command type for SSH fallback
            timeout: Request timeout

        Returns:
            Tuple of (success, response_dict or None)
        """
        state = self._get_state(node_id)

        # Try HTTP first if appropriate
        if state.should_try_http():
            url = f"http://{host}:{port}{path}"
            success, response = await self._try_http(
                node_id, url, method, payload, timeout
            )

            if success:
                state.record_http_success()
                return True, response
            else:
                state.record_http_failure()

        # Try Tailscale IP
        success, response = await self._try_tailscale(
            node_id, port, path, method, payload, timeout
        )
        if success:
            state.record_tailscale_success()
            state.record_http_success()  # Tailscale worked, HTTP layer is fine
            return True, response
        else:
            state.record_tailscale_failure()

        # Try Cloudflare Zero Trust tunnel
        if state.should_try_cloudflare():
            success, response = await self._try_cloudflare(
                node_id, path, method, payload, timeout
            )
            if success:
                state.record_cloudflare_success()
                return True, response
            else:
                state.record_cloudflare_failure()

        # Try SSH fallback for Vast nodes
        if node_id.startswith("vast-") and command_type:
            # Mark SSH as potentially available for Vast nodes
            state.ssh_available = True

            success, response = await self._try_ssh(
                node_id, command_type, payload or {}
            )

            if success:
                state.record_ssh_success()
                return True, response
            else:
                state.record_ssh_failure()

        return False, None

    async def send_heartbeat(
        self,
        node_id: str,
        host: str,
        port: int,
        self_info: Dict[str, Any],
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Send heartbeat to a node.

        Args:
            node_id: Target node
            host: Node's HTTP host
            port: Node's HTTP port
            self_info: Our node info

        Returns:
            Tuple of (success, response_dict or None)
        """
        return await self.send_request(
            node_id=node_id,
            host=host,
            port=port,
            path="/heartbeat",
            method="POST",
            payload=self_info,
            command_type="heartbeat",
        )

    async def send_relay_heartbeat(
        self,
        node_id: str,
        host: str,
        port: int,
        self_info: Dict[str, Any],
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Send relay heartbeat to a node.

        Args:
            node_id: Target node (relay hub)
            host: Node's HTTP host
            port: Node's HTTP port
            self_info: Our node info

        Returns:
            Tuple of (success, response_dict or None)
        """
        return await self.send_request(
            node_id=node_id,
            host=host,
            port=port,
            path="/relay/heartbeat",
            method="POST",
            payload=self_info,
            command_type="relay_heartbeat",
        )

    async def request_job_start(
        self,
        node_id: str,
        host: str,
        port: int,
        job_payload: Dict[str, Any],
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Request a node to start a job.

        Args:
            node_id: Target node
            host: Node's HTTP host
            port: Node's HTTP port
            job_payload: Job configuration

        Returns:
            Tuple of (success, response_dict or None)
        """
        return await self.send_request(
            node_id=node_id,
            host=host,
            port=port,
            path="/jobs/start",
            method="POST",
            payload=job_payload,
            command_type="start_job",
        )

    async def get_node_status(
        self,
        node_id: str,
        host: str,
        port: int,
    ) -> Optional[Dict[str, Any]]:
        """Get status from a node.

        Args:
            node_id: Target node
            host: Node's HTTP host
            port: Node's HTTP port

        Returns:
            Status dict if successful, None otherwise
        """
        success, response = await self.send_request(
            node_id=node_id,
            host=host,
            port=port,
            path="/status",
            method="GET",
            command_type="status",
        )
        return response if success else None

    def get_transport_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get transport statistics for all nodes.

        Returns:
            Dict mapping node_id to transport stats
        """
        stats = {}
        for node_id, state in self._states.items():
            stats[node_id] = {
                "preferred_transport": state.preferred_transport.value,
                "http_failures": state.http_consecutive_failures,
                "http_last_success": state.http_last_success,
                "tailscale_failures": state.tailscale_consecutive_failures,
                "cloudflare_available": state.cloudflare_tunnel is not None,
                "cloudflare_failures": state.cloudflare_consecutive_failures,
                "aria2_available": state.aria2_available,
                "ssh_available": state.ssh_available,
                "ssh_successes": state.ssh_consecutive_successes,
                "ssh_last_success": state.ssh_last_success,
            }
        return stats

    def configure_cloudflare(self, node_id: str, tunnel_hostname: str) -> None:
        """Configure Cloudflare Zero Trust tunnel for a node.

        Args:
            node_id: Target node identifier
            tunnel_hostname: Cloudflare tunnel hostname (e.g., 'node.tunnel.example.com')
        """
        state = self._get_state(node_id)
        state.cloudflare_tunnel = tunnel_hostname
        state.cloudflare_consecutive_failures = 0
        logger.info(f"[Transport] {node_id}: Configured Cloudflare tunnel {tunnel_hostname}")

    def configure_aria2(self, node_id: str, available: bool = True) -> None:
        """Configure aria2 availability for a node.

        Args:
            node_id: Target node identifier
            available: Whether aria2 is available for this node
        """
        state = self._get_state(node_id)
        state.aria2_available = available
        logger.info(f"[Transport] {node_id}: aria2 {'enabled' if available else 'disabled'}")

    async def download_file(
        self,
        node_id: str,
        urls: List[str],
        local_path: str,
    ) -> Tuple[bool, Optional[str]]:
        """Download a file using aria2 with multi-source parallel download.

        Best for large files like model checkpoints where multiple nodes
        may have copies.

        Args:
            node_id: Primary node identifier (for tracking)
            urls: List of URLs to download from (aria2 will use all)
            local_path: Local path to save the file

        Returns:
            Tuple of (success, local_path or None)
        """
        state = self._get_state(node_id)

        # Try aria2 first if available
        if state.should_try_aria2() or len(urls) > 1:
            success, path = await self._try_aria2(node_id, urls, local_path)
            if success:
                return True, path

        # Fallback to simple HTTP download from first URL
        if urls:
            try:
                from aiohttp import ClientSession, ClientTimeout
                timeout = ClientTimeout(total=300, connect=10)
                async with ClientSession(timeout=timeout) as session:
                    async with session.get(urls[0]) as resp:
                        if resp.status == 200:
                            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
                            with open(local_path, "wb") as f:
                                async for chunk in resp.content.iter_chunked(1024 * 1024):
                                    f.write(chunk)
                            return True, local_path
            except Exception as e:
                logger.debug(f"HTTP download failed: {e}")

        return False, None

    async def probe_all_transports(
        self,
        node_id: str,
        host: str,
        port: int,
    ) -> Dict[str, Tuple[bool, float]]:
        """Probe all transports for a node and return latencies.

        Args:
            node_id: Node to probe
            host: Node's HTTP host
            port: Node's HTTP port

        Returns:
            Dict mapping transport type to (reachable, latency_ms)
        """
        results = {}

        # Probe HTTP
        start = time.time()
        success, _ = await self._try_http(
            node_id, f"http://{host}:{port}/health", "GET", None, 10
        )
        results["http"] = (success, (time.time() - start) * 1000)

        # Probe Tailscale
        start = time.time()
        success, _ = await self._try_tailscale(node_id, port, "/health", "GET", None, 10)
        results["tailscale"] = (success, (time.time() - start) * 1000)

        # Probe Cloudflare
        state = self._get_state(node_id)
        if state.cloudflare_tunnel:
            start = time.time()
            success, _ = await self._try_cloudflare(node_id, "/health", "GET", None, 10)
            results["cloudflare"] = (success, (time.time() - start) * 1000)

        # Probe SSH
        ssh = await self._get_ssh_transport()
        if ssh and node_id.startswith("vast-"):
            start = time.time()
            reachable, _ = await ssh.check_connectivity(node_id)
            results["ssh"] = (reachable, (time.time() - start) * 1000)

        return results


# Global instance
_hybrid_transport: Optional[HybridTransport] = None


def get_hybrid_transport() -> HybridTransport:
    """Get or create global hybrid transport instance."""
    global _hybrid_transport
    if _hybrid_transport is None:
        _hybrid_transport = HybridTransport()
    return _hybrid_transport


async def diagnose_node_connectivity(
    node_id: str,
    host: str,
    port: int,
) -> Dict[str, Any]:
    """Run comprehensive connectivity diagnostics for a node.

    Args:
        node_id: Node to diagnose
        host: Node's HTTP host
        port: Node's HTTP port

    Returns:
        Diagnostic report dict
    """
    transport = get_hybrid_transport()
    probes = await transport.probe_all_transports(node_id, host, port)

    # Determine best transport
    best_transport = None
    best_latency = float("inf")

    for transport_type, (reachable, latency) in probes.items():
        if reachable and latency < best_latency:
            best_transport = transport_type
            best_latency = latency

    return {
        "node_id": node_id,
        "host": host,
        "port": port,
        "probes": {
            t: {"reachable": r, "latency_ms": round(l, 1)}
            for t, (r, l) in probes.items()
        },
        "best_transport": best_transport,
        "best_latency_ms": round(best_latency, 1) if best_transport else None,
        "recommendation": (
            "SSH recommended" if best_transport == "ssh" else
            "Cloudflare tunnel recommended" if best_transport == "cloudflare" else
            "Tailscale recommended" if best_transport == "tailscale" else
            "HTTP working normally" if best_transport == "http" else
            "Node unreachable via all transports"
        ),
    }
