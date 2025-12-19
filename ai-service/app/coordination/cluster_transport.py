"""Unified Cluster Transport Layer.

Provides multi-transport communication for cluster synchronization,
extracted from registry_sync_manager.py for reuse across sync systems.

Features:
- Multi-transport failover (Tailscale -> SSH -> HTTP)
- Circuit breaker pattern for fault tolerance
- Rsync wrapper for file-based transfers
- HTTP wrapper for API-based operations
- Async-first design with timeout handling

Usage:
    from app.coordination.cluster_transport import (
        ClusterTransport,
        CircuitBreaker,
        NodeConfig,
        TransportResult,
    )

    transport = ClusterTransport()

    # Transfer a file to a node
    result = await transport.transfer_file(
        local_path=Path("data/model.pth"),
        remote_path="ai-service/data/model.pth",
        node=NodeConfig(hostname="lambda-h100"),
    )

    # Execute HTTP request with failover
    result = await transport.http_request(
        node=NodeConfig(hostname="lambda-h100"),
        endpoint="/api/status",
    )

December 2025 - RingRift AI Service
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Default timeouts (increased for VAST.ai SSH proxy latency)
DEFAULT_CONNECT_TIMEOUT = 30  # Was 10, too short for VAST.ai
DEFAULT_OPERATION_TIMEOUT = 180  # Was 60, increased for large transfers
DEFAULT_HTTP_TIMEOUT = 30

# Circuit breaker defaults
DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_RECOVERY_TIMEOUT = 300  # 5 minutes


@dataclass
class NodeConfig:
    """Configuration for a cluster node."""
    hostname: str
    tailscale_ip: Optional[str] = None
    ssh_port: int = 22
    http_port: int = 8080
    http_scheme: str = "http"
    base_path: str = "ai-service"

    @property
    def http_base_url(self) -> str:
        """Get the HTTP base URL for this node."""
        host = self.tailscale_ip or self.hostname
        return f"{self.http_scheme}://{host}:{self.http_port}"

    @property
    def ssh_target(self) -> str:
        """Get the SSH target string."""
        return self.tailscale_ip or self.hostname


@dataclass
class TransportResult:
    """Result of a transport operation."""
    success: bool
    transport_used: Optional[str] = None
    data: Optional[Any] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    bytes_transferred: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "transport_used": self.transport_used,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "bytes_transferred": self.bytes_transferred,
        }


# Use canonical circuit breaker from distributed module
from app.distributed.circuit_breaker import CircuitBreaker, CircuitState


class ClusterTransport:
    """Unified transport layer for cluster communication.

    Provides multi-transport failover with automatic fallback:
    1. Tailscale (direct IP, fastest if available)
    2. SSH (hostname-based, reliable)
    3. HTTP (API endpoint, most flexible)
    """

    def __init__(
        self,
        p2p_url: Optional[str] = None,
        connect_timeout: int = DEFAULT_CONNECT_TIMEOUT,
        operation_timeout: int = DEFAULT_OPERATION_TIMEOUT,
    ):
        self.p2p_url = p2p_url or os.environ.get(
            "P2P_URL", "https://p2p.ringrift.ai"
        )
        self.connect_timeout = connect_timeout
        self.operation_timeout = operation_timeout
        # Canonical circuit breaker (tracks all targets internally)
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=DEFAULT_FAILURE_THRESHOLD,
            recovery_timeout=float(DEFAULT_RECOVERY_TIMEOUT),
            operation_type="cluster_transport",
        )

    def can_attempt(self, node_id: str) -> bool:
        """Check if circuit allows operation for a node."""
        return self._circuit_breaker.can_execute(node_id)

    def record_success(self, node_id: str) -> None:
        """Record successful operation for a node."""
        self._circuit_breaker.record_success(node_id)

    def record_failure(self, node_id: str) -> None:
        """Record failed operation for a node."""
        self._circuit_breaker.record_failure(node_id)

    async def transfer_file(
        self,
        local_path: Path,
        remote_path: str,
        node: NodeConfig,
        direction: str = "push",  # "push" or "pull"
    ) -> TransportResult:
        """Transfer a file to/from a node using transport failover.

        Args:
            local_path: Local file path
            remote_path: Remote file path (relative to node base_path)
            node: Target node configuration
            direction: "push" (local -> remote) or "pull" (remote -> local)

        Returns:
            TransportResult with success status and details
        """
        if not self.can_attempt(node.hostname):
            return TransportResult(
                success=False,
                error="Circuit breaker open",
            )

        start_time = time.time()
        full_remote_path = f"{node.base_path}/{remote_path}"

        # Try transports in order
        transports = [
            ("tailscale", self._transfer_via_tailscale),
            ("ssh", self._transfer_via_ssh),
        ]

        for transport_name, transport_fn in transports:
            try:
                result = await transport_fn(
                    local_path, full_remote_path, node, direction
                )
                if result.success:
                    self.record_success(node.hostname)
                    result.transport_used = transport_name
                    result.latency_ms = (time.time() - start_time) * 1000
                    return result
            except Exception as e:
                logger.debug(
                    f"Transport {transport_name} failed for {node.hostname}: {e}"
                )
                continue

        self.record_failure(node.hostname)
        return TransportResult(
            success=False,
            error="All transports failed",
            latency_ms=(time.time() - start_time) * 1000,
        )

    async def _transfer_via_tailscale(
        self,
        local_path: Path,
        remote_path: str,
        node: NodeConfig,
        direction: str,
    ) -> TransportResult:
        """Transfer file via Tailscale direct IP."""
        if not node.tailscale_ip:
            return TransportResult(success=False, error="No Tailscale IP")

        return await self._rsync_transfer(
            local_path,
            f"{node.tailscale_ip}:{remote_path}",
            direction,
            ssh_port=node.ssh_port,
        )

    async def _transfer_via_ssh(
        self,
        local_path: Path,
        remote_path: str,
        node: NodeConfig,
        direction: str,
    ) -> TransportResult:
        """Transfer file via SSH."""
        return await self._rsync_transfer(
            local_path,
            f"{node.hostname}:{remote_path}",
            direction,
            ssh_port=node.ssh_port,
        )

    async def _rsync_transfer(
        self,
        local_path: Path,
        remote_spec: str,
        direction: str,
        ssh_port: int = 22,
    ) -> TransportResult:
        """Execute rsync transfer."""
        if direction == "push":
            src, dst = str(local_path), remote_spec
        else:
            src, dst = remote_spec, str(local_path)

        cmd = [
            "rsync", "-az", f"--timeout={self.operation_timeout}",
            "-e", f"ssh -p {ssh_port} -o StrictHostKeyChecking=no "
                  f"-o ConnectTimeout={self.connect_timeout}",
            src, dst
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.operation_timeout + 10,
            )

            if proc.returncode == 0:
                bytes_transferred = local_path.stat().st_size if local_path.exists() else 0
                return TransportResult(
                    success=True,
                    bytes_transferred=bytes_transferred,
                )
            else:
                return TransportResult(
                    success=False,
                    error=stderr.decode()[:200] if stderr else "rsync failed",
                )

        except asyncio.TimeoutError:
            return TransportResult(success=False, error="Transfer timeout")
        except Exception as e:
            return TransportResult(success=False, error=str(e))

    async def http_request(
        self,
        node: NodeConfig,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> TransportResult:
        """Make an HTTP request to a node.

        Args:
            node: Target node configuration
            endpoint: API endpoint (e.g., "/api/status")
            method: HTTP method
            data: Form data for POST
            json_data: JSON body for POST
            timeout: Request timeout in seconds

        Returns:
            TransportResult with response data
        """
        http_target = f"{node.hostname}_http"
        if not self.can_attempt(http_target):
            return TransportResult(
                success=False,
                error="HTTP circuit breaker open",
            )

        try:
            import aiohttp
        except ImportError:
            return TransportResult(
                success=False,
                error="aiohttp not available",
            )

        timeout_val = timeout or DEFAULT_HTTP_TIMEOUT
        start_time = time.time()

        try:
            client_timeout = aiohttp.ClientTimeout(total=timeout_val)
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                url = f"{node.http_base_url}{endpoint}"

                kwargs: Dict[str, Any] = {}
                if data:
                    kwargs["data"] = data
                if json_data:
                    kwargs["json"] = json_data

                async with session.request(method, url, **kwargs) as resp:
                    response_data = None
                    try:
                        response_data = await resp.json()
                    except Exception:
                        response_data = await resp.text()

                    if resp.status >= 200 and resp.status < 300:
                        self.record_success(http_target)
                        return TransportResult(
                            success=True,
                            transport_used="http",
                            data=response_data,
                            latency_ms=(time.time() - start_time) * 1000,
                        )
                    else:
                        self.record_failure(http_target)
                        return TransportResult(
                            success=False,
                            error=f"HTTP {resp.status}",
                            data=response_data,
                            latency_ms=(time.time() - start_time) * 1000,
                        )

        except asyncio.TimeoutError:
            self.record_failure(http_target)
            return TransportResult(
                success=False,
                error="HTTP request timeout",
                latency_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            self.record_failure(http_target)
            return TransportResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def http_request_with_failover(
        self,
        node: NodeConfig,
        endpoint: str,
        method: str = "GET",
        **kwargs,
    ) -> TransportResult:
        """Make HTTP request with Tailscale/hostname failover.

        Tries Tailscale IP first, then falls back to hostname.
        """
        # Try Tailscale IP first if available
        if node.tailscale_ip:
            ts_node = NodeConfig(
                hostname=node.tailscale_ip,
                http_port=node.http_port,
                http_scheme=node.http_scheme,
            )
            result = await self.http_request(ts_node, endpoint, method, **kwargs)
            if result.success:
                result.transport_used = "http_tailscale"
                return result

        # Fall back to hostname
        result = await self.http_request(node, endpoint, method, **kwargs)
        if result.success:
            result.transport_used = "http_hostname"
        return result

    async def check_node_reachable(self, node: NodeConfig) -> bool:
        """Check if a node is reachable via any transport."""
        # Try quick HTTP health check first
        result = await self.http_request(
            node, "/health", timeout=5
        )
        if result.success:
            return True

        # Try SSH ping
        try:
            cmd = [
                "ssh", "-q", "-o", f"ConnectTimeout={self.connect_timeout}",
                "-o", "BatchMode=yes", "-p", str(node.ssh_port),
                node.ssh_target, "exit", "0"
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=self.connect_timeout + 5)
            return proc.returncode == 0
        except Exception:
            return False

    def reset_circuit_breakers(self) -> None:
        """Reset all circuit breakers."""
        self._circuit_breaker.reset_all()

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary of all circuit breakers."""
        all_states = self._circuit_breaker.get_all_states()
        return {
            target: {
                "state": status.state.value,
                "failures": status.failure_count,
                "can_attempt": self.can_attempt(target),
                "consecutive_opens": status.consecutive_opens,
            }
            for target, status in all_states.items()
        }


# Singleton instance for convenience
_transport_instance: Optional[ClusterTransport] = None


def get_cluster_transport() -> ClusterTransport:
    """Get the singleton ClusterTransport instance."""
    global _transport_instance
    if _transport_instance is None:
        _transport_instance = ClusterTransport()
    return _transport_instance
