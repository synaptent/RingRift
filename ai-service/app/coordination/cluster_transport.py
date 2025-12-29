"""Unified Cluster Transport Layer.

Provides multi-transport communication for cluster synchronization,
extracted from registry_sync_manager.py for reuse across sync systems.

Features:
- Multi-transport failover (Tailscale -> SSH/rsync -> Base64 -> HTTP)
- Circuit breaker pattern for fault tolerance
- Rsync wrapper for file-based transfers
- Base64 transfer for connections with binary stream issues
- HTTP wrapper for API-based operations
- Async-first design with timeout handling

Transport Failover Order:
1. Tailscale (direct IP, fastest if available)
2. SSH/rsync (hostname-based, reliable, supports resume)
3. Base64 (text-safe, works when binary streams fail - see note below)
4. HTTP (API endpoint, most flexible)

**Base64 Transport (December 2025):**
When SCP/rsync connections reset with "Connection reset by peer" errors,
the base64 transport provides a reliable fallback. It encodes files as
text-safe base64 and pipes through SSH stdin, avoiding binary stream
corruption issues caused by some firewalls, proxies, or network configs.

Keywords for searchability:
- connection reset workaround
- binary stream corruption fix
- SSH pipe transfer
- text-safe file transfer

Usage:
    from app.coordination.cluster_transport import (
        ClusterTransport,
        CircuitBreaker,
        NodeConfig,
        TransportResult,
    )

    transport = ClusterTransport()

    # Transfer a file to a node (automatic failover to base64 if needed)
    result = await transport.transfer_file(
        local_path=Path("data/model.pth"),
        remote_path="ai-service/data/model.pth",
        node=NodeConfig(hostname="node-001"),
    )

    # Execute HTTP request with failover
    result = await transport.http_request(
        node=NodeConfig(hostname="node-001"),
        endpoint="/api/status",
    )

See also:
- scripts/lib/transfer.py:base64_push - sync version for scripts
- scripts/lib/transfer.py:robust_push - auto-failover wrapper

December 2025 - RingRift AI Service
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Import centralized timeout constants (December 2025)
try:
    from app.config.thresholds import (
        CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
        CLUSTER_CONNECT_TIMEOUT,
        CLUSTER_OPERATION_TIMEOUT,
        HTTP_TIMEOUT,
    )
    DEFAULT_CONNECT_TIMEOUT = CLUSTER_CONNECT_TIMEOUT
    DEFAULT_OPERATION_TIMEOUT = CLUSTER_OPERATION_TIMEOUT
    DEFAULT_HTTP_TIMEOUT = HTTP_TIMEOUT
    DEFAULT_FAILURE_THRESHOLD = CIRCUIT_BREAKER_FAILURE_THRESHOLD
    DEFAULT_RECOVERY_TIMEOUT = CIRCUIT_BREAKER_RECOVERY_TIMEOUT
except ImportError:
    # Fallback for testing/standalone use
    DEFAULT_CONNECT_TIMEOUT = 30  # Was 10, too short for VAST.ai
    DEFAULT_OPERATION_TIMEOUT = 180  # Was 60, increased for large transfers
    DEFAULT_HTTP_TIMEOUT = 30
    DEFAULT_FAILURE_THRESHOLD = 3
    DEFAULT_RECOVERY_TIMEOUT = 300  # 5 minutes

# Import bandwidth limiting from cluster_config (December 2025)
try:
    from app.config.cluster_config import get_node_bandwidth_kbs
    HAS_BANDWIDTH_CONFIG = True
except ImportError:
    HAS_BANDWIDTH_CONFIG = False
    get_node_bandwidth_kbs = None

# Import centralized port configuration (December 2025)
try:
    from app.config.ports import P2P_DEFAULT_PORT
except ImportError:
    P2P_DEFAULT_PORT = 8770  # Fallback for testing/standalone use


# =============================================================================
# Error Classes - Import from canonical location (December 28, 2025)
# =============================================================================

# Canonical definitions now in transport_base.py
from app.coordination.transport_base import (
    TransportError as _CanonicalTransportError,
    RetryableTransportError,
    PermanentTransportError,
    TransportResult as _CanonicalTransportResult,
)

# Backward-compatible wrapper for TransportError (different __init__ signature)
# The canonical version is a dataclass; this preserves the old interface
class TransportError(_CanonicalTransportError):
    """Base transport error for cluster operations.

    December 28, 2025: Wrapper for backward compatibility.
    Canonical location is app.coordination.transport_base.TransportError.
    """

    def __init__(self, message: str, target: str | None = None, transport: str | None = None):
        # Initialize the dataclass fields directly
        object.__setattr__(self, 'message', message)
        object.__setattr__(self, 'transport', transport or "")
        object.__setattr__(self, 'target', target or "")
        object.__setattr__(self, 'cause', None)
        Exception.__init__(self, message)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TimeoutConfig:
    """Configuration for cluster transport timeouts and circuit breaker.

    December 29, 2025: Renamed from TransportConfig to TimeoutConfig for clarity.
    This config focuses on connection timeouts and circuit breaker settings.
    For file transfer settings, use transport_manager.TransportConfig.
    For sync protocol settings, use storage_provider.SyncProtocolConfig.
    """

    connect_timeout: int = DEFAULT_CONNECT_TIMEOUT
    operation_timeout: int = DEFAULT_OPERATION_TIMEOUT
    http_timeout: int = DEFAULT_HTTP_TIMEOUT
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    recovery_timeout: float = float(DEFAULT_RECOVERY_TIMEOUT)
    retry_attempts: int = 3
    retry_backoff: float = 1.5

    @classmethod
    def for_large_transfers(cls) -> "TimeoutConfig":
        """Config optimized for large file transfers (>100MB)."""
        return cls(
            connect_timeout=60,
            operation_timeout=600,
            retry_attempts=2,
        )

    @classmethod
    def for_quick_requests(cls) -> "TimeoutConfig":
        """Config optimized for quick API requests."""
        return cls(
            connect_timeout=10,
            operation_timeout=30,
            http_timeout=15,
            retry_attempts=2,
        )


# Backward-compatible alias (deprecated, will be removed Q2 2026)
TransportConfig = TimeoutConfig


@dataclass
class NodeConfig:
    """Configuration for a cluster node."""
    hostname: str
    tailscale_ip: str | None = None
    ssh_port: int = 22
    http_port: int = P2P_DEFAULT_PORT  # P2P port for HTTP file sync
    http_scheme: str = "http"
    base_path: str = "ai-service"
    p2p_port: int = P2P_DEFAULT_PORT  # Explicit P2P port for file download endpoints

    @property
    def http_base_url(self) -> str:
        """Get the HTTP base URL for this node."""
        host = self.tailscale_ip or self.hostname
        return f"{self.http_scheme}://{host}:{self.http_port}"

    @property
    def ssh_target(self) -> str:
        """Get the SSH target string."""
        return self.tailscale_ip or self.hostname


# TransportResult is now imported from transport_base.py (December 28, 2025)
# Re-export the canonical version for backward compatibility
TransportResult = _CanonicalTransportResult


# Use canonical circuit breaker from distributed module
from app.distributed.circuit_breaker import CircuitBreaker

# Import HealthCheckResult for DaemonManager integration
try:
    from app.coordination.protocols import HealthCheckResult
except ImportError:
    # Fallback if protocols not available
    HealthCheckResult = None  # type: ignore


class ClusterTransport:
    """Unified transport layer for cluster communication.

    Provides multi-transport failover with automatic fallback:
    1. Tailscale (direct IP, fastest if available)
    2. SSH (hostname-based, reliable)
    3. HTTP (API endpoint, most flexible)
    """

    def __init__(
        self,
        p2p_url: str | None = None,
        connect_timeout: int = DEFAULT_CONNECT_TIMEOUT,
        operation_timeout: int = DEFAULT_OPERATION_TIMEOUT,
    ):
        # Dec 2025: Use centralized P2P URL helper
        if p2p_url:
            self.p2p_url = p2p_url
        else:
            from app.config.ports import get_local_p2p_url
            self.p2p_url = get_local_p2p_url()
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

        # Try transports in order of preference:
        # 1. Tailscale (direct IP, fastest if available)
        # 2. SSH/rsync (hostname-based, reliable, supports resume)
        # 3. Base64 (text-safe, works when binary streams fail)
        # 4. HTTP (P2P file download endpoints, works when all SSH fails)
        transports = [
            ("tailscale", self._transfer_via_tailscale),
            ("ssh", self._transfer_via_ssh),
            ("base64", self._transfer_via_base64),
            ("http", self._transfer_via_http),
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
            except (OSError, ValueError, TypeError, RuntimeError) as e:
                # OSError: network/file errors
                # ValueError: invalid path or parameters
                # TypeError: invalid argument types
                # RuntimeError: transfer operation failed
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

    async def _transfer_via_base64(
        self,
        local_path: Path,
        remote_path: str,
        node: NodeConfig,
        direction: str,
    ) -> TransportResult:
        """Transfer file via base64 encoding through SSH stdin/stdout.

        This method avoids binary stream handling issues that cause
        "Connection reset by peer" errors during rsync/scp transfers.

        **When this is useful:**
        - SSH connections reset during binary transfers
        - Firewall or proxy is corrupting binary streams
        - rsync/scp fail but simple SSH commands work

        **How it works:**
        - Push: cat file | base64 | ssh 'base64 -d > file'
        - Pull: ssh 'base64 file' | base64 -d > file

        Keywords for searchability:
        - base64 transfer / base64 push / base64 pull
        - connection reset workaround
        - binary stream corruption fix
        - SSH pipe transfer
        - text-safe file transfer

        See also:
        - scripts/lib/transfer.py:base64_push - sync version
        """
        import base64

        start_time = time.time()

        if direction == "push":
            return await self._base64_push(local_path, remote_path, node)
        else:
            return await self._base64_pull(local_path, remote_path, node)

    async def _base64_push(
        self,
        local_path: Path,
        remote_path: str,
        node: NodeConfig,
    ) -> TransportResult:
        """Push file to remote using base64 encoding."""
        import base64

        if not local_path.exists():
            return TransportResult(success=False, error=f"File not found: {local_path}")

        file_size = local_path.stat().st_size

        # Warn for large files (>100MB)
        if file_size > 100 * 1024 * 1024:
            logger.warning(
                f"base64_push: Large file ({file_size / 1024 / 1024:.1f}MB) - "
                "consider chunked transfer for better memory efficiency"
            )

        try:
            # Read and encode file (non-blocking to avoid event loop stall)
            # Dec 29, 2025: Use asyncio.to_thread for large file reads
            def _read_and_encode():
                with open(local_path, "rb") as f:
                    return base64.b64encode(f.read()).decode("ascii")

            encoded_data = await asyncio.to_thread(_read_and_encode)

            # Ensure remote directory exists and decode file
            remote_dir = str(Path(remote_path).parent)
            decode_cmd = f"mkdir -p '{remote_dir}' && base64 -d > '{remote_path}'"

            cmd = [
                "ssh", "-q",
                "-o", "StrictHostKeyChecking=no",
                "-o", f"ConnectTimeout={self.connect_timeout}",
                "-p", str(node.ssh_port),
                node.ssh_target,
                decode_cmd,
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            _stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=encoded_data.encode("ascii")),
                timeout=self.operation_timeout,
            )

            if proc.returncode == 0:
                return TransportResult(
                    success=True,
                    transport_used="base64",
                    bytes_transferred=file_size,
                )
            else:
                return TransportResult(
                    success=False,
                    error=stderr.decode()[:200] if stderr else "base64 push failed",
                )

        except asyncio.TimeoutError:
            return TransportResult(success=False, error="Base64 transfer timeout")
        except MemoryError:
            return TransportResult(success=False, error="File too large for base64 in memory")
        except (OSError, IOError) as e:
            return TransportResult(success=False, error=str(e))

    async def _base64_pull(
        self,
        local_path: Path,
        remote_path: str,
        node: NodeConfig,
    ) -> TransportResult:
        """Pull file from remote using base64 encoding."""
        import base64

        try:
            cmd = [
                "ssh", "-q",
                "-o", "StrictHostKeyChecking=no",
                "-o", f"ConnectTimeout={self.connect_timeout}",
                "-p", str(node.ssh_port),
                node.ssh_target,
                f"base64 '{remote_path}'",
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.operation_timeout,
            )

            if proc.returncode == 0:
                # Decode and write locally
                file_data = base64.b64decode(stdout)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "wb") as f:
                    f.write(file_data)

                return TransportResult(
                    success=True,
                    transport_used="base64",
                    bytes_transferred=len(file_data),
                )
            else:
                return TransportResult(
                    success=False,
                    error=stderr.decode()[:200] if stderr else "base64 pull failed",
                )

        except asyncio.TimeoutError:
            return TransportResult(success=False, error="Base64 transfer timeout")
        except base64.binascii.Error as e:
            return TransportResult(success=False, error=f"Base64 decode error: {e}")
        except (OSError, IOError) as e:
            return TransportResult(success=False, error=str(e))

    async def _transfer_via_http(
        self,
        local_path: Path,
        remote_path: str,
        node: NodeConfig,
        direction: str,
    ) -> TransportResult:
        """Transfer file via HTTP using P2P file download endpoints.

        This transport uses the /files/models/ and /files/data/ endpoints
        on the P2P orchestrator (port 8770) to download files when SSH fails.

        **When this is useful:**
        - All SSH-based transports fail (connection resets, timeouts)
        - Pulling files from nodes with P2P orchestrator running
        - Large files where SSH connections are unstable

        **Limitations:**
        - Currently only supports "pull" direction (remote -> local)
        - Requires P2P orchestrator to be running on the remote node
        - Only supports files in models/ or data/ directories

        December 2025 - Added as permanent workaround for SSH connectivity issues
        """
        if direction == "push":
            # HTTP push not supported yet - would need to implement upload endpoint
            return TransportResult(
                success=False,
                error="HTTP push not implemented - use rsync/base64 for push",
            )

        try:
            import aiohttp
        except ImportError:
            return TransportResult(success=False, error="aiohttp not available")

        # Determine the file type and endpoint from the path
        # Expected remote_path format: ai-service/models/foo.pth or ai-service/data/foo.db
        if "models/" in remote_path:
            # Extract filename after models/
            file_name = remote_path.split("models/")[-1]
            endpoint = f"/files/models/{file_name}"
        elif "data/" in remote_path:
            # Extract filename after data/
            file_name = remote_path.split("data/")[-1]
            endpoint = f"/files/data/{file_name}"
        else:
            return TransportResult(
                success=False,
                error=f"HTTP transfer only supports models/ or data/ paths, got: {remote_path}",
            )

        # Build URL using P2P port
        host = node.tailscale_ip or node.hostname
        p2p_port = getattr(node, 'p2p_port', 8770)
        url = f"http://{host}:{p2p_port}{endpoint}"

        try:
            timeout = aiohttp.ClientTimeout(total=self.operation_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status == 404:
                        return TransportResult(
                            success=False,
                            error=f"File not found via HTTP: {endpoint}",
                        )
                    if resp.status != 200:
                        error_text = await resp.text()
                        return TransportResult(
                            success=False,
                            error=f"HTTP {resp.status}: {error_text[:100]}",
                        )

                    # Stream download to file
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    bytes_downloaded = 0
                    with open(local_path, "wb") as f:
                        async for chunk in resp.content.iter_chunked(1024 * 1024):
                            f.write(chunk)
                            bytes_downloaded += len(chunk)

                    logger.info(
                        f"HTTP download complete: {url} -> {local_path} "
                        f"({bytes_downloaded / 1024 / 1024:.1f} MB)"
                    )

                    return TransportResult(
                        success=True,
                        transport_used="http",
                        bytes_transferred=bytes_downloaded,
                    )

        except aiohttp.ClientError as e:
            return TransportResult(success=False, error=f"HTTP client error: {e}")
        except asyncio.TimeoutError:
            return TransportResult(success=False, error="HTTP transfer timeout")
        except (OSError, IOError) as e:
            return TransportResult(success=False, error=f"HTTP file write error: {e}")

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

        # Extract hostname from remote_spec (format: user@host:path)
        # December 2025: Add bandwidth limiting to prevent network saturation
        bwlimit_arg = ""
        if HAS_BANDWIDTH_CONFIG and get_node_bandwidth_kbs:
            try:
                # Parse hostname from remote_spec
                if "@" in remote_spec and ":" in remote_spec:
                    host_part = remote_spec.split("@")[1].split(":")[0]
                    bwlimit_kbs = get_node_bandwidth_kbs(host_part)
                    if bwlimit_kbs > 0:
                        bwlimit_arg = f"--bwlimit={bwlimit_kbs}"
            except (IndexError, ValueError):
                pass  # Use no bandwidth limit on parse error

        cmd = [
            "rsync", "-az", f"--timeout={self.operation_timeout}",
            "-e", f"ssh -p {ssh_port} -o StrictHostKeyChecking=no "
                  f"-o ConnectTimeout={self.connect_timeout}",
        ]
        if bwlimit_arg:
            cmd.append(bwlimit_arg)
        cmd.extend([src, dst])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await asyncio.wait_for(
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
        except (OSError, FileNotFoundError, PermissionError) as e:
            # OSError: subprocess/network errors
            # FileNotFoundError: rsync or file not found
            # PermissionError: insufficient permissions
            return TransportResult(success=False, error=str(e))

    async def http_request(
        self,
        node: NodeConfig,
        endpoint: str,
        method: str = "GET",
        data: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        timeout: int | None = None,
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

                kwargs: dict[str, Any] = {}
                if data:
                    kwargs["data"] = data
                if json_data:
                    kwargs["json"] = json_data

                async with session.request(method, url, **kwargs) as resp:
                    response_data = None
                    try:
                        response_data = await resp.json()
                    except (ValueError, TypeError, aiohttp.ContentTypeError):
                        # ValueError: invalid JSON
                        # TypeError: type conversion error
                        # ContentTypeError: invalid content type
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
        except (aiohttp.ClientError, OSError) as e:
            # aiohttp.ClientError: HTTP client errors (connection, request, response)
            # OSError: network errors
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
        except (asyncio.TimeoutError, OSError, FileNotFoundError) as e:
            # asyncio.TimeoutError: SSH connection timeout
            # OSError: subprocess/network errors
            # FileNotFoundError: ssh command not found
            logger.debug(f"SSH reachability check failed for {node.hostname}: {e}")
            return False

    def reset_circuit_breakers(self) -> None:
        """Reset all circuit breakers."""
        self._circuit_breaker.reset_all()

    def get_health_summary(self) -> dict[str, Any]:
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

    def health_check(self) -> "HealthCheckResult":
        """Return health status for DaemonManager integration.

        December 2025: Added for unified health monitoring across all
        coordination components.

        Returns:
            HealthCheckResult with transport layer status.
        """
        if HealthCheckResult is None:
            # Fallback when protocols not importable
            return {"healthy": True, "status": "unknown", "details": {}}  # type: ignore

        all_states = self._circuit_breaker.get_all_states()
        total_circuits = len(all_states)
        open_circuits = sum(
            1 for s in all_states.values()
            if s.state.value == "open"
        )
        half_open_circuits = sum(
            1 for s in all_states.values()
            if s.state.value == "half_open"
        )

        # Determine health status based on circuit breaker states
        if total_circuits == 0:
            status = "healthy"
            healthy = True
        elif open_circuits > total_circuits * 0.5:
            status = "unhealthy"
            healthy = False
        elif open_circuits > 0 or half_open_circuits > 0:
            status = "degraded"
            healthy = True
        else:
            status = "healthy"
            healthy = True

        return HealthCheckResult(
            healthy=healthy,
            status=status,
            details={
                "total_circuits": total_circuits,
                "open_circuits": open_circuits,
                "half_open_circuits": half_open_circuits,
                "closed_circuits": total_circuits - open_circuits - half_open_circuits,
                "p2p_url": self.p2p_url,
            },
        )


# Singleton instance for convenience
_transport_instance: ClusterTransport | None = None


def get_cluster_transport() -> ClusterTransport:
    """Get the singleton ClusterTransport instance."""
    global _transport_instance
    if _transport_instance is None:
        _transport_instance = ClusterTransport()
    return _transport_instance


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Main class
    "ClusterTransport",
    # Data classes
    "NodeConfig",
    "TimeoutConfig",  # Canonical name (Dec 2025)
    "TransportConfig",  # Deprecated alias for backward compat
    "TransportResult",
    # Error classes
    "TransportError",
    "RetryableTransportError",
    "PermanentTransportError",
    # Functions
    "get_cluster_transport",
]
