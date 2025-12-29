"""Transport Manager - Unified data transfer layer.

This module provides the TransportManager class which handles all data
transfers across the cluster with automatic transport selection and fallback.

Part of the Unified Data Plane Daemon architecture (December 2025).

Consolidates transfer logic from:
- cluster_transport.py (multi-transport)
- sync_bandwidth.py (bandwidth limiting)
- resilient_transfer.py (retry logic)
- dynamic_data_distribution.py (rsync/HTTP fallback)

Usage:
    from app.coordination.transport_manager import (
        TransportManager,
        get_transport_manager,
        Transport,
    )

    manager = get_transport_manager()

    # Transfer a file
    result = await manager.transfer_file(
        source_node="node-1",
        target_node="node-2",
        source_path="/data/games/hex8_2p.db",
        target_path="/data/games/hex8_2p.db",
    )

    # Or use transport directly
    result = await manager.transfer_via(
        transport=Transport.RSYNC,
        source="node-1:/data/file.db",
        target="node-2:/data/file.db",
    )
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from app.config.ports import P2P_DEFAULT_PORT

logger = logging.getLogger(__name__)

__all__ = [
    "Transport",
    "TransportResult",
    "TransportConfig",
    "TransportManager",
    "get_transport_manager",
    "reset_transport_manager",
]


class Transport(Enum):
    """Available transport mechanisms."""

    P2P_GOSSIP = "p2p"  # P2P HTTP gossip (fast for small files)
    HTTP_FETCH = "http"  # Direct HTTP download
    RSYNC = "rsync"  # rsync over SSH (reliable for large files)
    S3 = "s3"  # AWS S3 API
    SCP = "scp"  # SCP over SSH
    BASE64_SSH = "base64"  # Base64 over SSH (last resort)


@dataclass
class TransportResult:
    """Result of a transfer operation."""

    success: bool
    transport_used: Transport
    bytes_transferred: int = 0
    duration_seconds: float = 0.0
    error: str | None = None
    retries: int = 0
    checksum: str = ""  # SHA256 of transferred data

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "transport": self.transport_used.value,
            "bytes": self.bytes_transferred,
            "duration": self.duration_seconds,
            "error": self.error,
            "retries": self.retries,
            "checksum": self.checksum,
        }


@dataclass
class TransportConfig:
    """Configuration for TransportManager."""

    # Timeouts (seconds)
    connect_timeout: float = 10.0
    transfer_timeout: float = 600.0  # 10 minutes for large files
    small_file_timeout: float = 60.0

    # Size thresholds
    large_file_threshold_bytes: int = 100 * 1024 * 1024  # 100MB
    small_file_threshold_bytes: int = 1 * 1024 * 1024  # 1MB

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    retry_backoff_multiplier: float = 2.0

    # Bandwidth limits (bytes/second, 0 = unlimited)
    default_bandwidth_limit: int = 0
    rsync_bandwidth_limit: int = 50 * 1024 * 1024  # 50 MB/s

    # SSH settings
    ssh_key_path: str = "~/.ssh/id_cluster"
    ssh_connect_timeout: int = 10
    ssh_options: list[str] = field(
        default_factory=lambda: [
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
        ]
    )

    # P2P settings
    p2p_port: int = P2P_DEFAULT_PORT
    http_data_port: int = 8780

    # S3 settings
    s3_bucket: str = "ringrift-models-20251214"


# Transport selection chains by scenario
TRANSPORT_CHAINS: dict[str, list[Transport]] = {
    "small_file": [
        Transport.P2P_GOSSIP,
        Transport.HTTP_FETCH,
        Transport.SCP,
        Transport.BASE64_SSH,
    ],
    "large_file": [
        Transport.RSYNC,
        Transport.SCP,
        Transport.HTTP_FETCH,
        Transport.BASE64_SSH,
    ],
    "s3_backup": [Transport.S3, Transport.RSYNC],
    "ephemeral_urgent": [
        Transport.RSYNC,
        Transport.SCP,
        Transport.BASE64_SSH,
    ],  # Skip P2P for speed
    "model_distribution": [Transport.P2P_GOSSIP, Transport.RSYNC, Transport.SCP],
    "default": [Transport.RSYNC, Transport.SCP, Transport.HTTP_FETCH, Transport.BASE64_SSH],
}


class TransportManager:
    """Manages data transfers across the cluster.

    Provides unified interface for all transfer operations with:
    - Automatic transport selection based on file size and scenario
    - Fallback chains when primary transport fails
    - Bandwidth limiting per transport type
    - Retry with exponential backoff
    - Transfer verification via checksums
    """

    def __init__(self, config: TransportConfig | None = None):
        """Initialize the transport manager.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or TransportConfig()

        # Stats tracking
        self._stats: dict[str, Any] = {
            "total_transfers": 0,
            "successful_transfers": 0,
            "failed_transfers": 0,
            "total_bytes": 0,
            "by_transport": {t.value: 0 for t in Transport},
            "errors_by_transport": {t.value: 0 for t in Transport},
        }

        # Circuit breakers per transport (simple implementation)
        self._circuit_breakers: dict[Transport, dict] = {
            t: {"failures": 0, "last_failure": 0.0, "open_until": 0.0}
            for t in Transport
        }
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_reset_time = 300.0  # 5 minutes

        # Node info cache
        self._node_info_cache: dict[str, dict[str, Any]] = {}

    def select_transport_chain(
        self,
        size_bytes: int = 0,
        scenario: str | None = None,
        is_ephemeral: bool = False,
    ) -> list[Transport]:
        """Select transport chain based on file characteristics.

        Args:
            size_bytes: File size in bytes.
            scenario: Optional explicit scenario name.
            is_ephemeral: Whether target is an ephemeral node.

        Returns:
            List of transports to try in order.
        """
        if scenario and scenario in TRANSPORT_CHAINS:
            return TRANSPORT_CHAINS[scenario]

        if is_ephemeral:
            return TRANSPORT_CHAINS["ephemeral_urgent"]

        if size_bytes > self.config.large_file_threshold_bytes:
            return TRANSPORT_CHAINS["large_file"]

        if size_bytes < self.config.small_file_threshold_bytes:
            return TRANSPORT_CHAINS["small_file"]

        return TRANSPORT_CHAINS["default"]

    def _is_circuit_open(self, transport: Transport) -> bool:
        """Check if circuit breaker is open for transport."""
        cb = self._circuit_breakers[transport]
        if cb["open_until"] > time.time():
            return True
        return False

    def _record_transport_failure(self, transport: Transport) -> None:
        """Record a transport failure for circuit breaker."""
        cb = self._circuit_breakers[transport]
        cb["failures"] += 1
        cb["last_failure"] = time.time()

        if cb["failures"] >= self._circuit_breaker_threshold:
            cb["open_until"] = time.time() + self._circuit_breaker_reset_time
            logger.warning(
                f"Circuit breaker OPEN for {transport.value} "
                f"(failures: {cb['failures']})"
            )

    def _record_transport_success(self, transport: Transport) -> None:
        """Record transport success, reset circuit breaker."""
        cb = self._circuit_breakers[transport]
        cb["failures"] = 0
        cb["open_until"] = 0.0

    async def transfer_file(
        self,
        source_node: str,
        target_node: str,
        source_path: str,
        target_path: str,
        size_bytes: int = 0,
        scenario: str | None = None,
        verify_checksum: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> TransportResult:
        """Transfer a file between nodes with automatic transport selection.

        Args:
            source_node: Source node ID.
            target_node: Target node ID.
            source_path: Path on source node.
            target_path: Path on target node.
            size_bytes: File size (for transport selection).
            scenario: Optional explicit scenario for transport selection.
            verify_checksum: Whether to verify checksum after transfer.
            progress_callback: Optional callback for progress updates.

        Returns:
            TransportResult with success status and details.
        """
        start_time = time.time()
        self._stats["total_transfers"] += 1

        # Determine if target is ephemeral
        is_ephemeral = self._is_ephemeral_node(target_node)

        # Select transport chain
        chain = self.select_transport_chain(
            size_bytes=size_bytes,
            scenario=scenario,
            is_ephemeral=is_ephemeral,
        )

        # Filter out transports with open circuit breakers
        available_chain = [t for t in chain if not self._is_circuit_open(t)]
        if not available_chain:
            # All circuits open, reset the first one to try
            available_chain = [chain[0]]
            self._circuit_breakers[chain[0]]["open_until"] = 0.0

        logger.debug(
            f"Transfer {source_node}:{source_path} -> {target_node}:{target_path} "
            f"using chain: {[t.value for t in available_chain]}"
        )

        # Try each transport in chain
        all_errors: list[str] = []
        retries = 0

        for transport in available_chain:
            for attempt in range(self.config.max_retries):
                try:
                    result = await self._execute_transfer(
                        transport=transport,
                        source_node=source_node,
                        target_node=target_node,
                        source_path=source_path,
                        target_path=target_path,
                        size_bytes=size_bytes,
                    )

                    if result.success:
                        result.duration_seconds = time.time() - start_time
                        result.retries = retries

                        # Verify checksum if requested
                        if verify_checksum and result.checksum:
                            verified = await self._verify_checksum(
                                target_node, target_path, result.checksum
                            )
                            if not verified:
                                all_errors.append(
                                    f"{transport.value}: Checksum mismatch"
                                )
                                self._record_transport_failure(transport)
                                continue

                        self._record_transport_success(transport)
                        self._stats["successful_transfers"] += 1
                        self._stats["total_bytes"] += result.bytes_transferred
                        self._stats["by_transport"][transport.value] += 1

                        logger.info(
                            f"Transfer complete via {transport.value}: "
                            f"{result.bytes_transferred} bytes in {result.duration_seconds:.1f}s"
                        )
                        return result

                    all_errors.append(f"{transport.value}: {result.error}")
                    retries += 1

                except asyncio.TimeoutError:
                    all_errors.append(f"{transport.value}: Timeout")
                    retries += 1
                except Exception as e:
                    all_errors.append(f"{transport.value}: {e}")
                    retries += 1

                # Retry delay with backoff
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay_seconds * (
                        self.config.retry_backoff_multiplier**attempt
                    )
                    await asyncio.sleep(delay)

            # Transport exhausted, record failure
            self._record_transport_failure(transport)
            self._stats["errors_by_transport"][transport.value] += 1

        # All transports failed
        self._stats["failed_transfers"] += 1
        logger.error(f"Transfer failed after trying all transports: {all_errors}")

        return TransportResult(
            success=False,
            transport_used=available_chain[-1],
            duration_seconds=time.time() - start_time,
            error="; ".join(all_errors[-3:]),  # Last 3 errors
            retries=retries,
        )

    async def _execute_transfer(
        self,
        transport: Transport,
        source_node: str,
        target_node: str,
        source_path: str,
        target_path: str,
        size_bytes: int,
    ) -> TransportResult:
        """Execute transfer using specific transport."""
        if transport == Transport.RSYNC:
            return await self._transfer_rsync(
                source_node, target_node, source_path, target_path
            )
        elif transport == Transport.SCP:
            return await self._transfer_scp(
                source_node, target_node, source_path, target_path
            )
        elif transport == Transport.HTTP_FETCH:
            return await self._transfer_http(
                source_node, target_node, source_path, target_path
            )
        elif transport == Transport.P2P_GOSSIP:
            return await self._transfer_p2p(
                source_node, target_node, source_path, target_path
            )
        elif transport == Transport.S3:
            return await self._transfer_s3(
                source_node, target_node, source_path, target_path
            )
        elif transport == Transport.BASE64_SSH:
            return await self._transfer_base64(
                source_node, target_node, source_path, target_path
            )
        else:
            return TransportResult(
                success=False,
                transport_used=transport,
                error=f"Unknown transport: {transport}",
            )

    async def _transfer_rsync(
        self,
        source_node: str,
        target_node: str,
        source_path: str,
        target_path: str,
    ) -> TransportResult:
        """Transfer via rsync over SSH."""
        source_spec = self._get_rsync_spec(source_node, source_path)
        target_spec = self._get_rsync_spec(target_node, target_path)

        cmd = [
            "rsync",
            "-avz",
            "--progress",
            f"--bwlimit={self.config.rsync_bandwidth_limit // 1024}k",
            "-e",
            f"ssh {' '.join(self.config.ssh_options)} -i {Path(self.config.ssh_key_path).expanduser()}",
            source_spec,
            target_spec,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.transfer_timeout,
            )

            if process.returncode == 0:
                # Parse bytes transferred from rsync output
                bytes_transferred = self._parse_rsync_bytes(stdout.decode())
                return TransportResult(
                    success=True,
                    transport_used=Transport.RSYNC,
                    bytes_transferred=bytes_transferred,
                )
            else:
                return TransportResult(
                    success=False,
                    transport_used=Transport.RSYNC,
                    error=stderr.decode()[:200],
                )

        except asyncio.TimeoutError:
            return TransportResult(
                success=False,
                transport_used=Transport.RSYNC,
                error="Rsync timeout",
            )

    async def _transfer_scp(
        self,
        source_node: str,
        target_node: str,
        source_path: str,
        target_path: str,
    ) -> TransportResult:
        """Transfer via SCP."""
        source_spec = self._get_scp_spec(source_node, source_path)
        target_spec = self._get_scp_spec(target_node, target_path)

        cmd = [
            "scp",
            *self.config.ssh_options,
            "-i",
            str(Path(self.config.ssh_key_path).expanduser()),
            source_spec,
            target_spec,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.transfer_timeout,
            )

            if process.returncode == 0:
                return TransportResult(
                    success=True,
                    transport_used=Transport.SCP,
                )
            else:
                return TransportResult(
                    success=False,
                    transport_used=Transport.SCP,
                    error=stderr.decode()[:200],
                )

        except asyncio.TimeoutError:
            return TransportResult(
                success=False,
                transport_used=Transport.SCP,
                error="SCP timeout",
            )

    async def _transfer_http(
        self,
        source_node: str,
        target_node: str,
        source_path: str,
        target_path: str,
    ) -> TransportResult:
        """Transfer via HTTP download on target node."""
        # Get HTTP URL for source
        source_ip = self._get_node_ip(source_node)
        if not source_ip:
            return TransportResult(
                success=False,
                transport_used=Transport.HTTP_FETCH,
                error=f"Unknown source node: {source_node}",
            )

        url = f"http://{source_ip}:{self.config.http_data_port}/{source_path}"

        # Run wget/curl on target node
        target_ssh = self._get_ssh_target(target_node)
        if not target_ssh:
            return TransportResult(
                success=False,
                transport_used=Transport.HTTP_FETCH,
                error=f"Unknown target node: {target_node}",
            )

        # Ensure directory exists and download
        cmd = f"mkdir -p $(dirname {target_path}) && wget -q -O {target_path} '{url}'"

        ssh_cmd = [
            "ssh",
            *self.config.ssh_options,
            "-i",
            str(Path(self.config.ssh_key_path).expanduser()),
            target_ssh,
            cmd,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.transfer_timeout,
            )

            if process.returncode == 0:
                return TransportResult(
                    success=True,
                    transport_used=Transport.HTTP_FETCH,
                )
            else:
                return TransportResult(
                    success=False,
                    transport_used=Transport.HTTP_FETCH,
                    error=stderr.decode()[:200],
                )

        except asyncio.TimeoutError:
            return TransportResult(
                success=False,
                transport_used=Transport.HTTP_FETCH,
                error="HTTP fetch timeout",
            )

    async def _transfer_p2p(
        self,
        source_node: str,
        target_node: str,
        source_path: str,
        target_path: str,
    ) -> TransportResult:
        """Transfer via P2P gossip endpoint."""
        try:
            import aiohttp

            source_ip = self._get_node_ip(source_node)
            target_ip = self._get_node_ip(target_node)

            if not source_ip or not target_ip:
                return TransportResult(
                    success=False,
                    transport_used=Transport.P2P_GOSSIP,
                    error="Unknown node IP",
                )

            # Use P2P /pull_file endpoint
            url = f"http://{source_ip}:{self.config.p2p_port}/pull_file"
            payload = {
                "path": source_path,
                "target_node": target_node,
                "target_path": target_path,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.transfer_timeout),
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return TransportResult(
                            success=result.get("success", False),
                            transport_used=Transport.P2P_GOSSIP,
                            bytes_transferred=result.get("bytes", 0),
                        )
                    else:
                        text = await response.text()
                        return TransportResult(
                            success=False,
                            transport_used=Transport.P2P_GOSSIP,
                            error=f"HTTP {response.status}: {text[:100]}",
                        )

        except ImportError:
            return TransportResult(
                success=False,
                transport_used=Transport.P2P_GOSSIP,
                error="aiohttp not available",
            )
        except Exception as e:
            return TransportResult(
                success=False,
                transport_used=Transport.P2P_GOSSIP,
                error=str(e)[:200],
            )

    async def _transfer_s3(
        self,
        source_node: str,
        target_node: str,
        source_path: str,
        target_path: str,
    ) -> TransportResult:
        """Transfer via S3 (upload from source, download to target)."""
        s3_key = f"transfers/{Path(source_path).name}"
        s3_uri = f"s3://{self.config.s3_bucket}/{s3_key}"

        # Upload from source
        if source_node != "s3":
            source_ssh = self._get_ssh_target(source_node)
            if source_ssh:
                upload_cmd = f"aws s3 cp {source_path} {s3_uri}"
                ssh_cmd = [
                    "ssh",
                    *self.config.ssh_options,
                    "-i",
                    str(Path(self.config.ssh_key_path).expanduser()),
                    source_ssh,
                    upload_cmd,
                ]
                try:
                    process = await asyncio.create_subprocess_exec(
                        *ssh_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    _, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.config.transfer_timeout,
                    )
                    if process.returncode != 0:
                        return TransportResult(
                            success=False,
                            transport_used=Transport.S3,
                            error=f"S3 upload failed: {stderr.decode()[:100]}",
                        )
                except asyncio.TimeoutError:
                    return TransportResult(
                        success=False,
                        transport_used=Transport.S3,
                        error="S3 upload timeout",
                    )

        # Download to target
        if target_node != "s3":
            target_ssh = self._get_ssh_target(target_node)
            if target_ssh:
                download_cmd = f"mkdir -p $(dirname {target_path}) && aws s3 cp {s3_uri} {target_path}"
                ssh_cmd = [
                    "ssh",
                    *self.config.ssh_options,
                    "-i",
                    str(Path(self.config.ssh_key_path).expanduser()),
                    target_ssh,
                    download_cmd,
                ]
                try:
                    process = await asyncio.create_subprocess_exec(
                        *ssh_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    _, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.config.transfer_timeout,
                    )
                    if process.returncode != 0:
                        return TransportResult(
                            success=False,
                            transport_used=Transport.S3,
                            error=f"S3 download failed: {stderr.decode()[:100]}",
                        )
                except asyncio.TimeoutError:
                    return TransportResult(
                        success=False,
                        transport_used=Transport.S3,
                        error="S3 download timeout",
                    )

        return TransportResult(
            success=True,
            transport_used=Transport.S3,
        )

    async def _transfer_base64(
        self,
        source_node: str,
        target_node: str,
        source_path: str,
        target_path: str,
    ) -> TransportResult:
        """Transfer via base64 encoding over SSH (last resort).

        This works when binary transfers fail due to connection issues.
        """
        source_ssh = self._get_ssh_target(source_node)
        target_ssh = self._get_ssh_target(target_node)

        if not source_ssh or not target_ssh:
            return TransportResult(
                success=False,
                transport_used=Transport.BASE64_SSH,
                error="Unknown node SSH target",
            )

        # Pipe: cat on source | base64 | ssh target | base64 -d > file
        cmd = (
            f"ssh {' '.join(self.config.ssh_options)} "
            f"-i {Path(self.config.ssh_key_path).expanduser()} "
            f"{source_ssh} 'cat {source_path} | base64' | "
            f"ssh {' '.join(self.config.ssh_options)} "
            f"-i {Path(self.config.ssh_key_path).expanduser()} "
            f"{target_ssh} 'mkdir -p $(dirname {target_path}) && base64 -d > {target_path}'"
        )

        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.transfer_timeout,
            )

            if process.returncode == 0:
                return TransportResult(
                    success=True,
                    transport_used=Transport.BASE64_SSH,
                )
            else:
                return TransportResult(
                    success=False,
                    transport_used=Transport.BASE64_SSH,
                    error=stderr.decode()[:200],
                )

        except asyncio.TimeoutError:
            return TransportResult(
                success=False,
                transport_used=Transport.BASE64_SSH,
                error="Base64 transfer timeout",
            )

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _is_ephemeral_node(self, node_id: str) -> bool:
        """Check if node is ephemeral (Vast.ai, spot instances, etc.)."""
        ephemeral_patterns = ["vast-", "spot-", "preempt-"]
        return any(node_id.lower().startswith(p) for p in ephemeral_patterns)

    def _get_node_ip(self, node_id: str) -> str | None:
        """Get IP address for a node."""
        # Check cache first
        if node_id in self._node_info_cache:
            return self._node_info_cache[node_id].get("ip")

        # Try to load from cluster config
        try:
            from app.config.cluster_config import get_cluster_nodes

            nodes = get_cluster_nodes()
            if node_id in nodes:
                ip = nodes[node_id].best_ip
                self._node_info_cache[node_id] = {"ip": ip}
                return ip
        except ImportError:
            pass

        return None

    def _get_ssh_target(self, node_id: str) -> str | None:
        """Get SSH target (user@host) for a node."""
        try:
            from app.config.cluster_config import get_cluster_nodes

            nodes = get_cluster_nodes()
            if node_id in nodes:
                node = nodes[node_id]
                user = node.ssh_user or "ubuntu"
                host = node.best_ip
                return f"{user}@{host}"
        except ImportError:
            pass

        return None

    def _get_rsync_spec(self, node_id: str, path: str) -> str:
        """Get rsync path specification for a node."""
        if node_id == "local":
            return path

        ssh_target = self._get_ssh_target(node_id)
        if ssh_target:
            return f"{ssh_target}:{path}"

        return path

    def _get_scp_spec(self, node_id: str, path: str) -> str:
        """Get SCP path specification for a node."""
        return self._get_rsync_spec(node_id, path)

    def _parse_rsync_bytes(self, output: str) -> int:
        """Parse bytes transferred from rsync output."""
        import re

        # Look for "sent X bytes" or "total size is X"
        match = re.search(r"sent\s+([\d,]+)\s+bytes", output)
        if match:
            return int(match.group(1).replace(",", ""))

        match = re.search(r"total size is\s+([\d,]+)", output)
        if match:
            return int(match.group(1).replace(",", ""))

        return 0

    async def _verify_checksum(
        self, node_id: str, path: str, expected_checksum: str
    ) -> bool:
        """Verify file checksum on remote node."""
        ssh_target = self._get_ssh_target(node_id)
        if not ssh_target:
            return False

        cmd = f"sha256sum {path} | cut -d' ' -f1"
        ssh_cmd = [
            "ssh",
            *self.config.ssh_options,
            "-i",
            str(Path(self.config.ssh_key_path).expanduser()),
            ssh_target,
            cmd,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=30.0
            )

            if process.returncode == 0:
                actual = stdout.decode().strip()
                return actual == expected_checksum

        except (asyncio.TimeoutError, Exception):
            pass

        return False

    # =========================================================================
    # Health & Stats
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get transfer statistics."""
        return {
            **self._stats,
            "circuit_breakers": {
                t.value: {
                    "failures": cb["failures"],
                    "open": cb["open_until"] > time.time(),
                }
                for t, cb in self._circuit_breakers.items()
            },
        }

    def health_check(self) -> "HealthCheckResult":
        """Check transport manager health."""
        try:
            from app.coordination.contracts import CoordinatorStatus, HealthCheckResult
        except ImportError:
            from dataclasses import dataclass as dc

            @dc
            class HealthCheckResult:
                healthy: bool
                status: str
                message: str
                details: dict | None = None

            CoordinatorStatus = type(
                "CS", (), {"RUNNING": "running", "DEGRADED": "degraded"}
            )()

        stats = self.get_stats()

        # Check for issues
        open_breakers = sum(
            1
            for t in Transport
            if self._circuit_breakers[t]["open_until"] > time.time()
        )

        total = stats["total_transfers"]
        failed = stats["failed_transfers"]
        failure_rate = failed / total if total > 0 else 0

        issues = []
        if open_breakers >= len(Transport) - 1:
            issues.append(f"Most transports circuit-open ({open_breakers}/{len(Transport)})")
        if failure_rate > 0.5 and total > 10:
            issues.append(f"High failure rate: {failure_rate:.0%}")

        healthy = len(issues) == 0
        status = CoordinatorStatus.RUNNING if healthy else CoordinatorStatus.DEGRADED

        return HealthCheckResult(
            healthy=healthy,
            status=status,
            message="; ".join(issues) if issues else "TransportManager healthy",
            details={
                "total_transfers": total,
                "failed_transfers": failed,
                "failure_rate": f"{failure_rate:.1%}",
                "open_circuit_breakers": open_breakers,
                "by_transport": stats["by_transport"],
            },
        )


# =============================================================================
# Module-level singleton
# =============================================================================

_transport_singleton: TransportManager | None = None


def get_transport_manager(config: TransportConfig | None = None) -> TransportManager:
    """Get the global TransportManager singleton.

    Args:
        config: Configuration for manager. Only used on first call.

    Returns:
        The global TransportManager instance.
    """
    global _transport_singleton

    if _transport_singleton is None:
        _transport_singleton = TransportManager(config)
    return _transport_singleton


def reset_transport_manager() -> None:
    """Reset the transport manager singleton (for testing)."""
    global _transport_singleton
    _transport_singleton = None
