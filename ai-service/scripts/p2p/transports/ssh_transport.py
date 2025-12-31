"""
SSH tunnel transport implementation.

Tier 3 (TUNNELED): SSH port forwarding for NAT traversal.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from ..transport_cascade import BaseTransport, TransportResult, TransportTier

logger = logging.getLogger(__name__)


class SSHTunnelTransport(BaseTransport):
    """
    SSH tunnel transport for NAT traversal.

    Tier 3 (TUNNELED): Works through most firewalls via SSH.
    Uses asyncio subprocess for non-blocking operation.
    """

    name = "ssh_tunnel"
    tier = TransportTier.TIER_3_TUNNELED

    def __init__(
        self,
        ssh_key_path: str | None = None,
        ssh_user: str = "root",
        ssh_port: int = 22,
        timeout: float = 30.0,
    ):
        self._ssh_key_path = ssh_key_path or os.path.expanduser("~/.ssh/id_ed25519")
        self._ssh_user = ssh_user
        self._ssh_port = ssh_port
        self._timeout = timeout
        # node_id -> ssh_host mapping
        self._ssh_hosts: dict[str, str] = {}

    def configure_host(self, node_id: str, ssh_host: str, ssh_port: int = 22) -> None:
        """Configure SSH host for a node."""
        self._ssh_hosts[node_id] = f"{ssh_host}:{ssh_port}"

    async def send(self, target: str, payload: bytes) -> TransportResult:
        """Send payload via SSH tunnel using netcat or curl."""
        ssh_host = self._get_ssh_host(target)
        if not ssh_host:
            return self._make_result(
                success=False,
                latency_ms=0,
                error=f"No SSH host configured for: {target}",
            )

        host, port = ssh_host.split(":")
        start_time = time.time()

        try:
            # Use SSH to execute a curl command on the remote node
            # This sends the payload to localhost:8770 on the remote
            cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=10",
                "-o", "BatchMode=yes",
                "-i", self._ssh_key_path,
                "-p", port,
                f"{self._ssh_user}@{host}",
                f"curl -s -X POST -d @- http://localhost:8770/cascade",
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=payload),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                return self._make_result(
                    success=False,
                    latency_ms=(time.time() - start_time) * 1000,
                    error=f"SSH command timeout ({self._timeout}s)",
                )

            latency_ms = (time.time() - start_time) * 1000

            if proc.returncode == 0:
                return self._make_result(
                    success=True,
                    latency_ms=latency_ms,
                    response=stdout,
                    ssh_host=ssh_host,
                )
            else:
                error_msg = stderr.decode().strip() or f"SSH exit code {proc.returncode}"
                return self._make_result(
                    success=False,
                    latency_ms=latency_ms,
                    error=error_msg,
                )

        except Exception as e:
            return self._make_result(
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=f"{type(e).__name__}: {e}",
            )

    async def is_available(self, target: str) -> bool:
        """Check if SSH transport is available."""
        ssh_host = self._get_ssh_host(target)
        if not ssh_host:
            return False

        # Quick SSH connection test
        host, port = ssh_host.split(":")
        try:
            cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=3",
                "-o", "BatchMode=yes",
                "-i", self._ssh_key_path,
                "-p", port,
                f"{self._ssh_user}@{host}",
                "echo ok",
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                await asyncio.wait_for(proc.communicate(), timeout=5.0)
                return proc.returncode == 0
            except asyncio.TimeoutError:
                proc.kill()
                return False

        except Exception:
            return False

    def _get_ssh_host(self, target: str) -> str | None:
        """Get SSH host for a target."""
        # Check configured mapping
        if target in self._ssh_hosts:
            return self._ssh_hosts[target]

        # If target looks like host:port, use it directly
        if "@" in target or "." in target:
            if ":" not in target:
                return f"{target}:{self._ssh_port}"
            return target

        return None
