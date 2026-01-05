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

from ..loops.loop_constants import LoopTimeouts
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
        self._base_timeout = timeout
        # node_id -> ssh_host mapping
        self._ssh_hosts: dict[str, str] = {}
        # node_id -> provider mapping for timeout adjustments
        self._node_providers: dict[str, str] = {}

    def configure_host(
        self, node_id: str, ssh_host: str, ssh_port: int = 22, provider: str = ""
    ) -> None:
        """Configure SSH host for a node.

        Args:
            node_id: Unique node identifier
            ssh_host: SSH hostname or IP
            ssh_port: SSH port (default 22)
            provider: Cloud provider name for timeout adjustment (e.g., "vast", "lambda")
        """
        self._ssh_hosts[node_id] = f"{ssh_host}:{ssh_port}"
        if provider:
            self._node_providers[node_id] = provider

    def _get_timeout_for_node(self, target: str) -> float:
        """Get provider-adjusted timeout for a target node.

        January 5, 2026: Apply provider-specific multipliers for slow connections.
        - Vast.ai: 2.0x (consumer networks, variable latency)
        - Lambda: 1.5x (NAT-blocked GH200s)
        - RunPod: 1.5x (higher latency)
        """
        provider = self._node_providers.get(target, "")
        if not provider:
            # Try to infer provider from node_id
            provider = self._infer_provider(target)
        multiplier = LoopTimeouts.get_provider_multiplier(provider)
        return self._base_timeout * multiplier

    def _infer_provider(self, node_id: str) -> str:
        """Infer provider from node_id naming convention."""
        node_lower = node_id.lower()
        if node_lower.startswith("vast-") or "vast" in node_lower:
            return "vast"
        if node_lower.startswith("lambda-") or "gh200" in node_lower:
            return "lambda"
        if node_lower.startswith("runpod-"):
            return "runpod"
        if node_lower.startswith("nebius-"):
            return "nebius"
        if node_lower.startswith("vultr-"):
            return "vultr"
        if node_lower.startswith("hetzner-"):
            return "hetzner"
        return ""

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
            # January 5, 2026: Added UserKnownHostsFile=/dev/null for ephemeral hosts
            # (Vast.ai containers regenerate SSH keys on restart)
            cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
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

            # January 5, 2026: Use provider-adjusted timeout for slow connections
            timeout = self._get_timeout_for_node(target)
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=payload),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                return self._make_result(
                    success=False,
                    latency_ms=(time.time() - start_time) * 1000,
                    error=f"SSH command timeout ({timeout:.1f}s)",
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
            # January 5, 2026: Added UserKnownHostsFile=/dev/null for ephemeral hosts
            cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
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

            # January 5, 2026: Use provider-adjusted timeout for availability check
            # Base 5s timeout gets multiplied for slow providers (Vast.ai: 10s, Lambda: 7.5s)
            availability_timeout = 5.0 * LoopTimeouts.get_provider_multiplier(
                self._infer_provider(target)
            )
            try:
                await asyncio.wait_for(proc.communicate(), timeout=availability_timeout)
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
