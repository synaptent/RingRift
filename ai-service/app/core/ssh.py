"""Unified SSH Helper Module for RingRift AI Service.

CANONICAL SSH UTILITY
=====================
This is the single source of truth for ALL SSH operations in the codebase.
Other modules should import from here instead of implementing their own SSH logic.

Features:
- Unified SSHClient class with async and sync methods
- Connection pooling via SSH ControlMaster (5 min persist)
- Automatic timeout handling
- SSH config loading from cluster_hosts.yaml
- Multi-transport fallback (Tailscale -> Direct -> Cloudflare)
- Backward-compatible function exports

Usage:
    from app.core.ssh import (
        SSHClient,
        get_ssh_client,
        run_ssh_command,
        run_ssh_command_async,
        SSHConfig,
        SSHResult,
    )

    # Async usage (recommended)
    client = get_ssh_client("runpod-h100")
    result = await client.run_async("nvidia-smi")

    # Sync usage
    result = client.run("nvidia-smi", timeout=30)

    # Convenience functions (backward compatible)
    result = await run_ssh_command_async("runpod-h100", "echo hello")

Migration from existing modules:
    - app/execution/executor.py:SSHExecutor -> app/core/ssh.SSHClient
    - app/distributed/hosts.py:SSHExecutor -> app/core/ssh.SSHClient
    - app/distributed/ssh_transport.py:SSHTransport -> app/core/ssh.SSHClient
    - app/providers/base.py:run_ssh_command -> app/core/ssh.run_ssh_command_async
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "SSHClient",
    "SSHConfig",
    "SSHResult",
    "get_ssh_client",
    "run_ssh_command",
    "run_ssh_command_async",
    "run_ssh_command_sync",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SSHConfig:
    """SSH connection configuration."""
    host: str
    port: int = 22
    user: str | None = None
    key_path: str | None = None
    connect_timeout: int = 10
    command_timeout: int = 60
    use_control_master: bool = True
    control_persist: int = 300  # 5 minutes
    server_alive_interval: int = 30
    server_alive_count_max: int = 4
    tailscale_ip: str | None = None
    work_dir: str | None = None
    venv_activate: str | None = None

    @classmethod
    def from_cluster_node(cls, node_id: str) -> SSHConfig | None:
        """Load SSH configuration from cluster_hosts.yaml."""
        try:
            from app.sync.cluster_hosts import get_cluster_nodes
            nodes = get_cluster_nodes()
            if node_id not in nodes:
                return None

            node = nodes[node_id]
            return cls(
                host=node.get("ssh_host", node.get("tailscale_ip", "")),
                port=node.get("ssh_port", 22),
                user=node.get("ssh_user", "ubuntu"),
                key_path=node.get("ssh_key"),
                tailscale_ip=node.get("tailscale_ip"),
                work_dir=node.get("ringrift_path", "~/ringrift/ai-service"),
            )
        except Exception as e:
            logger.debug(f"Failed to load SSH config for {node_id}: {e}")
            return None


@dataclass
class SSHResult:
    """Result of SSH command execution."""
    success: bool
    returncode: int
    stdout: str
    stderr: str
    elapsed_ms: float
    command: str
    host: str
    transport_used: str | None = None
    timed_out: bool = False
    error: str | None = None

    @property
    def output(self) -> str:
        return f"{self.stdout}\n{self.stderr}".strip()

    def __bool__(self) -> bool:
        return self.success


# =============================================================================
# SSH Client
# =============================================================================

class SSHClient:
    """Unified SSH client with connection pooling and multi-transport support."""

    _clients: dict[str, SSHClient] = {}
    _lock = asyncio.Lock() if hasattr(asyncio, "Lock") else None

    def __init__(
        self,
        config: SSHConfig,
        control_path_dir: Path | None = None,
    ):
        self._config = config
        self._control_path_dir = control_path_dir or Path.home() / ".ssh" / "ringrift_control"
        self._control_path_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    @property
    def config(self) -> SSHConfig:
        return self._config

    def _get_control_path(self) -> str:
        """Get ControlMaster socket path for this host."""
        safe_host = self._config.host.replace(".", "_").replace(":", "_")
        return str(self._control_path_dir / f"{safe_host}_{self._config.port}")

    def _build_ssh_command(
        self,
        remote_command: str,
        use_tailscale: bool = False,
    ) -> list[str]:
        """Build SSH command with proper options."""
        cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"ConnectTimeout={self._config.connect_timeout}",
            "-o", "BatchMode=yes",
            "-o", "LogLevel=ERROR",
            "-o", f"ServerAliveInterval={self._config.server_alive_interval}",
            "-o", f"ServerAliveCountMax={self._config.server_alive_count_max}",
            "-o", "TCPKeepAlive=yes",
        ]

        # Connection pooling via ControlMaster
        if self._config.use_control_master:
            control_path = self._get_control_path()
            cmd.extend([
                "-o", f"ControlPath={control_path}",
                "-o", "ControlMaster=auto",
                "-o", f"ControlPersist={self._config.control_persist}",
            ])

        # Port
        if self._config.port != 22:
            cmd.extend(["-p", str(self._config.port)])

        # Key file
        if self._config.key_path:
            key_path = os.path.expanduser(self._config.key_path)
            if os.path.exists(key_path):
                cmd.extend(["-i", key_path])

        # SSH target
        target_host = self._config.tailscale_ip if use_tailscale else self._config.host
        if self._config.user:
            target = f"{self._config.user}@{target_host}"
        else:
            target = target_host

        cmd.append(target)
        cmd.append(remote_command)

        return cmd

    async def run_async(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SSHResult:
        """Execute command asynchronously."""
        timeout = timeout or self._config.command_timeout
        start_time = time.time()

        # Build remote command
        remote_cmd = command
        if cwd or self._config.work_dir:
            work_dir = cwd or self._config.work_dir
            remote_cmd = f"cd {work_dir} && {command}"
        if env:
            env_str = " ".join(f"{k}={v}" for k, v in env.items())
            remote_cmd = f"{env_str} {remote_cmd}"
        if self._config.venv_activate:
            remote_cmd = f"source {self._config.venv_activate} && {remote_cmd}"

        # Try Tailscale first, then direct
        transports = []
        if self._config.tailscale_ip:
            transports.append(("tailscale", True))
        transports.append(("direct", False))

        last_error = None
        for transport_name, use_tailscale in transports:
            try:
                ssh_cmd = self._build_ssh_command(remote_cmd, use_tailscale)

                proc = await asyncio.create_subprocess_exec(
                    *ssh_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=timeout,
                    )
                    elapsed_ms = (time.time() - start_time) * 1000

                    return SSHResult(
                        success=proc.returncode == 0,
                        returncode=proc.returncode or 0,
                        stdout=stdout.decode("utf-8", errors="replace"),
                        stderr=stderr.decode("utf-8", errors="replace"),
                        elapsed_ms=elapsed_ms,
                        command=command,
                        host=self._config.host,
                        transport_used=transport_name,
                    )

                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    elapsed_ms = (time.time() - start_time) * 1000
                    return SSHResult(
                        success=False,
                        returncode=-1,
                        stdout="",
                        stderr=f"Command timed out after {timeout}s",
                        elapsed_ms=elapsed_ms,
                        command=command,
                        host=self._config.host,
                        transport_used=transport_name,
                        timed_out=True,
                    )

            except Exception as e:
                last_error = str(e)
                logger.debug(f"SSH via {transport_name} failed: {e}")
                continue

        elapsed_ms = (time.time() - start_time) * 1000
        return SSHResult(
            success=False,
            returncode=-1,
            stdout="",
            stderr=last_error or "All transports failed",
            elapsed_ms=elapsed_ms,
            command=command,
            host=self._config.host,
            error=last_error,
        )

    async def run_async_with_retry(
        self,
        command: str,
        timeout: int | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> SSHResult:
        """Execute with automatic retry on transient failures."""
        last_result = None
        for attempt in range(max_retries):
            result = await self.run_async(command, timeout)
            if result.success:
                return result
            last_result = result
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
        return last_result or SSHResult(
            success=False, returncode=-1, stdout="", stderr="No attempts made",
            elapsed_ms=0, command=command, host=self._config.host,
        )

    def run(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SSHResult:
        """Execute command synchronously."""
        timeout = timeout or self._config.command_timeout
        start_time = time.time()

        remote_cmd = command
        if cwd or self._config.work_dir:
            work_dir = cwd or self._config.work_dir
            remote_cmd = f"cd {work_dir} && {command}"
        if env:
            env_str = " ".join(f"{k}={v}" for k, v in env.items())
            remote_cmd = f"{env_str} {remote_cmd}"

        transports = []
        if self._config.tailscale_ip:
            transports.append(("tailscale", True))
        transports.append(("direct", False))

        last_error = None
        for transport_name, use_tailscale in transports:
            try:
                ssh_cmd = self._build_ssh_command(remote_cmd, use_tailscale)

                result = subprocess.run(
                    ssh_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                elapsed_ms = (time.time() - start_time) * 1000

                return SSHResult(
                    success=result.returncode == 0,
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    elapsed_ms=elapsed_ms,
                    command=command,
                    host=self._config.host,
                    transport_used=transport_name,
                )

            except subprocess.TimeoutExpired:
                elapsed_ms = (time.time() - start_time) * 1000
                return SSHResult(
                    success=False,
                    returncode=-1,
                    stdout="",
                    stderr=f"Command timed out after {timeout}s",
                    elapsed_ms=elapsed_ms,
                    command=command,
                    host=self._config.host,
                    transport_used=transport_name,
                    timed_out=True,
                )

            except Exception as e:
                last_error = str(e)
                logger.debug(f"SSH via {transport_name} failed: {e}")
                continue

        elapsed_ms = (time.time() - start_time) * 1000
        return SSHResult(
            success=False,
            returncode=-1,
            stdout="",
            stderr=last_error or "All transports failed",
            elapsed_ms=elapsed_ms,
            command=command,
            host=self._config.host,
            error=last_error,
        )

    async def check_connectivity(self) -> tuple[bool, str]:
        """Check if host is reachable via SSH."""
        result = await self.run_async("echo ok", timeout=10)
        if result.success and "ok" in result.stdout:
            return True, f"Connected via {result.transport_used}"
        return False, result.stderr or result.error or "Unknown error"

    def is_alive(self) -> bool:
        """Check if remote host is reachable (sync)."""
        result = self.run("echo ok", timeout=10)
        return result.success and "ok" in result.stdout

    def scp_from(
        self,
        remote_path: str,
        local_path: str,
        timeout: int = 300,
    ) -> SSHResult:
        """Copy file from remote host."""
        start_time = time.time()
        target_host = self._config.tailscale_ip or self._config.host
        if self._config.user:
            target = f"{self._config.user}@{target_host}"
        else:
            target = target_host

        cmd = ["scp", "-o", f"ConnectTimeout={self._config.connect_timeout}"]
        if self._config.port != 22:
            cmd.extend(["-P", str(self._config.port)])
        if self._config.key_path:
            key_path = os.path.expanduser(self._config.key_path)
            if os.path.exists(key_path):
                cmd.extend(["-i", key_path])
        cmd.extend([f"{target}:{remote_path}", local_path])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            elapsed_ms = (time.time() - start_time) * 1000
            return SSHResult(
                success=result.returncode == 0,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                elapsed_ms=elapsed_ms,
                command=f"scp {remote_path} -> {local_path}",
                host=self._config.host,
            )
        except subprocess.TimeoutExpired:
            elapsed_ms = (time.time() - start_time) * 1000
            return SSHResult(
                success=False, returncode=-1, stdout="", stderr=f"SCP timed out after {timeout}s",
                elapsed_ms=elapsed_ms, command=f"scp {remote_path}", host=self._config.host, timed_out=True,
            )

    def scp_to(
        self,
        local_path: str,
        remote_path: str,
        timeout: int = 300,
    ) -> SSHResult:
        """Copy file to remote host."""
        start_time = time.time()
        target_host = self._config.tailscale_ip or self._config.host
        if self._config.user:
            target = f"{self._config.user}@{target_host}"
        else:
            target = target_host

        cmd = ["scp", "-o", f"ConnectTimeout={self._config.connect_timeout}"]
        if self._config.port != 22:
            cmd.extend(["-P", str(self._config.port)])
        if self._config.key_path:
            key_path = os.path.expanduser(self._config.key_path)
            if os.path.exists(key_path):
                cmd.extend(["-i", key_path])
        cmd.extend([local_path, f"{target}:{remote_path}"])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            elapsed_ms = (time.time() - start_time) * 1000
            return SSHResult(
                success=result.returncode == 0,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                elapsed_ms=elapsed_ms,
                command=f"scp {local_path} -> {remote_path}",
                host=self._config.host,
            )
        except subprocess.TimeoutExpired:
            elapsed_ms = (time.time() - start_time) * 1000
            return SSHResult(
                success=False, returncode=-1, stdout="", stderr=f"SCP timed out after {timeout}s",
                elapsed_ms=elapsed_ms, command=f"scp {local_path}", host=self._config.host, timed_out=True,
            )

    async def close(self) -> None:
        """Close ControlMaster connection."""
        if self._config.use_control_master:
            control_path = self._get_control_path()
            try:
                await asyncio.create_subprocess_exec(
                    "ssh", "-O", "exit", "-o", f"ControlPath={control_path}",
                    f"{self._config.user}@{self._config.host}" if self._config.user else self._config.host,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            except Exception:
                pass


# =============================================================================
# Client Pool / Factory
# =============================================================================

_client_pool: dict[str, SSHClient] = {}


def get_ssh_client(node_id_or_host: str) -> SSHClient:
    """Get SSH client for a node, creating if needed.

    Loads config from cluster_hosts.yaml if node_id matches.
    """
    if node_id_or_host in _client_pool:
        return _client_pool[node_id_or_host]

    # Try to load from cluster config
    config = SSHConfig.from_cluster_node(node_id_or_host)
    if config is None:
        # Fallback: treat as hostname
        config = SSHConfig(host=node_id_or_host)

    client = SSHClient(config)
    _client_pool[node_id_or_host] = client
    return client


async def close_all_clients() -> None:
    """Close all SSH connections."""
    for client in _client_pool.values():
        await client.close()
    _client_pool.clear()


# =============================================================================
# Backward-Compatible Function Exports
# =============================================================================

async def run_ssh_command_async(
    host: str,
    command: str,
    user: str | None = None,
    timeout: int | None = None,
    max_retries: int = 3,
    **kwargs: Any,
) -> SSHResult:
    """Run SSH command asynchronously (compatible with executor.py)."""
    client = get_ssh_client(host)
    if user and client.config.user != user:
        # Create a new config with the specified user
        new_config = SSHConfig(
            host=client.config.host,
            port=client.config.port,
            user=user,
            key_path=client.config.key_path,
            tailscale_ip=client.config.tailscale_ip,
        )
        client = SSHClient(new_config)
    return await client.run_async_with_retry(command, timeout, max_retries)


def run_ssh_command_sync(
    host: str,
    command: str,
    user: str | None = None,
    port: int = 22,
    key_path: str | None = None,
    timeout: int = 60,
    **kwargs: Any,
) -> SSHResult:
    """Run SSH command synchronously (compatible with executor.py)."""
    config = SSHConfig(
        host=host,
        port=port,
        user=user,
        key_path=key_path,
        command_timeout=timeout,
    )
    client = SSHClient(config)
    return client.run(command, timeout)


# Default to async for backward compatibility
run_ssh_command = run_ssh_command_async
