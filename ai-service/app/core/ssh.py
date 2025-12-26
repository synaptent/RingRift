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
    "SSHHealth",
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

    # Cloudflare Zero Trust support
    cloudflare_tunnel: str | None = None
    cloudflare_service_token_id: str | None = None
    cloudflare_service_token_secret: str | None = None

    @property
    def use_cloudflare(self) -> bool:
        """Check if Cloudflare Zero Trust tunnel is configured."""
        return bool(self.cloudflare_tunnel)

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
                cloudflare_tunnel=node.get("cloudflare_tunnel"),
                cloudflare_service_token_id=node.get("cloudflare_service_token_id"),
                cloudflare_service_token_secret=node.get("cloudflare_service_token_secret"),
            )
        except Exception as e:
            logger.debug(f"Failed to load SSH config for {node_id}: {e}")
            return None

    @classmethod
    def from_host_config(cls, host_config: Any) -> SSHConfig:
        """Create SSHConfig from a HostConfig object.

        Args:
            host_config: HostConfig instance from distributed_hosts.yaml

        Returns:
            SSHConfig instance
        """
        return cls(
            host=getattr(host_config, "ssh_host", getattr(host_config, "tailscale_ip", "")),
            port=getattr(host_config, "ssh_port", 22),
            user=getattr(host_config, "ssh_user", "ubuntu"),
            key_path=getattr(host_config, "ssh_key", None),
            tailscale_ip=getattr(host_config, "tailscale_ip", None),
            work_dir=getattr(host_config, "ringrift_path", "~/ringrift/ai-service"),
            cloudflare_tunnel=getattr(host_config, "cloudflare_tunnel", None),
            cloudflare_service_token_id=getattr(host_config, "cloudflare_service_token_id", None),
            cloudflare_service_token_secret=getattr(host_config, "cloudflare_service_token_secret", None),
        )


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


@dataclass
class SSHHealth:
    """SSH connection health tracking."""
    last_success: float | None = None
    last_failure: float | None = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_successes: int = 0
    total_failures: int = 0

    @property
    def is_healthy(self) -> bool:
        """Check if connection is considered healthy."""
        # Healthy if we have recent success and < 3 consecutive failures
        return self.consecutive_failures < 3

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        total = self.total_successes + self.total_failures
        if total == 0:
            return 0.0
        return self.total_successes / total


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
        self._health = SSHHealth()

    @property
    def config(self) -> SSHConfig:
        return self._config

    @property
    def health(self) -> SSHHealth:
        """Get connection health status."""
        return self._health

    def _record_success(self) -> None:
        """Record successful command execution."""
        self._health.last_success = time.time()
        self._health.consecutive_successes += 1
        self._health.consecutive_failures = 0
        self._health.total_successes += 1

    def _record_failure(self) -> None:
        """Record failed command execution."""
        self._health.last_failure = time.time()
        self._health.consecutive_failures += 1
        self._health.consecutive_successes = 0
        self._health.total_failures += 1

    def _get_control_path(self) -> str:
        """Get ControlMaster socket path for this host."""
        safe_host = self._config.host.replace(".", "_").replace(":", "_")
        return str(self._control_path_dir / f"{safe_host}_{self._config.port}")

    def _build_cloudflare_proxy_command(self) -> str:
        """Build cloudflared ProxyCommand for Cloudflare Zero Trust tunnel.

        Returns:
            ProxyCommand string for SSH config (-o ProxyCommand=...)
        """
        if not self._config.cloudflare_tunnel:
            raise ValueError("Cloudflare tunnel not configured")

        cmd_parts = ["cloudflared", "access", "ssh", "--hostname", self._config.cloudflare_tunnel]

        # Add service token authentication if configured
        if self._config.cloudflare_service_token_id and self._config.cloudflare_service_token_secret:
            cmd_parts.extend([
                "--service-token-id", self._config.cloudflare_service_token_id,
                "--service-token-secret", self._config.cloudflare_service_token_secret,
            ])

        return " ".join(cmd_parts)

    def _build_ssh_command(
        self,
        remote_command: str,
        use_tailscale: bool = False,
        use_cloudflare: bool = False,
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

        # Cloudflare Zero Trust tunnel via ProxyCommand
        if use_cloudflare:
            proxy_cmd = self._build_cloudflare_proxy_command()
            cmd.extend(["-o", f"ProxyCommand={proxy_cmd}"])

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

        # Transport fallback chain: Tailscale -> Direct -> Cloudflare
        transports = []
        if self._config.tailscale_ip:
            transports.append(("tailscale", True, False))
        transports.append(("direct", False, False))
        if self._config.use_cloudflare:
            transports.append(("cloudflare", False, True))

        last_error = None
        for transport_name, use_tailscale, use_cloudflare in transports:
            try:
                ssh_cmd = self._build_ssh_command(remote_cmd, use_tailscale, use_cloudflare)

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

                    result = SSHResult(
                        success=proc.returncode == 0,
                        returncode=proc.returncode or 0,
                        stdout=stdout.decode("utf-8", errors="replace"),
                        stderr=stderr.decode("utf-8", errors="replace"),
                        elapsed_ms=elapsed_ms,
                        command=command,
                        host=self._config.host,
                        transport_used=transport_name,
                    )

                    # Record health status
                    if result.success:
                        self._record_success()
                    else:
                        self._record_failure()

                    return result

                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    elapsed_ms = (time.time() - start_time) * 1000
                    self._record_failure()
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
        self._record_failure()
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

        # Transport fallback chain: Tailscale -> Direct -> Cloudflare
        transports = []
        if self._config.tailscale_ip:
            transports.append(("tailscale", True, False))
        transports.append(("direct", False, False))
        if self._config.use_cloudflare:
            transports.append(("cloudflare", False, True))

        last_error = None
        for transport_name, use_tailscale, use_cloudflare in transports:
            try:
                ssh_cmd = self._build_ssh_command(remote_cmd, use_tailscale, use_cloudflare)

                result = subprocess.run(
                    ssh_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                elapsed_ms = (time.time() - start_time) * 1000

                ssh_result = SSHResult(
                    success=result.returncode == 0,
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    elapsed_ms=elapsed_ms,
                    command=command,
                    host=self._config.host,
                    transport_used=transport_name,
                )

                # Record health status
                if ssh_result.success:
                    self._record_success()
                else:
                    self._record_failure()

                return ssh_result

            except subprocess.TimeoutExpired:
                elapsed_ms = (time.time() - start_time) * 1000
                self._record_failure()
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
        self._record_failure()
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

    async def run_background(
        self,
        command: str,
        log_file: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SSHResult:
        """Execute command in background using nohup.

        Args:
            command: Command to execute
            log_file: Optional log file path (defaults to nohup.out)
            cwd: Working directory
            env: Environment variables

        Returns:
            SSHResult with PID in stdout if successful
        """
        # Build nohup command
        log_redirect = f"> {log_file} 2>&1" if log_file else "> /dev/null 2>&1"
        bg_command = f"nohup {command} {log_redirect} & echo $!"

        result = await self.run_async(bg_command, cwd=cwd, env=env)

        if result.success:
            # Extract PID from output
            pid = result.stdout.strip()
            logger.debug(f"Started background process on {self._config.host}: PID={pid}")

        return result

    async def get_process_memory(self, pid: int) -> tuple[int, str] | None:
        """Get memory usage of a process in MB.

        Args:
            pid: Process ID to check

        Returns:
            Tuple of (memory_mb, command_name) or None if process not found
        """
        # Use ps to get RSS (resident set size) in KB and command name
        cmd = f"ps -p {pid} -o rss=,comm= 2>/dev/null || echo ''"
        result = await self.run_async(cmd)

        if not result.success or not result.stdout.strip():
            return None

        try:
            parts = result.stdout.strip().split(None, 1)
            if len(parts) < 2:
                return None

            rss_kb = int(parts[0])
            cmd_name = parts[1]
            memory_mb = rss_kb // 1024

            return memory_mb, cmd_name
        except (ValueError, IndexError):
            return None

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
