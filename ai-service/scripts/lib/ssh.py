"""SSH command execution utilities for scripts.

Provides unified SSH command execution with common patterns:
- Configurable host, port, user
- Optional SSH key
- Timeout handling
- Retry logic
- Both sync and async versions

Usage:
    from scripts.lib.ssh import run_ssh_command, SSHConfig

    # Simple usage
    success, output = run_ssh_command("user@host", "ls -la")

    # With port and timeout
    success, output = run_ssh_command("host", "uptime", port=22022, timeout=30)

    # With full config
    config = SSHConfig(host="host", port=22022, user="root", timeout=60)
    success, output = run_ssh_command_with_config(config, "nvidia-smi")

    # With retries
    success, output = run_ssh_command("host", "command", retries=2)
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SSHConfig:
    """SSH connection configuration."""
    host: str
    port: int = 22
    user: str = "root"
    ssh_key: str | None = None
    connect_timeout: int = 10
    strict_host_key_checking: str = "accept-new"  # "no", "yes", or "accept-new"
    batch_mode: bool = True
    server_alive_interval: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SSHConfig":
        """Create config from a dictionary (e.g., from YAML config)."""
        return cls(
            host=d.get("ssh_host") or d.get("host") or d.get("tailscale_ip", ""),
            port=int(d.get("ssh_port") or d.get("port", 22)),
            user=d.get("ssh_user") or d.get("user", "root"),
            ssh_key=d.get("ssh_key"),
            connect_timeout=int(d.get("connect_timeout", 10)),
        )

    def build_ssh_args(self) -> list[str]:
        """Build SSH command arguments."""
        args = ["ssh"]

        # SSH key
        if self.ssh_key:
            key_path = os.path.expanduser(self.ssh_key)
            args.extend(["-i", key_path])

        # Options
        args.extend(["-o", f"StrictHostKeyChecking={self.strict_host_key_checking}"])
        args.extend(["-o", f"ConnectTimeout={self.connect_timeout}"])

        if self.batch_mode:
            args.extend(["-o", "BatchMode=yes"])

        if self.server_alive_interval:
            args.extend(["-o", f"ServerAliveInterval={self.server_alive_interval}"])

        # Port (if non-default)
        if self.port != 22:
            args.extend(["-p", str(self.port)])

        # User@host
        args.append(f"{self.user}@{self.host}")

        return args


@dataclass
class SSHResult:
    """Result of an SSH command execution."""
    success: bool
    output: str
    exit_code: int = 0
    error: str | None = None
    duration_seconds: float = 0.0

    def __bool__(self) -> bool:
        return self.success


def run_ssh_command(
    host: str,
    command: str,
    *,
    port: int = 22,
    user: str = "root",
    ssh_key: str | None = None,
    timeout: int = 30,
    retries: int = 0,
    retry_delay: float = 2.0,
    connect_timeout: int | None = None,
    strict_host_key_checking: str = "accept-new",
    batch_mode: bool = True,
    server_alive_interval: int | None = None,
) -> tuple[bool, str]:
    """Run SSH command on remote host.

    Args:
        host: Remote hostname or IP address
        command: Command to execute on remote host
        port: SSH port (default: 22)
        user: SSH user (default: root)
        ssh_key: Path to SSH private key (optional)
        timeout: Command timeout in seconds (default: 30)
        retries: Number of retry attempts on failure (default: 0)
        retry_delay: Delay between retries in seconds (default: 2.0)
        connect_timeout: SSH connection timeout (default: min(timeout, 10))
        strict_host_key_checking: SSH option (default: "accept-new")
        batch_mode: Enable BatchMode (default: True)
        server_alive_interval: SSH keep-alive interval (optional)

    Returns:
        Tuple of (success: bool, output: str)
    """
    if connect_timeout is None:
        connect_timeout = min(timeout, 10)

    config = SSHConfig(
        host=host,
        port=port,
        user=user,
        ssh_key=ssh_key,
        connect_timeout=connect_timeout,
        strict_host_key_checking=strict_host_key_checking,
        batch_mode=batch_mode,
        server_alive_interval=server_alive_interval,
    )

    return run_ssh_command_with_config(
        config, command,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
    )


def run_ssh_command_with_config(
    config: SSHConfig,
    command: str,
    *,
    timeout: int = 30,
    retries: int = 0,
    retry_delay: float = 2.0,
) -> tuple[bool, str]:
    """Run SSH command using an SSHConfig object.

    Args:
        config: SSH connection configuration
        command: Command to execute on remote host
        timeout: Command timeout in seconds
        retries: Number of retry attempts on failure
        retry_delay: Delay between retries in seconds

    Returns:
        Tuple of (success: bool, output: str)
    """
    ssh_args = config.build_ssh_args()
    ssh_args.append(command)

    last_error = ""
    for attempt in range(retries + 1):
        try:
            result = subprocess.run(
                ssh_args,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout.strip() or result.stderr.strip()

            if result.returncode == 0:
                return True, output

            # Command failed but SSH succeeded
            last_error = output
            if attempt < retries:
                time.sleep(retry_delay)

        except subprocess.TimeoutExpired:
            last_error = "timeout"
            if attempt < retries:
                time.sleep(retry_delay)

        except Exception as e:
            last_error = str(e)
            if attempt < retries:
                time.sleep(retry_delay)

    return False, last_error


def run_ssh_command_from_dict(
    host_config: dict[str, Any],
    command: str,
    *,
    timeout: int = 30,
    retries: int = 0,
) -> tuple[bool, str]:
    """Run SSH command using host configuration from a dictionary.

    Useful for configs loaded from YAML files.

    Args:
        host_config: Dictionary with ssh_host, ssh_port, ssh_user, ssh_key
        command: Command to execute
        timeout: Command timeout in seconds
        retries: Number of retry attempts

    Returns:
        Tuple of (success: bool, output: str)
    """
    config = SSHConfig.from_dict(host_config)
    return run_ssh_command_with_config(
        config, command,
        timeout=timeout,
        retries=retries,
    )


async def run_ssh_command_async(
    ssh_args: list[str],
    command: str,
    *,
    timeout: int = 60,
) -> tuple[bool, str]:
    """Run SSH command asynchronously.

    Args:
        ssh_args: Pre-built SSH command arguments (without the remote command)
        command: Command to execute on remote host
        timeout: Command timeout in seconds

    Returns:
        Tuple of (success: bool, output: str)
    """
    full_cmd = ssh_args + [command]
    try:
        proc = await asyncio.create_subprocess_exec(
            *full_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = stdout.decode("utf-8", errors="replace").rstrip()
        return proc.returncode == 0, output

    except asyncio.TimeoutError:
        return False, "SSH timeout"

    except Exception as e:
        return False, f"SSH error: {e}"


async def run_ssh_command_async_with_config(
    config: SSHConfig,
    command: str,
    *,
    timeout: int = 60,
) -> tuple[bool, str]:
    """Run SSH command asynchronously using an SSHConfig object.

    Args:
        config: SSH connection configuration
        command: Command to execute on remote host
        timeout: Command timeout in seconds

    Returns:
        Tuple of (success: bool, output: str)
    """
    ssh_args = config.build_ssh_args()
    return await run_ssh_command_async(ssh_args, command, timeout=timeout)


# Convenience function for Vast.ai instances
def run_vast_ssh_command(
    host: str,
    port: int,
    command: str,
    *,
    timeout: int = 30,
    retries: int = 0,
) -> tuple[bool, str]:
    """Run SSH command on a Vast.ai instance.

    Uses root user and accept-new host key checking (common for Vast).

    Args:
        host: Vast instance SSH host
        port: Vast instance SSH port
        command: Command to execute
        timeout: Command timeout in seconds
        retries: Number of retry attempts

    Returns:
        Tuple of (success: bool, output: str)
    """
    return run_ssh_command(
        host=host,
        command=command,
        port=port,
        user="root",
        timeout=timeout,
        retries=retries,
        strict_host_key_checking="accept-new",
        batch_mode=True,
    )
