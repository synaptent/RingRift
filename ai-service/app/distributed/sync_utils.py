"""Sync utilities for distributed operations.

This module provides shared utilities for file synchronization operations:
- rsync-based file transfer with SSH support
- Remote file discovery

SSH Connection:
    This module uses hosts.py:HostConfig for SSH connection details.
    The canonical SSH utilities are:
    - hosts.py:SSHExecutor for synchronous command execution
    - ssh_transport.py:SSHTransport for async P2P operations

    For rsync, we build SSH options from HostConfig to ensure consistency
    with the rest of the codebase.

Used by:
- scripts/unified_data_sync.py
- scripts/external_drive_sync_daemon.py
- Other data sync scripts
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.distributed.hosts import HostConfig

logger = logging.getLogger(__name__)


def build_ssh_command_for_rsync(host: "HostConfig") -> str:
    """Build SSH command string for use with rsync -e option.

    Uses HostConfig properties to build a consistent SSH command string.
    This is the single place where rsync SSH options are constructed.

    Args:
        host: HostConfig with SSH connection details

    Returns:
        SSH command string for rsync -e option
    """
    parts = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]

    if host.ssh_key:
        key_path = os.path.expanduser(host.ssh_key)
        parts.extend(["-i", key_path])

    if host.ssh_port != 22:
        parts.extend(["-p", str(host.ssh_port)])

    return " ".join(parts)


def rsync_file(
    host: "HostConfig",
    remote_path: str,
    local_path: Path,
    timeout: int = 120,
) -> bool:
    """Rsync a single file from a remote host.

    Args:
        host: HostConfig object with SSH connection details
        remote_path: Path to the file on the remote host
        local_path: Local path to save the file
        timeout: Timeout in seconds for the rsync operation

    Returns:
        True if rsync succeeded, False otherwise
    """
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)

        rsync_args = ["rsync", "-az", f"--timeout={timeout}"]

        # Build SSH command using centralized helper
        ssh_cmd = build_ssh_command_for_rsync(host)
        rsync_args.extend(["-e", ssh_cmd])

        # Build source path using host's ssh_target
        rsync_args.extend([f"{host.ssh_target}:{remote_path}", str(local_path)])

        result = subprocess.run(
            rsync_args,
            capture_output=True,
            timeout=timeout + 30,  # Extra buffer for rsync
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.warning(f"Rsync timed out for {host.name}:{remote_path}")
        return False
    except Exception as e:
        logger.warning(f"Rsync failed for {host.name}:{remote_path}: {e}")
        return False


def rsync_directory(
    host: "HostConfig",
    remote_dir: str,
    local_dir: Path,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    timeout: int = 300,
    delete: bool = False,
) -> bool:
    """Rsync a directory from a remote host.

    Args:
        host: HostConfig object with SSH connection details
        remote_dir: Path to the directory on the remote host
        local_dir: Local directory path
        include_patterns: File patterns to include (e.g., ["*.db"])
        exclude_patterns: File patterns to exclude (e.g., ["*.tmp"])
        timeout: Timeout in seconds for the rsync operation
        delete: Whether to delete files in local_dir not present on remote

    Returns:
        True if rsync succeeded, False otherwise
    """
    try:
        local_dir.mkdir(parents=True, exist_ok=True)

        rsync_args = ["rsync", "-az", f"--timeout={timeout}"]

        if delete:
            rsync_args.append("--delete")

        # Add include patterns
        if include_patterns:
            for pattern in include_patterns:
                rsync_args.extend(["--include", pattern])

        # Add exclude patterns
        if exclude_patterns:
            for pattern in exclude_patterns:
                rsync_args.extend(["--exclude", pattern])

        # Build SSH command using centralized helper
        ssh_cmd = build_ssh_command_for_rsync(host)
        rsync_args.extend(["-e", ssh_cmd])

        # Ensure trailing slash for directory sync
        remote_dir = remote_dir.rstrip("/") + "/"

        # Build source path using host's ssh_target
        rsync_args.extend([f"{host.ssh_target}:{remote_dir}", str(local_dir)])

        result = subprocess.run(
            rsync_args,
            capture_output=True,
            timeout=timeout + 30,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.warning(f"Rsync directory timed out for {host.name}:{remote_dir}")
        return False
    except Exception as e:
        logger.warning(f"Rsync directory failed for {host.name}:{remote_dir}: {e}")
        return False


def rsync_push_file(
    host: "HostConfig",
    local_path: Path,
    remote_path: str,
    timeout: int = 120,
) -> bool:
    """Rsync a single file to a remote host.

    Args:
        host: HostConfig object with SSH connection details
        local_path: Local file path
        remote_path: Path on the remote host
        timeout: Timeout in seconds for the rsync operation

    Returns:
        True if rsync succeeded, False otherwise
    """
    try:
        if not local_path.exists():
            logger.warning(f"Local file not found: {local_path}")
            return False

        rsync_args = ["rsync", "-az", f"--timeout={timeout}"]

        # Build SSH command using centralized helper
        ssh_cmd = build_ssh_command_for_rsync(host)
        rsync_args.extend(["-e", ssh_cmd])

        # Build destination path using host's ssh_target
        rsync_args.extend([str(local_path), f"{host.ssh_target}:{remote_path}"])

        result = subprocess.run(
            rsync_args,
            capture_output=True,
            timeout=timeout + 30,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.warning(f"Rsync push timed out for {host.name}:{remote_path}")
        return False
    except Exception as e:
        logger.warning(f"Rsync push failed for {host.name}:{remote_path}: {e}")
        return False
