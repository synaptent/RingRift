"""Unified rsync command construction for all sync strategies.

December 2025: Extracted from sync_push_mixin.py, sync_pull_mixin.py, sync_ephemeral_mixin.py
to consolidate duplicated rsync command building logic.

This module provides:
- RsyncCommandBuilder: Unified rsync command construction
- TimeoutCalculator: Unified timeout calculation by strategy
- SSHOptions: Wrapper around centralized SSH config (coordination_defaults)
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from app.config.coordination_defaults import (
    build_ssh_options as centralized_build_ssh_options,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SSHOptions:
    """SSH connection options for rsync.

    Dec 30, 2025: Now wraps the centralized SSH config from coordination_defaults.py
    to ensure consistent timeouts across all sync operations.
    """

    key_path: str
    connect_timeout: int = 10  # Default, may be overridden by provider
    strict_host_key_checking: bool = False
    tcp_keepalive: bool = True
    provider: str = "default"  # For provider-aware timeouts

    def to_string(self) -> str:
        """Build SSH options string for rsync -e flag.

        Uses centralized SSH config for consistent provider-aware timeouts.
        """
        # Use centralized config for consistency
        return centralized_build_ssh_options(
            key_path=self.key_path,
            provider=self.provider,
            include_keepalive=self.tcp_keepalive,
        )


class TimeoutCalculator:
    """Unified timeout calculation for sync operations.

    Different sync strategies have different timeout requirements:
    - standard (push): 2s/MB, min 120s, max 1800s
    - ephemeral: 1.5s/MB, min 60s, max 900s (faster termination)
    - pull: 2.5s/MB, min 120s, max 2400s (larger databases)
    """

    STRATEGIES = {
        "standard": {"per_mb": 2.0, "min": 120, "max": 1800, "base": 60},
        "ephemeral": {"per_mb": 1.5, "min": 60, "max": 900, "base": 40},
        "pull": {"per_mb": 2.5, "min": 120, "max": 2400, "base": 60},
    }

    @classmethod
    def calculate(
        cls,
        file_size_mb: float,
        strategy: Literal["standard", "ephemeral", "pull"] = "standard",
    ) -> int:
        """Calculate appropriate timeout based on file size and strategy.

        Args:
            file_size_mb: Size of file(s) being transferred in megabytes.
            strategy: One of 'standard', 'ephemeral', or 'pull'.

        Returns:
            Timeout in seconds.
        """
        params = cls.STRATEGIES.get(strategy, cls.STRATEGIES["standard"])
        timeout = int(params["base"] + file_size_mb * params["per_mb"])
        return max(params["min"], min(params["max"], timeout))

    @classmethod
    def from_path(
        cls,
        path: Path,
        strategy: Literal["standard", "ephemeral", "pull"] = "standard",
        default_mb: float = 100.0,
    ) -> int:
        """Calculate timeout from file path.

        Args:
            path: Path to file or directory.
            strategy: Sync strategy.
            default_mb: Default size if path doesn't exist.

        Returns:
            Timeout in seconds.
        """
        try:
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
            else:
                size_mb = default_mb
        except OSError:
            size_mb = default_mb
        return cls.calculate(size_mb, strategy)


@dataclass
class RsyncCommand:
    """Rsync command with execution helpers."""

    args: list[str]
    timeout: int
    description: str = ""

    async def execute_async(self) -> tuple[int, str, str]:
        """Execute rsync command asynchronously.

        Returns:
            Tuple of (return_code, stdout, stderr).
        """
        proc = await asyncio.create_subprocess_exec(
            *self.args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=self.timeout,
        )
        return (
            proc.returncode or 0,
            stdout_bytes.decode() if stdout_bytes else "",
            stderr_bytes.decode() if stderr_bytes else "",
        )

    def execute_sync(self) -> tuple[int, str, str]:
        """Execute rsync command synchronously.

        Returns:
            Tuple of (return_code, stdout, stderr).
        """
        result = subprocess.run(
            self.args,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        return result.returncode, result.stdout, result.stderr


class RsyncCommandBuilder:
    """Unified rsync command construction for all sync strategies.

    This builder consolidates the rsync command patterns from:
    - sync_push_mixin.py:366-377 (broadcast_sync_to_target)
    - sync_pull_mixin.py:320-327 (rsync_pull)
    - sync_ephemeral_mixin.py:816-826 (_direct_rsync)
    """

    @classmethod
    def build_for_push(
        cls,
        source: Path,
        target_path: str,
        ssh_options: SSHOptions,
        bandwidth_kbps: int = 50000,
        use_checksum: bool = True,
        use_progress: bool = True,
    ) -> RsyncCommand:
        """Build rsync command for broadcast/push strategy.

        This is the safest mode: uses --delay-updates for atomic updates
        and --checksum for verification.

        Args:
            source: Local source path.
            target_path: Remote target (user@host:path format).
            ssh_options: SSH connection options.
            bandwidth_kbps: Bandwidth limit in KB/s.
            use_checksum: Enable checksum verification.
            use_progress: Show transfer progress.

        Returns:
            RsyncCommand ready for execution.
        """
        args = [
            "rsync",
            "-avz",
            f"--bwlimit={bandwidth_kbps}",
            "--timeout=60",
            "--delay-updates",
        ]
        if use_progress:
            args.append("--progress")
        if use_checksum:
            args.append("--checksum")
        args.extend(["-e", ssh_options.to_string()])
        args.extend([str(source), target_path])

        timeout = TimeoutCalculator.from_path(source, strategy="standard")
        return RsyncCommand(
            args=args,
            timeout=timeout,
            description=f"Push {source.name} to {target_path}",
        )

    @classmethod
    def build_for_pull(
        cls,
        remote_full: str,
        local_path: Path,
        ssh_options: SSHOptions,
        timeout: int | None = None,
    ) -> RsyncCommand:
        """Build rsync command for pull strategy.

        Args:
            remote_full: Remote source (user@host:path format).
            local_path: Local destination path.
            ssh_options: SSH connection options.
            timeout: Override default timeout.

        Returns:
            RsyncCommand ready for execution.
        """
        args = [
            "rsync",
            "-az",
            "--timeout=60",
            "-e", ssh_options.to_string(),
            remote_full,
            str(local_path),
        ]

        final_timeout = timeout or TimeoutCalculator.calculate(100, strategy="pull")
        return RsyncCommand(
            args=args,
            timeout=final_timeout,
            description=f"Pull from {remote_full}",
        )

    @classmethod
    def build_for_ephemeral(
        cls,
        source_dir: str,
        target_path: str,
        db_name: str,
        ssh_options: SSHOptions,
        bandwidth_kbps: int | None = None,
        include_wal: bool = True,
    ) -> RsyncCommand:
        """Build rsync command for ephemeral host sync.

        Ephemeral mode syncs specific database files including WAL files
        for data durability on termination-prone hosts.

        Args:
            source_dir: Source directory (with trailing /).
            target_path: Remote target (user@host:path format).
            db_name: Database filename to sync.
            ssh_options: SSH connection options.
            bandwidth_kbps: Optional bandwidth limit in KB/s.
            include_wal: Include WAL files (.db-wal, .db-shm).

        Returns:
            RsyncCommand ready for execution.
        """
        args = [
            "rsync",
            "-avz",
            "--compress",
        ]

        if bandwidth_kbps:
            args.append(f"--bwlimit={bandwidth_kbps}")

        if include_wal:
            args.extend(cls._get_wal_include_args(db_name))
        args.append("--exclude=*")  # Exclude other files

        args.extend(["-e", ssh_options.to_string()])
        args.extend([source_dir, target_path])

        timeout = TimeoutCalculator.calculate(100, strategy="ephemeral")
        return RsyncCommand(
            args=args,
            timeout=timeout,
            description=f"Ephemeral sync {db_name} to {target_path}",
        )

    @staticmethod
    def _get_wal_include_args(db_name: str) -> list[str]:
        """Get rsync include arguments for database and WAL files.

        Args:
            db_name: Database filename.

        Returns:
            List of rsync arguments to include db and WAL files.
        """
        return [
            f"--include={db_name}",
            f"--include={db_name}-wal",
            f"--include={db_name}-shm",
        ]


def build_ssh_options(
    key_path: str,
    connect_timeout: int = 10,
    strict_host_key_checking: bool = False,
    provider: str = "default",
) -> SSHOptions:
    """Convenience function to create SSHOptions.

    Dec 30, 2025: Added provider parameter for provider-aware timeouts.

    Args:
        key_path: Path to SSH private key.
        connect_timeout: Connection timeout in seconds (may be overridden by provider).
        strict_host_key_checking: Enable strict host key checking.
        provider: Provider name for timeout lookup (vast, nebius, runpod, etc.).

    Returns:
        SSHOptions instance that uses centralized SSH config.
    """
    return SSHOptions(
        key_path=key_path,
        connect_timeout=connect_timeout,
        strict_host_key_checking=strict_host_key_checking,
        provider=provider,
    )


def get_timeout(
    file_size_mb: float,
    strategy: Literal["standard", "ephemeral", "pull"] = "standard",
) -> int:
    """Convenience function for timeout calculation.

    Args:
        file_size_mb: File size in megabytes.
        strategy: Sync strategy.

    Returns:
        Timeout in seconds.
    """
    return TimeoutCalculator.calculate(file_size_mb, strategy)
