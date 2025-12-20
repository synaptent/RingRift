"""File transfer utilities for scripts.

Provides robust file transfer operations:
- SCP with retries and timeouts
- Rsync with resume support
- Checksum verification
- Compression support
- Progress tracking

Usage:
    from scripts.lib.transfer import (
        TransferConfig,
        scp_push,
        scp_pull,
        rsync_push,
        rsync_pull,
        compute_checksum,
        verify_transfer,
    )

    # Push file to remote
    config = TransferConfig(ssh_key="~/.ssh/id_rsa")
    result = scp_push("local.db", "host", 22, "/remote/path/", config)

    # Pull with rsync (resume support)
    result = rsync_pull("host:/path/file.db", "local/", config)
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Default SSH options for robust transfers
DEFAULT_SSH_OPTIONS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "BatchMode=yes",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=3",
]


@dataclass
class TransferConfig:
    """Configuration for file transfers.

    Attributes:
        ssh_key: Path to SSH private key
        ssh_user: SSH username (default: root)
        ssh_port: SSH port (default: 22)
        connect_timeout: Connection timeout in seconds
        transfer_timeout: Overall transfer timeout in seconds
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries in seconds
        compress: Use compression during transfer
        verify_checksum: Verify file integrity after transfer
        chunk_size_mb: Size of chunks for chunked transfer
        bandwidth_limit: Bandwidth limit in KB/s (0 = unlimited)
    """
    ssh_key: str | None = None
    ssh_user: str = "root"
    ssh_port: int = 22
    connect_timeout: int = 30
    transfer_timeout: int = 300
    max_retries: int = 3
    retry_delay: float = 2.0
    compress: bool = True
    verify_checksum: bool = True
    chunk_size_mb: int = 10
    bandwidth_limit: int = 0

    def get_ssh_options(self) -> list[str]:
        """Get SSH command options."""
        opts = DEFAULT_SSH_OPTIONS.copy()
        opts.extend(["-o", f"ConnectTimeout={self.connect_timeout}"])
        if self.ssh_key:
            key_path = os.path.expanduser(self.ssh_key)
            if os.path.exists(key_path):
                opts.extend(["-i", key_path])
        return opts


@dataclass
class TransferResult:
    """Result of a file transfer operation.

    Attributes:
        success: Whether transfer completed successfully
        bytes_transferred: Number of bytes transferred
        duration_seconds: Transfer duration
        method: Transfer method used (scp/rsync)
        source: Source path
        destination: Destination path
        checksum_verified: Whether checksum was verified
        error: Error message if failed
        attempts: Number of attempts made
    """
    success: bool
    bytes_transferred: int = 0
    duration_seconds: float = 0.0
    method: str = ""
    source: str = ""
    destination: str = ""
    checksum_verified: bool = False
    error: str = ""
    attempts: int = 1

    def __bool__(self) -> bool:
        return self.success

    @property
    def speed_mbps(self) -> float:
        """Calculate transfer speed in MB/s."""
        if self.duration_seconds > 0 and self.bytes_transferred > 0:
            return (self.bytes_transferred / (1024 * 1024)) / self.duration_seconds
        return 0.0


def compute_checksum(
    filepath: Union[str, Path],
    algorithm: str = "md5",
    chunk_size: int = 8192,
) -> str:
    """Compute checksum of a file.

    Args:
        filepath: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)
        chunk_size: Read chunk size

    Returns:
        Hexadecimal checksum string
    """
    algorithms = {
        "md5": hashlib.md5,
        "sha1": hashlib.sha1,
        "sha256": hashlib.sha256,
    }

    if algorithm not in algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    hasher = algorithms[algorithm]()
    filepath = Path(filepath)

    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def get_remote_checksum(
    host: str,
    remote_path: str,
    config: TransferConfig,
    algorithm: str = "md5",
) -> str | None:
    """Get checksum of a remote file.

    Args:
        host: Remote hostname
        remote_path: Path to remote file
        config: Transfer configuration
        algorithm: Hash algorithm

    Returns:
        Checksum string or None if failed
    """
    commands = {
        "md5": "md5sum",
        "sha1": "sha1sum",
        "sha256": "sha256sum",
    }

    if algorithm not in commands:
        return None

    cmd = commands[algorithm]
    ssh_opts = config.get_ssh_options()

    ssh_cmd = [
        "ssh",
        *ssh_opts,
        "-p", str(config.ssh_port),
        f"{config.ssh_user}@{host}",
        f"{cmd} {remote_path} 2>/dev/null | cut -d' ' -f1",
    ]

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        logger.debug(f"Failed to get remote checksum: {e}")

    return None


def scp_push(
    local_path: Union[str, Path],
    host: str,
    port: int,
    remote_path: str,
    config: TransferConfig,
    progress_callback: Callable[[int, int], None] | None = None,
) -> TransferResult:
    """Push a file to a remote host using SCP.

    Args:
        local_path: Local file path
        host: Remote hostname
        port: SSH port
        remote_path: Destination path on remote
        config: Transfer configuration
        progress_callback: Optional callback(bytes_sent, total_bytes)

    Returns:
        TransferResult with operation details
    """
    local_path = Path(local_path)
    if not local_path.exists():
        return TransferResult(
            success=False,
            error=f"Local file not found: {local_path}",
            method="scp",
            source=str(local_path),
            destination=f"{host}:{remote_path}",
        )

    file_size = local_path.stat().st_size
    start_time = time.time()
    last_error = ""
    attempts = 0

    for attempt in range(config.max_retries):
        attempts = attempt + 1
        try:
            ssh_opts = config.get_ssh_options()

            cmd = ["scp"]
            if config.compress:
                cmd.append("-C")
            if config.bandwidth_limit > 0:
                cmd.extend(["-l", str(config.bandwidth_limit)])

            cmd.extend(ssh_opts)
            cmd.extend(["-P", str(port)])
            cmd.extend([str(local_path), f"{config.ssh_user}@{host}:{remote_path}"])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.transfer_timeout,
            )

            if result.returncode == 0:
                duration = time.time() - start_time

                # Verify checksum if configured
                checksum_ok = False
                if config.verify_checksum:
                    local_checksum = compute_checksum(local_path)
                    remote_full_path = remote_path
                    if remote_path.endswith("/"):
                        remote_full_path = f"{remote_path}{local_path.name}"
                    remote_checksum = get_remote_checksum(
                        host, remote_full_path, config
                    )
                    checksum_ok = local_checksum == remote_checksum

                return TransferResult(
                    success=True,
                    bytes_transferred=file_size,
                    duration_seconds=duration,
                    method="scp",
                    source=str(local_path),
                    destination=f"{host}:{remote_path}",
                    checksum_verified=checksum_ok,
                    attempts=attempts,
                )

            last_error = result.stderr.strip() or f"Exit code {result.returncode}"

        except subprocess.TimeoutExpired:
            last_error = "Transfer timed out"
        except Exception as e:
            last_error = str(e)

        if attempt < config.max_retries - 1:
            logger.warning(f"SCP attempt {attempts} failed: {last_error}, retrying...")
            time.sleep(config.retry_delay)

    return TransferResult(
        success=False,
        bytes_transferred=0,
        duration_seconds=time.time() - start_time,
        method="scp",
        source=str(local_path),
        destination=f"{host}:{remote_path}",
        error=last_error,
        attempts=attempts,
    )


def scp_pull(
    host: str,
    port: int,
    remote_path: str,
    local_path: Union[str, Path],
    config: TransferConfig,
) -> TransferResult:
    """Pull a file from a remote host using SCP.

    Args:
        host: Remote hostname
        port: SSH port
        remote_path: Source path on remote
        local_path: Local destination path
        config: Transfer configuration

    Returns:
        TransferResult with operation details
    """
    local_path = Path(local_path)
    start_time = time.time()
    last_error = ""
    attempts = 0

    # Ensure parent directory exists
    local_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(config.max_retries):
        attempts = attempt + 1
        try:
            ssh_opts = config.get_ssh_options()

            cmd = ["scp"]
            if config.compress:
                cmd.append("-C")
            if config.bandwidth_limit > 0:
                cmd.extend(["-l", str(config.bandwidth_limit)])

            cmd.extend(ssh_opts)
            cmd.extend(["-P", str(port)])
            cmd.extend([f"{config.ssh_user}@{host}:{remote_path}", str(local_path)])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.transfer_timeout,
            )

            if result.returncode == 0 and local_path.exists():
                duration = time.time() - start_time
                file_size = local_path.stat().st_size

                # Verify checksum if configured
                checksum_ok = False
                if config.verify_checksum:
                    local_checksum = compute_checksum(local_path)
                    remote_checksum = get_remote_checksum(
                        host, remote_path, config
                    )
                    checksum_ok = local_checksum == remote_checksum

                return TransferResult(
                    success=True,
                    bytes_transferred=file_size,
                    duration_seconds=duration,
                    method="scp",
                    source=f"{host}:{remote_path}",
                    destination=str(local_path),
                    checksum_verified=checksum_ok,
                    attempts=attempts,
                )

            last_error = result.stderr.strip() or f"Exit code {result.returncode}"

        except subprocess.TimeoutExpired:
            last_error = "Transfer timed out"
        except Exception as e:
            last_error = str(e)

        if attempt < config.max_retries - 1:
            logger.warning(f"SCP attempt {attempts} failed: {last_error}, retrying...")
            time.sleep(config.retry_delay)

    return TransferResult(
        success=False,
        duration_seconds=time.time() - start_time,
        method="scp",
        source=f"{host}:{remote_path}",
        destination=str(local_path),
        error=last_error,
        attempts=attempts,
    )


def rsync_push(
    local_path: Union[str, Path],
    host: str,
    port: int,
    remote_path: str,
    config: TransferConfig,
    delete: bool = False,
    exclude: list[str] | None = None,
) -> TransferResult:
    """Push files to remote using rsync.

    Args:
        local_path: Local file or directory path
        host: Remote hostname
        port: SSH port
        remote_path: Destination path on remote
        config: Transfer configuration
        delete: Delete files on remote not in source
        exclude: Patterns to exclude

    Returns:
        TransferResult with operation details
    """
    local_path = Path(local_path)
    if not local_path.exists():
        return TransferResult(
            success=False,
            error=f"Local path not found: {local_path}",
            method="rsync",
            source=str(local_path),
            destination=f"{host}:{remote_path}",
        )

    start_time = time.time()
    last_error = ""
    attempts = 0

    for attempt in range(config.max_retries):
        attempts = attempt + 1
        try:
            ssh_opts = " ".join(config.get_ssh_options())
            ssh_cmd = f"ssh -p {port} {ssh_opts}"

            cmd = ["rsync", "-avz", "--progress"]
            cmd.extend(["-e", ssh_cmd])

            if config.compress:
                cmd.append("-z")
            if config.bandwidth_limit > 0:
                cmd.extend(["--bwlimit", str(config.bandwidth_limit)])
            if delete:
                cmd.append("--delete")

            if exclude:
                for pattern in exclude:
                    cmd.extend(["--exclude", pattern])

            # Ensure trailing slash for directory sync
            source = str(local_path)
            if local_path.is_dir() and not source.endswith("/"):
                source += "/"

            cmd.extend([source, f"{config.ssh_user}@{host}:{remote_path}"])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.transfer_timeout,
            )

            if result.returncode == 0:
                duration = time.time() - start_time

                # Parse bytes transferred from rsync output
                bytes_transferred = 0
                for line in result.stdout.split("\n"):
                    if "bytes" in line.lower() and "sent" in line.lower():
                        try:
                            # Parse "sent X,XXX bytes" format
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part.lower() == "sent" and i + 1 < len(parts):
                                    bytes_transferred = int(parts[i + 1].replace(",", ""))
                                    break
                        except (ValueError, IndexError):
                            pass

                return TransferResult(
                    success=True,
                    bytes_transferred=bytes_transferred,
                    duration_seconds=duration,
                    method="rsync",
                    source=str(local_path),
                    destination=f"{host}:{remote_path}",
                    attempts=attempts,
                )

            last_error = result.stderr.strip() or f"Exit code {result.returncode}"

        except subprocess.TimeoutExpired:
            last_error = "Transfer timed out"
        except Exception as e:
            last_error = str(e)

        if attempt < config.max_retries - 1:
            logger.warning(f"Rsync attempt {attempts} failed: {last_error}, retrying...")
            time.sleep(config.retry_delay)

    return TransferResult(
        success=False,
        duration_seconds=time.time() - start_time,
        method="rsync",
        source=str(local_path),
        destination=f"{host}:{remote_path}",
        error=last_error,
        attempts=attempts,
    )


def rsync_pull(
    host: str,
    port: int,
    remote_path: str,
    local_path: Union[str, Path],
    config: TransferConfig,
    delete: bool = False,
    exclude: list[str] | None = None,
) -> TransferResult:
    """Pull files from remote using rsync.

    Args:
        host: Remote hostname
        port: SSH port
        remote_path: Source path on remote
        local_path: Local destination path
        config: Transfer configuration
        delete: Delete local files not on remote
        exclude: Patterns to exclude

    Returns:
        TransferResult with operation details
    """
    local_path = Path(local_path)
    start_time = time.time()
    last_error = ""
    attempts = 0

    # Ensure parent directory exists
    local_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(config.max_retries):
        attempts = attempt + 1
        try:
            ssh_opts = " ".join(config.get_ssh_options())
            ssh_cmd = f"ssh -p {port} {ssh_opts}"

            cmd = ["rsync", "-avz", "--progress"]
            cmd.extend(["-e", ssh_cmd])

            if config.compress:
                cmd.append("-z")
            if config.bandwidth_limit > 0:
                cmd.extend(["--bwlimit", str(config.bandwidth_limit)])
            if delete:
                cmd.append("--delete")

            if exclude:
                for pattern in exclude:
                    cmd.extend(["--exclude", pattern])

            cmd.extend([f"{config.ssh_user}@{host}:{remote_path}", str(local_path)])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.transfer_timeout,
            )

            if result.returncode == 0:
                duration = time.time() - start_time

                # Calculate bytes transferred
                bytes_transferred = 0
                if local_path.is_file():
                    bytes_transferred = local_path.stat().st_size
                elif local_path.is_dir():
                    for f in local_path.rglob("*"):
                        if f.is_file():
                            bytes_transferred += f.stat().st_size

                return TransferResult(
                    success=True,
                    bytes_transferred=bytes_transferred,
                    duration_seconds=duration,
                    method="rsync",
                    source=f"{host}:{remote_path}",
                    destination=str(local_path),
                    attempts=attempts,
                )

            last_error = result.stderr.strip() or f"Exit code {result.returncode}"

        except subprocess.TimeoutExpired:
            last_error = "Transfer timed out"
        except Exception as e:
            last_error = str(e)

        if attempt < config.max_retries - 1:
            logger.warning(f"Rsync attempt {attempts} failed: {last_error}, retrying...")
            time.sleep(config.retry_delay)

    return TransferResult(
        success=False,
        duration_seconds=time.time() - start_time,
        method="rsync",
        source=f"{host}:{remote_path}",
        destination=str(local_path),
        error=last_error,
        attempts=attempts,
    )


def copy_local(
    source: Union[str, Path],
    destination: Union[str, Path],
    verify_checksum: bool = True,
) -> TransferResult:
    """Copy a local file with optional checksum verification.

    Args:
        source: Source file path
        destination: Destination path
        verify_checksum: Verify integrity after copy

    Returns:
        TransferResult with operation details
    """
    source = Path(source)
    destination = Path(destination)
    start_time = time.time()

    if not source.exists():
        return TransferResult(
            success=False,
            error=f"Source not found: {source}",
            method="local_copy",
            source=str(source),
            destination=str(destination),
        )

    try:
        # Ensure parent directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Compute source checksum if verifying
        source_checksum = None
        if verify_checksum:
            source_checksum = compute_checksum(source)

        # Copy file
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)

        duration = time.time() - start_time

        # Get size
        if destination.is_file():
            bytes_transferred = destination.stat().st_size
        else:
            bytes_transferred = sum(
                f.stat().st_size for f in destination.rglob("*") if f.is_file()
            )

        # Verify checksum
        checksum_ok = False
        if verify_checksum and source_checksum and destination.is_file():
            dest_checksum = compute_checksum(destination)
            checksum_ok = source_checksum == dest_checksum

        return TransferResult(
            success=True,
            bytes_transferred=bytes_transferred,
            duration_seconds=duration,
            method="local_copy",
            source=str(source),
            destination=str(destination),
            checksum_verified=checksum_ok,
        )

    except Exception as e:
        return TransferResult(
            success=False,
            duration_seconds=time.time() - start_time,
            method="local_copy",
            source=str(source),
            destination=str(destination),
            error=str(e),
        )


def compress_file(
    source: Union[str, Path],
    dest: Union[str, Path] | None = None,
    level: int = 6,
) -> tuple[Path, int]:
    """Compress a file using gzip.

    Args:
        source: Source file path
        dest: Destination path (default: source + .gz)
        level: Compression level (1-9)

    Returns:
        Tuple of (compressed file path, compressed size)
    """
    source = Path(source)
    dest = Path(dest) if dest else source.with_suffix(source.suffix + ".gz")

    with open(source, "rb") as f_in, gzip.open(dest, "wb", compresslevel=level) as f_out:
        shutil.copyfileobj(f_in, f_out)

    return dest, dest.stat().st_size


def decompress_file(
    source: Union[str, Path],
    dest: Union[str, Path] | None = None,
) -> tuple[Path, int]:
    """Decompress a gzipped file.

    Args:
        source: Source .gz file path
        dest: Destination path (default: source without .gz)

    Returns:
        Tuple of (decompressed file path, decompressed size)
    """
    source = Path(source)
    if dest is None:
        if source.suffix == ".gz":
            dest = source.with_suffix("")
        else:
            dest = source.with_suffix(source.suffix + ".decompressed")
    dest = Path(dest)

    with gzip.open(source, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return dest, dest.stat().st_size


def verify_transfer(
    local_path: Union[str, Path],
    host: str,
    remote_path: str,
    config: TransferConfig,
    algorithm: str = "md5",
) -> bool:
    """Verify that local and remote files match.

    Args:
        local_path: Local file path
        host: Remote hostname
        remote_path: Remote file path
        config: Transfer configuration
        algorithm: Hash algorithm to use

    Returns:
        True if checksums match
    """
    local_path = Path(local_path)
    if not local_path.exists():
        return False

    local_checksum = compute_checksum(local_path, algorithm)
    remote_checksum = get_remote_checksum(host, remote_path, config, algorithm)

    if remote_checksum is None:
        return False

    return local_checksum == remote_checksum
