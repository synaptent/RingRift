"""File transfer utilities for scripts.

.. note:: December 2025 - SSH Migration
    This module uses direct subprocess.run for SSH operations (line 1382).
    Consider migrating to the canonical SSH client for better error handling:
        from app.core.ssh import get_ssh_client, SSHClient
    See app/core/ssh.py for migration guide.

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
        base64_push,
        base64_pull,
        http_pull,      # NEW: HTTP via P2P endpoints
        robust_push,
        robust_pull,    # NEW: Multi-transport with HTTP fallback
        chunked_push,
        chunked_push_progressive,
        compute_checksum,
        verify_transfer,
    )

    # Push file to remote
    config = TransferConfig(ssh_key="~/.ssh/id_rsa")
    result = scp_push("local.db", "host", 22, "/remote/path/", config)

    # Pull with rsync (resume support)
    result = rsync_pull("host:/path/file.db", "local/", config)

    # Base64 transfer (for flaky connections with binary stream issues)
    result = base64_push("file.npz", "host", 22, "/path/file.npz", config)

    # HTTP transfer via P2P endpoints (when SSH fails completely)
    result = http_pull("100.127.112.31", "models/canonical_hex8_2p.pth", "/tmp/model.pth")

    # Robust pull with automatic fallback (rsync -> scp -> base64 -> http)
    result = robust_pull("host", 22, "models/model.pth", "local/model.pth", config)

    # Chunked transfer with progressive verification (for large files)
    result = chunked_push_progressive(
        "large_model.pth", "host", 22, "/path/model.pth",
        config, chunk_size_mb=10,
        resume_state_path="/tmp/transfer_state.json",
    )
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# WAL File Handling (December 2025)
# =============================================================================
# SQLite databases using WAL mode have companion files (.db-wal, .db-shm)
# that MUST be synced together with the main .db file to prevent data loss.


def checkpoint_database(db_path: str | Path) -> bool:
    """Force WAL checkpoint to flush pending transactions to main database.

    Call this BEFORE syncing a database to minimize WAL size and ensure
    all transactions are in the main .db file.

    Args:
        db_path: Path to the database file

    Returns:
        True if checkpoint succeeded, False otherwise
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return False

    try:
        with sqlite3.connect(str(db_path), timeout=30.0) as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            if mode.upper() == "WAL":
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            return True
    except Exception as e:
        logger.warning(f"WAL checkpoint failed for {db_path}: {e}")
        return False


def get_db_with_wal_files(db_path: str | Path) -> list[Path]:
    """Get database file along with any WAL files that should be synced.

    Args:
        db_path: Path to the .db file

    Returns:
        List of paths: [.db-wal, .db-shm, .db] (existing files only)
    """
    db_path = Path(db_path)
    files = []

    # Add WAL files first (order matters for consistency)
    wal_path = db_path.with_suffix(db_path.suffix + "-wal")
    if wal_path.exists():
        files.append(wal_path)

    shm_path = db_path.with_suffix(db_path.suffix + "-shm")
    if shm_path.exists():
        files.append(shm_path)

    # Add main database last
    if db_path.exists():
        files.append(db_path)

    return files

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
    filepath: str | Path,
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
    local_path: str | Path,
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
    local_path: str | Path,
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
    local_path: str | Path,
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

            cmd = ["rsync", "-avz", "--progress", "--partial"]  # Jan 2, 2026: Enable resume
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

            # Dec 2025: Handle database files with WAL
            source = str(local_path)
            is_db_file = local_path.is_file() and source.endswith(".db")

            if is_db_file:
                # Checkpoint WAL before sync to ensure data integrity
                checkpoint_database(source)

                # Use include patterns to sync .db and WAL files together
                db_name = local_path.name
                parent_dir = str(local_path.parent) + "/"
                cmd.extend([
                    f"--include={db_name}",
                    f"--include={db_name}-wal",
                    f"--include={db_name}-shm",
                    "--exclude=*",
                ])
                source = parent_dir
            elif local_path.is_dir() and not source.endswith("/"):
                # Ensure trailing slash for directory sync
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
    local_path: str | Path,
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

    # Dec 2025: Handle database files with WAL
    is_db_file = remote_path.endswith(".db")

    for attempt in range(config.max_retries):
        attempts = attempt + 1
        try:
            ssh_opts = " ".join(config.get_ssh_options())
            ssh_cmd = f"ssh -p {port} {ssh_opts}"

            cmd = ["rsync", "-avz", "--progress", "--partial"]  # Jan 2, 2026: Enable resume
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

            # Dec 2025: For database files, include WAL files
            if is_db_file:
                db_name = Path(remote_path).name
                remote_dir = str(Path(remote_path).parent) + "/"
                cmd.extend([
                    f"--include={db_name}",
                    f"--include={db_name}-wal",
                    f"--include={db_name}-shm",
                    "--exclude=*",
                ])
                cmd.extend([f"{config.ssh_user}@{host}:{remote_dir}", str(local_path.parent) + "/"])
            else:
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
    source: str | Path,
    destination: str | Path,
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
    source: str | Path,
    dest: str | Path | None = None,
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
    source: str | Path,
    dest: str | Path | None = None,
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
    local_path: str | Path,
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


def base64_push(
    local_path: str | Path,
    host: str,
    port: int,
    remote_path: str,
    config: TransferConfig,
) -> TransferResult:
    """Push a file using base64 encoding through SSH stdin.

    This method encodes the file as base64 and pipes it through SSH,
    which avoids binary stream handling issues that cause "Connection reset
    by peer" errors during SCP/rsync transfers.

    **When to use this method:**
    - SCP/rsync connections are resetting mid-transfer
    - Firewall or proxy is corrupting binary streams
    - Connection is unstable but short commands work
    - You've tried chunked transfer and it still fails

    **How it works:**
    1. Read local file and encode as base64 (text-safe)
    2. Pipe through SSH to remote host
    3. Remote decodes base64 back to binary
    4. Verify file size matches

    Keywords for searchability:
    - base64 transfer / base64 push / base64 pull
    - connection reset workaround
    - binary stream corruption fix
    - SSH pipe transfer
    - text-safe file transfer
    - flaky connection transfer
    - SCP alternative

    See also:
    - chunked_push() - for resumable transfers
    - app/coordination/cluster_transport.py - async version

    Args:
        local_path: Local file path
        host: Remote hostname or IP
        port: SSH port
        remote_path: Destination path on remote (file path, not directory)
        config: Transfer configuration

    Returns:
        TransferResult with operation details

    Example:
        # When SCP keeps failing with "Connection reset":
        result = base64_push(
            "data/training/hex8_4p.npz",
            "cluster-node",
            22,
            "/home/user/ringrift/ai-service/data/training/hex8_4p.npz",
            TransferConfig(ssh_key="~/.ssh/id_cluster"),
        )
    """
    import base64

    local_path = Path(local_path)
    if not local_path.exists():
        return TransferResult(
            success=False,
            error=f"Local file not found: {local_path}",
            method="base64_push",
            source=str(local_path),
            destination=f"{host}:{remote_path}",
        )

    file_size = local_path.stat().st_size
    start_time = time.time()

    # Feb 2026: Hard limit to prevent OOM - base64 reads entire file into memory
    # plus 33% overhead for encoding. Use chunked_push for large files instead.
    max_base64_size = 500 * 1024 * 1024  # 500MB
    if file_size > max_base64_size:
        return TransferResult(
            success=False,
            error=(
                f"File too large for base64_push ({file_size / 1024 / 1024:.0f}MB > "
                f"{max_base64_size / 1024 / 1024:.0f}MB limit). "
                "Use chunked_push or rsync instead to avoid OOM."
            ),
            method="base64_push",
            source=str(local_path),
            destination=f"{host}:{remote_path}",
        )

    # For large files (>100MB), warn about memory usage
    if file_size > 100 * 1024 * 1024:
        logger.warning(
            f"base64_push: Large file ({file_size / 1024 / 1024:.1f}MB) - "
            "consider chunked_push for better memory efficiency"
        )

    ssh_opts = config.get_ssh_options()
    last_error = ""
    attempts = 0

    for attempt in range(config.max_retries):
        attempts = attempt + 1
        try:
            # Read and encode file
            with open(local_path, "rb") as f:
                file_data = f.read()
            encoded_data = base64.b64encode(file_data).decode("ascii")

            # Ensure remote directory exists and decode file
            remote_dir = str(Path(remote_path).parent)
            decode_cmd = f"mkdir -p '{remote_dir}' && base64 -d > '{remote_path}'"

            ssh_cmd = [
                "ssh",
                *ssh_opts,
                "-p", str(port),
                f"{config.ssh_user}@{host}",
                decode_cmd,
            ]

            # Pipe base64 data through SSH
            result = subprocess.run(
                ssh_cmd,
                input=encoded_data,
                capture_output=True,
                text=True,
                timeout=config.transfer_timeout,
            )

            if result.returncode == 0:
                # Verify file size on remote
                verify_cmd = [
                    "ssh",
                    *ssh_opts,
                    "-p", str(port),
                    f"{config.ssh_user}@{host}",
                    f"stat -c%s '{remote_path}' 2>/dev/null || stat -f%z '{remote_path}'",
                ]
                verify_result = subprocess.run(
                    verify_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                remote_size = 0
                try:
                    remote_size = int(verify_result.stdout.strip())
                except ValueError:
                    pass

                size_match = remote_size == file_size
                if not size_match:
                    last_error = f"Size mismatch: local={file_size}, remote={remote_size}"
                    if attempt < config.max_retries - 1:
                        logger.warning(f"base64_push attempt {attempts} failed: {last_error}, retrying...")
                        time.sleep(config.retry_delay)
                        continue

                duration = time.time() - start_time
                return TransferResult(
                    success=True,
                    bytes_transferred=file_size,
                    duration_seconds=duration,
                    method="base64_push",
                    source=str(local_path),
                    destination=f"{host}:{remote_path}",
                    checksum_verified=size_match,
                    attempts=attempts,
                )

            last_error = result.stderr.strip() or f"Exit code {result.returncode}"

        except subprocess.TimeoutExpired:
            last_error = "Transfer timed out"
        except MemoryError:
            last_error = "File too large for base64 encoding in memory"
            break  # Don't retry memory errors
        except Exception as e:
            last_error = str(e)

        if attempt < config.max_retries - 1:
            logger.warning(f"base64_push attempt {attempts} failed: {last_error}, retrying...")
            time.sleep(config.retry_delay)

    return TransferResult(
        success=False,
        duration_seconds=time.time() - start_time,
        method="base64_push",
        source=str(local_path),
        destination=f"{host}:{remote_path}",
        error=last_error,
        attempts=attempts,
    )


def base64_pull(
    host: str,
    port: int,
    remote_path: str,
    local_path: str | Path,
    config: TransferConfig,
) -> TransferResult:
    """Pull a file using base64 encoding through SSH stdout.

    This method reads the remote file, encodes as base64, pipes through SSH,
    and decodes locally. Avoids binary stream issues that cause connection resets.

    Keywords for searchability:
    - base64 transfer / base64 push / base64 pull
    - connection reset workaround
    - binary stream corruption fix
    - SSH pipe transfer

    Args:
        host: Remote hostname or IP
        port: SSH port
        remote_path: Source path on remote
        local_path: Local destination path
        config: Transfer configuration

    Returns:
        TransferResult with operation details
    """
    import base64

    local_path = Path(local_path)
    start_time = time.time()
    last_error = ""
    attempts = 0

    # Ensure parent directory exists
    local_path.parent.mkdir(parents=True, exist_ok=True)

    ssh_opts = config.get_ssh_options()

    for attempt in range(config.max_retries):
        attempts = attempt + 1
        try:
            # Read file as base64 from remote
            ssh_cmd = [
                "ssh",
                *ssh_opts,
                "-p", str(port),
                f"{config.ssh_user}@{host}",
                f"base64 '{remote_path}'",
            ]

            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=config.transfer_timeout,
            )

            if result.returncode == 0:
                # Decode and write locally
                file_data = base64.b64decode(result.stdout)
                with open(local_path, "wb") as f:
                    f.write(file_data)

                duration = time.time() - start_time
                file_size = local_path.stat().st_size

                return TransferResult(
                    success=True,
                    bytes_transferred=file_size,
                    duration_seconds=duration,
                    method="base64_pull",
                    source=f"{host}:{remote_path}",
                    destination=str(local_path),
                    attempts=attempts,
                )

            last_error = result.stderr.strip() or f"Exit code {result.returncode}"

        except subprocess.TimeoutExpired:
            last_error = "Transfer timed out"
        except base64.binascii.Error as e:
            last_error = f"Base64 decode error: {e}"
        except Exception as e:
            last_error = str(e)

        if attempt < config.max_retries - 1:
            logger.warning(f"base64_pull attempt {attempts} failed: {last_error}, retrying...")
            time.sleep(config.retry_delay)

    return TransferResult(
        success=False,
        duration_seconds=time.time() - start_time,
        method="base64_pull",
        source=f"{host}:{remote_path}",
        destination=str(local_path),
        error=last_error,
        attempts=attempts,
    )


# =============================================================================
# HTTP P2P File Transfer (December 2025)
# =============================================================================
# HTTP-based file transfer using P2P orchestrator endpoints.
# Works when SSH connections are unstable but P2P is running.

def http_pull(
    host: str,
    remote_path: str,
    local_path: str | Path,
    port: int = 8770,
    timeout: int = 300,
) -> TransferResult:
    """Pull a file via HTTP from P2P orchestrator endpoints.

    Uses /files/models/ and /files/data/ endpoints on the P2P orchestrator
    to download files when SSH-based transfers fail.

    **When this is useful:**
    - SSH connections reset or timeout
    - All other transports fail
    - P2P orchestrator is running on remote node

    **Limitations:**
    - Only supports files in models/ or data/ directories
    - Requires P2P orchestrator running (port 8770)

    Keywords for searchability:
    - HTTP transfer / HTTP pull / HTTP download
    - P2P file sync / P2P download
    - connection reset workaround

    Args:
        host: Remote hostname or IP (Tailscale IP preferred)
        remote_path: Source path (e.g., "models/canonical_hex8_2p.pth")
        local_path: Local destination path
        port: P2P orchestrator port (default 8770)
        timeout: Download timeout in seconds

    Returns:
        TransferResult with operation details

    Example:
        result = http_pull(
            "100.127.112.31",  # Tailscale IP
            "models/canonical_hex8_2p.pth",
            "/tmp/model.pth",
        )
    """
    import urllib.request
    import urllib.error

    local_path = Path(local_path)
    start_time = time.time()

    # Determine endpoint from path
    if "models/" in remote_path or remote_path.endswith(".pth") or remote_path.endswith(".pt"):
        # Extract filename if path contains models/
        if "models/" in remote_path:
            file_name = remote_path.split("models/")[-1]
        else:
            file_name = remote_path
        endpoint = f"/files/models/{file_name}"
    elif "data/" in remote_path or remote_path.endswith(".db") or remote_path.endswith(".npz"):
        # Extract filename if path contains data/
        if "data/" in remote_path:
            file_name = remote_path.split("data/")[-1]
        else:
            file_name = remote_path
        endpoint = f"/files/data/{file_name}"
    else:
        return TransferResult(
            success=False,
            method="http_pull",
            source=f"http://{host}:{port}{remote_path}",
            destination=str(local_path),
            error=f"Cannot determine file type from path: {remote_path}",
        )

    url = f"http://{host}:{port}{endpoint}"

    try:
        logger.info(f"http_pull: Downloading {url} -> {local_path}")

        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Stream download
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB

            with open(local_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

        duration = time.time() - start_time
        file_size = local_path.stat().st_size

        logger.info(
            f"http_pull: Complete {file_size / 1024 / 1024:.1f} MB "
            f"in {duration:.1f}s ({file_size / duration / 1024 / 1024:.1f} MB/s)"
        )

        return TransferResult(
            success=True,
            bytes_transferred=file_size,
            duration_seconds=duration,
            method="http_pull",
            source=url,
            destination=str(local_path),
        )

    except urllib.error.HTTPError as e:
        return TransferResult(
            success=False,
            duration_seconds=time.time() - start_time,
            method="http_pull",
            source=url,
            destination=str(local_path),
            error=f"HTTP {e.code}: {e.reason}",
        )
    except urllib.error.URLError as e:
        return TransferResult(
            success=False,
            duration_seconds=time.time() - start_time,
            method="http_pull",
            source=url,
            destination=str(local_path),
            error=f"URL error: {e.reason}",
        )
    except TimeoutError:
        return TransferResult(
            success=False,
            duration_seconds=time.time() - start_time,
            method="http_pull",
            source=url,
            destination=str(local_path),
            error="HTTP transfer timed out",
        )
    except Exception as e:
        return TransferResult(
            success=False,
            duration_seconds=time.time() - start_time,
            method="http_pull",
            source=url,
            destination=str(local_path),
            error=str(e),
        )


def robust_pull(
    host: str,
    port: int,
    remote_path: str,
    local_path: str | Path,
    config: TransferConfig,
    p2p_port: int = 8770,
) -> TransferResult:
    """Pull a file using the most reliable method available.

    Tries transfer methods in order of preference, falling back automatically:
    1. rsync (fastest, supports resume)
    2. scp (simpler, widely compatible)
    3. base64 (works when binary streams fail)
    4. http (uses P2P endpoints, works when SSH fails completely)

    Args:
        host: Remote hostname or IP
        port: SSH port
        remote_path: Source path on remote
        local_path: Local destination path
        config: Transfer configuration
        p2p_port: P2P port for HTTP fallback (default 8770)

    Returns:
        TransferResult with operation details
    """
    local_path = Path(local_path)

    # Try methods in order
    methods = [
        ("rsync", lambda: rsync_pull(host, port, remote_path, local_path, config)),
        ("scp", lambda: scp_pull(host, port, remote_path, local_path, config)),
        ("base64", lambda: base64_pull(host, port, remote_path, local_path, config)),
        ("http", lambda: http_pull(host, remote_path, local_path, p2p_port)),
    ]

    for method_name, method_fn in methods:
        logger.debug(f"robust_pull: Trying {method_name}...")
        result = method_fn()
        if result.success:
            logger.info(f"robust_pull: Success with {method_name}")
            result.method = f"robust_pull/{method_name}"
            return result
        logger.debug(f"robust_pull: {method_name} failed: {result.error}")

    # All methods failed
    return TransferResult(
        success=False,
        method="robust_pull",
        source=f"{host}:{remote_path}",
        destination=str(local_path),
        error="All transfer methods failed (rsync, scp, base64, http)",
    )


def robust_push(
    local_path: str | Path,
    host: str,
    port: int,
    remote_path: str,
    config: TransferConfig,
) -> TransferResult:
    """Push a file using the most reliable method available.

    Tries transfer methods in order of preference, falling back automatically:
    1. rsync (fastest, supports resume)
    2. scp (simpler, widely compatible)
    3. base64 (works when binary streams fail)

    Keywords for searchability:
    - robust transfer / reliable transfer / failover transfer
    - connection reset recovery
    - automatic fallback
    - multi-transport transfer

    Args:
        local_path: Local file path
        host: Remote hostname or IP
        port: SSH port
        remote_path: Destination path on remote
        config: Transfer configuration

    Returns:
        TransferResult with operation details (includes method used)

    Example:
        # Automatically picks best working method:
        result = robust_push(
            "models/large_model.pth",
            "cluster-node",
            22,
            "/data/models/large_model.pth",
            TransferConfig(),
        )
        print(f"Transfer succeeded via: {result.method}")
    """
    local_path = Path(local_path)
    if not local_path.exists():
        return TransferResult(
            success=False,
            error=f"Local file not found: {local_path}",
            method="robust_push",
            source=str(local_path),
            destination=f"{host}:{remote_path}",
        )

    file_size = local_path.stat().st_size
    large_file_threshold = 500 * 1024 * 1024  # 500MB

    # Feb 2026: For large files, use chunked_push instead of base64_push
    # to avoid OOM. base64_push reads entire file into memory + 33% overhead.
    if file_size > large_file_threshold:
        methods = [
            ("rsync", lambda: rsync_push(local_path, host, port, remote_path, config)),
            ("scp", lambda: scp_push(local_path, host, port, remote_path, config)),
            ("chunked", lambda: chunked_push(local_path, host, port, remote_path, config, chunk_size_mb=config.chunk_size_mb)),
        ]
        logger.info(
            f"robust_push: Large file ({file_size / 1024 / 1024:.0f}MB), "
            "using chunked_push instead of base64 to avoid OOM"
        )
    else:
        methods = [
            ("rsync", lambda: rsync_push(local_path, host, port, remote_path, config)),
            ("scp", lambda: scp_push(local_path, host, port, remote_path, config)),
            ("base64", lambda: base64_push(local_path, host, port, remote_path, config)),
        ]

    for method_name, method_fn in methods:
        logger.debug(f"robust_push: Trying {method_name}...")
        result = method_fn()
        if result.success:
            logger.info(f"robust_push: Succeeded via {method_name}")
            return result
        logger.debug(f"robust_push: {method_name} failed: {result.error}")

    # All methods failed
    methods_tried = ", ".join(name for name, _ in methods)
    return TransferResult(
        success=False,
        error=f"All transfer methods failed ({methods_tried})",
        method="robust_push",
        source=str(local_path),
        destination=f"{host}:{remote_path}",
    )


def chunked_push(
    local_path: str | Path,
    host: str,
    port: int,
    remote_path: str,
    config: TransferConfig,
    chunk_size_mb: int = 5,
) -> TransferResult:
    """Push a large file in chunks for flaky connections.

    This function splits a large file into smaller chunks, transfers each
    chunk separately with retries, then reassembles on the remote end.
    Each chunk can succeed independently, making this robust for unstable
    connections that reset during large transfers.

    Keywords for searchability:
    - split file into chunks
    - chunked upload / chunked push
    - reassemble file on remote
    - flaky connection transfer
    - unstable network transfer
    - connection reset recovery
    - large file transfer
    - firewall-friendly transfer

    See also:
    - scripts/cluster_file_sync.py:chunked_transfer - more feature-rich version
    - app/distributed/resilient_transfer.py - BitTorrent/aria2 multi-transport

    Args:
        local_path: Local file path
        host: Remote hostname
        port: SSH port
        remote_path: Destination path on remote
        config: Transfer configuration
        chunk_size_mb: Size of each chunk in MB (default 5MB)

    Returns:
        TransferResult with operation details

    Example:
        result = chunked_push(
            "models/large_model.pth",
            "cluster-node",
            22,
            "/data/models/",
            TransferConfig(ssh_key="~/.ssh/id_rsa"),
            chunk_size_mb=5,
        )
    """
    import tempfile

    local_path = Path(local_path)
    if not local_path.exists():
        return TransferResult(
            success=False,
            error=f"Local file not found: {local_path}",
            method="chunked_push",
            source=str(local_path),
            destination=f"{host}:{remote_path}",
        )

    file_size = local_path.stat().st_size
    chunk_size = chunk_size_mb * 1024 * 1024
    start_time = time.time()

    # For small files, just use regular SCP
    if file_size <= chunk_size:
        result = scp_push(local_path, host, port, remote_path, config)
        result.method = "chunked_push (single)"
        return result

    # Calculate number of chunks
    num_chunks = (file_size + chunk_size - 1) // chunk_size
    logger.info(f"Splitting {local_path.name} ({file_size / 1024 / 1024:.1f}MB) into {num_chunks} chunks")

    # Create temp directory for chunks
    with tempfile.TemporaryDirectory(prefix="chunked_transfer_") as temp_dir:
        temp_path = Path(temp_dir)
        chunk_files = []

        # Split file into chunks
        with open(local_path, "rb") as f:
            for i in range(num_chunks):
                chunk_name = f"{local_path.stem}.chunk{i:04d}"
                chunk_path = temp_path / chunk_name
                chunk_data = f.read(chunk_size)
                with open(chunk_path, "wb") as cf:
                    cf.write(chunk_data)
                chunk_files.append(chunk_path)

        # Create manifest with checksums
        manifest = {
            "filename": local_path.name,
            "total_size": file_size,
            "num_chunks": num_chunks,
            "chunk_size": chunk_size,
            "checksum": compute_checksum(local_path),
            "chunks": [],
        }

        for i, chunk_path in enumerate(chunk_files):
            manifest["chunks"].append({
                "index": i,
                "name": chunk_path.name,
                "size": chunk_path.stat().st_size,
                "checksum": compute_checksum(chunk_path),
            })

        # Write manifest
        manifest_path = temp_path / f"{local_path.stem}.manifest"
        import json
        with open(manifest_path, "w") as mf:
            json.dump(manifest, mf, indent=2)

        # Determine remote temp directory
        remote_basename = Path(remote_path.rstrip("/")).name if not remote_path.endswith("/") else local_path.name
        remote_temp_dir = f"/tmp/chunked_{remote_basename}_{int(time.time())}"

        # Create remote temp directory
        ssh_opts = config.get_ssh_options()
        ssh_cmd = ["ssh"] + ssh_opts + ["-p", str(port), f"{config.ssh_user}@{host}", f"mkdir -p {remote_temp_dir}"]
        try:
            subprocess.run(ssh_cmd, capture_output=True, timeout=30)
        except Exception as e:
            return TransferResult(
                success=False,
                error=f"Failed to create remote temp dir: {e}",
                method="chunked_push",
                source=str(local_path),
                destination=f"{host}:{remote_path}",
            )

        # Transfer manifest first
        manifest_result = scp_push(manifest_path, host, port, f"{remote_temp_dir}/", config)
        if not manifest_result.success:
            return TransferResult(
                success=False,
                error=f"Failed to transfer manifest: {manifest_result.error}",
                method="chunked_push",
                source=str(local_path),
                destination=f"{host}:{remote_path}",
            )

        # Transfer each chunk with retries and per-chunk verification
        # December 2025: Added per-chunk verification to catch corruption early
        bytes_transferred = 0
        max_chunk_retries = 3

        for i, chunk_path in enumerate(chunk_files):
            expected_checksum = manifest["chunks"][i]["checksum"]
            chunk_verified = False

            for retry in range(max_chunk_retries):
                if retry > 0:
                    logger.info(f"Retrying chunk {i + 1}/{num_chunks} (attempt {retry + 1}/{max_chunk_retries})")

                logger.info(f"Transferring chunk {i + 1}/{num_chunks}")
                chunk_result = scp_push(chunk_path, host, port, f"{remote_temp_dir}/", config)

                if not chunk_result.success:
                    logger.warning(f"Chunk {i} transfer failed: {chunk_result.error}")
                    time.sleep(2 ** retry)  # Exponential backoff
                    continue

                # December 2025: Per-chunk verification - verify checksum immediately after transfer
                verify_cmd = [
                    "ssh"
                ] + ssh_opts + [
                    "-p", str(port),
                    f"{config.ssh_user}@{host}",
                    f"md5sum {remote_temp_dir}/{chunk_path.name} | cut -d' ' -f1"
                ]
                try:
                    verify_result = subprocess.run(
                        verify_cmd,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    remote_checksum = verify_result.stdout.strip()

                    if remote_checksum == expected_checksum:
                        chunk_verified = True
                        bytes_transferred += chunk_result.bytes_transferred
                        logger.debug(f"Chunk {i} verified successfully")
                        break
                    else:
                        logger.warning(
                            f"Chunk {i} checksum mismatch: expected {expected_checksum[:8]}..., "
                            f"got {remote_checksum[:8]}... - retrying"
                        )
                        # Delete corrupted chunk on remote before retry
                        delete_cmd = ["ssh"] + ssh_opts + [
                            "-p", str(port),
                            f"{config.ssh_user}@{host}",
                            f"rm -f {remote_temp_dir}/{chunk_path.name}"
                        ]
                        subprocess.run(delete_cmd, capture_output=True, timeout=30)
                        time.sleep(2 ** retry)  # Exponential backoff

                except subprocess.TimeoutExpired:
                    logger.warning(f"Chunk {i} verification timed out - retrying")
                    time.sleep(2 ** retry)
                except Exception as e:
                    logger.warning(f"Chunk {i} verification error: {e} - retrying")
                    time.sleep(2 ** retry)

            if not chunk_verified:
                # Clean up remote temp dir
                cleanup_cmd = ["ssh"] + ssh_opts + ["-p", str(port), f"{config.ssh_user}@{host}", f"rm -rf {remote_temp_dir}"]
                subprocess.run(cleanup_cmd, capture_output=True, timeout=30)

                return TransferResult(
                    success=False,
                    error=f"Failed to transfer and verify chunk {i} after {max_chunk_retries} attempts",
                    method="chunked_push",
                    source=str(local_path),
                    destination=f"{host}:{remote_path}",
                    bytes_transferred=bytes_transferred,
                )

        # Reassemble on remote
        final_remote_path = remote_path if not remote_path.endswith("/") else f"{remote_path}{local_path.name}"
        reassemble_script = f'''
import json
import os
import hashlib

manifest_path = "{remote_temp_dir}/{local_path.stem}.manifest"
with open(manifest_path) as f:
    manifest = json.load(f)

output_path = "{final_remote_path}"
os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

with open(output_path, "wb") as out:
    for chunk in manifest["chunks"]:
        chunk_path = "{remote_temp_dir}/" + chunk["name"]
        with open(chunk_path, "rb") as cf:
            out.write(cf.read())

# Verify checksum
hasher = hashlib.md5()
with open(output_path, "rb") as f:
    for chunk in iter(lambda: f.read(8192), b""):
        hasher.update(chunk)

if hasher.hexdigest() == manifest["checksum"]:
    print("OK")
else:
    print("CHECKSUM_MISMATCH")
    os.remove(output_path)
'''

        reassemble_cmd = ["ssh"] + ssh_opts + ["-p", str(port), f"{config.ssh_user}@{host}", f"python3 -c '{reassemble_script}'"]
        try:
            result = subprocess.run(reassemble_cmd, capture_output=True, text=True, timeout=120)
            if "OK" in result.stdout:
                # Clean up remote temp dir
                cleanup_cmd = ["ssh"] + ssh_opts + ["-p", str(port), f"{config.ssh_user}@{host}", f"rm -rf {remote_temp_dir}"]
                subprocess.run(cleanup_cmd, capture_output=True, timeout=30)

                duration = time.time() - start_time
                return TransferResult(
                    success=True,
                    bytes_transferred=file_size,
                    duration_seconds=duration,
                    method="chunked_push",
                    source=str(local_path),
                    destination=f"{host}:{remote_path}",
                    checksum_verified=True,
                )
            else:
                error = "Checksum mismatch after reassembly" if "CHECKSUM_MISMATCH" in result.stdout else result.stderr
                return TransferResult(
                    success=False,
                    error=error,
                    method="chunked_push",
                    source=str(local_path),
                    destination=f"{host}:{remote_path}",
                )
        except Exception as e:
            return TransferResult(
                success=False,
                error=f"Reassembly failed: {e}",
                method="chunked_push",
                source=str(local_path),
                destination=f"{host}:{remote_path}",
            )

    return TransferResult(
        success=False,
        error="Unknown error",
        method="chunked_push",
        source=str(local_path),
        destination=f"{host}:{remote_path}",
    )


def chunked_push_progressive(
    local_path: str | Path,
    host: str,
    port: int,
    remote_path: str,
    config: TransferConfig,
    chunk_size_mb: int = 5,
    resume_state_path: str | Path | None = None,
) -> TransferResult:
    """Push a large file with progressive chunk verification and resume support.

    Enhanced version of chunked_push that verifies each chunk immediately after
    transfer, allowing resume from the last successfully verified chunk if the
    transfer is interrupted.

    Keywords for searchability:
    - progressive verification / incremental verification
    - resumable chunked transfer / resume interrupted transfer
    - chunk-by-chunk verification
    - fault-tolerant file transfer
    - large file sync with checkpoints

    See also:
    - chunked_push() - simpler version without progressive verification
    - robust_push() - automatic method fallback

    Args:
        local_path: Local file path
        host: Remote hostname
        port: SSH port
        remote_path: Destination path on remote
        config: Transfer configuration
        chunk_size_mb: Size of each chunk in MB (default 5MB)
        resume_state_path: Optional path to store resume state (for crash recovery)

    Returns:
        TransferResult with operation details

    Example:
        # Resume-capable transfer with state persistence:
        result = chunked_push_progressive(
            "models/large_model.pth",
            "cluster-node",
            22,
            "/data/models/large_model.pth",
            TransferConfig(ssh_key="~/.ssh/id_rsa"),
            chunk_size_mb=10,
            resume_state_path="/tmp/transfer_state.json",
        )
    """
    import json
    import tempfile

    local_path = Path(local_path)
    if not local_path.exists():
        return TransferResult(
            success=False,
            error=f"Local file not found: {local_path}",
            method="chunked_push_progressive",
            source=str(local_path),
            destination=f"{host}:{remote_path}",
        )

    file_size = local_path.stat().st_size
    chunk_size = chunk_size_mb * 1024 * 1024
    start_time = time.time()

    # For small files, just use regular SCP
    if file_size <= chunk_size:
        result = scp_push(local_path, host, port, remote_path, config)
        result.method = "chunked_push_progressive (single)"
        return result

    # Calculate number of chunks
    num_chunks = (file_size + chunk_size - 1) // chunk_size
    logger.info(
        f"Progressive chunked transfer: {local_path.name} "
        f"({file_size / 1024 / 1024:.1f}MB) → {num_chunks} chunks"
    )

    # Load resume state if available
    verified_chunks: set[int] = set()
    remote_temp_dir: str | None = None

    if resume_state_path:
        resume_state_path = Path(resume_state_path)
        if resume_state_path.exists():
            try:
                with open(resume_state_path) as f:
                    state = json.load(f)
                # Validate state matches current file
                if (
                    state.get("file_checksum") == compute_checksum(local_path)
                    and state.get("file_size") == file_size
                    and state.get("host") == host
                ):
                    verified_chunks = set(state.get("verified_chunks", []))
                    remote_temp_dir = state.get("remote_temp_dir")
                    logger.info(
                        f"Resuming transfer: {len(verified_chunks)}/{num_chunks} "
                        "chunks already verified"
                    )
            except Exception as e:
                logger.warning(f"Failed to load resume state: {e}")

    ssh_opts = config.get_ssh_options()

    # Create remote temp directory if not resuming
    if not remote_temp_dir:
        remote_basename = (
            Path(remote_path.rstrip("/")).name
            if not remote_path.endswith("/")
            else local_path.name
        )
        remote_temp_dir = f"/tmp/chunked_{remote_basename}_{int(time.time())}"

        ssh_cmd = (
            ["ssh"]
            + ssh_opts
            + ["-p", str(port), f"{config.ssh_user}@{host}", f"mkdir -p {remote_temp_dir}"]
        )
        try:
            subprocess.run(ssh_cmd, capture_output=True, timeout=30, check=True)
        except Exception as e:
            return TransferResult(
                success=False,
                error=f"Failed to create remote temp dir: {e}",
                method="chunked_push_progressive",
                source=str(local_path),
                destination=f"{host}:{remote_path}",
            )

    def save_resume_state() -> None:
        """Save current progress for crash recovery."""
        if not resume_state_path:
            return
        try:
            state = {
                "file_checksum": compute_checksum(local_path),
                "file_size": file_size,
                "host": host,
                "remote_temp_dir": remote_temp_dir,
                "verified_chunks": list(verified_chunks),
                "timestamp": time.time(),
            }
            with open(resume_state_path, "w") as f:
                json.dump(state, f)
        except Exception as e:
            logger.warning(f"Failed to save resume state: {e}")

    def verify_remote_chunk(chunk_name: str, expected_checksum: str) -> bool:
        """Verify chunk checksum on remote host."""
        verify_cmd = (
            ["ssh"]
            + ssh_opts
            + [
                "-p",
                str(port),
                f"{config.ssh_user}@{host}",
                f"md5sum '{remote_temp_dir}/{chunk_name}' 2>/dev/null | cut -d' ' -f1",
            ]
        )
        try:
            result = subprocess.run(
                verify_cmd, capture_output=True, text=True, timeout=30
            )
            remote_checksum = result.stdout.strip()
            return remote_checksum == expected_checksum
        except Exception:
            return False

    # Process chunks with progressive verification
    bytes_transferred = 0
    failed_chunk: int | None = None

    with tempfile.TemporaryDirectory(prefix="chunked_progressive_") as temp_dir:
        temp_path = Path(temp_dir)

        # Create manifest with checksums
        manifest = {
            "filename": local_path.name,
            "total_size": file_size,
            "num_chunks": num_chunks,
            "chunk_size": chunk_size,
            "checksum": compute_checksum(local_path),
            "chunks": [],
        }

        with open(local_path, "rb") as f:
            for i in range(num_chunks):
                chunk_name = f"{local_path.stem}.chunk{i:04d}"
                chunk_path = temp_path / chunk_name

                # Read chunk data
                f.seek(i * chunk_size)
                chunk_data = f.read(chunk_size)
                chunk_checksum = hashlib.md5(chunk_data).hexdigest()

                manifest["chunks"].append(
                    {
                        "index": i,
                        "name": chunk_name,
                        "size": len(chunk_data),
                        "checksum": chunk_checksum,
                    }
                )

                # Skip already verified chunks
                if i in verified_chunks:
                    logger.debug(f"Chunk {i + 1}/{num_chunks} already verified, skipping")
                    bytes_transferred += len(chunk_data)
                    continue

                # Write chunk to temp file
                with open(chunk_path, "wb") as cf:
                    cf.write(chunk_data)

                # Transfer chunk
                logger.info(f"Transferring chunk {i + 1}/{num_chunks}")
                chunk_result = scp_push(
                    chunk_path, host, port, f"{remote_temp_dir}/", config
                )

                if not chunk_result.success:
                    failed_chunk = i
                    save_resume_state()
                    break

                # Progressive verification - verify immediately after transfer
                if verify_remote_chunk(chunk_name, chunk_checksum):
                    verified_chunks.add(i)
                    bytes_transferred += len(chunk_data)
                    save_resume_state()
                    logger.debug(f"Chunk {i + 1}/{num_chunks} verified ✓")
                else:
                    # Retry once on verification failure
                    logger.warning(
                        f"Chunk {i + 1}/{num_chunks} verification failed, retrying..."
                    )
                    chunk_result = scp_push(
                        chunk_path, host, port, f"{remote_temp_dir}/", config
                    )
                    if chunk_result.success and verify_remote_chunk(
                        chunk_name, chunk_checksum
                    ):
                        verified_chunks.add(i)
                        bytes_transferred += len(chunk_data)
                        save_resume_state()
                        logger.info(f"Chunk {i + 1}/{num_chunks} verified on retry ✓")
                    else:
                        failed_chunk = i
                        save_resume_state()
                        break

                # Clean up local chunk file to save disk space
                chunk_path.unlink(missing_ok=True)

        if failed_chunk is not None:
            return TransferResult(
                success=False,
                error=f"Failed at chunk {failed_chunk + 1}/{num_chunks}. "
                f"Resume with same arguments to continue.",
                method="chunked_push_progressive",
                source=str(local_path),
                destination=f"{host}:{remote_path}",
                bytes_transferred=bytes_transferred,
            )

        # All chunks verified - write manifest and reassemble
        manifest_path = temp_path / f"{local_path.stem}.manifest"
        with open(manifest_path, "w") as mf:
            json.dump(manifest, mf, indent=2)

        manifest_result = scp_push(
            manifest_path, host, port, f"{remote_temp_dir}/", config
        )
        if not manifest_result.success:
            save_resume_state()
            return TransferResult(
                success=False,
                error=f"Failed to transfer manifest: {manifest_result.error}",
                method="chunked_push_progressive",
                source=str(local_path),
                destination=f"{host}:{remote_path}",
                bytes_transferred=bytes_transferred,
            )

        # Reassemble on remote
        final_remote_path = (
            remote_path
            if not remote_path.endswith("/")
            else f"{remote_path}{local_path.name}"
        )
        reassemble_script = f'''
import json
import os
import hashlib

manifest_path = "{remote_temp_dir}/{local_path.stem}.manifest"
with open(manifest_path) as f:
    manifest = json.load(f)

output_path = "{final_remote_path}"
os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

with open(output_path, "wb") as out:
    for chunk in manifest["chunks"]:
        chunk_path = "{remote_temp_dir}/" + chunk["name"]
        with open(chunk_path, "rb") as cf:
            out.write(cf.read())

# Verify final checksum
hasher = hashlib.md5()
with open(output_path, "rb") as f:
    for block in iter(lambda: f.read(8192), b""):
        hasher.update(block)

if hasher.hexdigest() == manifest["checksum"]:
    print("OK")
else:
    print("CHECKSUM_MISMATCH")
    os.remove(output_path)
'''

        reassemble_cmd = (
            ["ssh"]
            + ssh_opts
            + [
                "-p",
                str(port),
                f"{config.ssh_user}@{host}",
                f"python3 -c '{reassemble_script}'",
            ]
        )
        try:
            result = subprocess.run(
                reassemble_cmd, capture_output=True, text=True, timeout=120
            )
            if "OK" in result.stdout:
                # Clean up remote temp dir and resume state
                cleanup_cmd = (
                    ["ssh"]
                    + ssh_opts
                    + [
                        "-p",
                        str(port),
                        f"{config.ssh_user}@{host}",
                        f"rm -rf {remote_temp_dir}",
                    ]
                )
                subprocess.run(cleanup_cmd, capture_output=True, timeout=30)

                if resume_state_path and Path(resume_state_path).exists():
                    Path(resume_state_path).unlink(missing_ok=True)

                duration = time.time() - start_time
                return TransferResult(
                    success=True,
                    bytes_transferred=file_size,
                    duration_seconds=duration,
                    method="chunked_push_progressive",
                    source=str(local_path),
                    destination=f"{host}:{remote_path}",
                    checksum_verified=True,
                )
            else:
                error = (
                    "Checksum mismatch after reassembly"
                    if "CHECKSUM_MISMATCH" in result.stdout
                    else result.stderr
                )
                return TransferResult(
                    success=False,
                    error=error,
                    method="chunked_push_progressive",
                    source=str(local_path),
                    destination=f"{host}:{remote_path}",
                )
        except Exception as e:
            return TransferResult(
                success=False,
                error=f"Reassembly failed: {e}",
                method="chunked_push_progressive",
                source=str(local_path),
                destination=f"{host}:{remote_path}",
            )
