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

    # For very large files (>100MB), warn about memory usage
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

    # Try methods in order
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
    return TransferResult(
        success=False,
        error="All transfer methods failed (rsync, scp, base64)",
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

        # Transfer each chunk with retries
        bytes_transferred = 0
        for i, chunk_path in enumerate(chunk_files):
            logger.info(f"Transferring chunk {i + 1}/{num_chunks}")
            chunk_result = scp_push(chunk_path, host, port, f"{remote_temp_dir}/", config)
            if not chunk_result.success:
                # Clean up remote temp dir
                cleanup_cmd = ["ssh"] + ssh_opts + ["-p", str(port), f"{config.ssh_user}@{host}", f"rm -rf {remote_temp_dir}"]
                subprocess.run(cleanup_cmd, capture_output=True, timeout=30)

                return TransferResult(
                    success=False,
                    error=f"Failed to transfer chunk {i}: {chunk_result.error}",
                    method="chunked_push",
                    source=str(local_path),
                    destination=f"{host}:{remote_path}",
                    bytes_transferred=bytes_transferred,
                )
            bytes_transferred += chunk_result.bytes_transferred

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
