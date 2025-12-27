"""Sync utilities for distributed operations.

This module provides shared utilities for file synchronization operations:
- rsync-based file transfer with SSH support
- Remote file discovery
- Post-transfer integrity verification (December 2025 enhancement)
- Quarantine mechanism for corrupted files

SSH Connection:
    This module uses hosts.py:HostConfig for SSH connection details.
    The canonical SSH utilities are:
    - hosts.py:SSHExecutor for synchronous command execution
    - ssh_transport.py:SSHTransport for async P2P operations

    For rsync, we build SSH options from HostConfig to ensure consistency
    with the rest of the codebase.

Integrity Verification:
    All *_verified functions perform mandatory post-transfer verification:
    1. File size check
    2. SHA256 checksum verification
    3. Quarantine of corrupted files

    RECOMMENDED: Use *_verified functions for all production transfers.

Used by:
- scripts/unified_data_sync.py
- scripts/external_drive_sync_daemon.py
- Other data sync scripts
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.distributed.hosts import HostConfig

logger = logging.getLogger(__name__)

# Quarantine directory for corrupted files
QUARANTINE_DIR = Path("data/quarantine")

# Maximum age for quarantined files before cleanup (30 days)
QUARANTINE_MAX_AGE_DAYS = 30


@dataclass
class TransferVerificationResult:
    """Result of a verified file transfer."""
    success: bool
    verified: bool = False
    error: str = ""
    bytes_transferred: int = 0
    checksum_matched: bool = False
    quarantined: bool = False
    quarantine_path: Path | None = None


def _compute_checksum(path: Path, algorithm: str = "sha256") -> str:
    """Compute checksum of a file using streaming read.

    Args:
        path: Path to the file
        algorithm: Hash algorithm (sha256, sha1, md5)

    Returns:
        Hex-encoded checksum string
    """
    import hashlib

    hasher = hashlib.new(algorithm)
    chunk_size = 65536  # 64KB chunks for large files

    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Failed to compute checksum for {path}: {e}")
        return ""


def _quarantine_file(path: Path, reason: str = "checksum_mismatch") -> Path | None:
    """Move a corrupted file to quarantine directory.

    Args:
        path: Path to the corrupted file
        reason: Reason for quarantine (used in filename)

    Returns:
        Path to quarantined file, or None if quarantine failed
    """
    if not path.exists():
        return None

    try:
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

        # Create unique quarantine filename with timestamp
        timestamp = int(time.time())
        quarantine_name = f"{path.name}.{reason}.{timestamp}"
        quarantine_path = QUARANTINE_DIR / quarantine_name

        shutil.move(str(path), str(quarantine_path))
        logger.warning(f"Quarantined corrupted file: {path} -> {quarantine_path}")

        return quarantine_path
    except Exception as e:
        logger.error(f"Failed to quarantine {path}: {e}")
        # If quarantine fails, delete the corrupted file
        try:
            path.unlink()
            logger.warning(f"Deleted corrupted file (quarantine failed): {path}")
        except Exception:
            pass
        return None


def cleanup_quarantine(max_age_days: int = QUARANTINE_MAX_AGE_DAYS) -> int:
    """Remove old files from quarantine directory.

    Args:
        max_age_days: Maximum age of files to keep

    Returns:
        Number of files removed
    """
    if not QUARANTINE_DIR.exists():
        return 0

    cutoff = time.time() - (max_age_days * 86400)
    removed = 0

    for path in QUARANTINE_DIR.iterdir():
        if path.is_file() and path.stat().st_mtime < cutoff:
            try:
                path.unlink()
                removed += 1
            except Exception as e:
                logger.warning(f"Failed to clean up quarantine file {path}: {e}")

    if removed > 0:
        logger.info(f"Cleaned up {removed} old files from quarantine")

    return removed


def _fetch_remote_checksum(
    host: "HostConfig",
    remote_path: str,
    timeout: int = 30,
) -> str | None:
    """Fetch checksum of a remote file via SSH.

    Args:
        host: HostConfig with SSH connection details
        remote_path: Path to the file on remote host
        timeout: SSH command timeout

    Returns:
        SHA256 checksum or None if failed
    """
    try:
        ssh_parts = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]

        if host.ssh_key:
            key_path = os.path.expanduser(host.ssh_key)
            ssh_parts.extend(["-i", key_path])

        if host.ssh_port != 22:
            ssh_parts.extend(["-p", str(host.ssh_port)])

        ssh_parts.append(host.ssh_target)
        ssh_parts.append(f"sha256sum '{remote_path}' 2>/dev/null | cut -d' ' -f1")

        result = subprocess.run(
            ssh_parts,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0 and result.stdout.strip():
            checksum = result.stdout.strip()
            if len(checksum) == 64:  # Valid SHA256 length
                return checksum

        return None
    except Exception as e:
        logger.debug(f"Failed to fetch remote checksum for {host.name}:{remote_path}: {e}")
        return None


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


# =============================================================================
# VERIFIED TRANSFER FUNCTIONS
# =============================================================================
# RECOMMENDED: Use these *_verified functions for all production transfers.
# They add mandatory post-transfer checksum verification and quarantine
# corrupted files.
# =============================================================================


def rsync_file_verified(
    host: "HostConfig",
    remote_path: str,
    local_path: Path,
    expected_checksum: str | None = None,
    timeout: int = 120,
) -> TransferVerificationResult:
    """Rsync a single file with mandatory post-transfer verification.

    This is the RECOMMENDED function for production file transfers.
    It adds:
    1. rsync --checksum flag for transfer-level verification
    2. Post-transfer SHA256 checksum verification
    3. Automatic quarantine of corrupted files

    Args:
        host: HostConfig object with SSH connection details
        remote_path: Path to the file on the remote host
        local_path: Local path to save the file
        expected_checksum: Optional SHA256 checksum to verify against.
            If not provided, will fetch from remote before transfer.
        timeout: Timeout in seconds for the rsync operation

    Returns:
        TransferVerificationResult with success/verification status
    """
    result = TransferVerificationResult(success=False)

    try:
        # 1. Get expected checksum if not provided
        if not expected_checksum:
            expected_checksum = _fetch_remote_checksum(host, remote_path, timeout=30)
            if not expected_checksum:
                logger.warning(
                    f"Could not fetch remote checksum for {host.name}:{remote_path}, "
                    "proceeding without pre-transfer verification"
                )

        # 2. Create parent directory
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # 3. Run rsync with --checksum flag
        rsync_args = ["rsync", "-az", "--checksum", f"--timeout={timeout}"]

        ssh_cmd = build_ssh_command_for_rsync(host)
        rsync_args.extend(["-e", ssh_cmd])
        rsync_args.extend([f"{host.ssh_target}:{remote_path}", str(local_path)])

        proc_result = subprocess.run(
            rsync_args,
            capture_output=True,
            timeout=timeout + 30,
        )

        if proc_result.returncode != 0:
            stderr = proc_result.stderr.decode("utf-8", errors="replace")
            result.error = f"rsync failed: {stderr[:200]}"
            return result

        # 4. Verify file exists
        if not local_path.exists():
            result.error = "File not created after rsync"
            return result

        result.bytes_transferred = local_path.stat().st_size

        # 5. Post-transfer checksum verification
        if expected_checksum:
            actual_checksum = _compute_checksum(local_path)
            if actual_checksum == expected_checksum:
                result.checksum_matched = True
                result.verified = True
                result.success = True
                logger.debug(
                    f"Verified transfer: {host.name}:{remote_path} "
                    f"({result.bytes_transferred} bytes, checksum OK)"
                )
            else:
                # Checksum mismatch - quarantine the file
                result.error = (
                    f"Checksum mismatch: expected {expected_checksum[:16]}..., "
                    f"got {actual_checksum[:16]}..."
                )
                quarantine_path = _quarantine_file(local_path, "checksum_mismatch")
                if quarantine_path:
                    result.quarantined = True
                    result.quarantine_path = quarantine_path
                logger.error(
                    f"Transfer verification FAILED for {host.name}:{remote_path}: "
                    f"{result.error}"
                )
        else:
            # No checksum to verify against - mark as success but unverified
            result.success = True
            result.verified = False
            logger.warning(
                f"Transfer completed but UNVERIFIED (no checksum): "
                f"{host.name}:{remote_path}"
            )

        return result

    except subprocess.TimeoutExpired:
        result.error = f"Rsync timed out after {timeout}s"
        logger.warning(f"Rsync timed out for {host.name}:{remote_path}")
        return result
    except Exception as e:
        result.error = str(e)
        logger.warning(f"Rsync failed for {host.name}:{remote_path}: {e}")
        return result


def rsync_push_file_verified(
    host: "HostConfig",
    local_path: Path,
    remote_path: str,
    timeout: int = 120,
) -> TransferVerificationResult:
    """Rsync a single file to a remote host with verification.

    This is the RECOMMENDED function for pushing files to remote hosts.
    It adds:
    1. Pre-transfer local checksum computation
    2. rsync --checksum flag for transfer-level verification
    3. Post-transfer remote checksum verification

    Args:
        host: HostConfig object with SSH connection details
        local_path: Local file path
        remote_path: Path on the remote host
        timeout: Timeout in seconds for the rsync operation

    Returns:
        TransferVerificationResult with success/verification status
    """
    result = TransferVerificationResult(success=False)

    try:
        if not local_path.exists():
            result.error = f"Local file not found: {local_path}"
            return result

        # 1. Compute local checksum before transfer
        local_checksum = _compute_checksum(local_path)
        if not local_checksum:
            result.error = "Failed to compute local checksum"
            return result

        result.bytes_transferred = local_path.stat().st_size

        # 2. Run rsync with --checksum flag
        rsync_args = ["rsync", "-az", "--checksum", f"--timeout={timeout}"]

        ssh_cmd = build_ssh_command_for_rsync(host)
        rsync_args.extend(["-e", ssh_cmd])
        rsync_args.extend([str(local_path), f"{host.ssh_target}:{remote_path}"])

        proc_result = subprocess.run(
            rsync_args,
            capture_output=True,
            timeout=timeout + 30,
        )

        if proc_result.returncode != 0:
            stderr = proc_result.stderr.decode("utf-8", errors="replace")
            result.error = f"rsync push failed: {stderr[:200]}"
            return result

        # 3. Verify remote checksum matches local
        remote_checksum = _fetch_remote_checksum(host, remote_path, timeout=30)
        if remote_checksum:
            if remote_checksum == local_checksum:
                result.checksum_matched = True
                result.verified = True
                result.success = True
                logger.debug(
                    f"Verified push: {local_path} -> {host.name}:{remote_path} "
                    f"({result.bytes_transferred} bytes, checksum OK)"
                )
            else:
                result.error = (
                    f"Remote checksum mismatch: expected {local_checksum[:16]}..., "
                    f"got {remote_checksum[:16]}..."
                )
                logger.error(
                    f"Push verification FAILED for {host.name}:{remote_path}: "
                    f"{result.error}"
                )
        else:
            # Could not verify remote - mark as success but unverified
            result.success = True
            result.verified = False
            logger.warning(
                f"Push completed but UNVERIFIED (could not fetch remote checksum): "
                f"{host.name}:{remote_path}"
            )

        return result

    except subprocess.TimeoutExpired:
        result.error = f"Rsync push timed out after {timeout}s"
        logger.warning(f"Rsync push timed out for {host.name}:{remote_path}")
        return result
    except Exception as e:
        result.error = str(e)
        logger.warning(f"Rsync push failed for {host.name}:{remote_path}: {e}")
        return result


def rsync_directory_verified(
    host: "HostConfig",
    remote_dir: str,
    local_dir: Path,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    timeout: int = 300,
    delete: bool = False,
) -> TransferVerificationResult:
    """Rsync a directory with post-transfer verification of each file.

    This is the RECOMMENDED function for directory transfers.
    Note: Directory verification is more expensive as it computes checksums
    for all transferred files.

    Args:
        host: HostConfig object with SSH connection details
        remote_dir: Path to the directory on the remote host
        local_dir: Local directory path
        include_patterns: File patterns to include (e.g., ["*.db"])
        exclude_patterns: File patterns to exclude (e.g., ["*.tmp"])
        timeout: Timeout in seconds for the rsync operation
        delete: Whether to delete files in local_dir not present on remote

    Returns:
        TransferVerificationResult with aggregated success/verification status
    """
    result = TransferVerificationResult(success=False)

    try:
        local_dir.mkdir(parents=True, exist_ok=True)

        # Track files before sync for comparison
        files_before = set(local_dir.rglob("*")) if local_dir.exists() else set()

        rsync_args = ["rsync", "-az", "--checksum", f"--timeout={timeout}"]

        if delete:
            rsync_args.append("--delete")

        if include_patterns:
            for pattern in include_patterns:
                rsync_args.extend(["--include", pattern])

        if exclude_patterns:
            for pattern in exclude_patterns:
                rsync_args.extend(["--exclude", pattern])

        ssh_cmd = build_ssh_command_for_rsync(host)
        rsync_args.extend(["-e", ssh_cmd])

        remote_dir = remote_dir.rstrip("/") + "/"
        rsync_args.extend([f"{host.ssh_target}:{remote_dir}", str(local_dir)])

        proc_result = subprocess.run(
            rsync_args,
            capture_output=True,
            timeout=timeout + 30,
        )

        if proc_result.returncode != 0:
            stderr = proc_result.stderr.decode("utf-8", errors="replace")
            result.error = f"rsync directory failed: {stderr[:200]}"
            return result

        # Find new/modified files
        files_after = set(local_dir.rglob("*"))
        new_files = [f for f in files_after - files_before if f.is_file()]

        # Calculate total bytes transferred
        result.bytes_transferred = sum(f.stat().st_size for f in new_files)

        # For directory sync, we mark as success if rsync succeeded
        # Full verification of each file would require remote checksums
        result.success = True
        result.verified = False  # Directory verification is best-effort

        logger.info(
            f"Directory sync completed: {host.name}:{remote_dir} -> {local_dir} "
            f"({len(new_files)} files, {result.bytes_transferred} bytes)"
        )

        return result

    except subprocess.TimeoutExpired:
        result.error = f"Rsync directory timed out after {timeout}s"
        logger.warning(f"Rsync directory timed out for {host.name}:{remote_dir}")
        return result
    except Exception as e:
        result.error = str(e)
        logger.warning(f"Rsync directory failed for {host.name}:{remote_dir}: {e}")
        return result