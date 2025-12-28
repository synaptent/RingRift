"""Disk Space Reservation System for Sync Operations.

This module provides atomic disk space reservation to prevent race conditions
where multiple sync operations check available space simultaneously, each finding
enough space, but then collectively filling the disk.

December 2025: Created to fix disk capacity race condition in sync operations.

The Problem
-----------
1. Sync A checks disk: 50GB free, needs 10GB -> proceeds
2. Sync B checks disk: 50GB free, needs 10GB -> proceeds
3. Sync C checks disk: 50GB free, needs 40GB -> proceeds
4. All three syncs run concurrently: 10+10+40 = 60GB needed, but only 50GB free
5. Disk fills up, transfers corrupt or fail

The Solution
------------
Each sync operation creates a reservation file with its estimated size.
When checking available space, we subtract all active reservations:
    actual_available = disk_free - sum(active_reservations)

Features:
- File-based reservations in /tmp/ringrift_sync_reserve_*
- Automatic cleanup of stale reservations (>1 hour old)
- 10% safety margin by default
- Context manager for easy use

Usage:
    from app.coordination.disk_space_reservation import (
        DiskSpaceReservation,
        disk_space_reservation,
        cleanup_stale_reservations,
    )

    # Context manager (recommended)
    async with disk_space_reservation(target_dir, estimated_bytes) as reserved:
        if reserved:
            # Safe to proceed with sync
            await do_sync()
        else:
            # Insufficient space
            logger.warning("Not enough disk space")

    # Or manual usage
    reservation = DiskSpaceReservation(target_dir, estimated_bytes)
    if reservation.acquire():
        try:
            await do_sync()
        finally:
            reservation.release()

    # Cleanup stale reservations at daemon startup
    cleaned = cleanup_stale_reservations()
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Constants
DEFAULT_RESERVATION_DIR = Path("/tmp/ringrift_sync_reserve")
DEFAULT_SAFETY_MARGIN = 0.10  # 10% extra buffer
STALE_RESERVATION_THRESHOLD_SECONDS = 3600  # 1 hour
RESERVATION_FILE_PREFIX = "reserve_"

# Import centralized defaults if available
try:
    from app.config.coordination_defaults import SyncDefaults
    DEFAULT_SAFETY_MARGIN = getattr(SyncDefaults, 'DISK_SAFETY_MARGIN', 0.10)
except ImportError:
    pass


class DiskSpaceError(Exception):
    """Raised when insufficient disk space is available."""

    def __init__(
        self,
        message: str,
        available_bytes: int = 0,
        required_bytes: int = 0,
        existing_reservations: int = 0,
    ):
        super().__init__(message)
        self.available_bytes = available_bytes
        self.required_bytes = required_bytes
        self.existing_reservations = existing_reservations


@dataclass
class ReservationInfo:
    """Information about a disk space reservation."""

    target_dir: str
    estimated_bytes: int
    created_at: float
    pid: int
    hostname: str
    reservation_file: str

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def is_stale(self) -> bool:
        return self.age_seconds > STALE_RESERVATION_THRESHOLD_SECONDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_dir": self.target_dir,
            "estimated_bytes": self.estimated_bytes,
            "created_at": self.created_at,
            "age_seconds": round(self.age_seconds, 1),
            "pid": self.pid,
            "hostname": self.hostname,
            "is_stale": self.is_stale,
        }


class DiskSpaceReservation:
    """Context manager for atomic disk space reservation.

    Creates a reservation file to "claim" disk space before starting a transfer.
    Other sync operations will see this reservation and account for it when
    checking available space.

    Args:
        target_dir: Directory where files will be written
        estimated_bytes: Estimated size of the transfer in bytes
        safety_margin: Extra buffer percentage (default: 10%)
        reservation_dir: Directory to store reservation files (default: /tmp/ringrift_sync_reserve)
    """

    def __init__(
        self,
        target_dir: str | Path,
        estimated_bytes: int,
        safety_margin: float = DEFAULT_SAFETY_MARGIN,
        reservation_dir: Path | None = None,
    ):
        self.target_dir = Path(target_dir)
        self.estimated_bytes = estimated_bytes
        self.safety_margin = safety_margin
        self.reservation_dir = reservation_dir or DEFAULT_RESERVATION_DIR

        # Add safety margin to reservation
        self.reserved_bytes = int(estimated_bytes * (1 + safety_margin))

        # Create unique reservation file name based on target path hash
        path_hash = hashlib.md5(str(self.target_dir).encode()).hexdigest()[:12]
        self.reservation_file = self.reservation_dir / f"{RESERVATION_FILE_PREFIX}{path_hash}_{os.getpid()}"

        self._acquired = False

    def _ensure_reservation_dir(self) -> None:
        """Create reservation directory if it doesn't exist."""
        self.reservation_dir.mkdir(parents=True, exist_ok=True)

    def _get_existing_reservations(self) -> int:
        """Sum up all active (non-stale) reservation sizes.

        Returns:
            Total bytes reserved by other processes
        """
        total = 0
        current_time = time.time()

        if not self.reservation_dir.exists():
            return 0

        for path in self.reservation_dir.glob(f"{RESERVATION_FILE_PREFIX}*"):
            try:
                # Skip our own reservation
                if path == self.reservation_file:
                    continue

                with open(path) as f:
                    data = json.load(f)

                created_at = data.get("created_at", 0)
                reserved_bytes = data.get("reserved_bytes", 0)

                # Skip stale reservations
                if current_time - created_at > STALE_RESERVATION_THRESHOLD_SECONDS:
                    continue

                # Only count reservations for the same target directory
                # This allows parallel syncs to different disks
                if data.get("target_dir") == str(self.target_dir):
                    total += reserved_bytes

            except (json.JSONDecodeError, OSError, KeyError) as e:
                logger.debug(f"[DiskSpaceReservation] Error reading {path}: {e}")
                continue

        return total

    def _get_available_space(self) -> int:
        """Get actual available space on the target filesystem.

        Returns:
            Available bytes on the target filesystem
        """
        try:
            # Use the target dir's parent if it doesn't exist yet
            check_dir = self.target_dir
            while not check_dir.exists() and check_dir.parent != check_dir:
                check_dir = check_dir.parent

            usage = shutil.disk_usage(check_dir)
            return usage.free
        except OSError as e:
            logger.warning(f"[DiskSpaceReservation] Could not check disk usage: {e}")
            # Return 0 to be safe - this will prevent the reservation
            return 0

    def _create_reservation_file(self) -> bool:
        """Create the reservation file.

        Returns:
            True if reservation file was created successfully
        """
        import socket

        try:
            self._ensure_reservation_dir()

            data = {
                "target_dir": str(self.target_dir),
                "estimated_bytes": self.estimated_bytes,
                "reserved_bytes": self.reserved_bytes,
                "created_at": time.time(),
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
            }

            # Write atomically by writing to temp file then renaming
            temp_file = self.reservation_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(data, f)

            temp_file.rename(self.reservation_file)
            return True

        except OSError as e:
            logger.error(f"[DiskSpaceReservation] Failed to create reservation: {e}")
            return False

    def _remove_reservation_file(self) -> bool:
        """Remove the reservation file.

        Returns:
            True if file was removed (or didn't exist)
        """
        try:
            if self.reservation_file.exists():
                self.reservation_file.unlink()
            return True
        except OSError as e:
            logger.warning(f"[DiskSpaceReservation] Failed to remove reservation: {e}")
            return False

    def acquire(self) -> bool:
        """Acquire a disk space reservation.

        Checks if there's enough space (accounting for existing reservations)
        and creates a reservation file if so.

        Returns:
            True if reservation was acquired, False if insufficient space

        Raises:
            DiskSpaceError: If raise_on_failure is True and there's insufficient space
        """
        if self._acquired:
            return True

        # Get current disk state
        available = self._get_available_space()
        existing = self._get_existing_reservations()
        effective_available = available - existing

        # Check if we have enough space
        if effective_available < self.reserved_bytes:
            logger.warning(
                f"[DiskSpaceReservation] Insufficient space for {self.target_dir}: "
                f"available={available / (1024**3):.2f}GB, "
                f"reserved_by_others={existing / (1024**3):.2f}GB, "
                f"effective_available={effective_available / (1024**3):.2f}GB, "
                f"needed={self.reserved_bytes / (1024**3):.2f}GB"
            )
            return False

        # Create reservation file
        if self._create_reservation_file():
            self._acquired = True
            logger.debug(
                f"[DiskSpaceReservation] Reserved {self.reserved_bytes / (1024**3):.2f}GB "
                f"for {self.target_dir}"
            )
            return True

        return False

    def release(self) -> None:
        """Release the disk space reservation."""
        if self._acquired:
            self._remove_reservation_file()
            self._acquired = False
            logger.debug(f"[DiskSpaceReservation] Released reservation for {self.target_dir}")

    def __enter__(self) -> "DiskSpaceReservation":
        """Context manager entry - acquire reservation."""
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - release reservation."""
        self.release()

    @property
    def is_acquired(self) -> bool:
        """Check if reservation is currently held."""
        return self._acquired


def cleanup_stale_reservations(
    reservation_dir: Path | None = None,
    max_age_seconds: float = STALE_RESERVATION_THRESHOLD_SECONDS,
) -> int:
    """Clean up stale reservation files.

    Should be called at daemon startup to clean up reservations from
    crashed processes.

    Args:
        reservation_dir: Directory containing reservation files
        max_age_seconds: Consider files older than this stale (default: 1 hour)

    Returns:
        Number of stale reservations cleaned up
    """
    reservation_dir = reservation_dir or DEFAULT_RESERVATION_DIR
    cleaned = 0
    current_time = time.time()

    if not reservation_dir.exists():
        return 0

    for path in reservation_dir.glob(f"{RESERVATION_FILE_PREFIX}*"):
        try:
            with open(path) as f:
                data = json.load(f)

            created_at = data.get("created_at", 0)
            age = current_time - created_at

            if age > max_age_seconds:
                path.unlink()
                cleaned += 1
                logger.info(
                    f"[DiskSpaceReservation] Cleaned stale reservation: {path.name} "
                    f"(age={age / 3600:.1f}h, pid={data.get('pid')}, "
                    f"host={data.get('hostname')})"
                )

        except (json.JSONDecodeError, OSError) as e:
            # If we can't read it, it's probably corrupt - remove it
            try:
                path.unlink()
                cleaned += 1
                logger.info(f"[DiskSpaceReservation] Removed corrupt reservation: {path.name}")
            except OSError:
                pass

    return cleaned


def get_active_reservations(
    reservation_dir: Path | None = None,
) -> list[ReservationInfo]:
    """Get all active (non-stale) reservations.

    Args:
        reservation_dir: Directory containing reservation files

    Returns:
        List of active reservation info
    """
    reservation_dir = reservation_dir or DEFAULT_RESERVATION_DIR
    reservations = []
    current_time = time.time()

    if not reservation_dir.exists():
        return []

    for path in reservation_dir.glob(f"{RESERVATION_FILE_PREFIX}*"):
        try:
            with open(path) as f:
                data = json.load(f)

            created_at = data.get("created_at", 0)

            # Skip stale reservations
            if current_time - created_at > STALE_RESERVATION_THRESHOLD_SECONDS:
                continue

            reservations.append(ReservationInfo(
                target_dir=data.get("target_dir", ""),
                estimated_bytes=data.get("estimated_bytes", 0),
                created_at=created_at,
                pid=data.get("pid", 0),
                hostname=data.get("hostname", ""),
                reservation_file=str(path),
            ))

        except (json.JSONDecodeError, OSError):
            continue

    return reservations


def get_effective_available_space(
    target_dir: str | Path,
    reservation_dir: Path | None = None,
) -> int:
    """Get available space accounting for existing reservations.

    This is the space actually available for a new sync operation.

    Args:
        target_dir: Directory to check
        reservation_dir: Directory containing reservation files

    Returns:
        Effective available bytes (disk free minus active reservations)
    """
    target_path = Path(target_dir)
    reservation_dir = reservation_dir or DEFAULT_RESERVATION_DIR

    # Get actual disk free space
    try:
        check_dir = target_path
        while not check_dir.exists() and check_dir.parent != check_dir:
            check_dir = check_dir.parent
        disk_free = shutil.disk_usage(check_dir).free
    except OSError:
        return 0

    # Sum up reservations for this target directory
    reserved = 0
    for info in get_active_reservations(reservation_dir):
        if info.target_dir == str(target_path):
            reserved += info.estimated_bytes

    return max(0, disk_free - reserved)


def get_total_reserved_bytes(
    target_dir: str | Path | None = None,
    reservation_dir: Path | None = None,
) -> int:
    """Get total bytes reserved across all active reservations.

    Args:
        target_dir: If provided, only count reservations for this target
        reservation_dir: Directory containing reservation files

    Returns:
        Total reserved bytes
    """
    total = 0
    target_str = str(target_dir) if target_dir else None

    for info in get_active_reservations(reservation_dir):
        if target_str is None or info.target_dir == target_str:
            total += info.estimated_bytes

    return total


@contextmanager
def disk_space_reservation_sync(
    target_dir: str | Path,
    estimated_bytes: int,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    raise_on_failure: bool = False,
):
    """Synchronous context manager for disk space reservation.

    Args:
        target_dir: Directory where files will be written
        estimated_bytes: Estimated size of the transfer in bytes
        safety_margin: Extra buffer percentage (default: 10%)
        raise_on_failure: If True, raise DiskSpaceError on insufficient space

    Yields:
        True if reservation was acquired, False otherwise

    Raises:
        DiskSpaceError: If raise_on_failure is True and there's insufficient space
    """
    reservation = DiskSpaceReservation(
        target_dir=target_dir,
        estimated_bytes=estimated_bytes,
        safety_margin=safety_margin,
    )

    acquired = reservation.acquire()

    if not acquired and raise_on_failure:
        available = reservation._get_available_space()
        existing = reservation._get_existing_reservations()
        raise DiskSpaceError(
            f"Insufficient disk space: need {reservation.reserved_bytes / (1024**3):.2f}GB, "
            f"available {(available - existing) / (1024**3):.2f}GB",
            available_bytes=available,
            required_bytes=reservation.reserved_bytes,
            existing_reservations=existing,
        )

    try:
        yield acquired
    finally:
        reservation.release()


@asynccontextmanager
async def disk_space_reservation(
    target_dir: str | Path,
    estimated_bytes: int,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    raise_on_failure: bool = False,
):
    """Async context manager for disk space reservation.

    Args:
        target_dir: Directory where files will be written
        estimated_bytes: Estimated size of the transfer in bytes
        safety_margin: Extra buffer percentage (default: 10%)
        raise_on_failure: If True, raise DiskSpaceError on insufficient space

    Yields:
        True if reservation was acquired, False otherwise

    Raises:
        DiskSpaceError: If raise_on_failure is True and there's insufficient space
    """
    reservation = DiskSpaceReservation(
        target_dir=target_dir,
        estimated_bytes=estimated_bytes,
        safety_margin=safety_margin,
    )

    acquired = reservation.acquire()

    if not acquired and raise_on_failure:
        available = reservation._get_available_space()
        existing = reservation._get_existing_reservations()
        raise DiskSpaceError(
            f"Insufficient disk space: need {reservation.reserved_bytes / (1024**3):.2f}GB, "
            f"available {(available - existing) / (1024**3):.2f}GB",
            available_bytes=available,
            required_bytes=reservation.reserved_bytes,
            existing_reservations=existing,
        )

    try:
        yield acquired
    finally:
        reservation.release()


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Constants
    "DEFAULT_RESERVATION_DIR",
    "DEFAULT_SAFETY_MARGIN",
    "STALE_RESERVATION_THRESHOLD_SECONDS",
    # Exceptions
    "DiskSpaceError",
    # Data classes
    "ReservationInfo",
    # Main class
    "DiskSpaceReservation",
    # Context managers
    "disk_space_reservation",
    "disk_space_reservation_sync",
    # Utility functions
    "cleanup_stale_reservations",
    "get_active_reservations",
    "get_effective_available_space",
    "get_total_reserved_bytes",
]
