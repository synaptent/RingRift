"""WAL Sync Utilities - SQLite Write-Ahead Log handling for database sync.

This module provides utilities for safely syncing SQLite databases that may be
using WAL (Write-Ahead Log) mode. Without proper handling, syncing only the .db
file while leaving out .db-wal and .db-shm files can cause:

1. Data corruption on target (WAL transactions lost)
2. Incomplete data (recent writes in WAL not transferred)
3. Database appears empty or corrupted after sync

IMPORTANT: The functions in this module should be called BEFORE syncing any
SQLite database file.

Usage:
    from app.coordination.wal_sync_utils import (
        checkpoint_database,
        get_wal_files,
        get_db_with_wal_files,
        prepare_db_for_sync,
        validate_synced_database,
        WAL_INCLUDE_PATTERNS,
    )

    # Before sync - checkpoint to minimize WAL size
    success = checkpoint_database("/path/to/games.db")

    # Get list of all files to sync
    files = get_db_with_wal_files("/path/to/games.db")
    # Returns: ["/path/to/games.db", "/path/to/games.db-wal", "/path/to/games.db-shm"]

    # After sync - validate database integrity
    is_valid = validate_synced_database("/path/to/synced.db")

December 2025: Created to address WAL files not being synced with SQLite databases.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Rsync include patterns for WAL files
# Use these when syncing .db files to ensure WAL files are included
WAL_INCLUDE_PATTERNS: list[str] = [
    "*.db",
    "*.db-wal",
    "*.db-shm",
]

# Rsync arguments to include WAL files
# Add these to rsync commands when syncing databases
WAL_RSYNC_INCLUDES: list[str] = [
    "--include=*.db",
    "--include=*.db-wal",
    "--include=*.db-shm",
]


def get_wal_files(db_path: str | Path) -> list[Path]:
    """Get the WAL and SHM files for a database.

    Args:
        db_path: Path to the .db file

    Returns:
        List of existing WAL-related files (.db-wal, .db-shm)
    """
    db_path = Path(db_path)
    wal_files = []

    # WAL file contains uncommitted transactions
    wal_path = db_path.with_suffix(db_path.suffix + "-wal")
    if wal_path.exists():
        wal_files.append(wal_path)

    # SHM file is shared memory map for WAL coordination
    shm_path = db_path.with_suffix(db_path.suffix + "-shm")
    if shm_path.exists():
        wal_files.append(shm_path)

    return wal_files


def get_db_with_wal_files(db_path: str | Path) -> list[Path]:
    """Get database file along with any WAL files.

    This returns all files that should be synced together to ensure
    database integrity. The order is important:
    1. .db-wal (contains recent data)
    2. .db-shm (shared memory map)
    3. .db (main database)

    Syncing in this order prevents races where the main .db file
    is synced before its WAL data.

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


def checkpoint_database(db_path: str | Path, truncate: bool = True) -> bool:
    """Force WAL checkpoint to flush pending transactions to main database.

    This should be called BEFORE syncing a database to minimize WAL size
    and ensure all transactions are in the main .db file.

    Args:
        db_path: Path to the database file
        truncate: If True, use TRUNCATE mode to also reset the WAL file.
                  This is recommended before sync to minimize data transfer.

    Returns:
        True if checkpoint succeeded, False otherwise

    Example:
        # Before rsync
        if checkpoint_database("/data/games/selfplay.db"):
            rsync_to_remote(...)
    """
    db_path = Path(db_path)

    if not db_path.exists():
        logger.warning(f"[WALSync] Database not found for checkpoint: {db_path}")
        return False

    try:
        # Open with timeout to avoid blocking on locked database
        with sqlite3.connect(str(db_path), timeout=30.0) as conn:
            # Check if WAL mode is enabled
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

            if mode.upper() != "WAL":
                logger.debug(f"[WALSync] Database not in WAL mode: {db_path}")
                return True  # Not WAL mode, nothing to checkpoint

            # Perform checkpoint
            if truncate:
                # TRUNCATE mode: checkpoint + reset WAL file to zero size
                # This is most efficient for pre-sync preparation
                result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            else:
                # PASSIVE mode: checkpoint without blocking writers
                result = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()

            # Result is (busy, log_frames, checkpointed_frames)
            if result:
                busy, log_frames, checkpointed_frames = result
                if busy:
                    logger.warning(
                        f"[WALSync] Checkpoint busy for {db_path}: "
                        f"log={log_frames}, checkpointed={checkpointed_frames}"
                    )
                else:
                    logger.debug(
                        f"[WALSync] Checkpointed {db_path}: "
                        f"log={log_frames}, checkpointed={checkpointed_frames}"
                    )

            return True

    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower():
            logger.warning(f"[WALSync] Database locked during checkpoint: {db_path}")
        else:
            logger.warning(f"[WALSync] Checkpoint failed for {db_path}: {e}")
        return False
    except sqlite3.DatabaseError as e:
        logger.error(f"[WALSync] Database error during checkpoint: {db_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"[WALSync] Unexpected error during checkpoint: {db_path}: {e}")
        return False


def prepare_db_for_sync(db_path: str | Path) -> tuple[bool, str]:
    """Prepare a database for sync by checkpointing and validating.

    This is a higher-level function that:
    1. Checkpoints WAL to minimize transfer size
    2. Verifies database can be opened
    3. Reports any issues

    Args:
        db_path: Path to the database file

    Returns:
        Tuple of (success, message)

    Example:
        success, msg = prepare_db_for_sync("/data/games/selfplay.db")
        if not success:
            logger.warning(f"Sync preparation failed: {msg}")
    """
    db_path = Path(db_path)

    if not db_path.exists():
        return False, f"Database not found: {db_path}"

    # Try to checkpoint
    if not checkpoint_database(db_path):
        # Non-fatal - may still be syncable
        pass

    # Verify database can be opened
    try:
        with sqlite3.connect(str(db_path), timeout=10.0) as conn:
            # Quick validation
            conn.execute("SELECT 1").fetchone()

            # Check for WAL files that should be synced
            wal_files = get_wal_files(db_path)
            if wal_files:
                return True, f"Ready to sync with {len(wal_files)} WAL files"
            else:
                return True, "Ready to sync (no WAL files)"

    except sqlite3.DatabaseError as e:
        return False, f"Database validation failed: {e}"
    except Exception as e:
        return False, f"Unexpected validation error: {e}"


def validate_synced_database(
    db_path: str | Path,
    check_integrity: bool = True,
    min_expected_rows: int | None = None,
    table_name: str = "games",
) -> tuple[bool, list[str]]:
    """Validate a database after sync.

    This should be called on the TARGET after receiving a synced database
    to ensure it was transferred correctly.

    Args:
        db_path: Path to the synced database
        check_integrity: Run PRAGMA integrity_check (slower but thorough)
        min_expected_rows: Minimum expected rows in table (None to skip)
        table_name: Table to check for row count

    Returns:
        Tuple of (is_valid, list of error messages)

    Example:
        is_valid, errors = validate_synced_database("/data/synced.db")
        if not is_valid:
            for err in errors:
                logger.error(f"Sync validation: {err}")
    """
    db_path = Path(db_path)
    errors = []

    if not db_path.exists():
        return False, ["Database file not found"]

    try:
        with sqlite3.connect(str(db_path), timeout=30.0) as conn:
            # Basic connectivity test
            conn.execute("SELECT 1").fetchone()

            # Check integrity if requested
            if check_integrity:
                try:
                    # Dec 29, 2025: Use quick_check for large databases (>100MB)
                    # to prevent timeouts while still detecting most corruption
                    db_size_mb = db_path.stat().st_size / (1024 * 1024)
                    pragma = "PRAGMA quick_check" if db_size_mb > 100 else "PRAGMA integrity_check"
                    result = conn.execute(pragma).fetchall()
                    if len(result) != 1 or result[0][0] != "ok":
                        errors.extend(str(row[0]) for row in result)
                except sqlite3.Error as e:
                    errors.append(f"Integrity check failed: {e}")

            # Check row count if requested
            if min_expected_rows is not None:
                try:
                    count = conn.execute(
                        f"SELECT COUNT(*) FROM {table_name}"
                    ).fetchone()[0]
                    if count < min_expected_rows:
                        errors.append(
                            f"Expected at least {min_expected_rows} rows in {table_name}, "
                            f"got {count}"
                        )
                except sqlite3.OperationalError:
                    # Table might not exist - not necessarily an error
                    pass

            return len(errors) == 0, errors

    except sqlite3.DatabaseError as e:
        return False, [f"Database error: {e}"]
    except Exception as e:
        return False, [f"Validation error: {e}"]


def build_rsync_command_for_db(
    db_path: str | Path,
    dest: str,
    ssh_options: list[str] | None = None,
    bwlimit_kbps: int | None = None,
    extra_options: list[str] | None = None,
) -> list[str]:
    """Build an rsync command that includes WAL files for a database.

    This helper constructs an rsync command with the correct include/exclude
    patterns to sync a database along with its WAL files.

    Args:
        db_path: Path to the database file
        dest: Destination (e.g., "user@host:/path/")
        ssh_options: SSH options (e.g., ["-e", "ssh -i ~/.ssh/key"])
        bwlimit_kbps: Bandwidth limit in KB/s
        extra_options: Additional rsync options

    Returns:
        List of command arguments ready for subprocess

    Example:
        cmd = build_rsync_command_for_db(
            "/data/games.db",
            "ubuntu@remote:/data/",
            ssh_options=["-e", "ssh -i ~/.ssh/id_cluster"],
        )
        subprocess.run(cmd)
    """
    db_path = Path(db_path)
    parent_dir = str(db_path.parent) + "/"
    db_name = db_path.name

    cmd = ["rsync", "-avz", "--compress", "--partial"]  # Jan 2, 2026: Enable resume

    # Include the database and its WAL files
    # Note: Order matters - rsync processes rules in order
    cmd.extend([
        f"--include={db_name}",
        f"--include={db_name}-wal",
        f"--include={db_name}-shm",
        "--exclude=*",  # Exclude everything else in the directory
    ])

    if ssh_options:
        cmd.extend(ssh_options)

    if bwlimit_kbps:
        cmd.append(f"--bwlimit={bwlimit_kbps}")

    if extra_options:
        cmd.extend(extra_options)

    # Source directory (with trailing slash to sync contents)
    cmd.append(parent_dir)

    # Destination
    cmd.append(dest)

    return cmd


def get_rsync_include_args_for_db(db_name: str) -> list[str]:
    """Get rsync include arguments for a specific database file.

    Args:
        db_name: Name of the database file (e.g., "games.db")

    Returns:
        List of rsync arguments to include the DB and its WAL files

    Example:
        args = get_rsync_include_args_for_db("selfplay.db")
        # Returns: ["--include=selfplay.db", "--include=selfplay.db-wal", ...]
    """
    return [
        f"--include={db_name}",
        f"--include={db_name}-wal",
        f"--include={db_name}-shm",
    ]


__all__ = [
    "WAL_INCLUDE_PATTERNS",
    "WAL_RSYNC_INCLUDES",
    "build_rsync_command_for_db",
    "checkpoint_database",
    "get_db_with_wal_files",
    "get_rsync_include_args_for_db",
    "get_wal_files",
    "prepare_db_for_sync",
    "validate_synced_database",
]
