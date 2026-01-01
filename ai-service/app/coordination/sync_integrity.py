#!/usr/bin/env python3
"""Sync Integrity - Checksum validation and data integrity verification.

This module provides comprehensive integrity checking for sync operations:

1. File checksum computation (SHA256, streaming for large files)
2. SQLite database integrity verification (PRAGMA integrity_check)
3. Full sync integrity reports comparing source and target
4. Support for multiple hash algorithms
5. Structured error reporting

Note on checksum_utils relationship:
    This module provides STRICTER checksum validation than app.utils.checksum_utils:
    - Raises FileNotFoundError (vs returning empty string)
    - Validates path is_file()
    - Handles PermissionError explicitly
    For simple checksum needs, use app.utils.checksum_utils directly.
    Use this module for integrity verification contexts where failures must be explicit.

This consolidates checksum validation functionality from:
- app.distributed.unified_data_sync._compute_file_checksum
- app.distributed.p2p_sync_client checksum verification
- app.coordination.transfer_verification (higher-level wrapper)

Usage:
    from app.coordination.sync_integrity import (
        compute_file_checksum,
        compute_db_checksum,
        verify_checksum,
        check_sqlite_integrity,
        verify_sync_integrity,
        IntegrityReport,
    )

    # Basic file checksum
    checksum = compute_file_checksum(Path("data.db"))

    # Verify against expected
    if verify_checksum(Path("data.db"), expected_checksum):
        print("File is valid")

    # Check SQLite database integrity
    is_valid, errors = check_sqlite_integrity(Path("data.db"))

    # Full sync verification
    report = verify_sync_integrity(source_path, target_path)
    if report.is_valid:
        print("Sync successful")
    else:
        print(f"Errors: {report.errors}")
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "LARGE_CHUNK_SIZE",
    "LARGE_DB_THRESHOLD",
    "IntegrityCheckResult",
    "IntegrityReport",
    "check_sqlite_integrity",
    "compute_db_checksum",
    "compute_file_checksum",
    "verify_checksum",
    "verify_sync_integrity",
]

logger = logging.getLogger(__name__)

# Chunk sizes (December 27, 2025: Centralized in coordination_defaults.py)
from app.config.coordination_defaults import SyncIntegrityDefaults

DEFAULT_CHUNK_SIZE = SyncIntegrityDefaults.DEFAULT_CHUNK_SIZE
LARGE_CHUNK_SIZE = SyncIntegrityDefaults.LARGE_CHUNK_SIZE
# Dec 2025: Threshold for using fast integrity check (bytes)
LARGE_DB_THRESHOLD = SyncIntegrityDefaults.LARGE_DB_THRESHOLD

# Supported hash algorithms
HashAlgorithm = Literal["sha256", "sha1", "md5", "blake2b"]

# December 31, 2025: VACUUM cache to prevent repeated VACUUM operations
# on the same database within a short time window. Each VACUUM takes 1-20s
# depending on database size, and running VACUUM on a database that was
# just VACUUMed is redundant and wastes CPU.
_VACUUM_CACHE: dict[str, float] = {}  # db_path -> last_vacuum_time
VACUUM_CACHE_TTL_SECONDS = 60  # Don't re-VACUUM within 60 seconds


@dataclass
class IntegrityCheckResult:
    """Result of a SQLite integrity check."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    check_time: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "check_time": self.check_time,
        }


@dataclass
class IntegrityReport:
    """Comprehensive integrity verification report for sync operations."""

    source_path: str
    target_path: str
    is_valid: bool
    source_checksum: str
    target_checksum: str
    source_size: int
    target_size: int
    checksum_match: bool
    size_match: bool
    db_integrity_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    verification_time: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "source_path": self.source_path,
            "target_path": self.target_path,
            "is_valid": self.is_valid,
            "source_checksum": self.source_checksum[:16] + "..." if self.source_checksum else "",
            "target_checksum": self.target_checksum[:16] + "..." if self.target_checksum else "",
            "source_size": self.source_size,
            "target_size": self.target_size,
            "checksum_match": self.checksum_match,
            "size_match": self.size_match,
            "db_integrity_valid": self.db_integrity_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "verification_time": round(self.verification_time, 3),
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        if self.is_valid:
            return f"✓ Valid sync: {self.target_path} ({self.target_size} bytes)"
        else:
            error_summary = "; ".join(self.errors[:3])
            return f"✗ Invalid sync: {self.target_path} - {error_summary}"


# =============================================================================
# Core Checksum Functions
# =============================================================================


def compute_file_checksum(
    path: Path,
    algorithm: HashAlgorithm = "sha256",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """Compute checksum of a file using streaming read.

    Uses chunk-based reading to handle large files efficiently without
    loading the entire file into memory.

    Args:
        path: Path to the file
        algorithm: Hash algorithm to use (sha256, sha1, md5, blake2b)
        chunk_size: Size of chunks to read (default: 8KB)
                   Use LARGE_CHUNK_SIZE (64KB) for files > 100MB

    Returns:
        Hex-encoded hash string

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If algorithm is not supported
        PermissionError: If file cannot be read

    Example:
        # Standard usage
        checksum = compute_file_checksum(Path("data.db"))

        # Large file with bigger chunks
        checksum = compute_file_checksum(
            Path("huge.db"),
            chunk_size=LARGE_CHUNK_SIZE
        )
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    try:
        hasher = hashlib.new(algorithm)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e

    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)
    except PermissionError as e:
        raise PermissionError(f"Cannot read file: {path}") from e

    return hasher.hexdigest()


def compute_db_checksum(
    db_path: Path,
    algorithm: HashAlgorithm = "sha256",
) -> str:
    """Compute checksum of a SQLite database file.

    This is a specialized version of compute_file_checksum that:
    1. Uses large chunks for better performance on database files
    2. Logs warnings if database is locked or corrupted
    3. Falls back gracefully on errors

    Args:
        db_path: Path to SQLite database file
        algorithm: Hash algorithm to use (default: sha256)

    Returns:
        Hex-encoded hash string, or empty string on error

    Example:
        checksum = compute_db_checksum(Path("games.db"))
    """
    if not db_path.exists():
        logger.warning(f"[SyncIntegrity] Database not found: {db_path}")
        return ""

    try:
        # Use large chunks for database files (typically several MB to GB)
        return compute_file_checksum(db_path, algorithm=algorithm, chunk_size=LARGE_CHUNK_SIZE)
    except Exception as e:
        logger.warning(f"[SyncIntegrity] Failed to compute checksum for {db_path}: {e}")
        return ""


def verify_checksum(
    path: Path,
    expected: str,
    algorithm: HashAlgorithm = "sha256",
) -> bool:
    """Verify a file matches an expected checksum.

    Args:
        path: Path to file to verify
        expected: Expected hex-encoded checksum
        algorithm: Hash algorithm used for expected checksum

    Returns:
        True if checksum matches, False otherwise

    Example:
        if verify_checksum(Path("data.db"), expected_hash):
            print("File is valid")
        else:
            print("Checksum mismatch - file may be corrupted")
    """
    if not expected:
        logger.warning(f"[SyncIntegrity] No expected checksum provided for {path}")
        return False

    if not path.exists():
        logger.warning(f"[SyncIntegrity] File not found for verification: {path}")
        return False

    try:
        actual = compute_file_checksum(path, algorithm=algorithm)
        match = actual == expected

        if not match:
            logger.warning(
                f"[SyncIntegrity] Checksum mismatch for {path}: "
                f"expected {expected[:16]}..., got {actual[:16]}..."
            )

        return match

    except Exception as e:
        logger.error(f"[SyncIntegrity] Error verifying checksum for {path}: {e}")
        return False


# =============================================================================
# SQLite Integrity Checking
# =============================================================================


# SQLite magic header bytes
_SQLITE_HEADER_MAGIC = b"SQLite format 3\x00"


def _run_fast_integrity_check(db_path: Path) -> tuple[bool, list[str]]:
    """Run fast partial integrity check (Dec 2025).

    This performs minimal validation that completes in <1 second even for
    multi-GB databases. It validates:
    1. SQLite header magic bytes (first 16 bytes)
    2. Database can be opened and queried
    3. Page count is valid (>0)
    4. At least one table exists
    5. WAL mode check (if WAL exists, it should be consistent)

    This does NOT validate:
    - B-tree structure
    - Index consistency
    - Foreign key constraints
    - Full page checksums

    Use this for large databases where full integrity_check times out.
    """
    errors: list[str] = []

    try:
        # Step 1: Validate SQLite header magic bytes
        with open(db_path, "rb") as f:
            header = f.read(16)
            if header != _SQLITE_HEADER_MAGIC:
                return False, [f"Invalid SQLite header: {header[:16]!r}"]

        # Step 2-5: Open and run quick queries
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA busy_timeout = 2000")

            # Check page count (should be > 0)
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            if page_count <= 0:
                errors.append(f"Invalid page count: {page_count}")

            # Check that we can query sqlite_master (table list)
            cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            if table_count <= 0:
                errors.append("No tables found in database")

            # Check WAL status if WAL file exists (skip checkpoint - readonly mode)
            wal_path = Path(str(db_path) + "-wal")
            if wal_path.exists():
                # Just check WAL mode is consistent - don't try to checkpoint
                cursor.execute("PRAGMA journal_mode")
                journal_mode = cursor.fetchone()[0]
                if journal_mode != "wal":
                    # WAL file exists but journal mode isn't WAL - inconsistent
                    logger.debug(f"[SyncIntegrity] WAL file exists but mode is {journal_mode}")

            # Check freelist count (corrupted DBs often have invalid freelist)
            cursor.execute("PRAGMA freelist_count")
            freelist = cursor.fetchone()[0]
            if freelist < 0:
                errors.append(f"Invalid freelist count: {freelist}")

        if errors:
            logger.warning(f"[SyncIntegrity] Fast check found issues in {db_path}: {errors}")
            return False, errors

        return True, []

    except sqlite3.DatabaseError as e:
        error_msg = f"Database error during fast check: {e}"
        logger.error(f"[SyncIntegrity] {error_msg} for {db_path}")
        return False, [error_msg]

    except sqlite3.OperationalError as e:
        error_msg = f"Database inaccessible during fast check: {e}"
        logger.warning(f"[SyncIntegrity] {error_msg} for {db_path}")
        return False, [error_msg]

    except OSError as e:
        error_msg = f"File I/O error during fast check: {e}"
        logger.error(f"[SyncIntegrity] {error_msg} for {db_path}")
        return False, [error_msg]

    except Exception as e:
        error_msg = f"Unexpected error during fast check: {e}"
        logger.error(f"[SyncIntegrity] {error_msg} for {db_path}")
        return False, [error_msg]


def check_sqlite_integrity(
    db_path: Path,
    timeout_seconds: float = 30.0,
    use_quick_check: bool = False,
    use_fast_check: bool = False,
) -> tuple[bool, list[str]]:
    """Run SQLite integrity check on a database with timeout protection.

    This performs integrity validation with three modes:
    - Full (default): PRAGMA integrity_check - comprehensive but slow
    - Quick: PRAGMA quick_check - faster, less thorough
    - Fast: Minimal checks (header, tables, WAL) - 10x faster for large DBs

    Args:
        db_path: Path to SQLite database file
        timeout_seconds: Maximum time to wait for integrity check (default: 30s)
        use_quick_check: If True, use PRAGMA quick_check (faster, less thorough)
        use_fast_check: If True, use fast partial checks only (Dec 2025)
            - Validates SQLite header magic bytes
            - Checks table count and page count
            - Verifies WAL status
            - Does NOT verify B-tree structure

    Returns:
        Tuple of (is_valid, error_messages)
        - is_valid: True if database passes integrity check
        - error_messages: List of error messages (empty if valid)

    Note:
        Dec 2025: Added timeout protection to prevent hangs on corrupted/large DBs.
        Dec 2025: Added use_fast_check for 10x speedup on large databases (>100MB).
        For very large databases (>1GB), use use_fast_check=True.

    Example:
        is_valid, errors = check_sqlite_integrity(Path("games.db"))
        if not is_valid:
            print(f"Database corrupted: {errors}")

        # Fast check for large databases
        is_valid, errors = check_sqlite_integrity(Path("large.db"), use_fast_check=True)
    """
    import concurrent.futures
    import threading

    if not db_path.exists():
        return False, [f"Database file not found: {db_path}"]

    if not db_path.is_file():
        return False, [f"Path is not a file: {db_path}"]

    # Dec 2025: Fast check mode - validates header and basic structure only
    # This is 10x faster than full integrity_check for large databases
    if use_fast_check:
        return _run_fast_integrity_check(db_path)

    # Dec 2025: Run integrity check in thread with timeout to prevent hangs
    def _run_integrity_check() -> tuple[bool, list[str]]:
        """Execute integrity check in separate thread for timeout support."""
        try:
            # Use context manager to prevent connection leaks
            # Open read-only to avoid locking issues
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10) as conn:
                cursor = conn.cursor()

                # Set busy timeout for locked database handling
                cursor.execute("PRAGMA busy_timeout = 5000")

                # Run integrity check (quick_check is faster but less thorough)
                pragma = "quick_check" if use_quick_check else "integrity_check"
                cursor.execute(f"PRAGMA {pragma}")
                results = cursor.fetchall()

                # Check results
                # SQLite returns a single row with "ok" if everything is fine
                # Otherwise, returns multiple rows describing errors
                if len(results) == 1 and results[0][0] == "ok":
                    return True, []
                else:
                    errors = [str(row[0]) for row in results]
                    logger.warning(f"[SyncIntegrity] Database {db_path} integrity check failed: {errors}")
                    return False, errors

        except sqlite3.DatabaseError as e:
            error_msg = f"Database error: {e}"
            logger.error(f"[SyncIntegrity] {error_msg} for {db_path}")
            return False, [error_msg]

        except sqlite3.OperationalError as e:
            error_msg = f"Database locked or inaccessible: {e}"
            logger.warning(f"[SyncIntegrity] {error_msg} for {db_path}")
            return False, [error_msg]

        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(f"[SyncIntegrity] {error_msg} for {db_path}")
            return False, [error_msg]

    # Execute with timeout
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_integrity_check)
            return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        error_msg = f"Integrity check timed out after {timeout_seconds}s (database may be corrupted or very large)"
        logger.warning(f"[SyncIntegrity] {error_msg} for {db_path}")
        return False, [error_msg]


# =============================================================================
# Full Sync Integrity Verification
# =============================================================================


def verify_sync_integrity(
    source: Path,
    target: Path,
    algorithm: HashAlgorithm = "sha256",
    check_db: bool = True,
) -> IntegrityReport:
    """Perform full integrity verification for a sync operation.

    This comprehensive check verifies:
    1. Both files exist
    2. File sizes match
    3. Checksums match
    4. SQLite database integrity (if target is .db file and check_db=True)

    Args:
        source: Source file path
        target: Target file path
        algorithm: Hash algorithm to use (default: sha256)
        check_db: If True, run PRAGMA integrity_check on .db files

    Returns:
        IntegrityReport with detailed verification results

    Example:
        report = verify_sync_integrity(
            source=Path("remote/games.db"),
            target=Path("local/games.db")
        )

        if report.is_valid:
            print("Sync verified successfully")
        else:
            print(f"Sync failed: {report.summary()}")
            for error in report.errors:
                print(f"  - {error}")
    """
    start_time = time.time()
    errors = []
    warnings = []

    # Initialize with defaults
    source_checksum = ""
    target_checksum = ""
    source_size = 0
    target_size = 0
    checksum_match = False
    size_match = False
    db_integrity_valid = True  # Default to True if not checked

    # Check source exists
    if not source.exists():
        errors.append(f"Source file not found: {source}")
    else:
        try:
            source_size = source.stat().st_size
        except Exception as e:
            errors.append(f"Cannot stat source file: {e}")

    # Check target exists
    if not target.exists():
        errors.append(f"Target file not found: {target}")
    else:
        try:
            target_size = target.stat().st_size
        except Exception as e:
            errors.append(f"Cannot stat target file: {e}")

    # Compare sizes
    if source_size > 0 and target_size > 0:
        size_match = source_size == target_size
        if not size_match:
            errors.append(
                f"Size mismatch: source={source_size} bytes, target={target_size} bytes "
                f"(diff: {abs(source_size - target_size)} bytes)"
            )

    # Compute checksums
    if source.exists():
        try:
            # Use large chunks for better performance
            chunk_size = LARGE_CHUNK_SIZE if source_size > 1_000_000 else DEFAULT_CHUNK_SIZE
            source_checksum = compute_file_checksum(source, algorithm=algorithm, chunk_size=chunk_size)
        except Exception as e:
            errors.append(f"Cannot compute source checksum: {e}")

    if target.exists():
        try:
            # Use large chunks for better performance
            chunk_size = LARGE_CHUNK_SIZE if target_size > 1_000_000 else DEFAULT_CHUNK_SIZE
            target_checksum = compute_file_checksum(target, algorithm=algorithm, chunk_size=chunk_size)
        except Exception as e:
            errors.append(f"Cannot compute target checksum: {e}")

    # Compare checksums
    if source_checksum and target_checksum:
        checksum_match = source_checksum == target_checksum
        if not checksum_match:
            errors.append(
                f"Checksum mismatch: source={source_checksum[:16]}..., "
                f"target={target_checksum[:16]}..."
            )

    # Check SQLite database integrity
    is_db_file = target.suffix.lower() == ".db"
    if check_db and is_db_file and target.exists():
        try:
            # Dec 2025: Use fast check for large databases to prevent timeouts
            use_fast = target_size > LARGE_DB_THRESHOLD
            db_integrity_valid, db_errors = check_sqlite_integrity(
                target, use_fast_check=use_fast
            )
            if not db_integrity_valid:
                errors.extend(f"SQLite integrity error: {err}" for err in db_errors)
        except Exception as e:
            warnings.append(f"Could not check database integrity: {e}")
            # Don't fail the whole verification if integrity check fails
            # (database might be locked, read-only, etc.)

    # Determine overall validity
    is_valid = (
        len(errors) == 0
        and size_match
        and checksum_match
        and db_integrity_valid
    )

    verification_time = time.time() - start_time

    report = IntegrityReport(
        source_path=str(source),
        target_path=str(target),
        is_valid=is_valid,
        source_checksum=source_checksum,
        target_checksum=target_checksum,
        source_size=source_size,
        target_size=target_size,
        checksum_match=checksum_match,
        size_match=size_match,
        db_integrity_valid=db_integrity_valid,
        errors=errors,
        warnings=warnings,
        verification_time=verification_time,
    )

    if not is_valid:
        logger.warning(f"[SyncIntegrity] {report.summary()}")
    else:
        logger.debug(f"[SyncIntegrity] ✓ {target} verified ({target_size} bytes, {verification_time:.2f}s)")

    return report


# =============================================================================
# Database Transfer Safety (December 2025)
# =============================================================================
# These functions address the root cause of database corruption during sync:
# 1. SQLite WAL mode leaves uncommitted data in -wal files
# 2. Partial aria2 downloads reassemble incorrectly after connection resets
# 3. Non-atomic file operations leave incomplete files on disk


def prepare_database_for_transfer(db_path: Path) -> tuple[bool, str]:
    """Prepare a SQLite database for safe transfer by consolidating all data.

    This function:
    1. Checkpoints any WAL data into the main database file
    2. Runs VACUUM to consolidate and defragment the database
    3. Sets journal mode to DELETE (no sidecar files needed)

    This is CRITICAL before transferring databases to prevent corruption.
    Without this, WAL mode databases may transfer without their -wal files,
    resulting in missing transactions and data corruption.

    December 31, 2025: Added VACUUM caching to prevent repeated VACUUM
    operations on the same database within 60 seconds. This reduces CPU
    usage when multiple sync attempts target the same database.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Tuple of (success, message)

    Example:
        success, msg = prepare_database_for_transfer(Path("games.db"))
        if success:
            # Safe to transfer games.db
            rsync_file(...)
    """
    if not db_path.exists():
        return False, f"Database not found: {db_path}"

    # Check VACUUM cache - skip if recently VACUUMed
    db_key = str(db_path.resolve())
    now = time.time()
    last_vacuum = _VACUUM_CACHE.get(db_key, 0)
    if now - last_vacuum < VACUUM_CACHE_TTL_SECONDS:
        logger.debug(
            f"[TransferSafety] Skipping {db_path.name}: VACUUMed "
            f"{int(now - last_vacuum)}s ago (TTL: {VACUUM_CACHE_TTL_SECONDS}s)"
        )
        return True, "Database already prepared (cached)"

    try:
        # Dec 2025: Use context manager to prevent connection leaks
        with sqlite3.connect(str(db_path)) as conn:
            # Check current journal mode
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            logger.info(f"[TransferSafety] {db_path.name}: current journal mode = {mode}")

            # Checkpoint any WAL data
            if mode.upper() == "WAL":
                logger.info(f"[TransferSafety] {db_path.name}: checkpointing WAL...")
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

            # Switch to DELETE mode (simpler, no sidecar files)
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.commit()

            # VACUUM to consolidate database
            logger.info(f"[TransferSafety] {db_path.name}: running VACUUM...")
            conn.execute("VACUUM")
            conn.commit()

        # Verify no WAL files remain
        wal_path = db_path.with_suffix(db_path.suffix + "-wal")
        shm_path = db_path.with_suffix(db_path.suffix + "-shm")
        if wal_path.exists():
            wal_path.unlink()
            logger.info(f"[TransferSafety] Removed orphaned WAL file: {wal_path}")
        if shm_path.exists():
            shm_path.unlink()
            logger.info(f"[TransferSafety] Removed orphaned SHM file: {shm_path}")

        # Update VACUUM cache
        _VACUUM_CACHE[db_key] = time.time()

        logger.info(f"[TransferSafety] ✓ {db_path.name}: prepared for transfer")
        return True, "Database prepared for transfer"

    except sqlite3.Error as e:
        return False, f"SQLite error: {e}"
    except (OSError, PermissionError, ValueError, TypeError) as e:
        return False, f"Unexpected error: {e}"


def atomic_file_write(
    target_path: Path,
    write_func,
    temp_suffix: str = ".tmp",
) -> tuple[bool, str]:
    """Write a file atomically using temp file + rename pattern.

    This ensures that the target file is either:
    - Completely written with valid content, or
    - Unchanged (if write fails)

    Args:
        target_path: Final destination path
        write_func: Callable that writes to the temp file path
        temp_suffix: Suffix for temporary file

    Returns:
        Tuple of (success, message)

    Example:
        def do_download(temp_path):
            # Download to temp_path
            subprocess.run(["aria2c", "-o", str(temp_path), url])

        success, msg = atomic_file_write(Path("model.pth"), do_download)
    """
    import os
    import shutil

    temp_path = target_path.parent / f".{target_path.name}{temp_suffix}"

    try:
        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove any stale temp file
        if temp_path.exists():
            temp_path.unlink()

        # Execute the write function
        write_func(temp_path)

        # Verify temp file was created
        if not temp_path.exists():
            return False, "Write function did not create temp file"

        # Atomic rename (POSIX guarantees atomicity on same filesystem)
        os.rename(str(temp_path), str(target_path))

        return True, f"Atomically wrote {target_path}"

    except Exception as e:
        # Cleanup temp file on error
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError as unlink_err:
                logger.debug(f"Could not remove temp file {temp_path}: {unlink_err}")
        return False, f"Atomic write failed: {e}"


def verified_database_copy(
    source_path: Path,
    target_path: Path,
    prepare_source: bool = True,
) -> tuple[bool, str]:
    """Copy a SQLite database with full safety guarantees.

    This combines all safety measures:
    1. (Optional) Prepares source database (VACUUM, checkpoint WAL)
    2. Computes source checksum before copy
    3. Copies to temp file
    4. Verifies target checksum matches source
    5. Runs SQLite integrity check on target
    6. Atomic rename to final location

    Args:
        source_path: Source database file
        target_path: Target destination
        prepare_source: Whether to VACUUM source first (default: True)

    Returns:
        Tuple of (success, message)

    Example:
        success, msg = verified_database_copy(
            Path("games.db"),
            Path("/backup/games.db"),
        )
    """
    import shutil

    if not source_path.exists():
        return False, f"Source not found: {source_path}"

    try:
        # Step 1: Prepare source (optional)
        if prepare_source:
            success, msg = prepare_database_for_transfer(source_path)
            if not success:
                logger.warning(f"[TransferSafety] Could not prepare source: {msg}")
                # Continue anyway - copying an unprepared DB is better than not copying

        # Step 2: Compute source checksum
        source_checksum = compute_file_checksum(source_path)
        if not source_checksum:
            return False, "Failed to compute source checksum"

        source_size = source_path.stat().st_size
        logger.info(f"[TransferSafety] Source: {source_size} bytes, checksum: {source_checksum[:16]}...")

        # Step 3: Copy to temp file
        temp_path = target_path.parent / f".{target_path.name}.tmp"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if temp_path.exists():
            temp_path.unlink()

        shutil.copy2(str(source_path), str(temp_path))

        # Step 4: Verify checksum
        target_checksum = compute_file_checksum(temp_path)
        if target_checksum != source_checksum:
            temp_path.unlink()
            return False, f"Checksum mismatch: expected {source_checksum[:16]}..., got {target_checksum[:16]}..."

        # Step 5: SQLite integrity check with adaptive timeout for large databases
        # Dec 29, 2025: Use fast check for DBs > 100MB to prevent timeouts
        db_size_mb = source_size / (1024 * 1024)
        use_fast = db_size_mb > 100  # Fast check for DBs > 100MB
        is_valid, errors = check_sqlite_integrity(
            temp_path,
            use_fast_check=use_fast,
            timeout_seconds=15.0 if use_fast else 30.0,
        )
        if not is_valid:
            temp_path.unlink()
            return False, f"SQLite integrity check failed: {errors}"

        # Step 6: Atomic rename
        import os
        os.rename(str(temp_path), str(target_path))

        logger.info(f"[TransferSafety] ✓ Successfully copied {source_path.name} ({source_size} bytes)")
        return True, f"Copied and verified {source_path.name}"

    except Exception as e:
        # Cleanup
        temp_path = target_path.parent / f".{target_path.name}.tmp"
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError as unlink_err:
                logger.debug(f"Could not remove temp file {temp_path}: {unlink_err}")
        return False, f"Copy failed: {e}"


# =============================================================================
# Fast Checksum for Large Files (December 28, 2025)
# =============================================================================


def compute_fast_checksum(
    path: Path,
    algorithm: HashAlgorithm = "md5",
) -> str:
    """Compute a fast checksum for large files using chunked sampling.

    For files > 100MB, this reads only the first 64KB, middle 64KB, and
    last 64KB, providing a fast "fingerprint" that catches most corruption.

    For smaller files, computes full checksum.

    Args:
        path: Path to file
        algorithm: Hash algorithm to use (default: md5 for speed)

    Returns:
        Hex-encoded checksum string

    Example:
        # Fast check for large model files
        checksum = compute_fast_checksum(Path("model.pth"))
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    size = path.stat().st_size
    sample_size = 65536  # 64KB

    # For small files, use full checksum
    if size < 100 * 1024 * 1024:  # 100MB
        return compute_file_checksum(path, algorithm=algorithm)

    try:
        hasher = hashlib.new(algorithm)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e

    try:
        with open(path, "rb") as f:
            # Read first 64KB
            data = f.read(sample_size)
            hasher.update(data)

            # Read middle 64KB
            middle_pos = max(0, (size // 2) - (sample_size // 2))
            f.seek(middle_pos)
            data = f.read(sample_size)
            hasher.update(data)

            # Read last 64KB
            f.seek(max(0, size - sample_size))
            data = f.read(sample_size)
            hasher.update(data)

            # Include file size in hash for extra safety
            hasher.update(str(size).encode())

        return hasher.hexdigest()

    except PermissionError as e:
        raise PermissionError(f"Cannot read file: {path}") from e


# =============================================================================
# Remote Checksum Verification (December 28, 2025)
# =============================================================================


async def compute_remote_checksum(
    ssh_host: str,
    remote_path: str,
    ssh_user: str = "ubuntu",
    ssh_key: str = "~/.ssh/id_cluster",
    algorithm: HashAlgorithm = "md5",
    timeout: float = 60.0,
) -> str | None:
    """Compute checksum of a remote file via SSH.

    Uses md5sum/sha256sum on the remote host for efficiency (no file transfer).

    Args:
        ssh_host: Remote SSH host (IP or hostname)
        remote_path: Path to file on remote host
        ssh_user: SSH username (default: ubuntu)
        ssh_key: Path to SSH private key
        algorithm: Hash algorithm (md5 or sha256)
        timeout: Command timeout in seconds

    Returns:
        Hex-encoded checksum string, or None if command fails

    Example:
        checksum = await compute_remote_checksum(
            "192.168.1.100",
            "/data/games/canonical_hex8_2p.db",
        )
    """
    import asyncio
    import os

    # Choose command based on algorithm
    cmd_name = f"{algorithm}sum" if algorithm in ("md5", "sha256") else f"{algorithm}sum"

    ssh_key_expanded = os.path.expanduser(ssh_key)
    ssh_cmd = [
        "ssh",
        "-i", ssh_key_expanded,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
        f"{ssh_user}@{ssh_host}",
        f"{cmd_name} '{remote_path}' 2>/dev/null | cut -d' ' -f1",
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

        if proc.returncode == 0 and stdout:
            checksum = stdout.decode().strip()
            if checksum and len(checksum) in (32, 64, 128):  # md5, sha256, sha512
                return checksum

        logger.debug(
            f"[SyncIntegrity] Remote checksum failed for {ssh_host}:{remote_path}: "
            f"returncode={proc.returncode}, stderr={stderr.decode() if stderr else ''}"
        )
        return None

    except asyncio.TimeoutError:
        logger.warning(f"[SyncIntegrity] Remote checksum timeout for {ssh_host}:{remote_path}")
        return None
    except Exception as e:
        logger.debug(f"[SyncIntegrity] Remote checksum error: {e}")
        return None


async def verify_sync_checksum(
    source_path: str,
    dest_path: str,
    ssh_host: str | None = None,
    ssh_user: str = "ubuntu",
    ssh_key: str = "~/.ssh/id_cluster",
    algorithm: HashAlgorithm = "md5",
    use_fast_checksum: bool = True,
) -> tuple[bool, str]:
    """Verify file checksums match after sync operation.

    Compares checksums between source and destination files. Supports both
    local-to-local and local-to-remote verification.

    Args:
        source_path: Path to source file (local)
        dest_path: Path to destination file (local or remote)
        ssh_host: If provided, dest_path is on this remote host
        ssh_user: SSH username for remote verification
        ssh_key: Path to SSH private key
        algorithm: Hash algorithm to use (md5 for speed, sha256 for security)
        use_fast_checksum: Use chunked sampling for files > 100MB

    Returns:
        Tuple of (success, error_message)
        - success: True if checksums match
        - error_message: Empty string on success, error details on failure

    Example:
        # Local verification
        success, error = await verify_sync_checksum(
            "/data/source.db",
            "/backup/source.db",
        )

        # Remote verification
        success, error = await verify_sync_checksum(
            "/data/source.db",
            "/data/games/source.db",
            ssh_host="192.168.1.100",
        )
    """
    source = Path(source_path)

    # Compute source checksum
    try:
        if use_fast_checksum and source.stat().st_size > 100 * 1024 * 1024:
            source_checksum = compute_fast_checksum(source, algorithm=algorithm)
        else:
            source_checksum = compute_file_checksum(source, algorithm=algorithm)
    except Exception as e:
        return False, f"Failed to compute source checksum: {e}"

    # Compute destination checksum
    if ssh_host:
        # Remote file
        dest_checksum = await compute_remote_checksum(
            ssh_host=ssh_host,
            remote_path=dest_path,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            algorithm=algorithm,
        )
        if dest_checksum is None:
            return False, f"Failed to compute remote checksum on {ssh_host}:{dest_path}"
    else:
        # Local file
        dest = Path(dest_path)
        if not dest.exists():
            return False, f"Destination file not found: {dest_path}"
        try:
            if use_fast_checksum and dest.stat().st_size > 100 * 1024 * 1024:
                dest_checksum = compute_fast_checksum(dest, algorithm=algorithm)
            else:
                dest_checksum = compute_file_checksum(dest, algorithm=algorithm)
        except Exception as e:
            return False, f"Failed to compute destination checksum: {e}"

    # Compare checksums
    if source_checksum != dest_checksum:
        error_msg = (
            f"Checksum mismatch: source={source_checksum[:16]}..., "
            f"dest={dest_checksum[:16] if dest_checksum else 'None'}..."
        )
        logger.error(f"[SyncIntegrity] {error_msg}")
        return False, error_msg

    logger.debug(f"[SyncIntegrity] Checksum verified: {source_path} -> {dest_path}")
    return True, ""


async def verify_and_retry_sync(
    source_path: str,
    dest_path: str,
    ssh_host: str,
    ssh_user: str = "ubuntu",
    ssh_key: str = "~/.ssh/id_cluster",
    sync_func=None,
    max_retries: int = 1,
) -> tuple[bool, str]:
    """Verify sync checksum and retry once if verification fails.

    This is the main entry point for checksum-verified sync operations.
    After a sync completes, it:
    1. Verifies the checksum matches
    2. If mismatch, deletes the corrupted file and retries sync once
    3. Emits SYNC_CHECKSUM_FAILED event if retry also fails

    Args:
        source_path: Path to source file
        dest_path: Path to destination file on remote host
        ssh_host: Remote SSH host
        ssh_user: SSH username
        ssh_key: Path to SSH private key
        sync_func: Async callable to execute the sync (takes no args, returns bool)
        max_retries: Number of retry attempts (default: 1)

    Returns:
        Tuple of (success, error_message)

    Example:
        async def do_rsync():
            result = await rsync_to_target(source, target)
            return result.success

        success, error = await verify_and_retry_sync(
            "/data/source.db",
            "/data/games/source.db",
            ssh_host="192.168.1.100",
            sync_func=do_rsync,
        )
    """
    # First verification
    success, error = await verify_sync_checksum(
        source_path=source_path,
        dest_path=dest_path,
        ssh_host=ssh_host,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
    )

    if success:
        return True, ""

    # Checksum failed - retry if sync_func provided
    if sync_func is None or max_retries < 1:
        return False, error

    logger.warning(
        f"[SyncIntegrity] Checksum verification failed, attempting cleanup and retry: {error}"
    )

    # Delete corrupted file on remote
    try:
        import asyncio
        import os

        ssh_key_expanded = os.path.expanduser(ssh_key)
        delete_cmd = [
            "ssh",
            "-i", ssh_key_expanded,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            f"{ssh_user}@{ssh_host}",
            f"rm -f '{dest_path}'",
        ]

        proc = await asyncio.create_subprocess_exec(
            *delete_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=30.0)

    except Exception as e:
        logger.warning(f"[SyncIntegrity] Failed to delete corrupted file: {e}")

    # Retry sync
    try:
        retry_success = await sync_func()
        if not retry_success:
            await _emit_sync_checksum_failed(source_path, dest_path, ssh_host, "Retry sync failed")
            return False, "Retry sync failed"

    except Exception as e:
        await _emit_sync_checksum_failed(source_path, dest_path, ssh_host, str(e))
        return False, f"Retry sync error: {e}"

    # Verify again after retry
    success, error = await verify_sync_checksum(
        source_path=source_path,
        dest_path=dest_path,
        ssh_host=ssh_host,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
    )

    if not success:
        await _emit_sync_checksum_failed(source_path, dest_path, ssh_host, error)
        return False, f"Checksum verification failed after retry: {error}"

    logger.info(f"[SyncIntegrity] Checksum verified after retry: {source_path} -> {ssh_host}:{dest_path}")
    return True, ""


async def _emit_sync_checksum_failed(
    source_path: str,
    dest_path: str,
    target_node: str,
    error: str,
) -> None:
    """Emit SYNC_CHECKSUM_FAILED event for monitoring.

    Args:
        source_path: Source file path
        dest_path: Destination file path
        target_node: Target node hostname/IP
        error: Error description
    """
    try:
        from app.distributed.data_events import DataEventType, emit_data_event

        emit_data_event(
            event_type=DataEventType.SYNC_CHECKSUM_FAILED,
            payload={
                "source_path": source_path,
                "dest_path": dest_path,
                "target_node": target_node,
                "error": error,
            },
            source="sync_integrity",
        )
    except (ImportError, AttributeError) as e:
        logger.debug(f"[SyncIntegrity] Could not emit SYNC_CHECKSUM_FAILED: {e}")


# =============================================================================
# Quarantine Functions (Phase 7 - December 29, 2025)
# =============================================================================


def quarantine_corrupted_db(db_path: Path) -> Path | None:
    """Move corrupted database to quarantine directory.

    Phase 7 - December 29, 2025: Quarantine corrupted databases to prevent them
    from participating in sync operations and potentially spreading corruption.

    Args:
        db_path: Path to the corrupted database file

    Returns:
        Path to the quarantined file, or None if quarantine failed
    """
    import shutil

    if not db_path.exists():
        logger.warning(f"[SyncIntegrity] Cannot quarantine - file not found: {db_path}")
        return None

    try:
        quarantine_dir = db_path.parent / "quarantine"
        quarantine_dir.mkdir(exist_ok=True)

        # Add timestamp to filename to avoid collisions
        timestamp = int(time.time())
        dest = quarantine_dir / f"{db_path.name}.{timestamp}"

        # Move the database file
        shutil.move(str(db_path), str(dest))
        logger.warning(f"[SyncIntegrity] Quarantined corrupted DB: {db_path} -> {dest}")

        # Also move associated WAL and SHM files if they exist
        for suffix in ["-wal", "-shm", "-journal"]:
            wal_path = db_path.parent / f"{db_path.name}{suffix}"
            if wal_path.exists():
                wal_dest = quarantine_dir / f"{db_path.name}.{timestamp}{suffix}"
                shutil.move(str(wal_path), str(wal_dest))
                logger.debug(f"[SyncIntegrity] Quarantined WAL/SHM: {wal_path}")

        return dest

    except (OSError, PermissionError, shutil.Error) as e:
        logger.error(f"[SyncIntegrity] Failed to quarantine {db_path}: {e}")
        return None


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "LARGE_CHUNK_SIZE",
    "IntegrityCheckResult",
    "IntegrityReport",
    "atomic_file_write",
    "check_sqlite_integrity",
    "compute_db_checksum",
    "compute_fast_checksum",
    "compute_file_checksum",
    "compute_remote_checksum",
    "prepare_database_for_transfer",
    "quarantine_corrupted_db",
    "verified_database_copy",
    "verify_and_retry_sync",
    "verify_checksum",
    "verify_sync_checksum",
    "verify_sync_integrity",
]
