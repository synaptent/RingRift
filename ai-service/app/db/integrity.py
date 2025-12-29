"""Database integrity checking and repair utilities.

This module provides consolidated functions for checking and repairing SQLite
databases across the RingRift cluster. Previously duplicated in:
- scripts/p2p_orchestrator.py
- scripts/unified_ai_loop.py

Usage:
    from app.db.integrity import (
        check_database_integrity,
        check_and_repair_databases,
        recover_corrupted_database,
    )

    # Check a single database
    is_healthy, error_msg = check_database_integrity(Path("games.db"))

    # Scan and repair all databases in a directory
    results = check_and_repair_databases(
        data_dir=Path("data/games"),
        auto_repair=True,
    )
"""

import os
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


# Default timeout for integrity checks (30 seconds)
# Can be overridden via environment variable
DEFAULT_INTEGRITY_CHECK_TIMEOUT = float(
    os.environ.get("RINGRIFT_INTEGRITY_CHECK_TIMEOUT", "30.0")
)


class DatabaseIntegrityError(Exception):
    """Base exception for database integrity issues."""
    pass


def check_database_integrity(
    db_path: Path,
    timeout_seconds: float | None = None,
) -> tuple[bool, str]:
    """Check SQLite database integrity using PRAGMA integrity_check.

    Args:
        db_path: Path to the SQLite database file
        timeout_seconds: Maximum time for integrity check. Defaults to
            RINGRIFT_INTEGRITY_CHECK_TIMEOUT env var or 30 seconds.
            Set to 0 or negative for no timeout (not recommended for large DBs).

    Returns:
        Tuple of (is_healthy: bool, error_message: str or "ok")

    Note:
        For large databases, PRAGMA integrity_check can take minutes.
        This function uses subprocess to enforce the timeout reliably.
        If the check times out, it falls back to a quick header validation.
    """
    if timeout_seconds is None:
        timeout_seconds = DEFAULT_INTEGRITY_CHECK_TIMEOUT

    # If timeout disabled, use direct connection (faster but can hang)
    if timeout_seconds <= 0:
        return _check_integrity_direct(db_path)

    # Use subprocess for reliable timeout enforcement
    try:
        result = subprocess.run(
            ["sqlite3", str(db_path), "PRAGMA integrity_check;"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            if output == "ok":
                return True, "ok"
            else:
                return False, output or "unknown integrity error"
        else:
            error_msg = result.stderr.strip() or "sqlite3 command failed"
            return False, error_msg
    except subprocess.TimeoutExpired:
        # Timeout - fall back to quick header check
        return _check_integrity_quick(db_path)
    except FileNotFoundError:
        # sqlite3 CLI not available, fall back to Python with interrupt
        return _check_integrity_with_interrupt(db_path, timeout_seconds)
    except OSError as e:
        # File system errors (permissions, disk full, etc.)
        return False, f"check failed: {e}"


def _check_integrity_direct(db_path: Path) -> tuple[bool, str]:
    """Direct integrity check without timeout (for small DBs or when timeout disabled)."""
    try:
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()
        conn.close()
        if result and result[0] == "ok":
            return True, "ok"
        else:
            return False, result[0] if result else "unknown error"
    except sqlite3.DatabaseError as e:
        return False, str(e)
    except OSError as e:
        # File system errors (permissions, disk full, etc.)
        return False, f"check failed: {e}"


def _check_integrity_quick(db_path: Path) -> tuple[bool, str]:
    """Quick integrity check when full check times out.

    This validates:
    1. File is readable
    2. SQLite header is valid (first 16 bytes)
    3. Can open and query basic schema

    Returns:
        Tuple of (is_likely_ok: bool, status_message: str)
    """
    try:
        # Check file exists and is readable
        if not db_path.exists():
            return False, "file does not exist"
        if db_path.stat().st_size < 100:  # Minimum SQLite header size
            return False, "file too small to be valid SQLite database"

        # Check SQLite magic header
        with open(db_path, "rb") as f:
            header = f.read(16)
        if not header.startswith(b"SQLite format 3"):
            return False, "invalid SQLite header"

        # Try to open and query schema (validates page structure)
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM sqlite_master")
        cursor.fetchone()
        conn.close()

        # Passed quick checks - likely OK but not verified
        return True, "ok (quick check - full check timed out)"

    except sqlite3.DatabaseError as e:
        return False, f"quick check failed: {e}"
    except OSError as e:
        # File system errors (permissions, disk full, etc.)
        return False, f"quick check error: {e}"


def _check_integrity_with_interrupt(
    db_path: Path,
    timeout_seconds: float,
) -> tuple[bool, str]:
    """Integrity check with interrupt capability (fallback when sqlite3 CLI unavailable)."""
    import threading

    conn = None
    result_container = {"result": None, "error": None}

    def run_check():
        nonlocal conn
        try:
            conn = sqlite3.connect(str(db_path), timeout=10.0)
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            result_container["result"] = cursor.fetchone()
        except Exception as e:
            result_container["error"] = e

    def interrupt_check():
        if conn is not None:
            try:
                conn.interrupt()
            except Exception:
                pass

    check_thread = threading.Thread(target=run_check)
    check_thread.start()
    check_thread.join(timeout=timeout_seconds)

    if check_thread.is_alive():
        # Timeout - interrupt the connection
        interrupt_check()
        check_thread.join(timeout=2.0)  # Give it a moment to clean up
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        # Fall back to quick check
        return _check_integrity_quick(db_path)

    # Thread completed - check result
    if conn:
        try:
            conn.close()
        except Exception:
            pass

    if result_container["error"]:
        return False, str(result_container["error"])

    result = result_container["result"]
    if result and result[0] == "ok":
        return True, "ok"
    else:
        return False, result[0] if result else "unknown error"


def recover_corrupted_database(
    db_path: Path,
    log_prefix: str = "[DBIntegrity]"
) -> bool:
    """Attempt to recover a corrupted database using .dump and reimport.

    This function:
    1. Dumps the database to SQL using sqlite3 .dump command
    2. Archives the corrupted original to a 'corrupted' subdirectory
    3. Reimports the dump into a new database file
    4. Verifies the recovered database

    Args:
        db_path: Path to the corrupted database
        log_prefix: Prefix for log messages (e.g., "[P2P]" or "[UnifiedLoop]")

    Returns:
        True if recovery succeeded, False otherwise
    """
    corrupted_dir = db_path.parent / "corrupted"
    corrupted_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = corrupted_dir / f"{db_path.stem}_{timestamp}.db.corrupted"

    try:
        # Try to dump the database
        dump_path = db_path.with_suffix(".sql")
        result = subprocess.run(
            ["sqlite3", str(db_path), ".dump"],
            capture_output=True,
            timeout=300
        )
        if result.returncode != 0 or not result.stdout:
            # Dump failed - just archive the corrupted file
            print(f"{log_prefix} Cannot dump {db_path.name}, archiving corrupted file")
            db_path.rename(backup_path)
            return False

        # Write dump to file
        with open(dump_path, "wb") as f:
            f.write(result.stdout)

        # Archive the corrupted original
        db_path.rename(backup_path)

        # Reimport from dump
        new_db_path = db_path
        result = subprocess.run(
            ["sqlite3", str(new_db_path)],
            input=result.stdout,
            capture_output=True,
            timeout=600
        )
        if result.returncode == 0:
            # Verify the recovered database
            is_healthy, _ = check_database_integrity(new_db_path)
            if is_healthy:
                print(f"{log_prefix} Successfully recovered {db_path.name}")
                dump_path.unlink()  # Remove dump file
                return True

        # Recovery failed - remove partial file
        if new_db_path.exists():
            new_db_path.unlink()
        print(f"{log_prefix} Recovery failed for {db_path.name}, kept backup at {backup_path}")
        return False

    except subprocess.TimeoutExpired:
        print(f"{log_prefix} Recovery timed out for {db_path.name}")
        return False
    except Exception as e:
        print(f"{log_prefix} Recovery error for {db_path.name}: {e}")
        return False


def check_and_repair_databases(
    data_dir: Path,
    auto_repair: bool = True,
    min_size_bytes: int = 1024 * 1024,  # Only check DBs > 1MB
    recursive: bool = False,
    log_prefix: str = "[DBIntegrity]"
) -> dict[str, Any]:
    """Scan and repair corrupted SQLite databases.

    Args:
        data_dir: Directory to scan for .db files
        auto_repair: Whether to attempt automatic recovery (True) or just move (False)
        min_size_bytes: Minimum file size to check (smaller files are skipped)
        recursive: Whether to scan subdirectories (using rglob)
        log_prefix: Prefix for log messages

    Returns:
        Dict with counts: checked, healthy, corrupted, recovered, failed, corrupted_files
    """
    results = {
        "checked": 0,
        "healthy": 0,
        "corrupted": 0,
        "recovered": 0,
        "failed": 0,
        "corrupted_files": [],
    }

    if not data_dir.exists():
        return results

    # Choose glob pattern based on recursive flag
    db_files = data_dir.rglob("*.db") if recursive else data_dir.glob("*.db")

    for db_path in db_files:
        # Skip small files (likely empty/test DBs)
        try:
            if db_path.stat().st_size < min_size_bytes:
                continue
        except OSError:
            continue

        # Skip files in corrupted directories
        if "corrupted" in str(db_path):
            continue

        results["checked"] += 1
        is_healthy, error_msg = check_database_integrity(db_path)

        if is_healthy:
            results["healthy"] += 1
        else:
            results["corrupted"] += 1
            results["corrupted_files"].append(str(db_path))
            print(f"{log_prefix} CORRUPTED: {db_path.name} - {error_msg}")

            if auto_repair:
                if recover_corrupted_database(db_path, log_prefix=log_prefix):
                    results["recovered"] += 1
                else:
                    results["failed"] += 1
            else:
                # Just move to corrupted directory without recovery attempt
                corrupted_dir = db_path.parent / "corrupted"
                corrupted_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = corrupted_dir / f"{db_path.stem}_{timestamp}.db.corrupted"
                try:
                    db_path.rename(backup_path)
                    results["failed"] += 1  # Counts as failed since no recovery attempted
                    print(f"{log_prefix} Moved corrupted DB to {backup_path}")
                except Exception as e:
                    print(f"{log_prefix} Failed to move corrupted DB: {e}")

    return results


def get_database_stats(db_path: Path) -> dict[str, Any] | None:
    """Get statistics about a database file.

    Args:
        db_path: Path to the SQLite database

    Returns:
        Dict with stats or None if database cannot be read
    """
    try:
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        cursor = conn.cursor()

        stats = {
            "path": str(db_path),
            "size_mb": db_path.stat().st_size / (1024 * 1024),
            "tables": {},
        }

        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                stats["tables"][table] = count
            except (sqlite3.Error, OSError) as e:
                stats["tables"][table] = -1  # Error reading table

        conn.close()
        return stats

    except (sqlite3.Error, OSError) as e:
        return None
