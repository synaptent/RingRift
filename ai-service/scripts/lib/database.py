"""Database utilities for scripts.

Provides consistent, safe database connection patterns for training scripts.

Usage:
    from scripts.lib.database import (
        safe_transaction,
        get_game_db_path,
        get_elo_db_path,
        count_games,
    )

    # Safe transaction with automatic rollback
    with safe_transaction(db_path) as conn:
        conn.execute("INSERT INTO games ...")

    # Get standard database paths
    games_db = get_game_db_path("square8_2p")
    elo_db = get_elo_db_path()
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Union

logger = logging.getLogger(__name__)

# Standard database paths
_AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _AI_SERVICE_ROOT / "data"
GAMES_DIR = DATA_DIR / "games"
UNIFIED_ELO_DB = DATA_DIR / "unified_elo.db"


@contextmanager
def safe_transaction(
    db_path: Union[str, Path],
    timeout: float = 30.0,
    row_factory: bool = True,
) -> Generator[sqlite3.Connection, None, None]:
    """Execute a SQLite transaction with automatic rollback on error.

    Args:
        db_path: Path to SQLite database
        timeout: Lock acquisition timeout in seconds
        row_factory: If True, use sqlite3.Row for dict-like access

    Yields:
        SQLite connection with active transaction

    Example:
        with safe_transaction("data.db") as conn:
            conn.execute("INSERT INTO games ...")
            conn.execute("UPDATE stats SET ...")
    """
    conn = sqlite3.connect(str(db_path), timeout=timeout)
    if row_factory:
        conn.row_factory = sqlite3.Row

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def read_only_connection(
    db_path: Union[str, Path],
    timeout: float = 30.0,
) -> Generator[sqlite3.Connection, None, None]:
    """Open a read-only database connection.

    Args:
        db_path: Path to SQLite database
        timeout: Lock acquisition timeout in seconds

    Yields:
        SQLite connection for reading
    """
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=timeout)
    conn.row_factory = sqlite3.Row

    try:
        yield conn
    finally:
        conn.close()


def get_game_db_path(config_key: str) -> Path:
    """Get the standard game database path for a config.

    Args:
        config_key: Board config key (e.g., "square8_2p", "hex7_3p")

    Returns:
        Path to the selfplay database for this config
    """
    return GAMES_DIR / f"selfplay_{config_key}.db"


def get_elo_db_path() -> Path:
    """Get the unified ELO database path."""
    return UNIFIED_ELO_DB


def count_games(db_path: Union[str, Path], config_key: Optional[str] = None) -> int:
    """Count games in a database.

    Args:
        db_path: Path to game database
        config_key: Optional config key filter

    Returns:
        Number of games in database
    """
    try:
        with read_only_connection(db_path) as conn:
            if config_key:
                result = conn.execute(
                    "SELECT COUNT(*) FROM games WHERE config_key = ?",
                    (config_key,)
                ).fetchone()
            else:
                result = conn.execute("SELECT COUNT(*) FROM games").fetchone()
            return result[0] if result else 0
    except (sqlite3.Error, OSError) as e:
        logger.warning(f"Failed to count games in {db_path}: {e}")
        return 0


def table_exists(db_path: Union[str, Path], table_name: str) -> bool:
    """Check if a table exists in the database.

    Args:
        db_path: Path to database
        table_name: Name of table to check

    Returns:
        True if table exists
    """
    try:
        with read_only_connection(db_path) as conn:
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            ).fetchone()
            return result is not None
    except (sqlite3.Error, OSError):
        return False


def get_db_size_mb(db_path: Union[str, Path]) -> float:
    """Get database file size in MB.

    Args:
        db_path: Path to database

    Returns:
        Size in megabytes, or 0 if file doesn't exist
    """
    path = Path(db_path)
    if path.exists():
        return path.stat().st_size / (1024 * 1024)
    return 0.0


def vacuum_database(db_path: Union[str, Path]) -> bool:
    """Vacuum a database to reclaim space.

    Args:
        db_path: Path to database

    Returns:
        True if vacuum succeeded
    """
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("VACUUM")
        conn.close()
        return True
    except sqlite3.Error as e:
        logger.error(f"Failed to vacuum {db_path}: {e}")
        return False


def check_integrity(db_path: Union[str, Path]) -> tuple[bool, str]:
    """Check database integrity.

    Args:
        db_path: Path to database

    Returns:
        Tuple of (is_healthy, message)
    """
    try:
        with read_only_connection(db_path) as conn:
            result = conn.execute("PRAGMA integrity_check").fetchone()
            if result[0] == "ok":
                return True, "OK"
            return False, f"Integrity check failed: {result[0]}"
    except sqlite3.DatabaseError as e:
        return False, f"Database error: {e}"
    except OSError as e:
        return False, f"File error: {e}"
