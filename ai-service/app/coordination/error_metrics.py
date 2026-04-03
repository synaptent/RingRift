"""Lightweight SQLite-backed daemon error metrics.

Records structured error events from HandlerBase._main_loop so the
pipeline watchdog can query error rates without parsing log files.

Schema: CREATE TABLE errors (ts REAL, daemon TEXT, error_class TEXT,
                             msg_hash TEXT, msg_preview TEXT)

Thread-safe, fail-open: SQLite errors are swallowed — metrics must
never crash a daemon.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "error_metrics.db"
_RETENTION_DAYS = 7
_PREVIEW_LEN = 200

_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    """Return (and cache) a thread-safe SQLite connection."""
    global _conn
    if _conn is not None:
        return _conn
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS errors ("
        "  ts REAL, daemon TEXT, error_class TEXT, msg_hash TEXT, msg_preview TEXT)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_errors_ts ON errors(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_errors_daemon ON errors(daemon)")
    # Rotate old rows + VACUUM on startup
    cutoff = time.time() - _RETENTION_DAYS * 86400
    conn.execute("DELETE FROM errors WHERE ts < ?", (cutoff,))
    conn.commit()
    try:
        conn.execute("VACUUM")
    except sqlite3.OperationalError:
        pass  # VACUUM can fail if another connection holds a lock
    _conn = conn
    return conn


def record_error(daemon_name: str, exc: BaseException) -> None:
    """Record a daemon error. Never raises."""
    try:
        conn = _get_conn()
        error_class = type(exc).__qualname__
        msg = str(exc)
        msg_hash = hashlib.md5(msg.encode("utf-8", errors="replace")).hexdigest()[:12]
        preview = msg[:_PREVIEW_LEN]
        conn.execute(
            "INSERT INTO errors (ts, daemon, error_class, msg_hash, msg_preview) "
            "VALUES (?, ?, ?, ?, ?)",
            (time.time(), daemon_name, error_class, msg_hash, preview),
        )
        conn.commit()
    except Exception:
        logger.debug("error_metrics: failed to record error", exc_info=True)


def get_error_rates(hours: float = 1) -> dict[str, int]:
    """Return error count per daemon in the last *hours* hours. Never raises."""
    try:
        conn = _get_conn()
        cutoff = time.time() - hours * 3600
        rows = conn.execute(
            "SELECT daemon, COUNT(*) FROM errors WHERE ts >= ? GROUP BY daemon",
            (cutoff,),
        ).fetchall()
        return {daemon: count for daemon, count in rows}
    except Exception:
        logger.debug("error_metrics: failed to query error rates", exc_info=True)
        return {}


def get_top_errors(hours: float = 1, limit: int = 10) -> list[dict[str, Any]]:
    """Return most frequent error classes in the last *hours*. Never raises."""
    try:
        conn = _get_conn()
        cutoff = time.time() - hours * 3600
        rows = conn.execute(
            "SELECT daemon, error_class, msg_hash, msg_preview, COUNT(*) as cnt "
            "FROM errors WHERE ts >= ? "
            "GROUP BY daemon, error_class, msg_hash "
            "ORDER BY cnt DESC LIMIT ?",
            (cutoff, limit),
        ).fetchall()
        return [
            {"daemon": r[0], "error_class": r[1], "msg_hash": r[2],
             "msg_preview": r[3], "count": r[4]}
            for r in rows
        ]
    except Exception:
        logger.debug("error_metrics: failed to query top errors", exc_info=True)
        return []
