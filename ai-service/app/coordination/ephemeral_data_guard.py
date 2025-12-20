#!/usr/bin/env python3
"""Ephemeral Data Guard - Insurance against data loss from ephemeral hosts.

This module provides safeguards for data on ephemeral hosts (like Vast.ai)
that can terminate without warning:

1. Checkpoint markers - Write progress markers to persistent storage
2. Write-through caching - Immediately push critical games to persistent hosts
3. Heartbeat monitoring - Detect when ephemeral hosts go silent
4. Priority evacuation - Trigger urgent sync when host becomes unstable

Usage:
    from app.coordination.ephemeral_data_guard import (
        EphemeralDataGuard,
        get_ephemeral_guard,
        checkpoint_games,
        request_evacuation,
        is_host_ephemeral,
    )

    # On ephemeral host - checkpoint after generating games
    guard = get_ephemeral_guard()
    guard.checkpoint(games_generated=100, last_game_id="abc123")

    # On coordinator - check for hosts needing evacuation
    hosts = guard.get_evacuation_candidates()
    for host in hosts:
        trigger_emergency_sync(host)
"""

from __future__ import annotations

import json
import logging
import socket
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default paths
from app.utils.paths import DATA_DIR

DEFAULT_GUARD_DB = DATA_DIR / "coordination" / "ephemeral_guard.db"

# Thresholds
CHECKPOINT_INTERVAL = 60  # Checkpoint every 60 seconds
HEARTBEAT_TIMEOUT = 120  # Consider host dead after 2 minutes without heartbeat
EVACUATION_THRESHOLD = 50  # Trigger evacuation if >50 unsynced games and no heartbeat
CRITICAL_GAME_THRESHOLD = 10  # Games this important get immediate write-through

# Known ephemeral host patterns
EPHEMERAL_HOST_PATTERNS = [
    "vast",
    "spot",
    "preemptible",
    "ephemeral",
]


@dataclass
class HostCheckpoint:
    """Checkpoint state for a host."""
    host: str
    is_ephemeral: bool
    last_checkpoint_time: float
    last_heartbeat_time: float
    games_generated: int
    games_synced: int
    last_game_id: str
    checkpoint_data: dict[str, Any] = field(default_factory=dict)

    @property
    def unsynced_games(self) -> int:
        return max(0, self.games_generated - self.games_synced)

    @property
    def seconds_since_heartbeat(self) -> float:
        return time.time() - self.last_heartbeat_time if self.last_heartbeat_time > 0 else float('inf')

    @property
    def needs_evacuation(self) -> bool:
        """Check if this host needs emergency data evacuation."""
        if not self.is_ephemeral:
            return False
        if self.unsynced_games < EVACUATION_THRESHOLD:
            return False
        return not self.seconds_since_heartbeat < HEARTBEAT_TIMEOUT

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "is_ephemeral": self.is_ephemeral,
            "last_checkpoint": datetime.fromtimestamp(self.last_checkpoint_time).isoformat() if self.last_checkpoint_time > 0 else None,
            "last_heartbeat": datetime.fromtimestamp(self.last_heartbeat_time).isoformat() if self.last_heartbeat_time > 0 else None,
            "seconds_since_heartbeat": round(self.seconds_since_heartbeat, 1),
            "games_generated": self.games_generated,
            "games_synced": self.games_synced,
            "unsynced_games": self.unsynced_games,
            "needs_evacuation": self.needs_evacuation,
        }


@dataclass
class WriteThrough:
    """A write-through request for critical data."""
    game_id: str
    host: str
    priority: int  # Higher = more important
    created_at: float
    data_path: str
    synced: bool = False
    synced_at: float = 0.0


class EphemeralDataGuard:
    """Guards against data loss from ephemeral hosts.

    This class provides:
    1. Checkpoint tracking for ephemeral hosts
    2. Write-through queue for critical games
    3. Evacuation detection and triggering
    """

    _instance: EphemeralDataGuard | None = None
    _lock = threading.RLock()

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DEFAULT_GUARD_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._checkpoints: dict[str, HostCheckpoint] = {}
        self._write_through_queue: list[WriteThrough] = []
        self._init_db()
        self._load_state()

    @classmethod
    def get_instance(cls, db_path: Path | None = None) -> EphemeralDataGuard:
        """Get or create singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(db_path)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton for testing."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._save_state()
            cls._instance = None

    # =========================================================================
    # Database Management
    # =========================================================================

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                timeout=10,
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Host checkpoints table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS host_checkpoints (
                host TEXT PRIMARY KEY,
                is_ephemeral INTEGER DEFAULT 0,
                last_checkpoint_time REAL DEFAULT 0,
                last_heartbeat_time REAL DEFAULT 0,
                games_generated INTEGER DEFAULT 0,
                games_synced INTEGER DEFAULT 0,
                last_game_id TEXT DEFAULT '',
                checkpoint_data TEXT DEFAULT '{}'
            )
        """)

        # Write-through queue
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS write_through_queue (
                game_id TEXT PRIMARY KEY,
                host TEXT NOT NULL,
                priority INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                data_path TEXT NOT NULL,
                synced INTEGER DEFAULT 0,
                synced_at REAL DEFAULT 0
            )
        """)

        # Evacuation history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evacuation_history (
                evacuation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                host TEXT NOT NULL,
                triggered_at REAL NOT NULL,
                games_at_risk INTEGER DEFAULT 0,
                games_recovered INTEGER DEFAULT 0,
                completed_at REAL DEFAULT 0,
                success INTEGER DEFAULT 0
            )
        """)

        conn.commit()

    def _load_state(self) -> None:
        """Load state from database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM host_checkpoints")
        for row in cursor.fetchall():
            host = row["host"]
            self._checkpoints[host] = HostCheckpoint(
                host=host,
                is_ephemeral=bool(row["is_ephemeral"]),
                last_checkpoint_time=row["last_checkpoint_time"] or 0,
                last_heartbeat_time=row["last_heartbeat_time"] or 0,
                games_generated=row["games_generated"] or 0,
                games_synced=row["games_synced"] or 0,
                last_game_id=row["last_game_id"] or "",
                checkpoint_data=json.loads(row["checkpoint_data"] or "{}"),
            )

        logger.info(f"[EphemeralGuard] Loaded checkpoints for {len(self._checkpoints)} hosts")

    def _save_checkpoint(self, checkpoint: HostCheckpoint) -> None:
        """Save a single checkpoint to database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO host_checkpoints
            (host, is_ephemeral, last_checkpoint_time, last_heartbeat_time,
             games_generated, games_synced, last_game_id, checkpoint_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            checkpoint.host,
            1 if checkpoint.is_ephemeral else 0,
            checkpoint.last_checkpoint_time,
            checkpoint.last_heartbeat_time,
            checkpoint.games_generated,
            checkpoint.games_synced,
            checkpoint.last_game_id,
            json.dumps(checkpoint.checkpoint_data),
        ))
        conn.commit()

    # =========================================================================
    # Checkpoint Management
    # =========================================================================

    def checkpoint(
        self,
        host: str | None = None,
        games_generated: int | None = None,
        games_synced: int | None = None,
        last_game_id: str | None = None,
        extra_data: dict[str, Any] | None = None,
    ) -> HostCheckpoint:
        """Record a checkpoint for a host.

        Call this periodically (every CHECKPOINT_INTERVAL) from selfplay workers.

        Args:
            host: Host name (defaults to current hostname)
            games_generated: Total games generated on this host
            games_synced: Total games successfully synced
            last_game_id: ID of the most recent game
            extra_data: Additional checkpoint data

        Returns:
            Updated checkpoint state
        """
        if host is None:
            host = socket.gethostname()

        now = time.time()

        if host not in self._checkpoints:
            is_ephemeral = self._is_ephemeral_host(host)
            self._checkpoints[host] = HostCheckpoint(
                host=host,
                is_ephemeral=is_ephemeral,
                last_checkpoint_time=now,
                last_heartbeat_time=now,
                games_generated=0,
                games_synced=0,
                last_game_id="",
            )

        checkpoint = self._checkpoints[host]
        checkpoint.last_checkpoint_time = now
        checkpoint.last_heartbeat_time = now

        if games_generated is not None:
            checkpoint.games_generated = games_generated
        if games_synced is not None:
            checkpoint.games_synced = games_synced
        if last_game_id is not None:
            checkpoint.last_game_id = last_game_id
        if extra_data:
            checkpoint.checkpoint_data.update(extra_data)

        self._save_checkpoint(checkpoint)

        # Log if ephemeral with significant unsynced data
        if checkpoint.is_ephemeral and checkpoint.unsynced_games > 20:
            logger.warning(f"[EphemeralGuard] {host} has {checkpoint.unsynced_games} unsynced games")

        return checkpoint

    def heartbeat(self, host: str | None = None) -> None:
        """Record a heartbeat for a host (lighter than full checkpoint)."""
        if host is None:
            host = socket.gethostname()

        now = time.time()

        if host in self._checkpoints:
            self._checkpoints[host].last_heartbeat_time = now
            # Batch heartbeat updates - don't write to DB on every heartbeat
        else:
            # Create minimal checkpoint
            self.checkpoint(host=host)

    def record_sync_complete(self, host: str, games_synced: int) -> None:
        """Record that games were successfully synced from a host."""
        if host in self._checkpoints:
            self._checkpoints[host].games_synced = games_synced
            self._save_checkpoint(self._checkpoints[host])

    def _is_ephemeral_host(self, host: str) -> bool:
        """Check if a host is ephemeral based on naming patterns."""
        host_lower = host.lower()
        return any(pattern in host_lower for pattern in EPHEMERAL_HOST_PATTERNS)

    # =========================================================================
    # Write-Through Queue
    # =========================================================================

    def queue_write_through(
        self,
        game_id: str,
        host: str,
        data_path: str,
        priority: int = 0,
    ) -> None:
        """Queue a game for immediate write-through to persistent storage.

        Use this for critical games that must not be lost.
        """
        conn = self._get_connection()
        now = time.time()

        conn.execute("""
            INSERT OR REPLACE INTO write_through_queue
            (game_id, host, priority, created_at, data_path, synced, synced_at)
            VALUES (?, ?, ?, ?, ?, 0, 0)
        """, (game_id, host, priority, now, data_path))
        conn.commit()

        logger.info(f"[EphemeralGuard] Queued write-through for game {game_id} from {host}")

    def get_pending_write_throughs(self, limit: int = 100) -> list[WriteThrough]:
        """Get pending write-through requests ordered by priority."""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT * FROM write_through_queue
            WHERE synced = 0
            ORDER BY priority DESC, created_at ASC
            LIMIT ?
        """, (limit,))

        return [
            WriteThrough(
                game_id=row["game_id"],
                host=row["host"],
                priority=row["priority"],
                created_at=row["created_at"],
                data_path=row["data_path"],
                synced=bool(row["synced"]),
                synced_at=row["synced_at"],
            )
            for row in cursor.fetchall()
        ]

    def mark_write_through_complete(self, game_id: str) -> None:
        """Mark a write-through as completed."""
        conn = self._get_connection()
        conn.execute("""
            UPDATE write_through_queue
            SET synced = 1, synced_at = ?
            WHERE game_id = ?
        """, (time.time(), game_id))
        conn.commit()

    # =========================================================================
    # Evacuation Management
    # =========================================================================

    def get_evacuation_candidates(self) -> list[HostCheckpoint]:
        """Get hosts that need emergency data evacuation."""
        candidates = []
        for checkpoint in self._checkpoints.values():
            if checkpoint.needs_evacuation:
                candidates.append(checkpoint)

        # Sort by unsynced games (most at risk first)
        candidates.sort(key=lambda c: c.unsynced_games, reverse=True)
        return candidates

    def request_evacuation(self, host: str) -> int:
        """Request emergency evacuation for a host.

        Returns:
            Evacuation ID for tracking
        """
        conn = self._get_connection()
        now = time.time()

        checkpoint = self._checkpoints.get(host)
        games_at_risk = checkpoint.unsynced_games if checkpoint else 0

        cursor = conn.execute("""
            INSERT INTO evacuation_history
            (host, triggered_at, games_at_risk)
            VALUES (?, ?, ?)
        """, (host, now, games_at_risk))
        conn.commit()

        evacuation_id = cursor.lastrowid
        logger.warning(f"[EphemeralGuard] Evacuation requested for {host}: {games_at_risk} games at risk (id={evacuation_id})")

        return evacuation_id

    def complete_evacuation(
        self,
        evacuation_id: int,
        games_recovered: int,
        success: bool,
    ) -> None:
        """Record evacuation completion."""
        conn = self._get_connection()
        conn.execute("""
            UPDATE evacuation_history
            SET completed_at = ?, games_recovered = ?, success = ?
            WHERE evacuation_id = ?
        """, (time.time(), games_recovered, 1 if success else 0, evacuation_id))
        conn.commit()

        logger.info(f"[EphemeralGuard] Evacuation {evacuation_id} completed: {games_recovered} games recovered, success={success}")

    # =========================================================================
    # Status and Monitoring
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """Get overall ephemeral guard status."""
        ephemeral_hosts = [c for c in self._checkpoints.values() if c.is_ephemeral]
        evacuation_candidates = self.get_evacuation_candidates()
        pending_write_throughs = len(self.get_pending_write_throughs(limit=1000))

        total_unsynced = sum(c.unsynced_games for c in ephemeral_hosts)

        return {
            "ephemeral_hosts": len(ephemeral_hosts),
            "total_unsynced_games": total_unsynced,
            "evacuation_candidates": [c.host for c in evacuation_candidates],
            "pending_write_throughs": pending_write_throughs,
            "hosts": {
                host: checkpoint.to_dict()
                for host, checkpoint in self._checkpoints.items()
            },
        }

    def get_checkpoint(self, host: str) -> HostCheckpoint | None:
        """Get checkpoint for a specific host."""
        return self._checkpoints.get(host)


# =============================================================================
# Module-level convenience functions
# =============================================================================

def get_ephemeral_guard() -> EphemeralDataGuard:
    """Get the singleton ephemeral data guard."""
    return EphemeralDataGuard.get_instance()


def checkpoint_games(
    host: str | None = None,
    games_generated: int | None = None,
    games_synced: int | None = None,
    last_game_id: str | None = None,
) -> HostCheckpoint:
    """Record a checkpoint for games generated."""
    return get_ephemeral_guard().checkpoint(
        host=host,
        games_generated=games_generated,
        games_synced=games_synced,
        last_game_id=last_game_id,
    )


def ephemeral_heartbeat(host: str | None = None) -> None:
    """Record a heartbeat for a host."""
    get_ephemeral_guard().heartbeat(host)


def is_host_ephemeral(host: str) -> bool:
    """Check if a host is ephemeral."""
    return get_ephemeral_guard()._is_ephemeral_host(host)


def get_evacuation_candidates() -> list[HostCheckpoint]:
    """Get hosts needing emergency evacuation."""
    return get_ephemeral_guard().get_evacuation_candidates()


def request_evacuation(host: str) -> int:
    """Request emergency evacuation for a host."""
    return get_ephemeral_guard().request_evacuation(host)


def queue_critical_game(game_id: str, host: str, data_path: str) -> None:
    """Queue a critical game for immediate write-through."""
    get_ephemeral_guard().queue_write_through(
        game_id=game_id,
        host=host,
        data_path=data_path,
        priority=CRITICAL_GAME_THRESHOLD,
    )


def reset_ephemeral_guard() -> None:
    """Reset the singleton."""
    EphemeralDataGuard.reset_instance()


def wire_ephemeral_guard_events() -> EphemeralDataGuard:
    """Wire ephemeral guard to the event bus for host monitoring.

    Subscribes to:
    - HOST_OFFLINE: Trigger evacuation check for the host
    - HOST_ONLINE: Update host status and clear evacuation flags
    - TASK_HEARTBEAT: Update heartbeat for ephemeral hosts

    Returns:
        The configured EphemeralDataGuard instance
    """
    guard = get_ephemeral_guard()

    try:
        from app.distributed.data_events import DataEventType, get_event_bus

        bus = get_event_bus()

        def _event_payload(event: Any) -> dict[str, Any]:
            if isinstance(event, dict):
                return event
            payload = getattr(event, "payload", None)
            return payload if isinstance(payload, dict) else {}

        def _on_host_offline(event: Any) -> None:
            """Handle host going offline - check for evacuation."""
            payload = _event_payload(event)
            host = payload.get("host") or payload.get("node_id")
            if not host:
                return
            checkpoint = guard.get_checkpoint(host)
            if checkpoint and checkpoint.is_ephemeral and checkpoint.unsynced_games > 0:
                logger.warning(
                    f"[EphemeralGuard] Host {host} offline with {checkpoint.unsynced_games} "
                    f"unsynced games, requesting evacuation"
                )
                guard.request_evacuation(host)

        def _on_host_online(event: Any) -> None:
            """Handle host coming online - update heartbeat."""
            payload = _event_payload(event)
            host = payload.get("host") or payload.get("node_id")
            if host:
                guard.heartbeat(host)

        def _on_task_heartbeat(event: Any) -> None:
            """Handle task heartbeat - update host heartbeat."""
            payload = _event_payload(event)
            host = payload.get("host") or payload.get("node_id")
            if host:
                guard.heartbeat(host)

        bus.subscribe(DataEventType.HOST_OFFLINE, _on_host_offline)
        bus.subscribe(DataEventType.HOST_ONLINE, _on_host_online)
        bus.subscribe(DataEventType.TASK_HEARTBEAT, _on_task_heartbeat)

        logger.info("[EphemeralGuard] Wired to event bus (HOST_OFFLINE, HOST_ONLINE, TASK_HEARTBEAT)")

    except ImportError:
        logger.warning("[EphemeralGuard] data_events not available, running without event bus")

    return guard


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Constants
    "CHECKPOINT_INTERVAL",
    "CRITICAL_GAME_THRESHOLD",
    "EPHEMERAL_HOST_PATTERNS",
    "EVACUATION_THRESHOLD",
    "HEARTBEAT_TIMEOUT",
    # Main class
    "EphemeralDataGuard",
    # Data classes
    "HostCheckpoint",
    "WriteThrough",
    "checkpoint_games",
    "ephemeral_heartbeat",
    # Functions
    "get_ephemeral_guard",
    "get_evacuation_candidates",
    "is_host_ephemeral",
    "queue_critical_game",
    "request_evacuation",
    "reset_ephemeral_guard",
    "wire_ephemeral_guard_events",
]
