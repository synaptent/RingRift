"""Integrity Check Daemon for RingRift Data Quality.

This daemon periodically scans game databases for integrity issues,
specifically focusing on games without move data (orphan games).

Features:
- Periodic scanning of all game databases
- Detection of games without move data
- Quarantine of invalid games for later review
- Cleanup of quarantined games older than threshold
- Event emission for monitoring

December 2025: Created for Phase 6 of Move Data Integrity Enforcement.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.coordination.base_daemon import BaseDaemon, DaemonConfig
from app.coordination.protocols import HealthCheckResult

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class IntegrityCheckConfig(DaemonConfig):
    """Configuration for IntegrityCheckDaemon.

    Attributes:
        check_interval_seconds: How often to scan for integrity issues (default: 1 hour)
        data_dir: Directory containing game databases
        quarantine_after_days: Days to keep quarantined games before cleanup (default: 7)
        max_orphans_per_scan: Max orphan games to process per scan (default: 1000)
        emit_events: Whether to emit events for monitoring
    """

    check_interval_seconds: int = 3600  # 1 hour
    data_dir: str = ""
    quarantine_after_days: int = 7
    max_orphans_per_scan: int = 1000
    emit_events: bool = True

    @classmethod
    def from_env(cls, prefix: str = "RINGRIFT_INTEGRITY") -> "IntegrityCheckConfig":
        """Load configuration from environment variables."""
        config = cls()

        # Call parent for common env vars
        parent_config = super().from_env(prefix)
        config.enabled = parent_config.enabled
        config.check_interval_seconds = parent_config.check_interval_seconds
        config.handle_signals = parent_config.handle_signals

        # Daemon-specific env vars
        if os.environ.get(f"{prefix}_DATA_DIR"):
            config.data_dir = os.environ.get(f"{prefix}_DATA_DIR", "")

        if os.environ.get(f"{prefix}_QUARANTINE_DAYS"):
            try:
                config.quarantine_after_days = int(
                    os.environ.get(f"{prefix}_QUARANTINE_DAYS", "7")
                )
            except ValueError:
                pass

        if os.environ.get(f"{prefix}_MAX_ORPHANS"):
            try:
                config.max_orphans_per_scan = int(
                    os.environ.get(f"{prefix}_MAX_ORPHANS", "1000")
                )
            except ValueError:
                pass

        return config


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class OrphanGame:
    """Information about an orphan game (game without move data)."""

    game_id: str
    db_path: str
    board_type: str
    num_players: int
    total_moves: int  # Claimed moves in games table
    created_at: str
    game_status: str


@dataclass
class IntegrityCheckResult:
    """Result of an integrity check scan."""

    scan_time: str = ""
    databases_scanned: int = 0
    orphan_games_found: int = 0
    orphan_games_quarantined: int = 0
    orphan_games_cleaned: int = 0
    errors: list[str] = field(default_factory=list)
    details_by_db: dict[str, int] = field(default_factory=dict)


# =============================================================================
# Daemon Implementation
# =============================================================================


class IntegrityCheckDaemon(BaseDaemon[IntegrityCheckConfig]):
    """Periodic data integrity validation daemon.

    Scans game databases for games without move data, quarantines them,
    and cleans up old quarantined games.
    """

    DAEMON_NAME = "integrity_check"

    def __init__(self, config: IntegrityCheckConfig | None = None):
        """Initialize IntegrityCheckDaemon.

        Args:
            config: Configuration. If None, loads from environment.
        """
        if config is None:
            config = IntegrityCheckConfig.from_env()

        super().__init__(config)

        # Default data directory
        if not config.data_dir:
            config.data_dir = str(
                Path(__file__).parent.parent.parent / "data" / "games"
            )

        self._last_result: IntegrityCheckResult | None = None
        self._total_orphans_found = 0
        self._total_orphans_cleaned = 0

    async def _run_cycle(self) -> None:
        """Run one integrity check cycle."""
        logger.info("Starting integrity check cycle...")

        result = IntegrityCheckResult(
            scan_time=datetime.now(timezone.utc).isoformat()
        )

        try:
            # Find all game databases
            databases = self._find_databases()
            result.databases_scanned = len(databases)

            if not databases:
                logger.warning(f"No databases found in {self._config.data_dir}")
                self._last_result = result
                return

            # Check each database for orphan games
            for db_path in databases:
                try:
                    orphans = await self._check_database(db_path)
                    result.details_by_db[str(db_path)] = len(orphans)

                    if orphans:
                        result.orphan_games_found += len(orphans)
                        logger.warning(
                            f"Found {len(orphans)} orphan games in {db_path.name}"
                        )

                        # Quarantine orphan games
                        quarantined = await self._quarantine_orphans(db_path, orphans)
                        result.orphan_games_quarantined += quarantined

                except Exception as e:
                    error_msg = f"Error checking {db_path.name}: {e}"
                    result.errors.append(error_msg)
                    logger.error(error_msg)

            # Cleanup old quarantined games
            for db_path in databases:
                try:
                    cleaned = await self._cleanup_quarantine(db_path)
                    result.orphan_games_cleaned += cleaned
                except Exception as e:
                    error_msg = f"Error cleaning quarantine in {db_path.name}: {e}"
                    result.errors.append(error_msg)
                    logger.error(error_msg)

            # Update totals
            self._total_orphans_found += result.orphan_games_found
            self._total_orphans_cleaned += result.orphan_games_cleaned

            # Emit event if configured
            if self._config.emit_events and result.orphan_games_found > 0:
                await self._emit_integrity_event(result)

            logger.info(
                f"Integrity check complete: "
                f"{result.databases_scanned} DBs scanned, "
                f"{result.orphan_games_found} orphans found, "
                f"{result.orphan_games_quarantined} quarantined, "
                f"{result.orphan_games_cleaned} cleaned"
            )

        except Exception as e:
            result.errors.append(f"Cycle error: {e}")
            logger.exception("Error in integrity check cycle")

        self._last_result = result

    def _find_databases(self) -> list[Path]:
        """Find all game databases to check."""
        data_dir = Path(self._config.data_dir)
        if not data_dir.exists():
            return []

        databases = []
        for pattern in ["*.db", "**/*.db"]:
            databases.extend(data_dir.glob(pattern))

        # Filter out non-game databases
        databases = [
            db
            for db in databases
            if "jsonl" not in db.name
            and "sync" not in db.name
            and "elo" not in db.name
            and "registry" not in db.name
        ]

        return list(set(databases))

    async def _check_database(self, db_path: Path) -> list[OrphanGame]:
        """Check a database for orphan games.

        Args:
            db_path: Path to database

        Returns:
            List of OrphanGame objects
        """
        orphans: list[OrphanGame] = []

        def _check_sync() -> list[OrphanGame]:
            conn = sqlite3.connect(str(db_path))

            # Check if game_moves table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'"
            )
            has_moves_table = cursor.fetchone() is not None

            if not has_moves_table:
                # No moves table means all games might be orphans, but we can't verify
                conn.close()
                return []

            # Find games with no moves
            cursor = conn.execute(
                """
                SELECT g.game_id, g.board_type, g.num_players, g.total_moves,
                       g.created_at, g.game_status
                FROM games g
                LEFT JOIN game_moves m ON g.game_id = m.game_id
                WHERE m.game_id IS NULL
                LIMIT ?
            """,
                (self._config.max_orphans_per_scan,),
            )

            results = []
            for row in cursor:
                results.append(
                    OrphanGame(
                        game_id=row[0],
                        db_path=str(db_path),
                        board_type=row[1],
                        num_players=row[2],
                        total_moves=row[3],
                        created_at=row[4],
                        game_status=row[5],
                    )
                )

            conn.close()
            return results

        # Run sync DB operation in executor to avoid blocking
        loop = asyncio.get_event_loop()
        orphans = await loop.run_in_executor(None, _check_sync)

        return orphans

    async def _quarantine_orphans(
        self, db_path: Path, orphans: list[OrphanGame]
    ) -> int:
        """Move orphan games to quarantine table.

        Args:
            db_path: Path to database
            orphans: List of orphan games to quarantine

        Returns:
            Number of games quarantined
        """

        def _quarantine_sync() -> int:
            conn = sqlite3.connect(str(db_path))

            # Create quarantine table if not exists
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS orphaned_games (
                    game_id TEXT PRIMARY KEY,
                    detected_at TEXT NOT NULL,
                    reason TEXT,
                    original_status TEXT,
                    board_type TEXT,
                    num_players INTEGER
                )
            """
            )

            quarantined = 0
            for orphan in orphans:
                try:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO orphaned_games
                        (game_id, detected_at, reason, original_status, board_type, num_players)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            orphan.game_id,
                            datetime.now(timezone.utc).isoformat(),
                            "No move data found",
                            orphan.game_status,
                            orphan.board_type,
                            orphan.num_players,
                        ),
                    )
                    quarantined += 1
                except sqlite3.Error:
                    pass

            conn.commit()
            conn.close()
            return quarantined

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _quarantine_sync)

    async def _cleanup_quarantine(self, db_path: Path) -> int:
        """Clean up old quarantined games.

        Args:
            db_path: Path to database

        Returns:
            Number of games cleaned up
        """

        def _cleanup_sync() -> int:
            conn = sqlite3.connect(str(db_path))

            # Check if quarantine table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='orphaned_games'"
            )
            if not cursor.fetchone():
                conn.close()
                return 0

            # Find games to clean up (older than threshold)
            cutoff = datetime.now(timezone.utc).isoformat()
            # Approximate days by string comparison (ISO format works for this)
            cursor = conn.execute(
                """
                SELECT game_id FROM orphaned_games
                WHERE detected_at < datetime('now', ?)
            """,
                (f"-{self._config.quarantine_after_days} days",),
            )

            game_ids = [row[0] for row in cursor.fetchall()]

            if not game_ids:
                conn.close()
                return 0

            # Delete from games table and all related tables
            RELATED_TABLES = [
                "game_moves",
                "game_initial_state",
                "game_state_snapshots",
                "game_players",
                "game_choices",
                "game_history_entries",
                "game_nnue_features",
                "games",
                "orphaned_games",
            ]

            cleaned = 0
            for table in RELATED_TABLES:
                try:
                    cursor = conn.execute(
                        f"SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                        (table,),
                    )
                    if cursor.fetchone():
                        placeholders = ",".join(["?" for _ in game_ids])
                        conn.execute(
                            f"DELETE FROM {table} WHERE game_id IN ({placeholders})",
                            game_ids,
                        )
                except sqlite3.Error:
                    pass

            cleaned = len(game_ids)
            conn.commit()
            conn.close()

            return cleaned

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _cleanup_sync)

    async def _emit_integrity_event(self, result: IntegrityCheckResult) -> None:
        """Emit event about integrity issues found."""
        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()
            await bus.publish(
                DataEventType.DATA_QUALITY_DEGRADED,
                {
                    "source": "IntegrityCheckDaemon",
                    "event_type": "orphan_games_detected",
                    "orphan_games_found": result.orphan_games_found,
                    "databases_scanned": result.databases_scanned,
                    "details": result.details_by_db,
                    "scan_time": result.scan_time,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit integrity event: {e}")

    def health_check(self) -> HealthCheckResult:
        """Return daemon health status."""
        is_healthy = self._running and not self._stopped

        details = {
            "running": self._running,
            "stopped": self._stopped,
            "total_orphans_found": self._total_orphans_found,
            "total_orphans_cleaned": self._total_orphans_cleaned,
            "data_dir": self._config.data_dir,
        }

        if self._last_result:
            details["last_scan"] = self._last_result.scan_time
            details["last_orphans_found"] = self._last_result.orphan_games_found
            details["last_errors"] = len(self._last_result.errors)

        return HealthCheckResult(
            is_healthy=is_healthy,
            status="healthy" if is_healthy else "unhealthy",
            message="Integrity check daemon running" if is_healthy else "Daemon not running",
            details=details,
        )


# =============================================================================
# Singleton Access
# =============================================================================

_instance: IntegrityCheckDaemon | None = None


def get_integrity_check_daemon() -> IntegrityCheckDaemon:
    """Get or create the singleton IntegrityCheckDaemon instance."""
    global _instance
    if _instance is None:
        _instance = IntegrityCheckDaemon()
    return _instance


def reset_integrity_check_daemon() -> None:
    """Reset the singleton instance (for testing)."""
    global _instance
    _instance = None
