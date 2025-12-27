"""Automatic Data Cleanup Daemon (December 2025).

Automatically cleans up poor quality game databases by:
- Quarantining databases with quality < 30% (recoverable)
- Deleting databases with quality < 10% (non-recoverable, logged)
- Logging all cleanup actions to audit file

Usage:
    from app.coordination.data_cleanup_daemon import DataCleanupDaemon

    daemon = DataCleanupDaemon()
    await daemon.start()
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import socket
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from app.coordination.protocols import (
    CoordinatorStatus,
    HealthCheckResult,
    register_coordinator,
    unregister_coordinator,
)
from app.core.async_context import safe_create_task
# December 2025: Use consolidated daemon stats base class
from app.coordination.daemon_stats import CleanupDaemonStats

logger = logging.getLogger(__name__)

# Import centralized thresholds
try:
    from app.config.thresholds import (
        CLEANUP_MIN_GAMES_BEFORE_DELETE,
        CLEANUP_MOVE_COVERAGE_THRESHOLD,
        CLEANUP_QUALITY_THRESHOLD_DELETE,
        CLEANUP_QUALITY_THRESHOLD_QUARANTINE,
        CLEANUP_SCAN_INTERVAL_SECONDS,
        SYNC_QUALITY_SAMPLE_SIZE,
    )
except ImportError:
    # Fallback if thresholds not available
    CLEANUP_QUALITY_THRESHOLD_DELETE = 0.1
    CLEANUP_QUALITY_THRESHOLD_QUARANTINE = 0.3
    CLEANUP_MOVE_COVERAGE_THRESHOLD = 0.1
    CLEANUP_MIN_GAMES_BEFORE_DELETE = 100
    CLEANUP_SCAN_INTERVAL_SECONDS = 3600
    SYNC_QUALITY_SAMPLE_SIZE = 20


@dataclass
class CleanupConfig:
    """Configuration for data cleanup daemon.

    Thresholds are loaded from app.config.thresholds for centralized configuration.
    """

    enabled: bool = True
    scan_interval_seconds: int = CLEANUP_SCAN_INTERVAL_SECONDS
    # Quality thresholds - from centralized config
    quality_threshold_delete: float = CLEANUP_QUALITY_THRESHOLD_DELETE
    quality_threshold_quarantine: float = CLEANUP_QUALITY_THRESHOLD_QUARANTINE
    move_coverage_threshold: float = CLEANUP_MOVE_COVERAGE_THRESHOLD
    # Sampling
    quality_sample_size: int = SYNC_QUALITY_SAMPLE_SIZE * 2  # More samples for cleanup
    # Directories
    data_dir: Path = field(default_factory=lambda: Path("data/games"))
    quarantine_subdir: str = "quarantine"
    # Safety
    min_games_before_delete: int = CLEANUP_MIN_GAMES_BEFORE_DELETE
    require_canonical_pattern: bool = True  # Only cleanup non-canonical DBs


@dataclass
class DatabaseAssessment:
    """Assessment result for a single database."""

    path: str
    total_games: int
    quality_score: float
    move_coverage: float
    valid_victory_types: float
    issues: list[str] = field(default_factory=list)
    sampled_games: int = 0

    def __post_init__(self):
        if isinstance(self.path, Path):
            self.path = str(self.path)


@dataclass
class CleanupStats(CleanupDaemonStats):
    """Statistics from cleanup operations.

    December 2025: Now extends CleanupDaemonStats for consistent tracking.
    Inherits: items_scanned, items_cleaned, items_quarantined, bytes_reclaimed,
              record_cleanup(), is_healthy(), to_dict(), etc.
    """

    # DataCleanup-specific fields
    databases_deleted: int = 0
    games_quarantined: int = 0
    games_deleted: int = 0

    # Backward compatibility aliases
    @property
    def databases_scanned(self) -> int:
        """Alias for items_scanned (backward compatibility)."""
        return self.items_scanned

    @property
    def databases_quarantined(self) -> int:
        """Alias for items_quarantined (backward compatibility)."""
        return self.items_quarantined

    def record_database_scan(self, scanned: int = 1) -> None:
        """Record databases scanned."""
        self.items_scanned += scanned
        self.last_scan_time = time.time()

    def record_database_quarantine(self, databases: int = 1, games: int = 0) -> None:
        """Record databases quarantined."""
        self.items_quarantined += databases
        if games > 0:
            self.games_quarantined += games

    def record_database_delete(self, databases: int = 1, games: int = 0) -> None:
        """Record databases deleted."""
        self.databases_deleted += databases
        self.items_cleaned += databases
        if games > 0:
            self.games_deleted += games


class DataCleanupDaemon:
    """Daemon that automatically cleans up poor quality game databases.

    Quality thresholds:
    - < 10% quality: Delete (with audit log)
    - < 30% quality: Quarantine (recoverable)
    - < 10% move coverage: Quarantine (likely corrupted)

    All actions are logged to cleanup_audit.jsonl for review.
    """

    def __init__(self, config: CleanupConfig | None = None):
        self.config = config or CleanupConfig()
        self.node_id = socket.gethostname()
        self._running = False
        self._stats = CleanupStats()
        self._scan_task: asyncio.Task | None = None

        # CoordinatorProtocol state
        self._coordinator_status = CoordinatorStatus.INITIALIZING
        self._start_time: float = 0.0
        self._events_processed: int = 0
        self._errors_count: int = 0
        self._last_error: str = ""

        # Resolve directories
        base_dir = Path(__file__).resolve().parent.parent.parent
        self._data_dir = base_dir / self.config.data_dir
        self._quarantine_dir = self._data_dir / self.config.quarantine_subdir
        self._audit_path = self._data_dir / "cleanup_audit.jsonl"

        logger.info(
            f"DataCleanupDaemon initialized: "
            f"delete_threshold={self.config.quality_threshold_delete}, "
            f"quarantine_threshold={self.config.quality_threshold_quarantine}"
        )

    # =========================================================================
    # CoordinatorProtocol Implementation
    # =========================================================================

    @property
    def name(self) -> str:
        """Unique name identifying this coordinator."""
        return "DataCleanupDaemon"

    @property
    def status(self) -> CoordinatorStatus:
        """Current status of the coordinator."""
        return self._coordinator_status

    @property
    def uptime_seconds(self) -> float:
        """Time since daemon started, in seconds."""
        if self._start_time <= 0:
            return 0.0
        return time.time() - self._start_time

    def is_running(self) -> bool:
        """Check if the daemon is running."""
        return self._running

    async def start(self) -> None:
        """Start the cleanup daemon."""
        if not self.config.enabled:
            self._coordinator_status = CoordinatorStatus.STOPPED
            logger.info("DataCleanupDaemon disabled by config")
            return

        if self._coordinator_status == CoordinatorStatus.RUNNING:
            return  # Already running

        self._running = True
        self._coordinator_status = CoordinatorStatus.RUNNING
        self._start_time = time.time()

        # Create quarantine directory
        self._quarantine_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting DataCleanupDaemon on {self.node_id}")

        # December 2025: Subscribe to events for reactive cleanup
        await self._subscribe_to_events()

        # Start scan loop
        self._scan_task = safe_create_task(
            self._scan_loop(),
            name="data_cleanup_loop",
        )

        # Register with coordinator registry
        register_coordinator(self)

        logger.info(
            f"DataCleanupDaemon started: "
            f"interval={self.config.scan_interval_seconds}s, "
            f"data_dir={self._data_dir}"
        )

    async def stop(self) -> None:
        """Stop the cleanup daemon."""
        if self._coordinator_status == CoordinatorStatus.STOPPED:
            return

        self._coordinator_status = CoordinatorStatus.STOPPING
        logger.info("Stopping DataCleanupDaemon...")
        self._running = False

        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass

        unregister_coordinator(self.name)
        self._coordinator_status = CoordinatorStatus.STOPPED
        logger.info("DataCleanupDaemon stopped")

    async def _subscribe_to_events(self) -> None:
        """Subscribe to events for reactive cleanup.

        December 2025: Added event-driven cleanup triggers for better
        integration with the data pipeline. The daemon responds to:
        - DATA_QUALITY_ALERT: Immediate scan when quality issues detected
        - DATA_SYNC_COMPLETED: Scan new data after sync completes
        """
        try:
            from app.coordination.event_router import get_event_router
            from app.distributed.data_events import DataEventType

            router = get_event_router()
            router.subscribe(DataEventType.DATA_QUALITY_ALERT, self._on_quality_alert)
            router.subscribe(DataEventType.DATA_SYNC_COMPLETED, self._on_sync_completed)
            logger.info(
                "[DataCleanupDaemon] Subscribed to DATA_QUALITY_ALERT, DATA_SYNC_COMPLETED"
            )
        except ImportError:
            logger.debug("[DataCleanupDaemon] Event router not available")
        except Exception as e:
            logger.warning(f"[DataCleanupDaemon] Failed to subscribe to events: {e}")

    async def _on_quality_alert(self, event: Any) -> None:
        """Handle DATA_QUALITY_ALERT - trigger immediate scan.

        Quality alerts indicate poor quality data has been detected.
        We immediately trigger a cleanup scan instead of waiting.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            db_path = payload.get("database") or payload.get("db_path")
            quality_score = payload.get("quality_score", 0.0)

            logger.info(
                f"[DataCleanupDaemon] Quality alert received: "
                f"db={db_path}, quality={quality_score:.1%}"
            )

            # Trigger immediate scan
            if self._running:
                await self._scan_and_cleanup()
        except Exception as e:
            logger.debug(f"[DataCleanupDaemon] Error handling quality alert: {e}")

    async def _on_sync_completed(self, event: Any) -> None:
        """Handle DATA_SYNC_COMPLETED - scan for quality after sync.

        After new data is synced, we scan to check quality of newly
        received databases.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            games_synced = payload.get("games_synced", 0)

            if games_synced > 100:  # Only scan if significant data was synced
                logger.info(
                    f"[DataCleanupDaemon] Sync completed ({games_synced} games), "
                    "triggering quality scan"
                )
                if self._running:
                    await self._scan_and_cleanup()
        except Exception as e:
            logger.debug(f"[DataCleanupDaemon] Error handling sync completed: {e}")

    async def _scan_loop(self) -> None:
        """Main scan loop."""
        # Initial delay to let other systems stabilize
        await asyncio.sleep(60)

        while self._running:
            try:
                await self._scan_and_cleanup()
                self._stats.last_scan_time = time.time()
            except asyncio.CancelledError:
                break
            except (RuntimeError, OSError) as e:
                self._errors_count += 1
                self._last_error = str(e)
                self._stats.last_error = str(e)
                logger.error(f"Cleanup scan error: {e}")

            await asyncio.sleep(self.config.scan_interval_seconds)

    async def _scan_and_cleanup(self) -> None:
        """Scan all databases and cleanup poor quality ones."""
        if not self._data_dir.exists():
            logger.debug(f"Data directory not found: {self._data_dir}")
            return

        databases = list(self._data_dir.glob("*.db"))
        databases = [
            db
            for db in databases
            if not db.name.startswith(".")
            and db.parent != self._quarantine_dir
            and "manifest" not in db.name
            and "schema" not in db.name
        ]

        if not databases:
            logger.debug("No databases found to scan")
            return

        logger.info(f"Scanning {len(databases)} databases for quality")

        quarantined = 0
        deleted = 0

        for db_path in databases:
            try:
                assessment = self._assess_database(db_path)
                self._stats.databases_scanned += 1

                # Skip canonical databases if configured
                if self.config.require_canonical_pattern:
                    if db_path.name.startswith("canonical_"):
                        logger.debug(f"Skipping canonical database: {db_path.name}")
                        continue

                # Check deletion threshold
                if assessment.quality_score < self.config.quality_threshold_delete:
                    if assessment.total_games >= self.config.min_games_before_delete:
                        await self._delete_database(db_path, assessment)
                        deleted += 1
                        continue
                    else:
                        logger.info(
                            f"Skipping delete of {db_path.name}: "
                            f"only {assessment.total_games} games"
                        )

                # Check quarantine threshold
                if assessment.quality_score < self.config.quality_threshold_quarantine:
                    await self._quarantine_database(db_path, assessment)
                    quarantined += 1
                    continue

                # Check move coverage threshold
                if assessment.move_coverage < self.config.move_coverage_threshold:
                    await self._quarantine_database(
                        db_path,
                        assessment,
                        reason=f"Low move coverage: {assessment.move_coverage:.1%}",
                    )
                    quarantined += 1

            except (RuntimeError, OSError, sqlite3.Error) as e:
                logger.warning(f"Failed to assess {db_path.name}: {e}")

        if quarantined > 0 or deleted > 0:
            logger.info(
                f"Cleanup complete: {deleted} deleted, {quarantined} quarantined"
            )

    def _assess_database(self, db_path: Path) -> DatabaseAssessment:
        """Assess database quality by sampling recent games."""
        try:
            from app.quality.unified_quality import compute_game_quality_from_params
        except ImportError:
            # Fallback if quality module not available
            return DatabaseAssessment(
                path=str(db_path),
                total_games=0,
                quality_score=0.5,  # Assume medium quality
                move_coverage=1.0,
                valid_victory_types=1.0,
                issues=["Quality module not available"],
            )

        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.row_factory = sqlite3.Row

        try:
            # Get total game count
            cursor = conn.execute("SELECT COUNT(*) FROM games")
            total_games = cursor.fetchone()[0]

            if total_games == 0:
                return DatabaseAssessment(
                    path=str(db_path),
                    total_games=0,
                    quality_score=0.0,
                    move_coverage=0.0,
                    valid_victory_types=0.0,
                    issues=["No games in database"],
                )

            # Sample recent games
            cursor = conn.execute(
                """
                SELECT game_id, game_status, winner, termination_reason,
                       total_moves, board_type
                FROM games
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (self.config.quality_sample_size,),
            )
            games = cursor.fetchall()

            # Check move coverage
            try:
                cursor = conn.execute(
                    """
                    SELECT COUNT(DISTINCT g.game_id)
                    FROM games g
                    JOIN game_moves m ON g.game_id = m.game_id
                    ORDER BY g.created_at DESC
                    LIMIT ?
                """,
                    (self.config.quality_sample_size,),
                )
                games_with_moves = cursor.fetchone()[0]
                move_coverage = games_with_moves / max(len(games), 1)
            except sqlite3.OperationalError:
                # game_moves table might not exist
                move_coverage = 1.0  # Assume OK if table doesn't exist

            # Check victory types
            from app.utils.victory_type import is_valid_victory_type

            valid_victory = 0
            for g in games:
                if is_valid_victory_type(g["termination_reason"]):
                    valid_victory += 1
            valid_victory_pct = valid_victory / max(len(games), 1)

            # Compute quality scores
            quality_scores = []
            issues = []

            for g in games:
                try:
                    q = compute_game_quality_from_params(
                        game_id=g["game_id"],
                        game_status=g["game_status"],
                        winner=g["winner"],
                        termination_reason=g["termination_reason"],
                        total_moves=g["total_moves"],
                        board_type=g["board_type"] or "square8",
                    )
                    quality_scores.append(q.quality_score)
                except (KeyError, TypeError, ValueError) as e:
                    issues.append(f"Quality error: {e}")
                    quality_scores.append(0.3)

            avg_quality = sum(quality_scores) / max(len(quality_scores), 1)

            # Add issues for low scores
            if avg_quality < self.config.quality_threshold_quarantine:
                issues.append(f"Low quality: {avg_quality:.2f}")
            if move_coverage < self.config.move_coverage_threshold:
                issues.append(f"Low move coverage: {move_coverage:.1%}")
            if valid_victory_pct < 0.5:
                issues.append(f"Low valid victory types: {valid_victory_pct:.1%}")

            return DatabaseAssessment(
                path=str(db_path),
                total_games=total_games,
                quality_score=avg_quality,
                move_coverage=move_coverage,
                valid_victory_types=valid_victory_pct,
                issues=issues,
                sampled_games=len(games),
            )

        finally:
            conn.close()

    async def _quarantine_database(
        self,
        db_path: Path,
        assessment: DatabaseAssessment,
        reason: str | None = None,
    ) -> None:
        """Move database to quarantine folder."""
        reason = reason or f"Low quality: {assessment.quality_score:.2f}"

        dest = self._quarantine_dir / db_path.name
        logger.warning(f"Quarantining {db_path.name}: {reason}")

        # Move database
        shutil.move(str(db_path), str(dest))

        # Write assessment sidecar
        assessment_path = dest.with_suffix(".assessment.json")
        with open(assessment_path, "w") as f:
            json.dump(
                {
                    **asdict(assessment),
                    "quarantine_reason": reason,
                    "quarantine_time": datetime.now().isoformat(),
                    "node_id": self.node_id,
                },
                f,
                indent=2,
            )

        # Log to audit file
        self._log_audit_action("quarantine", db_path, assessment, reason)

        self._stats.databases_quarantined += 1
        self._stats.games_quarantined += assessment.total_games
        self._events_processed += 1

    async def _delete_database(
        self,
        db_path: Path,
        assessment: DatabaseAssessment,
    ) -> None:
        """Delete database with logging."""
        reason = f"Very low quality: {assessment.quality_score:.2f}"
        logger.warning(f"Deleting {db_path.name}: {reason}")

        # Log BEFORE deleting
        self._log_audit_action("delete", db_path, assessment, reason)

        # Delete the file
        db_path.unlink()

        self._stats.databases_deleted += 1
        self._stats.games_deleted += assessment.total_games
        self._events_processed += 1

    def _log_audit_action(
        self,
        action: str,
        db_path: Path,
        assessment: DatabaseAssessment,
        reason: str,
    ) -> None:
        """Log cleanup action to audit file."""
        try:
            with open(self._audit_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "action": action,
                            "path": str(db_path),
                            "reason": reason,
                            "quality_score": assessment.quality_score,
                            "total_games": assessment.total_games,
                            "move_coverage": assessment.move_coverage,
                            "valid_victory_types": assessment.valid_victory_types,
                            "issues": assessment.issues,
                            "timestamp": datetime.now().isoformat(),
                            "node_id": self.node_id,
                        }
                    )
                    + "\n"
                )
        except (OSError, IOError) as e:
            logger.error(f"Failed to write audit log: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get daemon status."""
        return {
            "node_id": self.node_id,
            "running": self._running,
            "status": self._coordinator_status.value,
            "uptime_seconds": self.uptime_seconds,
            "config": {
                "enabled": self.config.enabled,
                "scan_interval_seconds": self.config.scan_interval_seconds,
                "quality_threshold_delete": self.config.quality_threshold_delete,
                "quality_threshold_quarantine": self.config.quality_threshold_quarantine,
                "move_coverage_threshold": self.config.move_coverage_threshold,
            },
            "stats": {
                "databases_scanned": self._stats.databases_scanned,
                "databases_quarantined": self._stats.databases_quarantined,
                "databases_deleted": self._stats.databases_deleted,
                "games_quarantined": self._stats.games_quarantined,
                "games_deleted": self._stats.games_deleted,
                "last_scan_time": self._stats.last_scan_time,
                "last_error": self._stats.last_error,
            },
            "directories": {
                "data_dir": str(self._data_dir),
                "quarantine_dir": str(self._quarantine_dir),
                "audit_path": str(self._audit_path),
            },
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get daemon metrics in protocol-compliant format."""
        return {
            "name": self.name,
            "status": self._coordinator_status.value,
            "uptime_seconds": self.uptime_seconds,
            "start_time": self._start_time,
            "events_processed": self._events_processed,
            "errors_count": self._errors_count,
            "last_error": self._last_error,
            # Cleanup-specific metrics
            "databases_scanned": self._stats.databases_scanned,
            "databases_quarantined": self._stats.databases_quarantined,
            "databases_deleted": self._stats.databases_deleted,
            "games_quarantined": self._stats.games_quarantined,
            "games_deleted": self._stats.games_deleted,
        }

    def health_check(self) -> HealthCheckResult:
        """Check daemon health."""
        if self._coordinator_status == CoordinatorStatus.ERROR:
            return HealthCheckResult.unhealthy(
                f"Daemon in error state: {self._last_error}"
            )

        if self._coordinator_status == CoordinatorStatus.STOPPED:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="Daemon is stopped",
            )

        if not self.config.enabled:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="Daemon disabled by configuration",
            )

        # Check for stale scan
        if self._stats.last_scan_time > 0:
            scan_age = time.time() - self._stats.last_scan_time
            if scan_age > self.config.scan_interval_seconds * 3:
                return HealthCheckResult.degraded(
                    f"No scan in {scan_age:.0f}s (interval: {self.config.scan_interval_seconds}s)",
                    seconds_since_last_scan=scan_age,
                )

        return HealthCheckResult(
            healthy=True,
            status=self._coordinator_status,
            details={
                "uptime_seconds": self.uptime_seconds,
                "databases_scanned": self._stats.databases_scanned,
                "databases_quarantined": self._stats.databases_quarantined,
                "databases_deleted": self._stats.databases_deleted,
            },
        )

    async def scan_now(self) -> dict[str, int]:
        """Trigger an immediate scan cycle.

        Returns:
            Dict with scan results (scanned, quarantined, deleted counts).
        """
        before = (
            self._stats.databases_scanned,
            self._stats.databases_quarantined,
            self._stats.databases_deleted,
        )

        await self._scan_and_cleanup()

        after = (
            self._stats.databases_scanned,
            self._stats.databases_quarantined,
            self._stats.databases_deleted,
        )

        return {
            "scanned": after[0] - before[0],
            "quarantined": after[1] - before[1],
            "deleted": after[2] - before[2],
        }


# Module-level instance for singleton access
_cleanup_daemon: DataCleanupDaemon | None = None


def get_cleanup_daemon() -> DataCleanupDaemon:
    """Get the singleton DataCleanupDaemon instance."""
    global _cleanup_daemon
    if _cleanup_daemon is None:
        _cleanup_daemon = DataCleanupDaemon()
    return _cleanup_daemon


def reset_cleanup_daemon() -> None:
    """Reset the singleton (for testing)."""
    global _cleanup_daemon
    _cleanup_daemon = None


__all__ = [
    "CleanupConfig",
    "CleanupStats",
    "DataCleanupDaemon",
    "DatabaseAssessment",
    "get_cleanup_daemon",
    "reset_cleanup_daemon",
]
