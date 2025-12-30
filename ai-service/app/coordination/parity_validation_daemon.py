"""Coordinator-Side Parity Validation Daemon for RingRift.

This daemon runs on the coordinator node (which has Node.js) and validates
TS/Python parity for canonical databases. Cluster nodes lack Node.js, so
they generate databases with "pending_gate" parity status. This daemon
periodically validates those databases and stores TS reference hashes.

December 2025: Created to address cluster model training failures caused by
unvalidated parity gates on cluster nodes.
December 2025: Migrated from BaseDaemon to HandlerBase pattern.

Features:
- Periodic validation of canonical databases (every 30 minutes)
- Stores TS reference hashes using existing parity_validator infrastructure
- Emits PARITY_VALIDATION_COMPLETED events for pipeline coordination
- Only runs on coordinator (has npx/Node.js)
- Graceful handling of databases with pending_gate status
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.coordination.contracts import CoordinatorStatus
from app.coordination.event_utils import parse_config_key
from app.coordination.handler_base import HandlerBase, HealthCheckResult

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ParityValidationConfig:
    """Configuration for ParityValidationDaemon.

    December 2025: Simplified - no longer inherits from DaemonConfig.
    HandlerBase uses cycle_interval directly.

    Attributes:
        check_interval_seconds: How often to validate databases (default: 30 minutes)
        data_dir: Directory containing canonical game databases
        max_games_per_db: Max games to validate per database per cycle
        fail_on_missing_npx: Whether to fail if npx is not available
        emit_events: Whether to emit events for monitoring
    """

    check_interval_seconds: int = 1800  # 30 minutes
    data_dir: str = ""
    max_games_per_db: int = 100
    fail_on_missing_npx: bool = False
    emit_events: bool = True

    @classmethod
    def from_env(cls, prefix: str = "RINGRIFT_PARITY") -> "ParityValidationConfig":
        """Load configuration from environment variables."""
        config = cls()

        # Check interval
        interval_key = f"{prefix}_INTERVAL"
        if os.environ.get(interval_key):
            try:
                config.check_interval_seconds = int(os.environ[interval_key])
            except ValueError:
                pass

        # Daemon-specific env vars
        if os.environ.get(f"{prefix}_DATA_DIR"):
            config.data_dir = os.environ.get(f"{prefix}_DATA_DIR", "")

        if os.environ.get(f"{prefix}_MAX_GAMES_PER_DB"):
            try:
                config.max_games_per_db = int(
                    os.environ.get(f"{prefix}_MAX_GAMES_PER_DB", "100")
                )
            except ValueError:
                pass

        return config


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ParityValidationResult:
    """Result of parity validation for a database."""

    db_path: str = ""
    board_type: str = ""
    num_players: int = 0
    total_games: int = 0
    games_validated: int = 0
    games_passed: int = 0
    games_failed: int = 0
    games_skipped: int = 0
    validation_time: str = ""
    errors: list[str] = field(default_factory=list)


@dataclass
class ParityValidationSummary:
    """Summary of all parity validations in a cycle."""

    scan_time: str = ""
    databases_scanned: int = 0
    total_games_validated: int = 0
    total_games_passed: int = 0
    total_games_failed: int = 0
    results_by_db: dict[str, ParityValidationResult] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


# =============================================================================
# Daemon Implementation
# =============================================================================


class ParityValidationDaemon(HandlerBase):
    """Daemon that validates TS/Python parity for canonical databases.

    This daemon runs on the coordinator node (which has Node.js installed)
    and validates parity for databases synced from cluster nodes.

    December 2025: Migrated to HandlerBase pattern.
    - Uses HandlerBase singleton (get_instance/reset_instance)
    - Uses _stats for metrics tracking
    """

    def __init__(
        self,
        config: ParityValidationConfig | None = None,
    ) -> None:
        """Initialize the daemon.

        Args:
            config: Configuration for the daemon. Uses environment if None.
        """
        self._daemon_config = config or ParityValidationConfig.from_env()

        super().__init__(
            name="ParityValidationDaemon",
            config=self._daemon_config,
            cycle_interval=float(self._daemon_config.check_interval_seconds),
        )

        # Stats
        self._total_validations = 0
        self._total_passed = 0
        self._total_failed = 0
        self._last_validation_time: datetime | None = None
        self._last_result: ParityValidationSummary | None = None
        self._npx_available: bool | None = None

    @property
    def config(self) -> ParityValidationConfig:
        """Get daemon configuration."""
        return self._daemon_config

    # -------------------------------------------------------------------------
    # Core Daemon Methods
    # -------------------------------------------------------------------------

    async def _run_cycle(self) -> None:
        """Run one validation cycle."""
        # December 30, 2025: Wrap blocking subprocess call with asyncio.to_thread
        if not await asyncio.to_thread(self._check_npx_available):
            logger.warning(
                "[ParityValidationDaemon] npx not available - skipping validation"
            )
            if self.config.fail_on_missing_npx:
                self._record_error("npx not available on this node")
            return

        summary = await self._validate_all_databases()
        self._last_result = summary
        self._last_validation_time = datetime.now(timezone.utc)

        # Update stats
        self._total_validations += summary.total_games_validated
        self._total_passed += summary.total_games_passed
        self._total_failed += summary.total_games_failed

        # Emit event
        if self.config.emit_events and summary.databases_scanned > 0:
            self._emit_validation_complete(summary)

        # Log summary
        logger.info(
            f"[ParityValidationDaemon] Validation complete: "
            f"{summary.databases_scanned} databases, "
            f"{summary.total_games_validated} games validated, "
            f"{summary.total_games_passed} passed, "
            f"{summary.total_games_failed} failed"
        )

    def health_check(self) -> HealthCheckResult:
        """Return health status."""
        is_healthy = self._running and self._npx_available is not False

        details: dict[str, Any] = {
            "running": self._running,
            "npx_available": self._npx_available,
            "total_validations": self._total_validations,
            "total_passed": self._total_passed,
            "total_failed": self._total_failed,
            "error_count": self._stats.errors_count,
        }

        if self._last_validation_time:
            details["last_validation"] = self._last_validation_time.isoformat()

        message = "healthy" if is_healthy else "unhealthy"
        if self._npx_available is False:
            message = "npx not available"

        return HealthCheckResult(
            healthy=is_healthy,
            status=CoordinatorStatus.RUNNING if is_healthy else CoordinatorStatus.STOPPED,
            message=message,
            details=details,
        )

    # -------------------------------------------------------------------------
    # Validation Methods
    # -------------------------------------------------------------------------

    def _check_npx_available(self) -> bool:
        """Check if npx is available on this node."""
        if self._npx_available is not None:
            return self._npx_available

        try:
            result = subprocess.run(
                ["which", "npx"],
                capture_output=True,
                timeout=10,
            )
            self._npx_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            self._npx_available = False

        if not self._npx_available:
            logger.warning(
                "[ParityValidationDaemon] npx not found - this node cannot validate TS parity"
            )

        return self._npx_available

    async def _validate_all_databases(self) -> ParityValidationSummary:
        """Validate all canonical databases."""
        summary = ParityValidationSummary(
            scan_time=datetime.now(timezone.utc).isoformat(),
        )

        data_dir = Path(self.config.data_dir or "data/games")
        if not data_dir.exists():
            logger.warning(f"[ParityValidationDaemon] Data dir not found: {data_dir}")
            return summary

        # Find canonical databases
        canonical_dbs = list(data_dir.glob("canonical_*.db"))
        logger.info(
            f"[ParityValidationDaemon] Found {len(canonical_dbs)} canonical databases"
        )

        for db_path in canonical_dbs:
            try:
                result = await self._validate_database(db_path)
                summary.results_by_db[str(db_path)] = result
                summary.databases_scanned += 1
                summary.total_games_validated += result.games_validated
                summary.total_games_passed += result.games_passed
                summary.total_games_failed += result.games_failed
            except Exception as e:
                logger.error(f"[ParityValidationDaemon] Error validating {db_path}: {e}")
                summary.errors.append(f"{db_path}: {e}")

        return summary

    async def _validate_database(self, db_path: Path) -> ParityValidationResult:
        """Validate a single database.

        Uses the existing parity validation infrastructure to:
        1. Run TS replay for games with pending_gate status
        2. Store TS hashes for validated games
        3. Track pass/fail statistics
        """
        result = ParityValidationResult(
            db_path=str(db_path),
            validation_time=datetime.now(timezone.utc).isoformat(),
        )

        # Parse board type and num_players from filename using canonical utility
        # Format: canonical_hex8_2p.db
        stem = db_path.stem  # canonical_hex8_2p
        config_key = stem.replace("canonical_", "").replace("selfplay_", "")
        parsed = parse_config_key(config_key)
        if parsed:
            result.board_type = parsed.board_type
            result.num_players = parsed.num_players

        # Import here to avoid circular imports
        try:
            from app.db.game_replay import GameReplayDB
            from app.db.parity_validator import (
                store_ts_hashes,
                validate_game_parity,
            )
        except ImportError as e:
            result.errors.append(f"Import error: {e}")
            return result

        try:
            db = GameReplayDB(str(db_path))

            # Get games that need validation (pending_gate or no TS hashes)
            games = self._get_games_needing_validation(db)
            result.total_games = len(games)

            # Limit games per cycle
            games_to_validate = games[: self.config.max_games_per_db]

            for game_id in games_to_validate:
                try:
                    # Run parity validation
                    divergence = validate_game_parity(db, game_id)

                    if divergence is None:
                        # Passed - store TS hashes
                        result.games_passed += 1
                        # Note: store_ts_hashes will run TS replay and store the hashes
                        self._store_validated_hashes(db, game_id)
                    else:
                        # Failed
                        result.games_failed += 1
                        logger.warning(
                            f"[ParityValidationDaemon] Parity failure in {db_path.name}: "
                            f"game {game_id} diverged at move {divergence.diverged_at}"
                        )

                    result.games_validated += 1

                except Exception as e:
                    result.games_skipped += 1
                    result.errors.append(f"Game {game_id}: {e}")

        except Exception as e:
            result.errors.append(f"Database error: {e}")

        return result

    def _get_games_needing_validation(self, db: "GameReplayDB") -> list[str]:
        """Get list of game IDs that need parity validation.

        Returns games that:
        1. Have pending_gate parity status
        2. Don't have stored TS hashes
        """
        try:
            from app.db.parity_validator import has_ts_hashes
        except ImportError:
            return []

        games_needing_validation = []

        # Get all completed games
        try:
            conn = db._connection  # Use internal connection
            cursor = conn.execute(
                """
                SELECT game_id, parity_gate
                FROM games
                WHERE game_status = 'completed'
                ORDER BY created_at DESC
                """
            )
            for row in cursor.fetchall():
                game_id = row[0]
                parity_gate = row[1] if len(row) > 1 else None

                # Check if needs validation
                if parity_gate in (None, "pending_gate", "failed"):
                    if not has_ts_hashes(db, game_id):
                        games_needing_validation.append(game_id)

        except Exception as e:
            logger.debug(f"Error getting games needing validation: {e}")

        return games_needing_validation

    def _store_validated_hashes(self, db: "GameReplayDB", game_id: str) -> None:
        """Store TS hashes for a validated game and update parity status."""
        try:
            from app.db.parity_validator import (
                populate_ts_hashes_from_validation,
            )

            # This function runs TS replay and stores the hashes
            populate_ts_hashes_from_validation(db, game_id)

            # Update parity_gate status to 'passed'
            try:
                conn = db._connection
                conn.execute(
                    "UPDATE games SET parity_gate = 'passed' WHERE game_id = ?",
                    (game_id,),
                )
                conn.commit()
            except Exception as e:
                logger.debug(f"Error updating parity_gate for {game_id}: {e}")

        except Exception as e:
            logger.debug(f"Error storing TS hashes for {game_id}: {e}")

    # -------------------------------------------------------------------------
    # Event Emission
    # -------------------------------------------------------------------------

    def _emit_validation_complete(self, summary: ParityValidationSummary) -> None:
        """Emit PARITY_VALIDATION_COMPLETED event."""
        try:
            from app.distributed.data_events import DataEventType

            # Try to get event bus
            try:
                from app.coordination.event_router import get_event_bus

                bus = get_event_bus()
            except ImportError:
                return

            payload = {
                "databases_scanned": summary.databases_scanned,
                "total_games_validated": summary.total_games_validated,
                "total_games_passed": summary.total_games_passed,
                "total_games_failed": summary.total_games_failed,
                "scan_time": summary.scan_time,
                "source": "ParityValidationDaemon",
            }

            bus.publish_event(
                DataEventType.PARITY_VALIDATION_COMPLETED.value,
                payload,
            )

            logger.debug(
                f"[ParityValidationDaemon] Emitted PARITY_VALIDATION_COMPLETED event"
            )

        except Exception as e:
            logger.debug(f"Error emitting parity validation event: {e}")


# =============================================================================
# Singleton Access (using HandlerBase class methods)
# =============================================================================


def get_parity_validation_daemon() -> ParityValidationDaemon:
    """Get or create the singleton ParityValidationDaemon instance.

    Uses HandlerBase.get_instance() for thread-safe singleton access.
    """
    return ParityValidationDaemon.get_instance()


def reset_parity_validation_daemon() -> None:
    """Reset the singleton instance (for testing).

    Uses HandlerBase.reset_instance() for thread-safe cleanup.
    """
    ParityValidationDaemon.reset_instance()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ParityValidationConfig",
    "ParityValidationDaemon",
    "ParityValidationResult",
    "ParityValidationSummary",
    "get_parity_validation_daemon",
    "reset_parity_validation_daemon",
]
