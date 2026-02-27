"""Tournament Data Pipeline Loop for automated training data export.

January 2026: Created as part of the comprehensive evaluation system.

This loop runs every hour to discover tournament and gauntlet game databases,
apply quality gates, export them to NPZ format for training, and trigger
training jobs on high-quality tournament data.

Features:
- Discovers tournament/gauntlet databases across the cluster
- Quality gate: minimum game count and quality score thresholds
- Exports games to NPZ with quality metadata
- Triggers training events for downstream consumption
- Tracks exported databases to avoid duplicate work

Usage:
    from scripts.p2p.loops.tournament_data_pipeline_loop import (
        TournamentDataPipelineLoop,
        TournamentDataPipelineConfig,
    )

    loop = TournamentDataPipelineLoop(
        get_role=lambda: NodeRole.LEADER,
    )
    await loop.run_forever()
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from scripts.p2p.db_helpers import p2p_db_connection
from .base import BaseLoop

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class TournamentDataPipelineConfig:
    """Configuration for tournament data pipeline loop."""

    # Loop interval in seconds (default: 1 hour)
    interval: float = 3600

    # Minimum games required before export
    min_games_for_export: int = 100

    # Quality threshold (0.0 - 1.0)
    quality_threshold: float = 0.6

    # Directories to scan for tournament databases
    tournament_db_patterns: list[str] = field(
        default_factory=lambda: [
            "data/games/tournament_*.db",
            "data/games/gauntlet_*.db",
            "data/games/evaluation_*.db",
        ]
    )

    # Output directory for NPZ exports
    output_dir: str = "data/training/tournament"

    # Whether to emit events for training trigger
    emit_training_events: bool = True

    # Whether loop is enabled
    enabled: bool = True

    # Exported databases tracking file
    exported_tracking_file: str = "data/.tournament_exports.json"


@dataclass
class DatabaseStats:
    """Statistics for a single database."""

    path: Path
    game_count: int = 0
    avg_quality: float = 0.0
    board_type: str = ""
    num_players: int = 0
    total_moves: int = 0
    config_key: str = ""

    @property
    def is_valid_for_export(self) -> bool:
        """Check if database meets export criteria."""
        return self.game_count > 0 and self.config_key != ""


@dataclass
class ExportResult:
    """Result of exporting a database to NPZ."""

    db_path: Path
    npz_path: Path | None = None
    success: bool = False
    game_count: int = 0
    sample_count: int = 0
    error: str | None = None
    export_time: float = 0.0


@dataclass
class PipelineCycleStats:
    """Statistics for a single pipeline cycle."""

    databases_discovered: int = 0
    databases_skipped_exported: int = 0
    databases_skipped_count: int = 0
    databases_skipped_quality: int = 0
    databases_exported: int = 0
    export_failures: int = 0
    total_games_exported: int = 0
    total_samples_generated: int = 0
    cycle_duration: float = 0.0
    export_results: list[ExportResult] = field(default_factory=list)


class TournamentDataPipelineLoop(BaseLoop):
    """Periodic pipeline for exporting tournament/gauntlet games to training data.

    Runs every hour on leader node. Discovers tournament databases, applies
    quality gates, exports to NPZ, and triggers training.

    Pipeline Steps:
    1. Discover tournament/gauntlet databases using configurable patterns
    2. Filter out already-exported databases
    3. Apply quality gate (min games, quality threshold)
    4. Export qualifying databases to NPZ format
    5. Emit TOURNAMENT_DATA_READY events for training
    6. Track exported databases
    """

    def __init__(
        self,
        get_role: Callable[[], Any],
        config: TournamentDataPipelineConfig | None = None,
    ):
        """Initialize the tournament data pipeline loop.

        Args:
            get_role: Callable returning current NodeRole (LEADER/FOLLOWER/VOTER)
            config: Loop configuration
        """
        self.config = config or TournamentDataPipelineConfig()

        super().__init__(
            name="tournament_data_pipeline",
            interval=self.config.interval,
            enabled=self.config.enabled,
        )

        self._get_role = get_role

        # Track exported databases
        self._exported_databases: dict[str, float] = {}  # db_hash -> export_timestamp
        self._load_exported_tracking()

        # Statistics
        self._last_cycle_stats: PipelineCycleStats | None = None
        self._total_exports: int = 0
        self._total_failures: int = 0

    def _is_leader(self) -> bool:
        """Check if current node is the cluster leader."""
        try:
            role = self._get_role()
            role_str = role.value if hasattr(role, "value") else str(role)
            return role_str.lower() == "leader"
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to get role: {e}")
            return False

    def _load_exported_tracking(self) -> None:
        """Load the exported databases tracking file."""
        try:
            tracking_path = Path(self.config.exported_tracking_file)
            if tracking_path.exists():
                import json

                with open(tracking_path) as f:
                    self._exported_databases = json.load(f)
                logger.debug(
                    f"[{self.name}] Loaded {len(self._exported_databases)} exported database records"
                )
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to load export tracking: {e}")
            self._exported_databases = {}

    def _save_exported_tracking(self) -> None:
        """Save the exported databases tracking file."""
        try:
            tracking_path = Path(self.config.exported_tracking_file)
            tracking_path.parent.mkdir(parents=True, exist_ok=True)
            import json

            with open(tracking_path, "w") as f:
                json.dump(self._exported_databases, f, indent=2)
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to save export tracking: {e}")

    def _compute_db_hash(self, db_path: Path) -> str:
        """Compute a hash for the database based on path and mtime."""
        stat = db_path.stat()
        hash_input = f"{db_path.absolute()}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _is_already_exported(self, db_path: Path) -> bool:
        """Check if a database has already been exported."""
        try:
            db_hash = self._compute_db_hash(db_path)
            return db_hash in self._exported_databases
        except Exception:
            return False

    def _mark_exported(self, db_path: Path) -> None:
        """Mark a database as exported."""
        try:
            db_hash = self._compute_db_hash(db_path)
            self._exported_databases[db_hash] = time.time()
            self._save_exported_tracking()
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to mark as exported: {e}")

    async def _run_once(self) -> None:
        """Execute one iteration of the tournament data pipeline."""
        # Only run on leader
        if not self._is_leader():
            logger.debug(f"[{self.name}] Not leader, skipping pipeline cycle")
            return

        logger.info(f"[{self.name}] Starting tournament data pipeline cycle")
        cycle_start = time.time()
        stats = PipelineCycleStats()

        try:
            # 1. Discover tournament databases
            databases = await self._discover_tournament_databases()
            stats.databases_discovered = len(databases)
            logger.info(
                f"[{self.name}] Discovered {stats.databases_discovered} tournament databases"
            )

            if not databases:
                logger.info(f"[{self.name}] No tournament databases found")
                return

            # 2. Filter and quality gate
            for db_path in databases:
                # Check if already exported
                if self._is_already_exported(db_path):
                    stats.databases_skipped_exported += 1
                    continue

                # Get database stats
                db_stats = await self._get_database_stats(db_path)
                if not db_stats.is_valid_for_export:
                    continue

                # Check game count threshold
                if db_stats.game_count < self.config.min_games_for_export:
                    stats.databases_skipped_count += 1
                    logger.debug(
                        f"[{self.name}] Skipping {db_path.name}: "
                        f"{db_stats.game_count} games < {self.config.min_games_for_export}"
                    )
                    continue

                # Check quality threshold
                if db_stats.avg_quality < self.config.quality_threshold:
                    stats.databases_skipped_quality += 1
                    logger.warning(
                        f"[{self.name}] Skipping {db_path.name}: "
                        f"quality {db_stats.avg_quality:.2f} < {self.config.quality_threshold}"
                    )
                    continue

                # 3. Export to NPZ
                result = await self._export_to_npz(db_stats)
                stats.export_results.append(result)

                if result.success:
                    stats.databases_exported += 1
                    stats.total_games_exported += result.game_count
                    stats.total_samples_generated += result.sample_count
                    self._total_exports += 1

                    # 4. Mark as exported
                    self._mark_exported(db_path)

                    # 5. Emit training event
                    if self.config.emit_training_events and result.npz_path:
                        await self._emit_training_event(result, db_stats)
                else:
                    stats.export_failures += 1
                    self._total_failures += 1

                # Brief delay between exports
                await asyncio.sleep(0.5)

            # Emit summary event
            self._emit_completion_event(stats)

        except Exception as e:
            logger.error(f"[{self.name}] Pipeline cycle failed: {e}", exc_info=True)
            raise

        finally:
            stats.cycle_duration = time.time() - cycle_start
            self._last_cycle_stats = stats

            logger.info(
                f"[{self.name}] Pipeline cycle complete: "
                f"{stats.databases_exported}/{stats.databases_discovered} exported "
                f"({stats.total_samples_generated} samples) "
                f"in {stats.cycle_duration:.1f}s"
            )

    async def _discover_tournament_databases(self) -> list[Path]:
        """Discover tournament/gauntlet databases matching configured patterns."""
        import glob

        databases = []

        for pattern in self.config.tournament_db_patterns:
            # Support glob patterns
            matches = glob.glob(pattern, recursive=True)
            for match in matches:
                path = Path(match)
                if path.exists() and path.suffix == ".db":
                    databases.append(path)

        # Remove duplicates
        return list(set(databases))

    async def _get_database_stats(self, db_path: Path) -> DatabaseStats:
        """Get statistics for a database."""
        stats = DatabaseStats(path=db_path)

        try:
            import sqlite3

            def _read_stats() -> DatabaseStats:
                with p2p_db_connection(db_path, timeout=10.0) as conn:
                    cursor = conn.cursor()

                    # Get game count
                    cursor.execute(
                        "SELECT COUNT(*) FROM games WHERE status = 'completed'"
                    )
                    stats.game_count = cursor.fetchone()[0]

                    # Get average quality score if available
                    cursor.execute(
                        """
                        SELECT AVG(quality_score) FROM games
                        WHERE status = 'completed' AND quality_score IS NOT NULL
                        """
                    )
                    result = cursor.fetchone()[0]
                    stats.avg_quality = result if result else 0.5  # Default to neutral

                    # Get board type and num_players from first game
                    cursor.execute(
                        """
                        SELECT board_type, num_players FROM games
                        WHERE status = 'completed' LIMIT 1
                        """
                    )
                    row = cursor.fetchone()
                    if row:
                        stats.board_type = row[0] or ""
                        stats.num_players = row[1] or 0
                        if stats.board_type and stats.num_players:
                            stats.config_key = f"{stats.board_type}_{stats.num_players}p"

                    # Get total moves
                    cursor.execute("SELECT COUNT(*) FROM moves")
                    stats.total_moves = cursor.fetchone()[0]

                return stats

            # Run in thread pool to avoid blocking
            stats = await asyncio.to_thread(_read_stats)

        except Exception as e:
            logger.warning(f"[{self.name}] Failed to get stats for {db_path}: {e}")

        return stats

    async def _export_to_npz(self, db_stats: DatabaseStats) -> ExportResult:
        """Export a database to NPZ format."""
        result = ExportResult(db_path=db_stats.path)
        export_start = time.time()

        try:
            # Create output directory
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate output filename
            timestamp = int(time.time())
            npz_name = f"tournament_{db_stats.config_key}_{timestamp}.npz"
            npz_path = output_dir / npz_name

            logger.info(
                f"[{self.name}] Exporting {db_stats.path.name} -> {npz_path.name}"
            )

            # Use export_replay_dataset module
            try:
                from app.training.parallel_encoding import (
                    export_database_to_npz,
                )

                sample_count = await asyncio.to_thread(
                    export_database_to_npz,
                    str(db_stats.path),
                    str(npz_path),
                    db_stats.board_type,
                    db_stats.num_players,
                )

                if sample_count and sample_count > 0:
                    result.success = True
                    result.npz_path = npz_path
                    result.game_count = db_stats.game_count
                    result.sample_count = sample_count
                else:
                    result.error = "No samples exported"

            except ImportError:
                # Fallback: use subprocess to call export script
                import subprocess

                # Feb 2026: Cross-process export coordination
                _config_key = f"{db_stats.board_type}_{db_stats.num_players}p"
                try:
                    from app.coordination.export_coordinator import get_export_coordinator
                    _coord = get_export_coordinator()
                    if not _coord.try_acquire(_config_key):
                        result.error = "Cross-process export slot unavailable"
                        return result
                    _release_slot = True
                except Exception:
                    _release_slot = False

                try:
                    proc = await asyncio.create_subprocess_exec(
                        "python",
                        "scripts/export_replay_dataset.py",
                        "--db",
                        str(db_stats.path),
                        "--board-type",
                        db_stats.board_type,
                        "--num-players",
                        str(db_stats.num_players),
                        "--output",
                        str(npz_path),
                        "--quiet",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    _, stderr = await proc.communicate()

                    if proc.returncode == 0 and npz_path.exists():
                        result.success = True
                        result.npz_path = npz_path
                        result.game_count = db_stats.game_count
                        # Estimate sample count from file size
                        result.sample_count = db_stats.game_count * 30  # ~30 samples/game
                    else:
                        result.error = stderr.decode() if stderr else "Export failed"
                finally:
                    if _release_slot:
                        try:
                            _coord.release(_config_key)
                        except Exception:
                            pass

        except Exception as e:
            result.error = str(e)
            logger.error(f"[{self.name}] Export failed for {db_stats.path}: {e}")

        result.export_time = time.time() - export_start
        return result

    async def _emit_training_event(
        self, result: ExportResult, db_stats: DatabaseStats
    ) -> None:
        """Emit TOURNAMENT_DATA_READY event for training consumption."""
        try:
            from app.coordination.event_router import emit_event
            from app.distributed.data_events import DataEventType

            emit_event(
                DataEventType.TOURNAMENT_DATA_READY,
                {
                    "npz_path": str(result.npz_path),
                    "source_db": str(result.db_path),
                    "config_key": db_stats.config_key,
                    "board_type": db_stats.board_type,
                    "num_players": db_stats.num_players,
                    "game_count": result.game_count,
                    "sample_count": result.sample_count,
                    "avg_quality": db_stats.avg_quality,
                    "export_time": result.export_time,
                },
            )
            logger.info(
                f"[{self.name}] Emitted TOURNAMENT_DATA_READY for {result.npz_path}"
            )
        except ImportError:
            logger.debug(f"[{self.name}] Event system not available")
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to emit training event: {e}")

    def _emit_completion_event(self, stats: PipelineCycleStats) -> None:
        """Emit event when pipeline cycle completes."""
        try:
            from app.coordination.event_router import emit_event
            from app.distributed.data_events import DataEventType

            emit_event(
                DataEventType.TOURNAMENT_PIPELINE_COMPLETED,
                {
                    "databases_discovered": stats.databases_discovered,
                    "databases_exported": stats.databases_exported,
                    "databases_skipped": (
                        stats.databases_skipped_exported
                        + stats.databases_skipped_count
                        + stats.databases_skipped_quality
                    ),
                    "export_failures": stats.export_failures,
                    "total_games": stats.total_games_exported,
                    "total_samples": stats.total_samples_generated,
                    "cycle_duration": stats.cycle_duration,
                },
            )
        except ImportError:
            logger.debug(f"[{self.name}] Event system not available")
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to emit completion event: {e}")

    def health_check(self) -> Any:
        """Return health check result for DaemonManager integration."""
        try:
            from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
        except ImportError:
            return {
                "healthy": self._running,
                "status": "running" if self._running else "stopped",
                "total_exports": self._total_exports,
            }

        if not self._running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="TournamentDataPipelineLoop is stopped",
            )

        if not self._is_leader():
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.IDLE,
                message="Not leader - pipeline loop idle",
                details={"role": "follower"},
            )

        # Check error rate
        total = self._total_exports + self._total_failures
        if total > 0:
            success_rate = self._total_exports / total
            if success_rate < 0.5:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"High export failure rate: {(1-success_rate)*100:.1f}%",
                    details={
                        "total_exports": self._total_exports,
                        "total_failures": self._total_failures,
                        "success_rate": f"{success_rate*100:.1f}%",
                    },
                )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="TournamentDataPipelineLoop healthy",
            details={
                "total_exports": self._total_exports,
                "total_failures": self._total_failures,
                "last_cycle": (
                    self._last_cycle_stats.cycle_duration
                    if self._last_cycle_stats
                    else None
                ),
                "databases_tracked": len(self._exported_databases),
            },
        )

    def get_last_cycle_stats(self) -> PipelineCycleStats | None:
        """Get statistics from the last pipeline cycle."""
        return self._last_cycle_stats
