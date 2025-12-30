"""Automated NPZ Export Daemon (December 2025).

Automatically exports training data (NPZ files) when game counts exceed thresholds.
This closes the gap between selfplay and training by eliminating the manual export step.

Key features:
- Subscribes to SELFPLAY_COMPLETE events
- Tracks accumulated games per configuration
- Triggers export when game threshold reached (default: 100 games)
- Emits NPZ_EXPORT_STARTED event when export begins
- Emits NPZ_EXPORT_COMPLETE event after successful export
- Integrates with GameDiscovery for finding databases
- Supports cooldown to prevent export spam

Usage:
    from app.coordination.auto_export_daemon import AutoExportDaemon

    daemon = AutoExportDaemon()
    await daemon.start()

December 2025: Created as part of Phase 1 automation improvements.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.core.async_context import safe_create_task
from app.coordination.event_handler_utils import extract_config_key
from app.coordination.event_utils import parse_config_key
from app.coordination.handler_base import HandlerBase, HealthCheckResult

logger = logging.getLogger(__name__)


@dataclass
class AutoExportConfig:
    """Configuration for automated NPZ export."""

    enabled: bool = True
    # Minimum games before triggering export
    # Lowered from 500→100→50 (Dec 2025) for faster training iteration
    min_games_threshold: int = 50
    # Cooldown between exports for same config (seconds)
    # Reduced from 30min→5min→1min (Dec 2025) to minimize training data lag
    export_cooldown_seconds: int = 60  # 1 minute
    # Maximum concurrent exports
    max_concurrent_exports: int = 2
    # Timeout for export subprocess (seconds)
    export_timeout_seconds: int = 3600  # 1 hour
    # Whether to use incremental export (--use-cache)
    use_incremental_export: bool = True
    # Quality filtering options
    require_completed_games: bool = True
    min_moves: int = 10
    # Output directory for NPZ files
    output_dir: Path = field(default_factory=lambda: Path("data/training"))
    # State persistence (Phase 8 Dec 2025)
    state_db_path: Path = field(default_factory=lambda: Path("data/export_daemon_state.db"))
    persist_state: bool = True  # Enable state persistence to recover on crash
    # Event-driven batch export settings (December 2025)
    # Part of 48-hour autonomous operation optimization.
    # Implements "N games OR M seconds, whichever first" semantics.
    event_driven: bool = True  # Enable event-driven mode (vs pure timer)
    batch_accumulation_timeout_seconds: int = 30  # Max wait before export
    immediate_threshold_multiplier: float = 2.0  # Export immediately if games >= threshold * 2


@dataclass
class ConfigExportState:
    """Tracks export state for a single configuration."""

    config_key: str
    board_type: str
    num_players: int
    games_since_last_export: int = 0
    last_export_time: float = 0.0
    last_export_games: int = 0
    total_exported_samples: int = 0
    export_in_progress: bool = False
    consecutive_failures: int = 0


@dataclass
class BatchAccumulator:
    """Tracks accumulated games for batch export triggering.

    December 2025: Part of event-driven export optimization.
    Implements "N games OR M seconds, whichever first" semantics.
    """

    config_key: str
    accumulated_games: int = 0
    accumulation_started: float = 0.0  # Timestamp when accumulation began
    timer_task: asyncio.Task | None = None

    def reset(self) -> None:
        """Reset accumulator after export."""
        self.accumulated_games = 0
        self.accumulation_started = 0.0
        if self.timer_task and not self.timer_task.done():
            self.timer_task.cancel()
        self.timer_task = None


class AutoExportDaemon(HandlerBase):
    """Daemon that automatically exports training data when thresholds are met.

    Inherits from HandlerBase (December 2025 migration) providing:
    - Automatic event subscription via _get_event_subscriptions()
    - Singleton pattern via get_instance()
    - Standardized health check format
    - Lifecycle management (start/stop)
    """

    def __init__(self, config: AutoExportConfig | None = None):
        self._daemon_config = config or AutoExportConfig()
        super().__init__(
            name="auto_export",
            config=self._daemon_config,
            cycle_interval=300.0,  # 5 minutes scan interval
        )
        self._export_states: dict[str, ConfigExportState] = {}
        self._export_semaphore = asyncio.Semaphore(self._daemon_config.max_concurrent_exports)
        self._state_db_initialized = False
        # December 2025: Deduplication guard - when StageEvent subscriptions are active,
        # skip DataEventType handlers to prevent double-counting games
        self._stage_events_active = False
        # December 2025 Phase 3: Track configs pending sync completion.
        # Export is gated on sync completion to prevent race conditions where
        # export starts before data from other nodes has arrived.
        self._pending_sync_configs: set[str] = set()
        # Track whether we should skip due to coordinator mode
        self._coordinator_skip = False
        # Event-driven batch accumulation (December 2025)
        # Part of 48-hour autonomous operation optimization.
        self._batch_accumulators: dict[str, BatchAccumulator] = {}
        self._pending_timer_exports: set[str] = set()  # Configs with active timers

    @property
    def config(self) -> AutoExportConfig:
        """Get the daemon configuration."""
        return self._daemon_config

    def _get_event_subscriptions(self) -> dict[str, Callable]:
        """Return event subscriptions for HandlerBase.

        Subscribes to:
        - SELFPLAY_COMPLETE: Track new games and trigger export
        - SYNC_COMPLETE: Cross-node data sync completion
        - NEW_GAMES_AVAILABLE: New games from local selfplay
        - DATA_SYNC_COMPLETED: Clear sync pending flag
        """
        return {
            "selfplay_complete": self._on_selfplay_complete,
            "sync_complete": self._on_sync_complete,
            "new_games_available": self._on_new_games,
            "data_sync_completed": self._on_data_sync_completed,
        }

    async def _on_start(self) -> None:
        """Hook called before main loop - check coordinator mode and init state DB."""
        from app.config.env import env
        if env.is_coordinator or not env.export_enabled:
            logger.info(
                f"[AutoExportDaemon] Skipped on coordinator node: {env.node_id} "
                f"(is_coordinator={env.is_coordinator}, export_enabled={env.export_enabled})"
            )
            self._coordinator_skip = True
            return

        # Initialize state persistence and load previous state (Phase 8)
        if self._daemon_config.persist_state:
            self._init_state_db()
            self._load_state()

    # ========== State Persistence (Phase 8 Dec 2025) ==========

    def _init_state_db(self) -> None:
        """Initialize SQLite database for state persistence.

        Phase 8 Dec 2025: Persists export state to survive daemon restarts.
        This prevents data loss when pending export counts are lost on crash.
        """
        import sqlite3

        try:
            db_path = self.config.state_db_path
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Use context manager to ensure connection is always closed
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS export_state (
                        config_key TEXT PRIMARY KEY,
                        board_type TEXT NOT NULL,
                        num_players INTEGER NOT NULL,
                        games_since_last_export INTEGER DEFAULT 0,
                        last_export_time REAL DEFAULT 0,
                        last_export_games INTEGER DEFAULT 0,
                        total_exported_samples INTEGER DEFAULT 0,
                        consecutive_failures INTEGER DEFAULT 0,
                        updated_at REAL DEFAULT 0
                    )
                """)

                conn.commit()
            self._state_db_initialized = True

            logger.info(f"[AutoExportDaemon] State database initialized: {db_path}")

        except (sqlite3.Error, OSError) as e:
            logger.error(f"[AutoExportDaemon] Failed to initialize state DB: {e}")
            self._state_db_initialized = False

    def _load_state(self) -> None:
        """Load persisted state from SQLite on startup.

        Recovers pending export counts and last export times that would
        otherwise be lost on daemon restart.
        """
        if not self._state_db_initialized:
            return

        import sqlite3

        try:
            # Use context manager to ensure connection is always closed
            with sqlite3.connect(self.config.state_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM export_state")
                rows = cursor.fetchall()

                loaded_count = 0
                for row in rows:
                    config_key = row["config_key"]
                    self._export_states[config_key] = ConfigExportState(
                        config_key=config_key,
                        board_type=row["board_type"],
                        num_players=row["num_players"],
                        games_since_last_export=row["games_since_last_export"],
                        last_export_time=row["last_export_time"],
                        last_export_games=row["last_export_games"],
                        total_exported_samples=row["total_exported_samples"],
                        consecutive_failures=row["consecutive_failures"],
                    )
                    loaded_count += 1

            if loaded_count > 0:
                logger.info(
                    f"[AutoExportDaemon] Loaded {loaded_count} config states from persistence"
                )

        except (sqlite3.Error, OSError) as e:
            logger.error(f"[AutoExportDaemon] Failed to load state: {e}")

    def _save_state(self, config_key: str) -> None:
        """Persist state for a single config to SQLite.

        Called after any state change to ensure durability.
        """
        if not self._state_db_initialized or not self.config.persist_state:
            return

        state = self._export_states.get(config_key)
        if not state:
            return

        import sqlite3

        try:
            # Use context manager to ensure connection is always closed
            with sqlite3.connect(self.config.state_db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO export_state
                    (config_key, board_type, num_players, games_since_last_export,
                     last_export_time, last_export_games, total_exported_samples,
                     consecutive_failures, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    config_key,
                    state.board_type,
                    state.num_players,
                    state.games_since_last_export,
                    state.last_export_time,
                    state.last_export_games,
                    state.total_exported_samples,
                    state.consecutive_failures,
                    time.time(),
                ))

                conn.commit()

        except (sqlite3.Error, OSError) as e:
            logger.debug(f"[AutoExportDaemon] Failed to save state for {config_key}: {e}")

    # ========== Event Handlers ==========

    async def _on_selfplay_complete(self, result: Any) -> None:
        """Handle selfplay completion event.

        December 2025 Phase 3: Mark config as pending sync before recording games.
        This prevents export from starting before sync has completed, avoiding
        race conditions where export uses incomplete data.
        """
        try:
            # Extract config info from result
            board_type = getattr(result, "board_type", None)
            num_players = getattr(result, "num_players", None)
            games_generated = getattr(result, "games_generated", 0)

            if not board_type or not num_players:
                # Try to extract from metadata
                metadata = getattr(result, "metadata", {})
                board_type = board_type or metadata.get("board_type")
                num_players = num_players or metadata.get("num_players")

            if not board_type or not num_players:
                logger.debug("[AutoExportDaemon] Missing config info in selfplay result")
                return

            config_key = f"{board_type}_{num_players}p"

            # Phase 3: Mark config as pending sync - export will wait for DATA_SYNC_COMPLETED
            self._pending_sync_configs.add(config_key)
            logger.debug(
                f"[AutoExportDaemon] {config_key}: Marked as pending sync "
                "(export gated on DATA_SYNC_COMPLETED)"
            )

            await self._record_games(config_key, board_type, num_players, games_generated)

        except Exception as e:  # noqa: BLE001
            logger.error(f"[AutoExportDaemon] Error handling selfplay complete: {e}")

    async def _on_sync_complete(self, result: Any) -> None:
        """Handle sync completion event (Phase 3A.3: Dec 2025).

        When games are synced from other nodes, check if export is needed.
        This enables cross-node data to trigger exports just like local selfplay.
        """
        try:
            # Extract sync info from result
            metadata = getattr(result, "metadata", {})
            games_synced = metadata.get("games_synced", 0) or metadata.get("files_synced", 0)
            config_key = metadata.get("config_key", "")

            if not games_synced or not config_key:
                return

            board_type, num_players = self._parse_config_key(config_key)
            if not board_type or not num_players:
                return

            logger.info(
                f"[AutoExportDaemon] Sync complete: {games_synced} games for {config_key}"
            )
            await self._record_games(config_key, board_type, num_players, games_synced)

        except Exception as e:  # noqa: BLE001
            logger.error(f"[AutoExportDaemon] Error handling sync complete: {e}")

    async def _on_new_games(self, event: Any) -> None:
        """Handle new games available event."""
        try:
            payload = getattr(event, "payload", {})
            new_games = payload.get("new_games", 0)
            board_type = payload.get("board_type")
            num_players = payload.get("num_players")

            if board_type and num_players:
                config_key = f"{board_type}_{num_players}p"
                await self._record_games(config_key, board_type, num_players, new_games)

        except Exception as e:  # noqa: BLE001
            logger.error(f"[AutoExportDaemon] Error handling new games event: {e}")

    async def _on_selfplay_complete_event(self, event: Any) -> None:
        """Handle SELFPLAY_COMPLETE data events.

        December 2025: Added deduplication guard. If StageEvent subscription is active,
        skip this handler to prevent double-counting games (the StageEvent handler
        already processed this event).
        """
        # Deduplication guard: skip if StageEvent already handled this
        if self._stage_events_active:
            logger.debug(
                "[AutoExportDaemon] Skipping DataEventType.SELFPLAY_COMPLETE "
                "(StageEvent already active)"
            )
            return

        try:
            payload = getattr(event, "payload", {}) or {}
            config_key = extract_config_key(payload)
            games_generated = payload.get("games_played", payload.get("games_generated", 0))
            if not config_key or not games_generated:
                return

            board_type, num_players = self._parse_config_key(config_key)
            if not board_type or not num_players:
                return

            await self._record_games(config_key, board_type, num_players, games_generated)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[AutoExportDaemon] Error handling SELFPLAY_COMPLETE event: {e}")

    async def _on_data_sync_completed(self, event: Any) -> None:
        """Handle DATA_SYNC_COMPLETED data events.

        December 2025 Phase 3: Clear pending sync flag and allow export to proceed.
        This completes the sync-gating mechanism that prevents export race conditions.
        """
        try:
            payload = getattr(event, "payload", {}) or {}
            config_key = payload.get("config") or payload.get("config_key")
            games_synced = payload.get("games_synced", 0) or payload.get("files_synced", 0)
            if not config_key or not games_synced:
                return

            board_type, num_players = self._parse_config_key(config_key)
            if not board_type or not num_players:
                return

            # Phase 3: Clear pending sync flag - export can now proceed safely
            was_pending = config_key in self._pending_sync_configs
            self._pending_sync_configs.discard(config_key)
            if was_pending:
                logger.info(
                    f"[AutoExportDaemon] {config_key}: Sync completed, export now allowed "
                    f"({games_synced} games synced)"
                )

            await self._record_games(config_key, board_type, num_players, games_synced)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[AutoExportDaemon] Error handling DATA_SYNC_COMPLETED event: {e}")

    def _parse_config_key(self, config_key: str) -> tuple[str | None, int | None]:
        """Parse a config key like "square8_2p" into (board_type, num_players).

        December 2025: Now delegates to event_utils.parse_config_key() for
        consistent parsing across the codebase.
        """
        from app.coordination.event_utils import parse_config_key

        parsed = parse_config_key(config_key)
        if parsed is None:
            return None, None
        return parsed.board_type, parsed.num_players

    # ========== Event-Driven Batch Export (December 2025) ==========

    async def _evaluate_batch_trigger(self, config_key: str) -> None:
        """Evaluate whether to trigger export based on batch accumulation.

        December 2025: Part of 48-hour autonomous operation optimization.
        Implements "N games OR M seconds, whichever first" semantics:

        1. Immediate trigger: games >= threshold * 2.0 → export NOW
        2. Threshold trigger: games >= threshold → export NOW
        3. Timer trigger: games > 0, start timer → export when timer fires

        This reduces export latency from ~150s average to ~15s average.
        """
        if not self._daemon_config.event_driven:
            return

        state = self._export_states.get(config_key)
        if not state or state.export_in_progress:
            return

        # Get or create batch accumulator
        if config_key not in self._batch_accumulators:
            self._batch_accumulators[config_key] = BatchAccumulator(config_key=config_key)

        accumulator = self._batch_accumulators[config_key]
        games = state.games_since_last_export

        # Calculate thresholds
        threshold = self._daemon_config.min_games_threshold
        immediate_threshold = int(threshold * self._daemon_config.immediate_threshold_multiplier)

        # Case 1: Immediate trigger - very large batch
        if games >= immediate_threshold:
            logger.info(
                f"[AutoExportDaemon] {config_key}: Immediate export triggered "
                f"({games} >= {immediate_threshold} games)"
            )
            accumulator.reset()
            self._pending_timer_exports.discard(config_key)
            safe_create_task(
                self._run_export(config_key),
                name=f"batch_export_{config_key}",
            )
            return

        # Case 2: Threshold trigger - normal threshold reached
        if games >= threshold:
            logger.info(
                f"[AutoExportDaemon] {config_key}: Threshold export triggered "
                f"({games} >= {threshold} games)"
            )
            accumulator.reset()
            self._pending_timer_exports.discard(config_key)
            safe_create_task(
                self._run_export(config_key),
                name=f"batch_export_{config_key}",
            )
            return

        # Case 3: Timer trigger - start timer if not already running
        if games > 0 and config_key not in self._pending_timer_exports:
            if accumulator.accumulation_started == 0.0:
                accumulator.accumulation_started = time.time()

            # Start timer for batch accumulation timeout
            timeout = self._daemon_config.batch_accumulation_timeout_seconds
            logger.debug(
                f"[AutoExportDaemon] {config_key}: Starting {timeout}s batch timer "
                f"({games} games accumulated)"
            )
            self._pending_timer_exports.add(config_key)
            accumulator.timer_task = safe_create_task(
                self._batch_timer_callback(config_key, timeout),
                name=f"batch_timer_{config_key}",
            )

    async def _batch_timer_callback(self, config_key: str, timeout: float) -> None:
        """Timer callback that triggers export when accumulation timeout fires.

        December 2025: Part of 48-hour autonomous operation optimization.
        This ensures exports happen within `batch_accumulation_timeout_seconds`
        even if the game threshold isn't reached.
        """
        try:
            await asyncio.sleep(timeout)

            # Check if still relevant (not already exported)
            if config_key not in self._pending_timer_exports:
                return

            state = self._export_states.get(config_key)
            if not state or state.export_in_progress:
                self._pending_timer_exports.discard(config_key)
                return

            games = state.games_since_last_export
            if games > 0:
                logger.info(
                    f"[AutoExportDaemon] {config_key}: Timer export triggered "
                    f"after {timeout}s ({games} games)"
                )
                # Reset accumulator
                if config_key in self._batch_accumulators:
                    self._batch_accumulators[config_key].reset()
                self._pending_timer_exports.discard(config_key)
                await self._run_export(config_key)
            else:
                self._pending_timer_exports.discard(config_key)

        except asyncio.CancelledError:
            # Timer was cancelled (export triggered by threshold)
            self._pending_timer_exports.discard(config_key)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[AutoExportDaemon] Timer callback error for {config_key}: {e}")
            self._pending_timer_exports.discard(config_key)

    async def _record_games(
        self, config_key: str, board_type: str, num_players: int, games: int
    ) -> None:
        """Record new games and check if export should be triggered."""
        if config_key not in self._export_states:
            self._export_states[config_key] = ConfigExportState(
                config_key=config_key,
                board_type=board_type,
                num_players=num_players,
            )

        state = self._export_states[config_key]
        state.games_since_last_export += games

        # Persist state to survive daemon restarts (Phase 8)
        self._save_state(config_key)

        logger.debug(
            f"[AutoExportDaemon] {config_key}: +{games} games, "
            f"total pending: {state.games_since_last_export}"
        )

        # Check if we should trigger export
        # December 2025: Use event-driven batch accumulation if enabled
        if self._daemon_config.event_driven:
            await self._evaluate_batch_trigger(config_key)
        else:
            # Fallback to original threshold-based triggering
            await self._maybe_trigger_export(config_key)

    async def _maybe_trigger_export(self, config_key: str) -> None:
        """Check conditions and trigger export if appropriate."""
        state = self._export_states.get(config_key)
        if not state:
            return

        # Phase 3: Check if sync is still pending for this config
        # Export is gated on DATA_SYNC_COMPLETED to prevent race conditions
        if config_key in self._pending_sync_configs:
            logger.debug(
                f"[AutoExportDaemon] {config_key}: Export deferred - waiting for sync completion "
                f"(games pending: {state.games_since_last_export})"
            )
            return

        # Check if already exporting
        if state.export_in_progress:
            logger.debug(f"[AutoExportDaemon] {config_key}: Export already in progress")
            return

        # Check game threshold
        if state.games_since_last_export < self.config.min_games_threshold:
            return

        # Check cooldown
        time_since_last = time.time() - state.last_export_time
        if time_since_last < self.config.export_cooldown_seconds:
            remaining = self.config.export_cooldown_seconds - time_since_last
            logger.debug(
                f"[AutoExportDaemon] {config_key}: Cooldown active, "
                f"{remaining:.0f}s remaining"
            )
            return

        # Trigger export
        safe_create_task(
            self._run_export(config_key),
            name=f"export_{config_key}",
        )

    async def _run_export(self, config_key: str) -> bool:
        """Run the export subprocess for a configuration."""
        state = self._export_states.get(config_key)
        if not state:
            return False

        async with self._export_semaphore:
            state.export_in_progress = True

            try:
                logger.info(
                    f"[AutoExportDaemon] Starting export for {config_key} "
                    f"({state.games_since_last_export} games pending)"
                )

                # Emit NPZ_EXPORT_STARTED event
                await self._emit_export_started(config_key, state.games_since_last_export)

                # Build export command
                base_dir = Path(__file__).resolve().parent.parent.parent
                script_path = base_dir / "scripts" / "export_replay_dataset.py"
                output_path = self.config.output_dir / f"{config_key}.npz"

                cmd = [
                    sys.executable,
                    str(script_path),
                    "--use-discovery",
                    "--board-type", state.board_type,
                    "--num-players", str(state.num_players),
                    "--output", str(output_path),
                    "--allow-noncanonical",  # Allow any database, not just registry
                ]

                if self.config.use_incremental_export:
                    cmd.append("--use-cache")

                if self.config.require_completed_games:
                    cmd.append("--require-completed")

                if self.config.min_moves > 0:
                    cmd.extend(["--min-moves", str(self.config.min_moves)])

                # Run export subprocess
                start_time = time.time()
                # December 29, 2025: Add RINGRIFT_ALLOW_PENDING_GATE to bypass parity
                # validation on cluster nodes that lack Node.js/npx
                export_env = {
                    **os.environ,
                    "PYTHONPATH": str(base_dir),
                    "RINGRIFT_ALLOW_PENDING_GATE": "true",
                }
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(base_dir),
                    env=export_env,
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.config.export_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    logger.error(f"[AutoExportDaemon] Export timed out for {config_key}")
                    state.consecutive_failures += 1
                    self._save_state(config_key)  # Persist failure count (Phase 8)
                    return False

                duration = time.time() - start_time

                if process.returncode == 0:
                    # Success
                    state.last_export_time = time.time()
                    state.last_export_games = state.games_since_last_export
                    state.games_since_last_export = 0
                    state.consecutive_failures = 0

                    # Parse sample count from output if available
                    samples = self._parse_sample_count(stdout.decode())
                    if samples:
                        state.total_exported_samples = samples

                    logger.info(
                        f"[AutoExportDaemon] Export complete for {config_key}: "
                        f"{state.last_export_games} games, {samples or '?'} samples, "
                        f"{duration:.1f}s"
                    )

                    # Emit completion event
                    await self._emit_export_complete(config_key, output_path, samples)

                    # December 2025: Emit NEW_GAMES_AVAILABLE for pipeline coordination
                    # This triggers downstream consumers (training, evaluation) immediately
                    await self._emit_new_games_available(config_key, samples)
                    return True

                else:
                    # Failure
                    state.consecutive_failures += 1
                    logger.error(
                        f"[AutoExportDaemon] Export failed for {config_key}: "
                        f"exit code {process.returncode}\n"
                        f"stderr: {stderr.decode()[:500]}"
                    )
                    return False

            except Exception as e:  # noqa: BLE001
                state.consecutive_failures += 1
                logger.error(f"[AutoExportDaemon] Export error for {config_key}: {e}")
                return False

            finally:
                state.export_in_progress = False
                # Persist updated state (Phase 8)
                self._save_state(config_key)

    def _parse_sample_count(self, output: str) -> int | None:
        """Parse sample count from export script output."""
        import re

        # Look for patterns like "Exported 12345 samples" or "samples: 12345"
        patterns = [
            r"Exported\s+(\d+)\s+samples",
            r"samples:\s*(\d+)",
            r"Total samples:\s*(\d+)",
            r"(\d+)\s+samples",
        ]

        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    async def _emit_export_started(
        self, config_key: str, games_pending: int
    ) -> None:
        """Emit NPZ_EXPORT_STARTED event."""
        try:
            from app.coordination.event_router import (
                StageCompletionResult,
                StageEvent,
                get_stage_event_bus,
            )

            # Dec 30, 2025: Use consolidated parse_config_key utility
            parsed = parse_config_key(config_key)
            board_type = parsed.board_type if parsed else config_key
            num_players = parsed.num_players if parsed else 2

            bus = get_stage_event_bus()
            await bus.emit(
                StageCompletionResult(
                    event=StageEvent.NPZ_EXPORT_STARTED,
                    success=True,
                    iteration=0,  # Export is not iteration-based
                    timestamp=datetime.datetime.now().isoformat(),
                    board_type=board_type,
                    num_players=num_players,
                    metadata={
                        "config": config_key,
                        "games_pending": games_pending,
                    },
                )
            )
            logger.debug(f"[AutoExportDaemon] Emitted NPZ_EXPORT_STARTED for {config_key}")

        except Exception as e:  # noqa: BLE001
            logger.warning(f"[AutoExportDaemon] Failed to emit export started event: {e}")

    async def _emit_export_complete(
        self, config_key: str, output_path: Path, samples: int | None
    ) -> None:
        """Emit NPZ_EXPORT_COMPLETE event."""
        try:
            from app.coordination.event_router import (
                StageCompletionResult,
                StageEvent,
                get_stage_event_bus,
            )

            state = self._export_states.get(config_key)
            # Dec 30, 2025: Use consolidated parse_config_key utility
            parsed = parse_config_key(config_key)
            board_type = parsed.board_type if parsed else config_key
            num_players = parsed.num_players if parsed else 2

            bus = get_stage_event_bus()
            await bus.emit(
                StageCompletionResult(
                    event=StageEvent.NPZ_EXPORT_COMPLETE,
                    success=True,
                    iteration=0,  # Export is not iteration-based
                    timestamp=datetime.datetime.now().isoformat(),
                    board_type=board_type,
                    num_players=num_players,
                    metadata={
                        "config": config_key,
                        "output_path": str(output_path),
                        "samples": samples,
                        "games_exported": state.last_export_games if state else 0,
                    },
                )
            )
            logger.debug(f"[AutoExportDaemon] Emitted NPZ_EXPORT_COMPLETE for {config_key}")

        except Exception as e:  # noqa: BLE001
            logger.warning(f"[AutoExportDaemon] Failed to emit export complete event: {e}")

    async def _emit_new_games_available(
        self, config_key: str, samples: int | None
    ) -> None:
        """Emit NEW_GAMES_AVAILABLE event for pipeline coordination.

        December 2025: Part of 48-hour autonomous operation optimization.
        This event signals to downstream consumers (training, evaluation)
        that new training data is immediately available, enabling reactive
        dispatch instead of polling-based detection.
        """
        try:
            from app.distributed.data_events import DataEvent, DataEventType
            from app.coordination.event_router import get_event_bus

            state = self._export_states.get(config_key)
            parts = config_key.rsplit("_", 1)
            board_type = parts[0] if len(parts) == 2 else config_key
            num_players = int(parts[1].replace("p", "")) if len(parts) == 2 else 2

            event = DataEvent(
                event_type=DataEventType.NEW_GAMES_AVAILABLE,
                payload={
                    "config_key": config_key,
                    "board_type": board_type,
                    "num_players": num_players,
                    "samples": samples,
                    "games_exported": state.last_export_games if state else 0,
                    "source": "auto_export_daemon",
                    "trigger": "event_driven_batch",
                },
                source="AutoExportDaemon",
            )

            bus = get_event_bus()
            await bus.publish(event)
            logger.debug(
                f"[AutoExportDaemon] Emitted NEW_GAMES_AVAILABLE for {config_key} "
                f"({samples or 0} samples)"
            )

        except Exception as e:  # noqa: BLE001
            logger.warning(f"[AutoExportDaemon] Failed to emit new games available event: {e}")

    async def _run_cycle(self) -> None:
        """Main work loop iteration - called by HandlerBase at 5 minute intervals."""
        # Skip if we're on a coordinator node
        if self._coordinator_skip:
            return

        # Scan for databases that may need export
        await self._scan_for_pending_exports()

    async def _scan_for_pending_exports(self) -> None:
        """Scan databases to find configs needing export."""
        try:
            from app.utils.game_discovery import GameDiscovery

            discovery = GameDiscovery()
            databases = discovery.find_all_databases()

            for db_info in databases:
                if not db_info.board_type or not db_info.num_players:
                    continue

                config_key = f"{db_info.board_type}_{db_info.num_players}p"

                if config_key not in self._export_states:
                    self._export_states[config_key] = ConfigExportState(
                        config_key=config_key,
                        board_type=db_info.board_type,
                        num_players=db_info.num_players,
                        games_since_last_export=db_info.game_count,
                    )
                else:
                    # Update game count from discovery
                    state = self._export_states[config_key]
                    # Only update if discovery shows more games
                    if db_info.game_count > state.games_since_last_export:
                        state.games_since_last_export = db_info.game_count

                # Check if export needed
                await self._maybe_trigger_export(config_key)

        except ImportError:
            logger.debug("[AutoExportDaemon] GameDiscovery not available")
        except Exception as e:  # noqa: BLE001
            logger.error(f"[AutoExportDaemon] Error scanning databases: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get current daemon status."""
        return {
            "running": self._running,
            "configs_tracked": len(self._export_states),
            "states": {
                key: {
                    "games_pending": state.games_since_last_export,
                    "last_export": state.last_export_time,
                    "last_export_games": state.last_export_games,
                    "total_samples": state.total_exported_samples,
                    "in_progress": state.export_in_progress,
                    "failures": state.consecutive_failures,
                }
                for key, state in self._export_states.items()
            },
        }

    def health_check(self) -> HealthCheckResult:
        """Check daemon health (December 2025: CoordinatorProtocol compliance).

        Overrides HandlerBase.health_check() with export-specific logic.

        Returns:
            HealthCheckResult with status and details
        """
        from app.coordination.contracts import CoordinatorStatus

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="AutoExport daemon not running",
            )

        # Check for high failure rate
        total_failures = sum(s.consecutive_failures for s in self._export_states.values())
        if total_failures > 5:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"AutoExport daemon has {total_failures} consecutive failures",
                details=self.get_status(),
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"AutoExport daemon running ({len(self._export_states)} configs tracked)",
            details=self.get_status(),
        )


# December 2025: Using HandlerBase singleton pattern
def get_auto_export_daemon() -> AutoExportDaemon:
    """Get or create the singleton auto export daemon.

    December 2025: Now uses HandlerBase.get_instance() singleton pattern.
    """
    return AutoExportDaemon.get_instance()


def reset_auto_export_daemon() -> None:
    """Reset the singleton instance (for testing).

    December 2025: Added for test isolation.
    """
    AutoExportDaemon.reset_instance()


async def start_auto_export_daemon() -> AutoExportDaemon:
    """Start the auto export daemon (convenience function)."""
    daemon = get_auto_export_daemon()
    await daemon.start()
    return daemon


__all__ = [
    "AutoExportConfig",
    "AutoExportDaemon",
    "BatchAccumulator",
    "ConfigExportState",
    "get_auto_export_daemon",
    "reset_auto_export_daemon",
    "start_auto_export_daemon",
]
