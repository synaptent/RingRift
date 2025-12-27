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
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.async_context import safe_create_task

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


class AutoExportDaemon:
    """Daemon that automatically exports training data when thresholds are met."""

    def __init__(self, config: AutoExportConfig | None = None):
        self.config = config or AutoExportConfig()
        self._running = False
        self._task: asyncio.Task | None = None
        self._export_states: dict[str, ConfigExportState] = {}
        self._export_semaphore = asyncio.Semaphore(self.config.max_concurrent_exports)
        self._event_subscriptions: list[Any] = []
        self._state_db_initialized = False
        # December 2025: Deduplication guard - when StageEvent subscriptions are active,
        # skip DataEventType handlers to prevent double-counting games
        self._stage_events_active = False
        # December 2025 Phase 3: Track configs pending sync completion.
        # Export is gated on sync completion to prevent race conditions where
        # export starts before data from other nodes has arrived.
        self._pending_sync_configs: set[str] = set()

    async def start(self) -> None:
        """Start the auto export daemon."""
        # December 2025: Coordinator-only mode check
        # Export is CPU-intensive - should NEVER run on coordinator nodes
        from app.config.env import env
        if env.is_coordinator or not env.export_enabled:
            logger.info(
                f"[AutoExportDaemon] Skipped on coordinator node: {env.node_id} "
                f"(is_coordinator={env.is_coordinator}, export_enabled={env.export_enabled})"
            )
            return

        if self._running:
            logger.warning("[AutoExportDaemon] Already running")
            return

        self._running = True
        logger.info("[AutoExportDaemon] Starting auto export daemon")

        # Initialize state persistence and load previous state (Phase 8)
        if self.config.persist_state:
            self._init_state_db()
            self._load_state()

        # Subscribe to selfplay events
        await self._subscribe_to_events()

        # Start background monitoring task
        self._task = safe_create_task(
            self._monitor_loop(),
            name="auto_export_monitor",
        )
        self._task.add_done_callback(self._on_task_done)

    async def stop(self) -> None:
        """Stop the auto export daemon."""
        self._running = False

        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

        # Unsubscribe from events
        for unsub in self._event_subscriptions:
            try:
                if callable(unsub):
                    unsub()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[AutoExportDaemon] Error unsubscribing: {e}")

        logger.info("[AutoExportDaemon] Stopped")

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Handle task completion or failure."""
        try:
            exc = task.exception()
            if exc:
                logger.error(f"[AutoExportDaemon] Task failed: {exc}")
        except asyncio.CancelledError:
            pass
        except asyncio.InvalidStateError:
            pass

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

    # ========== Event Subscriptions ==========

    async def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events.

        Phase 3A.3 (Dec 2025): Now subscribes to SYNC_COMPLETE to trigger
        export when games are synced from other nodes.

        December 2025: Added deduplication guard. When StageEvent subscriptions
        succeed, we set _stage_events_active=True to skip duplicate processing
        from DataEventType handlers.
        """
        try:
            # P0.5 (December 2025): Use get_router() instead of deprecated get_stage_event_bus()
            from app.coordination.event_router import StageEvent, get_router

            router = get_router()
            unsub = router.subscribe(StageEvent.SELFPLAY_COMPLETE, self._on_selfplay_complete)
            self._event_subscriptions.append(unsub)
            # Mark stage events active to prevent duplicate handling from DataEventType
            self._stage_events_active = True
            logger.info("[AutoExportDaemon] Subscribed to SELFPLAY_COMPLETE (StageEvent)")

            # Phase 3A.3: Also subscribe to SYNC_COMPLETE for cross-node data
            unsub = router.subscribe(StageEvent.SYNC_COMPLETE, self._on_sync_complete)
            self._event_subscriptions.append(unsub)
            logger.info("[AutoExportDaemon] Subscribed to SYNC_COMPLETE events")
        except ImportError:
            logger.warning("[AutoExportDaemon] Stage events not available")

        try:
            from app.coordination.event_router import DataEventType, get_router

            router = get_router()
            router.subscribe(DataEventType.NEW_GAMES_AVAILABLE, self._on_new_games)
            router.subscribe(DataEventType.SELFPLAY_COMPLETE, self._on_selfplay_complete_event)
            router.subscribe(DataEventType.DATA_SYNC_COMPLETED, self._on_data_sync_completed)
            logger.info(
                "[AutoExportDaemon] Subscribed to NEW_GAMES_AVAILABLE, SELFPLAY_COMPLETE, "
                "DATA_SYNC_COMPLETED events"
            )
        except ImportError:
            logger.warning("[AutoExportDaemon] Data events not available")

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
            config_key = payload.get("config_key") or payload.get("config")
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
        """Parse a config key like "square8_2p" into (board_type, num_players)."""
        if "_" not in config_key or not config_key.endswith("p"):
            return None, None

        parts = config_key.rsplit("_", 1)
        if len(parts) != 2:
            return None, None

        board_type = parts[0]
        try:
            num_players = int(parts[1].replace("p", ""))
        except ValueError:
            return None, None

        return board_type, num_players
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
                ]

                if self.config.use_incremental_export:
                    cmd.append("--use-cache")

                if self.config.require_completed_games:
                    cmd.append("--require-completed")

                if self.config.min_moves > 0:
                    cmd.extend(["--min-moves", str(self.config.min_moves)])

                # Run export subprocess
                start_time = time.time()
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(base_dir),
                    env={**dict(__import__("os").environ), "PYTHONPATH": str(base_dir)},
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

            parts = config_key.rsplit("_", 1)
            board_type = parts[0] if len(parts) == 2 else config_key
            num_players = int(parts[1].replace("p", "")) if len(parts) == 2 else 2

            bus = get_stage_event_bus()
            await bus.emit(
                StageCompletionResult(
                    event=StageEvent.NPZ_EXPORT_STARTED,
                    success=True,
                    iteration=0,  # Export is not iteration-based
                    timestamp=__import__("datetime").datetime.now().isoformat(),
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
            parts = config_key.rsplit("_", 1)
            board_type = parts[0] if len(parts) == 2 else config_key
            num_players = int(parts[1].replace("p", "")) if len(parts) == 2 else 2

            bus = get_stage_event_bus()
            await bus.emit(
                StageCompletionResult(
                    event=StageEvent.NPZ_EXPORT_COMPLETE,
                    success=True,
                    iteration=0,  # Export is not iteration-based
                    timestamp=__import__("datetime").datetime.now().isoformat(),
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

    async def _monitor_loop(self) -> None:
        """Background loop to periodically check for pending exports."""
        while self._running:
            try:
                # Scan for databases that may need export
                await self._scan_for_pending_exports()

                # Wait before next scan
                await asyncio.sleep(300)  # 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                logger.error(f"[AutoExportDaemon] Monitor loop error: {e}")
                await asyncio.sleep(60)

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

    def health_check(self) -> "HealthCheckResult":
        """Check daemon health (December 2025: CoordinatorProtocol compliance).

        Returns:
            HealthCheckResult with status and details
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

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


# Singleton instance
_daemon: AutoExportDaemon | None = None


def get_auto_export_daemon() -> AutoExportDaemon:
    """Get or create the singleton auto export daemon."""
    global _daemon
    if _daemon is None:
        _daemon = AutoExportDaemon()
    return _daemon


async def start_auto_export_daemon() -> AutoExportDaemon:
    """Start the auto export daemon (convenience function)."""
    daemon = get_auto_export_daemon()
    await daemon.start()
    return daemon


__all__ = [
    "AutoExportConfig",
    "AutoExportDaemon",
    "ConfigExportState",
    "get_auto_export_daemon",
    "start_auto_export_daemon",
]
