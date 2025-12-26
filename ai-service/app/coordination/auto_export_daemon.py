"""Automated NPZ Export Daemon (December 2025).

Automatically exports training data (NPZ files) when game counts exceed thresholds.
This closes the gap between selfplay and training by eliminating the manual export step.

Key features:
- Subscribes to SELFPLAY_COMPLETE events
- Tracks accumulated games per configuration
- Triggers export when game threshold reached (default: 500 games)
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
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AutoExportConfig:
    """Configuration for automated NPZ export."""

    enabled: bool = True
    # Minimum games before triggering export
    min_games_threshold: int = 500
    # Cooldown between exports for same config (seconds)
    export_cooldown_seconds: int = 1800  # 30 minutes
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

    async def start(self) -> None:
        """Start the auto export daemon."""
        if self._running:
            logger.warning("[AutoExportDaemon] Already running")
            return

        self._running = True
        logger.info("[AutoExportDaemon] Starting auto export daemon")

        # Subscribe to selfplay events
        await self._subscribe_to_events()

        # Start background monitoring task
        self._task = asyncio.create_task(self._monitor_loop())
        self._task.add_done_callback(self._on_task_done)

    async def stop(self) -> None:
        """Stop the auto export daemon."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Unsubscribe from events
        for unsub in self._event_subscriptions:
            try:
                if callable(unsub):
                    unsub()
            except Exception as e:
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

    async def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events."""
        try:
            from app.coordination.stage_events import StageEvent, get_event_bus

            bus = get_event_bus()
            unsub = bus.subscribe(StageEvent.SELFPLAY_COMPLETE, self._on_selfplay_complete)
            self._event_subscriptions.append(unsub)
            logger.info("[AutoExportDaemon] Subscribed to SELFPLAY_COMPLETE events")
        except ImportError:
            logger.warning("[AutoExportDaemon] Stage events not available")

        try:
            from app.distributed.data_events import DataEventType, get_event_bus

            bus = get_event_bus()
            unsub = bus.subscribe(DataEventType.NEW_GAMES_AVAILABLE, self._on_new_games)
            self._event_subscriptions.append(unsub)
            logger.info("[AutoExportDaemon] Subscribed to NEW_GAMES_AVAILABLE events")
        except ImportError:
            logger.warning("[AutoExportDaemon] Data events not available")

    async def _on_selfplay_complete(self, result: Any) -> None:
        """Handle selfplay completion event."""
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
            await self._record_games(config_key, board_type, num_players, games_generated)

        except Exception as e:
            logger.error(f"[AutoExportDaemon] Error handling selfplay complete: {e}")

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

        except Exception as e:
            logger.error(f"[AutoExportDaemon] Error handling new games event: {e}")

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
        asyncio.create_task(self._run_export(config_key))

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

            except Exception as e:
                state.consecutive_failures += 1
                logger.error(f"[AutoExportDaemon] Export error for {config_key}: {e}")
                return False

            finally:
                state.export_in_progress = False

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

    async def _emit_export_complete(
        self, config_key: str, output_path: Path, samples: int | None
    ) -> None:
        """Emit NPZ_EXPORT_COMPLETE event."""
        try:
            from app.coordination.stage_events import (
                StageEvent,
                StageCompletionResult,
                get_event_bus,
            )

            state = self._export_states.get(config_key)
            parts = config_key.rsplit("_", 1)
            board_type = parts[0] if len(parts) == 2 else config_key
            num_players = int(parts[1].replace("p", "")) if len(parts) == 2 else 2

            bus = get_event_bus()
            await bus.emit(
                StageCompletionResult(
                    event=StageEvent.NPZ_EXPORT_COMPLETE,
                    success=True,
                    timestamp=__import__("datetime").datetime.now().isoformat(),
                    metadata={
                        "config": config_key,
                        "board_type": board_type,
                        "num_players": num_players,
                        "output_path": str(output_path),
                        "samples": samples,
                        "games_exported": state.last_export_games if state else 0,
                    },
                )
            )
            logger.debug(f"[AutoExportDaemon] Emitted NPZ_EXPORT_COMPLETE for {config_key}")

        except Exception as e:
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
            except Exception as e:
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
        except Exception as e:
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
