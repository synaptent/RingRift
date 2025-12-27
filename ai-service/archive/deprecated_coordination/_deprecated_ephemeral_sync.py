"""Ephemeral Sync Daemon - Aggressive sync for ephemeral hosts.

.. deprecated:: December 2025
    This module is deprecated in favor of AutoSyncDaemon with strategy="ephemeral".
    Use: ``from app.coordination.auto_sync_daemon import create_ephemeral_sync_daemon``

Ephemeral hosts (Vast.ai, spot instances) can be terminated with 15-30 seconds
notice. This daemon provides aggressive sync to minimize data loss:

- 5-second poll interval (vs 60s for persistent hosts)
- Immediate push on game completion
- Termination signal handling with final sync
- Priority flag for cluster sync

Usage (DEPRECATED):
    from app.coordination.ephemeral_sync import EphemeralSyncDaemon

    daemon = EphemeralSyncDaemon()
    await daemon.start()

    # On game completion
    await daemon.on_game_complete(game_result)

New Usage (RECOMMENDED):
    from app.coordination.auto_sync_daemon import (
        create_ephemeral_sync_daemon,
        is_ephemeral_host,
    )

    daemon = create_ephemeral_sync_daemon()
    await daemon.start()

    # On game completion
    await daemon.on_game_complete(game_result)

Integration:
    # Automatically started by selfplay runners on ephemeral hosts
    from app.training.selfplay_runner import SelfplayRunner

    runner = SelfplayRunner(...)
    # Runner auto-detects ephemeral host and starts EphemeralSyncDaemon
"""

from __future__ import annotations

import warnings

warnings.warn(
    "ephemeral_sync module is deprecated as of December 2025. "
    "Use auto_sync_daemon.create_ephemeral_sync_daemon() instead.",
    DeprecationWarning,
    stacklevel=2,
)

import asyncio
import logging
import os
import signal
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.core.async_context import safe_create_task

logger = logging.getLogger(__name__)

__all__ = [
    # Data classes
    "EphemeralSyncConfig",
    "EphemeralSyncStats",
    # Main class
    "EphemeralSyncDaemon",
    # Singleton accessors
    "get_ephemeral_sync_daemon",
    "reset_ephemeral_sync_daemon",
    # Utility functions
    "is_ephemeral_host",
]


@dataclass
class EphemeralSyncConfig:
    """Configuration for ephemeral sync."""
    enabled: bool = True
    poll_interval_seconds: int = 5  # Very aggressive
    immediate_push_enabled: bool = True
    termination_handler_enabled: bool = True
    min_games_before_push: int = 1  # Push even single games
    max_concurrent_syncs: int = 2
    # Dec 2025: Use centralized timeout from thresholds.py
    sync_timeout_seconds: int = 30  # Overridden at runtime by EPHEMERAL_SYNC_TIMEOUT
    # December 2025: Write-through mode for zero data loss
    write_through_enabled: bool = True  # Wait for push confirmation
    # P1.3 Dec 2025: Increased from 30s to 60s for Vast.ai high-latency nodes (50-500ms)
    write_through_timeout_seconds: int = 60  # Max wait for confirmation
    # Phase 8 Dec 2025: Write-ahead log for pending games
    wal_enabled: bool = True  # Persist pending games to survive crashes
    wal_path: Path = field(default_factory=lambda: Path("data/ephemeral_sync_wal.jsonl"))


@dataclass
class EphemeralSyncStats:
    """Statistics for ephemeral sync."""
    games_pushed: int = 0
    bytes_transferred: int = 0
    immediate_pushes: int = 0
    poll_pushes: int = 0
    termination_syncs: int = 0
    failed_syncs: int = 0
    last_push_time: float = 0.0
    last_error: str | None = None
    # December 2025: Write-through tracking
    write_through_successes: int = 0
    write_through_failures: int = 0
    games_at_risk_on_termination: int = 0  # Games not yet synced when terminated
    games_confirmed_synced: int = 0  # Games with confirmed push


class EphemeralSyncDaemon:
    """Daemon for aggressive sync on ephemeral hosts.

    Designed for Vast.ai and spot instances with short termination notice.
    Uses more aggressive sync settings to minimize data loss.
    """

    def __init__(
        self,
        config: EphemeralSyncConfig | None = None,
        on_termination: Callable[[], None] | None = None,
    ):
        """Initialize ephemeral sync daemon.

        Args:
            config: Sync configuration
            on_termination: Optional callback for termination handling
        """
        self.config = config or EphemeralSyncConfig()
        self.node_id = socket.gethostname()
        self._running = False
        self._stats = EphemeralSyncStats()
        self._poll_task: asyncio.Task | None = None
        self._pending_games: list[dict[str, Any]] = []
        self._push_lock = asyncio.Lock()
        self._termination_callback = on_termination
        self._wal_initialized = False

        # Detect if we're actually on an ephemeral host
        self._is_ephemeral = self._detect_ephemeral()

        # Sync targets (populated on start)
        self._sync_targets: list[str] = []

        # Initialize WAL and recover pending games (Phase 8)
        if self.config.wal_enabled:
            self._init_wal()
            self._load_wal()

        if self._is_ephemeral:
            logger.info(f"EphemeralSyncDaemon initialized on {self.node_id}")
        else:
            logger.debug(f"EphemeralSyncDaemon on non-ephemeral host {self.node_id}")

    def _detect_ephemeral(self) -> bool:
        """Detect if running on an ephemeral host."""
        # Check Vast.ai
        if Path("/workspace").exists():
            return True

        # Check provider via canonical detection
        from app.config.cluster_config import get_host_provider
        if get_host_provider(self.node_id) == "vast":
            return True

        # Check for spot instance markers
        if os.environ.get("AWS_SPOT_INSTANCE"):
            return True

        # Check RAM disk (Vast.ai uses /dev/shm for temp storage)
        if Path("/dev/shm/ringrift").exists():
            return True

        return False

    # ========== Write-Ahead Log (Phase 8 Dec 2025) ==========

    def _init_wal(self) -> None:
        """Initialize write-ahead log for pending games.

        Phase 8 Dec 2025: Persists pending game IDs to survive daemon crashes.
        Uses JSONL format for append-only writes (fast, robust).
        """
        import json

        try:
            wal_path = self.config.wal_path
            wal_path.parent.mkdir(parents=True, exist_ok=True)

            # Create empty WAL if it doesn't exist
            if not wal_path.exists():
                wal_path.touch()

            self._wal_initialized = True
            logger.debug(f"[EphemeralSync] WAL initialized: {wal_path}")

        except Exception as e:
            logger.error(f"[EphemeralSync] Failed to initialize WAL: {e}")
            self._wal_initialized = False

    def _load_wal(self) -> None:
        """Load pending games from WAL on startup.

        Recovers games that were pending when daemon crashed or was terminated.
        """
        import json

        if not self._wal_initialized:
            return

        try:
            wal_path = self.config.wal_path
            if not wal_path.exists() or wal_path.stat().st_size == 0:
                return

            loaded_count = 0
            with open(wal_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        # Only load entries not marked as synced
                        if not entry.get("synced", False):
                            self._pending_games.append(entry)
                            loaded_count += 1
                    except json.JSONDecodeError:
                        continue

            if loaded_count > 0:
                logger.info(
                    f"[EphemeralSync] Recovered {loaded_count} pending games from WAL"
                )

        except Exception as e:
            logger.error(f"[EphemeralSync] Failed to load WAL: {e}")

    def _append_to_wal(self, game_entry: dict[str, Any]) -> None:
        """Append pending game to WAL for durability.

        Called when a game is added to pending list, before sync attempt.
        """
        import json

        if not self._wal_initialized or not self.config.wal_enabled:
            return

        try:
            with open(self.config.wal_path, 'a') as f:
                f.write(json.dumps(game_entry) + '\n')
                f.flush()  # Ensure durability
                os.fsync(f.fileno())  # Force to disk

        except Exception as e:
            logger.debug(f"[EphemeralSync] Failed to append to WAL: {e}")

    def _clear_wal(self) -> None:
        """Clear WAL after successful sync of all pending games.

        Called when all pending games have been confirmed synced.
        """
        if not self._wal_initialized:
            return

        try:
            # Truncate WAL file
            self.config.wal_path.write_text('')
            logger.debug("[EphemeralSync] WAL cleared after successful sync")

        except Exception as e:
            logger.debug(f"[EphemeralSync] Failed to clear WAL: {e}")

    # ========== Host Detection ==========

    @property
    def is_ephemeral(self) -> bool:
        """Check if this is an ephemeral host."""
        return self._is_ephemeral

    async def start(self) -> None:
        """Start the ephemeral sync daemon."""
        if not self.config.enabled:
            logger.debug("EphemeralSyncDaemon disabled by config")
            return

        if not self._is_ephemeral:
            logger.debug("Skipping EphemeralSyncDaemon (not ephemeral host)")
            return

        self._running = True

        # Setup termination handlers
        if self.config.termination_handler_enabled:
            self._setup_termination_handlers()

        # Find sync targets
        await self._discover_sync_targets()

        # Start poll loop
        self._poll_task = safe_create_task(
            self._poll_loop(),
            name="ephemeral_sync_poll",
        )

        logger.info(
            f"EphemeralSyncDaemon started: "
            f"interval={self.config.poll_interval_seconds}s, "
            f"targets={len(self._sync_targets)}"
        )

    async def stop(self) -> None:
        """Stop the ephemeral sync daemon."""
        logger.info("Stopping EphemeralSyncDaemon...")
        self._running = False

        # Do final sync
        await self._final_sync()

        # Stop poll task
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

        logger.info("EphemeralSyncDaemon stopped")

    async def sync_now(self) -> int:
        """Trigger an immediate sync/push.

        Dec 2025: Added to expose sync functionality to sync_facade.py.

        Returns:
            Number of games pushed.
        """
        if not self._running:
            logger.warning("[EphemeralSyncDaemon] sync_now() called but daemon not running")
            return 0

        # Force push any pending data
        await self._push_to_targets(force=True)
        return self._stats.games_pushed

    def _setup_termination_handlers(self) -> None:
        """Setup signal handlers for termination."""
        def handle_termination(sig, frame):
            logger.warning(f"Received termination signal {sig}")
            # Run final sync in separate event loop if needed
            try:
                # Dec 2025: Use get_running_loop() instead of deprecated get_event_loop()
                loop = asyncio.get_running_loop()
                # Schedule the sync with error handling
                safe_create_task(
                    self._handle_termination(),
                    name="ephemeral_termination_sync",
                )
            except RuntimeError:
                # No running loop - create one
                try:
                    asyncio.run(self._handle_termination())
                except Exception as e:
                    logger.error(f"Cannot run final sync: {e}")

        # Handle common termination signals
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, handle_termination)
            except Exception as e:
                logger.debug(f"Could not set handler for {sig}: {e}")

    async def _handle_termination(self) -> None:
        """Handle termination signal."""
        logger.warning("Handling termination - starting final sync")
        self._stats.termination_syncs += 1

        # Phase 5 (Dec 2025): Notify coordinator of impending termination
        # so work can be migrated to other nodes
        await self._emit_termination_event()

        # Run callback if provided
        if self._termination_callback:
            try:
                self._termination_callback()
            except Exception as e:
                logger.error(f"Termination callback error: {e}")

        # Do final sync
        await self._final_sync()

    async def _emit_termination_event(self) -> None:
        """Emit HOST_OFFLINE event to notify coordinator of termination.

        Phase 5 (Dec 2025): Enables coordinator to migrate work before
        this ephemeral node terminates, reducing job loss.
        """
        try:
            from app.coordination.event_router import emit_host_offline

            await emit_host_offline(
                host=self.node_id,
                reason=f"ephemeral_termination:pending_games={len(self._pending_games)}",
                last_seen=time.time(),
                source="ephemeral_sync_daemon",
            )
            logger.info(f"Emitted HOST_OFFLINE event for {self.node_id}")

        except ImportError:
            logger.debug("emit_host_offline not available")
        except Exception as e:
            logger.warning(f"Failed to emit termination event: {e}")

    async def _final_sync(self) -> None:
        """Perform final sync before shutdown."""
        if not self._pending_games:
            logger.debug("No pending games for final sync")
            return

        logger.info(f"Final sync: {len(self._pending_games)} games pending")

        try:
            await self._push_to_targets(force=True)
        except Exception as e:
            logger.error(f"Final sync failed: {e}")

    async def _discover_sync_targets(self) -> None:
        """Discover sync targets from SyncRouter."""
        try:
            from app.coordination.sync_router import get_sync_router

            router = get_sync_router()
            targets = router.get_sync_targets(
                data_type="game",
                max_targets=3,  # Only need a few targets
            )

            self._sync_targets = [t.node_id for t in targets]

            if self._sync_targets:
                logger.info(f"Sync targets: {self._sync_targets}")
            else:
                logger.warning("No sync targets found")

        except ImportError:
            logger.warning("SyncRouter not available")
        except Exception as e:
            logger.error(f"Failed to discover sync targets: {e}")

    async def _poll_loop(self) -> None:
        """Main poll loop for sync."""
        while self._running:
            try:
                await self._poll_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._stats.failed_syncs += 1
                self._stats.last_error = str(e)
                logger.error(f"Poll cycle error: {e}")

            await asyncio.sleep(self.config.poll_interval_seconds)

    async def _poll_cycle(self) -> None:
        """Execute one poll cycle."""
        if not self._pending_games:
            return

        if len(self._pending_games) >= self.config.min_games_before_push:
            self._stats.poll_pushes += 1
            await self._push_to_targets()

    async def on_game_complete(
        self,
        game_result: dict[str, Any],
        db_path: Path | str | None = None,
    ) -> bool:
        """Handle game completion - queue for immediate push.

        December 2025: Enhanced with write-through mode for zero data loss.
        When write_through_enabled=True, waits for push confirmation before
        returning, ensuring the game is safely synced to a persistent node.

        Args:
            game_result: Game result dict with game_id, moves, etc.
            db_path: Path to database containing the game

        Returns:
            True if game was successfully synced (write-through mode) or queued,
            False if write-through failed (data at risk)
        """
        game_id = game_result.get("game_id")

        # Add to pending
        game_entry = {
            "game_id": game_id,
            "db_path": str(db_path) if db_path else None,
            "timestamp": time.time(),
            "synced": False,  # Track sync status
        }
        # Dec 2025: Write WAL FIRST for durability - if crash occurs between
        # WAL write and memory append, game is recoverable from WAL on restart.
        # The reverse order (memory first, then WAL) would lose data on crash.
        self._append_to_wal(game_entry)
        self._pending_games.append(game_entry)

        # Immediate push if enabled
        if self.config.immediate_push_enabled and len(self._pending_games) >= 1:
            self._stats.immediate_pushes += 1

            # December 2025: Write-through mode - wait for confirmation
            if self.config.write_through_enabled:
                try:
                    success = await asyncio.wait_for(
                        self._push_to_targets_with_confirmation(),
                        timeout=self.config.write_through_timeout_seconds,
                    )
                    if success:
                        self._stats.write_through_successes += 1
                        self._stats.games_confirmed_synced += 1
                        logger.debug(f"Write-through success for game {game_id}")
                        return True
                    else:
                        self._stats.write_through_failures += 1
                        logger.warning(f"Write-through push failed for game {game_id}")
                        return False
                except asyncio.TimeoutError:
                    self._stats.write_through_failures += 1
                    logger.warning(
                        f"Write-through timeout for game {game_id} "
                        f"(timeout={self.config.write_through_timeout_seconds}s)"
                    )
                    # Fall back to async push with error handling
                    safe_create_task(
                        self._push_to_targets(),
                        name="ephemeral_fallback_push",
                    )
                    return False
            else:
                # Legacy async push (fire-and-forget)
                await self._push_to_targets()
                return True

        return True

    async def _push_to_targets_with_confirmation(self) -> bool:
        """Push pending games and return True if at least one target succeeds.

        December 2025: Write-through variant that returns sync status.
        """
        if not self._pending_games:
            return True

        if not self._sync_targets:
            logger.warning("No sync targets available")
            return False

        async with self._push_lock:
            games_to_push = self._pending_games.copy()
            self._pending_games.clear()

            logger.info(f"Write-through: pushing {len(games_to_push)} games")

            # Get unique DB paths
            db_paths = set()
            for game in games_to_push:
                if game.get("db_path"):
                    db_paths.add(game["db_path"])

            if not db_paths:
                logger.warning("No database paths to push")
                return False

            # Push each DB to at least one target
            any_success = False
            for db_path in db_paths:
                for target in self._sync_targets:
                    try:
                        success = await self._rsync_to_target(db_path, target)
                        if success:
                            self._stats.games_pushed += len(games_to_push)
                            self._stats.last_push_time = time.time()
                            any_success = True
                            # Mark games as synced
                            for game in games_to_push:
                                game["synced"] = True
                            break  # Only need one successful target
                    except Exception as e:
                        logger.debug(f"Push to {target} failed: {e}")

            if any_success:
                await self._emit_sync_event(
                    games_pushed=len(games_to_push),
                    target_nodes=self._sync_targets[:1],
                    db_paths=list(db_paths),
                )
                # Clear WAL after successful sync (Phase 8)
                self._clear_wal()

            return any_success

    async def _push_to_targets(self, force: bool = False) -> None:
        """Push pending games to sync targets."""
        if not self._pending_games:
            return

        if not self._sync_targets:
            logger.warning("No sync targets available")
            return

        async with self._push_lock:
            games_to_push = self._pending_games.copy()
            self._pending_games.clear()

            logger.info(f"Pushing {len(games_to_push)} games to {len(self._sync_targets)} targets")

            # Get unique DB paths
            db_paths = set()
            for game in games_to_push:
                if game.get("db_path"):
                    db_paths.add(game["db_path"])

            if not db_paths:
                logger.warning("No database paths to push")
                return

            # Push each DB to targets
            successful_targets = []
            for db_path in db_paths:
                for target in self._sync_targets:
                    try:
                        success = await self._rsync_to_target(db_path, target)
                        if success:
                            self._stats.games_pushed += len(games_to_push)
                            self._stats.last_push_time = time.time()
                            successful_targets.append(target)
                            break  # Only need one successful target
                    except Exception as e:
                        logger.debug(f"Push to {target} failed: {e}")

            # Emit sync event if any pushes succeeded
            if successful_targets:
                await self._emit_sync_event(
                    games_pushed=len(games_to_push),
                    target_nodes=successful_targets,
                    db_paths=list(db_paths),
                )
                # Clear WAL after successful sync (Phase 8)
                self._clear_wal()

    async def _rsync_to_target(self, db_path: str, target_node: str) -> bool:
        """Rsync a database to a target node.

        Args:
            db_path: Local database path
            target_node: Target node ID

        Returns:
            True if successful
        """
        try:
            from app.coordination.sync_bandwidth import rsync_with_bandwidth_limit

            # Get SSH config for target
            from app.coordination.sync_router import get_sync_router
            router = get_sync_router()
            cap = router.get_node_capability(target_node)

            if not cap:
                return False

            # Use sync_bandwidth for rate-limited rsync (Dec 2025: centralized timeout)
            from app.config.thresholds import EPHEMERAL_SYNC_TIMEOUT
            result = rsync_with_bandwidth_limit(
                source=db_path,
                target_host=target_node,
                timeout=EPHEMERAL_SYNC_TIMEOUT,
            )

            return result.success

        except ImportError:
            # Fallback to direct rsync
            return await self._direct_rsync(db_path, target_node)
        except Exception as e:
            logger.debug(f"Rsync error: {e}")
            return False

    async def _direct_rsync(self, db_path: str, target_node: str) -> bool:
        """Direct rsync without bandwidth management."""
        import subprocess
        import yaml

        # Load SSH config
        base_dir = Path(__file__).resolve().parent.parent.parent
        config_path = base_dir / "config" / "distributed_hosts.yaml"

        if not config_path.exists():
            return False

        with open(config_path) as f:
            config = yaml.safe_load(f)

        host_config = config.get("hosts", {}).get(target_node, {})
        ssh_host = host_config.get("tailscale_ip") or host_config.get("ssh_host")
        ssh_user = host_config.get("ssh_user", "ubuntu")
        ssh_key = host_config.get("ssh_key", "~/.ssh/id_cluster")
        remote_path = host_config.get("ringrift_path", "~/ringrift/ai-service")

        if not ssh_host:
            return False

        # Build rsync command
        ssh_key = os.path.expanduser(ssh_key)
        remote_full = f"{ssh_user}@{ssh_host}:{remote_path}/data/games/"

        rsync_cmd = [
            "rsync",
            "-avz",
            "--compress",
            "-e", f"ssh -i {ssh_key} -o StrictHostKeyChecking=no -o ConnectTimeout=10",
            db_path,
            remote_full,
        ]

        try:
            # Dec 2025: Use centralized timeout from thresholds.py
            from app.config.thresholds import EPHEMERAL_SYNC_TIMEOUT
            result = await asyncio.to_thread(
                subprocess.run,
                rsync_cmd,
                capture_output=True,
                text=True,
                timeout=EPHEMERAL_SYNC_TIMEOUT,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.warning(f"Rsync timeout to {target_node}")
            return False
        except Exception as e:
            logger.debug(f"Rsync error: {e}")
            return False

    async def _emit_sync_event(
        self,
        games_pushed: int,
        target_nodes: list[str],
        db_paths: list[str],
    ) -> None:
        """Emit GAME_SYNCED event for feedback loop coupling.

        Args:
            games_pushed: Number of games synced
            target_nodes: Nodes that received the data
            db_paths: Database paths that were synced
        """
        try:
            from app.coordination.event_router import get_router, DataEventType

            router = get_router()
            if router:
                await router.publish(
                    event_type=DataEventType.GAME_SYNCED,
                    payload={
                        "node_id": self.node_id,
                        "games_pushed": games_pushed,
                        "target_nodes": target_nodes,
                        "db_paths": db_paths,
                        "is_ephemeral": self._is_ephemeral,
                        "timestamp": time.time(),
                    },
                    source="EphemeralSyncDaemon",
                )
        except Exception as e:
            logger.debug(f"Could not emit GAME_SYNCED event: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get daemon status."""
        return {
            "node_id": self.node_id,
            "is_ephemeral": self._is_ephemeral,
            "running": self._running,
            "sync_targets": self._sync_targets,
            "pending_games": len(self._pending_games),
            "config": {
                "enabled": self.config.enabled,
                "poll_interval_seconds": self.config.poll_interval_seconds,
                "immediate_push_enabled": self.config.immediate_push_enabled,
                "write_through_enabled": self.config.write_through_enabled,
                "write_through_timeout_seconds": self.config.write_through_timeout_seconds,
            },
            "stats": {
                "games_pushed": self._stats.games_pushed,
                "immediate_pushes": self._stats.immediate_pushes,
                "poll_pushes": self._stats.poll_pushes,
                "termination_syncs": self._stats.termination_syncs,
                "failed_syncs": self._stats.failed_syncs,
                "last_push_time": self._stats.last_push_time,
                "last_error": self._stats.last_error,
                # December 2025: Write-through telemetry
                "write_through_successes": self._stats.write_through_successes,
                "write_through_failures": self._stats.write_through_failures,
                "games_confirmed_synced": self._stats.games_confirmed_synced,
                "games_at_risk_on_termination": self._stats.games_at_risk_on_termination,
            },
        }

    def health_check(self):
        """Check daemon health (December 2025: CoordinatorProtocol compliance).

        Returns:
            HealthCheckResult with status and details
        """
        from app.coordination.protocols import HealthCheckResult, CoordinatorStatus

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="EphemeralSync daemon not running",
            )

        # Check for high failure rate
        total = self._stats.games_pushed
        failed = self._stats.failed_syncs
        if total > 10 and failed / total > 0.3:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"EphemeralSync high failure rate: {failed}/{total} failed",
                details=self.get_status(),
            )

        # Check games at risk
        if self._stats.games_at_risk_on_termination > 100:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message=f"EphemeralSync: {self._stats.games_at_risk_on_termination} games at risk",
                details=self.get_status(),
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"EphemeralSync running ({self._stats.games_pushed} games synced)",
            details=self.get_status(),
        )


# Module-level singleton
_ephemeral_sync_daemon: EphemeralSyncDaemon | None = None


def get_ephemeral_sync_daemon() -> EphemeralSyncDaemon:
    """Get the singleton EphemeralSyncDaemon instance."""
    global _ephemeral_sync_daemon
    if _ephemeral_sync_daemon is None:
        _ephemeral_sync_daemon = EphemeralSyncDaemon()
    return _ephemeral_sync_daemon


def reset_ephemeral_sync_daemon() -> None:
    """Reset the singleton (for testing)."""
    global _ephemeral_sync_daemon
    _ephemeral_sync_daemon = None


def is_ephemeral_host() -> bool:
    """Check if current host is ephemeral."""
    daemon = get_ephemeral_sync_daemon()
    return daemon.is_ephemeral
