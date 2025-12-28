"""Ephemeral host and WAL handling mixin for AutoSyncDaemon.

December 2025: Extracted from auto_sync_daemon.py as part of mixin-based refactoring.
December 2025: Updated to inherit from SyncMixinBase for common functionality.

This mixin provides:
- Write-ahead log (WAL) initialization and management
- Pending game tracking for ephemeral hosts
- Write-through push with retry logic
- Game completion handlers for ephemeral mode
- Rsync operations for ephemeral sync targets
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.async_context import fire_and_forget
from app.coordination.sync_mixin_base import SyncMixinBase

if TYPE_CHECKING:
    from app.coordination.sync_strategies import AutoSyncConfig, SyncStats
    from app.distributed.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

# Import WAL utilities if available
try:
    from app.coordination.wal_sync_utils import (
        checkpoint_database,
        get_rsync_include_args_for_db,
    )
except ImportError:
    # Fallback implementations
    def checkpoint_database(db_path: str | Path) -> bool:
        """Checkpoint WAL file into main database."""
        try:
            import sqlite3
            with sqlite3.connect(str(db_path), timeout=10.0) as conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            return True
        except sqlite3.Error:
            return False

    def get_rsync_include_args_for_db(db_name: str) -> list[str]:
        """Get rsync include args for WAL files."""
        return [f"--include={db_name}", f"--include={db_name}-wal", f"--include={db_name}-shm"]


class SyncEphemeralMixin(SyncMixinBase):
    """Mixin providing ephemeral host and WAL handling for AutoSyncDaemon.

    Inherits from SyncMixinBase for common error handling and logging utilities.

    Ephemeral hosts (Vast.ai, spot instances) need aggressive sync to prevent
    data loss on termination. This mixin provides:
    - Write-ahead log for durability
    - Write-through push for immediate replication
    - Pending game tracking with retry queue

    Additional expected attributes from main class:
    - _is_ephemeral: bool
    - _pending_games: list[dict[str, Any]]
    - _push_lock: asyncio.Lock
    - _wal_initialized: bool
    - _pending_writes_file: Path
    """

    # Additional type hints specific to this mixin
    _is_ephemeral: bool
    _pending_games: list[dict[str, Any]]
    _push_lock: asyncio.Lock
    _wal_initialized: bool
    _pending_writes_file: Path

    def _init_ephemeral_wal(self) -> None:
        """Initialize write-ahead log for ephemeral mode durability.

        December 2025: Consolidated from ephemeral_sync.py
        """
        try:
            wal_path = Path("data/ephemeral_sync_wal.jsonl")
            wal_path.parent.mkdir(parents=True, exist_ok=True)

            if not wal_path.exists():
                wal_path.touch()

            self._wal_initialized = True
            self._wal_path = wal_path
            logger.debug(f"[AutoSyncDaemon] Ephemeral WAL initialized: {wal_path}")

            # Recover pending games from WAL
            self._load_ephemeral_wal()

        except OSError as e:
            logger.error(f"[AutoSyncDaemon] Failed to initialize ephemeral WAL: {e}")
            self._wal_initialized = False

    def _load_ephemeral_wal(self) -> None:
        """Load pending games from WAL on startup.

        December 2025: Consolidated from ephemeral_sync.py
        """
        if not self._wal_initialized or not hasattr(self, "_wal_path"):
            return

        try:
            if not self._wal_path.exists() or self._wal_path.stat().st_size == 0:
                return

            loaded_count = 0
            with open(self._wal_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if not entry.get("synced", False):
                            self._pending_games.append(entry)
                            loaded_count += 1
                    except json.JSONDecodeError:
                        continue

            if loaded_count > 0:
                logger.info(
                    f"[AutoSyncDaemon] Recovered {loaded_count} pending games from WAL"
                )

        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"[AutoSyncDaemon] Failed to load WAL: {e}")

    def _append_to_wal(self, game_entry: dict[str, Any]) -> None:
        """Append pending game to WAL for durability.

        December 2025: Consolidated from ephemeral_sync.py
        Called when a game is added to pending list, before sync attempt.
        """
        if not self._wal_initialized or not hasattr(self, "_wal_path"):
            return

        try:
            with open(self._wal_path, 'a') as f:
                f.write(json.dumps(game_entry) + '\n')
                f.flush()
                os.fsync(f.fileno())  # Force to disk

        except OSError as e:
            logger.debug(f"[AutoSyncDaemon] Failed to append to WAL: {e}")

    def _clear_wal(self) -> None:
        """Clear WAL after successful sync of all pending games.

        December 2025: Consolidated from ephemeral_sync.py
        Called when all pending games have been confirmed synced.
        """
        if not self._wal_initialized or not hasattr(self, "_wal_path"):
            return

        try:
            self._wal_path.write_text('')
            logger.debug("[AutoSyncDaemon] WAL cleared after successful sync")

        except OSError as e:
            logger.debug(f"[AutoSyncDaemon] Failed to clear WAL: {e}")

    def _init_pending_writes_file(self) -> None:
        """Initialize the pending writes retry queue file.

        December 2025: Added to prevent data loss from write-through timeouts.
        """
        try:
            self._pending_writes_file.parent.mkdir(parents=True, exist_ok=True)
            if not self._pending_writes_file.exists():
                self._pending_writes_file.touch()
            logger.debug(
                f"[AutoSyncDaemon] Pending writes file initialized: {self._pending_writes_file}"
            )
        except OSError as e:
            logger.error(f"[AutoSyncDaemon] Failed to initialize pending writes file: {e}")

    def _persist_failed_write(self, game_entry: dict[str, Any]) -> None:
        """Persist a failed write to the retry queue file.

        December 2025: Called when all retry attempts fail for write-through.

        Args:
            game_entry: The game entry that failed to sync
        """
        try:
            entry = {
                **game_entry,
                "failed_at": time.time(),
                "retry_count": 0,
            }
            with open(self._pending_writes_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
            logger.info(
                f"[AutoSyncDaemon] Persisted failed write for game {game_entry.get('game_id')} "
                "to retry queue"
            )
        except OSError as e:
            logger.error(f"[AutoSyncDaemon] Failed to persist write to retry queue: {e}")

    async def _push_with_retry(
        self,
        game_entry: dict[str, Any],
        max_attempts: int = 3,
        base_delay: float = 2.0,
    ) -> bool:
        """Push a game with exponential backoff retry.

        December 2025: Added to prevent data loss from transient network failures.

        Args:
            game_entry: The game entry to push
            max_attempts: Maximum retry attempts (default: 3)
            base_delay: Base delay in seconds (delays: 2s, 4s, 8s)

        Returns:
            True if push succeeded, False if all retries failed
        """
        db_path = game_entry.get("db_path")
        if not db_path:
            logger.warning("[AutoSyncDaemon] No db_path in game entry, cannot push")
            return False

        targets = await self._get_ephemeral_sync_targets()
        if not targets:
            logger.warning("[AutoSyncDaemon] No sync targets available for retry")
            return False

        for attempt in range(max_attempts):
            delay = base_delay * (2 ** attempt)  # 2s, 4s, 8s

            for target in targets[:3]:  # Try up to 3 targets per attempt
                try:
                    success = await asyncio.wait_for(
                        self._rsync_to_target(db_path, target),
                        timeout=self.config.ephemeral_write_through_timeout,
                    )
                    if success:
                        logger.debug(
                            f"[AutoSyncDaemon] Retry push succeeded on attempt {attempt + 1}"
                        )
                        return True
                except asyncio.TimeoutError:
                    logger.debug(
                        f"[AutoSyncDaemon] Retry push timeout to {target} "
                        f"(attempt {attempt + 1}/{max_attempts})"
                    )
                except (RuntimeError, OSError) as e:
                    logger.debug(
                        f"[AutoSyncDaemon] Retry push failed to {target}: {e} "
                        f"(attempt {attempt + 1}/{max_attempts})"
                    )

            # Wait before next retry attempt (except on last attempt)
            if attempt < max_attempts - 1:
                logger.debug(
                    f"[AutoSyncDaemon] Waiting {delay}s before retry attempt {attempt + 2}"
                )
                await asyncio.sleep(delay)

        # All retries failed
        logger.warning(
            f"[AutoSyncDaemon] All {max_attempts} retry attempts failed for "
            f"game {game_entry.get('game_id')}"
        )
        return False

    async def _process_pending_writes(self) -> None:
        """Background task to periodically retry failed writes.

        December 2025: Runs every 60 seconds to retry persisted failed writes.
        """
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute

                if not self._pending_writes_file.exists():
                    continue

                # Read pending writes
                pending_writes: list[dict[str, Any]] = []
                remaining_writes: list[dict[str, Any]] = []

                try:
                    with open(self._pending_writes_file) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                entry = json.loads(line)
                                pending_writes.append(entry)
                            except json.JSONDecodeError:
                                continue
                except OSError as e:
                    logger.debug(f"[AutoSyncDaemon] Failed to read pending writes: {e}")
                    continue

                if not pending_writes:
                    continue

                logger.info(
                    f"[AutoSyncDaemon] Processing {len(pending_writes)} pending writes"
                )

                for entry in pending_writes:
                    # Check if entry is too old (>24 hours)
                    failed_at = entry.get("failed_at", 0)
                    if time.time() - failed_at > 86400:  # 24 hours
                        logger.warning(
                            f"[AutoSyncDaemon] Abandoning stale pending write "
                            f"(age > 24h): {entry.get('game_id')}"
                        )
                        continue

                    # Try to push
                    success = await self._push_with_retry(entry, max_attempts=2)
                    if not success:
                        # Increment retry count and keep in queue
                        entry["retry_count"] = entry.get("retry_count", 0) + 1
                        if entry["retry_count"] < 5:  # Max 5 retry cycles
                            remaining_writes.append(entry)
                        else:
                            logger.error(
                                f"[AutoSyncDaemon] Permanently failed to sync game "
                                f"{entry.get('game_id')} after 5 retry cycles"
                            )
                            # Emit event for alerting
                            fire_and_forget(
                                self._emit_sync_failed(
                                    f"Permanent failure for game {entry.get('game_id')}"
                                ),
                                on_error=lambda exc: logger.debug(
                                    f"Failed to emit sync failed: {exc}"
                                ),
                            )

                # Rewrite the file with remaining writes
                try:
                    with open(self._pending_writes_file, "w") as f:
                        for entry in remaining_writes:
                            f.write(json.dumps(entry) + "\n")
                except OSError as e:
                    logger.error(
                        f"[AutoSyncDaemon] Failed to update pending writes file: {e}"
                    )

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[AutoSyncDaemon] Error in pending writes processor: {e}")

    async def on_game_complete(
        self,
        game_result: dict[str, Any],
        db_path: Path | str | None = None,
    ) -> bool:
        """Handle game completion - queue for immediate push (ephemeral mode).

        December 2025: Consolidated from ephemeral_sync.py
        When write_through_enabled=True, waits for push confirmation before
        returning, ensuring the game is safely synced to a persistent node.

        Args:
            game_result: Game result dict with game_id, moves, etc.
            db_path: Path to database containing the game

        Returns:
            True if game was successfully synced (write-through mode) or queued,
            False if write-through failed (data at risk)
        """
        if not self._is_ephemeral:
            # Non-ephemeral mode: just track the game, normal sync will handle it
            return True

        game_id = game_result.get("game_id")

        # Add to pending
        game_entry = {
            "game_id": game_id,
            "db_path": str(db_path) if db_path else None,
            "timestamp": time.time(),
            "synced": False,
        }
        self._pending_games.append(game_entry)

        # Persist to WAL for durability
        self._append_to_wal(game_entry)

        # Immediate push if we have pending games
        if len(self._pending_games) >= 1:
            self._events_processed += 1

            # Write-through mode - wait for confirmation
            if self.config.ephemeral_write_through:
                try:
                    success = await asyncio.wait_for(
                        self._push_pending_games_with_confirmation(),
                        timeout=self.config.ephemeral_write_through_timeout,
                    )
                    if success:
                        logger.debug(f"[AutoSyncDaemon] Write-through success for game {game_id}")
                        return True
                    else:
                        logger.warning(f"[AutoSyncDaemon] Write-through push failed for game {game_id}")
                        # Dec 2025: Emit DATA_SYNC_FAILED for write-through failures (critical for ephemeral nodes)
                        fire_and_forget(
                            self._emit_sync_failed(f"Write-through push failed for game {game_id}"),
                            on_error=lambda exc: logger.debug(f"Failed to emit sync failed: {exc}"),
                        )
                        return False
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[AutoSyncDaemon] Write-through timeout for game {game_id} "
                        f"(timeout={self.config.ephemeral_write_through_timeout}s)"
                    )
                    # Dec 2025: Retry with exponential backoff before giving up
                    retry_success = await self._push_with_retry(
                        game_entry, max_attempts=3, base_delay=2.0
                    )
                    if retry_success:
                        logger.info(
                            f"[AutoSyncDaemon] Write-through succeeded after retry for game {game_id}"
                        )
                        return True

                    # All retries failed - persist to retry queue to prevent data loss
                    self._persist_failed_write(game_entry)

                    # Emit SYNC_STALLED for failover routing (Dec 2025)
                    fire_and_forget(
                        self._emit_sync_stalled(
                            target_node="write_through_target",
                            timeout_seconds=self.config.ephemeral_write_through_timeout,
                            data_type="game",
                        )
                    )
                    # Fall back to async push (for any remaining pending games)
                    fire_and_forget(self._push_pending_games())
                    return False
            else:
                # Legacy async push (fire-and-forget)
                await self._push_pending_games()
                return True

        return True

    async def _push_pending_games_with_confirmation(self) -> bool:
        """Push pending games and return True if at least one target succeeds.

        December 2025: Consolidated from ephemeral_sync.py
        Write-through variant that returns sync status.
        """
        if not self._pending_games:
            return True

        async with self._push_lock:
            games_to_push = self._pending_games.copy()
            self._pending_games.clear()

            logger.info(f"[AutoSyncDaemon] Write-through: pushing {len(games_to_push)} games")

            # Get unique DB paths
            db_paths = set()
            for game in games_to_push:
                if game.get("db_path"):
                    db_paths.add(game["db_path"])

            if not db_paths:
                logger.warning("[AutoSyncDaemon] No database paths to push")
                return False

            # Get sync targets
            targets = await self._get_ephemeral_sync_targets()
            if not targets:
                logger.warning("[AutoSyncDaemon] No sync targets available")
                self._pending_games.extend(games_to_push)  # Put back
                return False

            # Push each DB to at least one target
            any_success = False
            for db_path in db_paths:
                for target in targets[:3]:  # Try up to 3 targets
                    try:
                        success = await self._rsync_to_target(db_path, target)
                        if success:
                            any_success = True
                            # Mark games as synced
                            for game in games_to_push:
                                game["synced"] = True
                            break
                    except (RuntimeError, OSError, asyncio.TimeoutError) as e:
                        logger.debug(f"[AutoSyncDaemon] Push to {target} failed: {e}")

            if any_success:
                await self._emit_game_synced(
                    games_pushed=len(games_to_push),
                    target_nodes=targets[:1],
                    db_paths=list(db_paths),
                )
                self._clear_wal()

            return any_success

    async def _push_pending_games(self, force: bool = False) -> None:
        """Push pending games to sync targets.

        December 2025: Consolidated from ephemeral_sync.py
        """
        if not self._pending_games:
            return

        async with self._push_lock:
            games_to_push = self._pending_games.copy()
            self._pending_games.clear()

            logger.info(f"[AutoSyncDaemon] Pushing {len(games_to_push)} games")

            # Get unique DB paths
            db_paths = set()
            for game in games_to_push:
                if game.get("db_path"):
                    db_paths.add(game["db_path"])

            if not db_paths:
                logger.warning("[AutoSyncDaemon] No database paths to push")
                return

            # Get sync targets
            targets = await self._get_ephemeral_sync_targets()
            if not targets:
                logger.warning("[AutoSyncDaemon] No sync targets available")
                self._pending_games.extend(games_to_push)  # Put back
                return

            # Push each DB to targets
            successful_targets = []
            for db_path in db_paths:
                for target in targets[:3]:
                    try:
                        success = await self._rsync_to_target(db_path, target)
                        if success:
                            self._stats.games_synced += len(games_to_push)
                            successful_targets.append(target)
                            break
                    except (RuntimeError, OSError, asyncio.TimeoutError) as e:
                        logger.debug(f"[AutoSyncDaemon] Push to {target} failed: {e}")

            if successful_targets:
                await self._emit_game_synced(
                    games_pushed=len(games_to_push),
                    target_nodes=successful_targets,
                    db_paths=list(db_paths),
                )
                self._clear_wal()

    async def _get_ephemeral_sync_targets(self) -> list[str]:
        """Get sync targets for ephemeral mode.

        December 2025: Consolidated from ephemeral_sync.py
        """
        try:
            from app.coordination.sync_router import get_sync_router

            router = get_sync_router()
            targets = router.get_sync_targets(
                data_type="game",
                max_targets=3,
            )
            return [t.node_id for t in targets]

        except ImportError:
            logger.warning("[AutoSyncDaemon] SyncRouter not available")
            return []
        except (RuntimeError, OSError, ValueError, KeyError) as e:
            logger.error(f"[AutoSyncDaemon] Failed to get sync targets: {e}")
            return []

    async def _rsync_to_target(
        self,
        db_path: str,
        target_node: str,
        verify_checksum: bool = True,
    ) -> bool:
        """Rsync a database to a target node.

        December 2025: Consolidated from ephemeral_sync.py
        December 2025: Added sync mutex to prevent concurrent syncs to same target
        December 2025: Added circuit breaker for fault tolerance
        December 2025: Added write lock check to prevent syncing incomplete data
        December 2025: Added checksum verification after sync

        Args:
            db_path: Local database path
            target_node: Target node ID
            verify_checksum: If True, verify checksum after sync (default: True)

        Returns:
            True if successful (and checksum verified if enabled)
        """
        from app.db.write_lock import is_database_safe_to_sync
        from app.coordination.sync_mutex import acquire_sync_lock, release_sync_lock
        from app.coordination.sync_integrity import check_sqlite_integrity

        # Check write lock - don't sync if database is being written to
        if not is_database_safe_to_sync(db_path):
            logger.debug(
                f"[AutoSyncDaemon] Database {db_path} has active write lock, skipping sync"
            )
            return False

        # Dec 2025: CRITICAL - Checkpoint WAL before transfer to prevent corruption
        # Without this, WAL mode databases may transfer without their -wal files,
        # resulting in missing transactions and data corruption
        try:
            from app.coordination.sync_integrity import prepare_database_for_transfer
            prep_success, msg = prepare_database_for_transfer(Path(db_path))
            if not prep_success:
                logger.warning(
                    f"[AutoSyncDaemon] Failed to prepare {db_path} for transfer: {msg}"
                )
                # Continue anyway - may still work for non-WAL databases
        except ImportError:
            logger.debug("[AutoSyncDaemon] sync_integrity not available, skipping prepare step")
        except (OSError, sqlite3.Error) as e:
            logger.warning(f"[AutoSyncDaemon] Error preparing database: {e}")
            # Continue anyway - database may still be transferable

        # Check circuit breaker before attempting sync (December 2025)
        if self._circuit_breaker and not self._circuit_breaker.allow_request(target_node):
            logger.debug(
                f"[AutoSyncDaemon] Circuit open for {target_node}, skipping sync"
            )
            return False

        # Create lock key: target_node + filename to prevent concurrent writes
        db_name = Path(db_path).name if db_path else "unknown"
        lock_key = f"{target_node}:{db_name}"

        # Acquire sync lock to prevent race conditions
        if not acquire_sync_lock(lock_key, operation="rsync", timeout=60):
            logger.debug(f"[AutoSyncDaemon] Could not acquire lock for {lock_key}, skipping")
            return False

        success = False
        remote_path = None
        ssh_info = None
        try:
            from app.coordination.sync_bandwidth import rsync_with_bandwidth_limit
            from app.coordination.sync_router import get_sync_router

            router = get_sync_router()
            cap = router.get_node_capability(target_node)

            if not cap:
                release_sync_lock(lock_key)
                return False

            # Get SSH info for checksum verification
            from app.config.cluster_config import get_cluster_nodes
            nodes = get_cluster_nodes()
            node = nodes.get(target_node)
            if node:
                ssh_info = {
                    "host": node.best_ip,
                    "user": node.ssh_user or "ubuntu",
                    "key": node.ssh_key or "~/.ssh/id_cluster",
                }
                remote_path = f"{node.get_storage_path('games')}/{db_name}"

            # Use centralized timeout (Dec 2025)
            from app.config.thresholds import RSYNC_TIMEOUT
            result = rsync_with_bandwidth_limit(
                source=db_path,
                target_host=target_node,
                timeout=RSYNC_TIMEOUT,
            )

            success = result.success

        except ImportError:
            success = await self._direct_rsync(db_path, target_node)
            # Get SSH info for checksum verification
            try:
                from app.config.cluster_config import get_cluster_nodes
                nodes = get_cluster_nodes()
                node = nodes.get(target_node)
                if node:
                    ssh_info = {
                        "host": node.best_ip,
                        "user": node.ssh_user or "ubuntu",
                        "key": node.ssh_key or "~/.ssh/id_cluster",
                    }
                    remote_path = f"{node.get_storage_path('games')}/{db_name}"
            except (ImportError, KeyError, AttributeError):
                pass
        except (RuntimeError, OSError, asyncio.TimeoutError) as e:
            logger.debug(f"[AutoSyncDaemon] Rsync error: {e}")
            success = False
            # Emit sync failure event (Dec 2025)
            await self._emit_sync_failure(target_node, db_path, str(e))
            release_sync_lock(lock_key)
            return False

        # Always release the lock
        release_sync_lock(lock_key)

        # December 2025: Checksum verification after sync
        if success and verify_checksum and ssh_info and remote_path:
            try:
                from app.coordination.sync_integrity import verify_and_retry_sync

                async def retry_rsync():
                    return await self._direct_rsync(db_path, target_node)

                checksum_ok, error = await verify_and_retry_sync(
                    source_path=db_path,
                    dest_path=remote_path,
                    ssh_host=ssh_info["host"],
                    ssh_user=ssh_info["user"],
                    ssh_key=ssh_info["key"],
                    sync_func=retry_rsync,
                    max_retries=1,
                )

                if not checksum_ok:
                    logger.error(
                        f"[AutoSyncDaemon] Checksum verification failed for {db_path} -> "
                        f"{target_node}: {error}"
                    )
                    success = False
                    self._stats.databases_verification_failed += 1
                else:
                    self._stats.databases_verified += 1
                    logger.debug(
                        f"[AutoSyncDaemon] Checksum verified: {db_path} -> {target_node}"
                    )

            except ImportError:
                logger.debug("[AutoSyncDaemon] sync_integrity not available, skipping verification")
            except Exception as e:
                logger.warning(f"[AutoSyncDaemon] Checksum verification error: {e}")
                # Don't fail the sync if verification fails - just log it
                self._stats.databases_verification_failed += 1

        # Record success/failure with circuit breaker (December 2025)
        if self._circuit_breaker:
            if success:
                self._circuit_breaker.record_success(target_node)
            else:
                self._circuit_breaker.record_failure(target_node)

        return success

    async def _direct_rsync(self, db_path: str, target_node: str) -> bool:
        """Direct rsync without bandwidth management.

        December 2025: Consolidated from ephemeral_sync.py
        December 2025: Updated to use cluster_config helpers instead of inline YAML
        """
        try:
            # December 2025: Use cluster_config helpers instead of inline YAML parsing
            from app.config.cluster_config import get_cluster_nodes, get_node_bandwidth_kbs

            nodes = get_cluster_nodes()
            node = nodes.get(target_node)

            if not node:
                logger.debug(f"[AutoSyncDaemon] Node {target_node} not found in cluster config")
                return False

            ssh_host = node.best_ip
            ssh_user = node.ssh_user or "ubuntu"
            ssh_key = node.ssh_key or "~/.ssh/id_cluster"

            # Dec 2025: Use storage path from node config (supports OWC routing)
            remote_games_path = node.get_storage_path("games")

            # Get bandwidth limit for this node
            bwlimit_args = []
            try:
                bwlimit_kbs = get_node_bandwidth_kbs(target_node)
                if bwlimit_kbs > 0:
                    bwlimit_args = [f"--bwlimit={bwlimit_kbs}"]
            except (KeyError, ValueError):
                pass

            if not ssh_host:
                return False

            ssh_key = os.path.expanduser(ssh_key)
            remote_full = f"{ssh_user}@{ssh_host}:{remote_games_path}/"

            # Dec 2025: Checkpoint WAL before sync to ensure all data is in main .db file
            # This prevents corruption from missing WAL transactions
            checkpoint_database(db_path)

            # Dec 2025: Include WAL files in rsync to prevent data loss
            # WAL files (.db-wal, .db-shm) contain uncommitted transactions
            db_name = Path(db_path).name
            wal_include_args = get_rsync_include_args_for_db(db_name)
            parent_dir = str(Path(db_path).parent) + "/"

            rsync_cmd = [
                "rsync",
                "-avz",
                "--compress",
                *bwlimit_args,
                *wal_include_args,
                "--exclude=*",  # Exclude other files in directory
                "-e", f"ssh -i {ssh_key} -o StrictHostKeyChecking=no -o ConnectTimeout=10",
                parent_dir,
                remote_full,
            ]

            # Use centralized timeout (Dec 2025)
            from app.config.thresholds import RSYNC_TIMEOUT
            result = await asyncio.to_thread(
                subprocess.run,
                rsync_cmd,
                capture_output=True,
                text=True,
                timeout=RSYNC_TIMEOUT,
            )
            return result.returncode == 0

        except subprocess.TimeoutExpired:
            logger.warning(f"[AutoSyncDaemon] Rsync timeout to {target_node}")
            # Emit sync stalled event for failover routing (Dec 2025)
            from app.config.thresholds import RSYNC_TIMEOUT
            await self._emit_sync_stalled(
                target_node=target_node,
                timeout_seconds=RSYNC_TIMEOUT,
                data_type="game",
            )
            # Also emit failure for general error tracking
            await self._emit_sync_failure(target_node, db_path, f"Rsync timeout after {RSYNC_TIMEOUT}s")
            return False
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            logger.debug(f"[AutoSyncDaemon] Rsync error: {e}")
            # Emit sync failure event (Dec 2025)
            await self._emit_sync_failure(target_node, db_path, str(e))
            return False

    async def _emit_game_synced(
        self,
        games_pushed: int,
        target_nodes: list[str],
        db_paths: list[str],
    ) -> None:
        """Emit GAME_SYNCED event for feedback loop coupling.

        December 2025: Consolidated from ephemeral_sync.py
        """
        try:
            from app.coordination.event_router import DataEventType, get_router

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
                    source="AutoSyncDaemon",
                )
        except (RuntimeError, AttributeError, ImportError) as e:
            logger.debug(f"[AutoSyncDaemon] Could not emit GAME_SYNCED event: {e}")

    # Note: _emit_sync_failed() is expected from main class
    # _emit_sync_failure() and _emit_sync_stalled() are inherited from SyncMixinBase
