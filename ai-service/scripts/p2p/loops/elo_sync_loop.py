"""Elo Sync Loop - Periodic Elo database synchronization.

Created as part of Phase 4 of the P2P orchestrator decomposition (December 2025).

This loop synchronizes the Elo database across cluster nodes. Only runs
when the node is not busy with training operations.

Phase 15.1.3 (Dec 2025): Added retry logic with exponential backoff.
Previously, initialization failure would permanently disable the loop.
Now, the loop retries initialization with backoff before giving up,
and schedules periodic re-enable checks even after max retries.

Usage:
    from scripts.p2p.loops import EloSyncLoop

    loop = EloSyncLoop(
        get_elo_sync_manager=lambda: orchestrator.elo_sync_manager,
        get_sync_in_progress=lambda: orchestrator.sync_in_progress,
    )
    await loop.run_forever()
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Callable

from app.core.async_context import safe_create_task
from .base import BackoffConfig, BaseLoop
from .loop_constants import LoopIntervals, LoopLimits

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Jan 5, 2026: Push debounce interval (seconds) to prevent spam
PUSH_DEBOUNCE_INTERVAL = 30.0

# Backward-compat aliases (Sprint 10: use LoopIntervals/LoopLimits instead)
DEFAULT_SYNC_INTERVAL = LoopIntervals.ELO_SYNC
MAX_INIT_RETRIES = LoopLimits.MAX_INIT_RETRIES
REENABLE_CHECK_INTERVAL = LoopIntervals.REENABLE_CHECK


class EloSyncLoop(BaseLoop):
    """Background loop for periodic Elo database synchronization.

    This loop:
    1. Initializes the EloSyncManager on first run
    2. Periodically syncs Elo data with cluster peers
    3. Skips sync if training is in progress

    Attributes:
        _elo_sync_manager: The EloSyncManager instance (obtained via callback)
    """

    def __init__(
        self,
        get_elo_sync_manager: Callable[[], Any | None],
        get_sync_in_progress: Callable[[], bool],
        *,
        interval: float = DEFAULT_SYNC_INTERVAL,
        backoff_config: BackoffConfig | None = None,
        enabled: bool = True,
    ):
        """Initialize the Elo sync loop.

        Args:
            get_elo_sync_manager: Callback to get EloSyncManager instance
            get_sync_in_progress: Callback to check if sync/training is in progress
            interval: Seconds between sync attempts (default: 300)
            backoff_config: Custom backoff configuration for errors
            enabled: Whether the loop is enabled
        """
        super().__init__(
            name="elo_sync",
            interval=interval,
            backoff_config=backoff_config or BackoffConfig(
                initial_delay=5.0,
                max_delay=300.0,
                multiplier=2.0,
            ),
            enabled=enabled,
        )
        self._get_elo_sync_manager = get_elo_sync_manager
        self._get_sync_in_progress = get_sync_in_progress
        self._initialized = False
        # Phase 15.1.3: Track initialization retry state
        self._init_attempts = 0
        self._reenable_task: asyncio.Task | None = None
        # Jan 5, 2026: Push-based sync tracking
        self._last_push_time: float = 0.0
        self._event_subscription_active = False
        # Feb 2026: Periodic reconciliation of canonical_ -> ringrift_best_ IDs
        self._last_reconciliation_time: float = 0.0
        self._reconciliation_interval: float = 86400.0  # 24 hours

    async def _on_start(self) -> None:
        """Initialize the EloSyncManager on loop start with retry logic.

        Phase 15.1.3: Implements exponential backoff retry before disabling.
        Previously, a single initialization failure would permanently disable
        the loop. Now retries up to MAX_INIT_RETRIES times with backoff.
        """
        manager = self._get_elo_sync_manager()
        if manager is None:
            logger.info(f"[{self.name}] EloSyncManager not available, scheduling re-enable check")
            self.enabled = False
            self._schedule_reenable_check()
            return

        # Phase 15.1.3: Retry initialization with exponential backoff
        for attempt in range(MAX_INIT_RETRIES):
            self._init_attempts = attempt + 1
            try:
                await manager.initialize()
                self._initialized = True
                # Update interval from manager if available
                if hasattr(manager, "sync_interval"):
                    self.interval = manager.sync_interval
                logger.info(
                    f"[{self.name}] Started with interval {self.interval}s "
                    f"(attempt {attempt + 1}/{MAX_INIT_RETRIES})"
                )
                # Jan 5, 2026: Subscribe to EVALUATION_COMPLETED for push-based sync
                self._subscribe_to_evaluation_events()
                return  # Success - exit retry loop
            except Exception as e:
                wait_time = min(300.0, (2 ** attempt) * 10.0)  # 10s, 20s, 40s, 80s, 160s, max 300s
                logger.warning(
                    f"[{self.name}] Initialization attempt {attempt + 1}/{MAX_INIT_RETRIES} "
                    f"failed: {e}. Retrying in {wait_time}s..."
                )
                if attempt < MAX_INIT_RETRIES - 1:  # Don't sleep after last attempt
                    await asyncio.sleep(wait_time)

        # All retries exhausted
        logger.error(
            f"[{self.name}] EloSyncManager initialization failed after {MAX_INIT_RETRIES} attempts. "
            f"Disabling loop, will re-check in {REENABLE_CHECK_INTERVAL}s."
        )
        self.enabled = False
        self._schedule_reenable_check()

    async def _run_once(self) -> None:
        """Execute one iteration of the Elo sync loop."""
        manager = self._get_elo_sync_manager()
        if manager is None:
            return

        # Skip if sync/training is in progress
        if self._get_sync_in_progress():
            logger.debug(f"[{self.name}] Skipping sync - training in progress")
            return

        try:
            # Sync unified_elo.db (existing)
            success = await manager.sync_with_cluster()
            if success and hasattr(manager, "state"):
                match_count = getattr(manager.state, "local_match_count", 0)
                logger.info(f"[{self.name}] Elo sync completed: {match_count} matches")

            # Sync elo_progress.db (Jan 2026: cluster-wide progress tracking)
            await self._sync_elo_progress()

            # Feb 2026: Periodic reconciliation of canonical_ -> ringrift_best_ IDs
            await self._reconcile_participant_ids()
        except Exception as e:
            logger.warning(f"[{self.name}] Sync error: {e}")
            raise  # Trigger backoff

    def _schedule_reenable_check(self, interval: float = REENABLE_CHECK_INTERVAL) -> None:
        """Schedule a periodic check to re-enable the loop (Phase 15.1.3).

        After initialization fails permanently, this schedules periodic checks
        to see if the EloSyncManager becomes available again. This prevents
        permanent disabling when the failure was temporary (e.g., network issue).

        Args:
            interval: Seconds between re-enable checks (default: 1 hour)
        """
        if self._reenable_task is not None and not self._reenable_task.done():
            return  # Already scheduled

        async def _reenable_check_loop() -> None:
            while not self.enabled:
                await asyncio.sleep(interval)
                manager = self._get_elo_sync_manager()
                if manager is not None:
                    logger.info(
                        f"[{self.name}] Re-enable check: EloSyncManager now available, "
                        f"attempting to restart..."
                    )
                    self.enabled = True
                    self._init_attempts = 0  # Reset retry counter
                    try:
                        await self._on_start()
                        if self._initialized:
                            logger.info(f"[{self.name}] Successfully re-enabled after failure")
                            return
                    except Exception as e:
                        logger.warning(f"[{self.name}] Re-enable attempt failed: {e}")
                        self.enabled = False
                else:
                    logger.debug(
                        f"[{self.name}] Re-enable check: EloSyncManager still not available"
                    )

        try:
            asyncio.get_running_loop()  # Guard: safe_create_task needs a running loop
            self._reenable_task = safe_create_task(_reenable_check_loop(), name="elo-sync-reenable")
            logger.info(
                f"[{self.name}] Scheduled re-enable check every {interval}s"
            )
        except RuntimeError:
            # No running event loop - can't schedule, but that's OK for testing
            logger.debug(f"[{self.name}] No event loop, skipping re-enable schedule")

    def get_status(self) -> dict[str, Any]:
        """Get extended loop status including sync manager and retry status."""
        status = super().get_status()
        status["initialized"] = self._initialized
        # Phase 15.1.3: Include retry tracking
        status["init_attempts"] = self._init_attempts
        status["max_init_retries"] = MAX_INIT_RETRIES
        status["reenable_check_active"] = (
            self._reenable_task is not None and not self._reenable_task.done()
        )
        status["reenable_check_interval"] = REENABLE_CHECK_INTERVAL

        manager = self._get_elo_sync_manager()
        if manager is not None and hasattr(manager, "state"):
            try:
                status["local_match_count"] = getattr(manager.state, "local_match_count", 0)
            except AttributeError:
                pass  # Attribute may not exist on all manager states

        return status

    def health_check(self) -> Any:
        """Check loop health with Elo sync-specific status.

        Jan 2026: Added for DaemonManager integration.
        Reports initialization status, sync manager health, and retry state.

        Returns:
            HealthCheckResult with Elo sync status
        """
        try:
            from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
        except ImportError:
            # Fallback if protocols not available
            return {
                "healthy": self.running and self._initialized,
                "status": "running" if self.running else "stopped",
                "message": f"EloSyncLoop {'initialized' if self._initialized else 'not initialized'}",
                "details": self.get_status(),
            }

        # Not running
        if not self.running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="EloSyncLoop is stopped",
                details={"running": False},
            )

        # Disabled with re-enable check pending
        if not self.enabled:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message=f"EloSyncLoop disabled, re-enable check active: {self._reenable_task is not None}",
                details={
                    "init_attempts": self._init_attempts,
                    "max_retries": MAX_INIT_RETRIES,
                    "reenable_check_active": self._reenable_task is not None and not self._reenable_task.done(),
                },
            )

        # Initialization failed
        if not self._initialized:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"EloSyncLoop not initialized after {self._init_attempts} attempts",
                details={
                    "init_attempts": self._init_attempts,
                    "max_retries": MAX_INIT_RETRIES,
                },
            )

        # Get sync manager health
        manager = self._get_elo_sync_manager()
        match_count = 0
        if manager and hasattr(manager, "state"):
            try:
                match_count = getattr(manager.state, "local_match_count", 0)
            except AttributeError:
                pass

        # Healthy
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"EloSyncLoop healthy ({match_count} matches synced)",
            details={
                "initialized": True,
                "local_match_count": match_count,
                "sync_interval": self.interval,
                "init_attempts": self._init_attempts,
            },
        )

    def _subscribe_to_evaluation_events(self) -> None:
        """Subscribe to EVALUATION_COMPLETED events for push-based sync.

        Jan 5, 2026: Enables immediate Elo push after evaluation completes,
        reducing Elo propagation time from 300s (pull interval) to <30s.
        """
        if self._event_subscription_active:
            return

        try:
            from app.coordination.event_router import get_event_router
            from app.distributed.data_events import DataEventType

            router = get_event_router()
            event_type = DataEventType.EVALUATION_COMPLETED.value
            router.subscribe(event_type, self._on_evaluation_completed)
            self._event_subscription_active = True
            logger.info(f"[{self.name}] Subscribed to {event_type} for push-based sync")
        except ImportError as e:
            logger.debug(f"[{self.name}] Event subscription not available: {e}")
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to subscribe to evaluation events: {e}")

    async def _on_evaluation_completed(self, event: Any) -> None:
        """Handle EVALUATION_COMPLETED event by triggering immediate Elo push.

        Jan 5, 2026: Push-based Elo sync reduces propagation latency from
        300s (periodic pull) to <30s (event-driven push).

        Uses 30s debounce to prevent excessive pushes when multiple
        evaluations complete in quick succession.
        """
        # Extract config_key from event
        payload = getattr(event, "payload", {}) or {}
        if isinstance(event, dict):
            payload = event
        config_key = payload.get("config_key", "")

        if not config_key:
            return

        # Debounce: don't push more than once per 30s
        now = time.time()
        if now - self._last_push_time < PUSH_DEBOUNCE_INTERVAL:
            logger.debug(
                f"[{self.name}] Skipping push (debounce), "
                f"last push was {now - self._last_push_time:.1f}s ago"
            )
            return

        # Get manager and trigger push
        manager = self._get_elo_sync_manager()
        if manager is None:
            logger.warning(f"[{self.name}] Cannot push - EloSyncManager not available")
            return

        # Check if push_to_cluster method exists
        if not hasattr(manager, "push_to_cluster"):
            logger.debug(f"[{self.name}] Manager does not support push_to_cluster")
            return

        logger.info(f"[{self.name}] Pushing Elo updates after evaluation for {config_key}")
        self._last_push_time = now

        try:
            # push_to_cluster is async (Dec 2025)
            results = await manager.push_to_cluster()
            successful = sum(results.values()) if results else 0
            total = len(results) if results else 0
            logger.info(f"[{self.name}] Elo push completed: {successful}/{total} nodes updated")
        except Exception as e:
            logger.warning(f"[{self.name}] Elo push failed: {e}")

    async def _reconcile_participant_ids(self) -> None:
        """Reconcile any remaining canonical_* participant IDs to ringrift_best_*.

        Feb 2026: Runs once per day on coordinator only. Catches drift from
        non-updated cluster nodes that may still write canonical_* entries.
        """
        import os

        # Only run on coordinator
        is_coordinator = os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() == "true"
        if not is_coordinator:
            return

        # Check reconciliation interval
        now = time.time()
        if now - self._last_reconciliation_time < self._reconciliation_interval:
            return

        self._last_reconciliation_time = now

        try:
            import sqlite3
            from pathlib import Path
            from app.training.composite_participant import normalize_nn_id

            db_path = Path("data/unified_elo.db")
            if not db_path.exists():
                return

            def _do_reconcile():
                conn = sqlite3.connect(str(db_path), timeout=30.0)
                try:
                    # Find columns
                    cursor = conn.execute("PRAGMA table_info(elo_ratings)")
                    columns = {row[1] for row in cursor.fetchall()}
                    id_col = "model_id" if "model_id" in columns else "participant_id"

                    # Find remaining canonical_* entries
                    cursor = conn.execute(f"""
                        SELECT {id_col}, board_type, num_players
                        FROM elo_ratings
                        WHERE {id_col} LIKE 'canonical_%'
                    """)
                    entries = cursor.fetchall()

                    if not entries:
                        return 0

                    renamed = 0
                    for old_id, board_type, num_players in entries:
                        # Normalize the ID
                        if ":" in old_id:
                            parts = old_id.split(":")
                            normalized_nn = normalize_nn_id(parts[0])
                            new_id = ":".join([normalized_nn] + parts[1:])
                        else:
                            new_id = normalize_nn_id(old_id)

                        if new_id == old_id:
                            continue

                        # Check if target exists
                        cursor = conn.execute(f"""
                            SELECT COUNT(*) FROM elo_ratings
                            WHERE {id_col} = ? AND board_type = ? AND num_players = ?
                        """, (new_id, board_type, num_players))
                        exists = cursor.fetchone()[0] > 0

                        if not exists:
                            # Simple rename
                            conn.execute(f"""
                                UPDATE elo_ratings SET {id_col} = ?
                                WHERE {id_col} = ? AND board_type = ? AND num_players = ?
                            """, (new_id, old_id, board_type, num_players))
                            renamed += 1
                        else:
                            # Merge conflict - log for manual attention
                            logger.warning(
                                f"[{self.name}] Reconciliation merge needed: "
                                f"{old_id} -> {new_id} (both exist for {board_type}/{num_players}p)"
                            )

                    conn.commit()
                    return renamed
                finally:
                    conn.close()

            renamed = await asyncio.to_thread(_do_reconcile)
            if renamed:
                logger.info(
                    f"[{self.name}] Reconciled {renamed} canonical_* -> ringrift_best_* entries"
                )
        except Exception as e:
            logger.warning(f"[{self.name}] Reconciliation error: {e}")

    async def _sync_elo_progress(self) -> None:
        """Sync elo_progress.db from coordinator for cluster-wide progress tracking.

        Jan 2026: Added to ensure all nodes have Elo progress data for dashboards.
        The elo_progress.db contains Elo snapshots over time, enabling training
        improvement visualization across the cluster.

        Non-coordinator nodes pull from the coordinator (source of truth).
        The coordinator generates elo_progress.db via EloProgressTracker.
        """
        import os
        import tempfile
        from pathlib import Path

        # Only non-coordinator nodes need to pull
        is_coordinator = os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() == "true"
        if is_coordinator:
            logger.debug(f"[{self.name}] Skipping elo_progress sync - this is coordinator")
            return

        # Get coordinator URL from manager or environment
        coordinator_url = self._get_coordinator_url()
        if not coordinator_url:
            logger.debug(f"[{self.name}] No coordinator URL available for elo_progress sync")
            return

        try:
            import aiohttp

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                url = f"{coordinator_url}/elo/progress/db"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        if content:
                            # Atomic write to prevent corruption
                            local_path = Path("data/elo_progress.db")
                            local_path.parent.mkdir(parents=True, exist_ok=True)

                            with tempfile.NamedTemporaryFile(
                                dir=local_path.parent, delete=False, suffix=".db"
                            ) as tmp:
                                tmp.write(content)
                                tmp_path = Path(tmp.name)

                            tmp_path.rename(local_path)
                            logger.info(
                                f"[{self.name}] Synced elo_progress.db ({len(content)} bytes) "
                                f"from coordinator"
                            )
                    elif resp.status == 404:
                        logger.debug(f"[{self.name}] elo_progress.db not found on coordinator")
                    else:
                        logger.warning(
                            f"[{self.name}] Failed to sync elo_progress.db: HTTP {resp.status}"
                        )
        except ImportError:
            logger.debug(f"[{self.name}] aiohttp not available for elo_progress sync")
        except Exception as e:
            logger.warning(f"[{self.name}] elo_progress sync error: {e}")

    def _get_coordinator_url(self) -> str | None:
        """Get the coordinator's P2P URL for database pulls.

        Returns the Tailscale IP + port for the cluster coordinator,
        or None if not available.
        """
        import os

        # Try environment variable first
        coordinator_url = os.environ.get("RINGRIFT_COORDINATOR_URL")
        if coordinator_url:
            return coordinator_url

        # Try to get from manager state if available
        manager = self._get_elo_sync_manager()
        if manager and hasattr(manager, "coordinator_url"):
            return getattr(manager, "coordinator_url", None)

        # Hardcoded fallback for mac-studio coordinator
        # TODO: Make this configurable via distributed_hosts.yaml
        return "http://100.107.168.125:8770"
