"""Training Sync Loop - Periodic data synchronization to training nodes.

Created as part of Phase 4 of the P2P orchestrator decomposition (December 2025).

This loop periodically syncs selfplay data to training nodes. Only runs
when this node is the cluster leader.

Usage:
    from scripts.p2p.loops import TrainingSyncLoop

    loop = TrainingSyncLoop(
        is_leader=lambda: orchestrator._is_leader(),
        sync_to_training_nodes=lambda: orchestrator._sync_selfplay_to_training_nodes(),
        get_last_sync_time=lambda: orchestrator.last_training_sync_time,
        check_disk_capacity=check_disk_has_capacity,
    )
    await loop.run_forever()
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from .base import BackoffConfig, BaseLoop

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Constants
DEFAULT_TRAINING_SYNC_INTERVAL = 300  # 5 minutes
MAX_DISK_USAGE_PERCENT = 70  # Don't sync if disk > 70%


class TrainingSyncLoop(BaseLoop):
    """Background loop for periodic selfplay data sync to training nodes.

    This loop:
    1. Only runs when this node is the cluster leader
    2. Checks disk capacity before syncing
    3. Emits DATA_SYNC_STARTED/COMPLETED/FAILED events
    4. Calls the actual sync function

    Attributes:
        _is_leader: Callback to check if we are the leader
        _sync_to_training_nodes: Async callback to perform the sync
        _get_last_sync_time: Callback to get last sync timestamp
        _check_disk_capacity: Callback to check disk capacity
    """

    def __init__(
        self,
        is_leader: Callable[[], bool],
        sync_to_training_nodes: Callable[[], Coroutine[Any, Any, dict[str, Any]]],
        get_last_sync_time: Callable[[], float],
        check_disk_capacity: Callable[[], tuple[bool, float]] | None = None,
        *,
        interval: float = DEFAULT_TRAINING_SYNC_INTERVAL,
        max_disk_percent: float = MAX_DISK_USAGE_PERCENT,
        backoff_config: BackoffConfig | None = None,
        enabled: bool = True,
    ):
        """Initialize the training sync loop.

        Args:
            is_leader: Callback that returns True if this node is the leader
            sync_to_training_nodes: Async callback that performs the sync
            get_last_sync_time: Callback to get last sync timestamp
            check_disk_capacity: Optional callback that returns (has_capacity, disk_percent)
            interval: Seconds between sync attempts (default: 300)
            max_disk_percent: Skip sync if disk usage exceeds this (default: 70)
            backoff_config: Custom backoff configuration for errors
            enabled: Whether the loop is enabled
        """
        super().__init__(
            name="training_sync",
            interval=interval,
            backoff_config=backoff_config or BackoffConfig(
                initial_delay=30.0,
                max_delay=600.0,
                multiplier=2.0,
            ),
            enabled=enabled,
        )
        self._is_leader = is_leader
        self._sync_to_training_nodes = sync_to_training_nodes
        self._get_last_sync_time = get_last_sync_time
        self._check_disk_capacity = check_disk_capacity
        self._max_disk_percent = max_disk_percent

        # Track sync stats
        self._total_syncs = 0
        self._successful_syncs = 0
        self._total_jobs_created = 0

    async def _run_once(self) -> None:
        """Execute one iteration of the training sync loop."""
        # Only leader performs syncs
        if not self._is_leader():
            return

        # Check if enough time has passed since last sync
        last_sync = self._get_last_sync_time()
        if time.time() - last_sync < self.interval:
            return

        # Check disk capacity before syncing
        if self._check_disk_capacity is not None:
            has_capacity, disk_percent = self._check_disk_capacity()
            if not has_capacity:
                logger.info(
                    f"[{self.name}] Skipping sync - disk {disk_percent:.1f}% "
                    f">= {self._max_disk_percent}%"
                )
                return

        logger.info(f"[{self.name}] Running periodic training node sync...")

        # Emit DATA_SYNC_STARTED event
        sync_start_time = time.time()
        await self._emit_sync_started()

        # Perform the sync
        self._total_syncs += 1
        result = await self._sync_to_training_nodes()
        sync_duration = time.time() - sync_start_time

        if result.get("success"):
            sync_jobs = result.get("sync_jobs_created", 0)
            self._successful_syncs += 1
            self._total_jobs_created += sync_jobs
            logger.info(
                f"[{self.name}] Sync completed: {sync_jobs} jobs created "
                f"in {sync_duration:.1f}s"
            )

            # Emit DATA_SYNC_COMPLETED event
            await self._emit_sync_completed(sync_jobs, sync_duration)
        else:
            error_msg = result.get("error", "Unknown error")
            logger.warning(f"[{self.name}] Sync failed: {error_msg}")

            # Emit DATA_SYNC_FAILED event
            await self._emit_sync_failed(error_msg)

            # Raise to trigger backoff
            raise RuntimeError(f"Training sync failed: {error_msg}")

    async def _emit_sync_started(self) -> None:
        """Emit DATA_SYNC_STARTED event."""
        try:
            # Dec 29, 2025: Fixed import path (was app.coordination.data_events)
            from app.distributed.data_events import DataEventType, emit_event

            emit_event(
                DataEventType.DATA_SYNC_STARTED,
                {
                    "host": "training_nodes",
                    "sync_type": "incremental",
                    "source": "training_sync_loop",
                },
            )
        except ImportError:
            pass  # data_events not available
        except Exception as e:
            logger.debug(f"[{self.name}] Failed to emit sync started event: {e}")

    async def _emit_sync_completed(self, jobs_created: int, duration: float) -> None:
        """Emit DATA_SYNC_COMPLETED event."""
        try:
            # Dec 29, 2025: Fixed import path (was app.coordination.data_events)
            from app.distributed.data_events import DataEventType, emit_event

            emit_event(
                DataEventType.DATA_SYNC_COMPLETED,
                {
                    "host": "training_nodes",
                    "games_synced": jobs_created,
                    "duration": duration,
                    "source": "training_sync_loop",
                },
            )
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[{self.name}] Failed to emit sync completed event: {e}")

    async def _emit_sync_failed(self, error: str) -> None:
        """Emit DATA_SYNC_FAILED event."""
        try:
            # Dec 29, 2025: Fixed import path (was app.coordination.data_events)
            from app.distributed.data_events import DataEventType, emit_event

            emit_event(
                DataEventType.DATA_SYNC_FAILED,
                {
                    "host": "training_nodes",
                    "error": error,
                    "source": "training_sync_loop",
                },
            )
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[{self.name}] Failed to emit sync failed event: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get extended loop status including sync statistics."""
        status = super().get_status()
        status["sync_stats"] = {
            "total_syncs": self._total_syncs,
            "successful_syncs": self._successful_syncs,
            "total_jobs_created": self._total_jobs_created,
            "success_rate": (
                self._successful_syncs / self._total_syncs * 100
                if self._total_syncs > 0
                else 100.0
            ),
        }
        status["is_leader"] = self._is_leader()
        status["last_sync_time"] = self._get_last_sync_time()
        return status
