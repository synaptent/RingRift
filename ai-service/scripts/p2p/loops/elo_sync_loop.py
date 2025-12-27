"""Elo Sync Loop - Periodic Elo database synchronization.

Created as part of Phase 4 of the P2P orchestrator decomposition (December 2025).

This loop synchronizes the Elo database across cluster nodes. Only runs
when the node is not busy with training operations.

Usage:
    from scripts.p2p.loops import EloSyncLoop

    loop = EloSyncLoop(
        get_elo_sync_manager=lambda: orchestrator.elo_sync_manager,
        get_sync_in_progress=lambda: orchestrator.sync_in_progress,
    )
    await loop.run_forever()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from .base import BackoffConfig, BaseLoop

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Constants
DEFAULT_SYNC_INTERVAL = 300  # 5 minutes default


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

    async def _on_start(self) -> None:
        """Initialize the EloSyncManager on loop start."""
        manager = self._get_elo_sync_manager()
        if manager is None:
            logger.info(f"[{self.name}] EloSyncManager not available, loop disabled")
            self.enabled = False
            return

        try:
            await manager.initialize()
            self._initialized = True
            # Update interval from manager if available
            if hasattr(manager, "sync_interval"):
                self.interval = manager.sync_interval
            logger.info(f"[{self.name}] Started with interval {self.interval}s")
        except Exception as e:
            logger.warning(f"[{self.name}] EloSyncManager initialization failed: {e}")
            self.enabled = False

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
            success = await manager.sync_with_cluster()
            if success and hasattr(manager, "state"):
                match_count = getattr(manager.state, "local_match_count", 0)
                logger.info(f"[{self.name}] Sync completed: {match_count} matches")
        except Exception as e:
            logger.warning(f"[{self.name}] Sync error: {e}")
            raise  # Trigger backoff

    def get_status(self) -> dict[str, Any]:
        """Get extended loop status including sync manager status."""
        status = super().get_status()
        status["initialized"] = self._initialized

        manager = self._get_elo_sync_manager()
        if manager is not None and hasattr(manager, "state"):
            try:
                status["local_match_count"] = getattr(manager.state, "local_match_count", 0)
            except Exception:
                pass

        return status
