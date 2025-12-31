"""Queue Populator Loop - Maintains minimum work queue depth.

Created as part of Phase 4 of the P2P orchestrator decomposition (December 2025).

This loop ensures there are always at least 50 work items in the queue until
all board/player configurations reach 2000 Elo. Only runs on leader node.

Usage:
    from scripts.p2p.loops import QueuePopulatorLoop

    loop = QueuePopulatorLoop(
        get_role=lambda: orchestrator.role,
        get_selfplay_scheduler=lambda: orchestrator.selfplay_scheduler,
        notifier=orchestrator.notifier,
    )
    await loop.run_forever()
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .base import BackoffConfig, BaseLoop

if TYPE_CHECKING:
    from scripts.p2p.types import NodeRole

logger = logging.getLogger(__name__)

# Constants
POPULATOR_INTERVAL = 60  # 1 minute between population attempts
INITIAL_DELAY = 30  # Delay before first run
ALL_TARGETS_MET_INTERVAL = 300  # 5 minutes when all targets met


class QueuePopulatorLoop(BaseLoop):
    """Background loop to maintain minimum work queue depth.

    This loop:
    1. Initializes a QueuePopulator on first run
    2. Only operates when this node is the leader
    3. Populates the work queue until all configs reach target Elo
    4. Uses SelfplayScheduler for priority-based config selection

    Attributes:
        populator: The QueuePopulator instance (initialized on first run)
    """

    def __init__(
        self,
        get_role: Callable[[], NodeRole],
        get_selfplay_scheduler: Callable[[], Any],
        notifier: Any | None = None,
        config_path: Path | None = None,
        *,
        interval: float = POPULATOR_INTERVAL,
        backoff_config: BackoffConfig | None = None,
        enabled: bool = True,
    ):
        """Initialize the queue populator loop.

        Args:
            get_role: Callback to get current node role (LEADER/FOLLOWER)
            get_selfplay_scheduler: Callback to get SelfplayScheduler instance
            notifier: Optional notifier for status updates
            config_path: Path to unified_loop.yaml config (auto-detected if None)
            interval: Seconds between population attempts (default: 60)
            backoff_config: Custom backoff configuration for errors
            enabled: Whether the loop is enabled
        """
        super().__init__(
            name="queue_populator",
            interval=interval,
            backoff_config=backoff_config or BackoffConfig(
                initial_delay=5.0,
                max_delay=120.0,
                multiplier=2.0,
            ),
            enabled=enabled,
        )
        self._get_role = get_role
        self._get_selfplay_scheduler = get_selfplay_scheduler
        self._notifier = notifier
        self._config_path = config_path
        self._populator: Any = None
        self._initialized = False

    async def _on_start(self) -> None:
        """Initialize the QueuePopulator on loop start."""
        import asyncio
        await asyncio.sleep(INITIAL_DELAY)
        logger.info(f"[{self.name}] Starting with {INITIAL_DELAY}s initial delay")

    async def _run_once(self) -> None:
        """Execute one iteration of the queue population loop.

        Dec 31, 2025: Added diagnostic logging for 48-hour autonomous operation.
        Helps debug why work queue may be empty.
        """
        # Lazy initialization of populator
        if not self._initialized:
            self._initialize_populator()
            self._initialized = True

        if self._populator is None:
            logger.debug(f"[{self.name}] Populator not initialized, waiting for retry")
            return  # Initialization failed, backoff will retry

        # Import NodeRole here to avoid circular imports
        from scripts.p2p.types import NodeRole

        # Only leader populates work queue
        role = self._get_role()
        if role != NodeRole.LEADER:
            # Log periodically (not every iteration)
            if hasattr(self, "_last_role_log") and (time.time() - self._last_role_log) < 300:
                return  # Skip logging, already logged recently
            self._last_role_log = time.time()
            logger.debug(f"[{self.name}] Skipping - role={role.value}, need LEADER")
            return

        # Check if populator is enabled
        if not self._populator.config.enabled:
            logger.debug(f"[{self.name}] Populator disabled in config")
            return

        # Dec 31, 2025: Log work queue availability
        from app.coordination.work_queue import get_work_queue
        wq = get_work_queue()
        if wq is None:
            logger.warning(f"[{self.name}] Work queue not available")
            return
        current_depth = wq.depth() if hasattr(wq, 'depth') else 0
        logger.debug(f"[{self.name}] Running as LEADER, queue depth={current_depth}")

        # Check if all targets are met (run in thread to avoid blocking)
        # Dec 31, 2025: Wrap blocking operations with asyncio.to_thread()
        # to prevent CPU spikes and HTTP server unresponsiveness on leader nodes
        all_met = await asyncio.to_thread(self._populator.all_targets_met)
        if all_met:
            logger.info(f"[{self.name}] All Elo targets met (2000+), checking less often")
            # Increase interval temporarily
            self.interval = ALL_TARGETS_MET_INTERVAL
            return
        else:
            # Reset to normal interval
            self.interval = POPULATOR_INTERVAL

        # Populate the queue (run in thread to avoid blocking SQLite operations)
        items_added = await asyncio.to_thread(self._populator.populate)
        if items_added > 0:
            status = await asyncio.to_thread(self._populator.get_status)
            logger.info(
                f"[{self.name}] Queue populated: +{items_added} items, "
                f"depth={status['current_queue_depth']}, "
                f"unmet={status['configs_unmet']}/{status['total_configs']}"
            )

            # Notify if significant change
            if items_added >= 10 and self._notifier is not None:
                try:
                    await self._notifier.send(
                        f"📋 Queue populated: +{items_added} work items",
                        severity="info",
                        context={
                            "queue_depth": status["current_queue_depth"],
                            "configs_unmet": status["configs_unmet"],
                        },
                    )
                except Exception as e:
                    logger.debug(f"[{self.name}] Failed to send notification: {e}")

    def _initialize_populator(self) -> None:
        """Initialize the QueuePopulator with config."""
        try:
            import yaml

            from app.coordination.unified_queue_populator import (
                UnifiedQueuePopulator as QueuePopulator,
                load_populator_config_from_yaml,
            )
            from app.coordination.work_queue import get_work_queue

            # Determine config path
            if self._config_path is None:
                self._config_path = Path(__file__).parent.parent.parent.parent / "config" / "unified_loop.yaml"

            # Load config from YAML
            populator_config = None
            if self._config_path.exists():
                with open(self._config_path) as f:
                    yaml_config = yaml.safe_load(f)
                populator_config = load_populator_config_from_yaml(yaml_config)

            # Create populator
            self._populator = QueuePopulator(config=populator_config)
            self._populator.set_work_queue(get_work_queue())

            # Wire SelfplayScheduler for priority-based config selection
            scheduler = self._get_selfplay_scheduler()
            if scheduler is not None:
                self._populator.set_selfplay_scheduler(scheduler)

            logger.info(f"[{self.name}] QueuePopulator initialized")

        except Exception as e:
            logger.error(f"[{self.name}] Failed to initialize QueuePopulator: {e}")
            self._populator = None
            raise  # Trigger backoff

    @property
    def populator(self) -> Any:
        """Get the QueuePopulator instance."""
        return self._populator

    def get_status(self) -> dict[str, Any]:
        """Get extended loop status including populator status."""
        status = super().get_status()
        if self._populator is not None:
            try:
                populator_status = self._populator.get_status()
                status["populator"] = populator_status
            except Exception as e:
                status["populator_error"] = str(e)
        return status
