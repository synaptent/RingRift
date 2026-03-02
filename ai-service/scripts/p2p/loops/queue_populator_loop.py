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
from .loop_constants import LoopIntervals

if TYPE_CHECKING:
    from scripts.p2p.types import NodeRole

logger = logging.getLogger(__name__)

# Backward-compat aliases (Sprint 10: use LoopIntervals.* instead)
POPULATOR_INTERVAL = LoopIntervals.QUEUE_POPULATOR
INITIAL_DELAY = LoopIntervals.QUEUE_POPULATOR_INITIAL_DELAY
ALL_TARGETS_MET_INTERVAL = LoopIntervals.QUEUE_POPULATOR_ALL_MET


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
            # Mar 2026: Explicit 120s timeout instead of default 600s (max(60*10, 300)).
            # The default 600s timeout causes queue_populator to hang for 10 minutes
            # when thread pool is saturated (asyncio.to_thread calls in _run_once
            # wait for a thread slot that never becomes available). 120s is long enough
            # for legitimate populate() calls (typically <30s) but short enough to
            # detect thread pool starvation quickly and allow recovery.
            run_timeout=120.0,
        )
        self._get_role = get_role
        self._get_selfplay_scheduler = get_selfplay_scheduler
        self._notifier = notifier
        self._config_path = config_path
        self._populator: Any = None
        self._initialized = False
        # Jan 5, 2026 (Session 17.28): Track leader unreachability for follower takeover
        self._leader_last_seen: float = time.time()
        self._leader_unreachable_threshold: float = 30.0  # seconds before follower takes over

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

        # Jan 5, 2026 (Session 17.28): Allow followers to populate if leader is unreachable
        # This prevents single point of failure for work distribution
        role = self._get_role()
        is_leader = role == NodeRole.LEADER

        # Check if we should allow follower takeover
        should_populate = is_leader
        leader_unreachable_for = 0.0

        if not is_leader:
            # Check leader reachability (via P2P cluster status)
            leader_alive = await self._check_leader_alive()
            if leader_alive:
                self._leader_last_seen = time.time()
            leader_unreachable_for = time.time() - self._leader_last_seen

            # Follower takes over if leader unreachable for > threshold
            if leader_unreachable_for > self._leader_unreachable_threshold:
                should_populate = True
                # Log only first time we take over and then every 5 minutes
                if not hasattr(self, "_follower_takeover_logged") or \
                   (time.time() - self._follower_takeover_logged) > 300:
                    logger.warning(
                        f"[{self.name}] FOLLOWER TAKEOVER: Leader unreachable for "
                        f"{leader_unreachable_for:.0f}s, populating queue"
                    )
                    self._follower_takeover_logged = time.time()

        if not should_populate:
            # Log periodically (not every iteration)
            if hasattr(self, "_last_role_log") and (time.time() - self._last_role_log) < 300:
                return  # Skip logging, already logged recently
            self._last_role_log = time.time()
            logger.debug(f"[{self.name}] Skipping - role={role.value}, leader alive")
            return

        # Check if populator is enabled
        if not self._populator.config.enabled:
            logger.debug(f"[{self.name}] Populator disabled in config")
            return

        # Dec 31, 2025: Log work queue availability
        # Jan 1, 2026: Fixed to use get_queue_status() instead of non-existent depth()
        from app.coordination.work_queue import get_work_queue
        wq = get_work_queue()
        if wq is None:
            logger.warning(f"[{self.name}] Work queue not available")
            return
        wq_status = wq.get_queue_status() if hasattr(wq, 'get_queue_status') else {}
        current_depth = wq_status.get('total_items', 0)
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

    async def _check_leader_alive(self) -> bool:
        """Check if the P2P leader is reachable.

        Jan 5, 2026 (Session 17.28): Added for follower takeover feature.
        Checks leader health via P2P status endpoint.

        Returns:
            True if leader is alive and reachable, False otherwise
        """
        try:
            import aiohttp

            # Get leader info from P2P status
            async with aiohttp.ClientSession() as session:
                # Check local P2P status to get leader info
                async with session.get(
                    "http://localhost:8770/status",
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as resp:
                    if resp.status != 200:
                        return False
                    status = await resp.json()
                    leader_id = status.get("leader_id")
                    if not leader_id:
                        return False

                    # Check if we're the leader (shouldn't happen, but be safe)
                    node_id = status.get("node_id")
                    if leader_id == node_id:
                        return True  # We're the leader, consider it alive

                    # Get leader's host info from all_peers
                    all_peers = status.get("all_peers", {})
                    leader_info = all_peers.get(leader_id)
                    if not leader_info:
                        # Leader not in peer list - likely unreachable
                        return False

                    # Check if leader is marked alive
                    return leader_info.get("is_alive", False)

        except asyncio.TimeoutError:
            logger.debug(f"[{self.name}] Timeout checking leader status")
            return False
        except Exception as e:
            logger.debug(f"[{self.name}] Error checking leader status: {e}")
            return False

    async def _on_consecutive_timeouts(self) -> None:
        """Handle consecutive timeouts by logging thread pool diagnostics.

        Mar 2026: Consecutive timeouts in queue_populator almost always indicate
        thread pool starvation — asyncio.to_thread(populate) is waiting for a
        thread that never becomes available because other callers (heartbeat's
        _update_self_info, leader ops, etc.) have consumed all 8 slots.

        We log diagnostics to help identify the bottleneck and temporarily
        increase the interval to reduce thread pool contention.
        """
        timeout_count = self._stats.consecutive_timeouts
        logger.warning(
            f"[{self.name}] {timeout_count} consecutive timeouts detected. "
            f"Likely cause: thread pool starvation (8 workers, too many "
            f"asyncio.to_thread callers). Increasing interval to reduce pressure."
        )

        # Temporarily increase interval to reduce thread pool pressure.
        # Normal interval is 60s; double it for each consecutive timeout batch,
        # capped at 5 minutes. This gives other loops breathing room.
        self.interval = min(300.0, self.interval * 1.5)
        logger.info(
            f"[{self.name}] Interval increased to {self.interval:.0f}s "
            f"(will reset to {POPULATOR_INTERVAL}s on next success)"
        )

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

    def health_check(self) -> Any:
        """Check loop health with queue populator-specific status.

        Jan 2026: Added for DaemonManager integration.
        Reports initialization status, queue depth, and config coverage.

        Returns:
            HealthCheckResult with queue populator status
        """
        try:
            from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
        except ImportError:
            # Fallback if protocols not available
            return {
                "healthy": self.running,
                "status": "running" if self.running else "stopped",
                "message": f"QueuePopulatorLoop {'running' if self.running else 'stopped'}",
                "details": self.get_status(),
            }

        # Not running
        if not self.running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="QueuePopulatorLoop is stopped",
                details={"running": False},
            )

        # Check if we're the leader (only leader populates, unless follower takeover)
        from scripts.p2p.types import NodeRole

        role = self._get_role()
        if role != NodeRole.LEADER:
            # Jan 5, 2026: Report follower takeover status
            leader_unreachable_for = time.time() - self._leader_last_seen
            is_taking_over = leader_unreachable_for > self._leader_unreachable_threshold
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message=f"QueuePopulatorLoop {'taking over (leader unreachable)' if is_taking_over else 'idle (not leader)'}",
                details={
                    "role": role.value,
                    "is_leader": False,
                    "leader_unreachable_seconds": round(leader_unreachable_for, 1),
                    "follower_takeover_active": is_taking_over,
                    "takeover_threshold_seconds": self._leader_unreachable_threshold,
                },
            )

        # Not initialized yet
        if not self._initialized or self._populator is None:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message="QueuePopulatorLoop not yet initialized",
                details={"initialized": False, "is_leader": True},
            )

        # Get populator status
        try:
            populator_status = self._populator.get_status()
            queue_depth = populator_status.get("current_queue_depth", 0)
            configs_unmet = populator_status.get("configs_unmet", 0)
            total_configs = populator_status.get("total_configs", 12)
            min_depth = populator_status.get("min_queue_depth", 50)

            # Check if queue is dangerously low
            if queue_depth < min_depth // 2:
                return HealthCheckResult(
                    healthy=True,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"Queue depth critically low ({queue_depth}/{min_depth})",
                    details={
                        "queue_depth": queue_depth,
                        "min_queue_depth": min_depth,
                        "configs_unmet": configs_unmet,
                        "total_configs": total_configs,
                        "is_leader": True,
                    },
                )

            # Healthy
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message=f"QueuePopulatorLoop healthy (depth={queue_depth}, unmet={configs_unmet}/{total_configs})",
                details={
                    "queue_depth": queue_depth,
                    "min_queue_depth": min_depth,
                    "configs_unmet": configs_unmet,
                    "total_configs": total_configs,
                    "interval_seconds": self.interval,
                    "is_leader": True,
                },
            )

        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"QueuePopulatorLoop status check failed: {e}",
                details={"error": str(e), "is_leader": True},
            )
