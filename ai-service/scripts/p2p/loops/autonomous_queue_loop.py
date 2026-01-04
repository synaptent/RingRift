"""Autonomous Queue Population Loop.

Jan 4, 2026 - Phase 2 of P2P Cluster Resilience.

Problem: QueuePopulatorLoop only runs on the leader node. When leader election fails
or the leader is unreachable, no work is added to the queue and GPU nodes sit idle.

Solution: AutonomousQueuePopulationLoop runs on ALL nodes as a fallback. It monitors:
1. Leader availability - activates if no leader_id for >5 minutes
2. Queue depth - activates if queue_depth < 10 for >2 minutes

When activated, it populates a LOCAL work queue that workers can pull from when
the P2P work queue is empty. This ensures GPU utilization even during partitions.

The loop automatically deactivates when:
- A leader is elected and has been stable for >60 seconds
- The P2P work queue has >50 items (healthy)

Usage:
    from scripts.p2p.loops.autonomous_queue_loop import (
        AutonomousQueuePopulationLoop,
        AutonomousQueueConfig,
    )

    config = AutonomousQueueConfig()
    loop = AutonomousQueuePopulationLoop(orchestrator, config)
    loop.start()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Default configuration constants
DEFAULT_NO_LEADER_THRESHOLD_SECONDS = 300.0  # 5 minutes without leader
DEFAULT_QUEUE_STARVATION_THRESHOLD = 10  # Queue depth below which is "starved"
DEFAULT_QUEUE_STARVATION_DURATION_SECONDS = 120.0  # 2 minutes of starvation
DEFAULT_HEALTHY_QUEUE_DEPTH = 50  # Queue depth above which is "healthy"
DEFAULT_LEADER_STABLE_DURATION_SECONDS = 60.0  # Leader must be stable for 60s to deactivate
DEFAULT_LOCAL_QUEUE_TARGET = 20  # Target local queue size when activated
DEFAULT_CHECK_INTERVAL_SECONDS = 30.0  # How often to check conditions
DEFAULT_POPULATION_BATCH_SIZE = 5  # How many items to add per cycle


@dataclass
class AutonomousQueueConfig:
    """Configuration for autonomous queue population.

    Attributes:
        no_leader_threshold_seconds: Activate after this many seconds without leader
        queue_starvation_threshold: Queue depth below this is "starved"
        queue_starvation_duration_seconds: Activate after starved for this long
        healthy_queue_depth: Queue depth above this is "healthy" (deactivate)
        leader_stable_duration_seconds: Leader must be stable this long to deactivate
        local_queue_target: Target size for local queue when activated
        check_interval_seconds: How often to check activation conditions
        population_batch_size: How many items to add per population cycle
        enabled: Whether this loop is enabled at all
    """

    no_leader_threshold_seconds: float = DEFAULT_NO_LEADER_THRESHOLD_SECONDS
    queue_starvation_threshold: int = DEFAULT_QUEUE_STARVATION_THRESHOLD
    queue_starvation_duration_seconds: float = DEFAULT_QUEUE_STARVATION_DURATION_SECONDS
    healthy_queue_depth: int = DEFAULT_HEALTHY_QUEUE_DEPTH
    leader_stable_duration_seconds: float = DEFAULT_LEADER_STABLE_DURATION_SECONDS
    local_queue_target: int = DEFAULT_LOCAL_QUEUE_TARGET
    check_interval_seconds: float = DEFAULT_CHECK_INTERVAL_SECONDS
    population_batch_size: int = DEFAULT_POPULATION_BATCH_SIZE
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "AutonomousQueueConfig":
        """Create config from environment variables."""
        import os

        return cls(
            no_leader_threshold_seconds=float(
                os.environ.get("RINGRIFT_AUTONOMOUS_QUEUE_NO_LEADER_THRESHOLD", DEFAULT_NO_LEADER_THRESHOLD_SECONDS)
            ),
            queue_starvation_threshold=int(
                os.environ.get("RINGRIFT_AUTONOMOUS_QUEUE_STARVATION_THRESHOLD", DEFAULT_QUEUE_STARVATION_THRESHOLD)
            ),
            queue_starvation_duration_seconds=float(
                os.environ.get("RINGRIFT_AUTONOMOUS_QUEUE_STARVATION_DURATION", DEFAULT_QUEUE_STARVATION_DURATION_SECONDS)
            ),
            healthy_queue_depth=int(
                os.environ.get("RINGRIFT_AUTONOMOUS_QUEUE_HEALTHY_DEPTH", DEFAULT_HEALTHY_QUEUE_DEPTH)
            ),
            leader_stable_duration_seconds=float(
                os.environ.get("RINGRIFT_AUTONOMOUS_QUEUE_LEADER_STABLE_DURATION", DEFAULT_LEADER_STABLE_DURATION_SECONDS)
            ),
            local_queue_target=int(
                os.environ.get("RINGRIFT_AUTONOMOUS_QUEUE_LOCAL_TARGET", DEFAULT_LOCAL_QUEUE_TARGET)
            ),
            check_interval_seconds=float(
                os.environ.get("RINGRIFT_AUTONOMOUS_QUEUE_CHECK_INTERVAL", DEFAULT_CHECK_INTERVAL_SECONDS)
            ),
            population_batch_size=int(
                os.environ.get("RINGRIFT_AUTONOMOUS_QUEUE_BATCH_SIZE", DEFAULT_POPULATION_BATCH_SIZE)
            ),
            enabled=os.environ.get("RINGRIFT_AUTONOMOUS_QUEUE_ENABLED", "true").lower() == "true",
        )


@dataclass
class AutonomousQueueState:
    """Tracks the state of autonomous queue activation.

    Attributes:
        activated: Whether autonomous mode is currently active
        activation_time: When autonomous mode was activated (None if not active)
        activation_reason: Why autonomous mode was activated
        last_leader_seen: When a leader was last seen
        queue_starved_since: When queue starvation started (None if not starved)
        items_populated: Total items added while in autonomous mode
        deactivation_count: How many times we've deactivated
    """

    activated: bool = False
    activation_time: float | None = None
    activation_reason: str = ""
    last_leader_seen: float = field(default_factory=time.time)
    queue_starved_since: float | None = None
    items_populated: int = 0
    deactivation_count: int = 0

    def activate(self, reason: str) -> None:
        """Activate autonomous mode with given reason."""
        self.activated = True
        self.activation_time = time.time()
        self.activation_reason = reason
        logger.warning(
            f"[AutonomousQueue] ACTIVATED: {reason} "
            f"(no_leader_duration={time.time() - self.last_leader_seen:.0f}s)"
        )

    def deactivate(self, reason: str) -> None:
        """Deactivate autonomous mode."""
        if self.activated:
            duration = time.time() - (self.activation_time or time.time())
            logger.info(
                f"[AutonomousQueue] Deactivated after {duration:.0f}s: {reason} "
                f"(populated {self.items_populated} items)"
            )
            self.deactivation_count += 1
        self.activated = False
        self.activation_time = None
        self.activation_reason = ""
        self.items_populated = 0

    def record_leader_seen(self) -> None:
        """Record that we've seen a leader."""
        self.last_leader_seen = time.time()

    def record_queue_healthy(self) -> None:
        """Record that queue depth is healthy."""
        self.queue_starved_since = None

    def record_queue_starved(self) -> None:
        """Record that queue is starved (if not already tracking)."""
        if self.queue_starved_since is None:
            self.queue_starved_since = time.time()

    def get_no_leader_duration(self) -> float:
        """Get how long we've been without a leader."""
        return time.time() - self.last_leader_seen

    def get_starvation_duration(self) -> float:
        """Get how long queue has been starved (0 if not starved)."""
        if self.queue_starved_since is None:
            return 0.0
        return time.time() - self.queue_starved_since


class AutonomousQueuePopulationLoop:
    """Fallback queue population when leader is unavailable.

    This loop runs on ALL nodes and activates as a fallback when:
    1. No leader has been seen for >5 minutes, OR
    2. Work queue has been starved (<10 items) for >2 minutes

    When activated, it maintains a local work queue that workers can pull from
    when the P2P work queue is empty. This ensures GPU utilization continues
    even during network partitions or leader election failures.

    The loop automatically deactivates when conditions return to normal.
    """

    def __init__(
        self,
        orchestrator: Any,
        config: AutonomousQueueConfig | None = None,
    ) -> None:
        """Initialize the autonomous queue loop.

        Args:
            orchestrator: P2POrchestrator instance (provides leader_id, work_queue access)
            config: Configuration for the loop (defaults to env-based config)
        """
        self._orchestrator = orchestrator
        self._config = config or AutonomousQueueConfig.from_env()
        self._state = AutonomousQueueState()
        self._local_queue: list[dict[str, Any]] = []
        self._local_queue_lock = asyncio.Lock()
        self._running = False
        self._task: asyncio.Task | None = None
        self._startup_time = time.time()

    @property
    def is_activated(self) -> bool:
        """Check if autonomous mode is currently active."""
        return self._state.activated

    @property
    def local_queue_depth(self) -> int:
        """Get current depth of local queue."""
        return len(self._local_queue)

    def start(self) -> None:
        """Start the autonomous queue loop."""
        if not self._config.enabled:
            logger.info("[AutonomousQueue] Disabled via config, not starting")
            return

        if self._running:
            return

        self._running = True
        self._startup_time = time.time()
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            f"[AutonomousQueue] Started with config: "
            f"no_leader_threshold={self._config.no_leader_threshold_seconds}s, "
            f"starvation_threshold={self._config.queue_starvation_threshold}, "
            f"check_interval={self._config.check_interval_seconds}s"
        )

    async def stop(self) -> None:
        """Stop the autonomous queue loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._state.activated:
            self._state.deactivate("loop_stopped")

        logger.info("[AutonomousQueue] Stopped")

    async def _run_loop(self) -> None:
        """Main loop that checks conditions and populates queue."""
        # Grace period at startup to allow normal initialization
        grace_period = 60.0  # 1 minute grace period
        await asyncio.sleep(min(grace_period, self._config.check_interval_seconds))

        while self._running:
            try:
                await self._check_and_update_state()

                if self._state.activated:
                    await self._populate_local_queue()

                await asyncio.sleep(self._config.check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[AutonomousQueue] Loop error: {e}")
                await asyncio.sleep(self._config.check_interval_seconds)

    async def _check_and_update_state(self) -> None:
        """Check activation conditions and update state."""
        # Get current leader_id
        leader_id = getattr(self._orchestrator, "leader_id", None)
        queue_depth = self._get_p2p_queue_depth()

        # Update leader tracking
        if leader_id:
            self._state.record_leader_seen()

        # Update queue tracking
        if queue_depth >= self._config.queue_starvation_threshold:
            self._state.record_queue_healthy()
        else:
            self._state.record_queue_starved()

        # Check activation conditions
        if not self._state.activated:
            self._check_should_activate(leader_id, queue_depth)
        else:
            self._check_should_deactivate(leader_id, queue_depth)

    def _check_should_activate(self, leader_id: str | None, queue_depth: int) -> None:
        """Check if we should activate autonomous mode."""
        no_leader_duration = self._state.get_no_leader_duration()
        starvation_duration = self._state.get_starvation_duration()

        # Condition 1: No leader for too long
        if no_leader_duration >= self._config.no_leader_threshold_seconds:
            self._state.activate(f"no_leader_for_{no_leader_duration:.0f}s")
            self._emit_activation_event("no_leader")
            return

        # Condition 2: Queue starved for too long (even with leader)
        if starvation_duration >= self._config.queue_starvation_duration_seconds:
            self._state.activate(
                f"queue_starved_for_{starvation_duration:.0f}s (depth={queue_depth})"
            )
            self._emit_activation_event("queue_starvation")
            return

    def _check_should_deactivate(self, leader_id: str | None, queue_depth: int) -> None:
        """Check if we should deactivate autonomous mode."""
        # Condition 1: Leader is back and stable
        if leader_id:
            no_leader_duration = self._state.get_no_leader_duration()
            if no_leader_duration < self._config.leader_stable_duration_seconds:
                # Leader has been present for a while, but check queue health too
                if queue_depth >= self._config.healthy_queue_depth:
                    self._state.deactivate(f"leader_stable_and_queue_healthy (depth={queue_depth})")
                    self._emit_deactivation_event("leader_recovered")
                    return

        # Condition 2: Queue is very healthy (even without stable leader)
        if queue_depth >= self._config.healthy_queue_depth * 2:
            self._state.deactivate(f"queue_very_healthy (depth={queue_depth})")
            self._emit_deactivation_event("queue_recovered")
            return

    async def _populate_local_queue(self) -> None:
        """Populate local queue with work items."""
        async with self._local_queue_lock:
            current_depth = len(self._local_queue)
            if current_depth >= self._config.local_queue_target:
                return

            items_to_add = min(
                self._config.population_batch_size,
                self._config.local_queue_target - current_depth,
            )

            for _ in range(items_to_add):
                work_item = self._create_work_item()
                if work_item:
                    self._local_queue.append(work_item)
                    self._state.items_populated += 1

            if items_to_add > 0:
                logger.debug(
                    f"[AutonomousQueue] Added {items_to_add} items to local queue "
                    f"(depth now {len(self._local_queue)})"
                )

    def _create_work_item(self) -> dict[str, Any] | None:
        """Create a work item for local queue.

        Returns a selfplay work item for a config that needs more games.
        """
        try:
            # Try to get config priorities from selfplay scheduler
            scheduler = getattr(self._orchestrator, "selfplay_scheduler", None)
            if scheduler and hasattr(scheduler, "get_config_priorities"):
                priorities = scheduler.get_config_priorities()
                if priorities:
                    # Pick highest priority config
                    config_key = max(priorities, key=lambda k: priorities[k])
                    return self._make_selfplay_work_item(config_key)

            # Fallback: rotate through canonical configs
            configs = [
                "hex8_2p", "hex8_3p", "hex8_4p",
                "square8_2p", "square8_3p", "square8_4p",
                "square19_2p", "square19_3p", "square19_4p",
                "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
            ]
            # Use items_populated to rotate
            config_key = configs[self._state.items_populated % len(configs)]
            return self._make_selfplay_work_item(config_key)

        except Exception as e:
            logger.debug(f"[AutonomousQueue] Failed to create work item: {e}")
            return None

    def _make_selfplay_work_item(self, config_key: str) -> dict[str, Any]:
        """Create a selfplay work item for the given config."""
        import uuid

        parts = config_key.rsplit("_", 1)
        board_type = parts[0]
        num_players = int(parts[1].rstrip("p")) if len(parts) > 1 else 2

        return {
            "work_id": f"autonomous-{uuid.uuid4().hex[:8]}",
            "work_type": "selfplay",
            "config_key": config_key,
            "board_type": board_type,
            "num_players": num_players,
            "source": "autonomous_queue",
            "priority": 50,  # Medium priority
            "created_at": time.time(),
            "games_requested": 10,  # Small batch for quick iteration
        }

    async def pop_local_work(self) -> dict[str, Any] | None:
        """Pop a work item from the local queue.

        Called by workers when P2P queue is empty and autonomous mode is active.

        Returns:
            Work item dict or None if queue is empty
        """
        if not self._state.activated:
            return None

        async with self._local_queue_lock:
            if self._local_queue:
                return self._local_queue.pop(0)
        return None

    def get_local_queue_depth(self) -> int:
        """Get current local queue depth."""
        return len(self._local_queue)

    def _get_p2p_queue_depth(self) -> int:
        """Get current P2P work queue depth."""
        try:
            # Try to get from orchestrator's work queue
            work_queue = getattr(self._orchestrator, "_work_queue", None)
            if work_queue and hasattr(work_queue, "qsize"):
                return work_queue.qsize()

            # Try state manager
            state_manager = getattr(self._orchestrator, "state_manager", None)
            if state_manager and hasattr(state_manager, "get_work_queue_depth"):
                return state_manager.get_work_queue_depth()

            # Fallback: check if we're the leader and have stats
            if getattr(self._orchestrator, "leader_id", None) == getattr(self._orchestrator, "node_id", None):
                stats = getattr(self._orchestrator, "_work_queue_stats", {})
                return stats.get("depth", 0)

            return 0
        except Exception:
            return 0

    def _emit_activation_event(self, reason: str) -> None:
        """Emit event when autonomous mode activates."""
        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import emit_event

            emit_event(DataEventType.AUTONOMOUS_QUEUE_ACTIVATED, {
                "node_id": getattr(self._orchestrator, "node_id", "unknown"),
                "reason": reason,
                "no_leader_duration": self._state.get_no_leader_duration(),
                "starvation_duration": self._state.get_starvation_duration(),
                "timestamp": time.time(),
            })
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[AutonomousQueue] Failed to emit activation event: {e}")

    def _emit_deactivation_event(self, reason: str) -> None:
        """Emit event when autonomous mode deactivates."""
        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import emit_event

            emit_event(DataEventType.AUTONOMOUS_QUEUE_DEACTIVATED, {
                "node_id": getattr(self._orchestrator, "node_id", "unknown"),
                "reason": reason,
                "items_populated": self._state.items_populated,
                "activation_duration": time.time() - (self._state.activation_time or time.time()),
                "timestamp": time.time(),
            })
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[AutonomousQueue] Failed to emit deactivation event: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get current status of autonomous queue system."""
        return {
            "enabled": self._config.enabled,
            "running": self._running,
            "activated": self._state.activated,
            "activation_reason": self._state.activation_reason,
            "activation_time": self._state.activation_time,
            "local_queue_depth": len(self._local_queue),
            "items_populated_total": self._state.items_populated,
            "deactivation_count": self._state.deactivation_count,
            "no_leader_duration": self._state.get_no_leader_duration(),
            "starvation_duration": self._state.get_starvation_duration(),
            "config": {
                "no_leader_threshold": self._config.no_leader_threshold_seconds,
                "queue_starvation_threshold": self._config.queue_starvation_threshold,
                "queue_starvation_duration": self._config.queue_starvation_duration_seconds,
                "healthy_queue_depth": self._config.healthy_queue_depth,
                "local_queue_target": self._config.local_queue_target,
            },
        }

    def health_check(self) -> dict[str, Any]:
        """Return health check result."""
        return {
            "healthy": self._running or not self._config.enabled,
            "details": self.get_status(),
        }
