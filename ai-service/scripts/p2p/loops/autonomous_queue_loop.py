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

from app.core.async_context import safe_create_task

from .base import BaseLoop, LoopStats

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Default configuration constants
DEFAULT_NO_LEADER_THRESHOLD_SECONDS = 15.0
DEFAULT_QUEUE_STARVATION_THRESHOLD = 10  # Queue depth below which is "starved"
DEFAULT_QUEUE_STARVATION_DURATION_SECONDS = 10.0  # 10s of starvation (was 30s)
DEFAULT_HEALTHY_QUEUE_DEPTH = 50  # Queue depth above which is "healthy"
DEFAULT_LEADER_STABLE_DURATION_SECONDS = 5.0  # Leader stable for 5s to deactivate (was 10s)
DEFAULT_LOCAL_QUEUE_TARGET = 20  # Target local queue size when activated
DEFAULT_CHECK_INTERVAL_SECONDS = 15.0  # Check every 15s (was 30s)
DEFAULT_POPULATION_BATCH_SIZE = 10  # Add more items per cycle (was 5)


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


class AutonomousQueuePopulationLoop(BaseLoop):
    """Fallback queue population when leader is unavailable.

    This loop runs on ALL nodes and activates as a fallback when:
    1. No leader has been seen for >5 minutes, OR
    2. Work queue has been starved (<10 items) for >2 minutes

    When activated, it maintains a local work queue that workers can pull from
    when the P2P work queue is empty. This ensures GPU utilization continues
    even during network partitions or leader election failures.

    The loop automatically deactivates when conditions return to normal.

    Jan 4, 2026: Refactored to inherit from BaseLoop for consistent lifecycle
    management, statistics tracking, and LoopManager compatibility.
    """

    # Class-level attributes for LoopManager registration
    depends_on: list[str] = []  # No dependencies - this loop is a fallback

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
        self._config = config or AutonomousQueueConfig.from_env()

        # Call BaseLoop __init__ with proper parameters
        super().__init__(
            name="autonomous_queue_population",
            interval=self._config.check_interval_seconds,
            enabled=self._config.enabled,
            depends_on=[],  # No dependencies - this is a fallback
        )

        self._orchestrator = orchestrator
        self._state = AutonomousQueueState()
        self._local_queue: list[dict[str, Any]] = []
        self._local_queue_lock = asyncio.Lock()
        self._startup_time = time.time()
        self._grace_period_complete = False
        self._selfplay_enabled: bool | None = None  # Cached from node config
        self._selfplay_enabled_checked = False

    @property
    def is_activated(self) -> bool:
        """Check if autonomous mode is currently active."""
        return self._state.activated

    @property
    def local_queue_depth(self) -> int:
        """Get current depth of local queue."""
        return len(self._local_queue)

    def _is_selfplay_enabled_for_node(self) -> bool:
        """Check if selfplay is enabled for this node via YAML config.

        Jan 5, 2026: Added to prevent coordinator nodes (mac-studio) from
        spawning local selfplay work when selfplay_enabled: false in config.
        This was causing OOM on coordinator nodes.

        Returns:
            True if selfplay is enabled (or config can't be read), False if disabled
        """
        if self._selfplay_enabled_checked:
            return self._selfplay_enabled if self._selfplay_enabled is not None else True

        self._selfplay_enabled_checked = True

        try:
            import socket

            # Try to get node config from orchestrator
            node_id = getattr(self._orchestrator, "node_id", None)
            hostname = socket.gethostname()
            if not node_id:
                return True  # Can't determine, allow by default

            # Check for cached cluster config
            cluster_config = getattr(self._orchestrator, "_cluster_config", None)
            hostname_lower = hostname.lower().replace("-", "").replace("_", "")
            if cluster_config:
                # YAML uses "hosts" key, not "nodes"
                hosts = cluster_config.get("hosts", {})
                if hosts:
                    # First try direct lookup by node_id
                    node_cfg = hosts.get(node_id, {})

                    # If not found, search by hostname or role match
                    if not node_cfg:
                        for config_name, cfg in hosts.items():
                            if not isinstance(cfg, dict):
                                continue
                            config_name_lower = config_name.lower().replace("-", "").replace("_", "")
                            # Match if hostname contains config name or vice versa
                            if hostname_lower in config_name_lower or config_name_lower in hostname_lower:
                                node_cfg = cfg
                                logger.debug(
                                    f"[AutonomousQueue] Matched hostname {hostname} to config {config_name}"
                                )
                                break
                            # Special case: MacBook hostname maps to mac-studio or local-mac config
                            if "macbook" in hostname_lower and config_name_lower in ("macstudio", "localmac"):
                                node_cfg = cfg
                                logger.info(
                                    f"[AutonomousQueue] Matched MacBook hostname {hostname} to {config_name} config"
                                )
                                break

                    if node_cfg:
                        self._selfplay_enabled = node_cfg.get("selfplay_enabled", True)
                        if not self._selfplay_enabled:
                            logger.info(
                                f"[AutonomousQueue] Node {node_id} (hostname={hostname}) has selfplay_enabled=false, "
                                "disabling autonomous queue population"
                            )
                        return self._selfplay_enabled

                # Also check elo_sync.coordinator as an indicator
                elo_sync = cluster_config.get("elo_sync", {})
                coordinator_name = elo_sync.get("coordinator", "")
                if coordinator_name and coordinator_name.lower().replace("-", "") in hostname_lower.replace("-", ""):
                    logger.info(
                        f"[AutonomousQueue] Node {node_id} matches elo_sync coordinator {coordinator_name}, "
                        "disabling autonomous queue population"
                    )
                    self._selfplay_enabled = False
                    return False

            # Fallback: check role - coordinators shouldn't run selfplay
            role = getattr(self._orchestrator, "_role", None)
            if role == "coordinator":
                logger.info(
                    f"[AutonomousQueue] Node {node_id} is coordinator, "
                    "disabling autonomous queue population"
                )
                self._selfplay_enabled = False
                return False

            return True  # Default to enabled if can't determine
        except Exception as e:
            logger.debug(f"[AutonomousQueue] Error checking selfplay_enabled config: {e}")
            return True  # Default to enabled on error

    def start_background(self) -> asyncio.Task | None:
        """Start the loop as a background task (for LoopManager compatibility).

        Returns:
            The asyncio.Task running the loop, or None if disabled/already running
        """
        if not self.enabled:
            logger.info("[AutonomousQueue] Disabled via config, not starting")
            return None
        # Use parent class start method
        self._task = safe_create_task(self.run_forever(), name="autonomous-queue-loop")
        logger.info(
            f"[AutonomousQueue] Started with config: "
            f"no_leader_threshold={self._config.no_leader_threshold_seconds}s, "
            f"starvation_threshold={self._config.queue_starvation_threshold}, "
            f"check_interval={self._config.check_interval_seconds}s"
        )
        return self._task

    async def _on_stop(self) -> None:
        """Called when the loop stops - deactivate if active."""
        if self._state.activated:
            self._state.deactivate("loop_stopped")
        logger.info("[AutonomousQueue] Stopped")

    def get_status(self) -> dict[str, Any]:
        """Get status for LoopManager health checks."""
        return {
            "name": self.name,
            "running": self.running,
            "enabled": self.enabled,
            "activated": self._state.activated,
            "activation_reason": self._state.activation_reason,
            "items_populated": self._state.items_populated,
            "local_queue_depth": len(self._local_queue),
        }

    async def _run_once(self) -> None:
        """Single iteration - called by BaseLoop.run_forever().

        Jan 4, 2026: Refactored from _run_loop to match BaseLoop interface.
        Jan 5, 2026: Added selfplay_enabled check to prevent OOM on coordinator nodes.
        Jan 13, 2026: Reduced grace period from 60s to 10s for faster GPU utilization.
        """
        # Grace period at startup to allow normal initialization
        # 10 seconds is enough for P2P to connect and discover leader
        grace_period = 10.0  # 10 seconds (was 60s - too long for GPU nodes)
        if not self._grace_period_complete:
            if time.time() - self._startup_time < grace_period:
                return  # Skip this iteration during grace period
            self._grace_period_complete = True

        # Check if selfplay is enabled for this node - prevents OOM on coordinators
        if not self._is_selfplay_enabled_for_node():
            # Deactivate if previously activated
            if self._state.activated:
                self._state.deactivate("selfplay_disabled_for_node")
            return

        await self._check_and_update_state()

        if self._state.activated:
            await self._populate_local_queue()

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

        January 2026: Added starvation override to ensure data-starved configs
        (especially 3p and 4p) get allocation even when they aren't highest priority.
        This fixes the issue where square19_4p and hexagonal_4p had only 61-161 games
        while 2p configs had thousands.
        """
        import random

        try:
            # Try to get config priorities from selfplay scheduler
            scheduler = getattr(self._orchestrator, "selfplay_scheduler", None)

            # January 2026: Starvation override for 3p/4p configs
            # Get game counts if available to find starved configs
            starved_configs = self._get_starved_configs(scheduler)
            if starved_configs:
                # 70% chance to force allocation to a starved config
                # This aggressively rebalances data while still allowing some
                # priority-based allocation
                if random.random() < 0.7:
                    config_key = random.choice(starved_configs)
                    logger.debug(
                        f"[AutonomousQueue] Starvation override: allocating to {config_key}"
                    )
                    return self._make_selfplay_work_item(config_key)

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

    def _get_starved_configs(self, scheduler: Any) -> list[str]:
        """Get list of data-starved configs that need urgent allocation.

        January 2026: Identifies configs with critically low game counts.

        Returns configs with:
        - 4p configs with < 500 games (critical starvation)
        - 3p configs with < 1000 games (moderate starvation)

        Args:
            scheduler: SelfplayScheduler instance (may be None)

        Returns:
            List of config keys that are starved, prioritizing 4p configs.
        """
        starved = []

        # Try to get game counts from scheduler
        game_counts: dict[str, int] = {}
        if scheduler:
            # Try multiple possible attributes for game counts
            for attr in ["_game_counts", "game_counts", "_config_game_counts"]:
                counts = getattr(scheduler, attr, None)
                if counts and isinstance(counts, dict):
                    game_counts = counts
                    break

        # If no game counts from scheduler, use fallback estimation
        if not game_counts:
            try:
                from app.coordination.selfplay_priority_types import get_config_game_counts
                game_counts = get_config_game_counts()
            except ImportError:
                pass

        # Define starvation thresholds
        # 4p configs have been severely underallocated - use aggressive threshold
        STARVATION_4P = 500   # 4p configs with < 500 games are critical
        STARVATION_3P = 1000  # 3p configs with < 1000 games are starved

        # All canonical configs for reference
        all_4p = ["hex8_4p", "square8_4p", "square19_4p", "hexagonal_4p"]
        all_3p = ["hex8_3p", "square8_3p", "square19_3p", "hexagonal_3p"]

        # First prioritize 4p configs (most underserved)
        for cfg in all_4p:
            count = game_counts.get(cfg, 0)
            if count < STARVATION_4P:
                starved.append(cfg)

        # Then add starved 3p configs
        for cfg in all_3p:
            count = game_counts.get(cfg, 0)
            if count < STARVATION_3P:
                starved.append(cfg)

        return starved

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
            "engine_mode": "mixed",  # Jan 12, 2026: Enable harness diversity
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

    async def claim_local_work(
        self,
        node_id: str,
        capabilities: list[str] | None = None
    ) -> dict[str, Any] | None:
        """Claim work from local autonomous queue.

        Jan 4, 2026: Added for Phase 4 - pull-based training job claim.

        Args:
            node_id: ID of the node claiming work
            capabilities: Optional list of capabilities (e.g., ["cuda", "training"])

        Returns:
            Claimed work item dict or None if no matching work available
        """
        if not self._state.activated:
            return None

        async with self._local_queue_lock:
            for i, item in enumerate(self._local_queue):
                # Check capability match if required
                required = item.get("required_capabilities", [])
                if required and capabilities:
                    if not set(required).issubset(set(capabilities)):
                        continue

                # Claim the item
                claimed = self._local_queue.pop(i)
                claimed["claimed_by"] = node_id
                claimed["claimed_at"] = time.time()
                return claimed

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
        """Emit event when autonomous mode activates.

        Jan 5, 2026: Migrated to safe_emit_event for consistent error handling.
        """
        try:
            from app.coordination.event_emission_helpers import safe_emit_event
            from app.distributed.data_events import DataEventType

            success = safe_emit_event(
                DataEventType.AUTONOMOUS_QUEUE_ACTIVATED,
                {
                    "node_id": getattr(self._orchestrator, "node_id", "unknown"),
                    "reason": reason,
                    "no_leader_duration": self._state.get_no_leader_duration(),
                    "starvation_duration": self._state.get_starvation_duration(),
                    "timestamp": time.time(),
                },
                context="AutonomousQueue",
                log_after=f"Autonomous queue activated: {reason}",
            )
            if not success:
                logger.debug(f"[AutonomousQueue] Event emission returned False for activation")
        except ImportError:
            pass  # Event modules not available

    def _emit_deactivation_event(self, reason: str) -> None:
        """Emit event when autonomous mode deactivates.

        Jan 5, 2026: Migrated to safe_emit_event for consistent error handling.
        """
        try:
            from app.coordination.event_emission_helpers import safe_emit_event
            from app.distributed.data_events import DataEventType

            success = safe_emit_event(
                DataEventType.AUTONOMOUS_QUEUE_DEACTIVATED,
                {
                    "node_id": getattr(self._orchestrator, "node_id", "unknown"),
                    "reason": reason,
                    "items_populated": self._state.items_populated,
                    "activation_duration": time.time() - (self._state.activation_time or time.time()),
                    "timestamp": time.time(),
                },
                context="AutonomousQueue",
                log_after=f"Autonomous queue deactivated: {reason}",
            )
            if not success:
                logger.debug(f"[AutonomousQueue] Event emission returned False for deactivation")
        except ImportError:
            pass  # Event modules not available

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
