"""Memory Pressure Controller for proactive OOM prevention.

This module implements a 4-tier graduated response system to prevent memory
exhaustion. Unlike reactive monitoring that only alerts, this controller
takes automated action at each tier to reduce memory pressure before OOM.

Tier Progression:
    NORMAL (< 60%): Full operation, no restrictions
    CAUTION (60-70%): Log warning, emit event for dashboards
    WARNING (70-80%): Pause new selfplay jobs, reduce batch sizes
    CRITICAL (80-90%): Kill non-essential daemons, force GC
    EMERGENCY (90%+): Notify standby coordinator, graceful shutdown

Architecture:
    MemoryPressureController monitors system RAM and takes graduated action.
    It integrates with:
    - SelfplayScheduler (pause/resume job spawning)
    - DaemonManager (kill non-essential daemons)
    - BackpressureSignal (contributes to overall pressure)
    - StandbyCoordinator (notify for failover)

Usage:
    from app.coordination.memory_pressure_controller import (
        get_memory_pressure_controller,
        MemoryPressureTier,
    )

    controller = get_memory_pressure_controller()
    await controller.start()

    # Check current tier
    tier = controller.current_tier
    if tier >= MemoryPressureTier.WARNING:
        logger.warning(f"High memory pressure: {tier.name}")

January 2026: Created for cluster resilience architecture after memory
exhaustion failure (Session 16).
"""

from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Callable

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

from app.config.coordination_defaults import MemoryPressureDefaults
from app.coordination.contracts import CoordinatorStatus, HealthCheckResult
from app.coordination.handler_base import HandlerBase

if TYPE_CHECKING:
    from app.coordination.daemon_manager import DaemonManager

logger = logging.getLogger(__name__)


class MemoryPressureTier(IntEnum):
    """Memory pressure tiers with graduated severity."""

    NORMAL = 0  # < 60% - Full operation
    CAUTION = 1  # 60-70% - Warning only
    WARNING = 2  # 70-80% - Throttle operations
    CRITICAL = 3  # 80-90% - Kill non-essential
    EMERGENCY = 4  # 90%+ - Graceful shutdown


@dataclass
class MemoryPressureState:
    """Current memory pressure state."""

    tier: MemoryPressureTier = MemoryPressureTier.NORMAL
    ram_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    swap_percent: float = 0.0
    timestamp: float = field(default_factory=time.time)
    consecutive_samples: int = 0
    last_action_time: float = 0.0
    actions_taken: list[str] = field(default_factory=list)

    @property
    def age_seconds(self) -> float:
        """Seconds since this state was captured."""
        return time.time() - self.timestamp

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tier": self.tier.name,
            "tier_value": self.tier.value,
            "ram_percent": self.ram_percent,
            "ram_used_gb": self.ram_used_gb,
            "ram_total_gb": self.ram_total_gb,
            "swap_percent": self.swap_percent,
            "timestamp": self.timestamp,
            "consecutive_samples": self.consecutive_samples,
            "last_action_time": self.last_action_time,
            "actions_taken": self.actions_taken[-10:],  # Last 10 actions
        }


class MemoryPressureController(HandlerBase):
    """Proactive memory pressure management with graduated response.

    This controller monitors system RAM usage and takes automated action
    to prevent memory exhaustion. Actions escalate through 4 tiers based
    on memory usage percentage.

    Key Features:
    - Hysteresis prevents oscillation between tiers
    - Cooldown prevents rapid repeated actions
    - Consecutive sample requirement prevents false positives
    - Integration with daemon manager for killing non-essentials
    - Event emission for external monitoring

    Inherits from HandlerBase (January 2026):
    - Singleton management (get_instance/reset_instance)
    - Lifecycle management (start/stop)
    - Health check returning HealthCheckResult
    - Fire-and-forget task helpers
    """

    def __init__(
        self,
        config: MemoryPressureDefaults | None = None,
        daemon_manager: DaemonManager | None = None,
    ):
        """Initialize the memory pressure controller.

        Args:
            config: Configuration defaults (uses MemoryPressureDefaults if None)
            daemon_manager: Optional daemon manager for killing daemons
        """
        self._pressure_config = config or MemoryPressureDefaults()

        # Initialize HandlerBase with cycle interval from config
        super().__init__(
            name="memory_pressure_controller",
            config=self._pressure_config,
            cycle_interval=self._pressure_config.CHECK_INTERVAL,
        )

        self._daemon_manager = daemon_manager

        # State
        self._pressure_state = MemoryPressureState()

        # Callbacks for integration
        self._on_tier_change: list[Callable[[MemoryPressureTier], None]] = []
        self._on_emergency: list[Callable[[], None]] = []

        # Tier thresholds (loaded from config)
        self._tier_thresholds = {
            MemoryPressureTier.CAUTION: self._pressure_config.TIER_CAUTION,
            MemoryPressureTier.WARNING: self._pressure_config.TIER_WARNING,
            MemoryPressureTier.CRITICAL: self._pressure_config.TIER_CRITICAL,
            MemoryPressureTier.EMERGENCY: self._pressure_config.TIER_EMERGENCY,
        }

        # Track selfplay pause state
        self._selfplay_paused = False

    @property
    def current_tier(self) -> MemoryPressureTier:
        """Get current memory pressure tier."""
        return self._pressure_state.tier

    @property
    def current_state(self) -> MemoryPressureState:
        """Get current state snapshot."""
        return self._pressure_state

    @property
    def pressure_health_score(self) -> float:
        """Get memory pressure health score for aggregation (0.0-1.0, higher is healthier)."""
        # Map tier to health score
        tier_scores = {
            MemoryPressureTier.NORMAL: 1.0,
            MemoryPressureTier.CAUTION: 0.8,
            MemoryPressureTier.WARNING: 0.5,
            MemoryPressureTier.CRITICAL: 0.2,
            MemoryPressureTier.EMERGENCY: 0.0,
        }
        return tier_scores.get(self._pressure_state.tier, 0.5)

    def _get_memory_usage(self) -> tuple[float, float, float, float]:
        """Get current memory usage.

        Returns:
            Tuple of (ram_percent, ram_used_gb, ram_total_gb, swap_percent)
        """
        if psutil is None:
            return 0.0, 0.0, 0.0, 0.0

        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            return (
                mem.percent,
                mem.used / (1024**3),
                mem.total / (1024**3),
                swap.percent,
            )
        except Exception as e:
            logger.error(f"[MemoryPressure] Failed to get memory info: {e}")
            return 0.0, 0.0, 0.0, 0.0

    def _determine_tier(self, ram_percent: float) -> MemoryPressureTier:
        """Determine which tier we're in based on RAM percentage.

        Args:
            ram_percent: Current RAM usage percentage

        Returns:
            The appropriate memory pressure tier
        """
        if ram_percent >= self._tier_thresholds[MemoryPressureTier.EMERGENCY]:
            return MemoryPressureTier.EMERGENCY
        elif ram_percent >= self._tier_thresholds[MemoryPressureTier.CRITICAL]:
            return MemoryPressureTier.CRITICAL
        elif ram_percent >= self._tier_thresholds[MemoryPressureTier.WARNING]:
            return MemoryPressureTier.WARNING
        elif ram_percent >= self._tier_thresholds[MemoryPressureTier.CAUTION]:
            return MemoryPressureTier.CAUTION
        else:
            return MemoryPressureTier.NORMAL

    def _should_downgrade_tier(self, new_tier: MemoryPressureTier) -> bool:
        """Check if we should downgrade to a lower tier (with hysteresis).

        We only downgrade if RAM has dropped enough below the threshold
        to prevent oscillation.

        Args:
            new_tier: The tier based on raw percentage

        Returns:
            True if we should accept the downgrade
        """
        current_tier = self._pressure_state.tier
        if new_tier >= current_tier:
            return True  # Not a downgrade

        # For downgrade, require hysteresis margin
        threshold_for_current = self._tier_thresholds.get(current_tier, 0)
        required_drop = threshold_for_current - self._pressure_config.HYSTERESIS

        return self._pressure_state.ram_percent < required_drop

    def _can_take_action(self) -> bool:
        """Check if we can take action (respecting cooldown)."""
        if self._pressure_state.last_action_time == 0:
            return True
        elapsed = time.time() - self._pressure_state.last_action_time
        return elapsed >= self._pressure_config.ACTION_COOLDOWN

    def _record_action(self, action: str) -> None:
        """Record an action taken."""
        self._pressure_state.last_action_time = time.time()
        self._pressure_state.actions_taken.append(f"{time.time():.0f}:{action}")
        logger.info(f"[MemoryPressure] Action taken: {action}")

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Emit event for external monitoring."""
        from app.coordination.event_emission_helpers import safe_emit_event

        safe_emit_event(
            event_type,
            payload,
            context="MemoryPressure",
            source="memory_pressure_controller",
        )

    async def _handle_caution_tier(self) -> None:
        """Handle CAUTION tier - logging and monitoring only."""
        logger.warning(
            f"[MemoryPressure] CAUTION: RAM at {self._pressure_state.ram_percent:.1f}% "
            f"({self._pressure_state.ram_used_gb:.1f}/{self._pressure_state.ram_total_gb:.1f} GB)"
        )

        self._emit_event(
            "MEMORY_PRESSURE_CAUTION",
            {
                "tier": "CAUTION",
                "ram_percent": self._pressure_state.ram_percent,
                "ram_used_gb": self._pressure_state.ram_used_gb,
                "node_id": os.environ.get("RINGRIFT_NODE_ID", "local"),
            },
        )

    async def _handle_warning_tier(self) -> None:
        """Handle WARNING tier - throttle operations."""
        if not self._can_take_action():
            return

        logger.warning(
            f"[MemoryPressure] WARNING: RAM at {self._pressure_state.ram_percent:.1f}% - "
            "Throttling operations"
        )

        # Pause selfplay job spawning
        await self._pause_selfplay()

        # Emit event
        self._emit_event(
            "MEMORY_PRESSURE_WARNING",
            {
                "tier": "WARNING",
                "ram_percent": self._pressure_state.ram_percent,
                "action": "pause_selfplay",
                "node_id": os.environ.get("RINGRIFT_NODE_ID", "local"),
            },
        )

        self._record_action("pause_selfplay")

    async def _handle_critical_tier(self) -> None:
        """Handle CRITICAL tier - aggressive memory recovery."""
        if not self._can_take_action():
            return

        logger.error(
            f"[MemoryPressure] CRITICAL: RAM at {self._pressure_state.ram_percent:.1f}% - "
            "Taking aggressive action"
        )

        # Ensure selfplay is paused
        await self._pause_selfplay()

        # Force garbage collection
        gc.collect()

        # January 2026 - P2P Stability Plan Phase 3: Trigger emergency gossip cleanup
        # This purges gossip state with aggressive TTLs to free memory
        await self._trigger_gossip_emergency_cleanup()

        # Kill non-essential daemons
        await self._kill_non_essential_daemons()

        # Emit event
        self._emit_event(
            "MEMORY_PRESSURE_CRITICAL",
            {
                "tier": "CRITICAL",
                "ram_percent": self._pressure_state.ram_percent,
                "action": "kill_non_essential",
                "node_id": os.environ.get("RINGRIFT_NODE_ID", "local"),
            },
        )

        self._record_action("kill_non_essential")

    async def _trigger_gossip_emergency_cleanup(self) -> None:
        """Trigger emergency gossip state cleanup.

        January 2026 - P2P Stability Plan Phase 3:
        When memory pressure is CRITICAL (>80%), force aggressive cleanup of
        gossip state data structures to free memory.
        """
        try:
            # Try to get the gossip cleanup loop from P2P orchestrator
            from scripts.p2p.loops.gossip_state_cleanup_loop import GossipStateCleanupLoop

            # Check if there's a global/singleton instance we can access
            # Try via loop manager first
            try:
                from scripts.p2p.loops.base import LoopManager
                loop_manager = LoopManager.get_instance()
                if loop_manager:
                    cleanup_loop = loop_manager.get_loop("gossip_state_cleanup")
                    if cleanup_loop and hasattr(cleanup_loop, "force_emergency_cleanup"):
                        result = await cleanup_loop.force_emergency_cleanup()
                        logger.info(
                            f"[MemoryPressure] Gossip emergency cleanup freed {result.get('total', 0)} entries"
                        )
                        return
            except Exception as e:
                logger.debug(f"[MemoryPressure] Could not access loop manager: {e}")

            # Fallback: emit event for P2P orchestrator to handle
            self._emit_event(
                "GOSSIP_EMERGENCY_CLEANUP_REQUEST",
                {
                    "trigger": "memory_pressure_critical",
                    "ram_percent": self._pressure_state.ram_percent,
                    "node_id": os.environ.get("RINGRIFT_NODE_ID", "local"),
                },
            )
            logger.info("[MemoryPressure] Emitted GOSSIP_EMERGENCY_CLEANUP_REQUEST event")

        except ImportError:
            logger.debug("[MemoryPressure] GossipStateCleanupLoop not available")
        except Exception as e:
            logger.warning(f"[MemoryPressure] Gossip emergency cleanup failed: {e}")

    async def _handle_emergency_tier(self) -> None:
        """Handle EMERGENCY tier - prepare for failover."""
        logger.critical(
            f"[MemoryPressure] EMERGENCY: RAM at {self._pressure_state.ram_percent:.1f}% - "
            "Initiating graceful shutdown"
        )

        # Notify callbacks (e.g., StandbyCoordinator)
        for callback in self._on_emergency:
            try:
                callback()
            except Exception as e:
                logger.error(f"[MemoryPressure] Emergency callback failed: {e}")

        # Emit event
        self._emit_event(
            "MEMORY_PRESSURE_EMERGENCY",
            {
                "tier": "EMERGENCY",
                "ram_percent": self._pressure_state.ram_percent,
                "action": "graceful_shutdown",
                "node_id": os.environ.get("RINGRIFT_NODE_ID", "local"),
            },
        )

        self._record_action("emergency_shutdown_requested")

        # Final attempt at memory recovery
        gc.collect()

    async def _pause_selfplay(self) -> None:
        """Pause selfplay job spawning."""
        if self._selfplay_paused:
            return

        try:
            # Try to get selfplay scheduler and pause
            from app.coordination.selfplay_scheduler import get_selfplay_scheduler

            scheduler = get_selfplay_scheduler()
            if hasattr(scheduler, "pause_spawning"):
                scheduler.pause_spawning()
                self._selfplay_paused = True
                logger.info("[MemoryPressure] Selfplay spawning paused")
        except Exception as e:
            logger.warning(f"[MemoryPressure] Failed to pause selfplay: {e}")

    async def _resume_selfplay(self) -> None:
        """Resume selfplay job spawning."""
        if not self._selfplay_paused:
            return

        try:
            from app.coordination.selfplay_scheduler import get_selfplay_scheduler

            scheduler = get_selfplay_scheduler()
            if hasattr(scheduler, "resume_spawning"):
                scheduler.resume_spawning()
                self._selfplay_paused = False
                logger.info("[MemoryPressure] Selfplay spawning resumed")
        except Exception as e:
            logger.warning(f"[MemoryPressure] Failed to resume selfplay: {e}")

    async def _kill_non_essential_daemons(self) -> None:
        """Kill non-essential daemons to free memory."""
        if self._daemon_manager is None:
            try:
                from app.coordination.daemon_manager import get_daemon_manager

                self._daemon_manager = get_daemon_manager()
            except Exception:
                logger.warning("[MemoryPressure] DaemonManager not available")
                return

        # List of non-essential daemon types (can be killed under memory pressure)
        non_essential = [
            "METRICS_ANALYSIS",
            "TOURNAMENT",
            "ARCHIVE",
            "ORPHAN_DETECTION",
            "STALE_EVALUATION",
        ]

        for daemon_type in non_essential:
            try:
                await self._daemon_manager.stop(daemon_type)
                logger.info(f"[MemoryPressure] Stopped daemon: {daemon_type}")
            except Exception as e:
                logger.debug(f"[MemoryPressure] Could not stop {daemon_type}: {e}")

    async def _handle_tier_change(
        self, old_tier: MemoryPressureTier, new_tier: MemoryPressureTier
    ) -> None:
        """Handle transition between tiers."""
        logger.info(
            f"[MemoryPressure] Tier change: {old_tier.name} -> {new_tier.name} "
            f"(RAM: {self._pressure_state.ram_percent:.1f}%)"
        )

        # Call tier-specific handlers
        if new_tier == MemoryPressureTier.CAUTION:
            await self._handle_caution_tier()
        elif new_tier == MemoryPressureTier.WARNING:
            await self._handle_warning_tier()
        elif new_tier == MemoryPressureTier.CRITICAL:
            await self._handle_critical_tier()
        elif new_tier == MemoryPressureTier.EMERGENCY:
            await self._handle_emergency_tier()
        elif new_tier == MemoryPressureTier.NORMAL and old_tier >= MemoryPressureTier.WARNING:
            # Recovering - resume operations
            await self._resume_selfplay()
            self._emit_event(
                "MEMORY_PRESSURE_RECOVERED",
                {
                    "tier": "NORMAL",
                    "ram_percent": self._pressure_state.ram_percent,
                    "node_id": os.environ.get("RINGRIFT_NODE_ID", "local"),
                },
            )

        # Notify callbacks
        for callback in self._on_tier_change:
            try:
                callback(new_tier)
            except Exception as e:
                logger.error(f"[MemoryPressure] Tier change callback failed: {e}")

    async def _on_start(self) -> None:
        """Hook called before main loop starts."""
        logger.info(
            f"[MemoryPressure] Started monitoring "
            f"(check_interval={self._cycle_interval}s)"
        )

    async def _run_cycle(self) -> None:
        """One iteration of memory pressure monitoring.

        Called by HandlerBase._main_loop every cycle_interval seconds.
        """
        # Get current memory usage
        ram_percent, ram_used_gb, ram_total_gb, swap_percent = (
            self._get_memory_usage()
        )

        # Update state
        self._pressure_state.ram_percent = ram_percent
        self._pressure_state.ram_used_gb = ram_used_gb
        self._pressure_state.ram_total_gb = ram_total_gb
        self._pressure_state.swap_percent = swap_percent
        self._pressure_state.timestamp = time.time()

        # Determine new tier
        new_tier = self._determine_tier(ram_percent)
        old_tier = self._pressure_state.tier

        # Check if tier changed (with hysteresis for downgrades)
        if new_tier != old_tier:
            if new_tier > old_tier:
                # Escalating - increment consecutive samples
                self._pressure_state.consecutive_samples += 1
                if self._pressure_state.consecutive_samples >= self._pressure_config.CONSECUTIVE_SAMPLES_REQUIRED:
                    self._pressure_state.tier = new_tier
                    self._pressure_state.consecutive_samples = 0
                    await self._handle_tier_change(old_tier, new_tier)
            elif self._should_downgrade_tier(new_tier):
                # De-escalating with hysteresis
                self._pressure_state.tier = new_tier
                self._pressure_state.consecutive_samples = 0
                await self._handle_tier_change(old_tier, new_tier)
        else:
            # Same tier, reset consecutive counter
            self._pressure_state.consecutive_samples = 0

        # Periodic handling for sustained high pressure
        if self._pressure_state.tier >= MemoryPressureTier.WARNING:
            if self._pressure_state.tier == MemoryPressureTier.WARNING:
                await self._handle_warning_tier()
            elif self._pressure_state.tier == MemoryPressureTier.CRITICAL:
                await self._handle_critical_tier()

    async def _on_stop(self) -> None:
        """Hook called after main loop stops."""
        logger.info("[MemoryPressure] Controller stopped")

    def register_tier_change_callback(
        self, callback: Callable[[MemoryPressureTier], None]
    ) -> None:
        """Register callback for tier changes."""
        self._on_tier_change.append(callback)

    def register_emergency_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for emergency tier."""
        self._on_emergency.append(callback)

    def health_check(self) -> HealthCheckResult:
        """Return health check result for daemon manager.

        Overrides HandlerBase.health_check() with memory-specific health logic.
        Returns unhealthy if tier is CRITICAL or above.
        """
        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="Memory pressure controller not running",
            )

        tier = self._pressure_state.tier

        if tier >= MemoryPressureTier.CRITICAL:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Memory pressure CRITICAL: {self._pressure_state.ram_percent:.1f}% RAM",
                details=self._pressure_state.to_dict(),
            )
        elif tier >= MemoryPressureTier.WARNING:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message=f"Memory pressure WARNING: {self._pressure_state.ram_percent:.1f}% RAM",
                details=self._pressure_state.to_dict(),
            )
        else:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message=f"Memory pressure normal: {self._pressure_state.ram_percent:.1f}% RAM",
                details=self._pressure_state.to_dict(),
            )


# Convenience accessor
def get_memory_pressure_controller() -> MemoryPressureController:
    """Get the singleton memory pressure controller."""
    return MemoryPressureController.get_instance()
