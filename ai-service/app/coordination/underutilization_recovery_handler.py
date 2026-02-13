"""Underutilization Recovery Handler.

Jan 4, 2026 - Phase 3 of P2P Cluster Resilience.

Problem: CLUSTER_UNDERUTILIZED events are emitted but only logged - no work injection occurs.
When 60%+ of GPU nodes are idle, training progress stalls.

Solution: This handler subscribes to underutilization events and injects high-priority
work items for underserved configurations. This keeps GPU utilization high even
during leader election failures or network partitions.

Events Subscribed:
- CLUSTER_UNDERUTILIZED: >50% of GPU nodes idle for >5 minutes
- WORK_QUEUE_EXHAUSTED: Work queue completely empty

Events Emitted:
- UTILIZATION_RECOVERY_STARTED: Recovery injection started
- UTILIZATION_RECOVERY_COMPLETED: Recovery injection finished
- UTILIZATION_RECOVERY_FAILED: Recovery injection failed

Usage:
    from app.coordination.underutilization_recovery_handler import (
        UnderutilizationRecoveryHandler,
        get_underutilization_handler,
    )

    handler = get_underutilization_handler()
    await handler.start()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.coordination.event_router import get_event_payload
from app.coordination.handler_base import HandlerBase

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Configuration defaults
DEFAULT_INJECTION_BATCH_SIZE = 20
DEFAULT_RECOVERY_COOLDOWN_SECONDS = 300.0  # 5 minutes between recovery attempts
DEFAULT_HIGH_PRIORITY = 100  # Priority for injected work items
DEFAULT_MIN_IDLE_PERCENT = 0.5  # Trigger at 50% idle
DEFAULT_CHECK_INTERVAL_SECONDS = 60.0


@dataclass
class UnderutilizationConfig:
    """Configuration for underutilization recovery.

    Attributes:
        injection_batch_size: How many work items to inject per recovery
        recovery_cooldown_seconds: Minimum time between recovery attempts
        high_priority: Priority level for injected work items
        min_idle_percent: Trigger threshold (0.0-1.0)
        check_interval_seconds: How often to check utilization
        enabled: Whether this handler is active
    """

    injection_batch_size: int = DEFAULT_INJECTION_BATCH_SIZE
    recovery_cooldown_seconds: float = DEFAULT_RECOVERY_COOLDOWN_SECONDS
    high_priority: int = DEFAULT_HIGH_PRIORITY
    min_idle_percent: float = DEFAULT_MIN_IDLE_PERCENT
    check_interval_seconds: float = DEFAULT_CHECK_INTERVAL_SECONDS
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "UnderutilizationConfig":
        """Create config from environment variables."""
        import os

        return cls(
            injection_batch_size=int(
                os.environ.get("RINGRIFT_UNDERUTIL_BATCH_SIZE", DEFAULT_INJECTION_BATCH_SIZE)
            ),
            recovery_cooldown_seconds=float(
                os.environ.get("RINGRIFT_UNDERUTIL_COOLDOWN", DEFAULT_RECOVERY_COOLDOWN_SECONDS)
            ),
            high_priority=int(
                os.environ.get("RINGRIFT_UNDERUTIL_PRIORITY", DEFAULT_HIGH_PRIORITY)
            ),
            min_idle_percent=float(
                os.environ.get("RINGRIFT_UNDERUTIL_IDLE_THRESHOLD", DEFAULT_MIN_IDLE_PERCENT)
            ),
            check_interval_seconds=float(
                os.environ.get("RINGRIFT_UNDERUTIL_CHECK_INTERVAL", DEFAULT_CHECK_INTERVAL_SECONDS)
            ),
            enabled=os.environ.get("RINGRIFT_UNDERUTIL_ENABLED", "true").lower() == "true",
        )


@dataclass
class RecoveryStats:
    """Statistics for recovery operations."""

    total_recoveries: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    items_injected: int = 0
    last_recovery_time: float = 0.0
    last_recovery_reason: str = ""
    configs_targeted: dict[str, int] = field(default_factory=dict)


class UnderutilizationRecoveryHandler(HandlerBase):
    """Handles cluster underutilization by injecting work queue items.

    This handler monitors for underutilization events and proactively
    injects work items to keep GPU nodes busy.

    Features:
    - Subscribes to CLUSTER_UNDERUTILIZED and WORK_QUEUE_EXHAUSTED events
    - Gets underserved configs from selfplay scheduler
    - Injects high-priority work items for underserved configs
    - Respects cooldown between recovery attempts
    - Tracks recovery statistics
    """

    # Singleton instance
    _instance: "UnderutilizationRecoveryHandler | None" = None

    def __init__(
        self,
        config: UnderutilizationConfig | None = None,
        work_queue: Any = None,
        selfplay_scheduler: Any = None,
    ) -> None:
        """Initialize the handler.

        Args:
            config: Handler configuration
            work_queue: Work queue instance (for injecting items)
            selfplay_scheduler: SelfplayScheduler instance (for config priorities)
        """
        resolved_config = config or UnderutilizationConfig.from_env()
        super().__init__(
            name="underutilization_recovery",
            config=resolved_config,  # Pass to HandlerBase
            cycle_interval=resolved_config.check_interval_seconds,
        )

        self._work_queue = work_queue
        self._selfplay_scheduler = selfplay_scheduler
        self._stats = RecoveryStats()
        self._last_recovery_attempt = 0.0
        self._recovery_in_progress = False
        self._pending_recovery_event: dict[str, Any] | None = None

    @classmethod
    def get_instance(
        cls,
        config: UnderutilizationConfig | None = None,
        work_queue: Any = None,
        selfplay_scheduler: Any = None,
    ) -> "UnderutilizationRecoveryHandler":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(
                config=config,
                work_queue=work_queue,
                selfplay_scheduler=selfplay_scheduler,
            )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    def set_work_queue(self, work_queue: Any) -> None:
        """Set the work queue reference (for late binding)."""
        self._work_queue = work_queue

    def set_selfplay_scheduler(self, scheduler: Any) -> None:
        """Set the selfplay scheduler reference (for late binding)."""
        self._selfplay_scheduler = scheduler

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Get event subscriptions for this handler."""
        return {
            "CLUSTER_UNDERUTILIZED": self._on_cluster_underutilized,
            "WORK_QUEUE_EXHAUSTED": self._on_work_queue_exhausted,
        }

    async def _on_cluster_underutilized(self, event: dict[str, Any]) -> None:
        """Handle CLUSTER_UNDERUTILIZED event.

        Args:
            event: Event payload with idle_percent, idle_nodes, etc.
        """
        if not self._config.enabled:
            return

        # Feb 2026: Extract payload from RouterEvent (was crashing with AttributeError)
        payload = get_event_payload(event)
        idle_percent = payload.get("idle_percent", 0.0)
        idle_nodes = payload.get("idle_nodes", [])

        logger.warning(
            f"[UnderutilizationRecovery] Cluster underutilized: "
            f"{idle_percent:.1%} idle ({len(idle_nodes)} nodes)"
        )

        # Queue recovery if cooldown has passed
        if self._can_attempt_recovery():
            self._pending_recovery_event = {
                "reason": "cluster_underutilized",
                "idle_percent": idle_percent,
                "idle_nodes": idle_nodes,
                "timestamp": time.time(),
            }

    async def _on_work_queue_exhausted(self, event: dict[str, Any]) -> None:
        """Handle WORK_QUEUE_EXHAUSTED event.

        Args:
            event: Event payload with queue depth, last_item_time, etc.
        """
        if not self._config.enabled:
            return

        logger.warning("[UnderutilizationRecovery] Work queue exhausted!")

        # Queue recovery if cooldown has passed
        if self._can_attempt_recovery():
            # Feb 2026: Extract payload from RouterEvent
            wq_payload = get_event_payload(event)
            self._pending_recovery_event = {
                "reason": "work_queue_exhausted",
                "queue_depth": wq_payload.get("queue_depth", 0),
                "timestamp": time.time(),
            }

    def _can_attempt_recovery(self) -> bool:
        """Check if we can attempt a recovery (respects cooldown)."""
        if self._recovery_in_progress:
            return False

        now = time.time()
        elapsed = now - self._last_recovery_attempt
        return elapsed >= self._config.recovery_cooldown_seconds

    async def _run_cycle(self) -> None:
        """Main cycle - check for pending recovery and execute."""
        if not self._config.enabled:
            return

        if self._pending_recovery_event and self._can_attempt_recovery():
            event = self._pending_recovery_event
            self._pending_recovery_event = None
            await self._execute_recovery(event)

    async def _execute_recovery(self, event: dict[str, Any]) -> None:
        """Execute work injection recovery.

        Args:
            event: Recovery trigger event with reason and context
        """
        if self._recovery_in_progress:
            return

        self._recovery_in_progress = True
        self._last_recovery_attempt = time.time()
        self._stats.total_recoveries += 1

        reason = event.get("reason", "unknown")
        self._stats.last_recovery_reason = reason

        logger.info(
            f"[UnderutilizationRecovery] Starting recovery: {reason} "
            f"(attempt #{self._stats.total_recoveries})"
        )

        try:
            # Emit start event
            self._emit_recovery_started(event)

            # Get underserved configs
            configs = await self._get_underserved_configs()
            if not configs:
                logger.warning("[UnderutilizationRecovery] No underserved configs found")
                self._stats.failed_recoveries += 1
                self._recovery_in_progress = False
                return

            # Inject work items
            items_injected = await self._inject_work_items(configs)

            if items_injected > 0:
                self._stats.successful_recoveries += 1
                self._stats.items_injected += items_injected
                self._stats.last_recovery_time = time.time()

                logger.info(
                    f"[UnderutilizationRecovery] Injected {items_injected} work items "
                    f"for {len(configs)} configs"
                )

                # Emit completion event
                self._emit_recovery_completed(event, items_injected, configs)
            else:
                self._stats.failed_recoveries += 1
                logger.warning("[UnderutilizationRecovery] Failed to inject any work items")
                self._emit_recovery_failed(event, "no_items_injected")

        except Exception as e:
            self._stats.failed_recoveries += 1
            logger.error(f"[UnderutilizationRecovery] Recovery failed: {e}")
            self._emit_recovery_failed(event, str(e))
        finally:
            self._recovery_in_progress = False

    async def _get_underserved_configs(self) -> list[str]:
        """Get list of underserved configurations.

        Returns:
            List of config keys that need more selfplay games
        """
        configs: list[str] = []

        # Try selfplay scheduler first
        if self._selfplay_scheduler:
            try:
                if hasattr(self._selfplay_scheduler, "get_underserved_configs"):
                    configs = self._selfplay_scheduler.get_underserved_configs(
                        limit=self._config.injection_batch_size // 4
                    )
                elif hasattr(self._selfplay_scheduler, "get_config_priorities"):
                    # Get top priorities
                    priorities = self._selfplay_scheduler.get_config_priorities()
                    if priorities:
                        sorted_configs = sorted(
                            priorities.keys(),
                            key=lambda k: priorities[k],
                            reverse=True,
                        )
                        configs = sorted_configs[:self._config.injection_batch_size // 4]
            except Exception as e:
                logger.debug(f"[UnderutilizationRecovery] Scheduler query failed: {e}")

        # Fallback: all canonical configs with rotation
        if not configs:
            all_configs = [
                "hex8_2p", "hex8_3p", "hex8_4p",
                "square8_2p", "square8_3p", "square8_4p",
                "square19_2p", "square19_3p", "square19_4p",
                "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
            ]
            # Rotate based on recovery count for diversity
            offset = self._stats.total_recoveries % len(all_configs)
            configs = all_configs[offset:] + all_configs[:offset]
            configs = configs[:self._config.injection_batch_size // 4]

        return configs

    async def _inject_work_items(self, configs: list[str]) -> int:
        """Inject work items for the given configs.

        Args:
            configs: List of config keys to inject work for

        Returns:
            Number of items successfully injected
        """
        if not self._work_queue:
            logger.warning("[UnderutilizationRecovery] No work queue available")
            return 0

        items_injected = 0
        items_per_config = max(1, self._config.injection_batch_size // len(configs))

        for config_key in configs:
            try:
                for i in range(items_per_config):
                    work_item = self._create_work_item(config_key, i)
                    if work_item:
                        if hasattr(self._work_queue, "push"):
                            await self._work_queue.push(work_item)
                        elif hasattr(self._work_queue, "put"):
                            await self._work_queue.put(work_item)
                        elif hasattr(self._work_queue, "add_item"):
                            self._work_queue.add_item(work_item)
                        else:
                            # Try dict-like interface
                            work_id = work_item.get("work_id", f"recovery-{time.time()}")
                            self._work_queue[work_id] = work_item

                        items_injected += 1

                        # Track config targeting
                        self._stats.configs_targeted[config_key] = (
                            self._stats.configs_targeted.get(config_key, 0) + 1
                        )

            except Exception as e:
                logger.debug(f"[UnderutilizationRecovery] Failed to inject for {config_key}: {e}")

        return items_injected

    def _create_work_item(self, config_key: str, index: int) -> dict[str, Any]:
        """Create a work item for injection.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            index: Item index within batch

        Returns:
            Work item dictionary
        """
        import uuid

        parts = config_key.rsplit("_", 1)
        board_type = parts[0]
        num_players = int(parts[1].rstrip("p")) if len(parts) > 1 else 2

        return {
            "work_id": f"recovery-{uuid.uuid4().hex[:8]}-{index}",
            "work_type": "selfplay",
            "config_key": config_key,
            "board_type": board_type,
            "num_players": num_players,
            "source": "underutilization_recovery",
            "priority": self._config.high_priority,
            "created_at": time.time(),
            "games_requested": 10,  # Small batch for quick turnaround
            "recovery_attempt": self._stats.total_recoveries,
        }

    def _emit_recovery_started(self, event: dict[str, Any]) -> None:
        """Emit UTILIZATION_RECOVERY_STARTED event."""
        from app.coordination.event_emission_helpers import safe_emit_event
        from app.distributed.data_events import DataEventType

        safe_emit_event(
            DataEventType.UTILIZATION_RECOVERY_STARTED,
            {
                "reason": event.get("reason"),
                "trigger_event": event,
                "recovery_attempt": self._stats.total_recoveries,
                "timestamp": time.time(),
            },
            context="UnderutilizationRecovery",
            source="underutilization_handler",
        )

    def _emit_recovery_completed(
        self,
        event: dict[str, Any],
        items_injected: int,
        configs: list[str],
    ) -> None:
        """Emit UTILIZATION_RECOVERY_COMPLETED event."""
        from app.coordination.event_emission_helpers import safe_emit_event
        from app.distributed.data_events import DataEventType

        safe_emit_event(
            DataEventType.UTILIZATION_RECOVERY_COMPLETED,
            {
                "reason": event.get("reason"),
                "items_injected": items_injected,
                "configs_targeted": configs,
                "recovery_attempt": self._stats.total_recoveries,
                "duration_seconds": time.time() - self._last_recovery_attempt,
                "timestamp": time.time(),
            },
            context="UnderutilizationRecovery",
            source="underutilization_handler",
        )

    def _emit_recovery_failed(self, event: dict[str, Any], error: str) -> None:
        """Emit UTILIZATION_RECOVERY_FAILED event."""
        from app.coordination.event_emission_helpers import safe_emit_event
        from app.distributed.data_events import DataEventType

        safe_emit_event(
            DataEventType.UTILIZATION_RECOVERY_FAILED,
            {
                "reason": event.get("reason"),
                "error": error,
                "recovery_attempt": self._stats.total_recoveries,
                "timestamp": time.time(),
            },
            context="UnderutilizationRecovery",
            source="underutilization_handler",
        )

    def get_stats(self) -> dict[str, Any]:
        """Get recovery statistics."""
        return {
            "enabled": self._config.enabled,
            "total_recoveries": self._stats.total_recoveries,
            "successful_recoveries": self._stats.successful_recoveries,
            "failed_recoveries": self._stats.failed_recoveries,
            "items_injected": self._stats.items_injected,
            "last_recovery_time": self._stats.last_recovery_time,
            "last_recovery_reason": self._stats.last_recovery_reason,
            "configs_targeted": dict(self._stats.configs_targeted),
            "recovery_in_progress": self._recovery_in_progress,
            "cooldown_remaining": max(
                0,
                self._config.recovery_cooldown_seconds - (time.time() - self._last_recovery_attempt),
            ),
        }

    def health_check(self) -> dict[str, Any]:
        """Return health check result."""
        return {
            "healthy": self._config.enabled and not self._recovery_in_progress,
            "details": self.get_stats(),
        }


# Singleton accessor
def get_underutilization_handler(
    config: UnderutilizationConfig | None = None,
    work_queue: Any = None,
    selfplay_scheduler: Any = None,
) -> UnderutilizationRecoveryHandler:
    """Get the singleton underutilization recovery handler."""
    return UnderutilizationRecoveryHandler.get_instance(
        config=config,
        work_queue=work_queue,
        selfplay_scheduler=selfplay_scheduler,
    )
