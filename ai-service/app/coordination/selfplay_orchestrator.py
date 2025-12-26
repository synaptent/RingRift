"""SelfplayOrchestrator - Unified selfplay event coordination (December 2025).

This module provides centralized monitoring and coordination of selfplay events.
It tracks selfplay tasks across nodes, emits completion events, and coordinates
between background selfplay and the broader pipeline.

Event Integration:
- Subscribes to TASK_SPAWNED: Track new selfplay tasks
- Subscribes to TASK_COMPLETED: Track selfplay completions
- Subscribes to TASK_FAILED: Track selfplay failures
- Emits SELFPLAY_COMPLETE: When canonical selfplay finishes
- Emits GPU_SELFPLAY_COMPLETE: When GPU-accelerated selfplay finishes

Key Responsibilities:
1. Track active selfplay tasks across all nodes
2. Emit completion events to trigger downstream pipeline stages
3. Coordinate background selfplay with main pipeline
4. Provide selfplay statistics and throughput metrics

Usage:
    from app.coordination.selfplay_orchestrator import (
        SelfplayOrchestrator,
        wire_selfplay_events,
        get_selfplay_orchestrator,
        emit_selfplay_completion,
    )

    # Wire selfplay events
    orchestrator = wire_selfplay_events()

    # Get selfplay statistics
    stats = orchestrator.get_stats()
    print(f"Active selfplay tasks: {stats['active_tasks']}")
    print(f"Games generated: {stats['total_games_generated']}")

    # Emit completion event (call when selfplay finishes)
    await emit_selfplay_completion(
        task_id="selfplay-123",
        board_type="square8",
        num_players=2,
        games_generated=500,
    )
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Use centralized event emitters (December 2025)
# Note: event_emitters.py handles all routing to data_events, stage_events, and cross-process
from app.coordination.event_emitters import emit_selfplay_complete as _emit_selfplay_event


class SelfplayType(Enum):
    """Types of selfplay tasks."""

    CANONICAL = "canonical"  # Standard MCTS selfplay
    GPU_ACCELERATED = "gpu_selfplay"  # GPU-accelerated selfplay
    HYBRID = "hybrid_selfplay"  # Mixed CPU/GPU selfplay
    BACKGROUND = "background"  # Background pipeline selfplay


@dataclass
class SelfplayTaskInfo:
    """Information about a selfplay task."""

    task_id: str
    selfplay_type: SelfplayType
    node_id: str
    board_type: str = "square8"
    num_players: int = 2
    games_requested: int = 0
    games_generated: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    success: bool = False
    error: str = ""
    iteration: int = 0

    @property
    def duration(self) -> float:
        """Get task duration in seconds."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    @property
    def games_per_second(self) -> float:
        """Get games per second throughput."""
        duration = self.duration
        if duration > 0 and self.games_generated > 0:
            return self.games_generated / duration
        return 0.0


@dataclass
class SelfplayStats:
    """Aggregate selfplay statistics."""

    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_games_generated: int = 0
    total_duration_seconds: float = 0.0
    average_games_per_task: float = 0.0
    average_games_per_second: float = 0.0
    by_node: dict[str, int] = field(default_factory=dict)
    by_type: dict[str, int] = field(default_factory=dict)
    last_activity_time: float = 0.0


class SelfplayOrchestrator:
    """Orchestrates selfplay coordination across the cluster.

    Tracks selfplay tasks, emits completion events, and provides
    unified monitoring of selfplay activity.
    """

    def __init__(
        self,
        max_history: int = 500,
        stats_window_seconds: float = 3600.0,  # 1 hour
    ):
        """Initialize SelfplayOrchestrator.

        Args:
            max_history: Maximum number of completed tasks to retain
            stats_window_seconds: Time window for throughput calculations
        """
        self.max_history = max_history
        self.stats_window_seconds = stats_window_seconds

        # Active tasks by task_id
        self._active_tasks: dict[str, SelfplayTaskInfo] = {}

        # Completed task history
        self._completed_history: list[SelfplayTaskInfo] = []

        # Statistics
        self._total_games_generated: int = 0
        self._total_duration_seconds: float = 0.0

        # Subscription state
        self._subscribed: bool = False

        # Callbacks for completion events
        self._completion_callbacks: list[Callable[[SelfplayTaskInfo], None]] = []

        # Backpressure tracking (December 2025)
        self._backpressure_nodes: dict[str, str] = {}  # node_id -> backpressure level
        self._paused_for_regression: bool = False
        self._resource_constrained_nodes: dict[str, float] = {}  # node_id -> timestamp

        # Curriculum weights (December 2025 - Phase 1 feedback loop)
        # Maps config_key (e.g., "square8_2p") to weight (0.5 to 2.0)
        # Higher weight = more selfplay jobs allocated to this config
        self._curriculum_weights: dict[str, float] = {}
        self._curriculum_weights_updated_at: float = 0.0

        # Quality-based budget multipliers (December 2025 - closes quality feedback gap)
        # Maps config_key to budget multiplier (0.8 to 1.5)
        # Low quality → higher multiplier (more exploration needed)
        # High quality → lower multiplier (model already performing well)
        self._quality_budget_multipliers: dict[str, float] = {}
        self._quality_scores: dict[str, float] = {}  # Track current quality scores

    def subscribe_to_events(self) -> bool:
        """Subscribe to selfplay-related events from the event router.

        Returns:
            True if successfully subscribed
        """
        if self._subscribed:
            return True

        try:
            from app.coordination.event_router import get_router
            from app.coordination.event_router import DataEventType  # Types still needed

            router = get_router()

            # Subscribe to task lifecycle events
            router.subscribe(DataEventType.TASK_SPAWNED.value, self._on_task_spawned)
            router.subscribe(DataEventType.TASK_COMPLETED.value, self._on_task_completed)
            router.subscribe(DataEventType.TASK_FAILED.value, self._on_task_failed)

            # Subscribe to task cleanup events (December 2025)
            router.subscribe(DataEventType.TASK_ORPHANED.value, self._on_task_orphaned)
            router.subscribe(DataEventType.TASK_ABANDONED.value, self._on_task_abandoned)

            # Subscribe to resource/backpressure events (December 2025)
            router.subscribe(DataEventType.BACKPRESSURE_ACTIVATED.value, self._on_backpressure_activated)
            router.subscribe(DataEventType.BACKPRESSURE_RELEASED.value, self._on_backpressure_released)
            router.subscribe(DataEventType.RESOURCE_CONSTRAINT.value, self._on_resource_constraint)
            router.subscribe(DataEventType.REGRESSION_DETECTED.value, self._on_regression_detected)

            # Subscribe to curriculum events (December 2025 - Phase 1 feedback loop)
            router.subscribe(DataEventType.CURRICULUM_REBALANCED.value, self._on_curriculum_rebalanced)

            # Subscribe to quality events (December 2025 - closes quality → selfplay gap)
            router.subscribe(DataEventType.QUALITY_SCORE_UPDATED.value, self._on_quality_updated)

            # Subscribe to idle resource detection (December 2025 - spawn selfplay on idle GPUs)
            router.subscribe(DataEventType.IDLE_RESOURCE_DETECTED.value, self._on_idle_resource_detected)

            self._subscribed = True
            logger.info("[SelfplayOrchestrator] Subscribed to task lifecycle and resource events via event router")
            return True

        except ImportError:
            logger.warning("[SelfplayOrchestrator] data_events not available")
            return False
        except Exception as e:
            logger.error(f"[SelfplayOrchestrator] Failed to subscribe: {e}")
            return False

    def subscribe_to_stage_events(self) -> bool:
        """Subscribe to stage completion events.

        Returns:
            True if successfully subscribed
        """
        try:
            from app.coordination.event_router import StageEvent, get_stage_event_bus

            bus = get_stage_event_bus()

            # Subscribe to selfplay stage events
            bus.subscribe(StageEvent.SELFPLAY_COMPLETE, self._on_selfplay_stage_complete)
            bus.subscribe(
                StageEvent.GPU_SELFPLAY_COMPLETE, self._on_gpu_selfplay_stage_complete
            )
            bus.subscribe(
                StageEvent.CANONICAL_SELFPLAY_COMPLETE,
                self._on_canonical_selfplay_stage_complete,
            )

            logger.info("[SelfplayOrchestrator] Subscribed to stage events")
            return True

        except ImportError:
            logger.warning("[SelfplayOrchestrator] stage_events not available")
            return False
        except Exception as e:
            logger.error(f"[SelfplayOrchestrator] Failed to subscribe to stages: {e}")
            return False

    def _is_selfplay_task(self, task_type: str) -> bool:
        """Check if task type is a selfplay task."""
        selfplay_types = {
            "selfplay",
            "gpu_selfplay",
            "hybrid_selfplay",
            "canonical_selfplay",
            "background_selfplay",
        }
        return task_type.lower() in selfplay_types

    def _get_selfplay_type(self, task_type: str) -> SelfplayType:
        """Convert task type string to SelfplayType enum."""
        mapping = {
            "selfplay": SelfplayType.CANONICAL,
            "canonical_selfplay": SelfplayType.CANONICAL,
            "gpu_selfplay": SelfplayType.GPU_ACCELERATED,
            "hybrid_selfplay": SelfplayType.HYBRID,
            "background_selfplay": SelfplayType.BACKGROUND,
        }
        return mapping.get(task_type.lower(), SelfplayType.CANONICAL)

    async def _on_task_spawned(self, event) -> None:
        """Handle TASK_SPAWNED event."""
        payload = event.payload
        task_type = payload.get("task_type", "")

        if not self._is_selfplay_task(task_type):
            return

        task_id = payload.get("task_id", "")
        node_id = payload.get("node_id", "")

        task = SelfplayTaskInfo(
            task_id=task_id,
            selfplay_type=self._get_selfplay_type(task_type),
            node_id=node_id,
            board_type=payload.get("board_type", "square8"),
            num_players=payload.get("num_players", 2),
            games_requested=payload.get("games_requested", 0),
            iteration=payload.get("iteration", 0),
        )

        self._active_tasks[task_id] = task
        logger.debug(
            f"[SelfplayOrchestrator] Tracking selfplay task {task_id} on {node_id}"
        )

    async def _on_task_completed(self, event) -> None:
        """Handle TASK_COMPLETED event."""
        payload = event.payload
        task_id = payload.get("task_id", "")

        if task_id not in self._active_tasks:
            # Check if it's a selfplay task we missed
            task_type = payload.get("task_type", "")
            if self._is_selfplay_task(task_type):
                # Create a task info from completion data
                task = SelfplayTaskInfo(
                    task_id=task_id,
                    selfplay_type=self._get_selfplay_type(task_type),
                    node_id=payload.get("node_id", ""),
                    games_generated=payload.get("games_generated", 0),
                    success=True,
                )
                self._record_completion(task, payload)
            return

        task = self._active_tasks.pop(task_id)
        task.success = True
        task.end_time = time.time()
        task.games_generated = payload.get(
            "games_generated", payload.get("result", {}).get("games_generated", 0)
        )

        self._record_completion(task, payload)

        # Emit stage completion event
        await self._emit_selfplay_complete(task)

    async def _on_task_failed(self, event) -> None:
        """Handle TASK_FAILED event."""
        payload = event.payload
        task_id = payload.get("task_id", "")

        if task_id not in self._active_tasks:
            return

        task = self._active_tasks.pop(task_id)
        task.success = False
        task.end_time = time.time()
        task.error = payload.get("error", "Unknown error")

        self._record_completion(task, payload)

        logger.warning(
            f"[SelfplayOrchestrator] Selfplay task {task_id} failed: {task.error}"
        )

    async def _on_task_orphaned(self, event) -> None:
        """Handle TASK_ORPHANED event - cleanup orphaned selfplay tasks.

        This event is received when a task hasn't sent a heartbeat for too long,
        indicating the worker may have crashed or become unresponsive.
        """
        payload = event.payload
        task_id = payload.get("task_id", "")
        task_type = payload.get("task_type", "")

        # Only handle selfplay tasks
        if not self._is_selfplay_task(task_type):
            return

        if task_id not in self._active_tasks:
            return

        task = self._active_tasks.pop(task_id)
        task.success = False
        task.end_time = time.time()
        task.error = f"Task orphaned: {payload.get('reason', 'no heartbeat')}"

        # Record as failed for statistics
        self._record_completion(task, payload)

        logger.warning(
            f"[SelfplayOrchestrator] Selfplay task orphaned: {task_id} on {task.node_id} - "
            f"cleaning up (partial games={task.games_generated})"
        )

    async def _on_task_abandoned(self, event) -> None:
        """Handle TASK_ABANDONED event - cleanup intentionally abandoned tasks.

        This event is received when a task is explicitly abandoned (e.g., due to
        backpressure, resource constraints, or pipeline requirements).
        """
        payload = event.payload
        task_id = payload.get("task_id", "")
        task_type = payload.get("task_type", "")

        # Only handle selfplay tasks
        if not self._is_selfplay_task(task_type):
            return

        if task_id not in self._active_tasks:
            return

        task = self._active_tasks.pop(task_id)
        task.success = False
        task.end_time = time.time()
        task.error = f"Task abandoned: {payload.get('reason', 'unknown')}"

        # Record as failed for statistics
        self._record_completion(task, payload)

        # If games were generated before abandonment, they may still be usable
        if task.games_generated > 0:
            logger.info(
                "[SelfplayOrchestrator] Selfplay task abandoned with partial results: "
                f"{task_id}, games={task.games_generated}"
            )
        else:
            logger.warning(
                f"[SelfplayOrchestrator] Selfplay task abandoned: {task_id} - "
                f"reason: {payload.get('reason', 'unknown')}"
            )

    def _record_completion(self, task: SelfplayTaskInfo, payload: dict) -> None:
        """Record task completion in history and stats."""
        # Update statistics
        if task.success:
            self._total_games_generated += task.games_generated
            self._total_duration_seconds += task.duration

        # Add to history
        self._completed_history.append(task)

        # Trim history
        if len(self._completed_history) > self.max_history:
            self._completed_history = self._completed_history[-self.max_history :]

        # Notify callbacks
        for callback in self._completion_callbacks:
            try:
                callback(task)
            except Exception as e:
                logger.error(f"[SelfplayOrchestrator] Callback error: {e}")

        logger.info(
            f"[SelfplayOrchestrator] Selfplay {'completed' if task.success else 'failed'}: "
            f"{task.task_id} on {task.node_id}, games={task.games_generated}, "
            f"duration={task.duration:.1f}s"
        )

    async def _emit_selfplay_complete(self, task: SelfplayTaskInfo) -> bool:
        """Emit SELFPLAY_COMPLETE event using centralized emitter.

        Note: event_emitters.py handles routing to all event systems
        (data_events, stage_events, cross-process) internally.
        """
        try:
            # Map selfplay type to string for centralized emitter
            selfplay_type_str = {
                SelfplayType.GPU_ACCELERATED: "gpu_accelerated",
                SelfplayType.CANONICAL: "canonical",
                SelfplayType.HYBRID: "hybrid",
                SelfplayType.BACKGROUND: "background",
            }.get(task.selfplay_type, "standard")

            result = await _emit_selfplay_event(
                task_id=task.task_id,
                board_type=task.board_type,
                num_players=task.num_players,
                games_generated=task.games_generated,
                success=task.success,
                node_id=task.node_id,
                duration_seconds=task.duration,
                selfplay_type=selfplay_type_str,
                iteration=task.iteration,
                error=task.error if not task.success else None,
                games_per_second=task.games_per_second,
            )
            if result:
                logger.debug("[SelfplayOrchestrator] Emitted selfplay complete event")
            return result

        except Exception as e:
            logger.warning(f"[SelfplayOrchestrator] Failed to emit selfplay event: {e}")
            return False

    async def _on_selfplay_stage_complete(self, result) -> None:
        """Handle SELFPLAY_COMPLETE stage event."""
        logger.debug(
            "[SelfplayOrchestrator] Stage event: SELFPLAY_COMPLETE "
            f"success={result.success}, games={result.games_generated}"
        )

    async def _on_gpu_selfplay_stage_complete(self, result) -> None:
        """Handle GPU_SELFPLAY_COMPLETE stage event."""
        logger.debug(
            "[SelfplayOrchestrator] Stage event: GPU_SELFPLAY_COMPLETE "
            f"success={result.success}, games={result.games_generated}"
        )

    async def _on_canonical_selfplay_stage_complete(self, result) -> None:
        """Handle CANONICAL_SELFPLAY_COMPLETE stage event."""
        logger.debug(
            "[SelfplayOrchestrator] Stage event: CANONICAL_SELFPLAY_COMPLETE "
            f"success={result.success}, games={result.games_generated}"
        )

    # =========================================================================
    # Resource/Backpressure Event Handlers (December 2025)
    # =========================================================================

    async def _on_backpressure_activated(self, event) -> None:
        """Handle BACKPRESSURE_ACTIVATED - slow down selfplay on affected node."""
        payload = event.payload
        node_id = payload.get("node_id", "")
        level = payload.get("level", "medium")

        if not node_id:
            return

        self._backpressure_nodes[node_id] = level

        # Log with different severity based on level
        if level in ("critical", "high"):
            logger.warning(
                f"[SelfplayOrchestrator] Backpressure {level.upper()} on {node_id} - "
                "selfplay may be throttled"
            )
        else:
            logger.info(
                f"[SelfplayOrchestrator] Backpressure {level} activated on {node_id}"
            )

    async def _on_backpressure_released(self, event) -> None:
        """Handle BACKPRESSURE_RELEASED - resume normal selfplay on node."""
        payload = event.payload
        node_id = payload.get("node_id", "")

        if node_id and node_id in self._backpressure_nodes:
            prev_level = self._backpressure_nodes.pop(node_id)
            logger.info(
                f"[SelfplayOrchestrator] Backpressure released on {node_id} "
                f"(was {prev_level})"
            )

    async def _on_resource_constraint(self, event) -> None:
        """Handle RESOURCE_CONSTRAINT_DETECTED - track constrained nodes."""
        payload = event.payload
        node_id = payload.get("node_id", "")
        constraint_type = payload.get("constraint_type", "unknown")

        if not node_id:
            return

        self._resource_constrained_nodes[node_id] = time.time()

        logger.warning(
            f"[SelfplayOrchestrator] Resource constraint ({constraint_type}) "
            f"detected on {node_id} - consider reducing selfplay load"
        )

    async def _on_regression_detected(self, event) -> None:
        """Handle REGRESSION_DETECTED - pause selfplay if model regressed."""
        payload = event.payload
        severity = payload.get("severity", "minor")
        metric_name = payload.get("metric_name") or payload.get("metric", "")

        if severity in ("severe", "critical"):
            self._paused_for_regression = True
            logger.warning(
                f"[SelfplayOrchestrator] {severity.upper()} regression detected "
                f"in {metric_name} - selfplay should pause until resolved"
            )
        else:
            logger.info(
                f"[SelfplayOrchestrator] Minor regression detected in {metric_name}"
            )

    async def _on_curriculum_rebalanced(self, event) -> None:
        """Handle CURRICULUM_REBALANCED - update selfplay allocation weights.

        December 2025: Phase 1 of self-improvement feedback loop.
        This closes the gap between curriculum feedback and selfplay allocation.
        Weights are used by get_config_weight() for job allocation.
        """
        payload = event.payload
        new_weights = payload.get("new_weights") or payload.get("all_weights", {})
        config_key = payload.get("config_key") or payload.get("config", "")
        trigger = payload.get("trigger", "unknown")

        if new_weights:
            # Update all weights
            old_weights = dict(self._curriculum_weights)
            self._curriculum_weights.update(new_weights)
            self._curriculum_weights_updated_at = time.time()

            # Log significant changes
            changed_configs = []
            for config, weight in new_weights.items():
                old_weight = old_weights.get(config, 1.0)
                if abs(weight - old_weight) >= 0.1:
                    changed_configs.append(f"{config}: {old_weight:.2f}→{weight:.2f}")

            if changed_configs:
                logger.info(
                    f"[SelfplayOrchestrator] Curriculum weights updated (trigger={trigger}): "
                    f"{', '.join(changed_configs)}"
                )
            else:
                logger.debug(
                    f"[SelfplayOrchestrator] Curriculum weights updated (trigger={trigger}), "
                    f"{len(new_weights)} configs"
                )
        elif config_key:
            # Single config update
            new_weight = payload.get("new_weight", 1.0)
            old_weight = self._curriculum_weights.get(config_key, 1.0)
            self._curriculum_weights[config_key] = new_weight
            self._curriculum_weights_updated_at = time.time()

            if abs(new_weight - old_weight) >= 0.1:
                logger.info(
                    f"[SelfplayOrchestrator] Curriculum weight for {config_key}: "
                    f"{old_weight:.2f}→{new_weight:.2f} (trigger={trigger})"
                )

    async def _on_quality_updated(self, event) -> None:
        """Handle QUALITY_SCORE_UPDATED - adjust selfplay budget based on quality.

        December 2025: Closes the quality → selfplay feedback gap.
        Low quality data triggers higher exploration budgets.
        High quality data allows lower budgets for efficiency.

        Budget multiplier range:
        - quality < 0.3: 1.5x budget (needs much more exploration)
        - quality 0.3-0.5: 1.3x budget (needs more exploration)
        - quality 0.5-0.7: 1.0x budget (normal)
        - quality > 0.7: 0.9x budget (already exploring well)
        """
        payload = event.payload
        config_key = payload.get("config_key") or payload.get("config", "")
        quality_score = payload.get("quality_score", 0.5)

        if not config_key:
            return

        # Store quality score for tracking
        old_quality = self._quality_scores.get(config_key, 0.5)
        self._quality_scores[config_key] = quality_score

        # Compute budget multiplier based on quality
        if quality_score < 0.3:
            multiplier = 1.5  # Very low quality - needs aggressive exploration
        elif quality_score < 0.5:
            multiplier = 1.3  # Low quality - needs more exploration
        elif quality_score < 0.7:
            multiplier = 1.0  # Normal quality - standard budget
        else:
            multiplier = 0.9  # High quality - can reduce budget for efficiency

        old_multiplier = self._quality_budget_multipliers.get(config_key, 1.0)
        self._quality_budget_multipliers[config_key] = multiplier

        # Log significant changes
        if abs(multiplier - old_multiplier) >= 0.1:
            logger.info(
                f"[SelfplayOrchestrator] Quality-based budget for {config_key}: "
                f"{old_multiplier:.1f}x→{multiplier:.1f}x (quality={quality_score:.2f})"
            )

    async def _on_idle_resource_detected(self, event) -> None:
        """Handle IDLE_RESOURCE_DETECTED - spawn selfplay on idle GPUs.

        December 2025: Closes the idle resource → selfplay feedback loop.
        When IdleResourceDaemon detects idle GPU resources, this handler
        spawns selfplay jobs to utilize them.

        The handler uses SelfplayScheduler priorities to decide which
        config to run selfplay for (curriculum weights, ELO velocity).
        """
        payload = event.payload if hasattr(event, "payload") else event
        node_id = payload.get("node_id", "")
        host = payload.get("host", "")
        gpu_utilization = payload.get("gpu_utilization", 0.0)
        idle_duration_seconds = payload.get("idle_duration_seconds", 0)

        if not node_id and not host:
            return

        # Skip if node is under backpressure
        if node_id and self.is_node_under_backpressure(node_id):
            logger.debug(f"[SelfplayOrchestrator] Skipping idle {node_id} - under backpressure")
            return

        # Get next config to run based on curriculum priorities
        try:
            from app.coordination.selfplay_scheduler import get_selfplay_scheduler

            scheduler = get_selfplay_scheduler()
            if scheduler is None:
                logger.debug("[SelfplayOrchestrator] No scheduler available for idle resource")
                return

            next_config = scheduler.get_next_selfplay_config()
            if next_config is None:
                logger.debug("[SelfplayOrchestrator] No selfplay config needed")
                return

            # Request selfplay for this config
            self.request_selfplay(
                config_key=next_config,
                num_games=100,  # Default batch size
                priority="normal",
                source="idle_resource_detection",
            )

            logger.info(
                f"[SelfplayOrchestrator] Spawning selfplay on idle {node_id or host}: "
                f"config={next_config}, gpu_util={gpu_utilization:.1f}%, "
                f"idle={idle_duration_seconds:.0f}s"
            )

        except ImportError:
            logger.debug("[SelfplayOrchestrator] SelfplayScheduler not available")
        except Exception as e:
            logger.warning(f"[SelfplayOrchestrator] Failed to spawn selfplay on idle resource: {e}")

    def get_quality_budget_multiplier(self, config_key: str) -> float:
        """Get the quality-based budget multiplier for a config.

        Args:
            config_key: Config identifier (e.g., "square8_2p")

        Returns:
            Budget multiplier (0.9 to 1.5), defaults to 1.0
        """
        return self._quality_budget_multipliers.get(config_key, 1.0)

    def get_effective_budget(self, config_key: str, base_budget: int) -> int:
        """Get the effective budget considering quality adjustments.

        Args:
            config_key: Config identifier
            base_budget: Base simulation budget

        Returns:
            Adjusted budget based on quality feedback
        """
        multiplier = self.get_quality_budget_multiplier(config_key)
        return max(16, int(base_budget * multiplier))

    def is_node_under_backpressure(self, node_id: str) -> bool:
        """Check if a node is under backpressure.

        Args:
            node_id: Node to check

        Returns:
            True if node has active backpressure
        """
        return node_id in self._backpressure_nodes

    def get_node_backpressure_level(self, node_id: str) -> str | None:
        """Get backpressure level for a node.

        Args:
            node_id: Node to check

        Returns:
            Backpressure level string or None if no backpressure
        """
        return self._backpressure_nodes.get(node_id)

    def is_paused_for_regression(self) -> bool:
        """Check if selfplay is paused due to model regression."""
        return self._paused_for_regression

    def clear_regression_pause(self) -> None:
        """Clear the regression pause flag."""
        if self._paused_for_regression:
            self._paused_for_regression = False
            logger.info("[SelfplayOrchestrator] Regression pause cleared")

    def get_constrained_nodes(self) -> list[str]:
        """Get list of nodes with recent resource constraints."""
        # Return nodes with constraints in the last 5 minutes
        cutoff = time.time() - 300.0
        return [
            node_id
            for node_id, timestamp in self._resource_constrained_nodes.items()
            if timestamp > cutoff
        ]

    # =========================================================================
    # Curriculum Weight Methods (December 2025 - Phase 1 Feedback Loop)
    # =========================================================================

    def get_config_weight(self, config_key: str) -> float:
        """Get the curriculum weight for a specific config.

        Weights control selfplay job allocation:
        - 1.0 = normal allocation
        - >1.0 = higher priority (needs more training)
        - <1.0 = lower priority (already performing well)

        Args:
            config_key: Config identifier (e.g., "square8_2p", "hex8_4p")

        Returns:
            Weight value (0.5 to 2.0), defaults to 1.0 if not set
        """
        return self._curriculum_weights.get(config_key, 1.0)

    def get_all_curriculum_weights(self) -> dict[str, float]:
        """Get all curriculum weights.

        Returns:
            Dict mapping config_key to weight
        """
        return dict(self._curriculum_weights)

    def set_curriculum_weight(self, config_key: str, weight: float) -> None:
        """Manually set a curriculum weight (for testing or override).

        Args:
            config_key: Config identifier
            weight: Weight value (will be clamped to 0.5-2.0)
        """
        weight = max(0.5, min(2.0, weight))
        self._curriculum_weights[config_key] = weight
        self._curriculum_weights_updated_at = time.time()
        logger.debug(f"[SelfplayOrchestrator] Set curriculum weight {config_key}={weight:.2f}")

    def load_curriculum_weights_from_feedback(self) -> bool:
        """Load curriculum weights from CurriculumFeedback singleton.

        Call this at startup to initialize weights from the feedback system.

        Returns:
            True if weights were loaded successfully
        """
        try:
            from app.training.curriculum_feedback import get_curriculum_weights
            weights = get_curriculum_weights()
            if weights:
                self._curriculum_weights.update(weights)
                self._curriculum_weights_updated_at = time.time()
                logger.info(
                    f"[SelfplayOrchestrator] Loaded {len(weights)} curriculum weights from feedback"
                )
                return True
        except ImportError:
            logger.debug("[SelfplayOrchestrator] curriculum_feedback not available")
        except Exception as e:
            logger.warning(f"[SelfplayOrchestrator] Failed to load curriculum weights: {e}")
        return False

    def calculate_job_allocation(
        self,
        base_jobs_per_config: int,
        configs: list[str] | None = None,
        skip_stuck_configs: bool = True,
        min_plateau_hours: int = 48,
    ) -> dict[str, int]:
        """Calculate job allocation for each config based on curriculum weights.

        This is the main method for applying curriculum weights to selfplay
        job distribution. Higher-weighted configs get more jobs.

        December 2025: Now supports stuck config detection (Phase 4 feedback loop).
        Configs with Elo plateau > min_plateau_hours get 0 jobs if skip_stuck_configs=True.

        Args:
            base_jobs_per_config: Base number of jobs per config (before weighting)
            configs: List of configs to allocate (None = all known configs)
            skip_stuck_configs: If True, skip configs that are permanently stuck (Elo plateau)
            min_plateau_hours: Minimum hours of plateau to consider a config stuck

        Returns:
            Dict mapping config_key to number of jobs

        Example:
            >>> orchestrator.set_curriculum_weight("square8_2p", 1.5)
            >>> orchestrator.set_curriculum_weight("hex8_2p", 0.8)
            >>> orchestrator.calculate_job_allocation(10, ["square8_2p", "hex8_2p"])
            {"square8_2p": 15, "hex8_2p": 8}
        """
        if configs is None:
            configs = list(self._curriculum_weights.keys())

        if not configs:
            return {}

        # Phase 4 Feedback Loop: Detect and filter stuck configs
        stuck_configs: set[str] = set()
        if skip_stuck_configs:
            try:
                from app.training.curriculum_feedback import get_curriculum_feedback

                cf = get_curriculum_feedback()
                stuck_configs = cf.get_stuck_config_keys(min_plateau_hours=min_plateau_hours)
                if stuck_configs:
                    logger.warning(
                        f"[SelfplayOrchestrator] Stuck configs detected (>{min_plateau_hours}h plateau): "
                        f"{sorted(stuck_configs)} - allocating 0 jobs"
                    )
            except Exception as e:
                logger.warning(f"[SelfplayOrchestrator] Failed to detect stuck configs: {e}")
                # Continue without stuck detection on error

        allocation = {}
        for config in configs:
            # Skip stuck configs - they get 0 jobs
            if config in stuck_configs:
                allocation[config] = 0
                continue

            weight = self.get_config_weight(config)
            jobs = int(round(base_jobs_per_config * weight))
            allocation[config] = max(1, jobs)  # At least 1 job per config

        logger.debug(
            f"[SelfplayOrchestrator] Job allocation: {allocation} "
            f"(base={base_jobs_per_config}, stuck_skipped={len(stuck_configs)})"
        )
        return allocation

    def get_curriculum_status(self) -> dict[str, Any]:
        """Get curriculum weight status for monitoring.

        Returns:
            Dict with curriculum weight info
        """
        return {
            "weights": dict(self._curriculum_weights),
            "num_configs": len(self._curriculum_weights),
            "updated_at": self._curriculum_weights_updated_at,
            "age_seconds": time.time() - self._curriculum_weights_updated_at
            if self._curriculum_weights_updated_at > 0 else None,
        }

    def register_task(
        self,
        task_id: str,
        selfplay_type: SelfplayType,
        node_id: str,
        board_type: str = "square8",
        num_players: int = 2,
        games_requested: int = 0,
        iteration: int = 0,
    ) -> SelfplayTaskInfo:
        """Manually register a selfplay task.

        Use this when spawning selfplay outside the task coordinator.

        Returns:
            The created SelfplayTaskInfo
        """
        task = SelfplayTaskInfo(
            task_id=task_id,
            selfplay_type=selfplay_type,
            node_id=node_id,
            board_type=board_type,
            num_players=num_players,
            games_requested=games_requested,
            iteration=iteration,
        )

        self._active_tasks[task_id] = task
        logger.debug(f"[SelfplayOrchestrator] Registered task {task_id}")
        return task

    async def complete_task(
        self,
        task_id: str,
        success: bool = True,
        games_generated: int = 0,
        error: str = "",
    ) -> SelfplayTaskInfo | None:
        """Manually mark a selfplay task as complete.

        Use this when selfplay completes outside the task coordinator.

        Returns:
            The completed SelfplayTaskInfo, or None if not found
        """
        if task_id not in self._active_tasks:
            logger.warning(f"[SelfplayOrchestrator] Unknown task {task_id}")
            return None

        task = self._active_tasks.pop(task_id)
        task.success = success
        task.end_time = time.time()
        task.games_generated = games_generated
        task.error = error

        self._record_completion(task, {})

        # Emit stage completion event
        await self._emit_selfplay_complete(task)

        return task

    def on_completion(self, callback: Callable[[SelfplayTaskInfo], None]) -> None:
        """Register a callback for selfplay completions.

        Args:
            callback: Function to call when selfplay completes
        """
        self._completion_callbacks.append(callback)

    def get_active_tasks(self) -> list[SelfplayTaskInfo]:
        """Get all active selfplay tasks."""
        return list(self._active_tasks.values())

    def get_task(self, task_id: str) -> SelfplayTaskInfo | None:
        """Get a specific task by ID."""
        return self._active_tasks.get(task_id)

    def get_history(self, limit: int = 50) -> list[SelfplayTaskInfo]:
        """Get recent task completion history."""
        return self._completed_history[-limit:]

    def get_stats(self) -> SelfplayStats:
        """Get aggregate selfplay statistics."""
        # Count by node
        by_node: dict[str, int] = {}
        for task in self._active_tasks.values():
            by_node[task.node_id] = by_node.get(task.node_id, 0) + 1

        # Count by type
        by_type: dict[str, int] = {}
        for task in self._active_tasks.values():
            type_key = task.selfplay_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1

        # Calculate averages
        completed = [t for t in self._completed_history if t.success]
        avg_games = (
            sum(t.games_generated for t in completed) / len(completed)
            if completed
            else 0.0
        )
        avg_gps = (
            sum(t.games_per_second for t in completed) / len(completed)
            if completed
            else 0.0
        )

        failed_count = sum(1 for t in self._completed_history if not t.success)

        return SelfplayStats(
            active_tasks=len(self._active_tasks),
            completed_tasks=len(completed),
            failed_tasks=failed_count,
            total_games_generated=self._total_games_generated,
            total_duration_seconds=self._total_duration_seconds,
            average_games_per_task=avg_games,
            average_games_per_second=avg_gps,
            by_node=by_node,
            by_type=by_type,
            last_activity_time=time.time(),
        )

    def get_status(self) -> dict[str, Any]:
        """Get orchestrator status for monitoring."""
        stats = self.get_stats()

        return {
            "active_tasks": stats.active_tasks,
            "completed_tasks": stats.completed_tasks,
            "failed_tasks": stats.failed_tasks,
            "total_games_generated": stats.total_games_generated,
            "average_games_per_task": round(stats.average_games_per_task, 1),
            "average_games_per_second": round(stats.average_games_per_second, 2),
            "by_node": stats.by_node,
            "by_type": stats.by_type,
            "subscribed": self._subscribed,
            "history_size": len(self._completed_history),
            # Backpressure tracking (December 2025)
            "nodes_under_backpressure": list(self._backpressure_nodes.keys()),
            "backpressure_levels": dict(self._backpressure_nodes),
            "paused_for_regression": self._paused_for_regression,
            "constrained_nodes": self.get_constrained_nodes(),
            # Curriculum weights (December 2025 - Phase 1 feedback loop)
            "curriculum_weights": dict(self._curriculum_weights),
            "curriculum_weights_updated_at": self._curriculum_weights_updated_at,
            # Quality-based budget multipliers (December 2025)
            "quality_scores": dict(self._quality_scores),
            "quality_budget_multipliers": dict(self._quality_budget_multipliers),
        }

    def clear_history(self) -> int:
        """Clear task history. Returns number of entries cleared."""
        count = len(self._completed_history)
        self._completed_history.clear()
        return count


# =============================================================================
# Singleton and convenience functions
# =============================================================================

# =============================================================================
# Engine Selection for Large Boards (December 2025)
# =============================================================================

# Default large boards - these use Gumbel MCTS for quality selfplay
_DEFAULT_LARGE_BOARDS: set[str] = {"square19", "hexagonal", "full_hex", "fullhex"}

# Default engine override for large boards
_DEFAULT_LARGE_BOARD_ENGINE = "gumbel_mcts"

# Default simulation budget for large boards
# CHANGED Dec 2025: Increased from 64 to 800 for 2000+ Elo quality training data
# 64 sims produces fast but low-quality data; 800 sims produces expert-level moves
_DEFAULT_LARGE_BOARD_BUDGET = 800  # QUALITY tier for 2000+ Elo target


def _load_large_board_config() -> dict:
    """Load large board configuration from unified_loop.yaml."""
    try:
        import yaml
        from pathlib import Path

        config_path = Path(__file__).parents[2] / "config" / "unified_loop.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            return config.get("selfplay", {}).get("large_board_config", {})
    except Exception as e:
        logger.debug(f"[SelfplayOrchestrator] Could not load large board config: {e}")
    return {}


def is_large_board(board_type: str) -> bool:
    """Check if a board type is classified as 'large'.

    Large boards require specialized engine selection (Gumbel MCTS) for
    quality selfplay due to their size (sq19=361 cells, hexagonal=469 cells).

    Args:
        board_type: Board type string (e.g., "square19", "hexagonal", "full hex")

    Returns:
        True if board is classified as large
    """
    # Normalize board type to canonical form (handles "full hex" -> "hexagonal")
    try:
        from app.utils.canonical_naming import normalize_board_type
        canonical = normalize_board_type(board_type)
    except Exception:
        canonical = board_type.lower()

    config = _load_large_board_config()
    large_boards = set(config.get("large_boards", _DEFAULT_LARGE_BOARDS))
    return canonical in large_boards


def get_engine_for_board(board_type: str, num_players: int = 2) -> str:
    """Get the recommended selfplay engine for a board configuration.

    For large boards (sq19, hexagonal, full hex), returns Gumbel MCTS which
    provides the best quality training data at acceptable speed.

    For small boards, returns empty string to use the default weighted selection
    from ai_type_weights config.

    Args:
        board_type: Board type string (e.g., "square19", "full hex")
        num_players: Number of players

    Returns:
        Engine name string (e.g., "gumbel_mcts") or empty string for default
    """
    if is_large_board(board_type):
        config = _load_large_board_config()
        return config.get("engine_override", _DEFAULT_LARGE_BOARD_ENGINE)
    return ""  # Use default weighted selection


def get_simulation_budget_for_board(board_type: str) -> int:
    """Get the Gumbel MCTS simulation budget for a board type.

    Large boards use THROUGHPUT tier (64 sims) by default for 5-10 games/sec.
    Small boards use the standard budget from gpu_mcts config.

    Args:
        board_type: Board type string (e.g., "square19", "full hex")

    Returns:
        Simulation budget (number of MCTS simulations per move)
    """
    if is_large_board(board_type):
        config = _load_large_board_config()
        return config.get("simulation_budget", _DEFAULT_LARGE_BOARD_BUDGET)
    return _DEFAULT_LARGE_BOARD_BUDGET


_selfplay_orchestrator: SelfplayOrchestrator | None = None


def get_selfplay_orchestrator() -> SelfplayOrchestrator:
    """Get the global SelfplayOrchestrator singleton."""
    global _selfplay_orchestrator
    if _selfplay_orchestrator is None:
        _selfplay_orchestrator = SelfplayOrchestrator()
    return _selfplay_orchestrator


def wire_selfplay_events() -> SelfplayOrchestrator:
    """Wire selfplay events to the orchestrator.

    This subscribes the orchestrator to task lifecycle events
    and stage completion events.

    Returns:
        The wired SelfplayOrchestrator instance
    """
    orchestrator = get_selfplay_orchestrator()
    orchestrator.subscribe_to_events()
    orchestrator.subscribe_to_stage_events()
    return orchestrator


def wire_curriculum_to_selfplay() -> SelfplayOrchestrator:
    """Wire curriculum feedback to selfplay orchestrator.

    December 2025: Phase 1 of self-improvement feedback loop.
    This ensures:
    1. Selfplay orchestrator subscribes to CURRICULUM_REBALANCED events
    2. Initial curriculum weights are loaded from CurriculumFeedback
    3. Job allocation uses curriculum weights

    Call this at startup to enable curriculum-aware selfplay allocation.

    Returns:
        The configured SelfplayOrchestrator instance

    Example:
        from app.coordination.selfplay_orchestrator import wire_curriculum_to_selfplay

        # At startup
        orchestrator = wire_curriculum_to_selfplay()

        # Later, get weighted job allocation
        allocation = orchestrator.calculate_job_allocation(
            base_jobs_per_config=10,
            configs=["square8_2p", "hex8_2p", "square8_4p"]
        )
        # Result: {"square8_2p": 12, "hex8_2p": 8, "square8_4p": 15}
    """
    orchestrator = get_selfplay_orchestrator()

    # Subscribe to events (including CURRICULUM_REBALANCED)
    orchestrator.subscribe_to_events()
    orchestrator.subscribe_to_stage_events()

    # Load initial weights from curriculum feedback
    orchestrator.load_curriculum_weights_from_feedback()

    logger.info("[wire_curriculum_to_selfplay] Curriculum feedback wired to selfplay orchestrator")
    return orchestrator


async def emit_selfplay_completion(
    task_id: str,
    board_type: str,
    num_players: int,
    games_generated: int,
    success: bool = True,
    node_id: str = "",
    selfplay_type: str = "canonical",
    iteration: int = 0,
    error: str = "",
) -> bool:
    """Emit a selfplay completion event.

    Call this when selfplay finishes to notify downstream pipeline stages.

    Args:
        task_id: Task identifier
        board_type: Board type (e.g., "square8")
        num_players: Number of players
        games_generated: Number of games generated
        success: Whether selfplay succeeded
        node_id: Node that ran selfplay
        selfplay_type: Type of selfplay (canonical, gpu_selfplay, etc.)
        iteration: Pipeline iteration number
        error: Error message if failed

    Returns:
        True if event was emitted successfully
    """
    orchestrator = get_selfplay_orchestrator()

    # Check if task is already tracked
    task = orchestrator.get_task(task_id)

    if task is None:
        # Create and register the task
        type_enum = SelfplayType.CANONICAL
        if selfplay_type == "gpu_selfplay":
            type_enum = SelfplayType.GPU_ACCELERATED
        elif selfplay_type == "hybrid_selfplay":
            type_enum = SelfplayType.HYBRID

        task = orchestrator.register_task(
            task_id=task_id,
            selfplay_type=type_enum,
            node_id=node_id,
            board_type=board_type,
            num_players=num_players,
            games_requested=games_generated,
            iteration=iteration,
        )

    # Complete the task (this emits the stage event)
    result = await orchestrator.complete_task(
        task_id=task_id,
        success=success,
        games_generated=games_generated,
        error=error,
    )

    return result is not None


def get_selfplay_stats() -> SelfplayStats:
    """Convenience function to get selfplay statistics."""
    return get_selfplay_orchestrator().get_stats()


__all__ = [
    "SelfplayOrchestrator",
    "SelfplayStats",
    "SelfplayTaskInfo",
    "SelfplayType",
    "emit_selfplay_completion",
    "get_engine_for_board",
    "get_selfplay_orchestrator",
    "get_selfplay_stats",
    "get_simulation_budget_for_board",
    "is_large_board",
    "wire_curriculum_to_selfplay",
    "wire_selfplay_events",
]
