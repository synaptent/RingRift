"""SyncPlanner v2 - Intelligent Sync Planning for Unified Data Plane.

December 28, 2025 - Phase 3 of Unified Data Plane implementation.

This module provides intelligent sync planning that determines:
- WHAT data to sync (based on DataCatalog queries)
- WHERE to sync it (target node selection)
- WHEN to sync (priority and deadline awareness)
- HOW to sync (transport preference)

Key features:
- Deadline-aware planning for training dependencies
- Replication factor enforcement (ensure min_replicas copies exist)
- Quality-based prioritization (sync high-quality games first)
- Ephemeral node urgency (prioritize data rescue from Vast.ai)
- Training dependency resolution (ensure data exists before training starts)
- Event-driven planning (responds to pipeline events)

Architecture:
    DataCatalog (what exists where)
         ↓
    SyncPlanner (decides what to sync)
         ↓
    TransportManager (executes transfers)
         ↓
    EventBridge (notifies completion)

Usage:
    from app.coordination.sync_planner_v2 import (
        SyncPlanner,
        SyncPlan,
        SyncPriority,
        get_sync_planner,
    )

    planner = get_sync_planner()

    # Plan for an event
    plans = planner.plan_for_event("SELFPLAY_COMPLETE", {"config_key": "hex8_2p"})

    # Plan training dependencies
    plan = planner.plan_training_deps("training-node-1", "hex8_2p")

    # Plan replication enforcement
    plans = planner.plan_replication(min_factor=3)

    # Execute plans
    for plan in plans:
        await transport_manager.execute_plan(plan)
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import socket
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from app.coordination.data_catalog import (
    DataCatalog,
    DataEntry,
    DataType,
    get_data_catalog,
)
from app.coordination.transport_manager import (
    Transport,
    TransportManager,
    get_transport_manager,
)
from app.coordination.protocols import (
    CoordinatorStatus,
    HealthCheckResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Enums
    "SyncPriority",
    # Data classes
    "SyncPlan",
    "PlannerConfig",
    "PlannerStats",
    # Main class
    "SyncPlanner",
    # Singleton accessors
    "get_sync_planner",
    "reset_sync_planner",
]


# =============================================================================
# Enums
# =============================================================================

# Import from canonical location (December 2025 consolidation)
from app.coordination.sync_constants import SyncPriority


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SyncPlan:
    """A planned sync operation.

    Describes what data to sync, where to sync it, and how urgent it is.
    Plans are executed by TransportManager.
    """

    source_node: str
    target_nodes: list[str]
    entries: list[DataEntry]
    priority: SyncPriority = SyncPriority.NORMAL
    reason: str = ""
    transport_preference: list[Transport] = field(default_factory=list)
    deadline: float | None = None  # Unix timestamp, None = no deadline
    config_key: str | None = None  # Optional config filter
    batch_id: str = ""  # For tracking related plans

    # Execution state (filled by executor)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    success: bool | None = None
    error: str | None = None

    def __lt__(self, other: "SyncPlan") -> bool:
        """Compare for priority queue (higher priority first)."""
        if self.priority != other.priority:
            return self.priority > other.priority
        # Same priority: earlier deadline first
        if self.deadline and other.deadline:
            return self.deadline < other.deadline
        # Plans with deadlines come first
        if self.deadline:
            return True
        return False

    @property
    def total_bytes(self) -> int:
        """Total bytes to transfer."""
        return sum(e.size_bytes for e in self.entries)

    @property
    def is_expired(self) -> bool:
        """Check if deadline has passed."""
        if self.deadline is None:
            return False
        return time.time() > self.deadline

    @property
    def time_to_deadline(self) -> float | None:
        """Seconds until deadline, or None if no deadline."""
        if self.deadline is None:
            return None
        return max(0, self.deadline - time.time())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_node": self.source_node,
            "target_nodes": self.target_nodes,
            "entry_count": len(self.entries),
            "total_bytes": self.total_bytes,
            "priority": self.priority.name,
            "reason": self.reason,
            "transport_preference": [t.value for t in self.transport_preference],
            "deadline": self.deadline,
            "config_key": self.config_key,
            "batch_id": self.batch_id,
            "created_at": self.created_at,
            "is_expired": self.is_expired,
        }


@dataclass
class PlannerConfig:
    """Configuration for SyncPlanner."""

    # Replication settings
    min_replication_factor: int = 3
    target_replication_factor: int = 5
    max_replication_factor: int = 10

    # Deadline settings
    training_deadline_seconds: float = 300.0  # 5 min to get training deps
    ephemeral_rescue_deadline_seconds: float = 60.0  # 1 min to rescue from ephemeral
    model_sync_deadline_seconds: float = 120.0  # 2 min for model promotion

    # Planning limits
    max_plans_per_event: int = 20
    max_targets_per_plan: int = 5
    max_entries_per_plan: int = 100
    max_bytes_per_plan: int = 1_000_000_000  # 1GB

    # Priority boosts
    quality_priority_boost: float = 20.0  # Max boost for high quality
    ephemeral_priority_boost: int = 30
    training_priority_boost: int = 25

    # Background planning
    replication_check_interval: float = 300.0  # 5 min
    orphan_check_interval: float = 60.0  # 1 min

    @classmethod
    def from_env(cls) -> "PlannerConfig":
        """Create config from environment variables."""
        import os

        return cls(
            min_replication_factor=int(
                os.getenv("RINGRIFT_MIN_REPLICATION", "3")
            ),
            target_replication_factor=int(
                os.getenv("RINGRIFT_TARGET_REPLICATION", "5")
            ),
            training_deadline_seconds=float(
                os.getenv("RINGRIFT_TRAINING_DEADLINE", "300")
            ),
            ephemeral_rescue_deadline_seconds=float(
                os.getenv("RINGRIFT_EPHEMERAL_DEADLINE", "60")
            ),
        )


@dataclass
class PlannerStats:
    """Statistics for sync planning."""

    plans_created: int = 0
    plans_executed: int = 0
    plans_succeeded: int = 0
    plans_failed: int = 0
    plans_expired: int = 0
    total_bytes_planned: int = 0
    total_entries_planned: int = 0

    # Per-priority counters
    critical_plans: int = 0
    high_plans: int = 0
    normal_plans: int = 0
    low_plans: int = 0
    background_plans: int = 0

    # Per-reason counters
    training_dep_plans: int = 0
    replication_plans: int = 0
    orphan_recovery_plans: int = 0
    event_triggered_plans: int = 0

    last_plan_time: float = 0.0
    last_execution_time: float = 0.0

    def record_plan(self, plan: SyncPlan) -> None:
        """Record a new plan."""
        self.plans_created += 1
        self.total_bytes_planned += plan.total_bytes
        self.total_entries_planned += len(plan.entries)
        self.last_plan_time = time.time()

        # Priority counter
        if plan.priority >= SyncPriority.CRITICAL:
            self.critical_plans += 1
        elif plan.priority >= SyncPriority.HIGH:
            self.high_plans += 1
        elif plan.priority >= SyncPriority.NORMAL:
            self.normal_plans += 1
        elif plan.priority >= SyncPriority.LOW:
            self.low_plans += 1
        else:
            self.background_plans += 1

    def record_execution(self, plan: SyncPlan, success: bool) -> None:
        """Record plan execution result."""
        self.plans_executed += 1
        self.last_execution_time = time.time()

        if plan.is_expired:
            self.plans_expired += 1
        elif success:
            self.plans_succeeded += 1
        else:
            self.plans_failed += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plans_created": self.plans_created,
            "plans_executed": self.plans_executed,
            "plans_succeeded": self.plans_succeeded,
            "plans_failed": self.plans_failed,
            "plans_expired": self.plans_expired,
            "total_bytes_planned": self.total_bytes_planned,
            "total_entries_planned": self.total_entries_planned,
            "success_rate": (
                self.plans_succeeded / self.plans_executed
                if self.plans_executed > 0
                else 0.0
            ),
            "priority_breakdown": {
                "critical": self.critical_plans,
                "high": self.high_plans,
                "normal": self.normal_plans,
                "low": self.low_plans,
                "background": self.background_plans,
            },
            "last_plan_time": self.last_plan_time,
            "last_execution_time": self.last_execution_time,
        }


# =============================================================================
# Main Class
# =============================================================================


class SyncPlanner:
    """Intelligent sync planner for the Unified Data Plane.

    Decides what to sync, where to sync it, and when. Works with DataCatalog
    to understand current data distribution and TransportManager to execute
    actual transfers.

    Key responsibilities:
    1. Event-driven planning (respond to SELFPLAY_COMPLETE, TRAINING_STARTED, etc.)
    2. Training dependency resolution (ensure data exists before training)
    3. Replication factor enforcement (maintain min copies)
    4. Ephemeral node rescue (urgent sync from Vast.ai before termination)
    5. Orphan game recovery (sync unregistered games)
    """

    def __init__(
        self,
        catalog: DataCatalog | None = None,
        transport_manager: TransportManager | None = None,
        config: PlannerConfig | None = None,
    ):
        """Initialize the sync planner.

        Args:
            catalog: DataCatalog for data location queries
            transport_manager: TransportManager for executing plans
            config: Planning configuration
        """
        self._catalog = catalog or get_data_catalog()
        self._transport = transport_manager or get_transport_manager()
        self._config = config or PlannerConfig.from_env()

        self._node_id = socket.gethostname()
        self._stats = PlannerStats()

        # Pending plans priority queue (max-heap via negation in __lt__)
        self._pending_plans: list[SyncPlan] = []
        self._plan_lock = asyncio.Lock()

        # Event handlers
        self._event_handlers: dict[str, Callable] = {}
        self._register_default_handlers()

        # Background tasks
        self._running = False
        self._replication_task: asyncio.Task | None = None
        self._execution_task: asyncio.Task | None = None

        # Node capability cache (lazy loaded)
        self._node_capabilities: dict[str, dict] = {}
        self._capabilities_loaded = False

        logger.info(
            f"[SyncPlanner] Initialized on {self._node_id} "
            f"(min_replication={self._config.min_replication_factor})"
        )

    def _register_default_handlers(self) -> None:
        """Register default event handlers."""
        self._event_handlers = {
            "SELFPLAY_COMPLETE": self._handle_selfplay_complete,
            "TRAINING_STARTED": self._handle_training_started,
            "TRAINING_STARTING": self._handle_training_started,  # Alias
            "MODEL_PROMOTED": self._handle_model_promoted,
            "ORPHAN_GAMES_DETECTED": self._handle_orphan_detected,
            "NODE_TERMINATING": self._handle_node_terminating,
            "DATA_SYNC_REQUESTED": self._handle_sync_requested,
            "SYNC_REQUEST": self._handle_sync_requested,  # Alias
        }

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """Start the planner's background tasks."""
        if self._running:
            return

        self._running = True

        # Start background replication check
        self._replication_task = asyncio.create_task(
            self._replication_check_loop(),
            name="sync_planner_replication",
        )

        # Start plan execution loop
        self._execution_task = asyncio.create_task(
            self._execution_loop(),
            name="sync_planner_execution",
        )

        logger.info("[SyncPlanner] Started background tasks")

    async def stop(self) -> None:
        """Stop the planner's background tasks."""
        self._running = False

        if self._replication_task:
            self._replication_task.cancel()
            try:
                await self._replication_task
            except asyncio.CancelledError:
                pass

        if self._execution_task:
            self._execution_task.cancel()
            try:
                await self._execution_task
            except asyncio.CancelledError:
                pass

        logger.info("[SyncPlanner] Stopped")

    # =========================================================================
    # Event-Driven Planning
    # =========================================================================

    def plan_for_event(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> list[SyncPlan]:
        """Generate sync plans in response to an event.

        Args:
            event_type: Type of event (e.g., "SELFPLAY_COMPLETE")
            payload: Event payload with details

        Returns:
            List of sync plans to execute
        """
        handler = self._event_handlers.get(event_type)
        if not handler:
            logger.debug(f"[SyncPlanner] No handler for event: {event_type}")
            return []

        try:
            plans = handler(payload)
            for plan in plans:
                self._stats.record_plan(plan)
                self._stats.event_triggered_plans += 1
            return plans
        except Exception as e:
            logger.error(f"[SyncPlanner] Error handling {event_type}: {e}")
            return []

    def _handle_selfplay_complete(self, payload: dict) -> list[SyncPlan]:
        """Handle SELFPLAY_COMPLETE event - sync new games to cluster."""
        source_node = payload.get("source_node", payload.get("node_id", self._node_id))
        config_key = payload.get("config_key", "")
        game_count = payload.get("game_count", payload.get("count", 0))
        db_path = payload.get("db_path", "")

        if not config_key:
            return []

        # Get entries for this config that exist only on source
        entries = self._catalog.get_by_config(config_key)
        source_only = [
            e for e in entries
            if source_node in e.locations and len(e.locations) == 1
        ]

        if not source_only:
            return []

        # Get training and GPU nodes as targets
        targets = self._get_training_targets(exclude=[source_node])
        if not targets:
            return []

        plan = SyncPlan(
            source_node=source_node,
            target_nodes=targets[:self._config.max_targets_per_plan],
            entries=source_only[:self._config.max_entries_per_plan],
            priority=SyncPriority.NORMAL,
            reason=f"selfplay_complete:{config_key}:{game_count}",
            transport_preference=[Transport.P2P_GOSSIP, Transport.RSYNC],
            config_key=config_key,
        )

        return [plan]

    def _handle_training_started(self, payload: dict) -> list[SyncPlan]:
        """Handle TRAINING_STARTED - ensure training deps are synced."""
        node_id = payload.get("node_id", payload.get("host", ""))
        config_key = payload.get("config_key", "")

        if not node_id or not config_key:
            return []

        # Plan to sync training dependencies
        plan = self.plan_training_deps(node_id, config_key)
        if plan:
            self._stats.training_dep_plans += 1
            return [plan]
        return []

    def _handle_model_promoted(self, payload: dict) -> list[SyncPlan]:
        """Handle MODEL_PROMOTED - sync model to all GPU nodes."""
        model_path = payload.get("model_path", "")
        source_node = payload.get("source_node", payload.get("node_id", self._node_id))
        config_key = payload.get("config_key", "")

        if not model_path:
            return []

        # Get model entry from catalog
        entry = self._catalog.get_by_path(model_path)
        if not entry:
            # Register the model first
            return []

        # Get all GPU nodes as targets
        targets = self._get_gpu_targets(exclude=[source_node])
        if not targets:
            return []

        plan = SyncPlan(
            source_node=source_node,
            target_nodes=targets,
            entries=[entry],
            priority=SyncPriority.HIGH,
            reason=f"model_promoted:{config_key}",
            transport_preference=[Transport.P2P_GOSSIP, Transport.HTTP_FETCH],
            deadline=time.time() + self._config.model_sync_deadline_seconds,
            config_key=config_key,
        )

        return [plan]

    def _handle_orphan_detected(self, payload: dict) -> list[SyncPlan]:
        """Handle ORPHAN_GAMES_DETECTED - urgent sync from ephemeral nodes."""
        source_node = payload.get("source_node", payload.get("node_id", ""))
        config_key = payload.get("config_key", "")
        game_ids = payload.get("game_ids", [])

        if not source_node:
            return []

        plan = self.plan_orphan_recovery(source_node, config_key, game_ids)
        if plan:
            self._stats.orphan_recovery_plans += 1
            return [plan]
        return []

    def _handle_node_terminating(self, payload: dict) -> list[SyncPlan]:
        """Handle NODE_TERMINATING - urgent rescue sync."""
        node_id = payload.get("node_id", "")

        if not node_id:
            return []

        # Get all data on this node
        entries = self._catalog.get_entries_on_node(node_id)
        if not entries:
            return []

        # Filter to under-replicated entries
        at_risk = [
            e for e in entries
            if self._catalog.get_replication_factor(e.path) < self._config.min_replication_factor
        ]

        if not at_risk:
            return []

        # Get stable targets
        targets = self._get_stable_targets(exclude=[node_id])
        if not targets:
            targets = [self._node_id]  # Fallback to coordinator

        plan = SyncPlan(
            source_node=node_id,
            target_nodes=targets[:3],  # Limit for urgency
            entries=at_risk[:50],  # Most important first
            priority=SyncPriority.CRITICAL,
            reason=f"node_terminating:{node_id}",
            transport_preference=[Transport.RSYNC, Transport.BASE64_SSH],
            deadline=time.time() + self._config.ephemeral_rescue_deadline_seconds,
        )

        return [plan]

    def _handle_sync_requested(self, payload: dict) -> list[SyncPlan]:
        """Handle explicit SYNC_REQUEST event."""
        source_node = payload.get("source", payload.get("source_node", self._node_id))
        target_nodes = payload.get("targets", payload.get("target_nodes", []))
        data_type = payload.get("data_type", "games")
        config_key = payload.get("config_key", "")
        priority_str = payload.get("priority", "NORMAL")

        # Parse priority
        try:
            priority = SyncPriority[priority_str.upper()]
        except (KeyError, AttributeError):
            priority = SyncPriority.NORMAL

        # Get entries to sync
        entries = []
        if config_key:
            entries = self._catalog.get_by_config(config_key)
        elif data_type:
            try:
                dt = DataType(data_type)
                entries = self._catalog.get_by_type(dt)
            except ValueError:
                pass

        if not entries or not target_nodes:
            return []

        plan = SyncPlan(
            source_node=source_node,
            target_nodes=target_nodes,
            entries=entries[:self._config.max_entries_per_plan],
            priority=priority,
            reason=f"sync_request:{config_key or data_type}",
            config_key=config_key,
        )

        return [plan]

    # =========================================================================
    # Specialized Planning Methods
    # =========================================================================

    def plan_training_deps(
        self,
        node_id: str,
        config_key: str,
    ) -> SyncPlan | None:
        """Plan sync to satisfy training dependencies.

        Ensures that the training node has all required data before
        training can start.

        Args:
            node_id: Training node that needs data
            config_key: Configuration (e.g., "hex8_2p")

        Returns:
            SyncPlan for training dependencies, or None if deps already met
        """
        # Get data missing on training node
        missing = self._catalog.get_missing_on_node(node_id, data_type=DataType.GAMES)
        if not missing:
            missing = self._catalog.get_missing_on_node(node_id, data_type=DataType.NPZ)

        # Filter by config
        if config_key:
            missing = [e for e in missing if e.config_key == config_key]

        if not missing:
            return None

        # Find best source (closest with most data)
        sources = self._find_sources_for_entries(missing)
        if not sources:
            return None

        best_source = sources[0]

        return SyncPlan(
            source_node=best_source,
            target_nodes=[node_id],
            entries=missing[:self._config.max_entries_per_plan],
            priority=SyncPriority.HIGH,
            reason=f"training_deps:{config_key}",
            transport_preference=[Transport.RSYNC, Transport.HTTP_FETCH],
            deadline=time.time() + self._config.training_deadline_seconds,
            config_key=config_key,
        )

    def plan_replication(
        self,
        min_factor: int | None = None,
    ) -> list[SyncPlan]:
        """Plan syncs to meet replication requirements.

        Scans catalog for under-replicated data and creates plans to
        increase replication factor.

        Args:
            min_factor: Minimum replication factor (defaults to config)

        Returns:
            List of sync plans for replication
        """
        min_factor = min_factor or self._config.min_replication_factor

        # Get under-replicated entries
        under_replicated = self._catalog.get_under_replicated(min_factor)
        if not under_replicated:
            return []

        plans = []
        processed_paths: set[str] = set()

        for entry in under_replicated[:self._config.max_plans_per_event]:
            if entry.path in processed_paths:
                continue
            processed_paths.add(entry.path)

            # Find source (has the data)
            if not entry.locations:
                continue
            source = entry.primary_location or list(entry.locations)[0]

            # Find targets (need the data)
            current_locations = set(entry.locations)
            targets = self._get_replication_targets(
                exclude=list(current_locations),
                count=min_factor - len(current_locations),
            )

            if not targets:
                continue

            plan = SyncPlan(
                source_node=source,
                target_nodes=targets,
                entries=[entry],
                priority=SyncPriority.LOW,
                reason=f"replication:{entry.path}:{len(current_locations)}/{min_factor}",
            )

            plans.append(plan)
            self._stats.record_plan(plan)
            self._stats.replication_plans += 1

        return plans

    def plan_orphan_recovery(
        self,
        source_node: str,
        config_key: str | None = None,
        game_ids: list[str] | None = None,
    ) -> SyncPlan | None:
        """Plan urgent sync for orphan game recovery.

        Orphan games are games on ephemeral nodes that haven't been
        synced to stable storage yet.

        Args:
            source_node: Node with orphan games
            config_key: Optional config filter
            game_ids: Optional specific game IDs

        Returns:
            SyncPlan for orphan recovery
        """
        # Get entries from source node
        entries = self._catalog.get_entries_on_node(source_node)
        if not entries:
            return None

        # Filter by config and type
        if config_key:
            entries = [e for e in entries if e.config_key == config_key]
        entries = [e for e in entries if e.data_type == DataType.GAMES]

        # Filter to under-replicated
        entries = [
            e for e in entries
            if self._catalog.get_replication_factor(e.path) < self._config.min_replication_factor
        ]

        if not entries:
            return None

        # Get stable targets (coordinator + training nodes)
        targets = self._get_stable_targets(exclude=[source_node])
        if not targets:
            targets = [self._node_id]  # Fallback to self

        # Check if source is ephemeral for priority boost
        is_ephemeral = self._is_ephemeral_node(source_node)
        priority = (
            SyncPriority.CRITICAL if is_ephemeral
            else SyncPriority.HIGH
        )

        return SyncPlan(
            source_node=source_node,
            target_nodes=targets[:3],
            entries=entries[:50],  # Limit for urgency
            priority=priority,
            reason=f"orphan_recovery:{config_key or 'unknown'}",
            transport_preference=[Transport.RSYNC, Transport.SCP, Transport.BASE64_SSH],
            deadline=time.time() + self._config.ephemeral_rescue_deadline_seconds,
            config_key=config_key,
        )

    # =========================================================================
    # Plan Execution
    # =========================================================================

    async def submit_plan(self, plan: SyncPlan) -> None:
        """Submit a plan for execution.

        Plans are added to the priority queue and executed by the
        background execution loop.

        Args:
            plan: SyncPlan to execute
        """
        async with self._plan_lock:
            heapq.heappush(self._pending_plans, plan)
            self._stats.record_plan(plan)

        logger.info(
            f"[SyncPlanner] Submitted plan: {plan.reason} "
            f"(priority={plan.priority.name}, targets={len(plan.target_nodes)})"
        )

    async def execute_plan(self, plan: SyncPlan) -> bool:
        """Execute a sync plan immediately.

        Args:
            plan: SyncPlan to execute

        Returns:
            True if successful
        """
        plan.started_at = time.time()

        if plan.is_expired:
            plan.error = "deadline_expired"
            self._stats.record_execution(plan, success=False)
            return False

        try:
            # Execute transfers for each target
            success_count = 0
            for target in plan.target_nodes:
                for entry in plan.entries:
                    result = await self._transport.transfer_file(
                        source_node=plan.source_node,
                        target_node=target,
                        source_path=entry.path,
                        target_path=entry.path,
                        size_bytes=entry.size_bytes,
                    )

                    if result.success:
                        success_count += 1
                        # Update catalog
                        self._catalog.mark_synced(entry.path, target)

            plan.completed_at = time.time()
            plan.success = success_count > 0

            self._stats.record_execution(plan, success=plan.success)

            if plan.success:
                logger.info(
                    f"[SyncPlanner] Plan completed: {plan.reason} "
                    f"({success_count} transfers)"
                )
            else:
                logger.warning(
                    f"[SyncPlanner] Plan failed: {plan.reason}"
                )

            return plan.success

        except Exception as e:
            plan.completed_at = time.time()
            plan.success = False
            plan.error = str(e)
            self._stats.record_execution(plan, success=False)
            logger.error(f"[SyncPlanner] Plan execution error: {e}")
            return False

    async def _execution_loop(self) -> None:
        """Background loop that executes pending plans."""
        while self._running:
            try:
                plan = None
                async with self._plan_lock:
                    if self._pending_plans:
                        plan = heapq.heappop(self._pending_plans)

                if plan:
                    await self.execute_plan(plan)
                else:
                    await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[SyncPlanner] Execution loop error: {e}")
                await asyncio.sleep(5.0)

    async def _replication_check_loop(self) -> None:
        """Background loop that checks replication requirements."""
        while self._running:
            try:
                await asyncio.sleep(self._config.replication_check_interval)

                if not self._running:
                    break

                # Generate replication plans
                plans = self.plan_replication()
                for plan in plans:
                    await self.submit_plan(plan)

                if plans:
                    logger.info(
                        f"[SyncPlanner] Replication check: {len(plans)} plans created"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[SyncPlanner] Replication loop error: {e}")
                await asyncio.sleep(60.0)

    # =========================================================================
    # Node Selection Helpers
    # =========================================================================

    def _load_node_capabilities(self) -> None:
        """Load node capabilities from cluster config."""
        if self._capabilities_loaded:
            return

        try:
            from app.config.cluster_config import get_cluster_nodes

            nodes = get_cluster_nodes()
            for name, node in nodes.items():
                self._node_capabilities[name] = {
                    "is_gpu": node.is_gpu_node if hasattr(node, "is_gpu_node") else False,
                    "is_training": (
                        node.role == "training"
                        if hasattr(node, "role")
                        else False
                    ),
                    "is_coordinator": (
                        node.is_coordinator
                        if hasattr(node, "is_coordinator")
                        else False
                    ),
                    "is_ephemeral": (
                        node.provider in ("vast", "runpod")
                        if hasattr(node, "provider")
                        else False
                    ),
                    "provider": node.provider if hasattr(node, "provider") else "unknown",
                }
            self._capabilities_loaded = True
        except Exception as e:
            logger.debug(f"[SyncPlanner] Could not load node capabilities: {e}")

    def _get_training_targets(self, exclude: list[str] | None = None) -> list[str]:
        """Get training nodes as sync targets."""
        self._load_node_capabilities()
        exclude_set = set(exclude or [])

        targets = [
            node_id for node_id, caps in self._node_capabilities.items()
            if node_id not in exclude_set
            and caps.get("is_training") or caps.get("is_gpu")
        ]

        return targets[:self._config.max_targets_per_plan]

    def _get_gpu_targets(self, exclude: list[str] | None = None) -> list[str]:
        """Get GPU nodes as sync targets."""
        self._load_node_capabilities()
        exclude_set = set(exclude or [])

        targets = [
            node_id for node_id, caps in self._node_capabilities.items()
            if node_id not in exclude_set
            and caps.get("is_gpu")
        ]

        return targets

    def _get_stable_targets(self, exclude: list[str] | None = None) -> list[str]:
        """Get stable (non-ephemeral) nodes as sync targets."""
        self._load_node_capabilities()
        exclude_set = set(exclude or [])

        targets = [
            node_id for node_id, caps in self._node_capabilities.items()
            if node_id not in exclude_set
            and not caps.get("is_ephemeral")
            and not caps.get("is_coordinator")
        ]

        return targets

    def _get_replication_targets(
        self,
        exclude: list[str] | None = None,
        count: int = 3,
    ) -> list[str]:
        """Get nodes for increasing replication factor."""
        self._load_node_capabilities()
        exclude_set = set(exclude or [])

        # Prefer stable nodes, then GPU nodes, then any
        stable = [
            node_id for node_id, caps in self._node_capabilities.items()
            if node_id not in exclude_set
            and not caps.get("is_ephemeral")
            and not caps.get("is_coordinator")
        ]

        return stable[:count]

    def _find_sources_for_entries(
        self,
        entries: list[DataEntry],
    ) -> list[str]:
        """Find nodes that have the given entries."""
        source_counts: dict[str, int] = {}

        for entry in entries:
            for loc in entry.locations:
                source_counts[loc] = source_counts.get(loc, 0) + 1

        # Sort by count (nodes with most entries first)
        sorted_sources = sorted(
            source_counts.keys(),
            key=lambda n: source_counts[n],
            reverse=True,
        )

        return sorted_sources

    def _is_ephemeral_node(self, node_id: str) -> bool:
        """Check if a node is ephemeral (Vast.ai, spot instance)."""
        self._load_node_capabilities()
        caps = self._node_capabilities.get(node_id, {})
        return caps.get("is_ephemeral", False)

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Check health status of SyncPlanner.

        Returns:
            HealthCheckResult with planner status
        """
        status = CoordinatorStatus.RUNNING if self._running else CoordinatorStatus.STOPPED
        message = ""
        errors = 0

        # Check catalog health
        try:
            catalog_health = self._catalog.health_check()
            if not catalog_health.healthy:
                status = CoordinatorStatus.DEGRADED
                message = f"Catalog: {catalog_health.message}"
        except Exception as e:
            errors += 1
            status = CoordinatorStatus.DEGRADED
            message = f"Catalog error: {e}"

        # Check for plan backlog
        pending_count = len(self._pending_plans)
        if pending_count > 100:
            status = CoordinatorStatus.DEGRADED
            message = f"Plan backlog: {pending_count}"

        # Check execution stats
        if self._stats.plans_executed > 10:
            success_rate = self._stats.plans_succeeded / self._stats.plans_executed
            if success_rate < 0.5:
                status = CoordinatorStatus.DEGRADED
                message = f"Low success rate: {success_rate:.1%}"

        return HealthCheckResult(
            healthy=status == CoordinatorStatus.RUNNING,
            status=status,
            message=message,
            details={
                "running": self._running,
                "pending_plans": pending_count,
                "stats": self._stats.to_dict(),
                "errors_count": errors,
            },
        )

    def get_status(self) -> dict[str, Any]:
        """Get current planner status."""
        return {
            "node_id": self._node_id,
            "running": self._running,
            "pending_plans": len(self._pending_plans),
            "config": {
                "min_replication": self._config.min_replication_factor,
                "target_replication": self._config.target_replication_factor,
                "training_deadline_seconds": self._config.training_deadline_seconds,
            },
            "stats": self._stats.to_dict(),
        }


# =============================================================================
# Module-Level Singleton
# =============================================================================

_sync_planner: SyncPlanner | None = None


def get_sync_planner() -> SyncPlanner:
    """Get the singleton SyncPlanner instance."""
    global _sync_planner
    if _sync_planner is None:
        _sync_planner = SyncPlanner()
    return _sync_planner


def reset_sync_planner() -> None:
    """Reset the singleton (for testing)."""
    global _sync_planner
    _sync_planner = None
