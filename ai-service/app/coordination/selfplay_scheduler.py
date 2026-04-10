"""SelfplayScheduler - Priority-based selfplay allocation across cluster.

This module provides intelligent scheduling of selfplay jobs across the cluster,
prioritizing configurations based on:
- Data staleness (fresher data gets lower priority)
- ELO improvement velocity (fast-improving configs get more resources)
- Training pipeline needs (configs waiting for training get boosted)
- Node capabilities (allocate based on GPU power)

Architecture:
    SelfplayScheduler
    ├── PriorityUpdateMixin: Priority calculation pipeline
    ├── FreshnessFetcherMixin: Track data age per config
    ├── AllocationMixin: Node allocation orchestration
    ├── CoreEventHandlerMixin: Event handler implementations
    ├── DataProviderMixin: Async data source access
    ├── NodeTargetingMixin: Node targeting logic
    ├── IdleWorkInjectionMixin: Idle node work injection
    ├── ArchitectureTrackerMixin: Architecture boost tracking
    ├── SelfplayVelocityMixin: Elo velocity tracking
    ├── SelfplayQualitySignalMixin: Quality signal handling
    └── SelfplayHealthMonitorMixin: P2P health monitoring

Usage:
    from app.coordination.selfplay_scheduler import (
        SelfplayScheduler,
        get_selfplay_scheduler,
    )

    scheduler = get_selfplay_scheduler()

    # Get priority-ordered configs for selfplay
    priorities = await scheduler.get_priority_configs()

    # Allocate games across nodes
    allocation = await scheduler.allocate_selfplay_batch(games_per_config=500)

    # Start jobs based on allocation
    for config, nodes in allocation.items():
        for node_id, num_games in nodes.items():
            await start_selfplay_job(node_id, config, num_games)

December 2025: Created as part of strategic integration plan.
February 2026: Priority calculation methods extracted to PriorityUpdateMixin.
"""

from __future__ import annotations

__all__ = [
    # Classes
    "ConfigPriority",
    "DynamicWeights",  # December 29, 2025: Now exported
    "NodeCapability",
    "SelfplayScheduler",
    # Constants (also available via SelfplayPriorityWeightDefaults in coordination_defaults)
    "ALL_CONFIGS",
    "CURRICULUM_WEIGHT",
    "DATA_DEFICIT_WEIGHT",
    "DATA_STARVATION_CRITICAL_MULTIPLIER",
    "DATA_STARVATION_CRITICAL_THRESHOLD",
    "DATA_STARVATION_EMERGENCY_MULTIPLIER",
    "DATA_STARVATION_EMERGENCY_THRESHOLD",
    "DATA_STARVATION_ULTRA_MULTIPLIER",
    "DATA_STARVATION_ULTRA_THRESHOLD",
    "DATA_POVERTY_MULTIPLIER",
    "DATA_POVERTY_THRESHOLD",
    "DATA_WARNING_MULTIPLIER",
    "DATA_WARNING_THRESHOLD",
    "DEFAULT_GAMES_PER_CONFIG",
    "DEFAULT_TRAINING_SAMPLES_TARGET",
    "DYNAMIC_WEIGHT_BOUNDS",  # December 29, 2025: Now exported
    "ELO_VELOCITY_WEIGHT",
    "EXPLORATION_BOOST_WEIGHT",
    "FRESH_DATA_THRESHOLD",
    "IMPROVEMENT_BOOST_WEIGHT",
    "LARGE_BOARD_TARGET_MULTIPLIER",
    "MAX_STALENESS_HOURS",
    "MIN_GAMES_PER_ALLOCATION",
    "PRIORITY_OVERRIDE_MULTIPLIERS",
    "QUALITY_WEIGHT",  # December 29, 2025: Now exported
    "SAMPLES_PER_GAME_BY_BOARD",
    "STALE_DATA_THRESHOLD",
    "STALENESS_WEIGHT",
    "TRAINING_NEED_WEIGHT",
    "VOI_WEIGHT",
    "VOI_SAMPLE_COST_BY_BOARD",
    # Functions
    "get_selfplay_scheduler",
    "reset_selfplay_scheduler",
    # New Dec 2025
    "get_priority_configs_sync",
]

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from app.config.thresholds import (
    SELFPLAY_GAMES_PER_NODE,
    is_ephemeral_node,
    get_gpu_weight,
)

from app.coordination.priority_calculator import (
    ALL_CONFIGS,
    PLAYER_COUNT_ALLOCATION_MULTIPLIER,
    PriorityCalculator,
    PRIORITY_OVERRIDE_MULTIPLIERS,
    SAMPLES_PER_GAME_BY_BOARD,
    VOI_SAMPLE_COST_BY_BOARD,
)
from app.coordination.node_allocator import NodeCapability
from app.coordination.budget_calculator import (
    parse_config_key,
)
from app.coordination.protocols import HealthCheckResult
from app.coordination.handler_base import HandlerBase
from app.coordination.event_handler_utils import extract_config_key
from app.coordination.selfplay_health_monitor import SelfplayHealthMonitorMixin
# Import from module (not package) to avoid circular import
from app.coordination.selfplay.quality_signal_handler import SelfplayQualitySignalMixin
from app.coordination.selfplay.velocity_mixin import SelfplayVelocityMixin
from app.coordination.selfplay.freshness_fetcher import FreshnessFetcherMixin
from app.coordination.selfplay.core_event_mixin import CoreEventHandlerMixin
from app.coordination.selfplay.allocation_mixin import AllocationMixin
from app.coordination.selfplay.priority_update_mixin import PriorityUpdateMixin

from app.coordination.config_state_cache import ConfigStateCache
from app.coordination.scheduler_metrics import SchedulerMetricsCollector
from app.coordination.interfaces import IBackpressureMonitor
from app.config.cluster_config import get_cluster_nodes
from app.coordination.node_circuit_breaker import get_node_circuit_registry
from app.coordination.selfplay.allocation_engine import (
    AllocationContext,
    AllocationEngine,
    AllocationResult,
)
from app.coordination.selfplay.data_providers import DataProviderMixin
from app.coordination.selfplay.node_targeting import NodeTargetingMixin
from app.coordination.selfplay.idle_injection import IdleWorkInjectionMixin
from app.coordination.selfplay.architecture_tracker_mixin import ArchitectureTrackerMixin

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration (re-exported from coordination_defaults for backward compat)
# For runtime tuning: export RINGRIFT_STALENESS_WEIGHT=0.40 etc.
# =============================================================================
from app.config.coordination_defaults import SelfplayPriorityWeightDefaults, SelfplayAllocationDefaults

_priority_weight_defaults = SelfplayPriorityWeightDefaults()

# Priority weights (base values, adjusted dynamically)
STALENESS_WEIGHT = _priority_weight_defaults.STALENESS_WEIGHT
ELO_VELOCITY_WEIGHT = _priority_weight_defaults.ELO_VELOCITY_WEIGHT
TRAINING_NEED_WEIGHT = _priority_weight_defaults.TRAINING_NEED_WEIGHT
EXPLORATION_BOOST_WEIGHT = _priority_weight_defaults.EXPLORATION_BOOST_WEIGHT
CURRICULUM_WEIGHT = _priority_weight_defaults.CURRICULUM_WEIGHT
IMPROVEMENT_BOOST_WEIGHT = _priority_weight_defaults.IMPROVEMENT_BOOST_WEIGHT
DATA_DEFICIT_WEIGHT = _priority_weight_defaults.DATA_DEFICIT_WEIGHT
QUALITY_WEIGHT = _priority_weight_defaults.QUALITY_WEIGHT
VOI_WEIGHT = _priority_weight_defaults.VOI_WEIGHT
DYNAMIC_WEIGHT_BOUNDS = _priority_weight_defaults.get_weight_bounds()

# Dynamic weight adjustment thresholds
VOI_ELO_TARGET = _priority_weight_defaults.VOI_ELO_TARGET
IDLE_GPU_HIGH_THRESHOLD = _priority_weight_defaults.IDLE_GPU_HIGH_THRESHOLD
IDLE_GPU_LOW_THRESHOLD = _priority_weight_defaults.IDLE_GPU_LOW_THRESHOLD
TRAINING_QUEUE_HIGH_THRESHOLD = _priority_weight_defaults.TRAINING_QUEUE_HIGH_THRESHOLD
CONFIGS_AT_TARGET_THRESHOLD = _priority_weight_defaults.CONFIGS_AT_TARGET_THRESHOLD
ELO_HIGH_THRESHOLD = _priority_weight_defaults.ELO_HIGH_THRESHOLD
ELO_MEDIUM_THRESHOLD = _priority_weight_defaults.ELO_MEDIUM_THRESHOLD
TARGET_GAMES_FOR_2000_ELO = _priority_weight_defaults.TARGET_GAMES_FOR_2000_ELO
LARGE_BOARD_TARGET_MULTIPLIER = _priority_weight_defaults.LARGE_BOARD_TARGET_MULTIPLIER

# Data starvation tiers (ULTRA > EMERGENCY > CRITICAL > WARNING)
DATA_STARVATION_ULTRA_THRESHOLD = _priority_weight_defaults.DATA_STARVATION_ULTRA_THRESHOLD
DATA_STARVATION_EMERGENCY_THRESHOLD = _priority_weight_defaults.DATA_STARVATION_EMERGENCY_THRESHOLD
DATA_STARVATION_CRITICAL_THRESHOLD = _priority_weight_defaults.DATA_STARVATION_CRITICAL_THRESHOLD
DATA_STARVATION_ULTRA_MULTIPLIER = _priority_weight_defaults.DATA_STARVATION_ULTRA_MULTIPLIER
DATA_STARVATION_EMERGENCY_MULTIPLIER = _priority_weight_defaults.DATA_STARVATION_EMERGENCY_MULTIPLIER
DATA_STARVATION_CRITICAL_MULTIPLIER = _priority_weight_defaults.DATA_STARVATION_CRITICAL_MULTIPLIER
DATA_POVERTY_THRESHOLD = _priority_weight_defaults.DATA_POVERTY_THRESHOLD
DATA_POVERTY_MULTIPLIER = _priority_weight_defaults.DATA_POVERTY_MULTIPLIER
DATA_WARNING_THRESHOLD = _priority_weight_defaults.DATA_WARNING_THRESHOLD
DATA_WARNING_MULTIPLIER = _priority_weight_defaults.DATA_WARNING_MULTIPLIER

# Staleness and allocation thresholds
DEFAULT_TRAINING_SAMPLES_TARGET = 50000
FRESH_DATA_THRESHOLD = _priority_weight_defaults.FRESH_DATA_THRESHOLD
STALE_DATA_THRESHOLD = _priority_weight_defaults.STALE_DATA_THRESHOLD
MAX_STALENESS_HOURS = _priority_weight_defaults.MAX_STALENESS_HOURS
DEFAULT_GAMES_PER_CONFIG = SelfplayAllocationDefaults.GAMES_PER_CONFIG
MIN_GAMES_PER_ALLOCATION = SelfplayAllocationDefaults.MIN_GAMES_PER_ALLOCATION
MIN_MEMORY_GB_FOR_TASKS = SelfplayAllocationDefaults.MIN_MEMORY_GB
DISK_WARNING_THRESHOLD = SelfplayAllocationDefaults.DISK_WARNING_THRESHOLD
MEMORY_WARNING_THRESHOLD = SelfplayAllocationDefaults.MEMORY_WARNING_THRESHOLD

from app.coordination.selfplay_priority_types import ConfigPriority, DynamicWeights


class SelfplayScheduler(
    PriorityUpdateMixin,
    FreshnessFetcherMixin,
    CoreEventHandlerMixin,
    AllocationMixin,
    DataProviderMixin,
    NodeTargetingMixin,
    IdleWorkInjectionMixin,
    ArchitectureTrackerMixin,
    SelfplayVelocityMixin,
    SelfplayQualitySignalMixin,
    SelfplayHealthMonitorMixin,
    HandlerBase,
):
    """Priority-based selfplay scheduler across cluster nodes.

    Responsibilities:
    - Track data freshness per configuration
    - Calculate priority scores for each config
    - Allocate selfplay games based on node capabilities
    - Integrate with feedback loop signals
    - Calculate target selfplay jobs per node (Dec 2025)

    December 30, 2025: Now inherits from HandlerBase for unified event handling,
    singleton management, and health check patterns.

    January 2026 Sprint 17.4: P2P health handlers extracted to SelfplayHealthMonitorMixin.
    January 2026 Sprint 17.9: Velocity/Elo handlers extracted to SelfplayVelocityMixin.
    """

    def __init__(
        self,
        # Optional callbacks for external integrations (Dec 2025)
        # These enable full delegation from P2P orchestrator
        get_cluster_elo_fn: Callable[[], dict[str, Any]] | None = None,
        load_curriculum_weights_fn: Callable[[], dict[str, float]] | None = None,
        get_board_priority_overrides_fn: Callable[[], dict[str, int]] | None = None,
        # Backpressure callbacks
        should_stop_production_fn: Callable[..., bool] | None = None,
        should_throttle_production_fn: Callable[..., bool] | None = None,
        get_throttle_factor_fn: Callable[..., float] | None = None,
        # Resource targeting callbacks
        record_utilization_fn: Callable[..., None] | None = None,
        get_host_targets_fn: Callable[[str], Any] | None = None,
        get_target_job_count_fn: Callable[..., int] | None = None,
        should_scale_up_fn: Callable[..., tuple[bool, str]] | None = None,
        should_scale_down_fn: Callable[..., tuple[bool, int, str]] | None = None,
        # Hardware-aware limits
        get_max_selfplay_for_node_fn: Callable[..., int] | None = None,
        get_hybrid_selfplay_limits_fn: Callable[..., dict[str, int]] | None = None,
        # Safeguard callback
        is_emergency_active_fn: Callable[[], bool] | None = None,
        # Dependency injection for breaking circular dependencies (Dec 2025)
        backpressure_monitor: Optional[IBackpressureMonitor] = None,
        # Verbosity
        verbose: bool = False,
    ):
        """Initialize selfplay scheduler with optional callback injections.

        All callback parameters enable full delegation from P2P orchestrator
        and break circular dependencies. See class docstring for usage.
        """
        # December 30, 2025: Initialize HandlerBase for unified event handling
        # SelfplayScheduler is primarily event-driven, so cycle_interval is long
        # (priority refresh happens on events, not periodic cycles)
        super().__init__(
            name="selfplay_scheduler",
            cycle_interval=300.0,  # 5 min - priorities refreshed on events, not cycles
            dedup_enabled=True,
        )

        # January 2026 Sprint 17.4: Initialize health monitor state from mixin
        self._init_health_monitor_state()

        # Store callbacks (Dec 2025)
        self._get_cluster_elo_fn = get_cluster_elo_fn
        self._load_curriculum_weights_fn = load_curriculum_weights_fn
        self._get_board_priority_overrides_fn = get_board_priority_overrides_fn
        self._should_stop_production_fn = should_stop_production_fn
        self._should_throttle_production_fn = should_throttle_production_fn
        self._get_throttle_factor_fn = get_throttle_factor_fn
        self._record_utilization_fn = record_utilization_fn
        self._get_host_targets_fn = get_host_targets_fn
        self._get_target_job_count_fn = get_target_job_count_fn
        self._should_scale_up_fn = should_scale_up_fn
        self._should_scale_down_fn = should_scale_down_fn
        self._get_max_selfplay_for_node_fn = get_max_selfplay_for_node_fn
        self._get_hybrid_selfplay_limits_fn = get_hybrid_selfplay_limits_fn
        self._is_emergency_active_fn = is_emergency_active_fn
        self._verbose = verbose

        # Priority tracking
        self._config_priorities: dict[str, ConfigPriority] = {
            cfg: ConfigPriority(config_key=cfg) for cfg in ALL_CONFIGS
        }

        # Node tracking
        self._node_capabilities: dict[str, NodeCapability] = {}

        # Timing
        self._last_priority_update = 0.0
        self._priority_update_interval = 15.0  # Dec 2025: Update every 15s (was 60s)

        # Node capability refresh timing.
        # NOTE: ClusterMonitor probes can be expensive (SSH/subprocess). We rate-limit
        # them and treat externally pre-seeded capabilities as already up-to-date.
        self._last_node_capability_update = 0.0
        self._node_capability_update_interval = 60.0

        # Event subscription
        self._subscribed = False

        # Dec 29, 2025: Dynamic priority weights (adjusted based on cluster state)
        self._dynamic_weights = DynamicWeights()
        self._last_dynamic_weights_update = 0.0
        self._dynamic_weights_update_interval = 60.0  # Update every 60 seconds

        # Dec 29, 2025 - Phase 2: Elo velocity tracking
        # Track Elo history per config: list of (timestamp, elo) tuples
        # Used to compute Elo/hour velocity for priority adjustment
        self._elo_history: dict[str, list[tuple[float, float]]] = {}
        self._elo_velocity: dict[str, float] = {}  # Computed Elo change per hour

        # January 2026 Sprint 10: Stall detection for PLATEAU_DETECTED emission
        # Track consecutive low-velocity updates per config
        self._low_velocity_count: dict[str, int] = {}
        self._last_plateau_emission: dict[str, float] = {}  # Avoid spamming events

        # January 2026 Sprint 10: Diversity tracking for opponent variety maximization
        # Track opponent types seen per config: config_key -> set of opponent types
        self._opponent_types_by_config: dict[str, set[str]] = {}
        # Computed diversity scores: config_key -> diversity score (0.0=low, 1.0=high)
        self._diversity_scores: dict[str, float] = {}
        # Maximum opponent types for diversity calculation (8 = full diversity)
        self._max_opponent_types = 8

        # Jan 5, 2026: Game counts cache for real-time NEW_GAMES_AVAILABLE updates
        # Initialized from DB at startup, updated by events for instant feedback
        self._cached_game_counts: dict[str, int] = {}

        # Dec 30, 2025: Extracted quality cache class (reduces code, enables testing)
        # ConfigStateCache handles TTL, invalidation, and daemon integration
        self._quality_cache = ConfigStateCache(
            ttl_seconds=30.0,
            default_quality=0.7,
            quality_provider=self._fetch_quality_from_daemon,
        )

        # January 2026 Phase 2: Cluster-wide game count cache
        # Caches cluster manifest data to avoid repeated lookups during priority calculation
        self._cluster_game_counts: dict[str, int] = {}
        self._cluster_game_counts_last_update: float = 0.0
        self._cluster_game_counts_ttl: float = 60.0  # Refresh every 60 seconds

        # Lazy dependencies
        self._training_freshness = None
        self._cluster_manifest = None
        # Injected backpressure monitor (breaks circular dep with backpressure.py)
        self._backpressure_monitor: Optional[IBackpressureMonitor] = backpressure_monitor

        # January 13, 2026: Memory pressure constraint flag
        # When memory is CRITICAL or EMERGENCY, pause selfplay allocation
        # to prevent OOM and allow cleanup daemons time to free space.
        self._memory_constrained: bool = False
        self._memory_constraint_source: str = ""  # For logging which node/source triggered

        # Jan 5, 2026: Idle node work injection state
        # Tracks when each node became idle (first seen with current_jobs=0)
        # If node is idle for > IDLE_THRESHOLD_SECONDS, inject priority work
        # Jan 5, 2026: Reduced from 300s (5 min) to 120s (2 min) for faster work injection.
        # Nodes were waiting too long before getting work during P2P recovery periods.
        self._node_idle_since: dict[str, float] = {}
        self._idle_threshold_seconds = float(
            os.environ.get("RINGRIFT_IDLE_NODE_THRESHOLD_SECONDS", "120")  # 2 min
        )
        self._last_idle_injection = 0.0
        self._idle_injection_cooldown = 60.0  # Don't spam work injection

        # Load priority overrides from config (Dec 2025)
        self._load_priority_overrides()

        # Dec 30, 2025: Extracted metrics collector class (reduces code, enables testing)
        # SchedulerMetricsCollector handles rolling window, throughput calculation
        self._metrics_collector = SchedulerMetricsCollector(window_seconds=3600.0)

        # December 29, 2025: PriorityCalculator for delegated score computation
        # Callbacks are bound methods so PriorityCalculator can access scheduler state
        self._priority_calculator = PriorityCalculator(
            dynamic_weights=self._dynamic_weights,
            get_quality_score_fn=self._get_config_data_quality,
            get_elo_velocity_fn=self.get_elo_velocity,
            get_cascade_priority_fn=self._get_cascade_priority,
            data_starvation_ultra_threshold=DATA_STARVATION_ULTRA_THRESHOLD,
            data_starvation_emergency_threshold=DATA_STARVATION_EMERGENCY_THRESHOLD,
            data_starvation_critical_threshold=DATA_STARVATION_CRITICAL_THRESHOLD,
            data_starvation_warning_threshold=DATA_WARNING_THRESHOLD,
            data_starvation_ultra_multiplier=DATA_STARVATION_ULTRA_MULTIPLIER,
            data_starvation_emergency_multiplier=DATA_STARVATION_EMERGENCY_MULTIPLIER,
            data_starvation_critical_multiplier=DATA_STARVATION_CRITICAL_MULTIPLIER,
            data_starvation_warning_multiplier=DATA_WARNING_MULTIPLIER,
        )

    def _load_priority_overrides(self) -> None:
        """Load board_priority_overrides from unified_loop.yaml.

        Maps priority levels (0-3) to multipliers for the priority score.
        0 = CRITICAL (3x), 1 = HIGH (2x), 2 = MEDIUM (1.25x), 3 = LOW (1x)
        """
        config_paths = [
            Path(__file__).parent.parent.parent / "config" / "unified_loop.yaml",
            Path("config/unified_loop.yaml"),
            Path("/etc/ringrift/unified_loop.yaml"),
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = yaml.safe_load(f)

                    overrides = config.get("selfplay", {}).get("board_priority_overrides", {})
                    if overrides:
                        for config_key, priority_level in overrides.items():
                            if config_key in self._config_priorities:
                                self._config_priorities[config_key].priority_override = priority_level
                                logger.debug(
                                    f"[SelfplayScheduler] Priority override: {config_key} = {priority_level}"
                                )
                        logger.info(
                            f"[SelfplayScheduler] Loaded {len(overrides)} priority overrides from config"
                        )
                    return
                except (yaml.YAMLError, OSError, KeyError, AttributeError) as e:
                    logger.warning(f"[SelfplayScheduler] Failed to load config {config_path}: {e}")

    # =========================================================================
    # Node Allocation
    # =========================================================================

    async def allocate_selfplay_batch(
        self,
        games_per_config: int = DEFAULT_GAMES_PER_CONFIG,
        max_configs: int = 15,
    ) -> dict[str, dict[str, int]]:
        """Allocate selfplay games across cluster nodes.

        Args:
            games_per_config: Target games per config
            max_configs: Maximum configs to allocate (default: 15 for better coverage)

        Returns:
            Dict mapping config_key to {node_id: num_games}

        December 28, 2025: Changed default from 6 to 12 to include all board/player configs.
        January 2026 Sprint 17.9: Delegates allocation logic to AllocationEngine for testability.
        January 2026 Session 17.35: Increased from 12 to 15 for +5-10% config coverage.
        """
        # January 13, 2026: Check memory constraint before allocating
        # Memory pressure events set this flag to prevent OOM during critical load
        if self._memory_constrained:
            logger.info(
                f"[SelfplayScheduler] Skipping allocation due to memory constraint "
                f"({self._memory_constraint_source})"
            )
            return {}

        # Check backpressure before allocating (Dec 2025)
        bp_signal = None
        try:
            # Use injected monitor if available, otherwise lazy import
            # (Dec 2025: IBackpressureMonitor protocol enables dependency injection)
            if self._backpressure_monitor is None:
                from app.coordination.backpressure import get_backpressure_monitor
                self._backpressure_monitor = get_backpressure_monitor()
            bp_signal = await self._backpressure_monitor.get_signal()

            if bp_signal.should_pause:
                logger.warning(
                    f"[SelfplayScheduler] Backpressure pause: pressure={bp_signal.overall_pressure:.2f}"
                )
                return {}
        except Exception as e:
            # Don't let backpressure failures block allocation
            logger.warning(f"[SelfplayScheduler] Backpressure check failed (continuing): {e}")

        # Get priority configs
        priorities = await self.get_priority_configs(top_n=max_configs)

        # Get available nodes
        await self._update_node_capabilities()

        # January 2026 Sprint 17.9: Build allocation context from current cluster state
        context = AllocationContext(
            unhealthy_nodes=getattr(self, "_unhealthy_nodes", set()),
            cluster_health_factor=getattr(self, "_cluster_health_factor", 1.0),
            backpressure_signal=bp_signal,
            demoted_nodes=getattr(self, "_demoted_nodes", set()),
            enforce_4p_minimums=True,
        )

        # January 2026 Sprint 17.9: Create AllocationEngine with current state snapshots
        # Engine receives copies of mutable state to ensure deterministic allocation
        engine = AllocationEngine(
            config_priorities=self._config_priorities,
            node_capabilities=self._node_capabilities,
            metrics_collector=self._metrics_collector,
            emit_event_fn=self._safe_emit_allocation_event,
        )

        # Delegate allocation logic to engine
        # Engine handles: priority-based allocation, starvation floor, 4p minimums, metrics
        result = engine.allocate_selfplay_batch(
            priorities=priorities,
            games_per_config=games_per_config,
            context=context,
        )

        # Update scheduler state from result (priority tracking updated by engine)
        # Note: ConfigPriority.games_allocated and nodes_allocated are updated
        # inside engine.allocate_selfplay_batch() via the shared _config_priorities dict

        return result.allocations

    def _safe_emit_allocation_event(self, event_name: str, payload: dict[str, Any]) -> None:
        """Safely emit an allocation event via the scheduler's event system.

        This is passed to AllocationEngine to enable event emission without
        the engine needing direct access to the event router.

        January 2026 Sprint 17.9: Created for AllocationEngine integration.
        """
        try:
            from app.coordination.event_emission_helpers import safe_emit_event
            from app.distributed.data_events import DataEventType

            # Map string event name to DataEventType
            event_type = getattr(DataEventType, event_name, None)
            if event_type is not None:
                safe_emit_event(event_type, payload, source="selfplay_scheduler")
            else:
                logger.warning(f"[SelfplayScheduler] Unknown event type: {event_name}")
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Event emission failed (non-critical): {e}")

    # =========================================================================
    # External Boost Interface
    # =========================================================================

    def boost_config_allocation(self, config_key: str, multiplier: float = 1.5) -> bool:
        """Boost selfplay allocation for a specific configuration.

        December 2025: Called by TrainingTriggerDaemon when gauntlet evaluation
        shows poor performance, triggering additional selfplay to generate
        more training data for struggling configurations.

        Args:
            config_key: Configuration to boost (e.g., "hex8_2p")
            multiplier: Boost multiplier (default 1.5x, capped at 2.0)

        Returns:
            True if boost was applied, False if config not found
        """
        if config_key not in self._config_priorities:
            logger.warning(
                f"[SelfplayScheduler] Cannot boost unknown config: {config_key}"
            )
            return False

        priority = self._config_priorities[config_key]

        # Apply multiplier to exploration boost (capped at 2.0)
        old_boost = priority.exploration_boost
        priority.exploration_boost = min(2.0, priority.exploration_boost * multiplier)

        # Also boost momentum multiplier temporarily
        old_momentum = priority.momentum_multiplier
        priority.momentum_multiplier = min(1.5, priority.momentum_multiplier * 1.2)

        logger.info(
            f"[SelfplayScheduler] Boosted {config_key}: "
            f"exploration {old_boost:.2f}x → {priority.exploration_boost:.2f}x, "
            f"momentum {old_momentum:.2f}x → {priority.momentum_multiplier:.2f}x"
        )

        # Emit SELFPLAY_RATE_CHANGED if momentum changed by >20%
        if abs(priority.momentum_multiplier - old_momentum) / max(old_momentum, 0.01) > 0.20:
            change_percent = ((priority.momentum_multiplier - old_momentum) / old_momentum) * 100.0
            from app.coordination.event_emission_helpers import safe_emit_event

            safe_emit_event(
                "SELFPLAY_RATE_CHANGED",
                {
                    "config_key": config_key,
                    "old_rate": old_momentum,
                    "new_rate": priority.momentum_multiplier,
                    "change_percent": change_percent,
                    "reason": "config_boost",
                },
                log_after=f"[SelfplayScheduler] Emitted SELFPLAY_RATE_CHANGED for {config_key}: "
                f"{old_momentum:.2f} → {priority.momentum_multiplier:.2f} ({change_percent:+.1f}%)",
                log_level=logging.DEBUG,
                context="SelfplayScheduler.boost_config_priority",
            )

        # Force priority recalculation
        self._last_priority_update = 0.0

        return True

    def get_config_priority(self, config_key: str) -> ConfigPriority | None:
        """Get current priority state for a configuration.

        December 2025: Useful for monitoring and debugging priority decisions.

        Args:
            config_key: Configuration to query

        Returns:
            ConfigPriority object or None if not found
        """
        return self._config_priorities.get(config_key)

    # =========================================================================
    # Event Integration
    # =========================================================================

    async def _run_cycle(self) -> None:
        """Main work loop iteration (HandlerBase abstract method).

        December 30, 2025: SelfplayScheduler is primarily event-driven, so
        this cycle is minimal - just refreshes stale priority data periodically.
        The main work happens in response to events like SELFPLAY_COMPLETE,
        TRAINING_COMPLETED, etc.

        January 5, 2026: Added idle node work injection to utilize idle GPU
        nodes for underserved configs during backpressure periods.
        """
        # Periodic priority refresh (in case events are missed)
        current_time = time.time()
        if current_time - self._last_priority_update > self._priority_update_interval:
            try:
                await self._update_priorities()
            except Exception as e:
                self._record_error(f"Priority update failed: {e}", e)

        # Jan 5, 2026: Inject work for idle GPU nodes
        # This ensures idle resources are used for underserved configs
        try:
            await self.inject_work_for_idle_nodes()
        except Exception as e:
            # Non-critical - log and continue
            logger.debug(f"[SelfplayScheduler] Idle work injection failed: {e}")

        # January 5, 2026 (Phase 7.6): Periodic cleanup of stale unhealthy nodes
        # Runs every 10 cycles (~5-10 minutes) to restore nodes whose circuits
        # have closed but missed the CIRCUIT_RESET event.
        if self.stats.cycles_completed % 10 == 0:
            try:
                await self._cleanup_stale_unhealthy_nodes()
            except Exception as e:
                logger.debug(f"[SelfplayScheduler] Unhealthy node cleanup failed: {e}")

    async def _cleanup_stale_unhealthy_nodes(self) -> None:
        """Remove nodes from _unhealthy_nodes if their circuit breaker is closed.

        January 5, 2026 (Phase 7.6): Periodic safety net for nodes that recovered
        but missed the CIRCUIT_RESET event. Checks circuit breaker state and
        restores nodes that are healthy.

        Expected impact: Nodes recover automatically instead of staying excluded.
        """
        try:
            cb_registry = get_node_circuit_registry()
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.debug(f"[SelfplayScheduler] CB registry unavailable: {e}")
            return  # CB registry not available

        unhealthy = getattr(self, "_unhealthy_nodes", set())
        if not unhealthy:
            return

        to_restore: list[str] = []
        for node_id in list(unhealthy):
            # If circuit is closed (not open), node is healthy
            if not cb_registry.is_circuit_open(node_id):
                to_restore.append(node_id)

        for node_id in to_restore:
            self._unhealthy_nodes.discard(node_id)
            # Also clear from demoted sets
            if hasattr(self, "_demoted_nodes"):
                self._demoted_nodes.discard(node_id)
            logger.info(
                f"[SelfplayScheduler] Auto-restored {node_id} (circuit closed, "
                f"was in _unhealthy_nodes)"
            )

        if to_restore:
            logger.info(
                f"[SelfplayScheduler] Periodic cleanup restored {len(to_restore)} nodes: "
                f"{to_restore}"
            )

    def _get_event_subscriptions(self) -> dict[str, Callable]:
        """Return event_type -> handler mapping (HandlerBase pattern).

        December 30, 2025: Replaces subscribe_to_events() with declarative mapping.
        All event handlers are registered via this method when start() is called.
        """
        from app.coordination.event_router import DataEventType

        # Build subscriptions dict - core events always included
        subs: dict[str, Callable] = {
            # Core subscriptions (always active)
            DataEventType.SELFPLAY_COMPLETE.value: self._on_selfplay_complete,
            DataEventType.TRAINING_COMPLETED.value: self._on_training_complete,
            DataEventType.MODEL_PROMOTED.value: self._on_promotion_complete,
            DataEventType.SELFPLAY_TARGET_UPDATED.value: self._on_selfplay_target_updated,
            DataEventType.QUALITY_DEGRADED.value: self._on_quality_degraded,
            DataEventType.CURRICULUM_REBALANCED.value: self._on_curriculum_rebalanced,
            DataEventType.SELFPLAY_RATE_CHANGED.value: self._on_selfplay_rate_changed,
            DataEventType.TRAINING_BLOCKED_BY_QUALITY.value: self._on_training_blocked_by_quality,
        }

        # Optional subscriptions (check if event type exists)
        optional_events = [
            ("OPPONENT_MASTERED", self._on_opponent_mastered),
            ("TRAINING_EARLY_STOPPED", self._on_training_early_stopped),
            ("ELO_VELOCITY_CHANGED", self._on_elo_velocity_changed),
            ("EXPLORATION_BOOST", self._on_exploration_boost),
            # Jan 7, 2026: Quality-driven exploration adjustment
            ("EXPLORATION_ADJUSTED", self._on_exploration_adjusted),
            ("CURRICULUM_ADVANCED", self._on_curriculum_advanced),
            ("ADAPTIVE_PARAMS_CHANGED", self._on_adaptive_params_changed),
            ("LOW_QUALITY_DATA_WARNING", self._on_low_quality_warning),
            # P2P cluster health events
            ("NODE_UNHEALTHY", self._on_node_unhealthy),
            ("NODE_RECOVERED", self._on_node_recovered),
            ("NODE_ACTIVATED", self._on_node_recovered),  # Same handler
            ("P2P_NODE_DEAD", self._on_node_unhealthy),  # Same handler
            ("P2P_CLUSTER_UNHEALTHY", self._on_cluster_unhealthy),
            ("P2P_CLUSTER_HEALTHY", self._on_cluster_healthy),
            ("HOST_OFFLINE", self._on_host_offline),
            ("NODE_TERMINATED", self._on_host_offline),  # Same handler
            # Regression events
            ("REGRESSION_DETECTED", self._on_regression_detected),
            # Backpressure events
            ("BACKPRESSURE_ACTIVATED", self._on_backpressure_activated),
            ("BACKPRESSURE_RELEASED", self._on_backpressure_released),
            # Jan 2026 Sprint 10: Evaluation-specific backpressure
            ("EVALUATION_BACKPRESSURE", self._on_evaluation_backpressure),
            ("EVALUATION_BACKPRESSURE_RELEASED", self._on_backpressure_released),
            # Jan 2026: Work queue specific backpressure events
            ("WORK_QUEUE_BACKPRESSURE", self._on_work_queue_backpressure),
            ("WORK_QUEUE_BACKPRESSURE_RELEASED", self._on_work_queue_backpressure_released),
            ("NODE_OVERLOADED", self._on_node_overloaded),
            # Elo velocity tracking
            ("ELO_UPDATED", self._on_elo_updated),
            # Jan 5, 2026: Real-time game count updates for faster feedback
            ("NEW_GAMES_AVAILABLE", self._on_new_games_available),
            # Progress monitoring
            ("PROGRESS_STALL_DETECTED", self._on_progress_stall),
            ("PROGRESS_RECOVERED", self._on_progress_recovered),
            # Architecture updates
            ("ARCHITECTURE_WEIGHTS_UPDATED", self._on_architecture_weights_updated),
            # Quality feedback
            ("QUALITY_FEEDBACK_ADJUSTED", self._on_quality_feedback_adjusted),
            # Jan 2026: Multi-harness evaluation feedback for harness performance tracking
            ("MULTI_HARNESS_EVALUATION_COMPLETED", self._on_multi_harness_evaluation_completed),
            ("CROSS_CONFIG_TOURNAMENT_COMPLETED", self._on_cross_config_tournament_completed),
            # Dec 30, 2025: P2P restart resilience
            ("P2P_RESTARTED", self._on_p2p_restarted),
            # Jan 13, 2026: Memory pressure handling - pause selfplay when memory critical
            ("MEMORY_PRESSURE", self._on_memory_pressure),
            ("RESOURCE_CONSTRAINT", self._on_resource_constraint),
        ]

        for event_name, handler in optional_events:
            if hasattr(DataEventType, event_name):
                event_type = getattr(DataEventType, event_name)
                subs[event_type.value] = handler

        return subs

    # =========================================================================
    # Status & Metrics
    # =========================================================================

    def _record_allocation(self, games_allocated: int) -> None:
        """Record allocation metrics for rolling throughput tracking.

        Dec 30, 2025: Refactored to use SchedulerMetricsCollector.
        """
        self._metrics_collector.record_allocation(games_allocated)

    def _emit_allocation_updated(
        self,
        allocation: dict[str, dict[str, int]] | None,
        total_games: int,
        trigger: str,
        config_key: str | None = None,
    ) -> None:
        """Emit SELFPLAY_ALLOCATION_UPDATED event.

        Jan 2026: Delegated to selfplay.allocation_events module.

        Args:
            allocation: Dict of config_key -> {node_id: games} for batch allocations
            total_games: Total games in this allocation
            trigger: What caused this allocation (e.g., "allocate_batch", "exploration_boost")
            config_key: Specific config that changed (for single-config updates)
        """
        from app.coordination.selfplay.allocation_events import emit_allocation_updated

        emit_allocation_updated(
            allocation=allocation,
            total_games=total_games,
            trigger=trigger,
            config_key=config_key,
            config_priorities=self._config_priorities,
        )

    def _emit_starvation_alert(
        self,
        config_key: str,
        game_count: int,
        tier: str,
    ) -> None:
        """Emit DATA_STARVATION_CRITICAL event to trigger priority dispatch.

        Jan 2026: Delegated to selfplay.allocation_events module.

        Args:
            config_key: Config with starvation (e.g., "square19_3p")
            game_count: Current game count for this config
            tier: Starvation tier ("ULTRA", "EMERGENCY", "CRITICAL")
        """
        from app.coordination.selfplay.allocation_events import emit_starvation_alert

        emit_starvation_alert(
            config_key=config_key,
            game_count=game_count,
            tier=tier,
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get throughput metrics for monitoring.

        Dec 30, 2025: Refactored to use SchedulerMetricsCollector.
        """
        return self._metrics_collector.get_metrics()

    def get_status(self) -> dict[str, Any]:
        """Get scheduler status."""
        return {
            "subscribed": self._subscribed,
            "last_priority_update": self._last_priority_update,
            "node_count": len(self._node_capabilities),
            "overloaded_nodes": self.get_overloaded_nodes(),
            "config_priorities": {
                cfg: {
                    "priority_score": p.priority_score,
                    "staleness_hours": p.staleness_hours,
                    "elo_velocity": p.elo_velocity,
                    "exploration_boost": p.exploration_boost,
                    "curriculum_weight": p.curriculum_weight,
                    "games_allocated": p.games_allocated,
                }
                for cfg, p in self._config_priorities.items()
            },
        }

    def get_top_priorities(self, n: int = 5) -> list[dict[str, Any]]:
        """Get top N priority configurations with details."""
        sorted_configs = sorted(
            self._config_priorities.values(),
            key=lambda p: -p.priority_score,
        )

        return [
            {
                "config": p.config_key,
                "priority": p.priority_score,
                "staleness_hours": p.staleness_hours,
                "elo_velocity": p.elo_velocity,
                "exploration_boost": p.exploration_boost,
                "curriculum_weight": p.curriculum_weight,
            }
            for p in sorted_configs[:n]
        ]

    def health_check(self) -> HealthCheckResult:
        """Check scheduler health.

        December 30, 2025: Now incorporates HandlerBase error tracking metrics
        alongside scheduler-specific metrics. Uses SchedulerMetricsCollector
        for allocation tracking.

        Returns:
            Health check result with scheduler status and metrics.
        """
        # Get base class health metrics (error rates, cycles, events processed)
        base_health = super().health_check()

        # Get metrics from collector (December 30, 2025: extracted to scheduler_metrics.py)
        games_in_window = self._metrics_collector.get_games_in_window()
        games_total = self._metrics_collector._games_allocated_total

        # Determine health status (combine base class + scheduler-specific)
        current_time = time.time()
        stale_priority = current_time - self._last_priority_update > 300  # 5 min
        healthy = base_health.healthy and self._subscribed and not stale_priority

        message = "Running" if healthy else (
            base_health.message if not base_health.healthy else
            "Not subscribed to events" if not self._subscribed else
            "Priority data stale (>5 min)"
        )

        # Merge base details with scheduler-specific details
        details = {
            **base_health.details,  # Includes events_processed, errors_count, etc.
            "subscribed": self._subscribed,
            "configs_tracked": len(self._config_priorities),
            "nodes_tracked": len(self._node_capabilities),
            "last_priority_update": self._last_priority_update,
            "priority_age_seconds": current_time - self._last_priority_update,
            "games_allocated_total": games_total,
            "games_in_last_hour": games_in_window,
        }

        return HealthCheckResult(
            healthy=healthy,
            message=message,
            details=details,
        )


# =============================================================================
# Singleton
# =============================================================================

# December 30, 2025: Module-level cache deprecated in favor of HandlerBase.get_instance()
# Kept for backward compatibility but delegates to HandlerBase singleton management.
_scheduler_instance: SelfplayScheduler | None = None


def get_selfplay_scheduler() -> SelfplayScheduler:
    """Get the singleton SelfplayScheduler instance.

    December 30, 2025: Now delegates to HandlerBase.get_instance() for unified
    singleton management. The subscribe_to_events() call is retained for backward
    compatibility with code that doesn't use start().
    """
    global _scheduler_instance

    # Use HandlerBase's singleton management
    scheduler = SelfplayScheduler.get_instance()

    # Keep module-level cache in sync for any legacy code that checks it
    _scheduler_instance = scheduler

    # Subscribe to events (safe to call multiple times)
    if not scheduler._subscribed:
        scheduler.subscribe_to_events()

    return scheduler


def reset_selfplay_scheduler() -> None:
    """Reset the scheduler singleton (for testing).

    December 30, 2025: Now delegates to HandlerBase.reset_instance().
    """
    global _scheduler_instance
    _scheduler_instance = None
    SelfplayScheduler.reset_instance()
