"""SelfplayScheduler - Priority-based selfplay allocation across cluster.

This module provides intelligent scheduling of selfplay jobs across the cluster,
prioritizing configurations based on:
- Data staleness (fresher data gets lower priority)
- ELO improvement velocity (fast-improving configs get more resources)
- Training pipeline needs (configs waiting for training get boosted)
- Node capabilities (allocate based on GPU power)

Architecture:
    SelfplayScheduler
    ├── TrainingFreshness: Track data age per config
    ├── ELO Velocity: Track improvement rate
    ├── Node Allocator: Distribute based on capabilities
    └── Event Integration: React to pipeline events

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

import asyncio
import contextlib
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import yaml

from app.config.thresholds import (
    SELFPLAY_GAMES_PER_NODE,
    is_ephemeral_node,
    get_gpu_weight,
)

# December 2025: Constants consolidated in priority_calculator.py
# December 29, 2025: Now also imports PriorityCalculator for delegation
from app.coordination.priority_calculator import (
    ALL_CONFIGS,
    ClusterState,
    PLAYER_COUNT_ALLOCATION_MULTIPLIER,
    PriorityCalculator,
    PriorityInputs,
    PRIORITY_OVERRIDE_MULTIPLIERS,
    SAMPLES_PER_GAME_BY_BOARD,
    VOI_SAMPLE_COST_BY_BOARD,
    compute_dynamic_weights,
)

# December 2025: NodeCapability extracted to node_allocator.py
from app.coordination.node_allocator import NodeCapability

# December 2025: Budget calculation extracted to budget_calculator.py
from app.coordination.budget_calculator import (
    get_adaptive_budget_for_elo as _get_budget_for_elo,
    get_adaptive_budget_for_games as _get_budget_for_games,
    get_budget_with_intensity as _get_budget_with_intensity,  # Sprint 10
    compute_target_games as _compute_target,
    parse_config_key,
)
from app.coordination.protocols import HealthCheckResult
from app.coordination.handler_base import HandlerBase
from app.coordination.event_handler_utils import extract_config_key

# January 2026 Sprint 17.4: Health monitoring extracted to mixin
from app.coordination.selfplay_health_monitor import SelfplayHealthMonitorMixin

# January 2026 Sprint 17.9: Quality signal handlers extracted to mixin
# Note: Import directly from module, not from package, to avoid circular import
from app.coordination.selfplay.quality_signal_handler import SelfplayQualitySignalMixin

# January 2026 Sprint 17.9: Velocity/Elo handlers extracted to mixin
from app.coordination.selfplay.velocity_mixin import SelfplayVelocityMixin

# December 30, 2025: Extracted cache and metrics classes
from app.coordination.config_state_cache import ConfigStateCache
from app.coordination.scheduler_metrics import SchedulerMetricsCollector

# Import interfaces for type hints (no circular dependency)
from app.coordination.interfaces import IBackpressureMonitor

# January 5, 2026 (Phase 7.4): Node circuit breaker for work allocation filtering
from app.coordination.node_circuit_breaker import get_node_circuit_registry

# January 2026 Sprint 17.9: AllocationEngine extracted for testability
from app.coordination.selfplay.allocation_engine import (
    AllocationContext,
    AllocationEngine,
    AllocationResult,
)

# Note: backpressure concrete import moved to lazy loading in allocate_selfplay_batch()
# to break circular dependency cycle (Dec 2025). The IBackpressureMonitor protocol
# from interfaces allows type hints without importing the concrete class.

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================
# Note: ALL_CONFIGS, PRIORITY_OVERRIDE_MULTIPLIERS, PLAYER_COUNT_ALLOCATION_MULTIPLIER,
# SAMPLES_PER_GAME_BY_BOARD, and VOI_SAMPLE_COST_BY_BOARD are imported from
# app.coordination.priority_calculator (December 2025 consolidation)

# Priority calculation weights (BASE values - adjusted dynamically)
# Dec 29, 2025: These are now baseline weights that get adjusted based on cluster state.
# See _compute_dynamic_weights() for adaptive reweighting logic.
#
# IMPORTANT: For runtime tuning via environment variables, use SelfplayPriorityWeightDefaults
# from app.config.coordination_defaults. Example:
#   export RINGRIFT_STALENESS_WEIGHT=0.40  # Boost data freshness priority
#   export RINGRIFT_ELO_VELOCITY_WEIGHT=0.15  # Reduce velocity priority
#
# The module-level constants below are for backward compatibility and internal use.
# New code should import from coordination_defaults for env var support.
from app.config.coordination_defaults import SelfplayPriorityWeightDefaults

# Create singleton instance for env var resolution
_priority_weight_defaults = SelfplayPriorityWeightDefaults()

STALENESS_WEIGHT = _priority_weight_defaults.STALENESS_WEIGHT
ELO_VELOCITY_WEIGHT = _priority_weight_defaults.ELO_VELOCITY_WEIGHT
TRAINING_NEED_WEIGHT = _priority_weight_defaults.TRAINING_NEED_WEIGHT
EXPLORATION_BOOST_WEIGHT = _priority_weight_defaults.EXPLORATION_BOOST_WEIGHT
CURRICULUM_WEIGHT = _priority_weight_defaults.CURRICULUM_WEIGHT
IMPROVEMENT_BOOST_WEIGHT = _priority_weight_defaults.IMPROVEMENT_BOOST_WEIGHT
DATA_DEFICIT_WEIGHT = _priority_weight_defaults.DATA_DEFICIT_WEIGHT
QUALITY_WEIGHT = _priority_weight_defaults.QUALITY_WEIGHT
VOI_WEIGHT = _priority_weight_defaults.VOI_WEIGHT

# =============================================================================
# Dynamic Weight Bounds (Dec 29, 2025)
# =============================================================================
# Min/max bounds for dynamic weight adjustment based on cluster state
# These prevent any single factor from dominating allocation decisions
# Now sourced from centralized config with env var support
DYNAMIC_WEIGHT_BOUNDS = _priority_weight_defaults.get_weight_bounds()

# VOI target Elo (from coordination_defaults)
VOI_ELO_TARGET = _priority_weight_defaults.VOI_ELO_TARGET

# Thresholds for dynamic weight adjustment triggers (now env-configurable)
IDLE_GPU_HIGH_THRESHOLD = _priority_weight_defaults.IDLE_GPU_HIGH_THRESHOLD
IDLE_GPU_LOW_THRESHOLD = _priority_weight_defaults.IDLE_GPU_LOW_THRESHOLD
TRAINING_QUEUE_HIGH_THRESHOLD = _priority_weight_defaults.TRAINING_QUEUE_HIGH_THRESHOLD
CONFIGS_AT_TARGET_THRESHOLD = _priority_weight_defaults.CONFIGS_AT_TARGET_THRESHOLD
ELO_HIGH_THRESHOLD = _priority_weight_defaults.ELO_HIGH_THRESHOLD
ELO_MEDIUM_THRESHOLD = _priority_weight_defaults.ELO_MEDIUM_THRESHOLD

# Target games per config for data deficit calculation
TARGET_GAMES_FOR_2000_ELO = _priority_weight_defaults.TARGET_GAMES_FOR_2000_ELO
LARGE_BOARD_TARGET_MULTIPLIER = _priority_weight_defaults.LARGE_BOARD_TARGET_MULTIPLIER

# Dec 29, 2025: Data starvation thresholds (now env-configurable)
# Configs with fewer games than these thresholds get priority boosts
# Especially critical for 4-player configs which have near-zero games
# ULTRA tier added Dec 29, 2025 for critically starved configs (< 20 games)
DATA_STARVATION_ULTRA_THRESHOLD = _priority_weight_defaults.DATA_STARVATION_ULTRA_THRESHOLD
DATA_STARVATION_EMERGENCY_THRESHOLD = _priority_weight_defaults.DATA_STARVATION_EMERGENCY_THRESHOLD
DATA_STARVATION_CRITICAL_THRESHOLD = _priority_weight_defaults.DATA_STARVATION_CRITICAL_THRESHOLD
DATA_STARVATION_ULTRA_MULTIPLIER = _priority_weight_defaults.DATA_STARVATION_ULTRA_MULTIPLIER
DATA_STARVATION_EMERGENCY_MULTIPLIER = _priority_weight_defaults.DATA_STARVATION_EMERGENCY_MULTIPLIER
DATA_STARVATION_CRITICAL_MULTIPLIER = _priority_weight_defaults.DATA_STARVATION_CRITICAL_MULTIPLIER

# Data poverty tier (Dec 30, 2025): Moderate boost for configs with <5000 games
# Bridges gap between CRITICAL (1000 games) and no boost
DATA_POVERTY_THRESHOLD = _priority_weight_defaults.DATA_POVERTY_THRESHOLD
DATA_POVERTY_MULTIPLIER = _priority_weight_defaults.DATA_POVERTY_MULTIPLIER

# Session 17.34 (Jan 5, 2026): WARNING tier for configs with <5000 games
# Catches configs like square8_3p (3,167 games) that are above CRITICAL but still underserved
DATA_WARNING_THRESHOLD = _priority_weight_defaults.DATA_WARNING_THRESHOLD
DATA_WARNING_MULTIPLIER = _priority_weight_defaults.DATA_WARNING_MULTIPLIER

# Default training sample target per config
DEFAULT_TRAINING_SAMPLES_TARGET = 50000

# Staleness thresholds (hours) - now env-configurable
FRESH_DATA_THRESHOLD = _priority_weight_defaults.FRESH_DATA_THRESHOLD
STALE_DATA_THRESHOLD = _priority_weight_defaults.STALE_DATA_THRESHOLD
MAX_STALENESS_HOURS = _priority_weight_defaults.MAX_STALENESS_HOURS

# Default allocation (December 27, 2025: Centralized in coordination_defaults.py)
from app.config.coordination_defaults import SelfplayAllocationDefaults

DEFAULT_GAMES_PER_CONFIG = SelfplayAllocationDefaults.GAMES_PER_CONFIG
MIN_GAMES_PER_ALLOCATION = SelfplayAllocationDefaults.MIN_GAMES_PER_ALLOCATION

# Resource management thresholds (for get_target_jobs_for_node)
MIN_MEMORY_GB_FOR_TASKS = SelfplayAllocationDefaults.MIN_MEMORY_GB
DISK_WARNING_THRESHOLD = SelfplayAllocationDefaults.DISK_WARNING_THRESHOLD
MEMORY_WARNING_THRESHOLD = SelfplayAllocationDefaults.MEMORY_WARNING_THRESHOLD


# January 2, 2026: DynamicWeights and ConfigPriority extracted to selfplay_priority_types.py
# for better modularity and testability (~200 LOC extracted)
from app.coordination.selfplay_priority_types import ConfigPriority, DynamicWeights

# Note: NodeCapability is imported from app.coordination.node_allocator
# DynamicWeights and ConfigPriority are now imported from selfplay_priority_types.py
# (January 2, 2026 extraction - ~200 LOC moved to separate module)


class SelfplayScheduler(SelfplayVelocityMixin, SelfplayQualitySignalMixin, SelfplayHealthMonitorMixin, HandlerBase):
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
            data_starvation_emergency_threshold=DATA_STARVATION_EMERGENCY_THRESHOLD,
            data_starvation_critical_threshold=DATA_STARVATION_CRITICAL_THRESHOLD,
            data_starvation_emergency_multiplier=DATA_STARVATION_EMERGENCY_MULTIPLIER,
            data_starvation_critical_multiplier=DATA_STARVATION_CRITICAL_MULTIPLIER,
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
    # Game Count Normalization (Dec 29, 2025)
    # =========================================================================

    def _get_samples_per_game_estimate(self, config_key: str) -> float:
        """Get estimated samples per game for a config.

        Uses historical averages from SAMPLES_PER_GAME_BY_BOARD, falling back
        to a conservative default of 50 samples/game if config not found.

        Args:
            config_key: Config key like "hex8_2p", "square19_4p"

        Returns:
            Estimated samples per game
        """
        board_type, num_players = parse_config_key(config_key)
        player_key = f"{num_players}p"
        if board_type in SAMPLES_PER_GAME_BY_BOARD:
            return float(SAMPLES_PER_GAME_BY_BOARD[board_type].get(player_key, 50))
        return 50.0  # Conservative default

    def set_target_training_samples(self, config_key: str, target_samples: int) -> None:
        """Set training sample target for a configuration.

        Dec 29, 2025: Part of game count normalization feature.
        DataPipelineOrchestrator calls this before requesting export to
        communicate how many samples are needed for training.

        This updates the ConfigPriority's target_training_samples and
        samples_per_game_estimate, which are then used by games_needed
        property to determine how many more games should be generated.

        Args:
            config_key: Config key (e.g., "hex8_2p", "square19_4p")
            target_samples: Number of training samples needed

        Example:
            scheduler.set_target_training_samples("hex8_2p", 100000)
            # Now scheduler.get_games_needed("hex8_2p") returns games needed
        """
        if config_key not in self._config_priorities:
            logger.warning(
                f"[SelfplayScheduler] Unknown config {config_key}, creating priority entry"
            )
            self._config_priorities[config_key] = ConfigPriority(config_key=config_key)

        priority = self._config_priorities[config_key]
        # Validate target: 0 or negative keeps existing value or uses default
        if target_samples <= 0:
            if priority.target_training_samples <= 0:
                # Use default if no existing value
                priority.target_training_samples = 100000
            # Otherwise keep existing value
            logger.debug(
                f"[SelfplayScheduler] Target {target_samples} <= 0 for {config_key}, "
                f"using existing/default: {priority.target_training_samples}"
            )
        else:
            priority.target_training_samples = target_samples
        priority.samples_per_game_estimate = self._get_samples_per_game_estimate(config_key)

        games_needed = priority.games_needed
        logger.info(
            f"[SelfplayScheduler] Set training target for {config_key}: "
            f"{target_samples} samples, {priority.samples_per_game_estimate:.0f} samples/game estimate, "
            f"{games_needed} games needed"
        )

    def get_games_needed(self, config_key: str) -> int:
        """Get number of additional games needed for a configuration.

        Dec 29, 2025: Part of game count normalization feature.
        Returns how many more games are needed to meet the training sample target.

        Args:
            config_key: Config key (e.g., "hex8_2p", "square19_4p")

        Returns:
            Number of games needed (0 if target already met or config unknown)
        """
        if config_key not in self._config_priorities:
            return 0
        return self._config_priorities[config_key].games_needed

    def get_all_games_needed(self) -> dict[str, int]:
        """Get games needed for all configurations.

        Dec 29, 2025: Part of game count normalization feature.
        Returns a dict mapping config_key to games_needed for all configs.

        Returns:
            Dict[config_key, games_needed] for all tracked configurations
        """
        return {
            cfg: priority.games_needed
            for cfg, priority in self._config_priorities.items()
        }

    # =========================================================================
    # Priority Calculation
    # =========================================================================

    async def get_priority_configs(self, top_n: int = 12) -> list[tuple[str, float]]:
        """Get configs ranked by priority for selfplay allocation.

        Args:
            top_n: Number of top priority configs to return (default: 12 for all configs)

        Returns:
            List of (config_key, priority_score) tuples, sorted by priority

        December 28, 2025: Changed default from 6 to 12 to include all board/player configs.
        """
        await self._update_priorities()

        priorities = [
            (cfg, p.priority_score)
            for cfg, p in self._config_priorities.items()
        ]

        # Sort by priority (descending)
        priorities.sort(key=lambda x: -x[1])

        return priorities[:top_n]

    def get_priority_configs_sync(
        self, top_n: int | None = None, filter_configs: list[str] | None = None
    ) -> list[tuple[str, float]]:
        """Get configs ranked by priority (sync version using cached data).

        This method returns cached priority data without triggering an async update.
        Use this from synchronous contexts where you need priority-ordered configs.

        Args:
            top_n: Optional limit on number of configs to return (default: all)
            filter_configs: Optional list of config keys to filter by

        Returns:
            List of (config_key, priority_score) tuples, sorted by priority descending

        December 2025: Added for IdleResourceDaemon and other sync callers.
        """
        priorities = [
            (cfg, p.priority_score)
            for cfg, p in self._config_priorities.items()
            if filter_configs is None or cfg in filter_configs
        ]

        # Sort by priority (descending)
        priorities.sort(key=lambda x: -x[1])

        if top_n is not None:
            return priorities[:top_n]
        return priorities

    async def _update_priorities(self) -> None:
        """Update priority scores for all configurations."""
        now = time.time()
        if now - self._last_priority_update < self._priority_update_interval:
            return

        self._last_priority_update = now

        # Jan 2026: Parallel fetch of all async data for priority updates
        # This significantly reduces priority update latency from 6 sequential calls to 1 parallel call
        (
            freshness_data,
            elo_data,
            feedback_data,
            curriculum_data,
            game_count_data,
            elo_current_data,
        ) = await asyncio.gather(
            self._get_data_freshness(),
            self._get_elo_velocities(),
            self._get_feedback_signals(),
            self._get_curriculum_weights(),
            self._get_game_counts(),
            self._get_current_elos(),
        )

        # Sync operations (fast, no parallelization needed)
        improvement_data = self._get_improvement_boosts()
        momentum_data = self._get_momentum_multipliers()
        architecture_data = self._get_architecture_boosts()

        # Update each config
        for config_key, priority in self._config_priorities.items():
            # Update staleness
            if config_key in freshness_data:
                priority.staleness_hours = freshness_data[config_key]

            # Update ELO velocity
            if config_key in elo_data:
                priority.elo_velocity = elo_data[config_key]

            # Update feedback signals
            if config_key in feedback_data:
                priority.exploration_boost = feedback_data[config_key].get("exploration_boost", 1.0)
                priority.training_pending = feedback_data[config_key].get("training_pending", False)

            # Update curriculum weight (Phase 2C.3)
            if config_key in curriculum_data:
                priority.curriculum_weight = curriculum_data[config_key]

            # Update improvement boost (Phase 5)
            if config_key in improvement_data:
                priority.improvement_boost = improvement_data[config_key]

            # Update momentum multiplier (Phase 19)
            if config_key in momentum_data:
                priority.momentum_multiplier = momentum_data[config_key]

            # Update architecture boost (Phase 5B)
            if config_key in architecture_data:
                priority.architecture_boost = architecture_data[config_key]

            # Update game count and large board flag (Dec 2025)
            if config_key in game_count_data:
                priority.game_count = game_count_data[config_key]
            # Mark large boards for higher data deficit weight
            priority.is_large_board = config_key.startswith("square19") or config_key.startswith("hexagonal")

            # Update current Elo and search budget (Dec 29, 2025)
            # Budget now considers BOTH game count AND Elo:
            # - Low game count (<1000): Use bootstrap budgets for faster data generation
            # - High game count (>=1000): Use Elo-based budgets for quality
            # January 2026 Sprint 10: Also factors in training intensity
            # - Higher intensity (hot_path, accelerated) → higher budget
            # - Lower intensity (reduced, paused) → lower budget
            if config_key in elo_current_data:
                current_elo = elo_current_data[config_key]
                priority.current_elo = current_elo  # Store for dynamic weight calculation
                game_count = priority.game_count
                # Sprint 10: Use intensity-coupled budget calculation
                new_budget = self._get_budget_with_intensity(game_count, current_elo, config_key)
                old_budget = priority.search_budget
                if new_budget != old_budget:
                    priority.search_budget = new_budget
                    intensity = self._get_training_intensity_for_config(config_key)
                    logger.info(
                        f"[SelfplayScheduler] Adaptive budget for {config_key}: "
                        f"{old_budget}→{new_budget} (games={game_count}, Elo={current_elo:.0f}, "
                        f"intensity={intensity})"
                    )

            # Dec 29, 2025: Update Elo uncertainty for VOI calculation
            # Uncertainty decreases with more games (statistical sampling theory)
            # Base uncertainty of 300 Elo, reduces with sqrt(game_count)
            BASE_UNCERTAINTY = 300.0
            MIN_UNCERTAINTY = 30.0  # Floor to prevent near-zero uncertainty
            if priority.game_count > 0:
                priority.elo_uncertainty = max(
                    MIN_UNCERTAINTY,
                    BASE_UNCERTAINTY / math.sqrt(priority.game_count)
                )
            else:
                priority.elo_uncertainty = BASE_UNCERTAINTY

            # January 2026 Sprint 10: Update diversity score and opponent count
            priority.diversity_score = self.get_diversity_score(config_key)
            priority.opponent_types_seen = self.get_opponent_types_seen(config_key)

            # Compute priority score
            priority.priority_score = self._compute_priority_score(priority)

        # Phase 12: Check for expired exploration boosts and decay them
        decayed_count = self._decay_expired_boosts(now)
        if decayed_count > 0:
            logger.info(f"[SelfplayScheduler] Decayed {decayed_count} expired exploration boosts")

        logger.debug(f"[SelfplayScheduler] Updated priorities for {len(self._config_priorities)} configs")

        # Phase 6: Record cluster utilization in ImprovementOptimizer
        # This enables the training loop to adapt based on resource usage
        try:
            from app.training.improvement_optimizer import get_improvement_optimizer

            # Get approximate utilization from node count and active configs
            active_configs = len([p for p in self._config_priorities.values() if p.training_pending])
            total_configs = len(self._config_priorities)

            # Estimate GPU utilization: assume 50% baseline + (active configs / total) * 50%
            gpu_util = min(100.0, 50.0 + (active_configs / max(1, total_configs)) * 50.0)
            cpu_util = gpu_util * 0.6  # CPU typically lower than GPU for selfplay

            optimizer = get_improvement_optimizer()
            rec = optimizer.record_cluster_utilization(
                cpu_utilization=cpu_util,
                gpu_utilization=gpu_util,
            )
            logger.debug(
                f"[SelfplayScheduler] Recorded cluster utilization: CPU={cpu_util:.0f}%, "
                f"GPU={gpu_util:.0f}% (signal: {rec.signal.name})"
            )
        except ImportError:
            pass  # Improvement optimizer not available
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Failed to record utilization: {e}")

    def _compute_dynamic_weights(self) -> DynamicWeights:
        """Compute dynamic priority weights based on current cluster state.

        Dec 29, 2025: Implements adaptive reweighting to optimize resource allocation.
        Now delegates to priority_calculator.compute_dynamic_weights() for the actual
        weight computation logic.

        Weight adjustment logic:
        - High idle GPU fraction → Boost staleness weight (generate more data)
        - Large training queue → Reduce staleness weight (don't flood queue)
        - Many configs at Elo target → Reduce velocity weight (focus on struggling configs)
        - High average Elo → Boost curriculum weight (need harder positions)

        Returns:
            DynamicWeights with adjusted values based on cluster state
        """
        now = time.time()

        # Rate-limit weight updates (expensive to compute cluster state)
        if now - self._last_dynamic_weights_update < self._dynamic_weights_update_interval:
            return self._dynamic_weights

        self._last_dynamic_weights_update = now

        # --- Gather cluster state ---
        cluster_state = self._gather_cluster_state()

        # --- Delegate weight computation to priority_calculator ---
        weights = compute_dynamic_weights(cluster_state)

        # Log significant weight changes
        old_weights = self._dynamic_weights
        if (abs(weights.staleness - old_weights.staleness) > 0.05 or
            abs(weights.curriculum - old_weights.curriculum) > 0.03):
            logger.info(
                f"[SelfplayScheduler] Dynamic weights updated: "
                f"staleness={weights.staleness:.2f} (was {old_weights.staleness:.2f}), "
                f"curriculum={weights.curriculum:.2f} (was {old_weights.curriculum:.2f}), "
                f"idle_gpus={cluster_state.idle_gpu_fraction:.1%}, "
                f"queue={cluster_state.training_queue_depth}, "
                f"at_target={cluster_state.configs_at_target_fraction:.1%}, "
                f"avg_elo={cluster_state.average_elo:.0f}"
            )

        self._dynamic_weights = weights

        # Update PriorityCalculator with new weights
        self._priority_calculator.update_weights(weights)

        return weights

    def _gather_cluster_state(self) -> ClusterState:
        """Gather current cluster state for dynamic weight computation.

        December 29, 2025: Extracted from _compute_dynamic_weights() for clarity.

        Returns:
            ClusterState with current metrics
        """
        # 1. Idle GPU fraction (from node capabilities)
        # Dec 31, 2025: Enhanced logging to identify which GPU nodes are idle
        idle_gpu_fraction = 0.0
        if self._node_capabilities:
            total_nodes = len(self._node_capabilities)
            idle_gpu_nodes = [
                node_id
                for node_id, cap in self._node_capabilities.items()
                if cap.current_jobs == 0 and cap.gpu_memory_gb > 0
            ]
            idle_nodes = len(idle_gpu_nodes)
            idle_gpu_fraction = idle_nodes / max(1, total_nodes)

            # Log idle GPU nodes for diagnostics (helps identify underutilized resources)
            if idle_gpu_nodes:
                logger.info(
                    f"[SelfplayScheduler] Idle GPU nodes ({idle_nodes}/{total_nodes}): "
                    f"{', '.join(idle_gpu_nodes[:5])}{'...' if len(idle_gpu_nodes) > 5 else ''}"
                )

        # 2. Training queue depth (check backpressure monitor)
        training_queue_depth = 0
        if self._backpressure_monitor:
            try:
                # Synchronous check for queue depth - use hasattr to avoid MagicMock issues
                if hasattr(self._backpressure_monitor, '_last_queue_depth'):
                    cached_depth = self._backpressure_monitor._last_queue_depth
                    # Ensure numeric value (handles MagicMock in tests)
                    if isinstance(cached_depth, (int, float)):
                        training_queue_depth = int(cached_depth)
            except (AttributeError, TypeError, ValueError):
                pass  # Handle missing attributes or type conversion issues

        # 3. Configs at Elo target fraction
        elo_target = 2000.0  # From thresholds
        configs_at_target = 0
        total_configs = len(self._config_priorities)
        for cfg, priority in self._config_priorities.items():
            # Check if config has reached target Elo
            if hasattr(priority, 'current_elo') and priority.current_elo >= elo_target:
                configs_at_target += 1
        configs_at_target_fraction = configs_at_target / max(1, total_configs)

        # 4. Average model Elo
        elo_sum = 0.0
        elo_count = 0
        for priority in self._config_priorities.values():
            if hasattr(priority, 'current_elo') and priority.current_elo > 0:
                elo_sum += priority.current_elo
                elo_count += 1
        average_elo = elo_sum / max(1, elo_count) if elo_count > 0 else 1500.0

        return ClusterState(
            idle_gpu_fraction=idle_gpu_fraction,
            training_queue_depth=training_queue_depth,
            configs_at_target_fraction=configs_at_target_fraction,
            average_elo=average_elo,
        )

    def _config_priority_to_inputs(self, priority: ConfigPriority) -> PriorityInputs:
        """Convert ConfigPriority to PriorityInputs for PriorityCalculator.

        December 29, 2025: Helper for delegating priority calculation.

        Args:
            priority: ConfigPriority from scheduler state

        Returns:
            PriorityInputs for use with PriorityCalculator
        """
        return PriorityInputs(
            config_key=priority.config_key,
            staleness_hours=priority.staleness_hours,
            elo_velocity=priority.elo_velocity,
            training_pending=priority.training_pending,
            exploration_boost=priority.exploration_boost,
            curriculum_weight=priority.curriculum_weight,
            improvement_boost=priority.improvement_boost,
            quality_penalty=priority.quality_penalty,
            architecture_boost=priority.architecture_boost,
            momentum_multiplier=priority.momentum_multiplier,
            game_count=priority.game_count,
            is_large_board=priority.is_large_board,
            priority_override=priority.priority_override,
            current_elo=priority.current_elo,
            elo_uncertainty=priority.elo_uncertainty,
            target_elo=priority.target_elo,
            # January 2026 Sprint 10: Diversity score from opponent tracking
            diversity_score=priority.diversity_score,
            # January 2026 Phase 2: Cluster-wide game count for deficit calculation
            cluster_game_count=self._get_cluster_game_count(priority.config_key),
        )

    def _compute_priority_score(self, priority: ConfigPriority) -> float:
        """Compute overall priority score for a configuration.

        Higher score = higher priority for selfplay allocation.

        December 2025: Refactored to delegate core computation to PriorityCalculator.
        Keeps scheduler-specific handling for:
        - Dynamic weight updates (triggers cluster state refresh)
        - ULTRA starvation tier (more severe than PriorityCalculator's emergency)
        - Detailed logging for starvation warnings
        - Momentum multiplier change logging
        """
        # Ensure dynamic weights are fresh and update calculator
        self._compute_dynamic_weights()

        # Convert ConfigPriority to PriorityInputs for calculator
        inputs = self._config_priority_to_inputs(priority)

        # Delegate core computation to PriorityCalculator
        # This handles: staleness, velocity, curriculum, quality, VOI, data deficit,
        # exploration boost, momentum, priority override, player count, cascade, starvation
        score = self._priority_calculator.compute_priority_score(inputs)

        # ULTRA starvation tier override (more severe than PriorityCalculator's emergency)
        # PriorityCalculator handles emergency (< 50) and critical (< 200) tiers
        # ULTRA (< 20) is even more urgent and needs special logging
        game_count = priority.game_count
        if game_count < DATA_STARVATION_ULTRA_THRESHOLD:
            # ULTRA tier: divide out the emergency multiplier already applied by calculator,
            # then apply ULTRA multiplier instead
            score = score / DATA_STARVATION_EMERGENCY_MULTIPLIER * DATA_STARVATION_ULTRA_MULTIPLIER
            logger.warning(
                f"[SelfplayScheduler] ULTRA STARVATION: {priority.config_key} has only "
                f"{game_count} games (<{DATA_STARVATION_ULTRA_THRESHOLD}). "
                f"Applying {DATA_STARVATION_ULTRA_MULTIPLIER}x priority boost. URGENT DATA NEEDED!"
            )
            # Jan 5, 2026: Emit starvation event for automatic dispatch trigger
            # Only emit once per 5 minutes to avoid flooding the event bus
            starvation_cooldown_key = f"starvation_alert_{priority.config_key}"
            last_alert = getattr(self, "_starvation_alert_times", {}).get(starvation_cooldown_key, 0)
            if time.time() - last_alert > 300:  # 5 minute cooldown
                self._emit_starvation_alert(priority.config_key, game_count, "ULTRA")
                if not hasattr(self, "_starvation_alert_times"):
                    self._starvation_alert_times: dict[str, float] = {}
                self._starvation_alert_times[starvation_cooldown_key] = time.time()
        elif game_count < DATA_STARVATION_EMERGENCY_THRESHOLD:
            # Log for visibility (calculator already applied the multiplier)
            logger.warning(
                f"[SelfplayScheduler] EMERGENCY: {priority.config_key} has only "
                f"{game_count} games (<{DATA_STARVATION_EMERGENCY_THRESHOLD}). "
                f"Applying {DATA_STARVATION_EMERGENCY_MULTIPLIER}x priority boost."
            )
        elif game_count < DATA_STARVATION_CRITICAL_THRESHOLD:
            # Log for visibility (calculator already applied the multiplier)
            logger.info(
                f"[SelfplayScheduler] CRITICAL: {priority.config_key} has only "
                f"{game_count} games (<{DATA_STARVATION_CRITICAL_THRESHOLD}). "
                f"Applying {DATA_STARVATION_CRITICAL_MULTIPLIER}x priority boost."
            )
        elif game_count < DATA_POVERTY_THRESHOLD:
            # POVERTY tier (Dec 30, 2025): Moderate boost for configs with <5000 games
            # PriorityCalculator doesn't handle this tier, so apply multiplier directly
            score = score * DATA_POVERTY_MULTIPLIER
            logger.info(
                f"[SelfplayScheduler] POVERTY: {priority.config_key} has only "
                f"{game_count} games (<{DATA_POVERTY_THRESHOLD}). "
                f"Applying {DATA_POVERTY_MULTIPLIER}x priority boost."
            )
        elif game_count < DATA_WARNING_THRESHOLD:
            # Session 17.34 (Jan 5, 2026): WARNING tier for configs with <5000 games
            # Catches configs like square8_3p (3,167 games) that are above CRITICAL/POVERTY
            # but still need a boost to catch up with well-represented configs
            score = score * DATA_WARNING_MULTIPLIER
            logger.info(
                f"[SelfplayScheduler] WARNING: {priority.config_key} has only "
                f"{game_count} games (<{DATA_WARNING_THRESHOLD}). "
                f"Applying {DATA_WARNING_MULTIPLIER}x priority boost."
            )

        # Log momentum multiplier changes (>10% change from baseline)
        if abs(priority.momentum_multiplier - 1.0) > 0.1:
            logger.info(
                f"[SelfplayScheduler] Momentum multiplier applied to {priority.config_key}: "
                f"{priority.momentum_multiplier:.2f}x"
            )

        # Debug logging for priority component breakdown
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"[SelfplayScheduler] Priority for {priority.config_key}: "
                f"score={score:.4f}, games={game_count}, "
                f"momentum={priority.momentum_multiplier:.2f}x"
            )

        return score

    async def _get_data_freshness(self) -> dict[str, float]:
        """Get data freshness (hours since last export) per config.

        Returns:
            Dict mapping config_key to hours since last export
        """
        result = {}

        try:
            # Try using TrainingFreshness if available
            if self._training_freshness is None:
                try:
                    from app.coordination.training_freshness import get_training_freshness
                    self._training_freshness = get_training_freshness()
                except ImportError:
                    pass

            if self._training_freshness:
                for config_key in ALL_CONFIGS:
                    staleness = await self._training_freshness.get_staleness(config_key)
                    result[config_key] = staleness
            else:
                # Fallback: check NPZ file modification times
                from pathlib import Path
                now = time.time()

                for config_key in ALL_CONFIGS:
                    board_type, num_players = parse_config_key(config_key)
                    npz_path = Path(f"data/training/{board_type}_{num_players}p.npz")

                    if npz_path.exists():
                        mtime = npz_path.stat().st_mtime
                        result[config_key] = (now - mtime) / 3600.0
                    else:
                        result[config_key] = MAX_STALENESS_HOURS

        except (OSError, ValueError, IndexError) as e:
            logger.warning(f"[SelfplayScheduler] Error getting freshness (using defaults): {e}")

        return result

    async def _get_elo_velocities(self) -> dict[str, float]:
        """Get ELO improvement velocities per config.

        Returns:
            Dict mapping config_key to ELO points per day
        """
        result = {}

        try:
            # Try using QueuePopulator's ConfigTarget if available
            from app.coordination.unified_queue_populator import get_queue_populator

            populator = get_queue_populator()
            if populator:
                for config_key, target in populator._targets.items():
                    result[config_key] = target.elo_velocity
        except ImportError:
            pass
        except (AttributeError, KeyError) as e:
            logger.warning(f"[SelfplayScheduler] Error getting ELO velocities (using defaults): {e}")

        return result

    async def _get_current_elos(self) -> dict[str, float]:
        """Get current Elo ratings per config.

        Dec 29, 2025: Added for adaptive Gumbel budget scaling.
        Jan 2026: Prefers composite Elo entries (with harness tracking) over legacy.

        Uses 4-layer fallback:
        1. Composite Elo (harness-tracked, highest quality)
        2. QueuePopulator (fast, in-memory)
        3. Legacy EloService (any participant)
        4. Default 1500

        Returns:
            Dict mapping config_key to current Elo rating
        """
        result = {}

        # Layer 1: Prefer composite Elo entries (Jan 2026)
        # These have harness_type and simulation_count, representing quality evaluations
        try:
            from app.training.elo_service import get_elo_service
            import sqlite3

            elo_service = get_elo_service()
            if hasattr(elo_service, '_db_path'):
                conn = sqlite3.connect(elo_service._db_path)
                # Query composite entries with highest budget (b800, b1600)
                # Format: canonical_hex8_2p:gumbel_mcts:b800
                for config_key in ALL_CONFIGS:
                    board_type, num_players = parse_config_key(config_key)
                    cur = conn.execute("""
                        SELECT participant_id, rating, games_played, simulation_count
                        FROM elo_ratings
                        WHERE board_type = ? AND num_players = ?
                          AND participant_id LIKE ?
                          AND games_played >= 10
                        ORDER BY simulation_count DESC NULLS LAST, rating DESC
                        LIMIT 1
                    """, (board_type, num_players, f"canonical_{config_key}:%"))
                    row = cur.fetchone()
                    if row:
                        result[config_key] = row[1]  # rating
                        logger.debug(
                            f"[SelfplayScheduler] Using composite Elo for {config_key}: "
                            f"{row[1]:.0f} ({row[0]}, {row[2]} games)"
                        )
                conn.close()
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Composite Elo query failed: {e}")

        # Layer 2: Try QueuePopulator's ConfigTarget (fastest, in-memory)
        try:
            from app.coordination.unified_queue_populator import get_queue_populator

            populator = get_queue_populator()
            if populator:
                for config_key, target in populator._targets.items():
                    if config_key not in result and target.current_best_elo > 0:
                        result[config_key] = target.current_best_elo
        except ImportError:
            pass
        except (AttributeError, KeyError) as e:
            logger.debug(f"[SelfplayScheduler] QueuePopulator unavailable: {e}")

        # Layer 3: Fallback to EloService database for missing configs
        missing_configs = [c for c in ALL_CONFIGS if c not in result]
        if missing_configs:
            try:
                from app.training.elo_service import get_elo_service

                elo_service = get_elo_service()
                for config_key in missing_configs:
                    # Parse config_key: "hex8_2p" -> board_type="hex8", num_players=2
                    board_type, num_players = parse_config_key(config_key)
                    # Get leaderboard and take top model's Elo
                    leaderboard = elo_service.get_leaderboard(
                        board_type=board_type,
                        num_players=num_players,
                        limit=1,
                        min_games=5,  # Require some confidence
                    )
                    if leaderboard:
                        result[config_key] = leaderboard[0].rating
                        logger.debug(
                            f"[SelfplayScheduler] Got Elo from EloService for {config_key}: "
                            f"{leaderboard[0].rating:.0f}"
                        )
            except ImportError:
                logger.debug("[SelfplayScheduler] EloService not available")
            except Exception as e:
                logger.debug(f"[SelfplayScheduler] EloService query failed: {e}")

        # Layer 4: Default 1500 for any still-missing configs
        for config_key in ALL_CONFIGS:
            if config_key not in result:
                result[config_key] = 1500.0  # Default starting Elo

        return result

    def _get_adaptive_budget_for_elo(self, elo: float) -> int:
        """Get Gumbel MCTS budget based on current Elo tier.

        December 2025: Delegated to budget_calculator module.
        See budget_calculator.get_adaptive_budget_for_elo() for full docs.
        """
        return _get_budget_for_elo(elo)

    def _get_adaptive_budget_for_games(self, game_count: int, elo: float) -> int:
        """Get Gumbel MCTS budget based on game count (prioritizes bootstrapping).

        December 2025: Delegated to budget_calculator module.
        See budget_calculator.get_adaptive_budget_for_games() for full docs.
        """
        return _get_budget_for_games(game_count, elo)

    def _get_budget_with_intensity(
        self, game_count: int, elo: float, config_key: str
    ) -> int:
        """Get Gumbel budget factoring in training intensity.

        January 2026 Sprint 10: Couples training intensity to Gumbel budget.
        Higher intensity configs get higher budgets for better quality games.

        Expected improvement: +20-30 Elo from better intensity/budget alignment.
        """
        intensity = self._get_training_intensity_for_config(config_key)
        return _get_budget_with_intensity(game_count, elo, intensity)

    def _get_training_intensity_for_config(self, config_key: str) -> str:
        """Get training intensity for a config from FeedbackLoopController.

        January 2026 Sprint 10: Retrieves intensity for budget coupling.

        Returns:
            Training intensity string: "hot_path", "accelerated", "normal",
            "reduced", or "paused". Defaults to "normal" if unavailable.
        """
        try:
            from app.coordination.feedback_loop_controller import get_feedback_loop_controller

            controller = get_feedback_loop_controller()
            if controller:
                state = controller._get_or_create_state(config_key)
                return getattr(state, "current_training_intensity", "normal")
        except (ImportError, AttributeError):
            pass
        return "normal"

    def _compute_target_games(self, config: str, current_elo: float) -> int:
        """Compute dynamic target games needed based on Elo gap and board difficulty.

        December 2025: Delegated to budget_calculator module.
        See budget_calculator.compute_target_games() for full docs.
        """
        return _compute_target(config, current_elo)

    def get_target_games_for_config(self, config: str) -> int:
        """Get dynamic target games for a config (public accessor).

        December 29, 2025: Phase 8 - Replaces static TARGET_GAMES_FOR_2000_ELO.
        """
        # Get current Elo from cached data
        current_elo = 1500.0  # Default if not available
        for cfg_key, priority in self._config_priorities.items():
            if cfg_key == config:
                current_elo = getattr(priority, 'current_elo', 1500.0)
                break

        return self._compute_target_games(config, current_elo)

    async def _get_feedback_signals(self) -> dict[str, dict[str, Any]]:
        """Get feedback loop signals per config.

        Returns:
            Dict mapping config_key to feedback data
        """
        result = {}

        try:
            from app.coordination.feedback_loop_controller import get_feedback_loop_controller

            controller = get_feedback_loop_controller()
            if controller:
                for config_key in ALL_CONFIGS:
                    state = controller._get_or_create_state(config_key)
                    result[config_key] = {
                        "exploration_boost": state.current_exploration_boost,
                        "training_pending": state.current_training_intensity == "accelerated",
                    }
        except ImportError:
            pass
        except (AttributeError, KeyError) as e:
            logger.warning(f"[SelfplayScheduler] Error getting feedback signals (using defaults): {e}")

        return result

    async def _get_curriculum_weights(self) -> dict[str, float]:
        """Get curriculum weights per config.

        December 2025 - Phase 2C.3: Wire curriculum weights into priority calculation.
        Higher curriculum weight = config needs more training data.

        Returns:
            Dict mapping config_key to curriculum weight (default 1.0)
        """
        result = {}

        try:
            from app.training.curriculum_feedback import get_curriculum_feedback

            feedback = get_curriculum_feedback()
            if feedback:
                # Use get_curriculum_weights() which computes weights from metrics
                # (win rates, Elo trends, weak opponent data) instead of accessing
                # _current_weights directly which may be empty initially
                computed_weights = feedback.get_curriculum_weights()
                if computed_weights:
                    result = computed_weights
                else:
                    # Fall back to manually tracked weights if metrics unavailable
                    result = dict(feedback._current_weights)

        except ImportError:
            logger.debug("[SelfplayScheduler] curriculum_feedback not available")
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error getting curriculum weights: {e}")

        # Ensure all configs have a default weight
        for config_key in ALL_CONFIGS:
            if config_key not in result:
                result[config_key] = 1.0

        # December 2025: Partial normalization of curriculum weights
        # Session 17.42: Changed from full normalization to 50% partial to preserve differentials
        # Full normalization was flattening weights (e.g., 1.4 -> 1.23), reducing curriculum influence
        # Now we apply only 50% of the scaling adjustment to preserve weight differentials
        if result:
            total = sum(result.values())
            target_sum = len(result)  # Average of 1.0
            if total > 0 and abs(total - target_sum) > 0.01:
                raw_scale = target_sum / total
                # Apply only 50% of the scaling to preserve differentials
                # e.g., if raw_scale=0.88, partial_scale=0.94 (keeps half the differential)
                scale = 1.0 + (raw_scale - 1.0) * 0.5
                result = {k: v * scale for k, v in result.items()}
                logger.debug(
                    f"[SelfplayScheduler] Partial normalized curriculum weights: "
                    f"raw_scale={raw_scale:.3f}, applied_scale={scale:.3f}, "
                    f"sum={sum(result.values()):.2f}"
                )

        return result

    def _fetch_quality_from_daemon(self, config: str) -> Optional[float]:
        """Fetch quality score from QualityMonitorDaemon.

        Dec 30, 2025: Extracted as provider for ConfigStateCache.

        Args:
            config: Config key like "hex8_2p"

        Returns:
            Quality score 0.0-1.0, or None if unavailable
        """
        try:
            from app.coordination.quality_monitor_daemon import get_quality_daemon

            daemon = get_quality_daemon()
            if daemon:
                return daemon.get_config_quality(config)
        except ImportError:
            logger.debug("[SelfplayScheduler] quality_monitor_daemon not available")
        except (AttributeError, KeyError) as e:
            logger.debug(f"[SelfplayScheduler] Error getting quality for {config}: {e}")
        return None

    def _get_config_data_quality(self, config: str) -> float:
        """Get data quality score for a config from QualityMonitorDaemon.

        Dec 29, 2025 - Phase 1: Quality-weighted selfplay allocation.
        Higher quality score = better training data (Gumbel MCTS, passed parity).
        Lower quality = heuristic-only games, parity failures.

        Dec 30, 2025: Refactored to use ConfigStateCache for TTL caching.

        Args:
            config: Config key like "hex8_2p"

        Returns:
            Quality score 0.0-1.0 (default 0.7 if unavailable)
        """
        return self._quality_cache.get_quality_or_fetch(config)

    async def _get_all_config_qualities(self) -> dict[str, float]:
        """Get data quality scores for all configs.

        Dec 29, 2025 - Phase 1: Batch quality lookup for priority calculation.
        Dec 30, 2025: Refactored to use ConfigStateCache.

        Returns:
            Dict mapping config_key to quality score (0.0-1.0)
        """
        return self._quality_cache.get_all_qualities(list(ALL_CONFIGS))

    def invalidate_quality_cache(self, config: str | None = None) -> int:
        """Invalidate quality cache for a config or all configs.

        Dec 30, 2025: Refactored to use ConfigStateCache.
        Call this when quality data changes externally
        (e.g., after evaluation completes, after parity gate updates).

        Args:
            config: Specific config to invalidate, or None to clear all

        Returns:
            Number of entries invalidated
        """
        return self._quality_cache.invalidate(config)

    def _get_cluster_game_counts(self) -> dict[str, int]:
        """Get cluster-wide game counts from UnifiedDataRegistry.

        Jan 2026 Phase 2: Cluster awareness for selfplay scheduling.
        Returns total games available across the cluster (local + remote nodes)
        for each config to prevent duplicate game generation.

        Uses TTL caching to avoid repeated lookups during priority calculation.
        Logs source breakdown for visibility and emits events on fallback.

        Returns:
            Dict mapping config_key to cluster-wide total game count
        """
        now = time.time()

        # Check if cache is fresh
        if now - self._cluster_game_counts_last_update < self._cluster_game_counts_ttl:
            return self._cluster_game_counts

        # Refresh from registry
        try:
            from app.distributed.data_catalog import get_data_registry

            registry = get_data_registry()
            status = registry.get_cluster_status()

            if status and sum(c.get("total", 0) for c in status.values()) > 0:
                # Update cache with total game counts
                self._cluster_game_counts = {
                    config_key: config_data.get("total", 0)
                    for config_key, config_data in status.items()
                }
                self._cluster_game_counts_last_update = now

                # Log source breakdown for visibility (Jan 2026 - cluster-wide measurement)
                local_total = sum(c.get("local", 0) for c in status.values())
                cluster_total = sum(c.get("cluster", 0) for c in status.values())
                owc_total = sum(c.get("owc", 0) for c in status.values())
                total = sum(self._cluster_game_counts.values())

                logger.info(
                    f"[SelfplayScheduler] Using CLUSTER-WIDE counts: "
                    f"local={local_total:,}, cluster={cluster_total:,}, owc={owc_total:,}, "
                    f"total={total:,} games across {len(self._cluster_game_counts)} configs"
                )
                return self._cluster_game_counts

            # Registry returned empty - fall through to fallback
            logger.warning("[SelfplayScheduler] Cluster registry returned empty counts")

        except ImportError as e:
            logger.warning(f"[SelfplayScheduler] DataRegistry not available: {e}")
        except (RuntimeError, ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"[SelfplayScheduler] Error getting cluster counts: {e}")

        # Fallback to local-only counts with explicit warning
        logger.warning(
            "[SelfplayScheduler] FALLBACK to LOCAL-ONLY game counts! "
            "Cluster data unavailable - progress measurement may be incomplete."
        )

        # Emit event for monitoring (Jan 2026 - cluster visibility)
        try:
            from app.distributed.data_events.event_types import DataEventType
            from app.coordination.event_router import emit_event

            emit_event(
                DataEventType.CLUSTER_VISIBILITY_DEGRADED,
                {
                    "reason": "cluster_manifest_unavailable",
                    "node_id": getattr(self, "_node_id", "unknown"),
                    "cached_count": sum(self._cluster_game_counts.values()),
                },
            )
        except Exception:
            pass  # Don't fail on event emission

        return self._cluster_game_counts

    def _get_cluster_game_count(self, config_key: str) -> int:
        """Get cluster-wide game count for a specific config.

        Jan 2026 Phase 2: Helper for priority calculation.

        Args:
            config_key: Config key like "hex8_2p"

        Returns:
            Total games available cluster-wide for this config (default: 0)
        """
        counts = self._get_cluster_game_counts()
        return counts.get(config_key, 0)

    def _get_cascade_priority(self, config_key: str) -> float:
        """Get cascade training priority boost for a config.

        Jan 2026: Delegated to selfplay.priority_boosts module.

        Args:
            config_key: Config key like "hex8_2p"

        Returns:
            Priority multiplier (1.0 = normal, >1.0 = boosted)
        """
        from app.coordination.selfplay.priority_boosts import get_cascade_priority
        return get_cascade_priority(config_key)

    def _get_improvement_boosts(self) -> dict[str, float]:
        """Get improvement boosts from ImprovementOptimizer per config.

        Jan 2026: Delegated to selfplay.priority_boosts module.

        Returns:
            Dict mapping config_key to boost value (-0.10 to +0.15)
        """
        from app.coordination.selfplay.priority_boosts import get_improvement_boosts
        return get_improvement_boosts()

    def _get_momentum_multipliers(self) -> dict[str, float]:
        """Get momentum multipliers from FeedbackAccelerator per config.

        Jan 2026: Delegated to selfplay.priority_boosts module.

        Returns:
            Dict mapping config_key to multiplier value (0.5 to 1.5)
        """
        from app.coordination.selfplay.priority_boosts import get_momentum_multipliers
        return get_momentum_multipliers()

    def _get_architecture_boosts(self) -> dict[str, float]:
        """Get architecture-based boosts per config.

        Jan 2026: Delegated to selfplay.priority_boosts module.

        Returns:
            Dict mapping config_key to boost value (0.0 to +0.30)
        """
        from app.coordination.selfplay.priority_boosts import get_architecture_boosts
        return get_architecture_boosts()

    async def _get_game_counts(self) -> dict[str, int]:
        """Get game counts per config - cluster-aware with fallback to local.

        January 2026: Made cluster-aware. Coordinator nodes don't have local
        canonical databases, so we must use cluster-wide aggregation first.

        Priority order:
        1. UnifiedDataRegistry (cluster manifest + local + OWC + S3)
        2. Local GameDiscovery (last resort fallback)

        Returns:
            Dict mapping config_key to game count
        """
        result: dict[str, int] = {}

        # Try 1: Cluster-wide counts from UnifiedDataRegistry
        try:
            cluster_counts = self._get_cluster_game_counts()
            if cluster_counts and sum(cluster_counts.values()) > 0:
                result = dict(cluster_counts)
                logger.debug(
                    f"[SelfplayScheduler] Using cluster game counts: "
                    f"{sum(result.values()):,} total across {len(result)} configs"
                )
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Cluster counts unavailable: {e}")

        # Try 2: Local-only fallback (for nodes without cluster connectivity)
        if not result or sum(result.values()) == 0:
            try:
                from app.utils.game_discovery import get_game_counts_summary

                result = get_game_counts_summary()
                if sum(result.values()) > 0:
                    logger.debug(
                        f"[SelfplayScheduler] Falling back to local counts: "
                        f"{sum(result.values()):,} total across {len(result)} configs"
                    )
            except ImportError:
                logger.debug("[SelfplayScheduler] game_discovery not available")
            except Exception as e:
                logger.warning(f"[SelfplayScheduler] Error getting local game counts: {e}")

        # Ensure all configs have a count (0 if not found)
        for config_key in ALL_CONFIGS:
            if config_key not in result:
                result[config_key] = 0

        return result

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
            from app.coordination.event_router import emit_event
            from app.distributed.data_events import DataEventType

            # Map string event name to DataEventType
            event_type = getattr(DataEventType, event_name, None)
            if event_type is not None:
                emit_event(event_type, payload)
            else:
                logger.warning(f"[SelfplayScheduler] Unknown event type: {event_name}")
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Event emission failed (non-critical): {e}")

    def _enforce_starvation_floor(
        self,
        allocation: dict[str, dict[str, int]],
        games_per_config: int,
    ) -> dict[str, dict[str, int]]:
        """Enforce minimum allocation for data-starved configs.

        Dec 29, 2025 - Phase 5: Data starvation floor enforcement.
        Any config with <100 games MUST receive allocation, even if it wasn't
        in the initial priority list. This addresses critical gaps like:
        - square19_4p: 1 game
        - hexagonal_4p: 400 games

        For emergency configs (<100 games): allocate 2x base games
        For critical configs (<1000 games): allocate 1.5x base games

        Args:
            allocation: Current allocation {config_key: {node_id: games}}
            games_per_config: Base games per config

        Returns:
            Adjusted allocation with starvation floors enforced
        """
        added_configs = []

        for config_key in ALL_CONFIGS:
            priority = self._config_priorities.get(config_key)
            if not priority:
                continue

            game_count = priority.game_count

            # Skip if already adequately allocated
            current_allocation = sum(
                allocation.get(config_key, {}).values()
            )

            # Determine minimum floor based on starvation level
            # Dec 29, 2025: Added ULTRA tier for critically starved configs
            if game_count < DATA_STARVATION_ULTRA_THRESHOLD:
                # ULTRA: <20 games - must get 3x allocation (highest priority)
                min_floor = int(games_per_config * 3.0)
                level = "ULTRA"
            elif game_count < DATA_STARVATION_EMERGENCY_THRESHOLD:
                # EMERGENCY: <100 games - must get 2x allocation
                min_floor = int(games_per_config * 2.0)
                level = "EMERGENCY"
            elif game_count < DATA_STARVATION_CRITICAL_THRESHOLD:
                # CRITICAL: <1000 games - must get 1.5x allocation
                min_floor = int(games_per_config * 1.5)
                level = "CRITICAL"
            elif game_count < DATA_POVERTY_THRESHOLD:
                # POVERTY (Dec 30, 2025): <5000 games - must get 1.25x allocation
                min_floor = int(games_per_config * 1.25)
                level = "POVERTY"
            else:
                # Not starved, no floor needed
                continue

            if current_allocation >= min_floor:
                continue  # Already meets floor

            # Need to allocate more for this starved config
            shortfall = min_floor - current_allocation

            # Allocate to available nodes
            additional_allocation = self._allocate_to_nodes(config_key, shortfall)

            if additional_allocation:
                if config_key not in allocation:
                    allocation[config_key] = {}

                # Merge with existing allocation
                for node_id, games in additional_allocation.items():
                    allocation[config_key][node_id] = (
                        allocation[config_key].get(node_id, 0) + games
                    )

                added_configs.append(
                    f"{config_key}({level}:{game_count}g→+{shortfall})"
                )
                logger.warning(
                    f"[SelfplayScheduler] Starvation floor: {config_key} "
                    f"({level}, {game_count} games) allocated +{shortfall} games"
                )

        if added_configs:
            logger.info(
                f"[SelfplayScheduler] Starvation floor enforcement: {', '.join(added_configs)}"
            )

        return allocation

    def _enforce_4p_allocation_minimums(
        self,
        allocation: dict[str, dict[str, int]],
        games_per_config: int,
    ) -> dict[str, dict[str, int]]:
        """Enforce minimum allocations for 3-player and 4-player configs.

        Dec 29, 2025 - Phase 4: Multiplayer allocation enforcement.
        Dec 30, 2025: Extended to 3p configs + increased targets.

        The 3p/4p multipliers can't be satisfied if cluster has limited GPUs.
        This method redistributes from 2p configs to ensure multiplayer gets minimum.

        Targets (Dec 30, 2025 - more aggressive):
        - 4p configs get at least 2.5x their proportional share
        - 3p configs get at least 1.5x their proportional share
        If short, steal from 2p configs (which are easiest to generate).

        Args:
            allocation: Current allocation {config_key: {node_id: games}}
            games_per_config: Base games per config

        Returns:
            Adjusted allocation with enforced multiplayer minimums
        """
        if not allocation:
            return allocation

        # Calculate per-config totals
        totals: dict[str, int] = {}
        for config_key, node_alloc in allocation.items():
            totals[config_key] = sum(node_alloc.values())

        # Identify configs by player count
        four_p_configs = [c for c in totals if "_4p" in c]
        three_p_configs = [c for c in totals if "_3p" in c]
        two_p_configs = [c for c in totals if "_2p" in c]

        if not two_p_configs:
            return allocation  # No donors available

        # Dec 30, 2025: More aggressive targets for multiplayer
        # 4p: 3.0x (was 2.5x, originally 1.5x) - these have fewest games
        # Jan 7, 2026: Increased to 3.0x for better 4-player model performance
        # Jan 14, 2026: Increased to 5.0x due to severe deficit (hex8_4p: 873 vs 16K)
        # 3p: 2.5x (was 1.5x) - also underrepresented
        min_4p_games = int(games_per_config * 5.0)
        min_3p_games = int(games_per_config * 2.5)
        redistributed = 0

        # Process 4p first (higher priority - most starved)
        for config in four_p_configs:
            current = totals.get(config, 0)
            shortfall = max(0, min_4p_games - current)

            if shortfall > 0:
                stolen = self._steal_from_donors(
                    allocation, totals, config, shortfall, two_p_configs, games_per_config
                )
                redistributed += stolen
                if stolen > 0:
                    logger.info(
                        f"[SelfplayScheduler] 4p enforcement: boosted {config} by {stolen} games "
                        f"(now {totals.get(config, 0)})"
                    )

        # Process 3p next (lower priority than 4p)
        for config in three_p_configs:
            current = totals.get(config, 0)
            shortfall = max(0, min_3p_games - current)

            if shortfall > 0:
                stolen = self._steal_from_donors(
                    allocation, totals, config, shortfall, two_p_configs, games_per_config
                )
                redistributed += stolen
                if stolen > 0:
                    logger.info(
                        f"[SelfplayScheduler] 3p enforcement: boosted {config} by {stolen} games "
                        f"(now {totals.get(config, 0)})"
                    )

        if redistributed > 0:
            logger.warning(
                f"[SelfplayScheduler] Multiplayer allocation enforcement: "
                f"redistributed {redistributed} games from 2p → 3p/4p configs"
            )

        return allocation

    def _steal_from_donors(
        self,
        allocation: dict[str, dict[str, int]],
        totals: dict[str, int],
        recipient: str,
        shortfall: int,
        donors: list[str],
        games_per_config: int,
    ) -> int:
        """Steal games from donor configs to fill recipient shortfall.

        Dec 30, 2025: Extracted helper for 3p/4p enforcement.

        Args:
            allocation: Current allocation dict
            totals: Current totals per config
            recipient: Config to receive games
            shortfall: Games needed
            donors: Configs to steal from
            games_per_config: Base allocation

        Returns:
            Number of games stolen
        """
        stolen_total = 0

        for donor in donors:
            if shortfall <= 0:
                break

            donor_current = totals.get(donor, 0)
            # Dec 30, 2025: 2p keeps at least 40% (was 50%) to allow more redistribution
            # Jan 14, 2026: Reduced to 20% to allow aggressive redistribution to 4p
            # 4p configs are at 5% of 2p levels and need urgent catchup
            donor_min = int(games_per_config * 0.2)

            available = max(0, donor_current - donor_min)
            steal = min(shortfall, available)

            if steal > 0 and donor in allocation:
                # Remove from donor (proportionally from nodes)
                for node_id in allocation[donor]:
                    node_games = allocation[donor][node_id]
                    if donor_current > 0:
                        node_steal = int(steal * node_games / donor_current)
                        allocation[donor][node_id] = max(0, node_games - node_steal)

                # Add to recipient config (to first available node)
                if recipient in allocation and allocation[recipient]:
                    first_node = next(iter(allocation[recipient]))
                    allocation[recipient][first_node] += steal
                elif recipient not in allocation:
                    # Find a node that had this config's board type
                    allocation[recipient] = {list(allocation[donor].keys())[0]: steal}

                totals[donor] -= steal
                totals[recipient] = totals.get(recipient, 0) + steal
                shortfall -= steal
                stolen_total += steal

                logger.debug(
                    f"[SelfplayScheduler] Multiplayer enforcement: {donor} -> {recipient}: {steal} games"
                )

        return stolen_total

    def _allocate_to_nodes(
        self,
        config_key: str,
        total_games: int,
    ) -> dict[str, int]:
        """Allocate games for a config across available nodes.

        December 2025 - Phase 2B.4: Ephemeral node short-job prioritization.
        Short jobs (square8, hex8) are boosted for ephemeral nodes.
        Long jobs (square19, hexagonal) are reduced for ephemeral nodes.

        December 2025 - P2P Integration: Excludes unhealthy nodes and applies
        cluster health factor to allocation.

        Args:
            config_key: Configuration key
            total_games: Total games to allocate

        Returns:
            Dict mapping node_id to num_games
        """
        # Get unhealthy nodes from P2P health tracking
        unhealthy_nodes = getattr(self, "_unhealthy_nodes", set())

        # January 5, 2026 (Phase 7.4): Get circuit-broken nodes from registry
        # This reduces failed dispatches from 30-40% to <5% by skipping
        # nodes with open circuit breakers that are known to be failing.
        cb_registry = get_node_circuit_registry()
        circuit_broken_nodes: set[str] = set()
        try:
            for node_id in self._node_capabilities.keys():
                if cb_registry.is_circuit_open(node_id):
                    circuit_broken_nodes.add(node_id)
            if circuit_broken_nodes:
                logger.debug(
                    f"[SelfplayScheduler] Excluding {len(circuit_broken_nodes)} "
                    f"circuit-broken nodes: {circuit_broken_nodes}"
                )
        except (ImportError, AttributeError, RuntimeError, KeyError, TypeError) as e:
            # Graceful fallback if CB registry unavailable or returns unexpected types
            logger.debug(f"[SelfplayScheduler] CB registry check failed: {e}")
            pass

        # Apply cluster health factor to total games
        cluster_health = getattr(self, "_cluster_health_factor", 1.0)
        if cluster_health < 1.0:
            adjusted_games = max(MIN_GAMES_PER_ALLOCATION, int(total_games * cluster_health))
            logger.debug(
                f"[SelfplayScheduler] Cluster health {cluster_health:.2f} reducing allocation: "
                f"{total_games} → {adjusted_games} games"
            )
            total_games = adjusted_games

        # Get available nodes sorted by capacity, excluding unhealthy and circuit-broken nodes
        available_nodes = sorted(
            [
                n for n in self._node_capabilities.values()
                if n.available_capacity > 0.1
                and n.node_id not in unhealthy_nodes  # Exclude unhealthy nodes
                and n.node_id not in circuit_broken_nodes  # Phase 7.4: Exclude CB nodes
            ],
            key=lambda n: (-n.available_capacity, n.data_lag_seconds),
        )

        if not available_nodes:
            return {}

        allocation: dict[str, int] = {}
        remaining = total_games

        # Phase 2B.4: Determine if this is a short job (quick selfplay)
        # Small boards complete in <10 minutes, good for ephemeral nodes
        is_short_job = config_key.startswith(("square8", "hex8"))

        # Calculate total capacity
        total_capacity = sum(n.available_capacity for n in available_nodes)

        for node in available_nodes:
            if remaining <= 0:
                break

            # Proportional allocation based on capacity
            proportion = node.available_capacity / total_capacity
            node_games = max(
                MIN_GAMES_PER_ALLOCATION,
                int(total_games * proportion),
            )

            # Adjust based on GPU type
            gpu_games = SELFPLAY_GAMES_PER_NODE.get(node.gpu_type, 500)
            node_games = min(node_games, gpu_games)

            # Phase 2B.4: Ephemeral node job-duration matching
            if node.is_ephemeral:
                if is_short_job:
                    # Boost allocation to ephemeral for short jobs
                    node_games = int(node_games * 1.5)
                    logger.debug(
                        f"[SelfplayScheduler] Boosted {node.node_id} (ephemeral) "
                        f"for short job {config_key}"
                    )
                else:
                    # Reduce for long jobs (risk of termination)
                    node_games = int(node_games * 0.5)
                    logger.debug(
                        f"[SelfplayScheduler] Reduced {node.node_id} (ephemeral) "
                        f"for long job {config_key}"
                    )

            # Cap at remaining games
            node_games = min(node_games, remaining)

            if node_games >= MIN_GAMES_PER_ALLOCATION:
                allocation[node.node_id] = node_games
                remaining -= node_games

        # Dec 29, 2025: Log node allocation breakdown for this config
        if allocation and logger.isEnabledFor(logging.DEBUG):
            total_assigned = sum(allocation.values())
            logger.debug(
                f"[SelfplayScheduler] Node allocation for {config_key}: "
                f"{total_assigned}/{total_games} games across {len(allocation)} nodes "
                f"({', '.join(f'{n}={g}' for n, g in allocation.items())})"
            )
        elif not allocation:
            logger.debug(
                f"[SelfplayScheduler] No allocation for {config_key}: "
                f"no nodes available (unhealthy={len(unhealthy_nodes)}, "
                f"total_requested={total_games})"
            )

        return allocation

    async def _update_node_capabilities(self) -> None:
        """Update node capability information from cluster.

        This is intentionally rate-limited because ClusterMonitor may probe remote
        hosts via subprocess/SSH. In unit tests and some callers, node capabilities
        may be pre-seeded directly on the instance; in that case we treat them as
        fresh and skip expensive probing.
        """
        now = time.time()

        # If capabilities were pre-seeded externally (common in tests) and we have
        # never performed a refresh, treat the injected snapshot as up-to-date.
        if self._node_capabilities and self._last_node_capability_update == 0.0:
            self._last_node_capability_update = now
            return

        if self._node_capabilities and (now - self._last_node_capability_update) < self._node_capability_update_interval:
            return

        # Mark refresh time up-front to avoid repeated expensive probes if the
        # monitor fails repeatedly.
        self._last_node_capability_update = now

        try:
            # Try getting from cluster monitor
            from app.coordination.cluster_status_monitor import ClusterMonitor

            monitor = ClusterMonitor()
            status = monitor.get_cluster_status()

            for node_id, node_status in status.nodes.items():
                if node_id not in self._node_capabilities:
                    self._node_capabilities[node_id] = NodeCapability(node_id=node_id)

                cap = self._node_capabilities[node_id]
                cap.gpu_type = node_status.gpu or "unknown"
                cap.gpu_memory_gb = node_status.gpu_memory_total_gb
                cap.is_ephemeral = is_ephemeral_node(node_id)
                cap.current_load = node_status.gpu_utilization_percent / 100.0
                cap.data_lag_seconds = node_status.sync_lag_seconds
                # Dec 29 2025: Track job count for dynamic weight calculation
                cap.current_jobs = getattr(node_status, 'selfplay_jobs', 0) or 0

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error updating node capabilities: {e}")

    # =========================================================================
    # External Boost Interface (December 2025)
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
            from app.coordination.event_router import safe_emit_event

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

    def subscribe_to_events(self) -> None:
        """Subscribe to relevant pipeline events.

        December 30, 2025: This method is retained for backward compatibility.
        New code should use start() which automatically subscribes via
        _get_event_subscriptions(). This method delegates to the HandlerBase
        subscription infrastructure.
        """
        if self._subscribed:
            return

        try:
            # Dec 2025 fix: Use get_router() instead of get_event_bus() because:
            # 1. get_router() always returns a valid UnifiedEventRouter singleton
            # 2. get_event_bus() can return None if data_events module unavailable
            # 3. UnifiedEventRouter handles event type normalization automatically
            from app.coordination.event_router import DataEventType, get_router

            router = get_router()

            # Dec 2025: Use per-subscription error handling to ensure one failure
            # doesn't prevent other subscriptions from being registered
            subscribed_count = 0
            failed_count = 0

            def _safe_subscribe(event_type, handler, name: str) -> bool:
                """Subscribe with individual error handling."""
                nonlocal subscribed_count, failed_count
                try:
                    # UnifiedEventRouter.subscribe() handles both enum and string types
                    router.subscribe(event_type, handler)
                    subscribed_count += 1
                    return True
                except Exception as e:
                    failed_count += 1
                    logger.warning(f"[SelfplayScheduler] Failed to subscribe to {name}: {e}")
                    return False

            # Core subscriptions (always attempt)
            _safe_subscribe(DataEventType.SELFPLAY_COMPLETE, self._on_selfplay_complete, "SELFPLAY_COMPLETE")
            _safe_subscribe(DataEventType.TRAINING_COMPLETED, self._on_training_complete, "TRAINING_COMPLETED")
            _safe_subscribe(DataEventType.MODEL_PROMOTED, self._on_promotion_complete, "MODEL_PROMOTED")
            # Phase 5: Subscribe to feedback events
            _safe_subscribe(DataEventType.SELFPLAY_TARGET_UPDATED, self._on_selfplay_target_updated, "SELFPLAY_TARGET_UPDATED")
            _safe_subscribe(DataEventType.QUALITY_DEGRADED, self._on_quality_degraded, "QUALITY_DEGRADED")
            # Phase 4A.1: Subscribe to curriculum rebalancing (December 2025)
            _safe_subscribe(DataEventType.CURRICULUM_REBALANCED, self._on_curriculum_rebalanced, "CURRICULUM_REBALANCED")
            # P0.1 (Dec 2025): Subscribe to SELFPLAY_RATE_CHANGED from FeedbackAccelerator
            _safe_subscribe(DataEventType.SELFPLAY_RATE_CHANGED, self._on_selfplay_rate_changed, "SELFPLAY_RATE_CHANGED")
            # P1.1 (Dec 2025): Subscribe to TRAINING_BLOCKED_BY_QUALITY for selfplay acceleration
            _safe_subscribe(DataEventType.TRAINING_BLOCKED_BY_QUALITY, self._on_training_blocked_by_quality, "TRAINING_BLOCKED_BY_QUALITY")

            # Optional subscriptions (graceful degradation if event type doesn't exist)
            if hasattr(DataEventType, 'OPPONENT_MASTERED'):
                _safe_subscribe(DataEventType.OPPONENT_MASTERED, self._on_opponent_mastered, "OPPONENT_MASTERED")
            if hasattr(DataEventType, 'TRAINING_EARLY_STOPPED'):
                _safe_subscribe(DataEventType.TRAINING_EARLY_STOPPED, self._on_training_early_stopped, "TRAINING_EARLY_STOPPED")
            if hasattr(DataEventType, 'ELO_VELOCITY_CHANGED'):
                _safe_subscribe(DataEventType.ELO_VELOCITY_CHANGED, self._on_elo_velocity_changed, "ELO_VELOCITY_CHANGED")
            if hasattr(DataEventType, 'EXPLORATION_BOOST'):
                _safe_subscribe(DataEventType.EXPLORATION_BOOST, self._on_exploration_boost, "EXPLORATION_BOOST")
            # Jan 7, 2026: Subscribe to EXPLORATION_ADJUSTED for quality-driven exploration
            if hasattr(DataEventType, 'EXPLORATION_ADJUSTED'):
                _safe_subscribe(DataEventType.EXPLORATION_ADJUSTED, self._on_exploration_adjusted, "EXPLORATION_ADJUSTED")
            # Dec 2025: Subscribe to CURRICULUM_ADVANCED for curriculum progression
            if hasattr(DataEventType, 'CURRICULUM_ADVANCED'):
                _safe_subscribe(DataEventType.CURRICULUM_ADVANCED, self._on_curriculum_advanced, "CURRICULUM_ADVANCED")
            # Dec 2025: Subscribe to ADAPTIVE_PARAMS_CHANGED for parameter adjustments
            if hasattr(DataEventType, 'ADAPTIVE_PARAMS_CHANGED'):
                _safe_subscribe(DataEventType.ADAPTIVE_PARAMS_CHANGED, self._on_adaptive_params_changed, "ADAPTIVE_PARAMS_CHANGED")
            if hasattr(DataEventType, 'LOW_QUALITY_DATA_WARNING'):
                _safe_subscribe(DataEventType.LOW_QUALITY_DATA_WARNING, self._on_low_quality_warning, "LOW_QUALITY_DATA_WARNING")

            # P2P cluster health events (December 2025)
            if hasattr(DataEventType, 'NODE_UNHEALTHY'):
                _safe_subscribe(DataEventType.NODE_UNHEALTHY, self._on_node_unhealthy, "NODE_UNHEALTHY")
            if hasattr(DataEventType, 'NODE_RECOVERED'):
                _safe_subscribe(DataEventType.NODE_RECOVERED, self._on_node_recovered, "NODE_RECOVERED")
            # Dec 28, 2025: NODE_ACTIVATED from cluster activator/watchdog also means node is available
            if hasattr(DataEventType, 'NODE_ACTIVATED'):
                _safe_subscribe(DataEventType.NODE_ACTIVATED, self._on_node_recovered, "NODE_ACTIVATED")
            if hasattr(DataEventType, 'P2P_NODE_DEAD'):
                _safe_subscribe(DataEventType.P2P_NODE_DEAD, self._on_node_unhealthy, "P2P_NODE_DEAD")
            if hasattr(DataEventType, 'P2P_CLUSTER_UNHEALTHY'):
                _safe_subscribe(DataEventType.P2P_CLUSTER_UNHEALTHY, self._on_cluster_unhealthy, "P2P_CLUSTER_UNHEALTHY")
            if hasattr(DataEventType, 'P2P_CLUSTER_HEALTHY'):
                _safe_subscribe(DataEventType.P2P_CLUSTER_HEALTHY, self._on_cluster_healthy, "P2P_CLUSTER_HEALTHY")
            if hasattr(DataEventType, 'HOST_OFFLINE'):
                _safe_subscribe(DataEventType.HOST_OFFLINE, self._on_host_offline, "HOST_OFFLINE")
            # Dec 2025: NODE_TERMINATED from idle shutdown - reuse host_offline handler
            if hasattr(DataEventType, 'NODE_TERMINATED'):
                _safe_subscribe(DataEventType.NODE_TERMINATED, self._on_host_offline, "NODE_TERMINATED")

            # Jan 3, 2026: Subscribe to voter demotion/promotion for allocation adjustment
            # When a voter is demoted, it signals potential node health issues
            if hasattr(DataEventType, 'VOTER_DEMOTED'):
                _safe_subscribe(DataEventType.VOTER_DEMOTED, self._on_voter_demoted, "VOTER_DEMOTED")
            if hasattr(DataEventType, 'VOTER_PROMOTED'):
                _safe_subscribe(DataEventType.VOTER_PROMOTED, self._on_voter_promoted, "VOTER_PROMOTED")

            # Jan 3, 2026 Session 10: Subscribe to CIRCUIT_RESET for proactive recovery monitoring
            # When a circuit breaker is reset after proactive health probe, boost node priority
            if hasattr(DataEventType, 'CIRCUIT_RESET'):
                _safe_subscribe(DataEventType.CIRCUIT_RESET, self._on_circuit_reset, "CIRCUIT_RESET")

            # Dec 2025: Subscribe to regression events for curriculum rebalancing
            if hasattr(DataEventType, 'REGRESSION_DETECTED'):
                _safe_subscribe(DataEventType.REGRESSION_DETECTED, self._on_regression_detected, "REGRESSION_DETECTED")

            # Dec 29, 2025: Subscribe to backpressure events for reactive scheduling
            if hasattr(DataEventType, 'BACKPRESSURE_ACTIVATED'):
                _safe_subscribe(DataEventType.BACKPRESSURE_ACTIVATED, self._on_backpressure_activated, "BACKPRESSURE_ACTIVATED")
            if hasattr(DataEventType, 'BACKPRESSURE_RELEASED'):
                _safe_subscribe(DataEventType.BACKPRESSURE_RELEASED, self._on_backpressure_released, "BACKPRESSURE_RELEASED")
            # Jan 2026 Sprint 10: Subscribe to evaluation-specific backpressure
            # When evaluation queue is backlogged, slow down selfplay to reduce queue pressure
            if hasattr(DataEventType, 'EVALUATION_BACKPRESSURE'):
                _safe_subscribe(DataEventType.EVALUATION_BACKPRESSURE, self._on_evaluation_backpressure, "EVALUATION_BACKPRESSURE")
            if hasattr(DataEventType, 'EVALUATION_BACKPRESSURE_RELEASED'):
                _safe_subscribe(DataEventType.EVALUATION_BACKPRESSURE_RELEASED, self._on_backpressure_released, "EVALUATION_BACKPRESSURE_RELEASED")
            # Dec 29, 2025: Subscribe to NODE_OVERLOADED for per-node backpressure
            if hasattr(DataEventType, 'NODE_OVERLOADED'):
                _safe_subscribe(DataEventType.NODE_OVERLOADED, self._on_node_overloaded, "NODE_OVERLOADED")

            # Dec 29, 2025 - Phase 2: Subscribe to ELO_UPDATED for velocity tracking
            if hasattr(DataEventType, 'ELO_UPDATED'):
                _safe_subscribe(DataEventType.ELO_UPDATED, self._on_elo_updated, "ELO_UPDATED")

            # Dec 29, 2025: Subscribe to progress stall events for 48h autonomous operation
            if hasattr(DataEventType, 'PROGRESS_STALL_DETECTED'):
                _safe_subscribe(DataEventType.PROGRESS_STALL_DETECTED, self._on_progress_stall, "PROGRESS_STALL_DETECTED")
            if hasattr(DataEventType, 'PROGRESS_RECOVERED'):
                _safe_subscribe(DataEventType.PROGRESS_RECOVERED, self._on_progress_recovered, "PROGRESS_RECOVERED")

            # Dec 29, 2025: Subscribe to architecture weight updates for allocation adjustment
            if hasattr(DataEventType, 'ARCHITECTURE_WEIGHTS_UPDATED'):
                _safe_subscribe(DataEventType.ARCHITECTURE_WEIGHTS_UPDATED, self._on_architecture_weights_updated, "ARCHITECTURE_WEIGHTS_UPDATED")

            # Dec 30, 2025: Subscribe to quality feedback for immediate cache invalidation
            # This enables faster response to quality degradation (+12-18 Elo improvement)
            if hasattr(DataEventType, 'QUALITY_FEEDBACK_ADJUSTED'):
                _safe_subscribe(DataEventType.QUALITY_FEEDBACK_ADJUSTED, self._on_quality_feedback_adjusted, "QUALITY_FEEDBACK_ADJUSTED")

            self._subscribed = subscribed_count > 0
            if self._subscribed:
                logger.info(
                    f"[SelfplayScheduler] Subscribed to {subscribed_count} pipeline events "
                    f"(failed: {failed_count}, includes P2P health events)"
                )
            else:
                logger.warning("[SelfplayScheduler] No subscriptions succeeded, reactive scheduling disabled")

        except ImportError as e:
            logger.warning(f"[SelfplayScheduler] Event router unavailable: {e}")
        except Exception as e:
            logger.warning(f"[SelfplayScheduler] Failed to subscribe to events: {e}")

    def _on_selfplay_complete(self, event: Any) -> None:
        """Handle selfplay completion event."""
        try:
            config_key = extract_config_key(event.payload)
            if config_key in self._config_priorities:
                # Reset staleness for this config
                self._config_priorities[config_key].staleness_hours = 0.0

                # January 2026 Sprint 10: Record opponent type for diversity tracking
                # Extract opponent type from event (e.g., "heuristic", "policy", "gumbel", "nnue")
                opponent_type = event.payload.get("opponent_type") or event.payload.get("engine_mode")
                if opponent_type:
                    self.record_opponent(config_key, str(opponent_type))
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling selfplay complete: {e}")

    def _on_training_complete(self, event: Any) -> None:
        """Handle training completion event."""
        try:
            config_key = extract_config_key(event.payload)
            if config_key in self._config_priorities:
                # Clear training pending flag
                self._config_priorities[config_key].training_pending = False
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling training complete: {e}")

    def _on_promotion_complete(self, event: Any) -> None:
        """Handle promotion completion event."""
        try:
            config_key = extract_config_key(event.payload)
            success = event.payload.get("success", False)

            if config_key in self._config_priorities:
                priority = self._config_priorities[config_key]
                if success:
                    # Boost for continued improvement
                    priority.exploration_boost = 1.0
                else:
                    # Increase exploration on failure
                    priority.exploration_boost = min(2.0, priority.exploration_boost * 1.3)
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling promotion complete: {e}")

    def _on_selfplay_target_updated(self, event: Any) -> None:
        """Handle selfplay target update request.

        Phase 5 (Dec 2025): Responds to requests for more/fewer selfplay games.
        Typically emitted when training needs more data urgently.

        Dec 28 2025: Extended to handle search_budget and velocity feedback
        from FeedbackLoopController for reaching 2000+ Elo.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            target_games = payload.get("target_games", 0)
            priority_val = payload.get("priority", "normal")
            search_budget = payload.get("search_budget", 0)
            exploration_boost = payload.get("exploration_boost", 1.0)
            velocity = payload.get("velocity", 0.0)
            reason = payload.get("reason", "")

            if config_key in self._config_priorities:
                priority = self._config_priorities[config_key]

                # Dec 28 2025: Apply search budget from velocity feedback
                if search_budget > 0 and reason == "velocity_feedback":
                    old_budget = getattr(priority, "search_budget", 400)
                    priority.search_budget = search_budget
                    logger.info(
                        f"[SelfplayScheduler] Updating {config_key} search budget: "
                        f"{old_budget}→{search_budget} (velocity={velocity:.1f} Elo/hr)"
                    )

                # Apply exploration boost
                if exploration_boost != 1.0:
                    priority.exploration_boost = exploration_boost

                # Boost priority based on urgency
                if priority_val.upper() == "HIGH":
                    priority.training_pending = True
                    priority.exploration_boost = max(1.2, priority.exploration_boost)
                    logger.info(
                        f"[SelfplayScheduler] Boosting {config_key} priority "
                        f"(target: {target_games} games, priority: {priority_val})"
                    )
                elif priority_val.lower() == "low":
                    # Reduce priority for this config
                    priority.exploration_boost = min(0.8, priority.exploration_boost)
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling selfplay target: {e}")

    # =========================================================================
    # Quality Signal Event Handlers - EXTRACTED to SelfplayQualitySignalMixin
    # January 2026 Sprint 17.9: Quality handlers moved to quality_signal_handler.py
    # Mixin provides: _on_quality_degraded, _on_regression_detected,
    #                 _on_training_blocked_by_quality, _on_quality_feedback_adjusted,
    #                 _on_opponent_mastered, _on_training_early_stopped,
    #                 _on_low_quality_warning, _emit_quality_penalty_applied
    # =========================================================================

    def _on_curriculum_rebalanced(self, event: Any) -> None:
        """Handle curriculum rebalancing event.

        Phase 4A.1 (December 2025): Updates priority weights when curriculum
        feedback adjusts config priorities based on training progress.

        December 30, 2025: Enhanced to handle regression-triggered curriculum
        updates with factor-based allocation reduction.

        December 2025 Guard: Skip events originated by SelfplayScheduler itself
        to prevent echo loops in the event system.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event

            # Loop guard: Skip events we emitted (prevents echo loops)
            source = payload.get("source", "")
            if source == "selfplay_scheduler":
                logger.debug("[SelfplayScheduler] Skipping self-originated CURRICULUM_REBALANCED")
                return

            trigger = payload.get("trigger", "")
            config_key = extract_config_key(payload)
            new_weight = payload.get("weight", 1.0)
            reason = payload.get("reason", "")

            # December 30, 2025: Handle regression-triggered curriculum emergency updates
            if trigger == "regression_detected":
                changed_configs = payload.get("changed_configs", [])
                factor = payload.get("factor", 0.5)
                elo_loss = payload.get("elo_loss", 0.0)

                for cfg in changed_configs:
                    if cfg in self._config_priorities:
                        old_weight = self._config_priorities[cfg].curriculum_weight
                        new_wt = old_weight * factor
                        self._config_priorities[cfg].curriculum_weight = new_wt
                        # Also boost exploration to encourage diversity
                        self._config_priorities[cfg].exploration_boost = min(
                            2.0, self._config_priorities[cfg].exploration_boost * 1.5
                        )
                        logger.warning(
                            f"[SelfplayScheduler] Regression-triggered curriculum reduction: "
                            f"{cfg} weight {old_weight:.2f} → {new_wt:.2f} (factor={factor}, "
                            f"elo_loss={elo_loss:.0f}), exploration boosted"
                        )
                return  # Handled regression case

            # Session 17.11: Handle quality critical drop - boost allocation for affected config
            # This enables immediate response to quality degradation (+8-12 Elo improvement)
            if trigger == "quality_critical_drop":
                changed_configs = payload.get("changed_configs", [])
                factor = payload.get("factor", 1.5)
                quality_drop = payload.get("quality_drop", 0.0)

                for cfg in changed_configs:
                    if cfg in self._config_priorities:
                        old_weight = self._config_priorities[cfg].curriculum_weight
                        # Boost allocation to prioritize data generation
                        new_wt = min(2.0, old_weight * factor)
                        self._config_priorities[cfg].curriculum_weight = new_wt
                        # Also increase exploration boost to improve diversity
                        self._config_priorities[cfg].exploration_boost = min(
                            2.0, self._config_priorities[cfg].exploration_boost * 1.3
                        )
                        logger.warning(
                            f"[SelfplayScheduler] Quality-drop allocation boost: "
                            f"{cfg} weight {old_weight:.2f} → {new_wt:.2f} (factor={factor}, "
                            f"quality_drop={quality_drop:.2f}), exploration boosted"
                        )
                return  # Handled quality_critical_drop case

            if config_key in self._config_priorities:
                old_weight = self._config_priorities[config_key].curriculum_weight
                self._config_priorities[config_key].curriculum_weight = new_weight

                # Only log significant changes
                if abs(new_weight - old_weight) > 0.1:
                    logger.info(
                        f"[SelfplayScheduler] Curriculum rebalanced: {config_key} "
                        f"weight {old_weight:.2f} → {new_weight:.2f}"
                        + (f" ({reason})" if reason else "")
                    )

            # Also handle batch updates (multiple configs at once)
            weights = payload.get("weights", {})
            if weights:
                for cfg, weight in weights.items():
                    if cfg in self._config_priorities:
                        old_w = self._config_priorities[cfg].curriculum_weight
                        self._config_priorities[cfg].curriculum_weight = weight
                        if abs(weight - old_w) > 0.1:
                            logger.info(
                                f"[SelfplayScheduler] Curriculum batch update: "
                                f"{cfg} weight {old_w:.2f} → {weight:.2f}"
                            )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling curriculum rebalanced: {e}")

    def _on_selfplay_rate_changed(self, event: Any) -> None:
        """Handle selfplay rate change event from FeedbackAccelerator.

        P0.1 (Dec 2025): Closes the Elo momentum → Selfplay rate feedback loop.
        FeedbackAccelerator emits SELFPLAY_RATE_CHANGED when it detects:
        - ACCELERATING: 1.5x multiplier (capitalize on positive momentum)
        - IMPROVING: 1.25x multiplier (boost for continued improvement)
        - STABLE: 1.0x multiplier (normal rate)
        - PLATEAU: 1.1x multiplier (slight boost to break plateau)
        - REGRESSING: 0.75x multiplier (reduce noise, focus on quality)
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            new_rate = payload.get("new_rate", 1.0) or payload.get("rate_multiplier", 1.0)
            momentum_state = payload.get("momentum_state", "unknown")

            if config_key in self._config_priorities:
                old_rate = self._config_priorities[config_key].momentum_multiplier
                self._config_priorities[config_key].momentum_multiplier = float(new_rate)

                # Log significant changes
                if abs(new_rate - old_rate) > 0.1:
                    logger.info(
                        f"[SelfplayScheduler] Selfplay rate changed: {config_key} "
                        f"multiplier {old_rate:.2f} → {new_rate:.2f} "
                        f"(momentum: {momentum_state})"
                    )
            else:
                logger.debug(
                    f"[SelfplayScheduler] Received rate change for unknown config: {config_key}"
                )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling selfplay rate changed: {e}")

    # =========================================================================
    # Memory Pressure Event Handlers (January 13, 2026)
    # =========================================================================

    def _on_memory_pressure(self, event: Any) -> None:
        """Handle MEMORY_PRESSURE event - pause selfplay when memory critical.

        January 13, 2026: Added to prevent OOM and allow cleanup daemons time to
        free disk space. When memory pressure is CRITICAL or EMERGENCY, we pause
        selfplay allocation to reduce memory load.

        Event payload:
            tier: str - "CAUTION", "WARNING", "CRITICAL", or "EMERGENCY"
            source: str - "coordinator", "gpu_vram", "system_ram", etc.
            node_id: str - Which node is under pressure (optional)
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            tier = payload.get("tier", "").upper()
            source = payload.get("source", "unknown")
            node_id = payload.get("node_id", "")

            if tier in ("CRITICAL", "EMERGENCY"):
                if not self._memory_constrained:
                    logger.warning(
                        f"[SelfplayScheduler] Memory pressure {tier} ({source}), "
                        f"pausing selfplay allocation"
                    )
                self._memory_constrained = True
                self._memory_constraint_source = f"{source}:{node_id}" if node_id else source
            elif tier in ("CAUTION", "WARNING"):
                # Resume on lower pressure tiers
                if self._memory_constrained:
                    logger.info(
                        f"[SelfplayScheduler] Memory pressure reduced to {tier}, "
                        f"resuming selfplay allocation"
                    )
                self._memory_constrained = False
                self._memory_constraint_source = ""

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling memory pressure: {e}")

    def _on_resource_constraint(self, event: Any) -> None:
        """Handle RESOURCE_CONSTRAINT event - general resource limits.

        January 13, 2026: Handles general resource constraints (disk space, etc.)
        that may require pausing selfplay to allow recovery.

        Event payload:
            resource_type: str - "disk", "memory", "gpu_vram", etc.
            level: str - "warning", "critical", "emergency"
            node_id: str - Which node is constrained (optional)
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            level = payload.get("level", "").lower()
            resource_type = payload.get("resource_type", "unknown")
            node_id = payload.get("node_id", "")

            if level in ("critical", "emergency"):
                if not self._memory_constrained:
                    logger.warning(
                        f"[SelfplayScheduler] Resource constraint {level} "
                        f"({resource_type}), pausing selfplay allocation"
                    )
                self._memory_constrained = True
                self._memory_constraint_source = f"{resource_type}:{node_id}" if node_id else resource_type
            elif level in ("normal", "ok", "released"):
                if self._memory_constrained:
                    logger.info(
                        f"[SelfplayScheduler] Resource constraint released ({resource_type}), "
                        f"resuming selfplay allocation"
                    )
                self._memory_constrained = False
                self._memory_constraint_source = ""

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling resource constraint: {e}")

    # =========================================================================
    # Velocity/Exploration Event Handlers - EXTRACTED to SelfplayVelocityMixin
    # January 2026 Sprint 17.9: Velocity handlers moved to velocity_mixin.py
    # Mixin provides: _on_elo_velocity_changed, _on_exploration_boost,
    #                 _on_curriculum_advanced, _on_adaptive_params_changed,
    #                 _decay_expired_boosts
    # =========================================================================

    # =========================================================================
    # P2P Cluster Health Event Handlers - EXTRACTED to SelfplayHealthMonitorMixin
    # January 2026 Sprint 17.4: Health handlers moved to selfplay_health_monitor.py
    # Mixin provides: _on_node_unhealthy, _on_node_recovered, _on_host_offline,
    #                 _on_voter_demoted, _on_voter_promoted, _on_circuit_reset,
    #                 _on_cluster_unhealthy, _on_cluster_healthy, _on_p2p_restarted,
    #                 _on_backpressure_activated, _on_evaluation_backpressure,
    #                 _on_backpressure_released, _on_node_overloaded,
    #                 _on_progress_stall, _on_progress_recovered
    # Also: is_node_under_backoff(), get_overloaded_nodes(), is_node_healthy(),
    #       get_cluster_health_factor(), is_backpressure_active()
    # =========================================================================

    # =========================================================================
    # Elo/Diversity Handlers - EXTRACTED to SelfplayVelocityMixin
    # January 2026 Sprint 17.9: Elo handlers moved to velocity_mixin.py
    # Mixin provides: _on_elo_updated, _emit_plateau_detected,
    #                 _on_architecture_weights_updated, get_elo_velocity,
    #                 initialize_elo_velocities_from_db, record_opponent,
    #                 _compute_diversity_score, get_diversity_score,
    #                 get_opponent_types_seen
    # =========================================================================

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

        December 2025: Notifies downstream consumers (IdleResourceDaemon, feedback
        loops) when selfplay allocation has changed. This enables:
        - IdleResourceDaemon to know which configs are prioritized
        - Feedback loops to track allocation changes from their signals
        - Monitoring to track allocation patterns over time

        Args:
            allocation: Dict of config_key -> {node_id: games} for batch allocations
            total_games: Total games in this allocation
            trigger: What caused this allocation (e.g., "allocate_batch", "exploration_boost")
            config_key: Specific config that changed (for single-config updates)
        """
        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()
            if bus is None:
                return

            # Build allocation summary
            if allocation:
                configs_allocated = list(allocation.keys())
                nodes_involved = set()
                for node_games in allocation.values():
                    nodes_involved.update(node_games.keys())
            else:
                configs_allocated = [config_key] if config_key else []
                nodes_involved = set()

            payload = {
                "trigger": trigger,
                "total_games": total_games,
                "configs": configs_allocated,
                "node_count": len(nodes_involved),
                "timestamp": time.time(),
            }

            # Include exploration boosts for tracking feedback loop efficacy
            if config_key and config_key in self._config_priorities:
                priority = self._config_priorities[config_key]
                payload["exploration_boost"] = priority.exploration_boost
                payload["curriculum_weight"] = priority.curriculum_weight

            bus.emit(DataEventType.SELFPLAY_ALLOCATION_UPDATED, payload)
            logger.debug(
                f"[SelfplayScheduler] Emitted SELFPLAY_ALLOCATION_UPDATED: "
                f"trigger={trigger}, games={total_games}, configs={len(configs_allocated)}"
            )

        except ImportError:
            pass  # Event system not available
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Failed to emit allocation update: {e}")

    def _emit_starvation_alert(
        self,
        config_key: str,
        game_count: int,
        tier: str,
    ) -> None:
        """Emit DATA_STARVATION_CRITICAL event to trigger priority dispatch.

        Jan 5, 2026: Added for automatic starvation response. When ULTRA starvation
        is detected (<20 games), this event enables QueuePopulatorLoop to auto-submit
        priority selfplay jobs without manual intervention.

        Args:
            config_key: Config with starvation (e.g., "square19_3p")
            game_count: Current game count for this config
            tier: Starvation tier ("ULTRA", "EMERGENCY", "CRITICAL")
        """
        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()
            if bus is None:
                return

            payload = {
                "config_key": config_key,
                "game_count": game_count,
                "tier": tier,
                "threshold": DATA_STARVATION_ULTRA_THRESHOLD,
                "multiplier": DATA_STARVATION_ULTRA_MULTIPLIER,
                "timestamp": time.time(),
            }

            bus.emit(DataEventType.DATA_STARVATION_CRITICAL, payload)
            logger.info(
                f"[SelfplayScheduler] Emitted DATA_STARVATION_CRITICAL: "
                f"{config_key} ({tier} tier, {game_count} games)"
            )

        except ImportError:
            pass  # Event system not available
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Failed to emit starvation alert: {e}")

    # =========================================================================
    # Idle Node Work Injection (Jan 5, 2026 - Sprint 17.30)
    # =========================================================================

    def _is_cpu_only_node(self, node_id: str) -> bool:
        """Check if a node is CPU-only (no GPU).

        Jan 5, 2026: CPU-only nodes (like Hetzner) can contribute
        heuristic-only selfplay data. This helper identifies them
        for appropriate work assignment.

        Args:
            node_id: Node identifier to check

        Returns:
            True if node has no GPU, False otherwise
        """
        cap = self._node_capabilities.get(node_id)
        if not cap:
            return False
        return cap.gpu_memory_gb == 0

    def _update_node_idle_tracking(self) -> None:
        """Update idle tracking state for all nodes.

        Called during node capability updates. Tracks when each node
        first became idle (current_jobs == 0).

        Jan 5, 2026: Part of idle node work injection feature.
        Jan 5, 2026 (Task 8.4): Now includes CPU-only nodes for heuristic selfplay.
        """
        now = time.time()

        for node_id, cap in self._node_capabilities.items():
            # Jan 5, 2026: Include CPU-only nodes now that they can run heuristic selfplay
            # Previously skipped CPU-only nodes, but they can contribute training data

            if cap.current_jobs == 0:
                # Node is idle - start tracking if not already
                if node_id not in self._node_idle_since:
                    self._node_idle_since[node_id] = now
                    logger.debug(
                        f"[SelfplayScheduler] Node {node_id} became idle, "
                        f"will inject work after {self._idle_threshold_seconds}s"
                    )
            else:
                # Node is working - clear idle tracking
                if node_id in self._node_idle_since:
                    del self._node_idle_since[node_id]

    def _detect_idle_nodes(self) -> list[str]:
        """Get list of nodes that have been idle longer than threshold.

        Returns:
            List of node IDs that are idle and eligible for work injection
        """
        now = time.time()
        idle_nodes = []

        # Get unhealthy nodes to skip
        unhealthy_nodes = getattr(self, "_unhealthy_nodes", set())

        for node_id, idle_since in self._node_idle_since.items():
            idle_duration = now - idle_since

            if idle_duration >= self._idle_threshold_seconds:
                # Skip unhealthy nodes
                if node_id in unhealthy_nodes:
                    logger.debug(
                        f"[SelfplayScheduler] Skipping unhealthy idle node: {node_id}"
                    )
                    continue

                idle_nodes.append(node_id)
                logger.debug(
                    f"[SelfplayScheduler] Node {node_id} idle for {idle_duration:.0f}s "
                    f"(threshold: {self._idle_threshold_seconds}s)"
                )

        return idle_nodes

    def _get_most_underserved_config(self) -> str | None:
        """Get the config with lowest game count that needs more data.

        Jan 5, 2026: Used for idle node work injection. Prioritizes
        configs in ULTRA/EMERGENCY starvation tiers.

        Returns:
            Config key of most underserved config, or None if all are healthy
        """
        most_underserved = None
        lowest_game_count = float("inf")

        for config_key, priority in self._config_priorities.items():
            game_count = priority.game_count

            # Skip configs that are adequately served
            if game_count >= DATA_STARVATION_CRITICAL_THRESHOLD:
                continue

            # Prefer the config with lowest game count
            if game_count < lowest_game_count:
                lowest_game_count = game_count
                most_underserved = config_key

        return most_underserved

    async def inject_work_for_idle_nodes(self) -> dict[str, Any]:
        """Inject priority selfplay work for nodes that have been idle.

        Jan 5, 2026 - Sprint 17.30: When GPU nodes are idle for more than
        IDLE_THRESHOLD_SECONDS (default 5 min), allocate emergency selfplay
        work for the most underserved config.

        This improves:
        - GPU utilization (+15-25% during backpressure periods)
        - Elo improvement (+8-15 Elo from additional data for starved configs)
        - Cluster efficiency (idle resources put to productive use)

        Returns:
            Dict with injection statistics
        """
        now = time.time()

        # Rate limit work injection
        if now - self._last_idle_injection < self._idle_injection_cooldown:
            return {"skipped": True, "reason": "cooldown"}

        # Update idle tracking from latest capabilities
        self._update_node_idle_tracking()

        # Detect nodes that are idle past threshold
        idle_nodes = self._detect_idle_nodes()

        if not idle_nodes:
            return {"idle_nodes": 0, "injected": 0}

        # Get most underserved config to work on
        target_config = self._get_most_underserved_config()

        if not target_config:
            # All configs have sufficient data
            logger.debug(
                f"[SelfplayScheduler] {len(idle_nodes)} idle nodes but no underserved configs"
            )
            return {
                "idle_nodes": len(idle_nodes),
                "injected": 0,
                "reason": "no_underserved_configs",
            }

        # Inject work for idle nodes
        injected_count = 0
        cpu_node_count = 0
        games_per_node = 50  # Emergency allocation per node
        # Jan 5, 2026 (Task 8.4): Lower batch for CPU-only nodes (heuristic is slower)
        games_per_cpu_node = 25

        allocation: dict[str, dict[str, int]] = {target_config: {}}

        for node_id in idle_nodes:
            # Jan 5, 2026 (Task 8.4): CPU nodes get fewer games (heuristic is slower)
            is_cpu = self._is_cpu_only_node(node_id)
            node_games = games_per_cpu_node if is_cpu else games_per_node

            # Allocate emergency games to this node
            allocation[target_config][node_id] = node_games
            injected_count += 1
            if is_cpu:
                cpu_node_count += 1

            # Clear idle tracking since we're injecting work
            if node_id in self._node_idle_since:
                del self._node_idle_since[node_id]

        self._last_idle_injection = now

        # Log the injection
        priority = self._config_priorities.get(target_config)
        game_count = priority.game_count if priority else 0
        total_games = sum(allocation[target_config].values())

        # Jan 5, 2026 (Task 8.4): Log CPU vs GPU node breakdown
        gpu_node_count = injected_count - cpu_node_count
        node_breakdown = f"{gpu_node_count} GPU, {cpu_node_count} CPU" if cpu_node_count > 0 else f"{gpu_node_count} GPU"

        logger.info(
            f"[SelfplayScheduler] Idle work injection: "
            f"{len(idle_nodes)} idle nodes ({node_breakdown}) → {target_config} ({game_count} games, "
            f"+{total_games} games allocated)"
        )

        # Emit event for observability
        try:
            from app.coordination.event_router import DataEventType, emit_event

            emit_event(
                DataEventType.IDLE_NODE_WORK_INJECTED,
                {
                    "idle_nodes": idle_nodes,
                    "target_config": target_config,
                    "games_injected": total_games,
                    "timestamp": now,
                    # Jan 5, 2026 (Task 8.4): Include CPU vs GPU breakdown
                    "gpu_nodes": gpu_node_count,
                    "cpu_nodes": cpu_node_count,
                },
            )
        except (ImportError, AttributeError):
            pass

        return {
            "idle_nodes": len(idle_nodes),
            "injected": injected_count,
            "target_config": target_config,
            "games_per_node": games_per_node,
            "games_per_cpu_node": games_per_cpu_node,
            "total_games": total_games,
            "gpu_nodes": gpu_node_count,
            "cpu_nodes": cpu_node_count,
        }

    # =========================================================================
    # Per-Node Job Targeting (Dec 2025)
    # =========================================================================

    def get_target_jobs_for_node(self, node: Any) -> int:
        """Return the desired selfplay concurrency for a node.

        Uses unified resource targets for consistent 60-80% utilization:
        - Backpressure-aware: Reduces jobs when training queue is full
        - Adaptive scaling: Increases jobs when underutilized, decreases when overloaded
        - Host-tier aware: Adjusts targets based on hardware capability

        Args:
            node: NodeInfo-like object with attributes:
                - node_id: str
                - memory_gb: int
                - has_gpu: bool
                - cpu_count: int
                - cpu_percent: float
                - memory_percent: float
                - disk_percent: float
                - gpu_percent: float
                - gpu_memory_percent: float
                - selfplay_jobs: int
                - gpu_name: str (optional)
                - gpu_count: int (optional)

        Returns:
            Target number of selfplay jobs for this node (always >= 1 unless blocked)
        """
        # Check safeguards first
        if self._is_emergency_active_fn:
            try:
                if self._is_emergency_active_fn():
                    return 0
            except (TypeError, AttributeError, RuntimeError) as e:
                # Dec 2025: Narrow exception - callback may be misconfigured
                logger.debug(f"[SelfplayScheduler] Emergency check callback error: {e}")

        # Check backpressure - reduce production when training queue is full
        backpressure_factor = 1.0
        if self._should_stop_production_fn:
            try:
                # Import QueueType lazily to avoid circular imports
                from app.coordination.backpressure import QueueType
                if self._should_stop_production_fn(QueueType.TRAINING_DATA):
                    if self._verbose:
                        logger.info(f"Backpressure STOP: training queue full, halting selfplay on {getattr(node, 'node_id', 'unknown')}")
                    return 0
            except Exception as e:
                if self._verbose:
                    logger.debug(f"Backpressure stop check error: {e}")

        if self._should_throttle_production_fn and self._get_throttle_factor_fn:
            try:
                from app.coordination.backpressure import QueueType
                if self._should_throttle_production_fn(QueueType.TRAINING_DATA):
                    backpressure_factor = self._get_throttle_factor_fn(QueueType.TRAINING_DATA)
                    if self._verbose:
                        logger.info(f"Backpressure throttle: factor={backpressure_factor:.2f}")
            except Exception as e:
                if self._verbose:
                    logger.debug(f"Backpressure throttle check error: {e}")

        # Extract node metrics
        node_id = getattr(node, "node_id", "unknown")
        memory_gb = int(getattr(node, "memory_gb", 0) or 0)
        has_gpu = bool(getattr(node, "has_gpu", False))
        cpu_count = int(getattr(node, "cpu_count", 0) or 0)
        cpu_percent = float(getattr(node, "cpu_percent", 0.0) or 0.0)
        mem_percent = float(getattr(node, "memory_percent", 0.0) or 0.0)
        disk_percent = float(getattr(node, "disk_percent", 0.0) or 0.0)
        gpu_percent = float(getattr(node, "gpu_percent", 0.0) or 0.0)
        gpu_mem_percent = float(getattr(node, "gpu_memory_percent", 0.0) or 0.0)
        current_jobs = int(getattr(node, "selfplay_jobs", 0) or 0)
        gpu_name = getattr(node, "gpu_name", "") or ""
        gpu_count = int(getattr(node, "gpu_count", 1) or 1) if has_gpu else 0

        # Minimum memory requirement
        if memory_gb > 0 and memory_gb < MIN_MEMORY_GB_FOR_TASKS:
            return 0

        # Record utilization for adaptive feedback
        if self._record_utilization_fn:
            with contextlib.suppress(Exception):
                self._record_utilization_fn(node_id, cpu_percent, gpu_percent, mem_percent, current_jobs)

        # Try unified resource targets if available
        if self._get_host_targets_fn and self._get_target_job_count_fn:
            try:
                host_targets = self._get_host_targets_fn(node_id)
                target_selfplay = self._get_target_job_count_fn(
                    node_id,
                    cpu_count if cpu_count > 0 else 8,
                    cpu_percent,
                    gpu_percent if has_gpu else 0.0,
                )

                # Scale up if underutilized
                if self._should_scale_up_fn:
                    scale_up, reason = self._should_scale_up_fn(
                        node_id, cpu_percent, gpu_percent, current_jobs
                    )
                    if scale_up and current_jobs < target_selfplay:
                        scale_up_increment = min(4, target_selfplay - current_jobs)
                        target_selfplay = current_jobs + scale_up_increment
                        if self._verbose:
                            logger.info(f"Scale-up on {node_id}: {reason}, target={target_selfplay}")

                # Scale down if overloaded
                if self._should_scale_down_fn:
                    scale_down, reduction, reason = self._should_scale_down_fn(
                        node_id, cpu_percent, gpu_percent, mem_percent
                    )
                    if scale_down:
                        target_selfplay = max(1, current_jobs - reduction)
                        logger.info(f"Scale-down on {node_id}: {reason}, target={target_selfplay}")

                # Apply backpressure factor
                target_selfplay = int(target_selfplay * backpressure_factor)

                # Apply host-specific max
                max_selfplay = getattr(host_targets, "max_selfplay", target_selfplay)
                target_selfplay = min(target_selfplay, max_selfplay)

                return int(max(1, target_selfplay))

            except Exception as e:
                if self._verbose:
                    logger.info(f"Resource targets error, falling back to hardware-aware: {e}")

        # FALLBACK: Use hardware-aware limits
        if self._get_max_selfplay_for_node_fn:
            max_selfplay = self._get_max_selfplay_for_node_fn(
                node_id=node_id,
                gpu_count=gpu_count,
                gpu_name=gpu_name,
                cpu_count=cpu_count,
                memory_gb=memory_gb,
                has_gpu=has_gpu,
            )
        else:
            # Minimal fallback when callback unavailable
            max_selfplay = self._compute_hardware_limit(
                has_gpu, gpu_name, gpu_count, cpu_count, memory_gb
            )

        target_selfplay = max_selfplay

        # Utilization-aware adjustments
        gpu_overloaded = gpu_percent > 85 or gpu_mem_percent > 85
        cpu_overloaded = cpu_percent > 80
        gpu_has_headroom = gpu_percent < 60 and gpu_mem_percent < 75
        cpu_has_headroom = cpu_percent < 60

        if gpu_overloaded:
            target_selfplay = max(2, target_selfplay - 2)
        if cpu_overloaded:
            target_selfplay = max(2, target_selfplay - 1)

        if ((has_gpu and gpu_has_headroom and cpu_has_headroom) or
            (not has_gpu and cpu_has_headroom)) and current_jobs < target_selfplay:
            target_selfplay = min(target_selfplay, current_jobs + 2)

        # Resource pressure warnings
        if disk_percent >= DISK_WARNING_THRESHOLD:
            target_selfplay = min(target_selfplay, 4)
        if mem_percent >= MEMORY_WARNING_THRESHOLD:
            target_selfplay = min(target_selfplay, 2)

        # Apply backpressure factor
        target_selfplay = int(target_selfplay * backpressure_factor)

        return int(max(1, target_selfplay))

    def _compute_hardware_limit(
        self,
        has_gpu: bool,
        gpu_name: str,
        gpu_count: int,
        cpu_count: int,
        memory_gb: int,
    ) -> int:
        """Compute hardware-based max selfplay limit.

        This is a fallback when resource_optimizer callbacks are unavailable.
        """
        if has_gpu:
            gpu_upper = gpu_name.upper()
            if any(g in gpu_upper for g in ["GH200"]):
                return int(cpu_count * 0.8) if cpu_count > 0 else 48
            elif any(g in gpu_upper for g in ["H100", "H200"]):
                return min(int(cpu_count * 0.5), 48) if cpu_count > 0 else 32
            elif any(g in gpu_upper for g in ["A100", "L40"]):
                return min(int(cpu_count * 0.4), 32) if cpu_count > 0 else 24
            elif any(g in gpu_upper for g in ["5090"]):
                return min(int(cpu_count * 0.3), gpu_count * 12, 64) if cpu_count > 0 else 48
            elif any(g in gpu_upper for g in ["A10", "4090", "3090"]):
                return min(int(cpu_count * 0.3), 24) if cpu_count > 0 else 16
            elif any(g in gpu_upper for g in ["4080", "4070", "3080", "4060"]):
                return min(int(cpu_count * 0.25), 12) if cpu_count > 0 else 8
            elif any(g in gpu_upper for g in ["3070", "3060", "2060", "2070", "2080"]):
                return min(int(cpu_count * 0.2), 10) if cpu_count > 0 else 6
            else:
                return min(int(cpu_count * 0.2), 8) if cpu_count > 0 else 6
        else:
            return min(int(cpu_count * 0.3), 32) if cpu_count > 0 else 8

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

    # =========================================================================
    # Architecture Performance Tracking (December 29, 2025)
    # =========================================================================

    def get_architecture_weights(
        self,
        board_type: str,
        num_players: int,
        temperature: float = 0.5,
    ) -> dict[str, float]:
        """Get allocation weights for architectures based on Elo performance.

        Higher-performing architectures get more weight for selfplay/training allocation.
        Uses softmax with temperature to control concentration on best architecture.

        Args:
            board_type: Board type (e.g., "hex8", "square8")
            num_players: Number of players (2, 3, or 4)
            temperature: Softmax temperature (lower = more concentrated on best)

        Returns:
            Dictionary mapping architecture name to allocation weight (sums to 1.0)

        Example:
            >>> weights = scheduler.get_architecture_weights("hex8", 2)
            >>> # {"v5": 0.4, "v4": 0.3, "v3": 0.2, "v2": 0.1}
        """
        try:
            from app.training.architecture_tracker import get_allocation_weights

            return get_allocation_weights(
                board_type=board_type,
                num_players=num_players,
                temperature=temperature,
            )
        except ImportError:
            logger.debug("[SelfplayScheduler] architecture_tracker not available")
            return {}
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error getting architecture weights: {e}")
            return {}

    def record_architecture_evaluation(
        self,
        architecture: str,
        board_type: str,
        num_players: int,
        elo: float,
        training_hours: float = 0.0,
        games_evaluated: int = 0,
    ) -> None:
        """Record an evaluation result for an architecture.

        Called after gauntlet evaluation to track architecture performance.
        The architecture tracker uses this data to compute allocation weights.

        Args:
            architecture: Architecture version (e.g., "v4", "v5_heavy")
            board_type: Board type (e.g., "hex8", "square8")
            num_players: Number of players
            elo: Elo rating from evaluation
            training_hours: Additional training time for this evaluation
            games_evaluated: Games used in evaluation
        """
        try:
            from app.training.architecture_tracker import record_evaluation

            stats = record_evaluation(
                architecture=architecture,
                board_type=board_type,
                num_players=num_players,
                elo=elo,
                training_hours=training_hours,
                games_evaluated=games_evaluated,
            )
            logger.info(
                f"[SelfplayScheduler] Architecture evaluation recorded: "
                f"{architecture} on {board_type}_{num_players}p -> Elo {elo:.0f} "
                f"(avg: {stats.avg_elo:.0f}, best: {stats.best_elo:.0f})"
            )
        except ImportError:
            logger.debug("[SelfplayScheduler] architecture_tracker not available")
        except Exception as e:
            logger.warning(f"[SelfplayScheduler] Failed to record architecture evaluation: {e}")

    def get_architecture_boost(
        self,
        architecture: str,
        board_type: str,
        num_players: int,
        threshold_elo_diff: float = 50.0,
    ) -> float:
        """Get boost factor for an architecture based on relative performance.

        Returns a factor > 1.0 if this architecture is better than average,
        < 1.0 if worse, exactly 1.0 if at average or no data available.

        Args:
            architecture: Architecture to check (e.g., "v4", "v5")
            board_type: Board type
            num_players: Player count
            threshold_elo_diff: Minimum Elo difference for boost

        Returns:
            Boost factor (1.0 = no boost)
        """
        try:
            from app.training.architecture_tracker import get_architecture_tracker

            tracker = get_architecture_tracker()
            return tracker.get_architecture_boost(
                architecture=architecture,
                board_type=board_type,
                num_players=num_players,
                threshold_elo_diff=threshold_elo_diff,
            )
        except ImportError:
            logger.debug("[SelfplayScheduler] architecture_tracker not available")
            return 1.0
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error getting architecture boost: {e}")
            return 1.0

    def get_best_architecture(
        self,
        board_type: str,
        num_players: int,
        metric: str = "avg_elo",
    ) -> str | None:
        """Get the best-performing architecture for a configuration.

        Args:
            board_type: Board type
            num_players: Player count
            metric: Metric to rank by ("avg_elo", "best_elo", "efficiency_score")

        Returns:
            Architecture name (e.g., "v5") or None if no data available
        """
        try:
            from app.training.architecture_tracker import get_best_architecture

            stats = get_best_architecture(
                board_type=board_type,
                num_players=num_players,
                metric=metric,
            )
            if stats:
                return stats.architecture
            return None
        except ImportError:
            logger.debug("[SelfplayScheduler] architecture_tracker not available")
            return None
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error getting best architecture: {e}")
            return None


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
