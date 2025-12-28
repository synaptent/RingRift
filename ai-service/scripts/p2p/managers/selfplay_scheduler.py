"""SelfplayScheduler: Priority-based selfplay configuration selection.

Extracted from p2p_orchestrator.py for better modularity.
Handles weighted config selection, job targeting, diversity tracking, and Elo-based priority.
"""

from __future__ import annotations

import contextlib
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from scripts.p2p.models import NodeInfo

from scripts.p2p.p2p_mixin_base import EventSubscriptionMixin

logger = logging.getLogger(__name__)


# Import constants from canonical source to avoid duplication
try:
    from scripts.p2p.constants import (
        DISK_WARNING_THRESHOLD,
        MEMORY_WARNING_THRESHOLD,
        MIN_MEMORY_GB_FOR_TASKS,
    )
except ImportError:
    # Fallback for testing/standalone use - match canonical values in app/p2p/constants.py
    MIN_MEMORY_GB_FOR_TASKS = 64  # Must match app/p2p/constants.py:84
    DISK_WARNING_THRESHOLD = 65  # Conservative: match constants.py
    MEMORY_WARNING_THRESHOLD = 75  # Conservative: match constants.py


@dataclass
class DiversityMetrics:
    """Diversity tracking metrics for selfplay scheduling."""

    games_by_engine_mode: dict[str, int] = field(default_factory=dict)
    games_by_board_config: dict[str, int] = field(default_factory=dict)
    games_by_difficulty: dict[str, int] = field(default_factory=dict)
    asymmetric_games: int = 0
    symmetric_games: int = 0
    last_reset: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with computed metrics."""
        total_games = self.asymmetric_games + self.symmetric_games
        asymmetric_ratio = (
            self.asymmetric_games / total_games if total_games > 0 else 0.0
        )

        engine_total = sum(self.games_by_engine_mode.values())
        engine_mode_distribution = (
            {k: v / engine_total for k, v in self.games_by_engine_mode.items()}
            if engine_total > 0
            else {}
        )

        return {
            "games_by_engine_mode": dict(self.games_by_engine_mode),
            "games_by_board_config": dict(self.games_by_board_config),
            "games_by_difficulty": dict(self.games_by_difficulty),
            "asymmetric_games": self.asymmetric_games,
            "symmetric_games": self.symmetric_games,
            "asymmetric_ratio": asymmetric_ratio,
            "engine_mode_distribution": engine_mode_distribution,
            "uptime_seconds": time.time() - self.last_reset,
        }


class SelfplayScheduler(EventSubscriptionMixin):
    """Manages selfplay configuration selection and job targeting.

    Responsibilities:
    - Weighted config selection based on static priority, Elo performance, curriculum
    - Job targeting per node based on hardware capabilities and utilization
    - Diversity tracking for monitoring
    - Integration with backpressure and resource optimization

    Inherits from EventSubscriptionMixin for standardized event handling (Dec 2025).

    Usage:
        scheduler = SelfplayScheduler(
            get_cluster_elo_fn=lambda: orchestrator._get_cluster_elo_summary(),
            verbose=True
        )

        # Pick a config for a node
        config = scheduler.pick_weighted_config(node)

        # Get job target for a node
        target = scheduler.get_target_jobs_for_node(node)

        # Track diversity
        scheduler.track_diversity(config)

        # Get metrics
        metrics = scheduler.get_diversity_metrics()
    """

    MIXIN_TYPE = "selfplay_scheduler"

    def __init__(
        self,
        get_cluster_elo_fn: Callable[[], dict[str, Any]] | None = None,
        load_curriculum_weights_fn: Callable[[], dict[str, float]] | None = None,
        get_board_priority_overrides_fn: Callable[[], dict[str, int]] | None = None,
        should_stop_production_fn: Callable[[Any], bool] | None = None,
        should_throttle_production_fn: Callable[[Any], bool] | None = None,
        get_throttle_factor_fn: Callable[[Any], float] | None = None,
        record_utilization_fn: Callable[[str, float, float, float, int], None]
        | None = None,
        get_host_targets_fn: Callable[[str], Any] | None = None,
        get_target_job_count_fn: Callable[[str, int, float, float], int] | None = None,
        should_scale_up_fn: Callable[[str, float, float, int], tuple[bool, str]]
        | None = None,
        should_scale_down_fn: Callable[
            [str, float, float, float], tuple[bool, int, str]
        ]
        | None = None,
        get_max_selfplay_for_node_fn: Callable[..., int] | None = None,
        get_hybrid_selfplay_limits_fn: Callable[..., dict[str, int]] | None = None,
        is_emergency_active_fn: Callable[[], bool] | None = None,
        verbose: bool = False,
    ):
        """Initialize the SelfplayScheduler.

        Args:
            get_cluster_elo_fn: Function to get cluster-wide Elo summary
            load_curriculum_weights_fn: Function to load curriculum weights
            get_board_priority_overrides_fn: Function to load board priority overrides
            should_stop_production_fn: Function to check if production should stop (backpressure)
            should_throttle_production_fn: Function to check if production should throttle
            get_throttle_factor_fn: Function to get throttle factor
            record_utilization_fn: Function to record node utilization
            get_host_targets_fn: Function to get host-specific targets
            get_target_job_count_fn: Function to calculate target job count
            should_scale_up_fn: Function to check if scaling up is needed
            should_scale_down_fn: Function to check if scaling down is needed
            get_max_selfplay_for_node_fn: Function to get max selfplay jobs for node
            get_hybrid_selfplay_limits_fn: Function to get hybrid selfplay limits
            is_emergency_active_fn: Function to check if emergency safeguards are active
            verbose: Enable verbose logging
        """
        self.get_cluster_elo = get_cluster_elo_fn or (lambda: {})
        self.load_curriculum_weights = load_curriculum_weights_fn or (lambda: {})
        self.get_board_priority_overrides = get_board_priority_overrides_fn or (
            lambda: {}
        )
        self.should_stop_production = should_stop_production_fn
        self.should_throttle_production = should_throttle_production_fn
        self.get_throttle_factor = get_throttle_factor_fn
        self.record_utilization = record_utilization_fn
        self.get_host_targets = get_host_targets_fn
        self.get_target_job_count = get_target_job_count_fn
        self.should_scale_up = should_scale_up_fn
        self.should_scale_down = should_scale_down_fn
        self.get_max_selfplay_for_node = get_max_selfplay_for_node_fn
        self.get_hybrid_selfplay_limits = get_hybrid_selfplay_limits_fn
        self.is_emergency_active = is_emergency_active_fn
        self.verbose = verbose

        # Diversity tracking
        self.diversity_metrics = DiversityMetrics()

        # Rate adjustment state (December 2025 - feedback loop integration)
        # Maps config_key -> current rate multiplier (1.0 = normal, >1 = boost, <1 = throttle)
        self._rate_multipliers: dict[str, float] = {}
        self._subscribed = False
        self._subscription_lock = threading.Lock()  # Dec 2025: Prevent race in subscribe_to_events

        # Track previous targets for change detection (P0.2 Dec 2025)
        self._previous_targets: dict[str, int] = {}
        self._previous_priorities: dict[str, int] = {}

        # Dec 2025: Initialize boost tracking dicts (fix for hasattr checks)
        # These track temporary boosts to selfplay rates for specific configs
        self._exploration_boosts: dict[str, tuple[float, float]] = {}  # config_key -> (boost_factor, expiry_time)
        self._training_complete_boosts: dict[str, float] = {}  # config_key -> expiry_time

    def get_elo_based_priority_boost(self, board_type: str, num_players: int) -> int:
        """Get priority boost based on ELO performance for this config.

        PRIORITY-BASED SCHEDULING: Configs with high-performing models get
        priority boost to allocate more resources to promising configurations.

        Args:
            board_type: Board type (e.g., "hex8", "square8")
            num_players: Number of players (2, 3, or 4)

        Returns:
            Priority boost (0-5) based on:
            - Top model ELO for this config
            - Recent improvement rate
            - Data coverage (inverse - underrepresented get boost)
        """
        boost = 0

        try:
            cluster_elo = self.get_cluster_elo()
            top_models = cluster_elo.get("top_models", [])

            # Find best model for this board/player combo
            best_elo = 0
            for model in top_models:
                model_name = model.get("name", "")
                # Model names typically include board type and player count
                if board_type in model_name or str(num_players) in model_name:
                    best_elo = max(best_elo, model.get("elo", 0))

            # ELO-based boost (every 100 ELO above 1200 = +1 priority)
            if best_elo > 1200:
                boost += min(3, (best_elo - 1200) // 100)

            # Underrepresented config boost
            # (hex and square19 often have fewer games)
            if board_type in ("hexagonal", "square19"):
                boost += 1
            if num_players > 2:
                boost += 1

        except AttributeError:
            pass

        return min(5, boost)  # Cap at +5

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Return event subscriptions for EventSubscriptionMixin.

        Dec 28, 2025: Migrated to use EventSubscriptionMixin pattern.

        Returns:
            Dict mapping event names to handler methods
        """
        return {
            "SELFPLAY_RATE_CHANGED": self._on_selfplay_rate_changed,
            "EXPLORATION_BOOST": self._on_exploration_boost,
            "TRAINING_COMPLETED": self._on_training_completed,
            "ELO_VELOCITY_CHANGED": self._on_elo_velocity_changed,
        }

    async def _on_selfplay_rate_changed(self, event) -> None:
        """Handle SELFPLAY_RATE_CHANGED events from feedback loop.

        Adjusts rate multipliers for configs based on Elo velocity and performance.

        Args:
            event: Event with payload containing config_key, new_rate, reason
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "")
        new_rate = payload.get("new_rate", 1.0)
        reason = payload.get("reason", "unknown")

        if not config_key:
            return

        old_rate = self._rate_multipliers.get(config_key, 1.0)
        self._rate_multipliers[config_key] = new_rate

        if abs(new_rate - old_rate) > 0.01:
            self._log_info(
                f"Rate changed for {config_key}: {old_rate:.2f} -> {new_rate:.2f} ({reason})"
            )

    def get_rate_multiplier(self, config_key: str) -> float:
        """Get current rate multiplier for a config.

        Args:
            config_key: Config key (e.g., "hex8_2p")

        Returns:
            Rate multiplier (1.0 = normal, >1 = boost, <1 = throttle)
        """
        return self._rate_multipliers.get(config_key, 1.0)

    async def _on_exploration_boost(self, event) -> None:
        """Handle EXPLORATION_BOOST events from FeedbackLoopController.

        Dec 2025: React to training anomalies (loss spikes, stalls) by increasing
        exploration to generate more diverse training data.

        Args:
            event: Event with payload containing config_key, boost_factor, reason
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "")
        boost_factor = payload.get("boost_factor", 1.3)
        duration = payload.get("duration_seconds", 900)
        reason = payload.get("reason", "training_anomaly")

        if not config_key:
            return

        # Delegate to existing set_exploration_boost method
        self.set_exploration_boost(config_key, boost_factor, duration)

        self._log_info(
            f"Exploration boost from event: {config_key} "
            f"{boost_factor:.2f}x for {duration}s ({reason})"
        )

    async def _on_training_completed(self, event) -> None:
        """Handle TRAINING_COMPLETED events to boost selfplay after training.

        Dec 2025: When training completes, boost selfplay for that config to
        generate more data for the next training cycle.

        Args:
            event: Event with payload containing config_key
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "")

        if not config_key:
            return

        # Delegate to existing on_training_complete method
        self.on_training_complete(config_key)

    async def _on_elo_velocity_changed(self, event) -> None:
        """Handle ELO_VELOCITY_CHANGED events for selfplay rate adjustment.

        Dec 2025: Adjusts selfplay rate based on Elo velocity trends.
        - accelerating: Increase selfplay to capitalize on momentum
        - decelerating: Reduce selfplay, shift focus to training quality
        - stable: Maintain current allocation

        Args:
            event: Event with payload containing config_key, velocity, trend
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "")
        velocity = payload.get("velocity", 0.0)
        trend = payload.get("trend", "stable")

        if not config_key:
            return

        old_rate = self._rate_multipliers.get(config_key, 1.0)

        # Adjust rate based on velocity trend
        if trend == "accelerating":
            # Capitalize on positive momentum - increase selfplay rate
            new_rate = min(1.5, old_rate * 1.2)
        elif trend == "decelerating":
            # Slow down and focus on quality
            new_rate = max(0.6, old_rate * 0.85)
        else:  # stable
            # Slight adjustment toward 1.0
            if old_rate > 1.0:
                new_rate = max(1.0, old_rate * 0.95)
            elif old_rate < 1.0:
                new_rate = min(1.0, old_rate * 1.05)
            else:
                new_rate = 1.0

        self._rate_multipliers[config_key] = new_rate

        if abs(new_rate - old_rate) > 0.01:
            self._log_info(
                f"Elo velocity {trend} for {config_key}: "
                f"velocity={velocity:.1f}, rate {old_rate:.2f} -> {new_rate:.2f}"
            )

    def _emit_selfplay_target_updated(
        self,
        config_key: str,
        priority: str,
        reason: str,
        *,
        target_jobs: int | None = None,
        effective_priority: int | None = None,
        exploration_boost: float | None = None,
    ) -> bool:
        """Emit SELFPLAY_TARGET_UPDATED event for feedback loop integration.

        P0.2 (December 2025): Enables pipeline coordination to respond to
        scheduling decisions. Events trigger:
        - DaemonManager workload scaling
        - FeedbackLoopController priority adjustments
        - Training pipeline data freshness checks

        Dec 2025 (P0-1 fix): Returns bool for caller to check success/failure.

        Args:
            config_key: Config key (e.g., "hex8_2p")
            priority: Priority level ("urgent", "high", "normal")
            reason: Descriptive reason for the update
            target_jobs: Optional target job count
            effective_priority: Optional effective priority value
            exploration_boost: Optional exploration boost multiplier

        Returns:
            True if event was emitted successfully, False otherwise.
        """
        try:
            from app.coordination.event_router import publish_sync

            payload: dict[str, Any] = {
                "config_key": config_key,
                "priority": priority,
                "reason": reason,
                "source": "p2p_selfplay_scheduler",
            }
            if target_jobs is not None:
                payload["target_jobs"] = target_jobs
            if effective_priority is not None:
                payload["effective_priority"] = effective_priority
            if exploration_boost is not None:
                payload["exploration_boost"] = exploration_boost

            publish_sync("SELFPLAY_TARGET_UPDATED", payload)
            if self.verbose:
                logger.info(
                    f"[SelfplayScheduler] Emitted SELFPLAY_TARGET_UPDATED: "
                    f"{config_key} priority={priority} reason={reason}"
                )
            return True
        except ImportError:
            logger.debug("[SelfplayScheduler] Event router not available for target updates")
            return False
        except (RuntimeError, AttributeError) as e:
            logger.debug(f"[SelfplayScheduler] Failed to emit target update: {e}")
            return False

    def pick_weighted_config(self, node: NodeInfo) -> dict[str, Any] | None:
        """Pick a selfplay config weighted by priority and node capabilities.

        PRIORITY-BASED SCHEDULING: Combines static priority with dynamic
        ELO-based boosts to allocate more resources to high-performing configs.

        Args:
            node: Node information for capability filtering

        Returns:
            Config dict with board_type, num_players, engine_mode, or None if no valid config
        """
        # Get the selfplay configs - DIVERSE mode prioritized for high-quality training data
        # Uses "mixed" engine mode for varied AI matchups (NNUE, MCTS, heuristic combinations)
        selfplay_configs = [
            # Priority 8: Underrepresented hex/sq19 combos with diverse AI (highest priority)
            {
                "board_type": "hexagonal",
                "num_players": 3,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "hexagonal",
                "num_players": 2,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "hexagonal",
                "num_players": 4,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "hex8",
                "num_players": 2,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "hex8",
                "num_players": 3,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "hex8",
                "num_players": 4,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "square19",
                "num_players": 3,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "square19",
                "num_players": 2,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "square19",
                "num_players": 4,
                "engine_mode": "mixed",
                "priority": 8,
            },
            # Priority 7: Square8 multi-player with diverse AI
            {
                "board_type": "square8",
                "num_players": 3,
                "engine_mode": "mixed",
                "priority": 7,
            },
            {
                "board_type": "square8",
                "num_players": 4,
                "engine_mode": "mixed",
                "priority": 7,
            },
            # Priority 6: Cross-AI matches (specific matchup types)
            {
                "board_type": "square8",
                "num_players": 2,
                "engine_mode": "heuristic-vs-mcts",
                "priority": 6,
            },
            {
                "board_type": "hexagonal",
                "num_players": 3,
                "engine_mode": "heuristic-vs-mcts",
                "priority": 6,
            },
            {
                "board_type": "square19",
                "num_players": 2,
                "engine_mode": "heuristic-vs-mcts",
                "priority": 6,
            },
            # Priority 5: Standard 2p square8 with diverse AI
            {
                "board_type": "square8",
                "num_players": 2,
                "engine_mode": "mixed",
                "priority": 5,
            },
            # Priority 4: Tournament varied (for evaluation-style games)
            {
                "board_type": "square8",
                "num_players": 2,
                "engine_mode": "tournament-varied",
                "priority": 4,
            },
            {
                "board_type": "hexagonal",
                "num_players": 2,
                "engine_mode": "tournament-varied",
                "priority": 4,
            },
            # Priority 3: CPU-bound specialized modes
            {
                "board_type": "square8",
                "num_players": 2,
                "engine_mode": "mcts-only",
                "priority": 3,
            },
            {
                "board_type": "hexagonal",
                "num_players": 2,
                "engine_mode": "descent-only",
                "priority": 3,
            },
        ]

        # Filter by node memory (avoid large boards on small nodes)
        node_mem = int(getattr(node, "memory_gb", 0) or 0)
        if node_mem and node_mem < 48:
            selfplay_configs = [
                c for c in selfplay_configs if c.get("board_type") == "square8"
            ]

        if not selfplay_configs:
            return None

        # PRIORITY-BASED SCHEDULING: Add ELO-based priority boosts
        # Phase 3.1: Also incorporate curriculum weights from unified AI loop
        curriculum_weights = {}
        try:
            curriculum_weights = self.load_curriculum_weights()
        except (OSError, ValueError, KeyError, ImportError):
            pass  # Use empty weights on error

        # Load board priority overrides from config (0=CRITICAL, 1=HIGH, 2=MEDIUM, 3=LOW)
        board_priority_overrides = self.get_board_priority_overrides()

        for cfg in selfplay_configs:
            elo_boost = self.get_elo_based_priority_boost(
                cfg.get("board_type", ""),
                cfg.get("num_players", 2),
            )

            # Phase 3.1: Apply curriculum weight boost
            # Config keys are formatted as "board_type_Np" (e.g., "square8_2p")
            config_key = f"{cfg.get('board_type', '')}_{cfg.get('num_players', 2)}p"
            curriculum_weight = curriculum_weights.get(config_key, 1.0)
            # Convert weight (0.7-1.5) to priority boost (0-3)
            # weight 0.7 = -1 boost, weight 1.0 = 0 boost, weight 1.5 = +2 boost
            curriculum_boost = int((curriculum_weight - 1.0) * 4)
            curriculum_boost = max(-2, min(3, curriculum_boost))  # Clamp to -2..+3

            # Apply board priority overrides from config
            # 0=CRITICAL adds +6, 1=HIGH adds +4, 2=MEDIUM adds +2, 3=LOW adds 0
            board_priority = board_priority_overrides.get(
                config_key, 3
            )  # Default to LOW (3)
            board_priority_boost = (3 - board_priority) * 2  # 0->6, 1->4, 2->2, 3->0

            # Apply rate multiplier from feedback loop (December 2025)
            # Rate multiplier > 1 = boost priority, < 1 = reduce priority
            rate_multiplier = self.get_rate_multiplier(config_key)
            rate_boost = int((rate_multiplier - 1.0) * 5)  # ±5 priority max
            rate_boost = max(-3, min(5, rate_boost))  # Clamp to -3..+5

            cfg["effective_priority"] = (
                cfg.get("priority", 1)
                + elo_boost
                + curriculum_boost
                + board_priority_boost
                + rate_boost
            )

        # Build weighted list by effective priority
        weighted = []
        for cfg in selfplay_configs:
            # Ensure minimum priority of 1
            priority = max(1, cfg.get("effective_priority", 1))
            weighted.extend([cfg] * priority)

        selected = random.choice(weighted) if weighted else None

        # P0.2 (Dec 2025): Emit event when priority changes significantly
        if selected:
            config_key = f"{selected.get('board_type', '')}_{selected.get('num_players', 2)}p"
            new_priority = selected.get("effective_priority", 1)
            old_priority = self._previous_priorities.get(config_key, new_priority)

            # Emit event if priority changed by 2+ or rate multiplier applied
            rate_mult = self.get_rate_multiplier(config_key)
            priority_change = abs(new_priority - old_priority)
            if priority_change >= 2 or (rate_mult != 1.0 and priority_change >= 1):
                reason = "priority_boost" if new_priority > old_priority else "priority_reduced"
                self._emit_selfplay_target_updated(
                    config_key=config_key,
                    priority="high" if priority_change >= 3 else "normal",
                    reason=f"{reason}:{priority_change}",
                    effective_priority=new_priority,
                    exploration_boost=rate_mult if rate_mult != 1.0 else None,
                )
                self._previous_priorities[config_key] = new_priority

        return selected

    def get_target_jobs_for_node(self, node: NodeInfo) -> int:
        """Return the desired selfplay concurrency for a node.

        Uses unified resource targets for consistent 60-80% utilization:
        - Backpressure-aware: Reduces jobs when training queue is full
        - Adaptive scaling: Increases jobs when underutilized, decreases when overloaded
        - Host-tier aware: Adjusts targets based on hardware capability

        Args:
            node: Node information

        Returns:
            Target number of selfplay jobs (minimum 1)

        Target: 60-80% CPU/GPU utilization for optimal training throughput.
        """
        # Check safeguards first - halt all selfplay during emergency
        if self.is_emergency_active is not None:
            try:
                if self.is_emergency_active():
                    return 0
            except (TypeError, AttributeError, RuntimeError, KeyError):
                pass  # Ignore errors in safeguards callback (non-critical)

        # Check backpressure - reduce production when training queue is full
        backpressure_factor = 1.0
        if (
            self.should_stop_production is not None
            and self.should_throttle_production is not None
        ):
            try:
                # Import QueueType here to avoid circular imports
                try:
                    from app.coordination import QueueType

                    queue_type = QueueType.TRAINING_DATA
                except ImportError:
                    queue_type = "TRAINING_DATA"

                if self.should_stop_production(queue_type):
                    logger.info(
                        f"Backpressure STOP: training queue full, halting selfplay on {node.node_id}"
                    )
                    return 0
                if self.should_throttle_production(queue_type):
                    if self.get_throttle_factor is not None:
                        backpressure_factor = self.get_throttle_factor(queue_type)
                        logger.info(f"Backpressure throttle: factor={backpressure_factor:.2f}")
            except (TypeError, AttributeError, ValueError, RuntimeError) as e:
                logger.info(f"Backpressure check error: {e}")

        # Minimum memory requirement - skip low-memory machines to avoid OOM
        memory_gb = int(getattr(node, "memory_gb", 0) or 0)
        if memory_gb > 0 and memory_gb < MIN_MEMORY_GB_FOR_TASKS:
            return 0

        # Extract node metrics
        has_gpu = bool(getattr(node, "has_gpu", False))
        cpu_count = int(getattr(node, "cpu_count", 0) or 0)
        cpu_percent = float(getattr(node, "cpu_percent", 0.0) or 0.0)
        mem_percent = float(getattr(node, "memory_percent", 0.0) or 0.0)
        disk_percent = float(getattr(node, "disk_percent", 0.0) or 0.0)
        gpu_percent = float(getattr(node, "gpu_percent", 0.0) or 0.0)
        gpu_mem_percent = float(getattr(node, "gpu_memory_percent", 0.0) or 0.0)
        current_jobs = int(getattr(node, "selfplay_jobs", 0) or 0)

        # Record utilization for adaptive feedback
        if self.record_utilization is not None:
            with contextlib.suppress(Exception):
                self.record_utilization(
                    node.node_id, cpu_percent, gpu_percent, mem_percent, current_jobs
                )

        # Use unified resource targets if available
        if (
            self.get_host_targets is not None
            and self.get_target_job_count is not None
        ):
            try:
                # Get host-specific targets adjusted for tier and backpressure
                host_targets = self.get_host_targets(node.node_id)

                # Use the unified target calculator
                target_selfplay = self.get_target_job_count(
                    node.node_id,
                    cpu_count if cpu_count > 0 else 8,
                    cpu_percent,
                    gpu_percent if has_gpu else 0.0,
                )

                # Check if we should scale up (underutilized)
                if self.should_scale_up is not None:
                    scale_up, reason = self.should_scale_up(
                        node.node_id, cpu_percent, gpu_percent, current_jobs
                    )
                    if scale_up and current_jobs < target_selfplay:
                        # Controlled scale-up: Add 2-4 jobs at a time, not all at once
                        scale_up_increment = min(4, target_selfplay - current_jobs)
                        target_selfplay = current_jobs + scale_up_increment
                        if self.verbose:
                            logger.info(
                                f"Scale-up on {node.node_id}: {reason}, target={target_selfplay}"
                            )

                # Check if we should scale down (overloaded)
                if self.should_scale_down is not None:
                    scale_down, reduction, reason = self.should_scale_down(
                        node.node_id, cpu_percent, gpu_percent, mem_percent
                    )
                    if scale_down:
                        target_selfplay = max(1, current_jobs - reduction)
                        logger.info(
                            f"Scale-down on {node.node_id}: {reason}, target={target_selfplay}"
                        )

                # Apply backpressure factor
                target_selfplay = int(target_selfplay * backpressure_factor)

                # Apply host-specific max
                target_selfplay = min(target_selfplay, host_targets.max_selfplay)

                final_target = int(max(1, target_selfplay))

                # P0.2 (Dec 2025): Emit event when target changes significantly
                old_target = self._previous_targets.get(node.node_id, final_target)
                target_change = abs(final_target - old_target)
                relative_change = target_change / max(1, old_target)
                if target_change >= 3 or relative_change >= 0.5:
                    reason = "target_increased" if final_target > old_target else "target_decreased"
                    evt_priority = "high" if target_change >= 5 or relative_change >= 0.75 else "normal"
                    self._emit_selfplay_target_updated(
                        config_key=f"node:{node.node_id}",
                        priority=evt_priority,
                        reason=f"{reason}:{target_change}",
                        target_jobs=final_target,
                    )
                    self._previous_targets[node.node_id] = final_target

                return final_target

            except (TypeError, AttributeError, ValueError, KeyError, RuntimeError) as e:
                logger.info(f"Resource targets error, falling back to hardware-aware: {e}")

        # FALLBACK: Use unified hardware-aware limits from resource_optimizer
        # This ensures consistent limits across all orchestrators
        gpu_name = getattr(node, "gpu_name", "") or ""
        gpu_count = int(getattr(node, "gpu_count", 1) or 1) if has_gpu else 0

        if self.get_max_selfplay_for_node is not None:
            # Use single source of truth from resource_optimizer
            max_selfplay = self.get_max_selfplay_for_node(
                node_id=node.node_id,
                gpu_count=gpu_count,
                gpu_name=gpu_name,
                cpu_count=cpu_count,
                memory_gb=memory_gb,
                has_gpu=has_gpu,
            )
        else:
            # Minimal fallback when resource_optimizer unavailable
            # Values calibrated from observed workloads (GH200: 48 jobs at 70% GPU)
            if has_gpu:
                gpu_upper = gpu_name.upper()
                if any(g in gpu_upper for g in ["GH200"]):
                    # GH200 with unified 480GB memory - CPU is bottleneck
                    max_selfplay = int(cpu_count * 0.8) if cpu_count > 0 else 48
                elif any(g in gpu_upper for g in ["H100", "H200"]):
                    max_selfplay = (
                        min(int(cpu_count * 0.5), 48) if cpu_count > 0 else 32
                    )
                elif any(g in gpu_upper for g in ["A100", "L40"]):
                    max_selfplay = (
                        min(int(cpu_count * 0.4), 32) if cpu_count > 0 else 24
                    )
                elif any(g in gpu_upper for g in ["5090"]):
                    # RTX 5090 (32GB) - very high capacity
                    max_selfplay = (
                        min(int(cpu_count * 0.3), gpu_count * 12, 64)
                        if cpu_count > 0
                        else 48
                    )
                elif any(g in gpu_upper for g in ["A10", "4090", "3090"]):
                    max_selfplay = (
                        min(int(cpu_count * 0.3), 24) if cpu_count > 0 else 16
                    )
                elif any(g in gpu_upper for g in ["4080", "4070", "3080", "4060"]):
                    max_selfplay = (
                        min(int(cpu_count * 0.25), 12) if cpu_count > 0 else 8
                    )
                elif any(
                    g in gpu_upper for g in ["3070", "3060", "2060", "2070", "2080"]
                ):
                    max_selfplay = (
                        min(int(cpu_count * 0.2), 10) if cpu_count > 0 else 6
                    )
                else:
                    max_selfplay = min(int(cpu_count * 0.2), 8) if cpu_count > 0 else 6
            else:
                # CPU-only: ~0.3 jobs per core, capped at 32
                max_selfplay = min(int(cpu_count * 0.3), 32) if cpu_count > 0 else 8

        target_selfplay = max_selfplay

        # Utilization-aware adjustments (target 60-80%)
        gpu_overloaded = gpu_percent > 85 or gpu_mem_percent > 85
        cpu_overloaded = cpu_percent > 80
        gpu_has_headroom = gpu_percent < 60 and gpu_mem_percent < 75
        cpu_has_headroom = cpu_percent < 60

        # Scale DOWN if overloaded
        if gpu_overloaded:
            target_selfplay = max(2, target_selfplay - 2)
        if cpu_overloaded:
            target_selfplay = max(2, target_selfplay - 1)

        # Scale UP only if both resources have headroom (gradual)
        if (
            not gpu_overloaded
            and not cpu_overloaded
            and current_jobs > 0
            and (has_gpu and gpu_has_headroom and cpu_has_headroom)
        ) or ((not has_gpu and cpu_has_headroom) and current_jobs < target_selfplay):
            target_selfplay = min(target_selfplay, current_jobs + 2)

        # Resource pressure warnings
        if disk_percent >= DISK_WARNING_THRESHOLD:
            target_selfplay = min(target_selfplay, 4)
        if mem_percent >= MEMORY_WARNING_THRESHOLD:
            target_selfplay = min(target_selfplay, 2)

        # Apply backpressure factor
        target_selfplay = int(target_selfplay * backpressure_factor)

        final_target = int(max(1, target_selfplay))

        # P0.2 (Dec 2025): Emit event when target changes significantly
        node_id = getattr(node, "node_id", "unknown")
        old_target = self._previous_targets.get(node_id, final_target)
        target_change = abs(final_target - old_target)

        # Emit if target changed by 3+ jobs or 50%+ relative change
        relative_change = target_change / max(1, old_target)
        if target_change >= 3 or relative_change >= 0.5:
            reason = "target_increased" if final_target > old_target else "target_decreased"
            priority = "high" if target_change >= 5 or relative_change >= 0.75 else "normal"
            self._emit_selfplay_target_updated(
                config_key=f"node:{node_id}",
                priority=priority,
                reason=f"{reason}:{target_change}",
                target_jobs=final_target,
            )
            self._previous_targets[node_id] = final_target

        return final_target

    def get_hybrid_job_targets(self, node: NodeInfo) -> dict[str, int]:
        """Get separate GPU and CPU-only selfplay job targets for hybrid mode.

        For high-CPU nodes with limited GPU VRAM (like Vast hosts), this enables:
        - Running GPU jobs up to VRAM limit
        - Running additional CPU-only jobs to utilize excess CPU capacity

        Args:
            node: Node information

        Returns:
            Dict with 'gpu_jobs', 'cpu_only_jobs', 'total_jobs'
        """
        has_gpu = bool(getattr(node, "has_gpu", False))
        cpu_count = int(getattr(node, "cpu_count", 0) or 0)
        memory_gb = int(getattr(node, "memory_gb", 0) or 0)
        gpu_name = getattr(node, "gpu_name", "") or ""
        gpu_count = int(getattr(node, "gpu_count", 1) or 1) if has_gpu else 0

        # Use hybrid limits function if available
        if self.get_hybrid_selfplay_limits is not None:
            try:
                limits = self.get_hybrid_selfplay_limits(
                    node_id=node.node_id,
                    gpu_count=gpu_count,
                    gpu_name=gpu_name,
                    cpu_count=cpu_count,
                    memory_gb=memory_gb,
                    has_gpu=has_gpu,
                )
                return limits
            except (TypeError, AttributeError, ValueError, KeyError, RuntimeError) as e:
                logger.info(f"Hybrid limits error: {e}")

        # Fallback: No CPU-only jobs, use standard target
        gpu_jobs = self.get_target_jobs_for_node(node)
        return {"gpu_jobs": gpu_jobs, "cpu_only_jobs": 0, "total_jobs": gpu_jobs}

    def should_spawn_cpu_only_jobs(self, node: NodeInfo) -> bool:
        """Check if a node should spawn CPU-only jobs in addition to GPU jobs.

        CPU-only jobs are beneficial when:
        1. Node has many CPU cores (64+)
        2. Node has limited GPU VRAM (<=16GB per GPU)
        3. GPU jobs are already at capacity (VRAM-limited)

        Args:
            node: Node information

        Returns:
            True if CPU-only jobs should be spawned
        """
        if self.get_hybrid_selfplay_limits is None:
            return False

        cpu_count = int(getattr(node, "cpu_count", 0) or 0)
        has_gpu = bool(getattr(node, "has_gpu", False))
        gpu_name = (getattr(node, "gpu_name", "") or "").upper()

        # Must have significant CPU resources (64+ cores)
        if cpu_count < 64:
            return False

        # For GPU nodes, only spawn CPU-only if GPU has limited VRAM
        if has_gpu:
            # High-end datacenter GPUs don't need CPU-only jobs (plenty of VRAM)
            if any(g in gpu_name for g in ["GH200", "H100", "H200", "A100", "L40"]):
                return False
            # Consumer GPUs with limited VRAM benefit from CPU-only supplement
            if any(
                g in gpu_name
                for g in ["3070", "3060", "2060", "2070", "2080", "4060", "4070"]
            ):
                return True
            # 5090/4090 with 24-32GB might not need it unless very high CPU count
            if any(g in gpu_name for g in ["5090", "4090", "3090"]):
                return cpu_count >= 128

        # CPU-only nodes always benefit from full CPU utilization
        return True

    def track_diversity(self, config: dict[str, Any]) -> None:
        """Track diversity metrics for a scheduled selfplay game.

        Args:
            config: Selfplay configuration dict with engine_mode, board_type, num_players, etc.
        """
        # Track engine mode
        engine_mode = config.get("engine_mode", "unknown")
        if engine_mode not in self.diversity_metrics.games_by_engine_mode:
            self.diversity_metrics.games_by_engine_mode[engine_mode] = 0
        self.diversity_metrics.games_by_engine_mode[engine_mode] += 1

        # Track board config
        board_key = (
            f"{config.get('board_type', 'unknown')}_{config.get('num_players', 0)}p"
        )
        if board_key not in self.diversity_metrics.games_by_board_config:
            self.diversity_metrics.games_by_board_config[board_key] = 0
        self.diversity_metrics.games_by_board_config[board_key] += 1

        # Track asymmetric vs symmetric
        if config.get("asymmetric"):
            self.diversity_metrics.asymmetric_games += 1
            strong = config.get("strong_config", {})
            weak = config.get("weak_config", {})
            logger.info(
                f"DIVERSE: Asymmetric game scheduled - "
                f"Strong({strong.get('engine_mode')}@D{strong.get('difficulty')}) vs "
                f"Weak({weak.get('engine_mode')}@D{weak.get('difficulty')}) "
                f"on {board_key}"
            )
        else:
            self.diversity_metrics.symmetric_games += 1

        # Track difficulty if available
        difficulty = config.get("difficulty", config.get("difficulty_band"))
        if difficulty:
            diff_key = str(difficulty)
            if diff_key not in self.diversity_metrics.games_by_difficulty:
                self.diversity_metrics.games_by_difficulty[diff_key] = 0
            self.diversity_metrics.games_by_difficulty[diff_key] += 1

    def get_diversity_metrics(self) -> dict[str, Any]:
        """Get diversity tracking metrics for monitoring.

        Returns:
            Dictionary with diversity metrics including computed statistics
        """
        return self.diversity_metrics.to_dict()

    def on_training_complete(self, config_key: str) -> None:
        """Handle training completion for a config.

        Called by P2P orchestrator when TRAINING_COMPLETED event fires.
        Refreshes selfplay priorities to potentially increase allocation
        for the just-trained config (more data needed for next training cycle).

        P0.1 (Dec 2025): Added to fix missing method referenced at
        p2p_orchestrator.py:2338.

        Args:
            config_key: The config key (e.g., "hex8_2p") that completed training.
        """
        logger.info(
            f"[SelfplayScheduler] Training completed for {config_key}, "
            f"refreshing priorities"
        )

        # Boost selfplay rate for this config temporarily (just trained = needs more data)
        try:
            # Increase rate multiplier for 30 minutes after training
            boost_duration = 1800  # 30 minutes
            expiry = time.time() + boost_duration
            # Dec 2025: _training_complete_boosts initialized in __init__
            self._training_complete_boosts[config_key] = expiry

            logger.debug(
                f"[SelfplayScheduler] Boosting {config_key} selfplay priority "
                f"for {boost_duration}s after training completion"
            )
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error boosting {config_key}: {e}")

    def set_exploration_boost(
        self, config_key: str, boost_factor: float, duration_seconds: float = 900
    ) -> None:
        """Set exploration boost for a config due to training anomaly.

        P0.1 (Dec 2025): Added to handle EXPLORATION_BOOST events from
        FeedbackLoopController when training anomalies are detected.

        When training has loss spikes or stalls, we boost exploration to
        generate more diverse training data.

        Args:
            config_key: The config key (e.g., "hex8_2p") to boost.
            boost_factor: Multiplicative boost factor (e.g., 1.3 for 30% more).
            duration_seconds: How long the boost should last.
        """
        try:
            expiry = time.time() + duration_seconds
            # Dec 2025: _exploration_boosts initialized in __init__
            self._exploration_boosts[config_key] = (boost_factor, expiry)

            logger.info(
                f"[SelfplayScheduler] Set exploration boost for {config_key}: "
                f"{boost_factor:.2f}x for {duration_seconds}s"
            )
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error setting exploration boost: {e}")

    def get_exploration_boost(self, config_key: str) -> float:
        """Get current exploration boost factor for a config.

        Args:
            config_key: The config key to check.

        Returns:
            Boost factor (1.0 = no boost, >1.0 = boosted).
        """
        # Dec 2025: _exploration_boosts always initialized in __init__
        boost_info = self._exploration_boosts.get(config_key)
        if not boost_info:
            return 1.0

        boost_factor, expiry = boost_info
        if time.time() > expiry:
            # Boost expired, clean up
            del self._exploration_boosts[config_key]
            return 1.0

        return boost_factor

    # =========================================================================
    # Health Check (December 2025)
    # =========================================================================

    def health_check(self) -> dict[str, Any]:
        """Check health status of SelfplayScheduler.

        Returns:
            Dict with status, scheduling metrics, and error info
        """
        status = "healthy"
        errors_count = 0
        last_error: str | None = None

        # Get diversity metrics
        diversity = self.get_diversity_metrics()

        # Check if subscribed to events
        if not self._subscribed:
            if status == "healthy":
                status = "degraded"
                last_error = "Not subscribed to events"

        # Check config coverage (all configs should have some targets)
        configs_with_targets = sum(
            1 for v in self._previous_targets.values() if v > 0
        )
        total_configs = len(self._previous_targets) if self._previous_targets else 0

        if total_configs > 0 and configs_with_targets == 0:
            status = "degraded"
            last_error = "No configs have selfplay targets"
        elif total_configs > 0 and configs_with_targets < total_configs // 2:
            if status == "healthy":
                status = "degraded"
                last_error = f"Only {configs_with_targets}/{total_configs} configs have targets"

        # Count active exploration boosts
        # Dec 2025: _exploration_boosts always initialized in __init__
        now = time.time()
        active_boosts = sum(
            1 for _, expiry in self._exploration_boosts.values()
            if expiry > now
        )

        return {
            "status": status,
            "operations_count": total_configs,
            "errors_count": errors_count,
            "last_error": last_error,
            "subscribed": self._subscribed,
            "configs_tracked": total_configs,
            "configs_with_targets": configs_with_targets,
            "active_exploration_boosts": active_boosts,
            "diversity_metrics": diversity,
        }

    def record_promotion_failure(self, config_key: str) -> None:
        """Record a promotion failure for curriculum feedback.

        December 27, 2025: Added to handle PROMOTION_FAILED events.
        When a model fails to promote (gauntlet failure), we temporarily
        reduce the selfplay priority for that config to avoid wasting
        resources on a potentially unstable training trajectory.

        Args:
            config_key: The configuration that failed promotion (e.g., "hex8_2p")
        """
        if not hasattr(self, "_promotion_failures"):
            self._promotion_failures: dict[str, list[float]] = {}

        # Track failure timestamps
        if config_key not in self._promotion_failures:
            self._promotion_failures[config_key] = []
        self._promotion_failures[config_key].append(time.time())

        # Keep only failures from last 24 hours
        cutoff = time.time() - 86400
        self._promotion_failures[config_key] = [
            t for t in self._promotion_failures[config_key] if t > cutoff
        ]

        failure_count = len(self._promotion_failures[config_key])
        logger.info(
            f"[SelfplayScheduler] Recorded promotion failure for {config_key} "
            f"({failure_count} failures in last 24h)"
        )

        # Apply temporary priority reduction based on failure count
        # More failures = longer penalty period
        if failure_count >= 3:
            # After 3 failures, significantly reduce priority for 2 hours
            penalty_duration = 7200
            penalty_factor = 0.3
        elif failure_count >= 2:
            # After 2 failures, reduce priority for 1 hour
            penalty_duration = 3600
            penalty_factor = 0.5
        else:
            # First failure, reduce priority for 30 minutes
            penalty_duration = 1800
            penalty_factor = 0.7

        # Store penalty in exploration boosts (negative boost = reduced priority)
        if not hasattr(self, "_promotion_penalties"):
            self._promotion_penalties: dict[str, tuple[float, float]] = {}
        self._promotion_penalties[config_key] = (penalty_factor, time.time() + penalty_duration)

        logger.info(
            f"[SelfplayScheduler] Applied {penalty_factor:.0%} priority penalty "
            f"to {config_key} for {penalty_duration}s"
        )
