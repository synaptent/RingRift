"""Priority update orchestration mixin for SelfplayScheduler.

February 2026: Extracted from selfplay_scheduler.py to reduce main module size.

This mixin handles the priority calculation pipeline:
- Gathering data from async providers (freshness, Elo, feedback, curriculum)
- Computing dynamic priority weights based on cluster state
- Scoring each config via PriorityCalculator
- Budget adaptation based on Elo and training intensity
- Starvation alerting for critically underserved configs

All methods access scheduler state via `self` (mixin pattern).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import TYPE_CHECKING, Any

from app.coordination.budget_calculator import (
    get_budget_with_intensity as _get_budget_with_intensity,
    compute_target_games as _compute_target,
    get_board_adjusted_budget,
)
from app.coordination.priority_calculator import (
    ClusterState,
    PriorityInputs,
    compute_dynamic_weights,
)

if TYPE_CHECKING:
    from app.coordination.selfplay_priority_types import ConfigPriority, DynamicWeights

logger = logging.getLogger(__name__)


class PriorityUpdateMixin:
    """Mixin providing priority calculation orchestration for SelfplayScheduler.

    Requires the host class to have:
    - _config_priorities: dict[str, ConfigPriority]
    - _last_priority_update: float
    - _priority_update_interval: float
    - _dynamic_weights: DynamicWeights
    - _last_dynamic_weights_update: float
    - _dynamic_weights_update_interval: float
    - _priority_calculator: PriorityCalculator
    - _node_capabilities: dict[str, NodeCapability]
    - _backpressure_monitor: IBackpressureMonitor | None
    - _quality_cache: ConfigStateCache
    - _cluster_game_counts: dict[str, int]
    - Various data provider methods from DataProviderMixin
    """

    async def get_priority_configs(self, top_n: int = 12) -> list[tuple[str, float]]:
        """Get configs ranked by priority for selfplay allocation.

        Args:
            top_n: Number of top priority configs to return (default: 12 for all configs)

        Returns:
            List of (config_key, priority_score) tuples, sorted by priority
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
            if config_key in elo_current_data:
                current_elo = elo_current_data[config_key]
                priority.current_elo = current_elo
                game_count = priority.game_count
                # Sprint 10: Use intensity-coupled budget calculation
                new_budget = self._get_budget_with_intensity(game_count, current_elo, config_key)
                # Feb 2026: Apply large board budget caps scaled by player count
                board_type = config_key.split("_")[0]
                num_players = int(config_key.split("_")[1].rstrip("p"))
                new_budget = get_board_adjusted_budget(board_type, new_budget, game_count, num_players)
                old_budget = priority.search_budget
                if new_budget != old_budget:
                    priority.search_budget = new_budget
                    intensity = self._get_training_intensity_for_config(config_key)
                    logger.info(
                        f"[SelfplayScheduler] Adaptive budget for {config_key}: "
                        f"{old_budget}→{new_budget} (games={game_count}, Elo={current_elo:.0f}, "
                        f"intensity={intensity})"
                    )

            # Feb 2026: Dynamic priority override based on live metrics
            from app.coordination.priority_calculator import compute_config_priority_override
            live_game_count = game_count_data.get(config_key)
            live_elo = elo_current_data.get(config_key)
            dynamic_override = compute_config_priority_override(config_key, live_game_count, live_elo)
            old_override = priority.priority_override
            priority.priority_override = dynamic_override
            if dynamic_override != old_override:
                tier_names = {-1: "EMERGENCY", 0: "CRITICAL", 1: "HIGH", 2: "MEDIUM", 3: "LOW"}
                source = "dynamic" if live_game_count is not None else "fallback"
                elo_str = f"{live_elo:.0f}" if live_elo is not None else "?"
                logger.info(
                    f"[SelfplayScheduler] Priority update: {config_key}="
                    f"{tier_names.get(dynamic_override, '?')}({source}, "
                    f"{live_game_count or '?'} games, {elo_str} Elo)"
                )

            # Dec 29, 2025: Update Elo uncertainty for VOI calculation
            BASE_UNCERTAINTY = 300.0
            MIN_UNCERTAINTY = 30.0
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
        try:
            from app.training.improvement_optimizer import get_improvement_optimizer

            active_configs = len([p for p in self._config_priorities.values() if p.training_pending])
            total_configs = len(self._config_priorities)

            gpu_util = min(100.0, 50.0 + (active_configs / max(1, total_configs)) * 50.0)
            cpu_util = gpu_util * 0.6

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
            pass
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Failed to record utilization: {e}")

    def _compute_dynamic_weights(self) -> DynamicWeights:
        """Compute dynamic priority weights based on current cluster state.

        Weight adjustment logic:
        - High idle GPU fraction -> Boost staleness weight (generate more data)
        - Large training queue -> Reduce staleness weight (don't flood queue)
        - Many configs at Elo target -> Reduce velocity weight (focus on struggling configs)
        - High average Elo -> Boost curriculum weight (need harder positions)

        Returns:
            DynamicWeights with adjusted values based on cluster state
        """
        now = time.time()

        # Rate-limit weight updates (expensive to compute cluster state)
        if now - self._last_dynamic_weights_update < self._dynamic_weights_update_interval:
            return self._dynamic_weights

        self._last_dynamic_weights_update = now

        cluster_state = self._gather_cluster_state()
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

        Returns:
            ClusterState with current metrics
        """
        # 1. Idle GPU fraction (from node capabilities)
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

            if idle_gpu_nodes:
                logger.info(
                    f"[SelfplayScheduler] Idle GPU nodes ({idle_nodes}/{total_nodes}): "
                    f"{', '.join(idle_gpu_nodes[:5])}{'...' if len(idle_gpu_nodes) > 5 else ''}"
                )

        # 2. Training queue depth (check backpressure monitor)
        training_queue_depth = 0
        if self._backpressure_monitor:
            try:
                if hasattr(self._backpressure_monitor, '_last_queue_depth'):
                    cached_depth = self._backpressure_monitor._last_queue_depth
                    if isinstance(cached_depth, (int, float)):
                        training_queue_depth = int(cached_depth)
            except (AttributeError, TypeError, ValueError):
                pass

        # 3. Configs at Elo target fraction
        elo_target = 2000.0
        configs_at_target = 0
        total_configs = len(self._config_priorities)
        for cfg, priority in self._config_priorities.items():
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
            diversity_score=priority.diversity_score,
            cluster_game_count=self._get_cluster_game_count(priority.config_key),
        )

    def _compute_priority_score(self, priority: ConfigPriority) -> float:
        """Compute overall priority score for a configuration.

        Delegates to PriorityCalculator for all computation including starvation
        tiers (ULTRA/EMERGENCY/CRITICAL/WARNING). This method handles dynamic
        weight refresh, logging, and starvation alert emission.
        """
        self._compute_dynamic_weights()

        inputs = self._config_priority_to_inputs(priority)
        score = self._priority_calculator.compute_priority_score(inputs)

        # Log starvation tier and emit alerts for severe cases
        game_count = priority.game_count
        tier = self._priority_calculator.get_starvation_tier(game_count)
        if tier == "ULTRA":
            logger.warning(
                f"[SelfplayScheduler] ULTRA STARVATION: {priority.config_key} has only "
                f"{game_count} games. URGENT DATA NEEDED!"
            )
            starvation_cooldown_key = f"starvation_alert_{priority.config_key}"
            last_alert = getattr(self, "_starvation_alert_times", {}).get(starvation_cooldown_key, 0)
            if time.time() - last_alert > 300:
                self._emit_starvation_alert(priority.config_key, game_count, "ULTRA")
                if not hasattr(self, "_starvation_alert_times"):
                    self._starvation_alert_times: dict[str, float] = {}
                self._starvation_alert_times[starvation_cooldown_key] = time.time()
        elif tier == "EMERGENCY":
            logger.warning(
                f"[SelfplayScheduler] EMERGENCY: {priority.config_key} has only "
                f"{game_count} games."
            )
        elif tier in ("CRITICAL", "WARNING"):
            logger.info(
                f"[SelfplayScheduler] {tier}: {priority.config_key} has only "
                f"{game_count} games."
            )

        if abs(priority.momentum_multiplier - 1.0) > 0.1:
            logger.info(
                f"[SelfplayScheduler] Momentum multiplier applied to {priority.config_key}: "
                f"{priority.momentum_multiplier:.2f}x"
            )

        return score

    def _get_budget_with_intensity(
        self, game_count: int, elo: float, config_key: str
    ) -> int:
        """Get Gumbel budget factoring in training intensity.

        Higher intensity configs get higher budgets for better quality games.
        """
        intensity = self._get_training_intensity_for_config(config_key)
        return _get_budget_with_intensity(game_count, elo, intensity)

    def _get_training_intensity_for_config(self, config_key: str) -> str:
        """Get training intensity for a config from FeedbackLoopController.

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

    def get_target_games_for_config(self, config: str) -> int:
        """Get dynamic target games for a config (public accessor)."""
        current_elo = 1500.0
        for cfg_key, priority in self._config_priorities.items():
            if cfg_key == config:
                current_elo = getattr(priority, 'current_elo', 1500.0)
                break
        return _compute_target(config, current_elo)

    def _get_cascade_priority(self, config_key: str) -> float:
        """Get cascade training priority boost for a config.

        Returns:
            Priority multiplier (1.0 = normal, >1.0 = boosted)
        """
        from app.coordination.selfplay.priority_boosts import get_cascade_priority
        return get_cascade_priority(config_key)

    def _get_improvement_boosts(self) -> dict[str, float]:
        """Get improvement boosts from ImprovementOptimizer per config.

        Returns:
            Dict mapping config_key to boost value (-0.10 to +0.15)
        """
        from app.coordination.selfplay.priority_boosts import get_improvement_boosts
        return get_improvement_boosts()

    def _get_momentum_multipliers(self) -> dict[str, float]:
        """Get momentum multipliers from FeedbackAccelerator per config.

        Returns:
            Dict mapping config_key to multiplier value (0.5 to 1.5)
        """
        from app.coordination.selfplay.priority_boosts import get_momentum_multipliers
        return get_momentum_multipliers()

    def _get_architecture_boosts(self) -> dict[str, float]:
        """Get architecture-based boosts per config.

        Returns:
            Dict mapping config_key to boost value (0.0 to +0.30)
        """
        from app.coordination.selfplay.priority_boosts import get_architecture_boosts
        return get_architecture_boosts()
