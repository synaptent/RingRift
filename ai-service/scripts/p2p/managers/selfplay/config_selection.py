"""Config selection logic for SelfplayScheduler.

Extracted from selfplay_scheduler.py for modularity (Phase 2 decomposition).
Contains the core scheduling decision logic: pick_weighted_config(),
training completion handling, exploration boosts, and promotion failure tracking.
"""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scripts.p2p.models import NodeInfo

# Jan 2026: Import adaptive budget calculator for dynamic budget selection
try:
    from app.coordination.budget_calculator import (
        get_adaptive_budget_for_games,
        get_budget_tier_name,
        get_board_adjusted_budget,
    )
    BUDGET_CALCULATOR_AVAILABLE = True
except ImportError:
    BUDGET_CALCULATOR_AVAILABLE = False
    def get_adaptive_budget_for_games(game_count: int, elo: float) -> int:
        """Fallback budget when calculator not available."""
        if game_count < 100:
            return 64
        elif game_count < 500:
            return 150
        elif game_count < 1000:
            return 200
        elif elo >= 2000:
            return 3200
        elif elo >= 1800:
            return 1600
        else:
            return 800

    def get_board_adjusted_budget(board_type: str, budget: int, game_count: int, num_players: int = 2) -> int:
        """Fallback: no board adjustment when calculator not available."""
        return budget

    def get_budget_tier_name(budget: int) -> str:
        """Fallback tier name when calculator not available."""
        names = {64: "BOOTSTRAP_T1", 150: "BOOTSTRAP_T2", 200: "BOOTSTRAP_T3",
                 800: "STANDARD", 1600: "ULTIMATE", 3200: "MASTER"}
        return names.get(budget, f"CUSTOM({budget})")

# Import constants from canonical source
try:
    from app.p2p.constants import (
        PROMOTION_PENALTY_DURATION_CRITICAL,
        PROMOTION_PENALTY_DURATION_MULTIPLE,
        PROMOTION_PENALTY_DURATION_SINGLE,
        PROMOTION_PENALTY_FACTOR_CRITICAL,
        PROMOTION_PENALTY_FACTOR_MULTIPLE,
        PROMOTION_PENALTY_FACTOR_SINGLE,
        TRAINING_BOOST_DURATION,
    )
except ImportError:
    TRAINING_BOOST_DURATION = 1800
    PROMOTION_PENALTY_DURATION_CRITICAL = 7200
    PROMOTION_PENALTY_DURATION_MULTIPLE = 3600
    PROMOTION_PENALTY_DURATION_SINGLE = 1800
    PROMOTION_PENALTY_FACTOR_CRITICAL = 0.3
    PROMOTION_PENALTY_FACTOR_MULTIPLE = 0.5
    PROMOTION_PENALTY_FACTOR_SINGLE = 0.7

logger = logging.getLogger(__name__)


class ConfigSelectionMixin:
    """Mixin for config selection, training boosts, and promotion tracking.

    Provides:
    - pick_weighted_config(): Core scheduling decision
    - on_training_complete(): Training boost callback
    - set_exploration_boost() / get_exploration_boost(): Boost management
    - record_promotion_failure(): Promotion penalty tracking

    Assumes the host class has attributes from SelfplayScheduler.__init__():
    - _exploration_boosts, _training_complete_boosts, _previous_priorities
    - load_curriculum_weights, get_board_priority_overrides
    And methods from other mixins:
    - _filter_configs_by_gpu, _node_has_gpu (EngineSelectionMixin)
    - get_elo_based_priority_boost, get_rate_multiplier (PriorityCalculatorMixin)
    - _emit_selfplay_target_updated, _emit_selfplay_allocation_updated (EventHandlersMixin)
    """

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
                "engine_mode": "gumbel-mcts",  # Jan 2026: High-quality GPU mode
                "priority": 3,
            },
            # Priority 2: Heterogeneous cross-AI games (December 2025)
            # Per-player AI configs for maximum training diversity
            # Neural net learning from diverse opponent behaviors
            {
                "board_type": "hex8",
                "num_players": 2,
                "engine_mode": "cross-ai",
                "priority": 2,
                "player_ai_configs": {
                    1: {"engine": "gumbel-mcts", "budget": 150},
                    2: {"engine": "heuristic", "difficulty": 5},
                },
            },
            {
                "board_type": "square8",
                "num_players": 3,
                "engine_mode": "cross-ai",
                "priority": 2,
                "player_ai_configs": {
                    1: {"engine": "gumbel-mcts", "budget": 150},
                    2: {"engine": "brs"},
                    3: {"engine": "maxn"},
                },
            },
            {
                "board_type": "hex8",
                "num_players": 4,
                "engine_mode": "cross-ai",
                "priority": 2,
                "player_ai_configs": {
                    1: {"engine": "gumbel-mcts", "budget": 200},
                    2: {"engine": "heuristic", "difficulty": 6},
                    3: {"engine": "brs"},
                    4: {"engine": "random"},
                },
            },
        ]

        # Filter by GPU VRAM (avoid large boards on low-VRAM GPUs)
        # NOTE: Use gpu_vram_gb (GPU memory), NOT memory_gb (system RAM)
        # Bug fix Dec 2025: Was incorrectly using system RAM which filtered out
        # Vast.ai nodes with 12-16GB VRAM but >48GB system RAM
        gpu_vram = int(
            getattr(node, "gpu_vram_gb", 0)
            or getattr(node, "gpu_memory_gb", 0)
            or 0
        )
        if gpu_vram and gpu_vram < 48:
            # Jan 12, 2026: Allow hex8 and square8 on smaller GPUs, only filter truly large boards
            # Previous filter was too restrictive (only square8), blocking hex8 which fits in 8GB
            selfplay_configs = [
                c for c in selfplay_configs
                if c.get("board_type") not in ("square19", "hexagonal")
            ]

        # December 2025: Filter by GPU capability
        # CPU-only nodes should only get CPU-compatible engine modes
        # This prevents wasting compute on GPU-required modes that will fall back to heuristic
        selfplay_configs = self._filter_configs_by_gpu(selfplay_configs, node)

        if not selfplay_configs:
            node_id = getattr(node, "node_id", "unknown")
            logger.warning(
                f"No compatible selfplay configs for node {node_id} "
                f"(has_gpu={self._node_has_gpu(node)}, gpu_vram={gpu_vram}GB)"
            )
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

            # Dec 2025 Phase 2: Apply data starvation boost
            # Configs with few recent games get priority boost to ensure training data
            starvation_multiplier = self._get_data_starvation_boost(config_key)
            # Convert multiplier (0.5-2.0) to additive boost (-3 to +5)
            starvation_boost = int((starvation_multiplier - 1.0) * 5)
            starvation_boost = max(-3, min(5, starvation_boost))  # Clamp to -3..+5

            # Sprint 17.9 (Jan 2026): Apply bootstrap boost for critically underserved configs
            # This provides very aggressive priority for configs that need initial data collection
            bootstrap_boost = self._get_bootstrap_priority_boost(config_key)

            # Dec 2025 Phase 4D: Apply plateau penalty
            # Configs in plateau state (no Elo improvement) get 50% priority reduction
            # to redirect resources to configs making progress
            base_priority = (
                cfg.get("priority", 1)
                + elo_boost
                + curriculum_boost
                + board_priority_boost
                + rate_boost
                + starvation_boost
                + bootstrap_boost
            )

            if self._is_config_plateaued(config_key):
                # Apply 50% penalty for plateaued configs, but exempt CRITICAL
                # configs (priority 0) to prevent death spiral: no data -> no
                # training -> no Elo improvement -> plateau penalty -> less data
                is_critical = board_priority_overrides.get(config_key, 3) == 0
                if is_critical:
                    cfg["effective_priority"] = base_priority
                else:
                    cfg["effective_priority"] = max(1, int(base_priority * 0.5))
            else:
                cfg["effective_priority"] = base_priority

        # Session 17.34: Apply multi-config preference for large GPUs
        # Boosts priority for configs NOT currently running on the node
        selfplay_configs = self._apply_multi_config_preference(selfplay_configs, node)

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

        # December 2025: Apply mixed-engine selection for ALL board types
        # Instead of returning "mixed" mode, select a specific engine from the mix
        # This ensures BRS/MaxN diversity in hex8/square8 as well as large boards
        if selected:
            board_type = selected.get("board_type", "")
            engine_mode = selected.get("engine_mode", "")
            num_players = selected.get("num_players", 0)
            config_key = f"{board_type}_{num_players}p"

            # January 2026: Check if config should use mixed opponent training
            # Weak configs benefit from diverse opponents to break weak-vs-weak cycles
            if self.should_use_mixed_opponents(config_key):
                # Override engine mode to use MixedOpponentSelfplayRunner
                # This provides diverse opponents (random, heuristic, mcts, minimax, etc.)
                selected["engine_mode"] = "mixed-opponents"
                logger.info(
                    f"[MixedOpponents] {config_key}: forcing mixed opponent mode for diverse training"
                )
                return selected

            # Apply engine mix for any board type with "mixed" or "diverse" mode
            if engine_mode in ("mixed", "diverse"):
                has_gpu = self._node_has_gpu(node)
                num_players = selected.get("num_players", 0)
                config_key = f"{board_type}_{num_players}p"

                # Jan 2026: Get adaptive budget based on game count and Elo
                # This replaces static budgets with dynamic calculation
                adaptive_budget = self.get_adaptive_selfplay_budget(config_key)
                # Feb 2026: Apply large board budget caps scaled by player count
                game_count = self._get_game_counts_per_config().get(config_key, 0)
                adaptive_budget = get_board_adjusted_budget(board_type, adaptive_budget, game_count, num_players)

                # Jan 2026 Sprint 10: Check for quality boost - forces high-quality modes
                quality_boost = self.get_quality_boost(config_key)

                if quality_boost > 1.0 and has_gpu:
                    # Quality boost active - force high-quality Gumbel MCTS
                    # Budget scales with boost, starting from adaptive budget
                    boosted_budget = int(adaptive_budget * quality_boost)
                    # Cap at MASTER tier (3200) to prevent excessive compute
                    boosted_budget = min(boosted_budget, 3200)

                    actual_engine = "gumbel-mcts"
                    extra_args = {"budget": boosted_budget}

                    logger.info(
                        f"Quality boost override: {config_key} using '{actual_engine}' "
                        f"with budget={boosted_budget} (adaptive={adaptive_budget}, boost={quality_boost:.2f}x)"
                    )
                else:
                    # Normal selection from engine mix
                    actual_engine, extra_args = self._select_board_engine(
                        has_gpu=has_gpu,
                        board_type=board_type,
                        num_players=num_players,
                    )

                    # Jan 2026: Override static budget with adaptive budget for gumbel-mcts
                    # This is the KEY FIX - configs with 1000+ games now get 800+ budget
                    if actual_engine == "gumbel-mcts" and extra_args:
                        old_budget = extra_args.get("budget", 0)
                        extra_args["budget"] = adaptive_budget
                        if old_budget != adaptive_budget:
                            logger.info(
                                f"[AdaptiveBudget] {config_key}: {old_budget} -> {adaptive_budget} "
                                f"({get_budget_tier_name(adaptive_budget)})"
                            )

                # Update the config with the selected engine
                selected["engine_mode"] = actual_engine
                if extra_args:
                    selected["engine_extra_args"] = extra_args

                # Determine board category for logging
                board_category = "large" if board_type in self.LARGE_BOARD_TYPES else "standard"
                if quality_boost <= 1.0:
                    logger.info(
                        f"{board_category.capitalize()} board engine mix: {board_type}_{num_players}p '{engine_mode}' -> "
                        f"'{actual_engine}' (gpu={has_gpu}, extra_args={extra_args})"
                    )

            # Jan 2026: Handle mode-specific mixes (minimax-only, mcts-only, descent-only)
            # These modes now resolve to specific engines (including BRS/MaxN) for diversity tracking
            elif engine_mode in self.MODE_SPECIFIC_MIXES:
                has_gpu = self._node_has_gpu(node)
                num_players = selected.get("num_players", 0)
                config_key = f"{board_type}_{num_players}p"

                # Get the appropriate mix based on GPU availability
                gpu_mix, cpu_mix = self.MODE_SPECIFIC_MIXES[engine_mode]
                engine_mix = gpu_mix if has_gpu else cpu_mix

                # Filter to available engines (respect GPU requirements)
                available_engines = [
                    (mode, weight, gpu_required, args)
                    for mode, weight, gpu_required, args in engine_mix
                    if not gpu_required or has_gpu
                ]

                if available_engines:
                    # Weighted random selection
                    weighted_engines = []
                    for mode, weight, _gpu, args in available_engines:
                        weighted_engines.extend([(mode, args)] * weight)

                    if weighted_engines:
                        actual_engine, extra_args = random.choice(weighted_engines)

                        # Update the config with the selected engine
                        selected["engine_mode"] = actual_engine
                        if extra_args:
                            selected["engine_extra_args"] = extra_args

                        logger.info(
                            f"Mode-specific engine mix: {config_key} '{engine_mode}' -> "
                            f"'{actual_engine}' (gpu={has_gpu}, extra_args={extra_args})"
                        )

        # Session 17.22: Add architecture selection based on Elo performance
        # This closes the feedback loop: better-performing architectures get more selfplay
        if selected:
            selected_arch = self._select_architecture_for_config(
                board_type=selected.get("board_type", ""),
                num_players=selected.get("num_players", 2),
            )
            selected["model_version"] = selected_arch

        return selected

    # =========================================================================
    # Training Completion and Exploration Boost
    # =========================================================================

    def on_training_complete(self, config_key: str) -> None:
        """Handle training completion for a config.

        Called by P2P orchestrator when TRAINING_COMPLETED event fires.
        Refreshes selfplay priorities to potentially increase allocation
        for the just-trained config (more data needed for next training cycle).

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
            boost_duration = TRAINING_BOOST_DURATION
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

            # P0.2 Dec 2025: Emit allocation updated event
            # Get current curriculum weights to include in the event
            curriculum_weights = {}
            try:
                curriculum_weights = self.load_curriculum_weights()
            except (OSError, ValueError, KeyError, ImportError):
                pass  # Config file read/parse/import errors - use empty weights
            self._emit_selfplay_allocation_updated(
                config_key, curriculum_weights, boost_factor, "exploration_boost"
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

    def record_promotion_failure(self, config_key: str) -> None:
        """Record a promotion failure for curriculum feedback.

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
            penalty_duration = PROMOTION_PENALTY_DURATION_CRITICAL
            penalty_factor = PROMOTION_PENALTY_FACTOR_CRITICAL
        elif failure_count >= 2:
            # After 2 failures, reduce priority for 1 hour
            penalty_duration = PROMOTION_PENALTY_DURATION_MULTIPLE
            penalty_factor = PROMOTION_PENALTY_FACTOR_MULTIPLE
        else:
            # First failure, reduce priority for 30 minutes
            penalty_duration = PROMOTION_PENALTY_DURATION_SINGLE
            penalty_factor = PROMOTION_PENALTY_FACTOR_SINGLE

        # Store penalty in exploration boosts (negative boost = reduced priority)
        if not hasattr(self, "_promotion_penalties"):
            self._promotion_penalties: dict[str, tuple[float, float]] = {}
        self._promotion_penalties[config_key] = (penalty_factor, time.time() + penalty_duration)

        logger.info(
            f"[SelfplayScheduler] Applied {penalty_factor:.0%} priority penalty "
            f"to {config_key} for {penalty_duration}s"
        )
