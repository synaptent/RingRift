"""Priority calculation methods for SelfplayScheduler.

Extracted from selfplay_scheduler.py for better modularity.
Contains PriorityCalculatorMixin and ArchitectureSelectionMixin.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Callable

# Feb 2026: Import Elo gap utilities for distance-to-target priority factor
try:
    from app.config.thresholds import (
        get_elo_gap as _get_elo_gap,
        is_target_met as _is_target_met,
        ELO_TARGET_ALL_CONFIGS,
    )
    ELO_GAP_AVAILABLE = True
except ImportError:
    ELO_GAP_AVAILABLE = False
    ELO_TARGET_ALL_CONFIGS = 2000.0

    def _get_elo_gap(config_key: str, current_elo: float) -> float:
        """Fallback when thresholds not available."""
        return max(0.0, ELO_TARGET_ALL_CONFIGS - current_elo)

    def _is_target_met(config_key: str, current_elo: float) -> bool:
        """Fallback when thresholds not available."""
        return current_elo >= ELO_TARGET_ALL_CONFIGS

# Jan 2026: Import adaptive budget calculator for dynamic budget selection
try:
    from app.coordination.budget_calculator import (
        get_adaptive_budget_for_games,
        get_budget_tier_name,
        get_board_adjusted_budget,  # Jan 2026: Large board budget caps
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

# Session 17.22: Architecture selection feedback loop
try:
    from app.training.architecture_tracker import (
        get_allocation_weights as _get_architecture_weights,
    )
except ImportError:
    _get_architecture_weights = None  # type: ignore

try:
    from scripts.p2p.config.selfplay_job_configs import SELFPLAY_CONFIGS
except ImportError:
    SELFPLAY_CONFIGS = []

logger = logging.getLogger(__name__)


class ArchitectureSelectionMixin:
    """Mixin providing architecture selection based on Elo performance weights.

    Requires the host class to have:
        - _architecture_weights_cache: dict[str, tuple[dict[str, float], float]]
        - _architecture_weight_cache_ttl: float
    """

    def _get_cached_architecture_weights(
        self,
        board_type: str,
        num_players: int,
    ) -> dict[str, float]:
        """Get architecture weights, using cache if fresh.

        Jan 5, 2026 Session 17.26: Reduces DB queries by caching weights.
        Cache is refreshed immediately when ARCHITECTURE_WEIGHTS_UPDATED event is received,
        or falls back to DB query if cache is stale (>30 min).

        Args:
            board_type: Board type (e.g., "hex8", "square8")
            num_players: Number of players (2, 3, or 4)

        Returns:
            Dict mapping architecture names to allocation weights (sum to 1.0)
        """
        config_key = f"{board_type}_{num_players}p"
        now = time.time()

        # Check cache first
        if config_key in self._architecture_weights_cache:
            cached_weights, cached_time = self._architecture_weights_cache[config_key]
            if now - cached_time < self._architecture_weight_cache_ttl:
                return cached_weights

        # Cache miss or expired - fetch fresh from ArchitectureTracker
        if _get_architecture_weights is None:
            return {}

        try:
            weights = _get_architecture_weights(
                board_type=board_type,
                num_players=num_players,
                temperature=0.5,  # Moderate exploration vs exploitation
            )

            # Cache result
            if weights:
                self._architecture_weights_cache[config_key] = (weights, now)

            return weights or {}

        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"Error fetching architecture weights for {config_key}: {e}")
            return {}

    def _select_architecture_for_config(
        self,
        board_type: str,
        num_players: int,
    ) -> str:
        """Select architecture version based on Elo performance weights.

        Session 17.22: This closes the architecture selection feedback loop.
        The ArchitectureTracker records per-(config, architecture) Elo ratings,
        and compute_allocation_weights() returns softmax weights biased toward
        better-performing architectures.

        Session 17.26: Now uses cached weights to reduce DB queries.

        Args:
            board_type: Board type (e.g., "hex8", "square8")
            num_players: Number of players (2, 3, or 4)

        Returns:
            Architecture version string (e.g., "v5", "v2", "v5-heavy")
        """
        default_arch = "v5"

        # Use cached weights (refreshed via events or TTL fallback)
        weights = self._get_cached_architecture_weights(board_type, num_players)

        if not weights:
            # Cold start: no evaluation data yet
            logger.debug(
                f"No architecture weights for {board_type}_{num_players}p, "
                f"using default: {default_arch}"
            )
            return default_arch

        # Weighted random selection based on Elo performance
        architectures = list(weights.keys())
        arch_weights = list(weights.values())

        # Use random.choices for weighted selection
        selected_arch = random.choices(architectures, weights=arch_weights, k=1)[0]

        logger.info(
            f"Architecture selection for {board_type}_{num_players}p: "
            f"{selected_arch} (weights: {weights})"
        )

        return selected_arch


class PriorityCalculatorMixin:
    """Mixin providing priority calculation methods for selfplay scheduling.

    Requires the host class to have:
        - get_cluster_elo: Callable[[], dict[str, Any]]
        - load_curriculum_weights: Callable[[], dict[str, float]]
        - _configs_in_training_pipeline: set[str]
        - _feedback_states: dict[str, Any]
        - _training_complete_boosts: dict[str, float]
        - _last_config_job_time: dict[str, float]  (lazily initialized)
        - _config_game_counts: dict[str, int]  (lazily initialized)
        - _p2p_game_counts: dict[str, int]  (lazily initialized)
        - _unified_game_counts: dict[str, int]  (lazily initialized)
        - verbose: bool
    """

    # Minimum allocation constants
    MINIMUM_ALLOCATION_PERCENT = 0.20  # Reserve 20% of jobs for underserved
    UNDERSERVED_THRESHOLD = 5000  # Configs below this game count are "underserved"
    CRITICAL_THRESHOLD = 1000  # Configs below this get highest priority
    MINIMUM_ENFORCE_INTERVAL = 30.0  # Seconds between enforcement checks

    # Bootstrap priority thresholds (Jan 7, 2026 - 48h Autonomous Operation)
    # Very aggressive priority boosts for configs with almost no games
    # These override normal priority calculation for critically underserved configs
    BOOTSTRAP_THRESHOLDS = {
        "critical": 50,     # < 50 games = max boost
        "low": 500,         # < 500 games = high boost (was 200)
        "medium": 2000,     # < 2000 games = moderate boost (was 500)
        "bootstrap": 5000,  # < 5000 games = minimal boost (NEW)
    }
    BOOTSTRAP_BOOSTS = {
        "critical": 100,    # Jan 7: Increased from 50 to 100 for data-starved configs
        "low": 75,          # Jan 7: Increased from 15 to 75 for low-data configs
        "medium": 50,       # Jan 7: Increased from 10 to 50 for medium-data configs
        "bootstrap": 25,    # Jan 7: NEW tier for configs approaching training threshold
    }

    def _pick_simple_weighted_config(self) -> str | None:
        """Pick a selfplay config key using simple priority-weighted selection.

        January 2026 Sprint 6: Simplified version of pick_weighted_config for
        use in preemptive job spawning where we don't have node info.

        Returns:
            Config key (e.g., "hex8_2p") or None if no valid config
        """
        # Standard configs with priority weights
        STANDARD_CONFIGS = [
            # High priority: Underserved/complex configs
            ("hexagonal_3p", 8),
            ("hexagonal_4p", 8),
            ("square19_3p", 7),
            ("square19_4p", 7),
            ("hex8_3p", 6),
            ("hex8_4p", 6),
            # Medium priority: Standard 2-player
            ("hex8_2p", 5),
            ("square8_2p", 5),
            ("hexagonal_2p", 5),
            # Lower priority: Well-covered configs
            ("square8_3p", 4),
            ("square8_4p", 4),
            ("square19_2p", 4),
        ]

        # Apply dynamic boosts from curriculum weights
        curriculum_weights = {}
        try:
            curriculum_weights = self.load_curriculum_weights()
        except (OSError, IOError):
            # File I/O errors
            pass
        except (ValueError, TypeError, KeyError, AttributeError):
            # Data structure errors
            pass

        weighted_configs = []
        for config_key, base_priority in STANDARD_CONFIGS:
            # Apply curriculum weight boost
            curriculum_mult = curriculum_weights.get(config_key, 1.0)

            # Apply staleness boost for underserved configs
            staleness_boost = self._get_staleness_boost(config_key)

            effective_priority = int(base_priority * curriculum_mult + staleness_boost)

            # Feb 2026: Apply Elo gap factor - configs further from target
            # get proportionally more selfplay allocation
            try:
                current_elo = self.get_config_elo(config_key)
                if _is_target_met(config_key, current_elo):
                    elo_gap_factor = 0.3  # Maintenance mode
                else:
                    elo_gap = _get_elo_gap(config_key, current_elo)
                    elo_gap_factor = min(3.0, 1.0 + (elo_gap / 500.0))
                effective_priority = int(effective_priority * elo_gap_factor)
            except (AttributeError, TypeError):
                pass  # Elo data unavailable, skip gap factor

            effective_priority = max(1, effective_priority)

            weighted_configs.extend([config_key] * effective_priority)

        if not weighted_configs:
            return None

        return random.choice(weighted_configs)

    def _get_staleness_boost(self, config_key: str) -> int:
        """Get staleness-based priority boost for a config.

        Configs that haven't had recent games get higher priority.
        """
        try:
            # Initialize tracking dict if needed
            if not hasattr(self, "_last_config_job_time"):
                self._last_config_job_time = {}

            # Check last job time for this config
            last_job_time = self._last_config_job_time.get(config_key, 0)
            if last_job_time == 0:
                return 3  # Never had a job, high boost

            age_hours = (time.time() - last_job_time) / 3600
            if age_hours > 24:
                return 3
            elif age_hours > 12:
                return 2
            elif age_hours > 6:
                return 1
            return 0
        except (TypeError, AttributeError, KeyError):
            # Defensive - shouldn't happen but gracefully handle
            return 0

    def record_job_dispatched(self, config_key: str) -> None:
        """Record that a job was dispatched for a config.

        January 2026 Sprint 6: Used for staleness-based priority boosting.
        Configs that haven't had recent jobs get higher priority.

        Args:
            config_key: Config identifier (e.g., "hex8_2p")
        """
        if not hasattr(self, "_last_config_job_time"):
            self._last_config_job_time = {}
        self._last_config_job_time[config_key] = time.time()

    def _get_enforced_minimum_allocation(self) -> str | None:
        """Check if we should force allocation to an underserved config.

        January 2026 Sprint 6: Implements minimum allocation enforcement.
        Reserves 20% of cluster capacity for underserved configs to guarantee
        they receive games even when higher-priority configs dominate.

        Returns:
            Config key if enforcement is active, None otherwise
        """
        try:
            now = time.time()

            # Rate limit enforcement checks
            if hasattr(self, "_last_enforcement_check"):
                if now - self._last_enforcement_check < self.MINIMUM_ENFORCE_INTERVAL:
                    return None
            self._last_enforcement_check = now

            # Random 20% chance to enforce (simulates 20% allocation)
            if random.random() > self.MINIMUM_ALLOCATION_PERCENT:
                return None

            # Find underserved configs
            underserved = self._get_underserved_configs()
            if not underserved:
                return None

            # Sort by game count (lowest first = most critical)
            underserved.sort(key=lambda x: x[1])

            # Pick the most critical config (lowest game count)
            selected_config, game_count = underserved[0]

            logger.info(
                f"[MinimumAllocation] Enforcing allocation to {selected_config} "
                f"(only {game_count} games, threshold={self.UNDERSERVED_THRESHOLD})"
            )

            return selected_config

        except Exception as e:
            logger.debug(f"[MinimumAllocation] Enforcement check failed: {e}")
            return None

    def _get_underserved_configs(self) -> list[tuple[str, int]]:
        """Get list of configs below the underserved threshold.

        Returns:
            List of (config_key, game_count) tuples for underserved configs
        """
        try:
            # Get game counts from tracking or discovery
            game_counts = self._get_game_counts_per_config()

            underserved = []
            all_configs = [
                "hex8_2p", "hex8_3p", "hex8_4p",
                "square8_2p", "square8_3p", "square8_4p",
                "square19_2p", "square19_3p", "square19_4p",
                "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
            ]

            for config_key in all_configs:
                count = game_counts.get(config_key, 0)
                if count < self.UNDERSERVED_THRESHOLD:
                    underserved.append((config_key, count))

            return underserved

        except Exception as e:
            logger.debug(f"[MinimumAllocation] Failed to get underserved configs: {e}")
            return []

    def _get_game_counts_per_config(self) -> dict[str, int]:
        """Get current game counts for each config.

        January 14, 2026: Updated to prefer unified counts which include all sources
        (LOCAL, CLUSTER, S3, OWC external drive on mac-studio).

        Priority order:
        1. Unified counts (via UnifiedGameAggregator) - most complete
        2. P2P manifest counts (cluster-wide view)
        3. Local tracking (fallback)

        Returns:
            Dict mapping config_key -> game_count
        """
        try:
            # First try unified counts (includes OWC, S3, all cluster nodes)
            if hasattr(self, "_unified_game_counts") and self._unified_game_counts:
                return dict(self._unified_game_counts)

            # Fall back to P2P manifest (cluster-wide but no OWC/S3)
            if hasattr(self, "_p2p_game_counts") and self._p2p_game_counts:
                return dict(self._p2p_game_counts)

            # Fall back to local tracking
            if hasattr(self, "_config_game_counts"):
                return dict(self._config_game_counts)

            # Return empty dict if no data available
            return {}

        except (TypeError, AttributeError):
            # Defensive - shouldn't happen but gracefully handle
            return {}

    def update_config_game_count(self, config_key: str, game_count: int) -> None:
        """Update the tracked game count for a config.

        Called when game data is synced or generated.

        Args:
            config_key: Config identifier (e.g., "hex8_2p")
            game_count: Current total game count
        """
        if not hasattr(self, "_config_game_counts"):
            self._config_game_counts = {}
        self._config_game_counts[config_key] = game_count

    def _get_bootstrap_priority_boost(self, config_key: str) -> int:
        """Get priority boost for configs with critically low game counts.

        Sprint 17.9 (Jan 2026): Implements bootstrap priority for underserved configs.
        Jan 7, 2026: Updated thresholds and boosts for 48h autonomous operation.

        This provides very aggressive priority boosts for configs that need immediate
        data collection to enable initial training.

        Args:
            config_key: Config identifier (e.g., "hex8_2p")

        Returns:
            Priority boost value (Jan 7, 2026 - 48h autonomous):
            - 100: Critical (<50 games) - needs immediate bootstrap
            - 75: Low (<500 games) - needs strong priority
            - 50: Medium (<2000 games) - needs moderate priority
            - 25: Bootstrap (<5000 games) - approaching training threshold
            - 0: Normal (>=5000 games) - no bootstrap boost needed
        """
        try:
            game_counts = self._get_game_counts_per_config()
            game_count = game_counts.get(config_key, 0)

            if game_count < self.BOOTSTRAP_THRESHOLDS["critical"]:
                boost = self.BOOTSTRAP_BOOSTS["critical"]
                logger.debug(
                    f"[BootstrapPriority] {config_key}: {game_count} games "
                    f"(< {self.BOOTSTRAP_THRESHOLDS['critical']}) -> +{boost} critical boost"
                )
                return boost
            elif game_count < self.BOOTSTRAP_THRESHOLDS["low"]:
                boost = self.BOOTSTRAP_BOOSTS["low"]
                logger.debug(
                    f"[BootstrapPriority] {config_key}: {game_count} games "
                    f"(< {self.BOOTSTRAP_THRESHOLDS['low']}) -> +{boost} low boost"
                )
                return boost
            elif game_count < self.BOOTSTRAP_THRESHOLDS["medium"]:
                boost = self.BOOTSTRAP_BOOSTS["medium"]
                logger.debug(
                    f"[BootstrapPriority] {config_key}: {game_count} games "
                    f"(< {self.BOOTSTRAP_THRESHOLDS['medium']}) -> +{boost} medium boost"
                )
                return boost
            elif game_count < self.BOOTSTRAP_THRESHOLDS["bootstrap"]:
                boost = self.BOOTSTRAP_BOOSTS["bootstrap"]
                logger.debug(
                    f"[BootstrapPriority] {config_key}: {game_count} games "
                    f"(< {self.BOOTSTRAP_THRESHOLDS['bootstrap']}) -> +{boost} bootstrap boost"
                )
                return boost
            else:
                return 0

        except Exception as e:
            logger.debug(f"[BootstrapPriority] Failed to compute boost for {config_key}: {e}")
            return 0

    def get_config_priorities(self) -> dict[str, float]:
        """Get priority scores for all configs, used by autonomous queue loop.

        Session 17.42: Added to fix selfplay not prioritizing starved configs.
        The autonomous_queue_loop was calling this method but it didn't exist,
        causing it to fall back to round-robin selection which ignored starvation.

        Feb 2026: Added Elo gap factor to prioritize configs furthest from target.
        Configs at 1483 Elo (517 gap) get ~2x, configs at 2000+ get 0.3x maintenance.

        Returns:
            Dict mapping config_key -> priority score (higher = more priority).
            Incorporates:
            - Bootstrap priority boost for data-starved configs (<50 games = +50)
            - Staleness boost from curriculum state
            - Base weights from standard config priorities
            - Elo gap factor: multiplier based on distance to 2000 Elo target
        """
        # All 12 canonical configs with base weights
        BASE_WEIGHTS = {
            "hexagonal_3p": 8,
            "hexagonal_4p": 8,
            "square19_3p": 7,
            "square19_4p": 7,
            "hex8_3p": 6,
            "hex8_4p": 6,
            "hex8_2p": 5,
            "square8_2p": 5,
            "hexagonal_2p": 5,
            "square8_3p": 4,
            "square8_4p": 4,
            "square19_2p": 4,
        }

        priorities = {}
        game_counts = self._get_game_counts_per_config()

        for config_key, base_weight in BASE_WEIGHTS.items():
            # Start with base weight
            priority = float(base_weight)

            # Add bootstrap priority boost for data-starved configs
            # This is the CRITICAL part - gives +50 boost to configs with <50 games
            bootstrap_boost = self._get_bootstrap_priority_boost(config_key)
            priority += bootstrap_boost

            # Add staleness boost from curriculum state
            staleness_boost = self._get_staleness_boost(config_key)
            priority += staleness_boost

            # Feb 2026: Apply Elo gap factor as multiplier.
            # Configs further from 2000 Elo target get more selfplay time.
            # This is the key insight: hexagonal_2p at 1483 needs ~3x more
            # iterations than square19_4p at 1982, but was getting equal allocation.
            #
            # Scale: gap_factor = 1.0 + (elo_gap / 500)
            #   - 0 gap (at/above target): 0.3x (maintenance mode)
            #   - 200 gap (1800 Elo):      1.4x
            #   - 500 gap (1500 Elo):      2.0x
            #   - Capped at 3.0x
            current_elo = self.get_config_elo(config_key)
            if _is_target_met(config_key, current_elo):
                elo_gap_factor = 0.3
            else:
                elo_gap = _get_elo_gap(config_key, current_elo)
                elo_gap_factor = min(3.0, 1.0 + (elo_gap / 500.0))
            priority *= elo_gap_factor

            # Log configs with significant boosts for visibility
            game_count = game_counts.get(config_key, 0)
            if bootstrap_boost > 0 or staleness_boost > 5 or elo_gap_factor > 1.5:
                logger.debug(
                    f"[ConfigPriorities] {config_key}: base={base_weight}, "
                    f"bootstrap={bootstrap_boost}, staleness={staleness_boost}, "
                    f"elo_gap_factor={elo_gap_factor:.2f}, "
                    f"total={priority:.1f} (games={game_count}, elo={current_elo:.0f})"
                )

            priorities[config_key] = priority

        return priorities

    def update_p2p_game_counts(self, counts: dict[str, int]) -> None:
        """Update game counts from P2P manifest data.

        Called by P2P orchestrator when manifest data is refreshed.

        Args:
            counts: Dict mapping config_key -> game_count
        """
        self._p2p_game_counts = dict(counts)

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

    def get_config_elo(self, config_key: str) -> float:
        """Get the current Elo rating for a config.

        Jan 2026: Added to support adaptive budget calculation based on Elo.

        Args:
            config_key: Config key (e.g., "hex8_2p")

        Returns:
            Current Elo rating for the config (default 1500 if unknown)
        """
        try:
            cluster_elo = self.get_cluster_elo()
            top_models = cluster_elo.get("top_models", [])

            # Find best model for this config
            for model in top_models:
                model_name = model.get("name", "")
                # Match config key pattern in model name
                if config_key in model_name or config_key.replace("_", "") in model_name.replace("_", ""):
                    return model.get("elo", 1500)

            # Also check by board type + player count
            parts = config_key.replace("_", "").split("p")
            if len(parts) == 2:
                board_type = parts[0].rstrip("0123456789")
                num_players = config_key.split("_")[-1].replace("p", "")
                for model in top_models:
                    model_name = model.get("name", "")
                    if board_type in model_name and num_players in model_name:
                        return model.get("elo", 1500)

        except (AttributeError, TypeError):
            pass

        return 1500  # Default Elo for unknown configs

    def should_use_mixed_opponents(self, config_key: str) -> bool:
        """Determine if config should use mixed opponent training.

        January 2026: Added to fix training feedback loop - weak configs benefit
        from mixed opponents to break weak-vs-weak cycles. When models only play
        against themselves, they can get stuck in local optima.

        Mixed opponent training provides diverse signal from:
        - Random opponents (exploration)
        - Heuristic opponents (tactical patterns)
        - MCTS opponents (strategic depth)
        - Minimax opponents (game-theoretic optimal)
        - Policy-only opponents (neural patterns)

        Thresholds based on data poverty and Elo stagnation:
        - Less than 5000 games (data poverty)
        - Elo below 1200 (still learning fundamentals)
        - 3p or 4p configs (benefit more from diverse opponents)

        Args:
            config_key: Config key (e.g., "hex8_4p")

        Returns:
            True if config should use mixed opponent training
        """
        game_counts = self._get_game_counts_per_config()
        game_count = game_counts.get(config_key, 0)
        current_elo = self.get_config_elo(config_key)

        # Data poverty - config needs more diverse training signal
        if game_count < 5000:
            logger.debug(
                f"[MixedOpponents] {config_key}: using mixed (data poverty: {game_count} games < 5000)"
            )
            return True

        # Elo still low - model is learning fundamentals, needs diverse opponents
        if current_elo < 1200:
            logger.debug(
                f"[MixedOpponents] {config_key}: using mixed (low Elo: {current_elo:.0f} < 1200)"
            )
            return True

        # 3p and 4p configs benefit more from mixed opponents due to complexity
        if config_key.endswith("_3p") or config_key.endswith("_4p"):
            # Use mixed if not yet at high game count
            if game_count < 10000:
                logger.debug(
                    f"[MixedOpponents] {config_key}: using mixed (multiplayer with {game_count} games)"
                )
                return True

        return False

    def get_adaptive_selfplay_budget(self, config_key: str) -> int:
        """Get adaptive Gumbel budget based on game count and Elo.

        Jan 2026: Replaces static budget with dynamic calculation using the
        budget calculator. Configs with more games and higher Elo get higher
        budgets for better quality training data.

        Budget tiers:
        - Bootstrap (<100 games): 64 (max throughput)
        - Bootstrap (<500 games): 150 (fast iteration)
        - Bootstrap (<1000 games): 200 (balanced)
        - Standard (>=1000 games, <1500 Elo): 800
        - Quality (>=1000 games, 1500+ Elo): 800
        - Ultimate (>=1000 games, 1800+ Elo): 1600
        - Master (>=1000 games, 2000+ Elo): 3200

        Args:
            config_key: Config key (e.g., "hex8_2p")

        Returns:
            Gumbel MCTS budget
        """
        game_counts = self._get_game_counts_per_config()
        game_count = game_counts.get(config_key, 0)
        elo = self.get_config_elo(config_key)

        budget = get_adaptive_budget_for_games(game_count, elo)
        tier_name = get_budget_tier_name(budget)

        logger.debug(
            f"[Budget] {config_key}: games={game_count}, elo={elo:.0f} -> "
            f"budget={budget} ({tier_name})"
        )

        return budget

    def _get_data_starvation_boost(self, config_key: str) -> float:
        """Get priority boost for configs that are data-starved for training.

        December 2025 Phase 2: Boost selfplay allocation for configs that have
        few recent games, ensuring training data availability across all configs.

        Args:
            config_key: Config key (e.g., "hex8_2p")

        Returns:
            Boost multiplier:
            - 0.5: Config is actively training (reduce selfplay)
            - 1.0: Normal priority (no boost)
            - 1.5: Moderate data starvation (<50 games since training)
            - 2.0: High data starvation (<25 games since training)
        """
        # If config is currently in training pipeline, reduce selfplay priority
        # to avoid wasting resources on a config that's actively learning
        if config_key in self._configs_in_training_pipeline:
            return 0.5

        # Check recent game count from feedback state
        state = self._feedback_states.get(config_key)
        if state is not None:
            # Try to get games_since_last_training attribute
            games = getattr(state, "games_since_last_training", None)
            if games is not None:
                if games < 25:
                    # Very data-starved - high boost to generate more data
                    return 2.0
                elif games < 50:
                    # Moderately data-starved
                    return 1.5

        # Also check if there's a training complete boost active
        # (recently trained configs need more data for next cycle)
        if config_key in self._training_complete_boosts:
            expiry = self._training_complete_boosts[config_key]
            if time.time() < expiry:
                return 1.3  # Slight boost after training completes

        return 1.0  # No boost
