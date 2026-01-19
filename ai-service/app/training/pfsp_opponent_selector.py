"""Prioritized Fictitious Self-Play (PFSP) Opponent Selector.

This module implements PFSP for smarter opponent selection during selfplay.
Instead of uniform random selection, PFSP prioritizes opponents where the
current model has a win rate near 50%, maximizing learning signal.

Key insights:
- Games against too-weak opponents teach nothing (always win)
- Games against too-strong opponents also teach little (always lose)
- Games with ~50% win rate provide maximum learning signal

Algorithm:
1. Track historical win rates against each opponent model version
2. Compute priority: higher for win rates near 50%
3. Sample opponents proportional to priority

Usage:
    from app.training.pfsp_opponent_selector import (
        PFSPOpponentSelector,
        get_pfsp_selector,
    )

    selector = get_pfsp_selector()

    # Select opponent for current model
    opponent = selector.select_opponent(
        current_model="hex8_2p_v123",
        available_opponents=["hex8_2p_v120", "hex8_2p_v121", "hex8_2p_v122"],
    )

    # Record game result
    selector.record_game_result(
        current_model="hex8_2p_v123",
        opponent="hex8_2p_v120",
        current_model_won=True,
    )

Based on: MuZero Reanalyze and AlphaZero PFSP techniques.
Created: December 2025
Purpose: Strengthen AI training self-improvement loop (Phase 14)
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PFSPConfig:
    """Configuration for PFSP opponent selection.

    Zone of Proximal Development (ZPD) Mode:
        Research shows optimal learning occurs when training against slightly
        stronger opponents (~40% win rate) rather than equal-strength opponents
        (50% win rate). This is because harder games provide more learning signal
        per sample, pushing the model toward better strategies.

        Set target_win_rate=0.4 and zpd_mode=True for ZPD training (recommended).
    """

    # Priority function parameters
    # ZPD optimal: 0.4 (slightly harder opponents maximize learning)
    # Traditional: 0.5 (equal-strength opponents)
    target_win_rate: float = 0.4  # ZPD: optimal ~40% win rate (was 0.5)
    zpd_elo_offset: float = 100.0  # Prefer opponents ~100 Elo higher when selecting
    zpd_mode: bool = True  # Enable Zone of Proximal Development opponent selection
    priority_sharpness: float = 0.1  # Lower = sharper priority (more focus on target)
    min_priority: float = 0.1  # Minimum priority to ensure exploration

    # Win rate estimation
    min_games_for_estimate: int = 10  # Minimum games before using historical win rate
    default_win_rate: float = 0.5  # Default when no history available
    decay_factor: float = 0.99  # Exponential decay for older games

    # Model history
    max_opponents_tracked: int = 100  # Maximum opponent versions to track
    max_games_per_matchup: int = 500  # Maximum games to store per matchup

    # Exploration
    exploration_epsilon: float = 0.1  # Probability of random opponent selection


@dataclass
class MatchupRecord:
    """Record of games between two model versions."""

    wins: int = 0
    losses: int = 0
    draws: int = 0
    last_played: float = 0.0
    weighted_wins: float = 0.0  # Decay-weighted wins
    weighted_games: float = 0.0  # Decay-weighted total games

    @property
    def total_games(self) -> int:
        """Total games played."""
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        """Win rate (0-1)."""
        if self.total_games == 0:
            return 0.5
        return self.wins / self.total_games

    @property
    def weighted_win_rate(self) -> float:
        """Decay-weighted win rate."""
        if self.weighted_games < 1.0:
            return 0.5
        return self.weighted_wins / self.weighted_games


# =============================================================================
# Main Selector
# =============================================================================


class PFSPOpponentSelector:
    """Select opponents using Prioritized Fictitious Self-Play.

    PFSP prioritizes opponents where the current model has a win rate
    close to 50%, as these games provide the most learning signal.
    """

    def __init__(self, config: PFSPConfig | None = None):
        """Initialize the PFSP selector.

        Args:
            config: Configuration for PFSP behavior
        """
        self.config = config or PFSPConfig()

        # Matchup history: (current_model, opponent) -> MatchupRecord
        self._matchups: dict[tuple[str, str], MatchupRecord] = defaultdict(MatchupRecord)

        # Model pool management
        self._model_versions: dict[str, list[str]] = defaultdict(list)  # config -> versions
        self._model_elos: dict[str, float] = {}  # model_id -> ELO
        self._last_decay_time: float = time.time()

        # Statistics
        self._selections: list[dict[str, Any]] = []

    def select_opponent(
        self,
        current_model: str,
        available_opponents: list[str],
        config_key: str | None = None,
    ) -> str:
        """Select an opponent for selfplay using PFSP.

        Args:
            current_model: Current model version ID
            available_opponents: List of available opponent model IDs
            config_key: Optional config key for filtering (e.g., "hex8_2p")

        Returns:
            Selected opponent model ID
        """
        if not available_opponents:
            logger.warning("[PFSP] No opponents available, returning current model")
            return current_model

        # Exploration: random selection with epsilon probability
        if random.random() < self.config.exploration_epsilon:
            selected = random.choice(available_opponents)
            self._record_selection(current_model, selected, "exploration")
            return selected

        # Compute priorities for each opponent
        priorities = []
        for opponent in available_opponents:
            priority = self._compute_priority(current_model, opponent)
            priorities.append(priority)

        # Normalize to probabilities
        total_priority = sum(priorities)
        if total_priority <= 0:
            # Fallback to uniform
            selected = random.choice(available_opponents)
            self._record_selection(current_model, selected, "fallback_uniform")
            return selected

        probs = [p / total_priority for p in priorities]

        # Sample based on priorities
        selected = random.choices(available_opponents, weights=probs, k=1)[0]
        self._record_selection(
            current_model, selected, "pfsp", priority=priorities[available_opponents.index(selected)]
        )

        return selected

    def _compute_priority(self, current_model: str, opponent: str) -> float:
        """Compute PFSP priority for an opponent.

        Priority is highest when win rate is near target (default 40% for ZPD).

        When ZPD mode is enabled, we also add a bonus for opponents that are
        slightly stronger (~100 Elo higher). This implements Zone of Proximal
        Development training, where learning from harder games maximizes
        improvement per training sample.

        Args:
            current_model: Current model ID
            opponent: Opponent model ID

        Returns:
            Priority value (higher = more likely to be selected)
        """
        # Get historical win rate
        win_rate = self._get_win_rate(current_model, opponent)

        # PFSP priority: 1 / (|win_rate - target| + sharpness)
        # Higher when win_rate is close to target
        deviation = abs(win_rate - self.config.target_win_rate)
        priority = 1.0 / (deviation + self.config.priority_sharpness)

        # ZPD mode: boost priority for slightly stronger opponents
        # This makes us prefer opponents ~100 Elo higher, as those games
        # provide maximum learning signal per sample
        if self.config.zpd_mode:
            current_elo = self._model_elos.get(current_model, 1500.0)
            opponent_elo = self._model_elos.get(opponent, 1500.0)
            elo_diff = opponent_elo - current_elo  # Positive = opponent is stronger

            # Boost for stronger opponents, maximum at zpd_elo_offset Elo higher
            # Gaussian-like boost: exp(-((diff - target)^2) / (2 * sigma^2))
            # Peak at +100 Elo (zpd_elo_offset), decays away from there
            target_diff = self.config.zpd_elo_offset
            sigma = 100.0  # Width of the bonus window
            zpd_bonus = math.exp(-((elo_diff - target_diff) ** 2) / (2 * sigma ** 2))

            # Scale the bonus (up to 2x priority for perfect ZPD match)
            priority *= (1.0 + zpd_bonus)

        # Ensure minimum exploration
        priority = max(priority, self.config.min_priority)

        return priority

    def _get_win_rate(self, current_model: str, opponent: str) -> float:
        """Get historical win rate against an opponent.

        Args:
            current_model: Current model ID
            opponent: Opponent model ID

        Returns:
            Win rate (0-1), or default if insufficient history
        """
        key = (current_model, opponent)
        record = self._matchups.get(key)

        if record is None or record.total_games < self.config.min_games_for_estimate:
            # Use ELO-based estimate if available
            return self._estimate_from_elo(current_model, opponent)

        # Use weighted win rate (more recent games count more)
        return record.weighted_win_rate

    def _estimate_from_elo(self, current_model: str, opponent: str) -> float:
        """Estimate win rate from ELO ratings.

        Args:
            current_model: Current model ID
            opponent: Opponent model ID

        Returns:
            Estimated win rate based on ELO difference
        """
        current_elo = self._model_elos.get(current_model, 1500.0)
        opponent_elo = self._model_elos.get(opponent, 1500.0)

        # Expected score formula (ELO)
        elo_diff = current_elo - opponent_elo
        expected = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

        return expected

    def record_game_result(
        self,
        current_model: str,
        opponent: str,
        current_model_won: bool,
        draw: bool = False,
    ) -> None:
        """Record a game result for updating PFSP statistics.

        Args:
            current_model: Current model ID
            opponent: Opponent model ID
            current_model_won: Whether current model won
            draw: Whether game was a draw
        """
        key = (current_model, opponent)
        record = self._matchups[key]

        now = time.time()

        # Apply decay to existing records periodically
        if now - self._last_decay_time > 3600:  # Every hour
            self._apply_decay()
            self._last_decay_time = now

        # Update record
        if draw:
            record.draws += 1
            record.weighted_wins += 0.5
            record.weighted_games += 1.0
        elif current_model_won:
            record.wins += 1
            record.weighted_wins += 1.0
            record.weighted_games += 1.0
        else:
            record.losses += 1
            record.weighted_games += 1.0

        record.last_played = now

        # Prune if too many games
        if record.total_games > self.config.max_games_per_matchup:
            self._prune_matchup(key)

    def _apply_decay(self) -> None:
        """Apply exponential decay to all matchup records."""
        decay = self.config.decay_factor

        for record in self._matchups.values():
            record.weighted_wins *= decay
            record.weighted_games *= decay

        logger.debug(f"[PFSP] Applied decay factor {decay} to {len(self._matchups)} matchups")

    def _prune_matchup(self, key: tuple[str, str]) -> None:
        """Prune excess games from a matchup record.

        Keeps only weighted statistics, resets raw counts to prevent
        unbounded growth.
        """
        record = self._matchups[key]

        # Reset raw counts but keep weighted
        record.wins = int(record.wins * 0.5)
        record.losses = int(record.losses * 0.5)
        record.draws = int(record.draws * 0.5)

    def register_model(
        self,
        model_id: str,
        config_key: str,
        elo: float = 1500.0,
    ) -> None:
        """Register a model version for opponent selection.

        Args:
            model_id: Model version ID
            config_key: Configuration key (e.g., "hex8_2p")
            elo: ELO rating of the model
        """
        if model_id not in self._model_versions[config_key]:
            self._model_versions[config_key].append(model_id)

            # Prune if too many versions
            if len(self._model_versions[config_key]) > self.config.max_opponents_tracked:
                oldest = self._model_versions[config_key].pop(0)
                # Also clean up matchups involving this model
                self._cleanup_model(oldest)

        self._model_elos[model_id] = elo

    def _cleanup_model(self, model_id: str) -> None:
        """Clean up matchup records involving a retired model.

        Args:
            model_id: Model ID to clean up
        """
        keys_to_remove = [
            key for key in self._matchups.keys()
            if model_id in key
        ]
        for key in keys_to_remove:
            del self._matchups[key]

        if model_id in self._model_elos:
            del self._model_elos[model_id]

    def get_available_opponents(self, config_key: str) -> list[str]:
        """Get list of available opponents for a config.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")

        Returns:
            List of model IDs available as opponents
        """
        return list(self._model_versions.get(config_key, []))

    def _record_selection(
        self,
        current_model: str,
        selected: str,
        method: str,
        priority: float = 0.0,
    ) -> None:
        """Record a selection for statistics.

        Args:
            current_model: Current model ID
            selected: Selected opponent ID
            method: Selection method used
            priority: Priority value (if PFSP)
        """
        self._selections.append({
            "timestamp": time.time(),
            "current_model": current_model,
            "selected": selected,
            "method": method,
            "priority": priority,
        })

        # Keep only recent selections
        if len(self._selections) > 1000:
            self._selections = self._selections[-1000:]

        logger.debug(f"[PFSP] Selected {selected} for {current_model} via {method}")

    def get_statistics(self) -> dict[str, Any]:
        """Get PFSP statistics.

        Returns:
            Dictionary of statistics
        """
        method_counts = defaultdict(int)
        for s in self._selections:
            method_counts[s["method"]] += 1

        return {
            "total_matchups_tracked": len(self._matchups),
            "models_tracked": sum(len(v) for v in self._model_versions.values()),
            "total_selections": len(self._selections),
            "selection_methods": dict(method_counts),
            "config_versions": {k: len(v) for k, v in self._model_versions.items()},
        }

    def get_matchup_win_rates(self, current_model: str) -> dict[str, float]:
        """Get win rates against all opponents for a model.

        Args:
            current_model: Model ID to query

        Returns:
            Dict mapping opponent ID to win rate
        """
        result = {}
        for (curr, opp), record in self._matchups.items():
            if curr == current_model and record.total_games >= self.config.min_games_for_estimate:
                result[opp] = record.win_rate
        return result


# =============================================================================
# Singleton & Integration
# =============================================================================

_selector_instance: PFSPOpponentSelector | None = None


def get_pfsp_selector(config: PFSPConfig | None = None) -> PFSPOpponentSelector:
    """Get the singleton PFSP selector instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        The singleton selector instance
    """
    global _selector_instance

    if _selector_instance is None:
        _selector_instance = PFSPOpponentSelector(config=config)
    return _selector_instance


def reset_pfsp_selector() -> None:
    """Reset the singleton PFSP selector (for testing)."""
    global _selector_instance
    _selector_instance = None


# =============================================================================
# Event Integration
# =============================================================================


def wire_pfsp_events() -> bool:
    """Subscribe PFSP selector to relevant events.

    Subscribes to:
    - MODEL_PROMOTED: Register new model versions
    - EVALUATION_COMPLETED: Update ELO ratings

    Returns:
        True if successfully subscribed
    """
    try:
        from app.coordination.event_router import (
            DataEventType,
            get_event_bus,
        )

        bus = get_event_bus()
        if bus is None:
            logger.warning("[PFSP] Event bus not available")
            return False

        selector = get_pfsp_selector()

        def on_model_promoted(event):
            """Handle MODEL_PROMOTED event."""
            payload = event.payload if hasattr(event, "payload") else {}
            model_id = payload.get("model_id", "")
            config_key = payload.get("config", "")
            elo = payload.get("elo", 1500.0)

            if model_id and config_key:
                selector.register_model(model_id, config_key, elo)
                logger.debug(f"[PFSP] Registered promoted model {model_id} (ELO: {elo:.0f})")

        def on_evaluation_completed(event):
            """Handle EVALUATION_COMPLETED event to update ELO."""
            payload = event.payload if hasattr(event, "payload") else {}
            model_id = payload.get("model_id", "")
            elo = payload.get("elo", 0)

            if model_id and elo > 0:
                selector._model_elos[model_id] = elo
                logger.debug(f"[PFSP] Updated ELO for {model_id}: {elo:.0f}")

        bus.subscribe(DataEventType.MODEL_PROMOTED, on_model_promoted)
        bus.subscribe(DataEventType.EVALUATION_COMPLETED, on_evaluation_completed)

        logger.info("[PFSP] Subscribed to MODEL_PROMOTED and EVALUATION_COMPLETED events")
        return True

    except Exception as e:
        logger.warning(f"[PFSP] Failed to wire events: {e}")
        return False


def bootstrap_pfsp_opponents() -> int:
    """Bootstrap PFSP with existing canonical models.

    This solves the cold-start problem where no opponents are registered
    until MODEL_PROMOTED events occur. By discovering existing canonical
    models, selfplay can immediately use PFSP opponent selection.

    Returns:
        Number of models registered
    """
    try:
        from pathlib import Path

        selector = get_pfsp_selector()
        models_dir = Path("models")
        count = 0

        if not models_dir.exists():
            logger.debug("[PFSP] No models directory found")
            return 0

        # Discover canonical models by naming convention
        for model_path in models_dir.glob("canonical_*.pth"):
            # Parse: canonical_{board}_{n}p.pth
            name = model_path.stem  # e.g., "canonical_hex8_2p"
            parts = name.split("_")
            if len(parts) >= 3:
                board_type = parts[1]  # hex8, square8, etc.
                players_str = parts[2]  # 2p, 3p, 4p
                if players_str.endswith("p"):
                    num_players = players_str[:-1]
                    config_key = f"{board_type}_{num_players}p"
                    model_id = name

                    # Register with default ELO
                    selector.register_model(model_id, config_key, elo=1500.0)
                    count += 1
                    logger.debug(f"[PFSP] Bootstrapped {model_id} for {config_key}")

        if count > 0:
            logger.info(f"[PFSP] Bootstrapped {count} canonical models as opponents")

        return count

    except Exception as e:
        logger.warning(f"[PFSP] Bootstrap failed: {e}")
        return 0
