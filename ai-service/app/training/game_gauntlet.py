"""Unified Game Gauntlet Module for RingRift AI Evaluation.

Consolidates game-playing logic that was duplicated across:
- scripts/select_best_checkpoint_by_elo.py
- app/training/tier_eval_runner.py
- app/training/background_eval.py

This module provides:
- play_single_game(): Play one game between two AIs
- run_baseline_gauntlet(): Evaluate a model against baseline opponents
- BaselineOpponent: Enum of standard baseline types

Usage:
    from app.training.game_gauntlet import run_baseline_gauntlet, BaselineOpponent

    results = run_baseline_gauntlet(
        model_path="models/my_model.pth",
        board_type=BoardType.SQUARE8,
        opponents=[BaselineOpponent.RANDOM, BaselineOpponent.HEURISTIC],
        games_per_opponent=20,
    )
    print(f"Win rate vs random: {results['random']['win_rate']:.1%}")
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies and heavy imports at module load
# Note: TYPE_CHECKING imports removed - using Any for lazy-loaded modules
_torch_loaded = False
_game_modules_loaded = False

# Declare globals for lazy-loaded modules (actual imports happen in _ensure_game_modules)
BoardType: Any = None
AIType: Any = None
AIConfig: Any = None
GameStatus: Any = None
HeuristicAI: Any = None
RandomAI: Any = None
PolicyOnlyAI: Any = None
UniversalAI: Any = None
create_initial_state: Any = None
DefaultRulesEngine: Any = None


def _ensure_game_modules():
    """Lazy load game-related modules."""
    global _game_modules_loaded
    if _game_modules_loaded:
        return

    global BoardType, AIType, AIConfig, GameStatus
    global HeuristicAI, RandomAI, PolicyOnlyAI, UniversalAI
    global create_initial_state, DefaultRulesEngine

    from app.ai.heuristic_ai import HeuristicAI
    from app.ai.policy_only_ai import PolicyOnlyAI
    from app.ai.random_ai import RandomAI
    from app.ai.universal_ai import UniversalAI
    from app.models import AIConfig, AIType, BoardType, GameStatus
    from app.rules.default_engine import DefaultRulesEngine
    from app.training.generate_data import create_initial_state

    _game_modules_loaded = True


# ============================================
# Adaptive Resource Management (Dec 2025)
# ============================================
# Prevents resource exhaustion by scaling workers based on system load


def get_adaptive_max_workers(requested: int = 4) -> int:
    """Reduce workers if system is under load.

    This helps prevent resource exhaustion when multiple gauntlets run
    concurrently or when the system is already under heavy load.

    Args:
        requested: Requested number of workers (default: 4)

    Returns:
        Adjusted worker count based on system load
    """
    try:
        import os
        load_avg = os.getloadavg()[0]
        cpu_count = os.cpu_count() or 1
        load_ratio = load_avg / cpu_count

        if load_ratio > 0.8:
            # Heavy load: use minimal workers
            return max(1, requested // 4)
        elif load_ratio > 0.5:
            # Moderate load: use half workers
            return max(2, requested // 2)
        else:
            # Normal load: use requested workers
            return requested
    except Exception:
        # If we can't check load, use conservative default
        return min(requested, 2)


class BaselineOpponent(Enum):
    """Standard baseline opponents for evaluation."""
    RANDOM = "random"
    HEURISTIC = "heuristic"


# Import baseline Elo estimates from centralized config
try:
    from app.config.thresholds import (
        BASELINE_ELO_HEURISTIC,
        BASELINE_ELO_RANDOM,
        MIN_WIN_RATE_VS_HEURISTIC,
        MIN_WIN_RATE_VS_RANDOM,
        get_min_win_rate_vs_heuristic,
        get_min_win_rate_vs_random,
    )
except ImportError:
    # Fallback values - keep in sync with app/config/thresholds.py
    BASELINE_ELO_RANDOM = 400
    BASELINE_ELO_HEURISTIC = 1200
    MIN_WIN_RATE_VS_RANDOM = 0.70  # 70% (matches thresholds.py)
    MIN_WIN_RATE_VS_HEURISTIC = 0.50  # 50% (matches thresholds.py)

    def get_min_win_rate_vs_random(num_players: int = 2) -> float:
        return 0.50 if num_players >= 4 else MIN_WIN_RATE_VS_RANDOM

    def get_min_win_rate_vs_heuristic(num_players: int = 2) -> float:
        return 0.20 if num_players >= 4 else MIN_WIN_RATE_VS_HEURISTIC

BASELINE_ELOS = {
    BaselineOpponent.RANDOM: BASELINE_ELO_RANDOM,
    BaselineOpponent.HEURISTIC: BASELINE_ELO_HEURISTIC,
}

# Static fallback (use get_min_win_rate_* functions for player-aware thresholds)
MIN_WIN_RATES = {
    BaselineOpponent.RANDOM: MIN_WIN_RATE_VS_RANDOM,
    BaselineOpponent.HEURISTIC: MIN_WIN_RATE_VS_HEURISTIC,
}


@dataclass
class GameResult:
    """Result of a single game."""
    winner: int | None  # Player number who won, or None for draw
    move_count: int
    victory_reason: str
    candidate_player: int  # Which player was the candidate
    candidate_won: bool


@dataclass
class GauntletResult:
    """Aggregated results from a gauntlet evaluation."""
    total_games: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_draws: int = 0
    win_rate: float = 0.0

    # Per-opponent results
    opponent_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Baseline gating
    passes_baseline_gating: bool = True
    failed_baselines: list[str] = field(default_factory=list)

    # Elo estimate
    estimated_elo: float = 1500.0


def create_baseline_ai(
    baseline: BaselineOpponent,
    player: int,
    board_type: Any,  # BoardType
    difficulty: int | None = None,
    game_seed: int | None = None,
) -> Any:
    """Create an AI instance for a baseline opponent.

    Args:
        baseline: Which baseline to create
        player: Player number (1 or 2)
        board_type: Board type enum
        difficulty: Optional difficulty override
        game_seed: Optional seed for RNG variation per game

    Returns:
        AI instance ready to play
    """
    _ensure_game_modules()

    # Derive player-specific seed for varied but reproducible behavior
    ai_rng_seed = None
    if game_seed is not None:
        ai_rng_seed = (game_seed * 104729 + player * 7919) & 0xFFFFFFFF

    if baseline == BaselineOpponent.RANDOM:
        config = AIConfig(
            ai_type=AIType.RANDOM,
            board_type=board_type,
            difficulty=difficulty or 1,
            rngSeed=ai_rng_seed,
        )
        return RandomAI(player, config)

    elif baseline == BaselineOpponent.HEURISTIC:
        config = AIConfig(
            ai_type=AIType.HEURISTIC,
            board_type=board_type,
            difficulty=difficulty or 5,
            rngSeed=ai_rng_seed,
        )
        return HeuristicAI(player, config)

    else:
        raise ValueError(f"Unknown baseline: {baseline}")


def create_neural_ai(
    player: int,
    board_type: Any,  # BoardType
    model_path: str | Path | None = None,
    model_getter: Callable[[], Any] | None = None,
    temperature: float = 0.5,
    game_seed: int | None = None,
    num_players: int = 2,
    model_type: str = "cnn",
) -> Any:
    """Create a neural network AI instance.

    Args:
        player: Player number
        board_type: Board type enum
        model_path: Path to model checkpoint (for file-based loading)
        model_getter: Callable that returns model weights (for in-memory loading)
        num_players: Number of players in the game (2, 3, or 4)
        temperature: Policy temperature for move selection
        game_seed: Optional seed for RNG variation per game
        model_type: Type of model - "cnn" (default), "gnn", or "hybrid"

    Returns:
        AI instance (PolicyOnlyAI for CNN, GNNAI for GNN/hybrid)
    """
    _ensure_game_modules()

    # GNN models use dedicated GNNAI class
    if model_type in ("gnn", "hybrid"):
        from app.ai.gnn_ai import create_gnn_ai

        # Derive device - prefer GPU if available
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        return create_gnn_ai(
            player_number=player,
            model_path=model_path,
            device=device,
            temperature=temperature,
        )

    # Derive player-specific seed for varied but reproducible behavior
    ai_rng_seed = None
    if game_seed is not None:
        ai_rng_seed = (game_seed * 104729 + player * 7919) & 0xFFFFFFFF

    if model_path is not None:
        from pathlib import Path
        path_obj = Path(model_path)

        # Try UniversalAI.from_checkpoint first for proper architecture inference
        # This handles checkpoints with different architectures (num_filters, num_res_blocks)
        try:
            # Resolve to absolute path if needed
            if not path_obj.is_absolute():
                # Check common locations
                for prefix in [Path("."), Path("models"), Path("models/checkpoints")]:
                    candidate = prefix / model_path
                    if candidate.exists():
                        path_obj = candidate
                        break
                # Also check with .pth extension
                if not path_obj.exists() and not str(path_obj).endswith(".pth"):
                    for prefix in [Path("."), Path("models"), Path("models/checkpoints")]:
                        candidate = prefix / f"{model_path}.pth"
                        if candidate.exists():
                            path_obj = candidate
                            break

            if path_obj.exists():
                return UniversalAI.from_checkpoint(
                    str(path_obj),
                    player_number=player,
                    board_type=board_type,
                    num_players=num_players,
                    policy_temperature=temperature,
                )
        except Exception as e:
            logger.warning(f"UniversalAI.from_checkpoint failed: {e}, falling back to PolicyOnlyAI")

        # Fallback to legacy loading via PolicyOnlyAI
        model_id = str(model_path)
        if not path_obj.is_absolute():
            if model_id.startswith("models/"):
                model_id = model_id[7:]
            if model_id.endswith(".pth"):
                model_id = model_id[:-4]

        config = AIConfig(
            ai_type=AIType.POLICY_ONLY,
            board_type=board_type,
            difficulty=8,
            use_neural_net=True,
            nn_model_id=model_id,
            policy_temperature=temperature,
            rngSeed=ai_rng_seed,
        )
        return PolicyOnlyAI(player, config, board_type=board_type)

    elif model_getter is not None:
        # In-memory model loading for BackgroundEvaluator (zero disk I/O)
        model_info = model_getter()

        # Extract state_dict from model_info
        if isinstance(model_info, dict):
            if 'state_dict' in model_info:
                state_dict = model_info['state_dict']
            elif 'model_state_dict' in model_info:
                # Versioned checkpoint format
                state_dict = model_info['model_state_dict']
            else:
                # Assume the dict is the state_dict itself
                state_dict = model_info
        elif hasattr(model_info, 'state_dict'):
            # It's an nn.Module - extract state_dict
            state_dict = model_info.state_dict()
        else:
            raise ValueError(
                f"model_getter must return a dict with 'state_dict', a state_dict, "
                f"or an nn.Module. Got: {type(model_info)}"
            )

        config = AIConfig(
            ai_type=AIType.POLICY_ONLY,
            board_type=board_type,
            difficulty=8,
            use_neural_net=True,
            nn_state_dict=state_dict,
            policy_temperature=temperature,
            rngSeed=ai_rng_seed,
        )
        return PolicyOnlyAI(player, config, board_type=board_type)

    else:
        raise ValueError("Must provide either model_path or model_getter")


def play_single_game(
    candidate_ai: Any,
    opponent_ai: Any,
    board_type: Any,  # BoardType
    num_players: int = 2,
    candidate_player: int = 1,
    max_moves: int = 500,
    seed: int | None = None,
    opponent_ais: dict[int, Any] | None = None,
) -> GameResult:
    """Play a single game between candidate and opponent(s).

    Args:
        candidate_ai: The AI being evaluated
        opponent_ai: The baseline/opponent AI (for 2-player games or fallback)
        board_type: Board type for the game
        num_players: Number of players
        candidate_player: Which player number is the candidate
        max_moves: Maximum moves before draw
        seed: Optional random seed
        opponent_ais: Dict mapping player numbers to AI instances for multiplayer.
                      If not provided, all non-candidate players use opponent_ai.

    Returns:
        GameResult with game outcome
    """
    _ensure_game_modules()

    engine = DefaultRulesEngine()
    state = create_initial_state(board_type, num_players)

    # Build player->AI mapping for multiplayer support
    player_ais: dict[int, Any] = {candidate_player: candidate_ai}
    if opponent_ais is not None:
        # Use provided AIs for each opponent player
        player_ais.update(opponent_ais)
    else:
        # For 2-player games, use the single opponent_ai for the other player
        for p in range(1, num_players + 1):
            if p != candidate_player:
                player_ais[p] = opponent_ai

    move_count = 0
    while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_player = state.current_player

        # Get the AI for the current player
        ai = player_ais.get(current_player)
        if ai is None:
            logger.error(f"No AI assigned for player {current_player}")
            break

        move = ai.select_move(state)

        if move:
            state = engine.apply_move(state, move)
        else:
            # No valid move available - this shouldn't happen in normal games
            logger.warning(f"Player {current_player} returned no move at turn {move_count}")
            break
        move_count += 1

    # Determine outcome
    victory_reason = getattr(state, 'victory_reason', 'unknown')
    if hasattr(victory_reason, 'value'):
        victory_reason = victory_reason.value

    winner = state.winner
    candidate_won = winner == candidate_player if winner is not None else False

    return GameResult(
        winner=winner,
        move_count=move_count,
        victory_reason=str(victory_reason) if victory_reason else "max_moves",
        candidate_player=candidate_player,
        candidate_won=candidate_won,
    )


def _emit_gauntlet_result_event(
    config_key: str,
    elo: float,
    win_rate: float,
    games: int,
) -> None:
    """Emit EVALUATION_COMPLETED event to close eval→curriculum feedback loop.

    This function emits an event that curriculum_feedback.py's
    TournamentToCurriculumWatcher can consume to adjust training weights.

    Args:
        config_key: Configuration key (e.g., "square8_2p")
        elo: Estimated Elo rating
        win_rate: Overall win rate (0.0-1.0)
        games: Total games played
    """
    import asyncio

    try:
        from app.distributed.event_helpers import emit_evaluation_completed_safe

        # Try to emit in async context, otherwise schedule
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(
                emit_evaluation_completed_safe(
                    config=config_key,
                    elo=elo,
                    games=games,
                    win_rate=win_rate,
                    source="game_gauntlet",
                )
            )
            logger.debug(
                f"[gauntlet] Emitted EVALUATION_COMPLETED for {config_key}: "
                f"elo={elo:.0f}, win_rate={win_rate:.1%}"
            )
        except RuntimeError:
            # No running loop - try to run synchronously
            try:
                asyncio.run(
                    emit_evaluation_completed_safe(
                        config=config_key,
                        elo=elo,
                        games=games,
                        win_rate=win_rate,
                        source="game_gauntlet",
                    )
                )
            except Exception as e:
                logger.debug(f"[gauntlet] Could not emit event (no async context): {e}")
    except ImportError:
        logger.debug("[gauntlet] Event helpers not available, skipping event emission")
    except Exception as e:
        logger.warning(f"[gauntlet] Failed to emit EVALUATION_COMPLETED: {e}")


def run_baseline_gauntlet(
    model_path: str | Path | None = None,
    board_type: Any = None,  # BoardType
    opponents: list[BaselineOpponent] | None = None,
    games_per_opponent: int = 20,
    num_players: int = 2,
    check_baseline_gating: bool = True,
    verbose: bool = False,
    model_getter: Callable[[], Any] | None = None,
    model_type: str = "cnn",
) -> GauntletResult:
    """Run a gauntlet evaluation against baseline opponents.

    Args:
        model_path: Path to the model checkpoint (file-based loading)
        board_type: Board type for games
        opponents: List of baselines to test against (default: RANDOM, HEURISTIC)
        games_per_opponent: Number of games per opponent
        num_players: Number of players in each game
        check_baseline_gating: Whether to check minimum win rate thresholds
        verbose: Whether to log per-game results
        model_getter: Callable returning model weights (in-memory loading, zero disk I/O)
        model_type: Type of model - "cnn" (default), "gnn", or "hybrid"

    Returns:
        GauntletResult with aggregated statistics
    """
    if model_path is None and model_getter is None:
        raise ValueError("Must provide either model_path or model_getter")
    _ensure_game_modules()

    # Log encoder expectations for debugging channel mismatches
    if board_type is not None:
        try:
            from app.training.encoder_registry import get_encoder_config
            for version in ["v2", "v3"]:
                config = get_encoder_config(board_type, version)
                logger.debug(
                    f"[gauntlet] {board_type.name} {version}: expects {config.in_channels} channels "
                    f"({config.base_channels} base × {config.frames} frames)"
                )
        except Exception:
            pass  # Registry not available, continue without

    if opponents is None:
        opponents = [BaselineOpponent.RANDOM, BaselineOpponent.HEURISTIC]

    result = GauntletResult()

    for baseline in opponents:
        baseline_name = baseline.value
        opponent_stats = {"wins": 0, "games": 0, "win_rate": 0.0}

        for game_num in range(games_per_opponent):
            # Rotate which player the candidate plays as
            candidate_player = (game_num % num_players) + 1

            # Derive unique seed per game for varied behavior
            game_seed = random.randint(0, 0xFFFFFFFF)

            try:
                candidate_ai = create_neural_ai(
                    candidate_player, board_type,
                    model_path=model_path,
                    model_getter=model_getter,
                    game_seed=game_seed,
                    num_players=num_players,
                    model_type=model_type,
                )

                # Create baseline AIs for all other players
                opponent_ais: dict[int, Any] = {}
                for p in range(1, num_players + 1):
                    if p != candidate_player:
                        opponent_ais[p] = create_baseline_ai(
                            baseline, p, board_type,
                            game_seed=game_seed,
                        )

                # For backwards compatibility, also pass opponent_ai (player after candidate)
                first_opponent = (candidate_player % num_players) + 1
                opponent_ai = opponent_ais.get(first_opponent, list(opponent_ais.values())[0])

                game_result = play_single_game(
                    candidate_ai=candidate_ai,
                    opponent_ai=opponent_ai,
                    board_type=board_type,
                    num_players=num_players,
                    candidate_player=candidate_player,
                    opponent_ais=opponent_ais,
                )

                result.total_games += 1
                opponent_stats["games"] += 1

                if game_result.candidate_won:
                    result.total_wins += 1
                    opponent_stats["wins"] += 1
                elif game_result.winner is not None:
                    result.total_losses += 1
                else:
                    result.total_draws += 1

                if verbose:
                    outcome = "WIN" if game_result.candidate_won else "LOSS"
                    logger.info(
                        f"[gauntlet] Game {game_num+1}/{games_per_opponent} vs {baseline_name}: "
                        f"{outcome} ({game_result.victory_reason}, {game_result.move_count} moves)"
                    )

            except Exception as e:
                logger.error(f"Error in game {game_num} vs {baseline_name}: {e}")
                continue

        # Calculate win rate for this opponent
        if opponent_stats["games"] > 0:
            opponent_stats["win_rate"] = opponent_stats["wins"] / opponent_stats["games"]

        result.opponent_results[baseline_name] = opponent_stats

        # Check baseline gating (use player-aware thresholds)
        if check_baseline_gating:
            if baseline == BaselineOpponent.RANDOM:
                min_required = get_min_win_rate_vs_random(num_players)
            elif baseline == BaselineOpponent.HEURISTIC:
                min_required = get_min_win_rate_vs_heuristic(num_players)
            else:
                min_required = MIN_WIN_RATES.get(baseline, 0.0)

            if opponent_stats["win_rate"] < min_required:
                result.passes_baseline_gating = False
                result.failed_baselines.append(baseline_name)
                logger.warning(
                    f"[gauntlet] Failed baseline gating vs {baseline_name}: "
                    f"{opponent_stats['win_rate']:.1%} < {min_required:.0%} required"
                    f" ({num_players}p thresholds)"
                )

    # Calculate overall win rate
    if result.total_games > 0:
        result.win_rate = result.total_wins / result.total_games

    # Estimate Elo from win rates
    result.estimated_elo = _estimate_elo_from_results(result.opponent_results)

    # Emit EVALUATION_COMPLETED event for curriculum feedback (December 2025)
    # This closes the eval→curriculum feedback loop
    _emit_gauntlet_result_event(
        config_key=f"{board_type.value}_{num_players}p",
        elo=result.estimated_elo,
        win_rate=result.win_rate,
        games=result.total_games,
    )

    return result


def _estimate_elo_from_results(
    opponent_results: dict[str, dict[str, Any]]
) -> float:
    """Estimate Elo rating from gauntlet results.

    Uses weighted average of Elo estimates from each opponent.
    """
    import math

    total_elo = 0.0
    total_weight = 0.0

    for baseline_name, stats in opponent_results.items():
        baseline = BaselineOpponent(baseline_name)
        opponent_elo = BASELINE_ELOS.get(baseline, 1000)
        win_rate = stats.get("win_rate", 0.5)
        games = stats.get("games", 0)

        if games == 0:
            continue

        # Elo formula: E = 1 / (1 + 10^((Rb-Ra)/400))
        # Solving for Ra: Ra = Rb - 400 * log10(1/E - 1)
        if win_rate <= 0:
            estimated = opponent_elo - 400
        elif win_rate >= 1:
            estimated = opponent_elo + 400
        else:
            estimated = opponent_elo - 400 * math.log10(1/win_rate - 1)

        total_elo += estimated * games
        total_weight += games

    if total_weight > 0:
        return total_elo / total_weight
    return 1500.0


# Convenience function for quick evaluation
def quick_evaluate(
    model_path: str | Path,
    games: int = 10,
) -> dict[str, float]:
    """Quick evaluation against baselines.

    Args:
        model_path: Path to model checkpoint
        games: Games per opponent

    Returns:
        Dict with win rates per opponent
    """
    _ensure_game_modules()

    result = run_baseline_gauntlet(
        model_path=model_path,
        board_type=BoardType.SQUARE8,
        games_per_opponent=games,
        verbose=True,
    )

    return {
        name: stats["win_rate"]
        for name, stats in result.opponent_results.items()
    }
