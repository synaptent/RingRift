#!/usr/bin/env python
"""AI Model Evaluation Framework for RingRift.

This script provides comprehensive evaluation of AI models against each other
with detailed metrics collection and statistical analysis.

Supported AI Types:
- baseline_heuristic: HeuristicAI with BASE_V1_BALANCED_WEIGHTS
- cmaes_heuristic: HeuristicAI with CMA-ES optimized weights
- neural_network: DescentAI with trained neural network model
- random: RandomAI as control baseline
- minimax: MinimaxAI with configurable depth
- policy_only: PolicyOnlyAI - uses neural network policy head without search
- gumbel_mcts: GumbelMCTSAI - Gumbel MCTS with neural network guidance

Usage Examples:
    # Baseline vs Random (50 games)
    python scripts/evaluate_ai_models.py \\
        --player1 baseline_heuristic \\
        --player2 random \\
        --games 50 \\
        --board square8 \\
        --seed 42 \\
        --output results/baseline_vs_random.json

    # CMA-ES vs Baseline comparison
    python scripts/evaluate_ai_models.py \\
        --player1 cmaes_heuristic \\
        --player2 baseline_heuristic \\
        --games 100 \\
        --output results/cmaes_vs_baseline.json

    # Neural network vs Minimax
    python scripts/evaluate_ai_models.py \\
        --player1 neural_network \\
        --player2 minimax \\
        --games 50 \\
        --checkpoint checkpoints/checkpoint_final_epoch_5.pth

    # Neural network vs Neural network (dual-checkpoint comparison)
    python scripts/evaluate_ai_models.py \\
        --player1 neural_network \\
        --player2 neural_network \\
        --games 100 \\
        --checkpoint models/ringrift_from_mcts.pth \\
        --checkpoint2 models/ringrift_from_descent.pth
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np

# Allow imports from app/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fallback for tqdm if not installed
try:
    from tqdm import tqdm as tqdm_lib

    tqdm: Callable[..., Any] = tqdm_lib
except ImportError:

    def tqdm_fallback(iterable: Any, **kwargs: Any) -> Generator[Any]:
        """Simple fallback for tqdm."""
        total = kwargs.get("total")
        if total is None and hasattr(iterable, "__len__"):
            total = len(iterable)
        desc = kwargs.get("desc", "")
        for i, item in enumerate(iterable):
            if total:
                print(f"\r{desc}: {i+1}/{total}", end="", flush=True)
            yield item
        print()

    tqdm = tqdm_fallback

from app.ai.base import BaseAI
from app.ai.descent_ai import DescentAI
from app.ai.gumbel_mcts_ai import GumbelMCTSAI
from app.ai.heuristic_ai import HeuristicAI
from app.ai.heuristic_weights import (
    BASE_V1_BALANCED_WEIGHTS,
    HEURISTIC_WEIGHT_PROFILES,
)
from app.ai.minimax_ai import MinimaxAI
from app.ai.policy_only_ai import PolicyOnlyAI
from app.ai.random_ai import RandomAI
from app.models import (
    AIConfig,
    BoardType,
    GameState,
    GameStatus,
)
from app.training.env import (
    TrainingEnvConfig,
    make_env,
)
from app.utils.progress_reporter import ProgressReporter
from scripts.lib.cli import BOARD_TYPE_MAP

# AI Type Constants
AI_TYPE_BASELINE_HEURISTIC = "baseline_heuristic"
AI_TYPE_CMAES_HEURISTIC = "cmaes_heuristic"
AI_TYPE_NEURAL_NETWORK = "neural_network"
AI_TYPE_RANDOM = "random"
AI_TYPE_MINIMAX = "minimax"
AI_TYPE_POLICY_ONLY = "policy_only"
AI_TYPE_GUMBEL_MCTS = "gumbel_mcts"

SUPPORTED_AI_TYPES = [
    AI_TYPE_BASELINE_HEURISTIC,
    AI_TYPE_CMAES_HEURISTIC,
    AI_TYPE_NEURAL_NETWORK,
    AI_TYPE_RANDOM,
    AI_TYPE_MINIMAX,
    AI_TYPE_POLICY_ONLY,
    AI_TYPE_GUMBEL_MCTS,
]


def _format_ai_label(ai_type: str, checkpoint: str | None) -> str:
    if not checkpoint:
        return ai_type
    base = os.path.basename(checkpoint)
    if base.endswith(".pth"):
        base = base[: -len(".pth")]
    return f"{ai_type}@{base}"


@dataclass
class GameResult:
    """Result of a single game."""

    winner: int | None  # 1, 2, or None for draw
    length: int  # Number of moves
    victory_type: str | None  # elimination, resignation, etc
    p1_decision_times: list[float] = field(default_factory=list)
    p2_decision_times: list[float] = field(default_factory=list)
    p1_final_pieces: int = 0  # Pieces remaining at game end
    p2_final_pieces: int = 0
    p1_was_player: str = ""  # Which AI type was player 1
    p2_was_player: str = ""  # Which AI type was player 2
    error: str | None = None


@dataclass
class EvaluationResults:
    """Aggregated results from evaluation."""

    config: dict[str, Any]
    player1_wins: int = 0
    player2_wins: int = 0
    draws: int = 0
    games: list[dict[str, Any]] = field(default_factory=list)
    total_runtime_seconds: float = 0.0

    # Per-game statistics
    game_lengths: list[int] = field(default_factory=list)
    p1_decision_times: list[float] = field(default_factory=list)
    p2_decision_times: list[float] = field(default_factory=list)

    # Victory type breakdown
    victory_types: dict[str, int] = field(default_factory=dict)

    # Piece advantage tracking
    p1_final_pieces_list: list[int] = field(default_factory=list)
    p2_final_pieces_list: list[int] = field(default_factory=list)


def wilson_score_interval(wins: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate Wilson score confidence interval for win rate.

    The Wilson score interval is more accurate than the normal approximation,
    especially for extreme probabilities and small sample sizes.

    Args:
        wins: Number of wins
        total: Total number of games
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound) for the win rate
    """
    if total == 0:
        return (0.0, 0.0)

    # Z-score for confidence intervals
    if confidence == 0.95:
        z = 1.96
    elif confidence == 0.90:
        z = 1.645
    else:
        z = 2.576

    p = wins / total
    n = total

    # Wilson score interval formula
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    spread = (z / denominator) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)

    return (round(lower, 4), round(upper, 4))


def load_cmaes_weights(path: str = "heuristic_weights_optimized.json") -> dict[str, float]:
    """Load CMA-ES optimized weights from JSON file.

    Args:
        path: Path to the optimized weights JSON file

    Returns:
        Dictionary of weight name to value
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    paths_to_try = [
        path,
        os.path.join(base_dir, path),
        os.path.join(base_dir, "heuristic_weights_optimized.json"),
    ]

    for p in paths_to_try:
        if os.path.exists(p):
            with open(p) as f:
                data = json.load(f)
                return data.get("weights", data)

    raise FileNotFoundError(f"Could not find CMA-ES weights file. Tried: {paths_to_try}")


def create_ai(
    ai_type: str,
    player_num: int,
    board_type: BoardType,
    checkpoint: str | None = None,
    mm_depth: int = 3,
    cmaes_path: str | None = None,
    game_seed: int | None = None,
) -> BaseAI:
    """Create an AI instance based on the specified type.

    Args:
        ai_type: Type of AI to create
        player_num: Player number (1 or 2)
        checkpoint: Path to neural network checkpoint
        mm_depth: Search depth for minimax AI
        cmaes_path: Path to CMA-ES optimized weights file
        game_seed: Unique seed for this game instance to ensure stochastic AIs
                   (like RandomAI) behave differently in each game.

    Returns:
        BaseAI instance
    """
    # Derive a unique per-AI seed for this game.
    # Combine game_seed with player_num so P1 and P2 use
    # different RNG streams.
    ai_rng_seed: int | None = None
    if game_seed is not None:
        # Use a hash-like combination to get a unique seed
        # per (game, player)
        ai_rng_seed = (game_seed * 104729 + player_num * 7919) & 0xFFFFFFFF

    # Base config with no thinking delay for faster evaluation
    base_config = AIConfig(
        difficulty=5,
        think_time=0,  # No artificial delay
        randomness=0.0,  # Deterministic play for evaluation
        rngSeed=ai_rng_seed,
        heuristic_profile_id=None,
    )

    if ai_type == AI_TYPE_RANDOM:
        return RandomAI(player_num, base_config)

    if ai_type == AI_TYPE_BASELINE_HEURISTIC:
        # Register baseline profile if not already registered
        if "baseline_v1_balanced" not in HEURISTIC_WEIGHT_PROFILES:
            profile = BASE_V1_BALANCED_WEIGHTS
            HEURISTIC_WEIGHT_PROFILES["baseline_v1_balanced"] = profile

        config = AIConfig(
            difficulty=5,
            think_time=0,
            randomness=0.0,
            rngSeed=ai_rng_seed,
            heuristic_profile_id="baseline_v1_balanced",
        )
        return HeuristicAI(player_num, config)

    if ai_type == AI_TYPE_CMAES_HEURISTIC:
        # Load and register CMA-ES optimized weights
        weights_path = cmaes_path or "heuristic_weights_optimized.json"
        cmaes_weights = load_cmaes_weights(weights_path)

        profile_id = "cmaes_optimized"
        HEURISTIC_WEIGHT_PROFILES[profile_id] = cmaes_weights

        config = AIConfig(
            difficulty=5,
            think_time=0,
            randomness=0.0,
            rngSeed=ai_rng_seed,
            heuristic_profile_id=profile_id,
        )
        return HeuristicAI(player_num, config)

    if ai_type == AI_TYPE_MINIMAX:
        # Configure minimax with specified depth
        # Difficulty maps to depth: difficulty 4-6 -> depth 3
        difficulty = max(4, min(9, mm_depth + 1))
        config = AIConfig(
            difficulty=difficulty,
            think_time=5000,  # 5 second time limit for search
            randomness=0.0,
            rngSeed=ai_rng_seed,
            heuristic_profile_id=None,
        )
        return MinimaxAI(player_num, config)

    if ai_type == AI_TYPE_NEURAL_NETWORK:
        # Find checkpoint if not specified
        ckpt = checkpoint
        if not ckpt:
            checkpoints_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
            if os.path.exists(checkpoints_dir):
                # Use the most recent final checkpoint
                checkpoints = sorted([f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")])
                if checkpoints:
                    # Prefer final checkpoints
                    final_cps = [c for c in checkpoints if "final" in c]
                    if final_cps:
                        ckpt = os.path.join(checkpoints_dir, final_cps[-1])
                    else:
                        ckpt = os.path.join(checkpoints_dir, checkpoints[-1])

        config = AIConfig(
            difficulty=5,
            think_time=0,
            randomness=0.0,
            rngSeed=ai_rng_seed,
            heuristic_profile_id=None,
            nn_model_id=ckpt if ckpt else None,
            allow_fresh_weights=False,
        )

        # Create DescentAI
        ai = DescentAI(player_num, config)

        # Ensure the neural net is initialized (and checkpoint loaded) if available.
        if ckpt and os.path.exists(ckpt):
            if ai.neural_net is None:
                raise RuntimeError(f"NeuralNetAI unavailable; cannot load checkpoint: {ckpt}")
            try:
                ai.neural_net._ensure_model_initialized(board_type)
                print(f"Loaded checkpoint: {ckpt}")
            except RuntimeError as e:
                # Handle shape mismatch (checkpoint for different board type)
                if "size mismatch" in str(e) or "shape" in str(e).lower():
                    print(f"Checkpoint {ckpt} incompatible with {board_type}, using fresh weights")
                    # Reinitialize with fresh weights
                    config.allow_fresh_weights = True
                    ai = DescentAI(player_num, config)
                    ai.neural_net._ensure_model_initialized(board_type)
                else:
                    raise RuntimeError(f"Failed to load checkpoint {ckpt}: {e}") from e

        return ai

    if ai_type == AI_TYPE_POLICY_ONLY:
        # PolicyOnlyAI - uses neural network policy head without search
        ckpt = checkpoint
        if not ckpt:
            # Try to find a checkpoint in models/ or checkpoints/
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
            checkpoints_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
            for search_dir in [models_dir, checkpoints_dir]:
                if os.path.exists(search_dir):
                    checkpoints = sorted([f for f in os.listdir(search_dir) if f.endswith(".pth")])
                    if checkpoints:
                        ckpt = os.path.join(search_dir, checkpoints[-1])
                        break

        config = AIConfig(
            difficulty=5,
            think_time=0,
            randomness=0.0,
            rngSeed=ai_rng_seed,
            nn_model_id=ckpt if ckpt else None,
            allow_fresh_weights=False,
        )
        ai = PolicyOnlyAI(player_num, config)
        if ckpt:
            print(f"PolicyOnlyAI loaded checkpoint: {ckpt}")
        return ai

    if ai_type == AI_TYPE_GUMBEL_MCTS:
        # GumbelMCTSAI - Gumbel MCTS with neural network
        ckpt = checkpoint
        if not ckpt:
            # Try to find a checkpoint in models/ or checkpoints/
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
            checkpoints_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
            for search_dir in [models_dir, checkpoints_dir]:
                if os.path.exists(search_dir):
                    checkpoints = sorted([f for f in os.listdir(search_dir) if f.endswith(".pth")])
                    if checkpoints:
                        ckpt = os.path.join(search_dir, checkpoints[-1])
                        break

        config = AIConfig(
            difficulty=5,
            think_time=2000,  # 2 second think time for MCTS
            randomness=0.0,
            rngSeed=ai_rng_seed,
            nn_model_id=ckpt if ckpt else None,
            allow_fresh_weights=False,
        )
        ai = GumbelMCTSAI(player_num, config)
        if ckpt:
            print(f"GumbelMCTSAI loaded checkpoint: {ckpt}")
        return ai

    raise ValueError(f"Unknown AI type: {ai_type}. Supported: {SUPPORTED_AI_TYPES}")


def count_player_pieces(game_state: GameState, player_num: int) -> int:
    """Count total pieces (rings in play + rings in hand) for a player.

    Args:
        game_state: Current game state
        player_num: Player number to count pieces for

    Returns:
        Total piece count
    """
    player = game_state.players[player_num - 1]
    rings_in_hand = player.rings_in_hand

    # Count rings on the board
    rings_on_board = 0
    if game_state.board and game_state.board.stacks:
        for stack in game_state.board.stacks.values():
            if hasattr(stack, "rings"):
                for ring in stack.rings:
                    if ring == player_num:
                        rings_on_board += 1
            elif isinstance(stack, list):
                for ring in stack:
                    if ring == player_num:
                        rings_on_board += 1

    return rings_in_hand + rings_on_board


def determine_victory_type(game_state: GameState) -> str | None:
    """Determine the type of victory from the game state.

    Args:
        game_state: Final game state

    Returns:
        Victory type string or None
    """
    if game_state.game_status != GameStatus.COMPLETED:
        return None

    if game_state.winner is None:
        return "draw"

    # Check for elimination victory
    for player in game_state.players:
        threshold = game_state.victory_threshold
        if player.eliminated_rings >= threshold:
            return "elimination"

    # Check for territory victory
    for player in game_state.players:
        threshold = game_state.territory_victory_threshold
        if player.territory_spaces >= threshold:
            return "territory"

    # Default to elimination if winner is set but type unclear
    return "elimination"


def _tiebreak_winner(game_state: GameState) -> int | None:
    """Deterministically select a winner for evaluation-only timeouts.

    This does not change canonical rules; it is used only to avoid draw-heavy
    evaluation results when a max-moves budget is hit.
    """
    if not getattr(game_state, "players", None):
        return None

    best_player: int | None = None
    best_key: tuple | None = None

    for idx, player in enumerate(game_state.players):
        player_num = getattr(player, "player_number", None)
        if player_num is None:
            player_num = idx + 1
        try:
            eliminated = int(getattr(player, "eliminated_rings", 0) or 0)
        except Exception:
            eliminated = 0
        try:
            territory = int(getattr(player, "territory_spaces", 0) or 0)
        except Exception:
            territory = 0
        pieces = count_player_pieces(game_state, int(player_num))

        key = (eliminated, territory, pieces, -int(player_num))
        if best_key is None or key > best_key:
            best_key = key
            best_player = int(player_num)

    return best_player


def play_single_game(
    ai_p1: BaseAI,
    ai_p2: BaseAI,
    env: Any,
    max_moves: int = 10000,
    p1_type: str = "",
    p2_type: str = "",
    verbose: bool = False,
) -> GameResult:
    """Play a single game between two AIs.

    Args:
        ai_p1: AI instance for player 1
        ai_p2: AI instance for player 2
        env: Game environment
        max_moves: Maximum moves before declaring draw
        p1_type: Type name for player 1 AI
        p2_type: Type name for player 2 AI
        verbose: Whether to print move-by-move output

    Returns:
        GameResult with game outcome and statistics
    """
    result = GameResult(
        winner=None,
        length=0,
        victory_type=None,
        p1_was_player=p1_type,
        p2_was_player=p2_type,
    )

    try:
        game_state = env.reset()
        move_count = 0

        while game_state.game_status == GameStatus.ACTIVE and move_count < max_moves:
            current_player = game_state.current_player
            current_ai = ai_p1 if current_player == 1 else ai_p2

            # Ensure AI has correct player number
            current_ai.player_number = current_player

            # Time the decision
            start_time = time.perf_counter()
            move = current_ai.select_move(game_state)
            decision_time = time.perf_counter() - start_time

            if current_player == 1:
                result.p1_decision_times.append(decision_time)
            else:
                result.p2_decision_times.append(decision_time)

            if move is None:
                # No valid move - current player loses
                game_state.game_status = GameStatus.COMPLETED
                game_state.winner = 2 if current_player == 1 else 1
                break

            # Apply the move
            game_state, _reward, done, _info = env.step(move)
            move_count += 1

            if verbose:
                print(f"Move {move_count}: P{current_player} -> {move.type}")

            if done:
                break

        # Record final state
        result.length = move_count
        result.winner = game_state.winner
        result.victory_type = determine_victory_type(game_state)

        # Record final piece counts
        result.p1_final_pieces = count_player_pieces(game_state, 1)
        result.p2_final_pieces = count_player_pieces(game_state, 2)

        # Handle timeout/draw (evaluation-only deterministic tie-break)
        if move_count >= max_moves and game_state.winner is None:
            winner = _tiebreak_winner(game_state)
            if winner is not None:
                result.winner = winner
                result.victory_type = "timeout_tiebreak"
            else:
                result.victory_type = "timeout"

    except Exception as e:
        result.error = str(e)
        result.victory_type = "error"

    return result


def run_evaluation(
    player1_type: str,
    player2_type: str,
    num_games: int,
    board_type: BoardType,
    seed: int | None,
    checkpoint_path: str | None,
    checkpoint_path2: str | None,
    cmaes_weights_path: str | None,
    minimax_depth: int,
    max_moves_per_game: int,
    verbose: bool,
) -> EvaluationResults:
    """Run the full evaluation between two AI types.

    Args:
        player1_type: Type of AI for player 1
        player2_type: Type of AI for player 2
        num_games: Number of games to play
        board_type: Board type to use
        seed: Random seed for reproducibility
        checkpoint_path: Path to neural network checkpoint for player 1
        checkpoint_path2: Path to neural network checkpoint for player 2
            (if None, uses checkpoint_path for both players)
        cmaes_weights_path: Path to CMA-ES weights file
        minimax_depth: Depth for minimax search
        max_moves_per_game: Maximum moves per game before draw
        verbose: Whether to print verbose output

    Returns:
        EvaluationResults with all metrics
    """
    # If checkpoint2 not specified, both players use checkpoint_path
    p1_checkpoint = checkpoint_path
    p2_checkpoint = checkpoint_path2 if checkpoint_path2 else checkpoint_path
    start_time = time.time()

    p1_label = _format_ai_label(player1_type, p1_checkpoint)
    p2_label = _format_ai_label(player2_type, p2_checkpoint)

    board_value = board_type.value if hasattr(board_type, "value") else str(board_type)
    results = EvaluationResults(
        config={
            "player1": player1_type,
            "player2": player2_type,
            "player1_label": p1_label,
            "player2_label": p2_label,
            "player1_checkpoint": p1_checkpoint,
            "player2_checkpoint": p2_checkpoint,
            "games": num_games,
            "board": board_value,
            "seed": seed,
            "max_moves_per_game": max_moves_per_game,
            "minimax_depth": minimax_depth,
            "timestamp": datetime.now().isoformat(),
        }
    )

    # Create environment via canonical factory
    env_config = TrainingEnvConfig(
        board_type=board_type,
        num_players=2,
        max_moves=max_moves_per_game,
        reward_mode="terminal",
    )
    env = make_env(env_config)

    # Set up random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    progress_desc = f"{p1_label} vs {p2_label}"

    progress_label = f"{p1_label} vs {p2_label} | board={board_value}"
    progress_reporter = ProgressReporter(
        total_units=num_games,
        unit_name="game",
        report_interval_sec=10.0,
        context_label=progress_label,
    )

    for i in tqdm(range(num_games), desc=progress_desc, total=num_games):
        # Derive a unique game seed for this specific game.
        # This ensures AI instances get unique RNG streams per game while
        # keeping the overall evaluation reproducible.
        game_seed = (seed + i) if seed is not None else None

        # Alternate colors for fairness
        # Even games: player1_type as P1, odd games: player2_type as P1
        if i % 2 == 0:
            # player1_type plays as Player 1
            ai_p1 = create_ai(
                player1_type,
                1,
                board_type,
                p1_checkpoint,
                minimax_depth,
                cmaes_weights_path,
                game_seed,
            )
            ai_p2 = create_ai(
                player2_type,
                2,
                board_type,
                p2_checkpoint,
                minimax_depth,
                cmaes_weights_path,
                game_seed,
            )
            p1_is_player1_type = True
        else:
            # player2_type plays as Player 1 (color swap)
            # When swapping colors, also swap checkpoints so each AI type
            # always uses its designated checkpoint
            ai_p1 = create_ai(
                player2_type,
                1,
                board_type,
                p2_checkpoint,
                minimax_depth,
                cmaes_weights_path,
                game_seed,
            )
            ai_p2 = create_ai(
                player1_type,
                2,
                board_type,
                p1_checkpoint,
                minimax_depth,
                cmaes_weights_path,
                game_seed,
            )
            p1_is_player1_type = False

        # Reset environment with different seed for variety
        if seed is not None:
            env.reset(seed=seed + i)
        else:
            env.reset()

        p1_ai_type = player1_type if p1_is_player1_type else player2_type
        p2_ai_type = player2_type if p1_is_player1_type else player1_type
        p1_ai_label = p1_label if p1_is_player1_type else p2_label
        p2_ai_label = p2_label if p1_is_player1_type else p1_label

        game_result = play_single_game(
            ai_p1,
            ai_p2,
            env,
            max_moves=max_moves_per_game,
            p1_type=p1_ai_type,
            p2_type=p2_ai_type,
            verbose=verbose,
        )

        # Map result back to player1_type/player2_type win
        if game_result.winner is not None:
            if p1_is_player1_type:
                if game_result.winner == 1:
                    results.player1_wins += 1
                else:
                    results.player2_wins += 1
            else:
                # Colors were swapped
                if game_result.winner == 1:
                    results.player2_wins += 1
                else:
                    results.player1_wins += 1
        else:
            results.draws += 1

        # Record game statistics
        results.game_lengths.append(game_result.length)

        # Map decision times back to correct AIs
        if p1_is_player1_type:
            results.p1_decision_times.extend(game_result.p1_decision_times)
            results.p2_decision_times.extend(game_result.p2_decision_times)
            results.p1_final_pieces_list.append(game_result.p1_final_pieces)
            results.p2_final_pieces_list.append(game_result.p2_final_pieces)
        else:
            # Swap stats since colors were swapped
            results.p1_decision_times.extend(game_result.p2_decision_times)
            results.p2_decision_times.extend(game_result.p1_decision_times)
            results.p1_final_pieces_list.append(game_result.p2_final_pieces)
            results.p2_final_pieces_list.append(game_result.p1_final_pieces)

        # Track victory types
        if game_result.victory_type:
            vtype = game_result.victory_type
            results.victory_types[vtype] = results.victory_types.get(vtype, 0) + 1

        # Determine winner type name
        winner_is_p1_type = (p1_is_player1_type and game_result.winner == 1) or (
            not p1_is_player1_type and game_result.winner == 2
        )
        if game_result.winner:
            winner_name = player1_type if winner_is_p1_type else player2_type
            winner_label = p1_label if winner_is_p1_type else p2_label
            winner_num = 1 if winner_is_p1_type else 2
        else:
            winner_name = "draw"
            winner_label = "draw"
            winner_num = 0

        # Store game record
        results.games.append(
            {
                "game_number": i + 1,
                "winner": winner_name,
                "winner_label": winner_label,
                "winner_number": winner_num,
                "length": game_result.length,
                "victory_type": game_result.victory_type,
                "p1_was": p1_ai_type,
                "p2_was": p2_ai_type,
                "p1_was_label": p1_ai_label,
                "p2_was_label": p2_ai_label,
                "error": game_result.error,
            }
        )

        games_completed = i + 1
        progress_reporter.update(
            completed=games_completed,
            extra_metrics={
                "p1_wins": results.player1_wins,
                "p2_wins": results.player2_wins,
                "draws": results.draws,
            },
        )

    elapsed = time.time() - start_time
    results.total_runtime_seconds = elapsed

    games_per_sec = num_games / elapsed if elapsed > 0 and num_games > 0 else 0.0
    progress_reporter.finish(
        extra_metrics={
            "p1_wins": results.player1_wins,
            "p2_wins": results.player2_wins,
            "draws": results.draws,
            "games_per_sec": games_per_sec,
        },
    )

    return results


def format_results_json(results: EvaluationResults) -> dict[str, Any]:
    """Format EvaluationResults as a JSON-serializable dictionary.

    Args:
        results: Evaluation results

    Returns:
        Dictionary suitable for JSON serialization
    """
    total_games = results.player1_wins + results.player2_wins + results.draws

    # Calculate statistics
    p1_win_rate = results.player1_wins / total_games if total_games > 0 else 0
    p1_win_rate_ci = wilson_score_interval(results.player1_wins, total_games)

    avg_game_length = sum(results.game_lengths) / len(results.game_lengths) if results.game_lengths else 0

    game_length_std = 0.0
    if len(results.game_lengths) > 1:
        mean = avg_game_length
        variance = sum((x - mean) ** 2 for x in results.game_lengths) / len(results.game_lengths)
        game_length_std = math.sqrt(variance)

    avg_decision_time_p1 = (
        sum(results.p1_decision_times) / len(results.p1_decision_times) if results.p1_decision_times else 0
    )

    avg_decision_time_p2 = (
        sum(results.p2_decision_times) / len(results.p2_decision_times) if results.p2_decision_times else 0
    )

    # Piece advantage calculation
    avg_p1_pieces = (
        sum(results.p1_final_pieces_list) / len(results.p1_final_pieces_list) if results.p1_final_pieces_list else 0
    )
    avg_p2_pieces = (
        sum(results.p2_final_pieces_list) / len(results.p2_final_pieces_list) if results.p2_final_pieces_list else 0
    )

    return {
        "config": results.config,
        "results": {
            "player1_wins": results.player1_wins,
            "player2_wins": results.player2_wins,
            "draws": results.draws,
            "player1_win_rate": round(p1_win_rate, 4),
            "player1_win_rate_ci95": list(p1_win_rate_ci),
            "avg_game_length": round(avg_game_length, 2),
            "avg_game_length_std": round(game_length_std, 2),
            "avg_decision_time_p1": round(avg_decision_time_p1, 4),
            "avg_decision_time_p2": round(avg_decision_time_p2, 4),
            "total_runtime_seconds": round(results.total_runtime_seconds, 2),
            "victory_types": results.victory_types,
            "avg_p1_final_pieces": round(avg_p1_pieces, 2),
            "avg_p2_final_pieces": round(avg_p2_pieces, 2),
            "piece_advantage_p1": round(avg_p1_pieces - avg_p2_pieces, 2),
        },
        "games": results.games,
    }


def print_summary(results: EvaluationResults) -> None:
    """Print a human-readable summary of evaluation results.

    Args:
        results: Evaluation results to summarize
    """
    formatted = format_results_json(results)
    config = formatted["config"]
    res = formatted["results"]
    p1_label = config.get("player1_label") or config["player1"]
    p2_label = config.get("player2_label") or config["player2"]

    print("\n" + "=" * 60)
    print("AI MODEL EVALUATION RESULTS")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Player 1: {p1_label}")
    print(f"  Player 2: {p2_label}")
    print(f"  Games:    {config['games']}")
    print(f"  Board:    {config['board']}")
    print(f"  Seed:     {config.get('seed', 'None')}")

    print("\nResults:")
    print(f"  {p1_label} wins: {res['player1_wins']}")
    print(f"  {p2_label} wins: {res['player2_wins']}")
    print(f"  Draws: {res['draws']}")

    print(f"\nWin Rate ({p1_label}):")
    print(f"  Rate:   {res['player1_win_rate']:.1%}")
    ci_lo = res["player1_win_rate_ci95"][0]
    ci_hi = res["player1_win_rate_ci95"][1]
    print(f"  95% CI: [{ci_lo:.1%}, {ci_hi:.1%}]")

    print("\nGame Statistics:")
    avg_len = res["avg_game_length"]
    std_len = res["avg_game_length_std"]
    print(f"  Avg length: {avg_len:.1f} +/- {std_len:.1f} moves")
    p1_time = res["avg_decision_time_p1"] * 1000
    p2_time = res["avg_decision_time_p2"] * 1000
    print(f"  Avg decision time ({p1_label}): {p1_time:.1f}ms")
    print(f"  Avg decision time ({p2_label}): {p2_time:.1f}ms")

    print("\nVictory Types:")
    for vtype, count in res["victory_types"].items():
        print(f"  {vtype}: {count}")

    print("\nPiece Advantage:")
    print(f"  Avg final pieces ({p1_label}): " f"{res['avg_p1_final_pieces']:.1f}")
    print(f"  Avg final pieces ({p2_label}): " f"{res['avg_p2_final_pieces']:.1f}")
    print(f"  {p1_label} advantage: " f"{res['piece_advantage_p1']:+.1f}")

    print(f"\nTotal runtime: {res['total_runtime_seconds']:.1f}s")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate AI models with comprehensive metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline heuristic vs Random (50 games)
  python scripts/evaluate_ai_models.py \\
      --player1 baseline_heuristic --player2 random --games 50

  # CMA-ES optimized vs baseline heuristic
  python scripts/evaluate_ai_models.py \\
      --player1 cmaes_heuristic --player2 baseline_heuristic --games 100

  # Neural network vs minimax
  python scripts/evaluate_ai_models.py \\
      --player1 neural_network --player2 minimax \\
      --checkpoint checkpoints/checkpoint_final_epoch_5.pth

Supported AI Types:
  - baseline_heuristic: HeuristicAI with BASE_V1_BALANCED_WEIGHTS
  - cmaes_heuristic: HeuristicAI with CMA-ES optimized weights
  - neural_network: DescentAI with trained neural network
  - random: RandomAI baseline
  - minimax: MinimaxAI with alpha-beta pruning
        """,
    )

    parser.add_argument("--player1", type=str, choices=SUPPORTED_AI_TYPES, required=True, help="AI type for player 1")

    parser.add_argument("--player2", type=str, choices=SUPPORTED_AI_TYPES, required=True, help="AI type for player 2")

    parser.add_argument("--games", type=int, default=50, help="Number of games to play (default: 50)")

    parser.add_argument(
        "--board",
        type=str,
        choices=list(BOARD_TYPE_MAP.keys()),
        default="square8",
        help="Board type to use (default: square8)",
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    parser.add_argument("--output", type=str, default=None, help="Output JSON file path for results")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to neural network checkpoint for player 1 (or both if --checkpoint2 not set)",
    )

    parser.add_argument(
        "--checkpoint2",
        type=str,
        default=None,
        help="Path to neural network checkpoint for player 2 (enables NN vs NN comparison)",
    )

    parser.add_argument("--cmaes-weights", type=str, default=None, help="Path to CMA-ES optimized weights JSON file")

    parser.add_argument("--minimax-depth", type=int, default=3, help="Search depth for minimax AI (default: 3)")

    parser.add_argument(
        "--max-moves",
        type=int,
        default=10000,
        help="Maximum moves per game before draw (default: 10000)",
    )

    parser.add_argument("--verbose", action="store_true", help="Print verbose move-by-move output")

    parser.add_argument("--quiet", action="store_true", help="Suppress summary output (only write to file)")

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Convert board type
    board_type = BOARD_TYPE_MAP[args.board]

    print(f"\nStarting evaluation: {args.player1} vs {args.player2}")
    print(f"Games: {args.games}, Board: {args.board}, Seed: {args.seed}")
    if args.checkpoint:
        print(f"P1 checkpoint: {args.checkpoint}")
    if args.checkpoint2:
        print(f"P2 checkpoint: {args.checkpoint2}")

    # Run evaluation
    results = run_evaluation(
        player1_type=args.player1,
        player2_type=args.player2,
        num_games=args.games,
        board_type=board_type,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        checkpoint_path2=args.checkpoint2,
        cmaes_weights_path=args.cmaes_weights,
        minimax_depth=args.minimax_depth,
        max_moves_per_game=args.max_moves,
        verbose=args.verbose,
    )

    # Format results
    formatted_results = format_results_json(results)

    # Print summary unless quiet
    if not args.quiet:
        print_summary(results)

    # Write to output file if specified
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(formatted_results, f, indent=2)
        print(f"\nResults written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
