#!/usr/bin/env python3
"""Evaluate experimental neural network models against baselines.

Production-grade evaluation script for comparing GMO, GMO v2, IG-GMO, and EBMO
models against baselines (Random, Heuristic, MCTS).

Usage:
    # Evaluate single model
    python scripts/evaluate_nn_models.py \
        --model gmo --checkpoint models/gmo/sq8_2p_mega/gmo_best.pt \
        --games 50 --device cuda

    # Evaluate multiple models
    python scripts/evaluate_nn_models.py \
        --model gmo --checkpoint models/gmo/gmo_best.pt \
        --model ig_gmo --checkpoint models/ig_gmo/sq8_2p_mega/ig_gmo_best.pt \
        --baselines random,heuristic \
        --games 100 --device cuda --output results/eval_20251222.json

    # Full tournament (all models vs all baselines)
    python scripts/evaluate_nn_models.py \
        --tournament --games 50 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.game_engine import GameEngine
from app.models import AIConfig, BoardType
from app.training.train_gmo_selfplay import create_initial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a match between two AIs."""
    model_name: str
    opponent_name: str
    model_player: int  # 1 or 2
    winner: int | None
    moves: int
    game_id: str


@dataclass
class EvaluationResult:
    """Aggregated evaluation results for a model."""
    model_name: str
    model_path: str | None
    opponent_name: str
    games_played: int
    wins_as_p1: int
    wins_as_p2: int
    total_wins: int
    win_rate: float
    win_rate_as_p1: float
    win_rate_as_p2: float
    avg_game_length: float


def create_ai(
    model_type: str,
    player_number: int,
    checkpoint_path: str | None = None,
    device: str = "cuda",
    game_seed: int | None = None,
):
    """Create an AI instance based on model type.

    Supported types: gmo, gmo_v2, ig_gmo, ebmo, random, heuristic, mcts

    Args:
        game_seed: Optional per-game seed for varied randomness (Jan 2026 fix).
                   Without this, RandomAI/HeuristicAI produce identical games.
    """
    # Derive player-specific seed for varied but reproducible behavior
    rng_seed = None
    if game_seed is not None:
        rng_seed = (game_seed * 104729 + player_number * 7919) & 0xFFFFFFFF

    if model_type == "random":
        from app.ai.random_ai import RandomAI
        return RandomAI(player_number=player_number, config=AIConfig(difficulty=1, rng_seed=rng_seed))

    elif model_type == "heuristic":
        from app.ai.heuristic_ai import HeuristicAI
        return HeuristicAI(player_number=player_number, config=AIConfig(difficulty=3, rng_seed=rng_seed))

    elif model_type == "mcts":
        from app.ai.factory import AIFactory
        from app.models import AIType
        config = AIConfig(difficulty=5, think_time=3000, use_neural_net=False)
        return AIFactory.create(AIType.MCTS, player_number=player_number, config=config)

    elif model_type == "gmo":
        from app.ai.gmo_ai import GMOAI, GMOConfig
        gmo_config = GMOConfig(device=device)
        ai = GMOAI(
            player_number=player_number,
            config=AIConfig(difficulty=6),
            gmo_config=gmo_config,
        )
        if checkpoint_path:
            ai.load_checkpoint(Path(checkpoint_path))
        return ai

    elif model_type == "gmo_v2":
        from app.ai.gmo_v2 import create_gmo_v2
        return create_gmo_v2(
            player_number=player_number,
            device=device,
            checkpoint_path=checkpoint_path,
        )

    elif model_type == "ig_gmo":
        from app.ai.ig_gmo import create_ig_gmo
        path = Path(checkpoint_path) if checkpoint_path else None
        return create_ig_gmo(
            player_number=player_number,
            device=device,
            checkpoint_path=path,
        )

    elif model_type == "ebmo":
        from app.ai.ebmo_ai import create_ebmo_ai
        ebmo_config = AIConfig(difficulty=6)
        return create_ebmo_ai(
            player_number=player_number,
            config=ebmo_config,
            model_path=checkpoint_path,
        )

    elif model_type == "cnn":
        # Standalone CNN/NNUE neural network (no MCTS)
        from app.ai.neural_net import NeuralNetAI
        config = AIConfig(difficulty=6, nn_model_id=checkpoint_path)
        return NeuralNetAI(
            player_number=player_number,
            config=config,
            board_type=BoardType.SQUARE8,
        )

    elif model_type == "mcts_d5":
        # MCTS depth 5 without neural net
        from app.ai.factory import AIFactory
        from app.models import AIType
        config = AIConfig(difficulty=5, think_time=5000, use_neural_net=False)
        return AIFactory.create(AIType.MCTS, player_number=player_number, config=config)

    elif model_type == "mcts_d7":
        # MCTS depth 7 without neural net
        from app.ai.factory import AIFactory
        from app.models import AIType
        config = AIConfig(difficulty=7, think_time=10000, use_neural_net=False)
        return AIFactory.create(AIType.MCTS, player_number=player_number, config=config)

    elif model_type == "gmo_gumbel":
        # GMO + Gumbel MCTS hybrid
        from app.ai.gmo_gumbel_hybrid import GMOGumbelConfig, GumbelMCTSGMOAI
        gumbel_config = GMOGumbelConfig(device=device, simulation_budget=150)
        ai = GumbelMCTSGMOAI(
            player_number=player_number,
            config=AIConfig(difficulty=8),
            gumbel_config=gumbel_config,
        )
        if checkpoint_path:
            ai.load_gmo_checkpoint(checkpoint_path)
        return ai

    elif model_type == "gumbel_mcts":
        # Standard Gumbel MCTS with CNN
        from app.ai.gumbel_mcts_ai import GumbelMCTSAI
        config = AIConfig(
            difficulty=8,
            nn_model_id=checkpoint_path,
            gumbel_simulation_budget=150,
        )
        return GumbelMCTSAI(player_number, config, BoardType.SQUARE8)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def play_game(
    player1,
    player2,
    game_id: str,
    board_type: BoardType = BoardType.SQUARE8,
    max_moves: int = 600,
) -> tuple[int | None, int]:
    """Play a game between two AIs.

    Returns:
        Tuple of (winner player number or None, move count)
    """
    state = create_initial_state(
        game_id=game_id,
        board_type=board_type,
        rng_seed=hash(game_id) % (2**31),
    )

    move_count = 0
    while state.game_status.value == "active" and move_count < max_moves:
        current_player = state.current_player
        legal_moves = GameEngine.get_valid_moves(state, current_player)

        if not legal_moves:
            # Check for bookkeeping moves
            requirement = GameEngine.get_phase_requirement(state, current_player)
            if requirement is not None:
                move = GameEngine.synthesize_bookkeeping_move(requirement, state)
                if move:
                    state = GameEngine.apply_move(state, move)
                    move_count += 1
                    continue
            break

        ai = player1 if current_player == 1 else player2
        move = ai.select_move(state) if hasattr(ai, 'select_move') else ai.get_move(state)

        if move is None:
            requirement = GameEngine.get_phase_requirement(state, current_player)
            if requirement is not None:
                move = GameEngine.synthesize_bookkeeping_move(requirement, state)

        if move is None:
            break

        state = GameEngine.apply_move(state, move)
        move_count += 1

    return state.winner, move_count


def evaluate_model(
    model_type: str,
    model_checkpoint: str | None,
    opponent_type: str,
    num_games: int = 50,
    device: str = "cuda",
    board_type: BoardType = BoardType.SQUARE8,
) -> EvaluationResult:
    """Evaluate a model against an opponent, playing as both P1 and P2.

    Args:
        model_type: Type of model (gmo, gmo_v2, ig_gmo, ebmo)
        model_checkpoint: Path to model checkpoint
        opponent_type: Type of opponent (random, heuristic, mcts)
        num_games: Total number of games (split between P1 and P2)
        device: Device to use (cuda, mps, cpu)
        board_type: Board type for games

    Returns:
        EvaluationResult with aggregated statistics
    """
    games_per_side = num_games // 2
    wins_as_p1 = 0
    wins_as_p2 = 0
    total_moves = 0

    model_name = f"{model_type}_{Path(model_checkpoint).stem}" if model_checkpoint else model_type

    logger.info(f"Evaluating {model_name} vs {opponent_type} ({num_games} games)...")

    # Play as P1
    logger.info(f"  Playing as P1 ({games_per_side} games)...")
    model_ai = create_ai(model_type, player_number=1, checkpoint_path=model_checkpoint, device=device)
    opponent_ai = create_ai(opponent_type, player_number=2, device=device)

    for i in range(games_per_side):
        # Reset AIs with unique seeds per game for variety
        if hasattr(model_ai, 'reset_for_new_game'):
            model_ai.reset_for_new_game(rng_seed=i * 1000 + 1)
        if hasattr(opponent_ai, 'reset_for_new_game'):
            opponent_ai.reset_for_new_game(rng_seed=i * 1000 + 2)

        winner, moves = play_game(
            model_ai, opponent_ai,
            game_id=f"{model_name}_p1_{i}",
            board_type=board_type,
        )
        if winner == 1:
            wins_as_p1 += 1
        total_moves += moves

    # Play as P2
    logger.info(f"  Playing as P2 ({games_per_side} games)...")
    model_ai = create_ai(model_type, player_number=2, checkpoint_path=model_checkpoint, device=device)
    opponent_ai = create_ai(opponent_type, player_number=1, device=device)

    for i in range(games_per_side):
        # Reset AIs with unique seeds per game for variety (offset to avoid overlap with P1 games)
        if hasattr(opponent_ai, 'reset_for_new_game'):
            opponent_ai.reset_for_new_game(rng_seed=(games_per_side + i) * 1000 + 1)
        if hasattr(model_ai, 'reset_for_new_game'):
            model_ai.reset_for_new_game(rng_seed=(games_per_side + i) * 1000 + 2)

        winner, moves = play_game(
            opponent_ai, model_ai,
            game_id=f"{model_name}_p2_{i}",
            board_type=board_type,
        )
        if winner == 2:
            wins_as_p2 += 1
        total_moves += moves

    total_wins = wins_as_p1 + wins_as_p2

    result = EvaluationResult(
        model_name=model_name,
        model_path=model_checkpoint,
        opponent_name=opponent_type,
        games_played=num_games,
        wins_as_p1=wins_as_p1,
        wins_as_p2=wins_as_p2,
        total_wins=total_wins,
        win_rate=total_wins / num_games if num_games > 0 else 0,
        win_rate_as_p1=wins_as_p1 / games_per_side if games_per_side > 0 else 0,
        win_rate_as_p2=wins_as_p2 / games_per_side if games_per_side > 0 else 0,
        avg_game_length=total_moves / num_games if num_games > 0 else 0,
    )

    logger.info(f"  Result: {total_wins}/{num_games} wins ({result.win_rate*100:.1f}%)")
    logger.info(f"    P1: {wins_as_p1}/{games_per_side} ({result.win_rate_as_p1*100:.0f}%)")
    logger.info(f"    P2: {wins_as_p2}/{games_per_side} ({result.win_rate_as_p2*100:.0f}%)")

    return result


def run_tournament(
    models: list[tuple[str, str | None]],  # List of (model_type, checkpoint_path)
    baselines: list[str],
    num_games: int = 50,
    device: str = "cuda",
    board_type: BoardType = BoardType.SQUARE8,
) -> list[EvaluationResult]:
    """Run a tournament evaluating all models against all baselines.

    Args:
        models: List of (model_type, checkpoint_path) tuples
        baselines: List of baseline types to evaluate against
        num_games: Games per model-baseline pair
        device: Device to use
        board_type: Board type for games

    Returns:
        List of EvaluationResult objects
    """
    results = []

    for model_type, checkpoint in models:
        for baseline in baselines:
            try:
                result = evaluate_model(
                    model_type=model_type,
                    model_checkpoint=checkpoint,
                    opponent_type=baseline,
                    num_games=num_games,
                    device=device,
                    board_type=board_type,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate {model_type} vs {baseline}: {e}")

    return results


def print_summary(results: list[EvaluationResult]) -> None:
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    # Group by model
    by_model: dict[str, list[EvaluationResult]] = {}
    for r in results:
        if r.model_name not in by_model:
            by_model[r.model_name] = []
        by_model[r.model_name].append(r)

    for model_name, model_results in by_model.items():
        print(f"\n{model_name}:")
        print("-" * 60)
        print(f"{'Opponent':<20} {'Win Rate':>10} {'As P1':>10} {'As P2':>10}")
        print("-" * 60)

        for r in model_results:
            bar = "█" * int(r.win_rate * 10) + "░" * (10 - int(r.win_rate * 10))
            print(f"{r.opponent_name:<20} {bar} {r.win_rate*100:5.1f}% {r.win_rate_as_p1*100:5.0f}% {r.win_rate_as_p2*100:5.0f}%")

    print("\n" + "=" * 80)


def save_results(results: list[EvaluationResult], output_path: str) -> None:
    """Save results to JSON file."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(),
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate neural network models")

    parser.add_argument(
        "--model", action="append", dest="models",
        help="Model type to evaluate (gmo, gmo_v2, ig_gmo, ebmo)"
    )
    parser.add_argument(
        "--checkpoint", action="append", dest="checkpoints",
        help="Checkpoint path for each model (must match --model order)"
    )
    parser.add_argument(
        "--baselines", type=str, default="random",
        help="Comma-separated list of baselines (random, heuristic, mcts)"
    )
    parser.add_argument(
        "--games", type=int, default=50,
        help="Number of games per evaluation"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda, mps, cpu)"
    )
    parser.add_argument(
        "--board-type", type=str, default="square8",
        help="Board type (square8, hexagonal, etc.)"
    )
    parser.add_argument(
        "--output", type=str,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--tournament", action="store_true",
        help="Run full tournament with all available models"
    )

    args = parser.parse_args()

    # Parse board type
    board_type_map = {
        "square8": BoardType.SQUARE8,
        "hexagonal": BoardType.HEXAGONAL,
    }
    board_type = board_type_map.get(args.board_type, BoardType.SQUARE8)

    # Parse baselines
    baselines = [b.strip() for b in args.baselines.split(",")]

    # Build model list
    if args.tournament:
        # Default tournament models
        models = [
            ("gmo", "models/gmo/gmo_best.pt"),
            ("gmo", "models/gmo/sq8_2p_mega/gmo_best.pt"),
            ("ig_gmo", "models/ig_gmo/sq8_2p_mega/ig_gmo_best.pt"),
        ]
        baselines = ["random", "heuristic"]
    elif args.models:
        checkpoints = args.checkpoints or [None] * len(args.models)
        if len(checkpoints) != len(args.models):
            parser.error("Number of --checkpoint must match number of --model")
        models = list(zip(args.models, checkpoints))
    else:
        parser.error("Either --model or --tournament is required")

    # Run evaluation
    logger.info(f"Running evaluation on {args.device}")
    logger.info(f"Models: {models}")
    logger.info(f"Baselines: {baselines}")

    results = run_tournament(
        models=models,
        baselines=baselines,
        num_games=args.games,
        device=args.device,
        board_type=board_type,
    )

    # Print summary
    print_summary(results)

    # Save results
    if args.output:
        save_results(results, args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_output = f"results/nn_eval_{timestamp}.json"
        save_results(results, default_output)


if __name__ == "__main__":
    main()
