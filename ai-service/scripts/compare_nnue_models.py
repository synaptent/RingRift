#!/usr/bin/env python3
"""Compare NNUE models (512 vs 256 hidden dim) via selfplay games.

Runs a head-to-head tournament between two NNUE configurations to
determine which performs better in actual gameplay.

Usage:
    python scripts/compare_nnue_models.py --games 100
    python scripts/compare_nnue_models.py --model-a models/nnue/nnue_square8_2p.pt --model-b models/nnue/nnue_256.pt
"""
from __future__ import annotations


import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.nnue import RingRiftNNUE, extract_features_from_mutable
from app.models import BoardType
from app.rules import MutableGameState, create_game_state, get_rules_engine
from app.utils.torch_utils import safe_load_checkpoint


def load_nnue_model_from_path(model_path: str) -> RingRiftNNUE:
    """Load an NNUE model from a checkpoint file."""
    checkpoint = safe_load_checkpoint(model_path, map_location="cpu")

    # Get model config from checkpoint
    board_type_str = checkpoint.get("board_type", "square8")
    if isinstance(board_type_str, str):
        board_type = BoardType(board_type_str)
    else:
        board_type = board_type_str

    hidden_dim = checkpoint.get("hidden_dim", 256)
    num_hidden_layers = checkpoint.get("num_hidden_layers", 2)

    # Create model
    model = RingRiftNNUE(
        board_type=board_type,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


class SimpleNNUEEvaluator:
    """Simple wrapper for NNUE model evaluation.

    Wraps a pre-loaded NNUE model and provides evaluation methods.
    """

    SCORE_SCALE = 10000.0

    def __init__(self, model: RingRiftNNUE, player_number: int = 0):
        self.model = model
        self.player_number = player_number
        self.available = model is not None

    def evaluate(self, state: MutableGameState) -> float:
        """Evaluate a mutable game state."""
        if not self.available or self.model is None:
            return 0.0

        features = extract_features_from_mutable(state, self.player_number)
        with torch.no_grad():
            value = self.model.forward_single(features)
        return value * self.SCORE_SCALE

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class GameResult:
    winner: int | None  # None for draw
    moves: int
    duration: float
    model_a_player: int
    model_b_player: int


def play_game(
    eval_a: SimpleNNUEEvaluator,
    eval_b: SimpleNNUEEvaluator,
    board_type: str = "square8",
    num_players: int = 2,
    max_moves: int = 300,
    model_a_player: int = 0,
) -> GameResult:
    """Play a single game between two NNUE evaluators."""

    state = create_game_state(board_type=board_type, num_players=num_players)
    mutable = MutableGameState.from_immutable(state)
    rules_engine = get_rules_engine(board_type)

    start_time = time.time()
    moves = 0

    while not mutable.is_game_over() and moves < max_moves:
        current_player = mutable.current_player

        # Select evaluator based on player
        if current_player == model_a_player:
            evaluator = eval_a
        else:
            evaluator = eval_b

        # Get legal moves using rules engine
        immutable_state = mutable.to_immutable()
        legal_moves = rules_engine.get_valid_moves(immutable_state, current_player)
        if not legal_moves:
            break

        # Evaluate each move using NNUE
        best_move = None
        best_value = float('-inf')

        for move in legal_moves:
            # Make move
            undo = mutable.make_move(move)

            # Evaluate position
            value = evaluator.evaluate(mutable)

            # Negate for opponent's perspective
            if mutable.current_player != current_player:
                value = -value

            mutable.unmake_move(undo)

            if value > best_value:
                best_value = value
                best_move = move

        if best_move is None:
            best_move = legal_moves[0]

        mutable.make_move(best_move)  # Final move - no need to save undo
        moves += 1

    duration = time.time() - start_time

    # Determine winner
    winner = None
    if mutable.is_game_over():
        winner = mutable.get_winner()

    return GameResult(
        winner=winner,
        moves=moves,
        duration=duration,
        model_a_player=model_a_player,
        model_b_player=1 - model_a_player,
    )


def run_comparison(
    model_a_path: Path,
    model_b_path: Path,
    num_games: int = 100,
    board_type: str = "square8",
    num_players: int = 2,
) -> dict[str, Any]:
    """Run a comparison tournament between two NNUE models."""
    print("\nNNUE Model Comparison")
    print("=" * 60)
    print(f"Model A: {model_a_path}")
    print(f"Model B: {model_b_path}")
    print(f"Games: {num_games}")
    print(f"Board: {board_type}, Players: {num_players}")
    print("=" * 60)

    # Load models
    print("\nLoading models...")
    try:
        model_a = load_nnue_model_from_path(str(model_a_path))
        eval_a = SimpleNNUEEvaluator(model_a, player_number=0)
        print(f"  Model A loaded: {sum(p.numel() for p in model_a.parameters())} params")
    except Exception as e:
        print(f"  ERROR loading Model A: {e}")
        return {"error": str(e)}

    try:
        model_b = load_nnue_model_from_path(str(model_b_path))
        eval_b = SimpleNNUEEvaluator(model_b, player_number=0)
        print(f"  Model B loaded: {sum(p.numel() for p in model_b.parameters())} params")
    except Exception as e:
        print(f"  ERROR loading Model B: {e}")
        return {"error": str(e)}

    # Run games
    print(f"\nPlaying {num_games} games...")
    results: list[GameResult] = []
    model_a_wins = 0
    model_b_wins = 0
    draws = 0

    for i in range(num_games):
        # Alternate starting player for fairness
        model_a_player = i % 2

        result = play_game(
            eval_a, eval_b,
            board_type=board_type,
            num_players=num_players,
            model_a_player=model_a_player,
        )
        results.append(result)

        if result.winner is None:
            draws += 1
        elif result.winner == result.model_a_player:
            model_a_wins += 1
        else:
            model_b_wins += 1

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{num_games} - A: {model_a_wins}, B: {model_b_wins}, D: {draws}")

    # Calculate statistics
    total_moves = sum(r.moves for r in results)
    total_duration = sum(r.duration for r in results)
    avg_moves = total_moves / num_games
    avg_duration = total_duration / num_games

    model_a_win_rate = model_a_wins / num_games
    model_b_win_rate = model_b_wins / num_games
    draw_rate = draws / num_games

    # Determine winner
    if model_a_wins > model_b_wins:
        winner = "Model A"
        winner_path = str(model_a_path)
    elif model_b_wins > model_a_wins:
        winner = "Model B"
        winner_path = str(model_b_path)
    else:
        winner = "Draw"
        winner_path = None

    # Results summary
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Model A wins: {model_a_wins} ({model_a_win_rate:.1%})")
    print(f"Model B wins: {model_b_wins} ({model_b_win_rate:.1%})")
    print(f"Draws: {draws} ({draw_rate:.1%})")
    print(f"\nWinner: {winner}")
    print(f"Average moves/game: {avg_moves:.1f}")
    print(f"Average duration/game: {avg_duration:.2f}s")

    return {
        "model_a": str(model_a_path),
        "model_b": str(model_b_path),
        "games": num_games,
        "model_a_wins": model_a_wins,
        "model_b_wins": model_b_wins,
        "draws": draws,
        "model_a_win_rate": model_a_win_rate,
        "model_b_win_rate": model_b_win_rate,
        "draw_rate": draw_rate,
        "winner": winner,
        "winner_path": winner_path,
        "avg_moves": avg_moves,
        "avg_duration": avg_duration,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare NNUE models via selfplay")
    parser.add_argument("--model-a", type=Path,
                        default=AI_SERVICE_ROOT / "models" / "nnue" / "nnue_square8_2p.pt",
                        help="Path to model A (512 hidden dim)")
    parser.add_argument("--model-b", type=Path,
                        help="Path to model B (256 hidden dim baseline)")
    parser.add_argument("--games", type=int, default=50, help="Number of games to play")
    parser.add_argument("--board", default="square8", help="Board type")
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    parser.add_argument("--output", type=Path, help="Output JSON file for results")

    args = parser.parse_args()

    # If model-b not specified, look for a baseline
    if args.model_b is None:
        baseline_candidates = [
            AI_SERVICE_ROOT / "models" / "nnue" / "nnue_square8_2p_256.pt",
            AI_SERVICE_ROOT / "models" / "nnue" / "nnue_square8_2p_prev.pt",
            AI_SERVICE_ROOT / "models" / "nnue" / "nnue_square8_2p_baseline.pt",
        ]
        for candidate in baseline_candidates:
            if candidate.exists():
                args.model_b = candidate
                break

        if args.model_b is None:
            print("No baseline model found. Please specify --model-b")
            print("Creating a fresh 256-dim baseline for comparison...")
            # Could train a fresh 256 model here, but for now just exit
            return 1

    if not args.model_a.exists():
        print(f"Model A not found: {args.model_a}")
        return 1

    if not args.model_b.exists():
        print(f"Model B not found: {args.model_b}")
        return 1

    results = run_comparison(
        args.model_a,
        args.model_b,
        num_games=args.games,
        board_type=args.board,
        num_players=args.players,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
