#!/usr/bin/env python3
"""Evaluate GNN policy model against baselines.

Runs the trained GNN model against Random and Heuristic opponents
to verify gameplay performance matches validation accuracy improvements.

Usage:
    python scripts/evaluate_gnn_model.py --games 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import torch

from app.game_engine import GameEngine
from app.models import BoardType, AIConfig
from app.ai.random_ai import RandomAI
from app.ai.heuristic_ai import HeuristicAI
from app.ai.gnn_ai import GNNAI, HAS_PYG
from app.training.initial_state import create_initial_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _resolve_board_type(value: str | BoardType | None) -> BoardType:
    """Normalize board type from checkpoints or CLI."""
    if isinstance(value, BoardType):
        return value
    if isinstance(value, str):
        normalized = value.lower()
        mapping = {
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hex8": BoardType.HEX8,
            "hexagonal": BoardType.HEXAGONAL,
        }
        if normalized in mapping:
            return mapping[normalized]
    return BoardType.HEX8


def _load_checkpoint(path: str):
    """Load a torch checkpoint with PyTorch 2.6+ weights_only fallback."""
    from app.utils.torch_utils import safe_load_checkpoint
    return safe_load_checkpoint(path, map_location="cpu")


def _build_gnn_ai(model_path: str, player_number: int, device: str) -> GNNAI:
    """Instantiate a GNNAI player with the given checkpoint."""
    return GNNAI(
        player_number=player_number,
        config=AIConfig(difficulty=8),
        model_path=model_path,
        device=device,
    )


def play_game(p1, p2, game_id: str, board_type: BoardType, max_moves: int = 300):
    """Play a game between two players."""
    state = create_initial_state(
        board_type=board_type,
        num_players=2,
    )

    move_count = 0
    while state.game_status.value == "active" and move_count < max_moves:
        current = state.current_player
        legal = GameEngine.get_valid_moves(state, current)

        if not legal:
            req = GameEngine.get_phase_requirement(state, current)
            if req:
                move = GameEngine.synthesize_bookkeeping_move(req, state)
                if move:
                    state = GameEngine.apply_move(state, move)
                    move_count += 1
                    continue
            break

        player = p1 if current == 1 else p2
        move = player.select_move(state) if hasattr(player, 'select_move') else player.get_move(state)

        if move is None:
            break

        state = GameEngine.apply_move(state, move)
        move_count += 1

    return state.winner, move_count


def evaluate_against_baseline(
    model_path: str,
    baseline: str,
    num_games: int = 20,
    board_type: BoardType = BoardType.HEXAGONAL,
    device: str = "cpu",
    max_moves: int = 300,
):
    """Evaluate GNN model against a baseline."""

    wins_as_p1 = 0
    wins_as_p2 = 0
    games_per_side = num_games // 2

    logger.info(f"Evaluating GNN vs {baseline} ({num_games} games)...")

    # Pre-create GNN players
    gnn_p1 = _build_gnn_ai(model_path, player_number=1, device=device)
    gnn_p2 = _build_gnn_ai(model_path, player_number=2, device=device)

    # Play as P1
    logger.info(f"  Playing as P1 ({games_per_side} games)...")
    for i in range(games_per_side):
        if baseline == "random":
            opponent = RandomAI(player_number=2, config=AIConfig(difficulty=1, rng_seed=i))
        else:
            opponent = HeuristicAI(player_number=2, config=AIConfig(difficulty=3, rng_seed=i))

        winner, moves = play_game(gnn_p1, opponent, f"gnn_p1_{i}", board_type, max_moves=max_moves)
        if winner == 1:
            wins_as_p1 += 1

    # Play as P2
    logger.info(f"  Playing as P2 ({games_per_side} games)...")
    for i in range(games_per_side):
        if baseline == "random":
            opponent = RandomAI(player_number=1, config=AIConfig(difficulty=1, rng_seed=i + 10000))
        else:
            opponent = HeuristicAI(player_number=1, config=AIConfig(difficulty=3, rng_seed=i + 10000))

        winner, moves = play_game(opponent, gnn_p2, f"gnn_p2_{i}", board_type, max_moves=max_moves)
        if winner == 2:
            wins_as_p2 += 1

    total_wins = wins_as_p1 + wins_as_p2
    win_rate = total_wins / num_games

    logger.info(f"  Result: {total_wins}/{num_games} ({win_rate*100:.1f}%)")
    logger.info(f"    As P1: {wins_as_p1}/{games_per_side}")
    logger.info(f"    As P2: {wins_as_p2}/{games_per_side}")

    return {
        "baseline": baseline,
        "total_wins": total_wins,
        "win_rate": win_rate,
        "wins_as_p1": wins_as_p1,
        "wins_as_p2": wins_as_p2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/gnn_hex8_2p/gnn_policy_best.pt")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--baselines", default="random,heuristic")
    parser.add_argument("--max-moves", type=int, default=200)
    args = parser.parse_args()

    if not HAS_PYG:
        raise SystemExit("PyTorch Geometric required for GNN evaluation.")

    baselines = args.baselines.split(",")

    print("=" * 60)
    print("GNN MODEL GAMEPLAY EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Games per baseline: {args.games}")
    print(f"Baselines: {baselines}")
    print()

    # Load model info
    ckpt = _load_checkpoint(args.model)
    print(f"GNN Validation Accuracy: {ckpt['val_acc']*100:.2f}%")
    print(f"Architecture: {ckpt['conv_type'].upper()}, {ckpt['num_layers']} layers")
    print()

    board_type = _resolve_board_type(ckpt.get("board_type"))

    results = []
    for baseline in baselines:
        result = evaluate_against_baseline(
            args.model,
            baseline.strip(),
            args.games,
            board_type,
            device="cpu",
            max_moves=args.max_moves,
        )
        results.append(result)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        bar = "█" * int(r["win_rate"] * 10) + "░" * (10 - int(r["win_rate"] * 10))
        print(f"vs {r['baseline']:<12}: {bar} {r['win_rate']*100:5.1f}%")

    print("\nNote: GNN moves use canonical action encoding via GNNAI.")
    print("=" * 60)


if __name__ == "__main__":
    main()
