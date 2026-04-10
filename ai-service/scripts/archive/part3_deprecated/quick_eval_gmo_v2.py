#!/usr/bin/env python3
"""Quick evaluation of GMO v2 model against baselines.

Uses direct value scoring (no gradient optimization) for fast evaluation.
"""
from __future__ import annotations


import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ai.gmo_v2 import create_gmo_v2, GMOv2AI
from app.game_engine import GameEngine
from app.models import BoardType, GameStatus
from app.training.train_gmo_selfplay import create_initial_state


def get_move_by_value(ai: GMOv2AI, state):
    """Select move by direct value scoring (no gradient optimization)."""
    legal_moves = GameEngine.get_valid_moves(state, state.current_player)
    if not legal_moves:
        return None

    ai.state_encoder.eval()
    ai.move_encoder.eval()
    ai.value_net.eval()

    with torch.no_grad():
        # Encode state once
        state_embed = ai.state_encoder(state)

        # Score all legal moves
        best_move = None
        best_value = float('-inf')

        for move in legal_moves:
            move_embed = ai.move_encoder(move)
            value, _ = ai.value_net(
                state_embed.unsqueeze(0),
                move_embed.unsqueeze(0)
            )
            if value.item() > best_value:
                best_value = value.item()
                best_move = move

    return best_move


def play_game(ai, opponent_type: str, ai_player: int, game_id: str) -> tuple[int | None, int]:
    """Play one game and return (winner, move_count)."""
    state = create_initial_state(
        game_id=game_id,
        board_type=BoardType.SQUARE8,
        rng_seed=hash(game_id) % (2**31),
    )

    move_count = 0
    max_moves = 400

    while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current = state.current_player
        legal_moves = GameEngine.get_valid_moves(state, current)

        if not legal_moves:
            # Check for bookkeeping
            req = GameEngine.get_phase_requirement(state, current)
            if req:
                move = GameEngine.synthesize_bookkeeping_move(req, state)
                if move:
                    state = GameEngine.apply_move(state, move)
                    move_count += 1
                    continue
            break

        if current == ai_player:
            move = get_move_by_value(ai, state)
        else:
            # Random opponent
            move = random.choice(legal_moves)

        if move is None:
            break

        state = GameEngine.apply_move(state, move)
        move_count += 1

    return state.winner, move_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    ai = create_gmo_v2(
        player_number=1,
        device=args.device,
        checkpoint_path=args.checkpoint,
    )

    wins_as_p1 = 0
    wins_as_p2 = 0
    games_per_side = args.games // 2

    print(f"\nPlaying as P1 ({games_per_side} games)...")
    for i in range(games_per_side):
        winner, moves = play_game(ai, "random", 1, f"p1_game_{i}")
        if winner == 1:
            wins_as_p1 += 1
        print(f"  Game {i+1}: {'Win' if winner == 1 else 'Loss'} ({moves} moves)")

    print(f"\nPlaying as P2 ({games_per_side} games)...")
    ai = create_gmo_v2(
        player_number=2,
        device=args.device,
        checkpoint_path=args.checkpoint,
    )
    for i in range(games_per_side):
        winner, moves = play_game(ai, "random", 2, f"p2_game_{i}")
        if winner == 2:
            wins_as_p2 += 1
        print(f"  Game {i+1}: {'Win' if winner == 2 else 'Loss'} ({moves} moves)")

    total_wins = wins_as_p1 + wins_as_p2
    total_games = args.games

    print(f"\n{'='*60}")
    print(f"RESULTS: GMO v2 vs Random")
    print(f"{'='*60}")
    print(f"Wins as P1: {wins_as_p1}/{games_per_side} ({100*wins_as_p1/games_per_side:.1f}%)")
    print(f"Wins as P2: {wins_as_p2}/{games_per_side} ({100*wins_as_p2/games_per_side:.1f}%)")
    print(f"Total: {total_wins}/{total_games} ({100*total_wins/total_games:.1f}%)")


if __name__ == "__main__":
    main()
