#!/usr/bin/env python3
"""
Run selfplay games using purely random AI (uniform random move selection).
This explores the game tree more uniformly and may hit rare scenarios like recovery.
"""

import argparse
import fcntl
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add app/ to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import GameState, Move, MoveType, GamePhase, GameStatus
from app.game_engine import GameEngine
from app.training.generate_data import create_initial_state
from app.utils.victory_type import derive_victory_type


def play_random_game(
    board_type: str = "square8",
    num_players: int = 2,
    max_moves: int = 10000,
    seed: Optional[int] = None,
    game_index: int = 0,
) -> dict:
    """Play a game using purely random move selection."""
    if seed is not None:
        random.seed(seed)

    state = create_initial_state(board_type=board_type, num_players=num_players)
    # Capture initial state for training data export (required for NPZ conversion)
    initial_state_snapshot = state.model_dump(mode="json")
    game_start_time = time.time()
    moves_played: List[dict] = []
    recovery_opportunities = 0

    for move_num in range(max_moves):
        if state.game_status != GameStatus.ACTIVE:
            break

        current_player = state.current_player

        # Get valid moves
        valid_moves = GameEngine.get_valid_moves(state, current_player)

        # Check for bookkeeping moves
        req = GameEngine.get_phase_requirement(state, current_player)
        bookkeeping = None
        if req is not None:
            bookkeeping = GameEngine.synthesize_bookkeeping_move(req, state)

        # If no valid moves, apply bookkeeping
        if not valid_moves and bookkeeping:
            move = bookkeeping
        elif valid_moves:
            # Check if any recovery moves are available
            recovery_moves = [m for m in valid_moves if m.type == MoveType.RECOVERY_SLIDE]
            if recovery_moves:
                recovery_opportunities += 1

            # Random selection from valid moves
            move = random.choice(valid_moves)
        else:
            # No moves available
            break

        # Record move
        move_dict = {
            'type': move.type.value,
            'player': move.player,
        }
        if move.to:
            move_dict['to'] = {'x': move.to.x, 'y': move.to.y}
        if hasattr(move, 'from_pos') and move.from_pos:
            move_dict['from'] = {'x': move.from_pos.x, 'y': move.from_pos.y}
        moves_played.append(move_dict)

        # Apply move
        try:
            state = GameEngine.apply_move(state, move)
        except Exception as e:
            print(f"Error applying move {move_num}: {e}")
            break

    # Determine winner
    winner = None
    if state.game_status == GameStatus.COMPLETED:
        if hasattr(state, 'winner'):
            winner = state.winner

    # Derive standardized victory type using shared module
    vtype, stalemate_tb = derive_victory_type(state, max_moves)
    game_time = time.time() - game_start_time
    status = "completed" if state.game_status == GameStatus.COMPLETED else str(state.game_status.value)

    return {
        # === Core game identifiers ===
        'game_id': f"random_{board_type}_{num_players}p_{game_index}_{int(time.time())}",
        'board_type': board_type,  # square8, square19, hexagonal
        'num_players': num_players,
        # === Game outcome ===
        'winner': winner,
        'move_count': len(moves_played),
        'total_moves': len(moves_played),  # Alias for compatibility
        'status': status,
        'game_status': status,
        'victory_type': vtype,
        'stalemate_tiebreaker': stalemate_tb,
        'termination_reason': f"status:{status}:{vtype}",
        'completed': state.game_status == GameStatus.COMPLETED,
        # === Engine/opponent metadata ===
        'engine_mode': 'random-only',
        'opponent_type': 'selfplay',
        'player_types': ['random'] * num_players,
        'recovery_opportunities': recovery_opportunities,
        # === Training data (required for NPZ export) ===
        'moves': moves_played,
        'initial_state': initial_state_snapshot,
        # === Timing metadata ===
        'game_time_seconds': game_time,
        'timestamp': datetime.now().isoformat(),
        'created_at': datetime.now().isoformat(),
        # === Source tracking ===
        'source': 'run_random_selfplay.py',
    }


def main():
    parser = argparse.ArgumentParser(description='Run random AI selfplay games')
    parser.add_argument('--num-games', type=int, default=100, help='Number of games')
    parser.add_argument('--board-type', default='square8', help='Board type')
    parser.add_argument('--num-players', type=int, default=2, help='Number of players')
    parser.add_argument('--output-dir', default='data/selfplay/random_ai', help='Output directory')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--max-moves', type=int, default=500, help='Max moves per game')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    games_file = output_dir / "games.jsonl"
    stats_file = output_dir / "stats.json"

    start_time = time.time()
    total_recovery = 0
    games_with_recovery = 0
    wins = {}
    victory_types = {}
    stalemate_tiebreakers = {}

    print(f"Running {args.num_games} random AI games on {args.board_type} {args.num_players}p...")

    with open(games_file, 'a') as f:
        # Acquire exclusive lock to prevent concurrent writes from multiple processes
        # This prevents JSONL corruption when multiple selfplay runs target the same file
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"ERROR: Cannot acquire lock on {games_file} - another process is writing to it.")
            print("Use a different output file or wait for the other process to finish.")
            sys.exit(1)

        for i in range(args.num_games):
            game_seed = args.seed + i if args.seed else None
            game = play_random_game(
                board_type=args.board_type,
                num_players=args.num_players,
                max_moves=args.max_moves,
                seed=game_seed,
                game_index=i,
            )

            f.write(json.dumps(game) + '\n')
            f.flush()  # Minimize data loss on abnormal termination

            # Track stats
            if game['recovery_opportunities'] > 0:
                games_with_recovery += 1
                total_recovery += game['recovery_opportunities']

            if game['winner']:
                wins[game['winner']] = wins.get(game['winner'], 0) + 1
            if game['victory_type']:
                victory_types[game['victory_type']] = victory_types.get(game['victory_type'], 0) + 1
            if game.get('stalemate_tiebreaker'):
                stalemate_tiebreakers[game['stalemate_tiebreaker']] = stalemate_tiebreakers.get(game['stalemate_tiebreaker'], 0) + 1

            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{args.num_games} games, {games_with_recovery} had recovery opportunities")

    elapsed = time.time() - start_time

    stats = {
        'total_games': args.num_games,
        'board_type': args.board_type,
        'num_players': args.num_players,
        'games_with_recovery_opportunities': games_with_recovery,
        'total_recovery_opportunities': total_recovery,
        'wins_by_player': wins,
        'victory_types': victory_types,
        'stalemate_tiebreakers': stalemate_tiebreakers,
        'elapsed_seconds': elapsed,
        'games_per_second': args.num_games / elapsed,
        'timestamp': datetime.now().isoformat(),
    }

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nCompleted in {elapsed:.1f}s ({args.num_games/elapsed:.2f} games/sec)")
    print(f"Games with recovery opportunities: {games_with_recovery}/{args.num_games}")
    print(f"Total recovery opportunities: {total_recovery}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
