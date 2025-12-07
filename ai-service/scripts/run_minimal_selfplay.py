#!/usr/bin/env python3
"""
Minimal self-play game generator using only RandomAI.

This script avoids the neural net and heavy imports to prevent memory crashes.
It generates a small number of games for replay DB population.

Usage:
    cd ai-service
    PYTHONPATH=. python scripts/run_minimal_selfplay.py
"""

import argparse
import gc
import sys
import time

# Limit threads before any imports
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from app.training.env import TrainingEnvConfig, make_env
from app.ai.random_ai import RandomAI
from app.ai.neural_net import clear_model_cache
from app.models import AIConfig, BoardType
from app.db import get_or_create_db, record_completed_game_with_parity_check, ParityValidationError


def run_single_game(env, num_players: int, max_moves: int, seed: int):
    """Run a single game with RandomAI players. Returns (final_state, initial_state, moves, stats)."""
    state = env.reset(seed=seed)
    initial_state = state.model_copy(deep=True)

    # Create RandomAI for each player
    ais = {}
    for pnum in range(1, num_players + 1):
        config = AIConfig(difficulty=1, randomness=1.0, think_time=0, rngSeed=seed + pnum)
        ais[pnum] = RandomAI(player_number=pnum, config=config)

    moves = []
    move_count = 0

    # GameStatus is a str Enum, so compare with .value or str
    def is_active(s):
        status = s.game_status
        if hasattr(status, 'value'):
            return status.value == "active"
        return str(status) == "active"

    while is_active(state) and move_count < max_moves:
        current_player = state.current_player
        ai = ais.get(current_player)
        if ai is None:
            break

        try:
            move = ai.select_move(state)
            if move is None:
                break
            moves.append(move)
            state, reward, done, info = env.step(move)
            move_count += 1
            if done:
                break
        except Exception as e:
            print(f"  Error at move {move_count}: {e}", file=sys.stderr)
            break

    # Normalize game_status to string
    status = state.game_status
    status_str = status.value if hasattr(status, 'value') else str(status)

    stats = {
        "game_status": status_str,
        "move_count": move_count,
        "winner": state.winner,
    }
    return state, initial_state, moves, stats


def main():
    parser = argparse.ArgumentParser(description="Minimal self-play generator (RandomAI only)")
    parser.add_argument(
        "--board-type",
        default="square8",
        choices=["square8", "square19", "hexagonal"],
        help="Board type: square8, square19, or hexagonal.",
    )
    parser.add_argument("--num-players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--num-games", type=int, default=3)
    parser.add_argument("--max-moves", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--record-db", type=str, default=None,
                       help="Path to SQLite DB for recording games")
    args = parser.parse_args()

    print(f"Minimal self-play: {args.board_type} {args.num_players}p, {args.num_games} games")

    # Create environment
    board_type_enum = BoardType(args.board_type)
    env_config = TrainingEnvConfig(
        board_type=board_type_enum,
        num_players=args.num_players,
        max_moves=args.max_moves,
        reward_mode="terminal",
    )
    env = make_env(env_config)

    # Setup recording DB if specified
    replay_db = None
    if args.record_db:
        replay_db = get_or_create_db(args.record_db)
        print(f"Recording to: {args.record_db}")

    stats = {"finished": 0, "active": 0, "total_moves": 0, "recorded": 0}

    for i in range(args.num_games):
        game_seed = args.seed + i * 1000
        print(f"  Game {i+1}/{args.num_games} (seed={game_seed})...", end=" ", flush=True)

        start = time.time()
        final_state, initial_state, moves, result = run_single_game(
            env=env,
            num_players=args.num_players,
            max_moves=args.max_moves,
            seed=game_seed,
        )
        elapsed = time.time() - start

        print(f"done in {elapsed:.1f}s - {result['game_status']}, {result['move_count']} moves")

        stats[result["game_status"]] = stats.get(result["game_status"], 0) + 1
        stats["total_moves"] += result["move_count"]

        # Record to database if enabled
        # Use final_state.move_history instead of the AI-selected moves list
        # because move_history includes host-generated no-action moves
        # (NO_LINE_ACTION, NO_TERRITORY_ACTION) inserted to satisfy
        # RR-CANON-R075/R076 for canonical phase coverage.
        if replay_db:
            try:
                game_id = record_completed_game_with_parity_check(
                    db=replay_db,
                    initial_state=initial_state,
                    final_state=final_state,
                    moves=final_state.move_history,
                    metadata={"source": "minimal_selfplay"},
                )
                stats["recorded"] += 1
            except ParityValidationError as pve:
                print(
                    f"[PARITY ERROR] Game diverged at k={pve.divergence.move_index}:\n"
                    f"  Bundle: {pve.divergence.bundle_path or 'N/A'}",
                    file=sys.stderr,
                )
                raise
            except Exception as e:
                print(f"    Recording failed: {e}", file=sys.stderr)

        # Force garbage collection and clear model cache between games
        clear_model_cache()
        gc.collect()

    print(f"\nSummary: {stats}")

    if replay_db:
        print(f"Games recorded to {args.record_db}")


if __name__ == "__main__":
    main()
