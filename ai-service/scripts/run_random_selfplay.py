#!/usr/bin/env python3
"""
Run selfplay games using purely random AI (uniform random move selection).
This explores the game tree more uniformly and may hit rare scenarios like recovery.

Uses unified SelfplayConfig for configuration (December 2025).
"""

import fcntl
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add app/ to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lib.logging_config import get_logger

logger = get_logger(__name__)

from app.game_engine import GameEngine
from app.models import GameStatus, MoveType
from app.training.initial_state import create_initial_state
from app.training.selfplay_config import SelfplayConfig, create_argument_parser
from app.utils.victory_type import derive_victory_type

# Import coordination for task limits and duration tracking
try:
    from app.coordination import (
        TaskType,
        can_spawn_safe,
        record_task_completion,
        register_running_task,
    )
    HAS_COORDINATION = True
except ImportError:
    HAS_COORDINATION = False
    TaskType = None
    can_spawn_safe = None


def play_random_game(
    board_type: str = "square8",
    num_players: int = 2,
    max_moves: int = 10000,
    seed: int | None = None,
    game_index: int = 0,
) -> dict:
    """Play a game using purely random move selection."""
    if seed is not None:
        random.seed(seed)

    state = create_initial_state(board_type=board_type, num_players=num_players)
    # Capture initial state for training data export (required for NPZ conversion)
    initial_state_snapshot = state.model_dump(mode="json")
    game_start_time = time.time()
    moves_played: list[dict] = []
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

        # Record move - only skip legacy event markers that are not canonical moves.
        # All other moves INCLUDING bookkeeping (NO_LINE_ACTION, NO_TERRITORY_ACTION, etc.)
        # must be recorded because they are required for replay - they advance the phase machine.
        LEGACY_EVENT_MARKERS = {
            MoveType.LINE_FORMATION,      # Legacy event marker, not a canonical move
            MoveType.TERRITORY_CLAIM,     # Legacy event marker, not a canonical move
        }
        if move.type not in LEGACY_EVENT_MARKERS:
            move_dict = {
                'type': move.type.value,
                'player': move.player,
            }
            if move.to:
                pos_dict = {'x': move.to.x, 'y': move.to.y}
                if move.to.z is not None:
                    pos_dict['z'] = move.to.z
                move_dict['to'] = pos_dict
            if hasattr(move, 'from_pos') and move.from_pos:
                pos_dict = {'x': move.from_pos.x, 'y': move.from_pos.y}
                if move.from_pos.z is not None:
                    pos_dict['z'] = move.from_pos.z
                move_dict['from'] = pos_dict
            if hasattr(move, 'capture_target') and move.capture_target:
                pos_dict = {'x': move.capture_target.x, 'y': move.capture_target.y}
                if move.capture_target.z is not None:
                    pos_dict['z'] = move.capture_target.z
                move_dict['capture_target'] = pos_dict
            # Record placement_count for place_ring moves (determines how many rings to place)
            if hasattr(move, 'placement_count') and move.placement_count is not None:
                move_dict['placement_count'] = move.placement_count
            moves_played.append(move_dict)

        # Apply move
        try:
            state = GameEngine.apply_move(state, move)
        except Exception as e:
            print(f"Error applying move {move_num}: {e}")
            break

    # Determine winner
    winner = None
    if state.game_status == GameStatus.COMPLETED and hasattr(state, 'winner'):
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
    # Use unified argument parser from SelfplayConfig
    parser = create_argument_parser(
        description='Run random AI selfplay games',
        include_gpu=False,  # Random selfplay doesn't use GPU
        include_ramdrive=False,
    )
    # Add script-specific arguments
    parser.add_argument('--max-moves', type=int, default=500, help='Max moves per game')
    parsed = parser.parse_args()

    # Create config from parsed args (uses canonical board type normalization)
    config = SelfplayConfig(
        board_type=parsed.board,
        num_players=parsed.num_players,
        num_games=parsed.num_games,
        output_dir=parsed.output_dir or 'data/selfplay/random_ai',
        seed=parsed.seed,
        source='run_random_selfplay.py',
    )

    # Use config values (provides validation and normalization)
    args = type('Args', (), {
        'num_games': config.num_games,
        'board_type': config.board_type,
        'num_players': config.num_players,
        'output_dir': config.output_dir,
        'seed': config.seed,
        'max_moves': parsed.max_moves,
    })()

    # Check coordination before spawning
    task_id = None
    coord_start_time = time.time()
    if HAS_COORDINATION:
        import socket
        node_id = socket.gethostname()
        allowed, reason = can_spawn_safe(TaskType.SELFPLAY, node_id)
        if not allowed:
            print(f"[Coordination] Warning: {reason}")
            print("[Coordination] Proceeding anyway (coordination is advisory)")

        # Register task for tracking
        task_id = f"random_selfplay_{args.board_type}_{args.num_players}p_{os.getpid()}"
        try:
            register_running_task(task_id, "selfplay", node_id, os.getpid())
            print(f"[Coordination] Registered task {task_id}")
        except Exception as e:
            print(f"[Coordination] Warning: Failed to register task: {e}")

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

    logger.info(
        "[random-selfplay] Starting %d games on %s %dp",
        args.num_games,
        args.board_type,
        args.num_players,
    )
    progress_interval = max(1, min(10, args.num_games // 20))  # Report ~20 times during run

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

            # Progress logging with ETA and throughput
            if (i + 1) % progress_interval == 0 or (i + 1) == args.num_games:
                elapsed = time.time() - start_time
                games_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = args.num_games - (i + 1)
                eta_seconds = remaining / games_per_sec if games_per_sec > 0 else 0
                pct = (i + 1) / args.num_games * 100
                logger.info(
                    "[random-selfplay] Game %d/%d (%.1f%%) | %.2f games/s | ETA: %.0fs | "
                    "recovery: %d games | victory: %s",
                    i + 1,
                    args.num_games,
                    pct,
                    games_per_sec,
                    eta_seconds,
                    games_with_recovery,
                    dict(victory_types),
                )

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

    logger.info(
        "[random-selfplay] Completed %d games in %.1fs (%.2f games/s)",
        args.num_games,
        elapsed,
        args.num_games / elapsed if elapsed > 0 else 0,
    )
    logger.info(
        "[random-selfplay] Recovery: %d/%d games | Total opportunities: %d",
        games_with_recovery,
        args.num_games,
        total_recovery,
    )
    logger.info("[random-selfplay] Victory types: %s", victory_types)
    logger.info("[random-selfplay] Results saved to %s", output_dir)

    # Record task completion for duration learning
    if HAS_COORDINATION and task_id:
        try:
            import socket
            node_id = socket.gethostname()
            config = f"{args.board_type}_{args.num_players}p"
            # Args: task_type, host, started_at, completed_at, success, config
            record_task_completion("selfplay", node_id, coord_start_time, time.time(), True, config)
            print("[Coordination] Recorded task completion")
        except Exception as e:
            print(f"[Coordination] Warning: Failed to record task completion: {e}")


if __name__ == "__main__":
    main()
