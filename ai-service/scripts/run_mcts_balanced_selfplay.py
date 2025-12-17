#!/usr/bin/env python3
"""Run balanced MCTS-only self-play for unbiased game data generation.

This script generates games using pure MCTS (no heuristic biases) to produce
balanced training data that reflects true game balance rather than AI biases.

Usage:
    python scripts/run_mcts_balanced_selfplay.py \
        --num-games 100 \
        --board-type square8 \
        --num-players 2 \
        --output-dir data/games/mcts_balanced

    # Run on all board types
    python scripts/run_mcts_balanced_selfplay.py \
        --num-games 50 \
        --all-boards \
        --output-dir data/games/mcts_balanced
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ai.mcts_ai import MCTSAI
from app.models import AIConfig, BoardType, GameStatus
from app.training.env import RingRiftEnv

# Unified logging setup
try:
    from app.core.logging_config import setup_logging
    logger = setup_logging("run_mcts_balanced_selfplay", log_dir="logs")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)


def play_mcts_game(
    board_type: BoardType,
    num_players: int,
    mcts_iterations: int = 400,
    seed: int = None,
    max_moves: int = 10000,
    randomness: float = 0.15,
) -> Dict[str, Any]:
    """Play a single game with MCTS-only players.

    Returns game record dict with moves, winner, and metadata.
    """
    # Create environment with proper phase handling
    env = RingRiftEnv(
        board_type=board_type,
        num_players=num_players,
        max_moves=max_moves,
        reward_on="terminal",
    )

    # Reset to get initial state
    state = env.reset(seed=seed)

    # Create MCTS AI for each player
    # Use varied seeds per player and game for diverse training data
    ais = {}
    for p in range(1, num_players + 1):
        # Derive unique seed per player per game using prime multiplication
        player_seed = ((seed or 0) * 1_000_003 + p * 97_911) & 0xFFFFFFFF if seed else None
        config = AIConfig(
            difficulty=8,  # High difficulty for strong MCTS play
            randomness=randomness,  # Configurable randomness for move diversity
            searchDepth=6,
            rngSeed=player_seed,
            use_neural_net=False,  # Pure MCTS without neural evaluation for unbiased games
        )
        ais[p] = MCTSAI(p, config)
        # Configure MCTS iterations
        ais[p].num_iterations = mcts_iterations

    moves_record = []
    move_count = 0
    start_time = time.time()
    done = False

    while not done and move_count < max_moves:
        current_player = state.current_player
        ai = ais[current_player]

        # Get legal moves (handles all phases including bookkeeping)
        legal_moves = env.legal_moves()
        if not legal_moves:
            # No moves available - game should end
            break

        # Get AI move
        move = ai.select_move(state)
        if move is None:
            # AI couldn't select a move - pick first legal
            move = legal_moves[0]

        # Record move
        moves_record.append({
            "player": current_player,
            "type": move.type.value if hasattr(move.type, 'value') else str(move.type),
            "from": {"x": move.from_pos.x, "y": move.from_pos.y} if move.from_pos else None,
            "to": {"x": move.to.x, "y": move.to.y} if move.to else None,
        })

        # Apply move using env.step (handles phase transitions properly)
        state, reward, done, info = env.step(move)
        move_count += 1

    elapsed = time.time() - start_time

    # Determine winner from final state or info
    winner = state.winner
    victory_type = "unknown"
    if done and 'victory_reason' in info:
        victory_type = info['victory_reason']
    elif state.game_status == GameStatus.COMPLETED:
        victory_type = "completed"
    elif move_count >= max_moves:
        victory_type = "timeout"

    return {
        "board_type": board_type.value,
        "num_players": num_players,
        "winner": winner,
        "victory_type": victory_type,
        "moves": moves_record,
        "move_count": move_count,
        "elapsed_seconds": elapsed,
        "mcts_iterations": mcts_iterations,
        "timestamp": datetime.utcnow().isoformat(),
        "ai_type": "mcts",
        "source": f"mcts_balanced_{board_type.value}_{num_players}p",
    }


def run_selfplay(
    board_type: BoardType,
    num_players: int,
    num_games: int,
    output_dir: Path,
    mcts_iterations: int = 400,
    base_seed: int = 42,
    randomness: float = 0.15,
) -> Dict[str, Any]:
    """Run multiple MCTS self-play games and save results."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"mcts_{board_type.value}_{num_players}p.jsonl"

    logger.info(f"Running {num_games} MCTS games: {board_type.value} {num_players}p")
    logger.info(f"Output: {output_file}")

    stats = {
        "total_games": 0,
        "wins_by_player": {str(p): 0 for p in range(1, num_players + 1)},
        "victory_types": {},
        "total_moves": 0,
        "total_time": 0,
    }

    with open(output_file, "a") as f:
        for i in range(num_games):
            seed = base_seed + i
            try:
                game = play_mcts_game(
                    board_type=board_type,
                    num_players=num_players,
                    mcts_iterations=mcts_iterations,
                    seed=seed,
                    randomness=randomness,
                )

                # Write to file
                f.write(json.dumps(game) + "\n")
                f.flush()

                # Update stats
                stats["total_games"] += 1
                stats["total_moves"] += game["move_count"]
                stats["total_time"] += game["elapsed_seconds"]

                if game["winner"]:
                    stats["wins_by_player"][str(game["winner"])] += 1

                vtype = game["victory_type"]
                stats["victory_types"][vtype] = stats["victory_types"].get(vtype, 0) + 1

                # Progress
                if (i + 1) % 10 == 0 or i == 0:
                    rate = stats["total_games"] / stats["total_time"] if stats["total_time"] > 0 else 0
                    logger.info(f"  Game {i+1}/{num_games} - {rate:.2f} games/sec - winner: {game['winner']}")

            except Exception as e:
                logger.error(f"Game {i+1} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Calculate win rates
    total = stats["total_games"]
    if total > 0:
        stats["win_rates"] = {
            p: (wins / total * 100)
            for p, wins in stats["wins_by_player"].items()
        }
        stats["games_per_second"] = total / stats["total_time"]
        stats["avg_moves_per_game"] = stats["total_moves"] / total

    # Save stats
    stats_file = output_dir / f"mcts_{board_type.value}_{num_players}p_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Completed {total} games. Win rates: {stats.get('win_rates', {})}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Run balanced MCTS self-play")
    parser.add_argument("--num-games", type=int, default=100, help="Games per config")
    parser.add_argument("--board-type", type=str, choices=["square8", "square19", "hexagonal"],
                        help="Board type (if not --all-boards)")
    parser.add_argument("--num-players", type=int, default=2, choices=[2, 3, 4],
                        help="Number of players")
    parser.add_argument("--all-boards", action="store_true",
                        help="Run on all board types")
    parser.add_argument("--mcts-iterations", type=int, default=400,
                        help="MCTS iterations per move")
    parser.add_argument("--randomness", type=float, default=0.15,
                        help="Randomness factor for move diversity (0.0-1.0, default 0.15)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/games/mcts_balanced"),
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")

    args = parser.parse_args()

    configs = []
    if args.all_boards:
        for board in [BoardType.SQUARE8, BoardType.SQUARE19, BoardType.HEXAGONAL]:
            for players in [2, 3, 4]:
                configs.append((board, players))
    else:
        if not args.board_type:
            parser.error("--board-type required unless --all-boards specified")
        board_map = {
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hexagonal": BoardType.HEXAGONAL,
        }
        configs.append((board_map[args.board_type], args.num_players))

    all_stats = {}
    for board_type, num_players in configs:
        key = f"{board_type.value}_{num_players}p"
        stats = run_selfplay(
            board_type=board_type,
            num_players=num_players,
            num_games=args.num_games,
            output_dir=args.output_dir,
            mcts_iterations=args.mcts_iterations,
            base_seed=args.seed,
            randomness=args.randomness,
        )
        all_stats[key] = stats

    # Print summary
    print("\n" + "=" * 60)
    print("MCTS Balanced Self-Play Summary")
    print("=" * 60)
    for key, stats in all_stats.items():
        print(f"\n{key}:")
        print(f"  Games: {stats['total_games']}")
        print(f"  Win rates: {stats.get('win_rates', {})}")
        print(f"  Avg moves: {stats.get('avg_moves_per_game', 0):.1f}")
        print(f"  Victory types: {stats.get('victory_types', {})}")


if __name__ == "__main__":
    main()
