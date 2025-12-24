#!/usr/bin/env python3
"""Multi-game parallel Gumbel MCTS selfplay generation.

Generates high-quality training data using Gumbel MCTS with batched NN
evaluation across multiple games for 10-20x speedup.

Usage:
    python scripts/multi_game_gumbel_selfplay.py \
        --board square8 \
        --num-players 2 \
        --num-games 100 \
        --batch-size 64 \
        --simulation-budget 800 \
        --output data/selfplay/multi_game_sq8_2p.jsonl

    # With specific model
    python scripts/multi_game_gumbel_selfplay.py \
        --board hex8 \
        --num-players 2 \
        --model models/canonical_hex8_2p.pth \
        --num-games 500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

# Ensure app imports resolve
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.multi_game_gumbel import MultiGameGumbelRunner, GameResult
from app.models import BoardType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_board_type(s: str) -> BoardType:
    """Parse board type string."""
    s = s.lower().strip()
    if "square8" in s or "sq8" in s:
        return BoardType.SQUARE8
    elif "square19" in s or "sq19" in s:
        return BoardType.SQUARE19
    elif "hex8" in s:
        return BoardType.HEX8
    elif "hex" in s:
        return BoardType.HEXAGONAL
    raise ValueError(f"Unknown board type: {s}")


def load_neural_net(model_path: str | None, board_type: BoardType, num_players: int):
    """Load neural network for evaluation."""
    if model_path is None:
        logger.info("No model specified, using uniform policy")
        return None

    try:
        from app.ai.neural_net import NeuralNetAI
        from app.models import AIConfig

        config = AIConfig(
            difficulty=9,
            use_neural_net=True,
            nn_model_id=model_path,
        )
        nn = NeuralNetAI(player_number=1, config=config, board_type=board_type)
        logger.info(f"Loaded neural network from {model_path}")
        return nn
    except Exception as e:
        logger.warning(f"Failed to load neural network: {e}")
        return None


def json_serializer(obj):
    """Custom JSON serializer for non-standard types."""
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def save_results_jsonl(
    results: list[GameResult],
    output_path: str,
    board_type: str = "square8",
    num_players: int = 2,
) -> None:
    """Save game results to JSONL format."""
    with open(output_path, 'a') as f:
        for result in results:
            record = {
                "game_id": str(uuid.uuid4()),
                "board_type": board_type,
                "num_players": num_players,
                "winner": result.winner,
                "status": result.status,
                "move_count": result.move_count,
                "moves": result.moves,
                "initial_state": result.initial_state,
                "timestamp": datetime.utcnow().isoformat(),
            }
            f.write(json.dumps(record, default=json_serializer) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Multi-game parallel Gumbel MCTS selfplay")
    parser.add_argument("--board", type=str, default="square8", help="Board type")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    parser.add_argument("--num-games", type=int, default=100, help="Total games to generate")
    parser.add_argument("--batch-size", type=int, default=64, help="Games per batch")
    parser.add_argument("--simulation-budget", type=int, default=800, help="MCTS simulations per move")
    parser.add_argument("--num-sampled-actions", type=int, default=16, help="Gumbel-Top-K actions")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path")
    parser.add_argument("--max-moves", type=int, default=500, help="Max moves per game")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    board_type = parse_board_type(args.board)
    logger.info(f"Board: {board_type.value}, Players: {args.num_players}")
    logger.info(f"Games: {args.num_games}, Batch size: {args.batch_size}")
    logger.info(f"Simulation budget: {args.simulation_budget}")

    # Load neural network
    nn = load_neural_net(args.model, board_type, args.num_players)

    # Create output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/selfplay/multi_game_{board_type.value}_{args.num_players}p_{timestamp}.jsonl"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Create runner
    runner = MultiGameGumbelRunner(
        num_games=args.batch_size,
        simulation_budget=args.simulation_budget,
        num_sampled_actions=args.num_sampled_actions,
        board_type=board_type,
        num_players=args.num_players,
        neural_net=nn,
        device=args.device,
        max_moves_per_game=args.max_moves,
    )

    # Run batches
    total_games = 0
    total_start = time.time()
    batches_needed = (args.num_games + args.batch_size - 1) // args.batch_size

    logger.info(f"Running {batches_needed} batches...")

    for batch_idx in range(batches_needed):
        remaining = args.num_games - total_games
        games_this_batch = min(args.batch_size, remaining)

        batch_start = time.time()
        results = runner.run_batch(num_games=games_this_batch)
        batch_elapsed = time.time() - batch_start

        # Save results
        save_results_jsonl(results, output_path, board_type.value, args.num_players)

        total_games += len(results)
        winners = [r.winner for r in results if r.winner is not None]
        avg_moves = sum(r.move_count for r in results) / len(results) if results else 0

        logger.info(
            f"Batch {batch_idx + 1}/{batches_needed}: {len(results)} games in {batch_elapsed:.1f}s, "
            f"avg moves: {avg_moves:.1f}, completed: {len(winners)}/{len(results)}"
        )

    total_elapsed = time.time() - total_start
    games_per_hour = (total_games / total_elapsed) * 3600

    logger.info(f"\n=== Summary ===")
    logger.info(f"Total games: {total_games}")
    logger.info(f"Total time: {total_elapsed:.1f}s")
    logger.info(f"Games/hour: {games_per_hour:.0f}")
    logger.info(f"NN calls: {runner.total_nn_calls}")
    logger.info(f"Leaves evaluated: {runner.total_leaves_evaluated}")
    logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
