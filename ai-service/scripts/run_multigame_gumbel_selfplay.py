#!/usr/bin/env python3
"""Multi-game batched Gumbel MCTS selfplay for high-throughput training data.

Runs 64+ games in parallel with synchronized Sequential Halving and batched NN
evaluation via MultiGameGumbelRunner, producing 10-20x speedup over sequential
generate_gumbel_selfplay.py on GPU nodes.

Output is saved to GameReplayDB (SQLite) with MCTS policy distributions,
compatible with the existing training pipeline.

Usage:
    # Run 256 games in batches of 64 on square8 2-player
    python scripts/run_multigame_gumbel_selfplay.py \
        --board square8 --num-players 2 --num-games 256 \
        --batch-size 64 --simulation-budget 800 \
        --db data/games/canonical_square8_2p.db

    # With explicit model path
    python scripts/run_multigame_gumbel_selfplay.py \
        --board hex8 --num-players 2 --num-games 128 \
        --model models/canonical_hex8_2p.pth \
        --db data/games/canonical_hex8_2p.db

February 2026: Initial implementation for GH200 cluster acceleration.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import uuid
from pathlib import Path

# Disable torch.compile on GH200 nodes (driver incompatibility)
if not os.environ.get("RINGRIFT_DISABLE_TORCH_COMPILE"):
    os.environ["RINGRIFT_DISABLE_TORCH_COMPILE"] = "1"

# Ensure app imports resolve
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models import AIConfig, BoardType, Move

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_board_type(board_str: str) -> BoardType:
    """Parse board type string to enum."""
    board_str = board_str.lower()
    if "square8" in board_str or "sq8" in board_str:
        return BoardType.SQUARE8
    elif "square19" in board_str or "sq19" in board_str:
        return BoardType.SQUARE19
    elif "hex8" in board_str:
        return BoardType.HEX8
    elif "hex" in board_str:
        return BoardType.HEXAGONAL
    return BoardType.SQUARE8


def resolve_model_path(
    board_type: str, num_players: int, model_id: str, model_version: str
) -> str | None:
    """Resolve model path from explicit ID or auto-discovery."""
    if model_id:
        if os.path.exists(model_id):
            return model_id
        logger.warning(f"Model path not found: {model_id}")
        return None

    # Try version-specific model selector
    try:
        from app.training.selfplay_model_selector import get_model_for_config

        resolved = get_model_for_config(
            board_type, num_players, prefer_nnue=False, model_version=model_version
        )
        if resolved:
            logger.info(f"Resolved model: {resolved}")
            return str(resolved)
    except ImportError:
        pass

    # Fallback to canonical model path
    canonical = PROJECT_ROOT / "models" / f"canonical_{board_type}_{num_players}p.pth"
    if canonical.exists():
        logger.info(f"Using canonical model: {canonical}")
        return str(canonical)

    logger.warning(f"No model found for {board_type}_{num_players}p")
    return None


def create_neural_net(
    model_path: str, board_type: BoardType, allow_fresh: bool = False
):
    """Load a NeuralNetAI for batch evaluation."""
    from app.ai.neural_net import NeuralNetAI

    ai_config = AIConfig(
        difficulty=8,
        use_neural_net=True,
        nn_model_id=model_path,
        allow_fresh_weights=allow_fresh,
    )

    return NeuralNetAI(
        player_number=1,
        config=ai_config,
        board_type=board_type,
    )


def save_results_to_db(results, db_path: str, board_type: str, num_players: int):
    """Save MultiGameGumbelRunner results to GameReplayDB.

    Converts GumbelGameResult objects to GameReplayDB records with MCTS
    policy distributions (move_probs) for training.
    """
    from app.db.game_replay import GameReplayDB
    from app.db.unified_recording import GameRecorder
    from app.game_engine import GameEngine
    from app.models import Move as MoveModel
    from app.rules.serialization import deserialize_game_state

    db = GameReplayDB(db_path, enforce_canonical_history=False)
    saved = 0

    for result in results:
        if not result.initial_state or not result.moves:
            continue

        game_id = f"mgumbel_{uuid.uuid4().hex[:12]}"

        try:
            # Reconstruct initial GameState from serialized dict
            initial_state = deserialize_game_state(result.initial_state)
        except Exception as e:
            logger.warning(f"Failed to deserialize initial state: {e}")
            continue

        metadata = {
            "source": "multigame_gumbel_selfplay",
            "board_type": board_type,
            "num_players": num_players,
            "winner": result.winner,
            "engine_mode": "multigame-gumbel",
            "move_count": result.move_count,
        }

        try:
            with GameRecorder(db, initial_state, game_id) as recorder:
                state = initial_state
                for idx, move_data in enumerate(result.moves):
                    # Extract MCTS policy
                    move_probs = move_data.get("mcts_policy")

                    # Create Move object (exclude extra fields)
                    move_dict = {
                        k: v
                        for k, v in move_data.items()
                        if k not in ("mcts_policy", "value", "moveNumber", "search_stats")
                    }
                    move_dict.setdefault("id", f"{game_id}:{idx}")
                    move = MoveModel.model_validate(move_dict)

                    # Apply move to get state_after
                    state_after = GameEngine.apply_move(state, move, trace_mode=True)

                    recorder.add_move(
                        move,
                        state_after=state_after,
                        state_before=state,
                        move_probs=move_probs,
                    )
                    state = state_after

                recorder.finalize(state, metadata)
            saved += 1

        except Exception as e:
            logger.warning(f"Failed to save game {game_id}: {e}")
            continue

    return saved


def run_multigame_selfplay(args):
    """Main selfplay loop using MultiGameGumbelRunner."""
    from app.ai.multi_game_gumbel import MultiGameGumbelRunner

    board_type_enum = parse_board_type(args.board)

    # Resolve model
    model_path = resolve_model_path(
        args.board, args.num_players, args.model_id, args.model_version
    )

    # Load neural network
    neural_net = None
    if model_path:
        try:
            neural_net = create_neural_net(
                model_path, board_type_enum, args.allow_fresh_weights
            )
            logger.info(f"Loaded NN from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load NN: {e}")
            if not args.allow_fresh_weights:
                logger.error("No neural net available and --allow-fresh-weights not set")
                return
    elif args.allow_fresh_weights:
        logger.info("No model found, using uniform policy (fresh weights)")
    else:
        logger.error(
            f"No model found for {args.board}_{args.num_players}p. "
            "Use --allow-fresh-weights to run without a trained model."
        )
        return

    # Determine device
    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
    except ImportError:
        pass

    logger.info(
        f"Starting multigame Gumbel selfplay: {args.board} {args.num_players}P, "
        f"{args.num_games} games, batch={args.batch_size}, "
        f"budget={args.simulation_budget}, device={device}"
    )

    # Create runner
    runner = MultiGameGumbelRunner(
        num_games=args.batch_size,
        simulation_budget=args.simulation_budget,
        num_sampled_actions=args.num_sampled_actions,
        board_type=board_type_enum,
        num_players=args.num_players,
        neural_net=neural_net,
        device=device,
        max_moves_per_game=args.max_moves or 500,
        temperature=args.temperature,
        temperature_threshold=args.temperature_threshold,
    )

    # Run in batches
    all_results = []
    total_start = time.time()
    games_remaining = args.num_games
    batch_num = 0

    while games_remaining > 0:
        batch_count = min(args.batch_size, games_remaining)
        batch_num += 1

        logger.info(
            f"Batch {batch_num}: running {batch_count} games "
            f"({args.num_games - games_remaining}/{args.num_games} done)"
        )

        batch_start = time.time()
        results = runner.run_batch(num_games=batch_count)
        batch_elapsed = time.time() - batch_start

        completed = sum(1 for r in results if r.status == "completed")
        avg_moves = (
            sum(r.move_count for r in results) / len(results) if results else 0
        )

        logger.info(
            f"Batch {batch_num}: {completed}/{batch_count} completed, "
            f"avg moves={avg_moves:.1f}, {batch_elapsed:.1f}s "
            f"({batch_count / batch_elapsed:.1f} games/s)"
        )

        all_results.extend(results)
        games_remaining -= batch_count

    total_elapsed = time.time() - total_start

    # Save to database
    if args.db:
        os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
        saved = save_results_to_db(
            all_results, args.db, args.board, args.num_players
        )
        logger.info(f"Saved {saved}/{len(all_results)} games to {args.db}")

    # Summary
    total_completed = sum(1 for r in all_results if r.status == "completed")
    total_moves = sum(r.move_count for r in all_results)
    winners = [r.winner for r in all_results if r.winner is not None]
    win_dist = {
        p: winners.count(p) for p in range(1, args.num_players + 1)
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"MULTIGAME GUMBEL SELFPLAY SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Config: {args.board} {args.num_players}P")
    logger.info(f"  Games: {total_completed}/{len(all_results)} completed")
    logger.info(f"  Total moves: {total_moves}")
    logger.info(
        f"  Avg moves/game: {total_moves / max(1, len(all_results)):.1f}"
    )
    logger.info(f"  Duration: {total_elapsed:.1f}s")
    logger.info(
        f"  Throughput: {len(all_results) / total_elapsed:.1f} games/s"
    )
    logger.info(f"  Win distribution: {win_dist}")
    logger.info(
        f"  NN calls: {runner.total_nn_calls}, "
        f"leaves evaluated: {runner.total_leaves_evaluated}"
    )

    # Emit completion event for pipeline coordination
    try:
        import asyncio
        import socket

        from app.coordination.selfplay_orchestrator import emit_selfplay_completion

        node_id = socket.gethostname()
        task_id = f"mgumbel_{args.board}_{args.num_players}p_{int(time.time())}"

        async def emit():
            return await emit_selfplay_completion(
                task_id=task_id,
                board_type=args.board,
                num_players=args.num_players,
                games_generated=total_completed,
                success=True,
                node_id=node_id,
                selfplay_type="multigame_gumbel",
            )

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(emit())
        except RuntimeError:
            asyncio.run(emit())
    except (ImportError, Exception):
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Multi-game batched Gumbel MCTS selfplay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--board", "-b", type=str, default="square8",
        choices=["square8", "square19", "hex8", "hexagonal"],
        help="Board type (default: square8)",
    )
    parser.add_argument(
        "--num-players", "-p", type=int, default=2, choices=[2, 3, 4],
        help="Number of players (default: 2)",
    )
    parser.add_argument(
        "--num-games", "-n", type=int, default=256,
        help="Total number of games to generate (default: 256)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Games per parallel batch (default: 64)",
    )
    parser.add_argument(
        "--simulation-budget", type=int, default=800,
        help="Gumbel MCTS simulations per move (default: 800)",
    )
    parser.add_argument(
        "--num-sampled-actions", type=int, default=16,
        help="Gumbel-Top-K actions to sample (default: 16)",
    )
    parser.add_argument(
        "--db", type=str, default="",
        help="Output GameReplayDB path (SQLite)",
    )
    parser.add_argument(
        "--model-id", type=str, default="",
        help="Neural network model path or ID",
    )
    parser.add_argument(
        "--model-version", type=str, default="v2",
        help="Architecture version (v2, v4, v5-heavy) for model selection",
    )
    parser.add_argument(
        "--max-moves", type=int, default=0,
        help="Max moves per game (0 = auto, default: 0)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Policy temperature for exploration (default: 1.0)",
    )
    parser.add_argument(
        "--temperature-threshold", type=int, default=30,
        help="Move after which to reduce temperature (default: 30)",
    )
    parser.add_argument(
        "--allow-fresh-weights", action="store_true",
        help="Allow running without a trained model checkpoint",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed (0 = random)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.seed:
        import numpy as np

        np.random.seed(args.seed)

    run_multigame_selfplay(args)


if __name__ == "__main__":
    main()
