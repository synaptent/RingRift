#!/usr/bin/env python3
"""
Descent vs MCTS Training Experiment

End-to-end script that:
1. Generates self-play data with Descent AI (soft policies from TT values)
2. Generates self-play data with MCTS AI (soft policies from visit counts)
3. Trains a neural network on each dataset
4. Evaluates both trained models against baseline and each other

Usage:
    python scripts/run_descent_vs_mcts_experiment.py \
        --games 500 \
        --epochs 20 \
        --eval-games 100 \
        --output-dir experiments/descent_vs_mcts
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure app package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import BoardType
from app.training.generate_data import generate_dataset
from app.training.train import train_from_file
from app.training.config import TrainConfig
from app.training.env import get_theoretical_max_moves

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("descent_vs_mcts")

def run_experiment(
    num_games: int = 500,
    epochs: int = 20,
    eval_games: int = 100,
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
    output_dir: str = "experiments/descent_vs_mcts",
    seed: int = 42,
    think_time: int = 500,
    max_moves: int = None,  # Auto-calculated if not specified
) -> dict:
    """
    Run the full Descent vs MCTS experiment.

    Returns:
        Dict with experiment results and file paths.
    """
    # Auto-calculate max_moves if not specified
    if max_moves is None:
        board_str = board_type.value if isinstance(board_type, BoardType) else str(board_type)
        max_moves = get_theoretical_max_moves(board_str, num_players)
        logger.info(f"[Auto] max_moves={max_moves} for {board_str} {num_players}p")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": timestamp,
        "config": {
            "num_games": num_games,
            "epochs": epochs,
            "eval_games": eval_games,
            "board_type": board_type.value,
            "num_players": num_players,
            "seed": seed,
            "think_time": think_time,
        },
        "descent": {},
        "mcts": {},
        "comparison": {},
    }

    # Save config
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(results["config"], f, indent=2)

    # =========================================================================
    # Phase 1: Generate data with Descent AI
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: Generating Descent AI self-play data")
    logger.info("=" * 60)

    descent_data_path = run_dir / "descent_data.npz"
    t0 = time.time()

    generate_dataset(
        num_games=num_games,
        output_file=str(descent_data_path),
        board_type=board_type,
        seed=seed,
        max_moves=max_moves,
        num_players=num_players,
        engine="descent",
        engine_mix="single",
    )

    descent_data_time = time.time() - t0
    results["descent"]["data_path"] = str(descent_data_path)
    results["descent"]["data_time_sec"] = descent_data_time
    logger.info(f"Descent data generated in {descent_data_time:.1f}s")

    # =========================================================================
    # Phase 2: Generate data with MCTS AI
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 2: Generating MCTS AI self-play data")
    logger.info("=" * 60)

    mcts_data_path = run_dir / "mcts_data.npz"
    t0 = time.time()

    generate_dataset(
        num_games=num_games,
        output_file=str(mcts_data_path),
        board_type=board_type,
        seed=seed + 10000,  # Different seed for variety
        max_moves=max_moves,
        num_players=num_players,
        engine="mcts",
        engine_mix="single",
    )

    mcts_data_time = time.time() - t0
    results["mcts"]["data_path"] = str(mcts_data_path)
    results["mcts"]["data_time_sec"] = mcts_data_time
    logger.info(f"MCTS data generated in {mcts_data_time:.1f}s")

    # =========================================================================
    # Phase 3: Train model on Descent data
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 3: Training model on Descent data")
    logger.info("=" * 60)

    descent_model_path = run_dir / "descent_model.pth"
    t0 = time.time()

    descent_config = TrainConfig(
        board_type=board_type,
        epochs_per_iter=epochs,
        learning_rate=1e-3,
        batch_size=32,
        seed=seed,
        model_id="descent_trained",
    )

    descent_losses = train_from_file(
        data_path=str(descent_data_path),
        output_path=str(descent_model_path),
        config=descent_config,
    )

    descent_train_time = time.time() - t0
    results["descent"]["model_path"] = str(descent_model_path)
    results["descent"]["train_time_sec"] = descent_train_time
    results["descent"]["final_loss"] = descent_losses.get("total", 0)
    logger.info(f"Descent model trained in {descent_train_time:.1f}s " f"(loss={descent_losses.get('total', 0):.4f})")

    # =========================================================================
    # Phase 4: Train model on MCTS data
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 4: Training model on MCTS data")
    logger.info("=" * 60)

    mcts_model_path = run_dir / "mcts_model.pth"
    t0 = time.time()

    mcts_config = TrainConfig(
        board_type=board_type,
        epochs_per_iter=epochs,
        learning_rate=1e-3,
        batch_size=32,
        seed=seed + 1,
        model_id="mcts_trained",
    )

    mcts_losses = train_from_file(
        data_path=str(mcts_data_path),
        output_path=str(mcts_model_path),
        config=mcts_config,
    )

    mcts_train_time = time.time() - t0
    results["mcts"]["model_path"] = str(mcts_model_path)
    results["mcts"]["train_time_sec"] = mcts_train_time
    results["mcts"]["final_loss"] = mcts_losses.get("total", 0)
    logger.info(f"MCTS model trained in {mcts_train_time:.1f}s " f"(loss={mcts_losses.get('total', 0):.4f})")

    # =========================================================================
    # Phase 5: Evaluate models
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 5: Evaluating models")
    logger.info("=" * 60)

    try:
        from scripts.evaluate_ai_models import run_evaluation

        # Descent vs baseline (heuristic)
        logger.info("Evaluating Descent-trained model vs Heuristic baseline...")
        descent_vs_heuristic = run_evaluation(
            player1_type="neural_network",
            player2_type="heuristic",
            num_games=eval_games,
            board_type=board_type,
            seed=seed,
            checkpoint_path=str(descent_model_path),
            checkpoint_path2=None,
            cmaes_weights_path=None,
            minimax_depth=3,
            max_moves_per_game=max_moves,
            verbose=False,
        )
        results["descent"]["vs_heuristic"] = {
            "win_rate": descent_vs_heuristic.p1_wins / max(1, eval_games),
            "wins": descent_vs_heuristic.p1_wins,
            "losses": descent_vs_heuristic.p2_wins,
            "draws": descent_vs_heuristic.draws,
        }

        # MCTS vs baseline (heuristic)
        logger.info("Evaluating MCTS-trained model vs Heuristic baseline...")
        mcts_vs_heuristic = run_evaluation(
            player1_type="neural_network",
            player2_type="heuristic",
            num_games=eval_games,
            board_type=board_type,
            seed=seed,
            checkpoint_path=str(mcts_model_path),
            checkpoint_path2=None,
            cmaes_weights_path=None,
            minimax_depth=3,
            max_moves_per_game=max_moves,
            verbose=False,
        )
        results["mcts"]["vs_heuristic"] = {
            "win_rate": mcts_vs_heuristic.p1_wins / max(1, eval_games),
            "wins": mcts_vs_heuristic.p1_wins,
            "losses": mcts_vs_heuristic.p2_wins,
            "draws": mcts_vs_heuristic.draws,
        }

        # Descent vs MCTS (head-to-head)
        logger.info("Evaluating Descent-trained vs MCTS-trained (head-to-head)...")
        descent_vs_mcts = run_evaluation(
            player1_type="neural_network",
            player2_type="neural_network",
            num_games=eval_games,
            board_type=board_type,
            seed=seed,
            checkpoint_path=str(descent_model_path),
            checkpoint_path2=str(mcts_model_path),
            cmaes_weights_path=None,
            minimax_depth=3,
            max_moves_per_game=max_moves,
            verbose=False,
        )
        results["comparison"]["descent_vs_mcts"] = {
            "descent_wins": descent_vs_mcts.p1_wins,
            "mcts_wins": descent_vs_mcts.p2_wins,
            "draws": descent_vs_mcts.draws,
            "descent_win_rate": descent_vs_mcts.p1_wins / max(1, eval_games),
        }

    except ImportError as e:
        logger.warning(f"Could not run evaluation: {e}")
        results["evaluation_error"] = str(e)

    # =========================================================================
    # Save final results
    # =========================================================================
    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {run_dir}")
    logger.info("")
    logger.info("Summary:")
    logger.info(f"  Descent model loss: {results['descent'].get('final_loss', 'N/A'):.4f}")
    logger.info(f"  MCTS model loss: {results['mcts'].get('final_loss', 'N/A'):.4f}")

    if "vs_heuristic" in results["descent"]:
        logger.info(f"  Descent vs Heuristic: {results['descent']['vs_heuristic']['win_rate']:.1%}")
    if "vs_heuristic" in results["mcts"]:
        logger.info(f"  MCTS vs Heuristic: {results['mcts']['vs_heuristic']['win_rate']:.1%}")
    if "descent_vs_mcts" in results["comparison"]:
        logger.info(f"  Descent vs MCTS: {results['comparison']['descent_vs_mcts']['descent_win_rate']:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Descent vs MCTS training experiment")
    parser.add_argument("--games", type=int, default=500, help="Number of self-play games per engine (default: 500)")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per model (default: 20)")
    parser.add_argument("--eval-games", type=int, default=100, help="Evaluation games per matchup (default: 100)")
    parser.add_argument(
        "--board-type",
        choices=["square8", "square19", "hexagonal"],
        default="square8",
        help="Board type (default: square8)",
    )
    parser.add_argument("--num-players", type=int, default=2, choices=[2, 3, 4], help="Number of players (default: 2)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/descent_vs_mcts",
        help="Output directory (default: experiments/descent_vs_mcts)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    board_type_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
    }

    run_experiment(
        num_games=args.games,
        epochs=args.epochs,
        eval_games=args.eval_games,
        board_type=board_type_map[args.board_type],
        num_players=args.num_players,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
