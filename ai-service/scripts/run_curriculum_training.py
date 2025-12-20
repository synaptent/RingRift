#!/usr/bin/env python3
"""
Curriculum Training Driver Script

Convenience wrapper for running curriculum training with sensible defaults.
Supports engine mixing, model pool evaluation, and progressive training.

Usage:
    # Basic curriculum training
    python scripts/run_curriculum_training.py

    # With engine mixing for data diversity
    python scripts/run_curriculum_training.py --engine-mix per_game --engine-ratio 0.5

    # With model pool evaluation
    python scripts/run_curriculum_training.py --use-model-pool --model-pool-size 3

    # Starting from an existing checkpoint
    python scripts/run_curriculum_training.py --base-model models/checkpoint.pth
"""

import argparse
import sys
from pathlib import Path

# Ensure app package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import BoardType
from app.training.curriculum import CurriculumConfig, CurriculumTrainer

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("curriculum_training")

def main():
    parser = argparse.ArgumentParser(
        description="Run curriculum training for RingRift neural network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core settings
    parser.add_argument(
        "--board-type", choices=["square8", "square19", "hexagonal", "hex8"], default="square8", help="Board type to train on"
    )
    parser.add_argument("--generations", type=int, default=10, help="Number of curriculum generations")
    parser.add_argument("--games-per-gen", type=int, default=500, help="Self-play games per generation")
    parser.add_argument("--training-epochs", type=int, default=20, help="Training epochs per generation")
    parser.add_argument("--eval-games", type=int, default=50, help="Evaluation games for promotion decision")
    parser.add_argument("--num-players", type=int, default=2, choices=[2, 3, 4], help="Number of players for self-play")

    # Engine mixing
    parser.add_argument(
        "--engine", choices=["descent", "mcts"], default="descent", help="Base engine for self-play data generation"
    )
    parser.add_argument(
        "--engine-mix", choices=["single", "per_game", "per_player"], default="single", help="Engine mixing strategy"
    )
    parser.add_argument("--engine-ratio", type=float, default=0.5, help="MCTS ratio when using engine mixing (0.0-1.0)")

    # Model pool
    parser.add_argument("--use-model-pool", action="store_true", help="Enable model pool evaluation (more robust)")
    parser.add_argument("--model-pool-size", type=int, default=5, help="Maximum models in evaluation pool")
    parser.add_argument("--pool-eval-games", type=int, default=20, help="Games per opponent in pool evaluation")

    # Training hyperparameters
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Training learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--promotion-threshold", type=float, default=0.55, help="Win rate threshold for promotion")

    # Data retention
    parser.add_argument("--data-retention", type=int, default=3, help="Generations of historical data to retain")
    parser.add_argument(
        "--historical-decay", type=float, default=0.8, help="Decay factor for historical data weighting"
    )

    # Other settings
    parser.add_argument("--base-model", type=str, default=None, help="Path to initial model checkpoint")
    parser.add_argument("--output-dir", type=str, default="curriculum_runs", help="Output directory for artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-moves", type=int, default=200, help="Maximum moves per self-play game")

    args = parser.parse_args()

    # Map board type
    board_type_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
    }

    # Build configuration
    config = CurriculumConfig(
        board_type=board_type_map[args.board_type],
        generations=args.generations,
        games_per_generation=args.games_per_gen,
        training_epochs=args.training_epochs,
        eval_games=args.eval_games,
        num_players=args.num_players,
        max_moves=args.max_moves,
        # Engine settings
        engine=args.engine,
        engine_mix=args.engine_mix,
        engine_ratio=args.engine_ratio,
        # Model pool settings
        use_model_pool=args.use_model_pool,
        model_pool_size=args.model_pool_size,
        pool_eval_games=args.pool_eval_games,
        pool_promotion_threshold=args.promotion_threshold,
        # Training settings
        promotion_threshold=args.promotion_threshold,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        data_retention=args.data_retention,
        historical_decay=args.historical_decay,
        base_seed=args.seed,
        output_dir=args.output_dir,
    )

    # Print configuration summary
    logger.info("=" * 60)
    logger.info("CURRICULUM TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"  Board type: {args.board_type}")
    logger.info(f"  Generations: {args.generations}")
    logger.info(f"  Games/generation: {args.games_per_gen}")
    logger.info(f"  Training epochs: {args.training_epochs}")
    logger.info(f"  Eval games: {args.eval_games}")
    logger.info(f"  Players: {args.num_players}")
    logger.info(f"  Engine: {args.engine}")
    logger.info(f"  Engine mix: {args.engine_mix}")
    if args.engine_mix != "single":
        logger.info(f"  Engine ratio: {args.engine_ratio}")
    logger.info(f"  Model pool: {'enabled' if args.use_model_pool else 'disabled'}")
    if args.use_model_pool:
        logger.info(f"  Pool size: {args.model_pool_size}")
    logger.info(f"  Base model: {args.base_model or 'random init'}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info("=" * 60)

    # Create trainer and run
    trainer = CurriculumTrainer(config, args.base_model)
    results = trainer.run()

    # Print summary
    print("\n" + "=" * 60)
    print("CURRICULUM TRAINING COMPLETE")
    print("=" * 60)
    promoted_count = sum(1 for r in results if r.promoted)
    print(f"Total generations: {len(results)}")
    print(f"Promotions: {promoted_count}")
    print(f"Output directory: {trainer.run_dir}")
    print()
    for r in results:
        status = "PROMOTED" if r.promoted else "skipped"
        print(f"Gen {r.generation}: {status} " f"(win={r.win_rate:.1%}, loss={r.training_loss:.4f})")


if __name__ == "__main__":
    main()
