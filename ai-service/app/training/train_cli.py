"""Command-line interface for RingRift training.

This module provides the CLI argument parsing and main entry point for training.
The actual training logic is in train.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from app.models import BoardType
from app.training.config import TrainConfig, get_model_version_for_board

logger = logging.getLogger(__name__)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        args: Optional list of argument strings. If None, uses sys.argv.
              Useful for testing.
    """
    parser = argparse.ArgumentParser(
        description='Train RingRift Neural Network AI'
    )

    # Config file (overrides individual arguments)
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to TrainingPipelineConfig YAML/JSON file. Overrides individual arguments.'
    )

    # Data and model paths
    parser.add_argument(
        '--data-path', type=str, default=None,
        help='Path to training data (.npz file)'
    )
    parser.add_argument(
        '--save-path', type=str, default=None,
        help='Path to save best model weights'
    )

    # Training configuration
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Training batch size'
    )
    parser.add_argument(
        '--auto-tune-batch-size', action='store_true',
        help='Auto-tune batch size via profiling (15-30%% faster, overrides --batch-size)'
    )
    parser.add_argument(
        '--track-calibration', action='store_true',
        help='Track value head calibration metrics during training'
    )

    # 2024-12 Hot Data Buffer and Integrated Enhancements
    parser.add_argument(
        '--use-hot-data-buffer', action='store_true',
        help='Enable hot data buffer for priority experience replay'
    )
    parser.add_argument(
        '--hot-buffer-size', type=int, default=10000,
        help='Size of hot data buffer (default: 10000)'
    )
    parser.add_argument(
        '--hot-buffer-mix-ratio', type=float, default=0.3,
        help='Ratio of samples from hot buffer vs regular data (default: 0.3)'
    )
    parser.add_argument(
        '--use-integrated-enhancements', action='store_true',
        help='Enable integrated training enhancements (curriculum, augmentation, etc.)'
    )
    parser.add_argument(
        '--enable-curriculum', action='store_true',
        help='Enable curriculum learning (progressive difficulty)'
    )
    parser.add_argument(
        '--enable-augmentation', action='store_true',
        help='Enable data augmentation (symmetry transforms)'
    )
    parser.add_argument(
        '--enable-elo-weighting', action='store_true',
        help='Enable ELO-based sample weighting'
    )
    parser.add_argument(
        '--enable-auxiliary-tasks', action='store_true',
        help='Enable auxiliary prediction tasks (outcome classification)'
    )
    parser.add_argument(
        '--enable-batch-scheduling', action='store_true',
        help='Enable dynamic batch size scheduling (linear ramp-up)'
    )
    parser.add_argument(
        '--enable-background-eval', action='store_true',
        help='Enable background Elo evaluation during training'
    )

    parser.add_argument(
        '--learning-rate', type=float, default=None,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--policy-label-smoothing', type=float, default=0.0,
        help='Policy label smoothing factor (default: 0.0)'
    )
    parser.add_argument(
        '--weight-decay', type=float, default=None,
        help='Weight decay for optimizer'
    )
    parser.add_argument(
        '--label-smoothing', type=float, default=0.0,
        help='Alias for --policy-label-smoothing'
    )
    parser.add_argument(
        '--filter-empty-policies', action='store_true',
        help='Filter out samples with empty policy targets'
    )
    parser.add_argument(
        '--feature-version', type=int, default=None,
        help='Feature encoding version (1=legacy, 2=enhanced features)'
    )

    # Early stopping
    parser.add_argument(
        '--early-stopping-patience', type=int, default=None,
        help='Number of epochs without improvement before stopping (0 to disable)'
    )

    # Checkpointing
    parser.add_argument(
        '--checkpoint-dir', type=str, default='checkpoints',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--checkpoint-interval', type=int, default=5,
        help='Epochs between checkpoints'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from (loads model, optimizer, scheduler, etc.)'
    )
    parser.add_argument(
        '--init-weights', type=str, default=None,
        help='Path to model weights for initialization (transfer learning). '
             'Only loads model weights, not optimizer state. Useful for 2p->4p transfer.'
    )
    parser.add_argument(
        '--init-weights-strict', action='store_true',
        help='Require all weights to match when using --init-weights (default: allow partial loading)'
    )
    parser.add_argument(
        '--freeze-policy', action='store_true',
        help='Freeze policy head and feature extractor, only train value head. '
             'Useful for fine-tuning value estimates without changing policy.'
    )

    # Learning rate scheduling
    parser.add_argument(
        '--warmup-epochs', type=int, default=None,
        help='Number of warmup epochs'
    )
    parser.add_argument(
        '--lr-scheduler', type=str, default=None,
        choices=['cosine', 'step', 'plateau', 'warmrestart'],
        help='Learning rate scheduler type'
    )
    parser.add_argument(
        '--lr-min', type=float, default=None,
        help='Minimum learning rate for scheduler'
    )
    parser.add_argument(
        '--lr-t0', type=int, default=10,
        help='Initial period for warm restart scheduler'
    )
    parser.add_argument(
        '--lr-t-mult', type=int, default=2,
        help='Period multiplier for warm restart scheduler'
    )

    # Board type and model configuration
    parser.add_argument(
        '--board-type', type=str, default=None,
        choices=['square8', 'square19', 'hex8', 'hexagonal'],
        help='Board type (default: square8)'
    )
    parser.add_argument(
        '--model-version', type=str, default=None,
        choices=['v2', 'v2_lite', 'v3', 'v3_lite', 'v4'],
        help='Model architecture version (for CNN models)'
    )
    parser.add_argument(
        '--model-type', type=str, default='cnn',
        choices=['cnn', 'gnn', 'hybrid'],
        help='Model type: cnn (default), gnn (Graph Neural Network), or hybrid (CNN-GNN). '
             'GNN/hybrid require PyTorch Geometric.'
    )
    parser.add_argument(
        '--num-res-blocks', type=int, default=None,
        help='Number of residual blocks (overrides default for model version)'
    )
    parser.add_argument(
        '--num-filters', type=int, default=None,
        help='Number of filters per layer (overrides default for model version)'
    )

    # Multi-player support
    parser.add_argument(
        '--multi-player', action='store_true',
        help='Enable multi-player mode (3+ players)'
    )
    parser.add_argument(
        '--num-players', type=int, default=2,
        help='Number of players for multi-player mode'
    )

    # Data augmentation
    parser.add_argument(
        '--augment-hex-symmetry', action='store_true',
        help='Apply hex symmetry augmentation during training'
    )
    parser.add_argument(
        '--sampling-weights', type=str, default='uniform',
        choices=['uniform', 'recency', 'policy_entropy', 'late_game', 'source', 'combined_source'],
        help='Sample weighting strategy. Use "source" for Gumbel 3x weight, '
             '"combined_source" for combined late_game + phase + source weighting (recommended)'
    )

    # Distributed training
    parser.add_argument(
        '--distributed', action='store_true',
        help='Enable distributed training via PyTorch DDP'
    )
    parser.add_argument(
        '--local-rank', type=int, default=-1,
        help='Local rank for distributed training'
    )
    parser.add_argument(
        '--scale-lr', action='store_true',
        help='Scale learning rate by world size'
    )
    parser.add_argument(
        '--lr-scale-mode', type=str, default='linear',
        choices=['linear', 'sqrt'],
        help='Learning rate scaling mode for distributed training'
    )
    parser.add_argument(
        '--find-unused-parameters', action='store_true',
        help='Enable find_unused_parameters for DDP'
    )

    # Fault tolerance (2025-12)
    parser.add_argument(
        '--disable-circuit-breaker', action='store_true',
        help='Disable circuit breaker for fault tolerance'
    )
    parser.add_argument(
        '--disable-anomaly-detection', action='store_true',
        help='Disable anomaly detection during training'
    )
    parser.add_argument(
        '--gradient-clip-mode', type=str, default='adaptive',
        choices=['none', 'fixed', 'adaptive'],
        help='Gradient clipping mode'
    )
    parser.add_argument(
        '--gradient-clip-max-norm', type=float, default=1.0,
        help='Maximum gradient norm for clipping'
    )
    parser.add_argument(
        '--anomaly-spike-threshold', type=float, default=3.0,
        help='Threshold for loss spike anomaly detection'
    )
    parser.add_argument(
        '--anomaly-gradient-threshold', type=float, default=100.0,
        help='Threshold for gradient anomaly detection'
    )
    parser.add_argument(
        '--disable-graceful-shutdown', action='store_true',
        help='Disable graceful shutdown handling'
    )

    # Pipeline automation (2025-12)
    parser.add_argument(
        '--enable-pipeline-auto-trigger', action='store_true',
        help='Enable automatic triggering of pipeline stages (sync → export → train → evaluate → promote). '
             'Requires bootstrap_coordination() to be called. Also respects COORDINATOR_AUTO_TRIGGER_PIPELINE env var.'
    )

    # Regularization (2025-12)
    parser.add_argument(
        '--dropout', type=float, default=0.08,
        help='Dropout rate (default: 0.08)'
    )

    # CMA-ES Heuristic Optimization (offline, expert use only)
    parser.add_argument(
        '--cmaes-heuristic', action='store_true',
        help='Run CMA-ES heuristic weight optimization (offline mode)'
    )
    parser.add_argument(
        '--cmaes-tier-id', type=str, default='sq8_heuristic_baseline_v1',
        help='Tier ID for CMA-ES heuristic optimization'
    )
    parser.add_argument(
        '--cmaes-base-profile-id', type=str, default='baseline_v1',
        help='Base heuristic profile ID for CMA-ES optimization'
    )
    parser.add_argument(
        '--cmaes-generations', type=int, default=20,
        help='Number of CMA-ES generations'
    )
    parser.add_argument(
        '--cmaes-population-size', type=int, default=16,
        help='CMA-ES population size'
    )
    parser.add_argument(
        '--cmaes-seed', type=int, default=42,
        help='Random seed for CMA-ES optimization'
    )
    parser.add_argument(
        '--cmaes-games-per-candidate', type=int, default=50,
        help='Games per candidate for fitness evaluation'
    )

    # Curriculum training mode (2025-12)
    parser.add_argument(
        '--curriculum', action='store_true',
        help='Run curriculum training (iterative self-play with promotion)'
    )
    parser.add_argument(
        '--curriculum-generations', type=int, default=10,
        help='Number of curriculum generations'
    )
    parser.add_argument(
        '--curriculum-games-per-gen', type=int, default=1000,
        help='Games to generate per generation'
    )
    parser.add_argument(
        '--curriculum-training-epochs', type=int, default=10,
        help='Training epochs per generation'
    )
    parser.add_argument(
        '--curriculum-eval-games', type=int, default=100,
        help='Evaluation games for promotion decision'
    )
    parser.add_argument(
        '--curriculum-promotion-threshold', type=float, default=0.55,
        help='Win rate threshold for promotion'
    )
    parser.add_argument(
        '--curriculum-data-retention', type=int, default=3,
        help='Number of past generations to retain data from'
    )
    parser.add_argument(
        '--curriculum-num-players', type=int, default=2,
        help='Number of players for curriculum games'
    )
    parser.add_argument(
        '--curriculum-engine', type=str, default='descent',
        choices=['descent', 'mcts', 'heuristic', 'neural', 'mix'],
        help='Engine type for self-play generation'
    )
    parser.add_argument(
        '--curriculum-engine-mix', type=str, default=None,
        help='Comma-separated engine:weight pairs for mix mode (e.g., "descent:0.5,mcts:0.3,heuristic:0.2")'
    )
    parser.add_argument(
        '--curriculum-engine-ratio', type=float, default=0.8,
        help='Ratio of games using primary engine (rest use random opponent)'
    )
    parser.add_argument(
        '--curriculum-output-dir', type=str, default='curriculum_runs',
        help='Output directory for curriculum artifacts (default: curriculum_runs)',
    )
    parser.add_argument(
        '--curriculum-base-model',
        type=str,
        default=None,
        help='Path to initial model checkpoint for curriculum training',
    )

    return parser.parse_args(args)


def main() -> None:
    """Main entry point for training."""
    # Import here to avoid circular imports and reduce startup time
    from app.training.train import run_cmaes_heuristic_optimization, train_model

    args = parse_args()

    # Initialize pipeline auto-trigger if requested (2025-12)
    # This wires up the coordination infrastructure for automatic stage progression
    if getattr(args, 'enable_pipeline_auto_trigger', False):
        try:
            from app.coordination.coordination_bootstrap import bootstrap_coordination
            bootstrap_result = bootstrap_coordination(pipeline_auto_trigger=True)
            logger.info(
                f"[TrainCLI] Pipeline auto-trigger enabled. "
                f"Initialized {len([v for v in bootstrap_result.values() if v])} coordinators."
            )
        except ImportError:
            logger.warning(
                "[TrainCLI] --enable-pipeline-auto-trigger requires app.coordination module. "
                "Continuing without pipeline automation."
            )
        except Exception as e:
            logger.warning(f"[TrainCLI] Failed to enable pipeline auto-trigger: {e}")

    # Load config file if provided (overrides individual arguments)
    if args.config:
        from app.training.config import TrainingPipelineConfig
        try:
            pipeline_config = TrainingPipelineConfig.load(args.config)
            logger.info(f"Loaded config from {args.config}")

            # Apply config values to args (config takes precedence if args not set)
            if args.data_path is None:
                args.data_path = pipeline_config.data.data_dir
            if args.epochs is None:
                args.epochs = pipeline_config.train.epochs_per_iter
            if args.batch_size is None:
                args.batch_size = pipeline_config.train.batch_size
            if args.learning_rate is None:
                args.learning_rate = pipeline_config.train.learning_rate
            if args.checkpoint_dir is None or args.checkpoint_dir == 'checkpoints':
                args.checkpoint_dir = pipeline_config.checkpoint.checkpoint_dir
            if args.board_type is None:
                args.board_type = pipeline_config.train.board_type.value

            # Log config summary
            logger.info(f"  Board type: {pipeline_config.train.board_type.value}")
            logger.info(f"  Learning rate: {pipeline_config.train.learning_rate}")
            logger.info(f"  Batch size: {pipeline_config.train.batch_size}")

        except Exception as e:
            logger.error(f"Failed to load config from {args.config}: {e}")
            raise

    # Curriculum training mode: iterative self-play with model promotion
    if getattr(args, "curriculum", False):
        from app.training.curriculum import CurriculumConfig, CurriculumTrainer

        # Map board type string to enum
        board_type_map = {
            'square8': BoardType.SQUARE8,
            'square19': BoardType.SQUARE19,
            'hex8': BoardType.HEX8,
            'hexagonal': BoardType.HEXAGONAL,
        }
        board_type = board_type_map.get(
            args.board_type or 'square8',
            BoardType.SQUARE8,
        )

        config = CurriculumConfig(
            board_type=board_type,
            generations=args.curriculum_generations,
            games_per_generation=args.curriculum_games_per_gen,
            training_epochs=args.curriculum_training_epochs,
            eval_games=args.curriculum_eval_games,
            promotion_threshold=args.curriculum_promotion_threshold,
            data_retention=args.curriculum_data_retention,
            num_players=args.curriculum_num_players,
            # Learning rate from standard args if provided
            learning_rate=args.learning_rate or 1e-3,
            batch_size=args.batch_size or 32,
            base_seed=args.seed or 42,
            output_dir=args.curriculum_output_dir,
            # Engine configuration
            engine=args.curriculum_engine,
            engine_mix=args.curriculum_engine_mix,
            engine_ratio=args.curriculum_engine_ratio,
        )

        trainer = CurriculumTrainer(config, args.curriculum_base_model)
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
            print(
                f"Gen {r.generation}: {status} (win={r.win_rate:.1%}, "
                f"loss={r.training_loss:.4f})"
            )
        return

    # Offline heuristic CMA-ES optimisation mode is explicitly opt-in and
    # does not affect the neural-network training path.
    if getattr(args, "cmaes_heuristic", False):
        report = run_cmaes_heuristic_optimization(
            tier_id=args.cmaes_tier_id,
            base_profile_id=args.cmaes_base_profile_id,
            generations=args.cmaes_generations,
            population_size=args.cmaes_population_size,
            rng_seed=args.cmaes_seed,
            games_per_candidate=args.cmaes_games_per_candidate,
        )
        out_dir = Path("results") / "ai_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"cmaes_heuristic_square8_{ts}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote CMA-ES heuristic report to {out_path}")
        return

    # Create config
    config = TrainConfig()

    # Override config from CLI args
    if args.epochs is not None:
        config.epochs_per_iter = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.seed is not None:
        config.seed = args.seed
    if args.policy_label_smoothing > 0:
        config.policy_label_smoothing = args.policy_label_smoothing
    if hasattr(args, 'weight_decay') and args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if hasattr(args, 'label_smoothing') and args.label_smoothing > 0:
        config.policy_label_smoothing = args.label_smoothing
    if args.feature_version is not None:
        config.feature_version = args.feature_version
    if args.filter_empty_policies:
        config.allow_empty_policies = False
    if args.board_type is not None:
        board_type_map = {
            'square8': BoardType.SQUARE8,
            'square19': BoardType.SQUARE19,
            'hex8': BoardType.HEX8,
            'hexagonal': BoardType.HEXAGONAL,
        }
        config.board_type = board_type_map[args.board_type]

    # Determine paths
    data_path = args.data_path or os.path.join(config.data_dir, "dataset.npz")
    save_path = args.save_path or os.path.join(
        config.model_dir,
        f"{config.model_id}.pth",
    )
    # Board-aware default model version (centralized in config.py)
    model_version = args.model_version
    if model_version is None:
        model_version = get_model_version_for_board(config.board_type)

    # Route to GNN/hybrid training if model-type specified (2025-12)
    model_type = getattr(args, 'model_type', 'cnn')
    if model_type in ('gnn', 'hybrid'):
        import subprocess
        import sys

        logger.info(f"[TrainCLI] Routing to GNN training pipeline (model_type={model_type})")

        board_type_str = config.board_type.value if hasattr(config.board_type, 'value') else str(config.board_type)

        # Determine output directory from save_path
        save_dir = str(Path(save_path).parent) if save_path else f"models/gnn_{board_type_str}_{args.num_players}p"

        # Build command for GNN training script
        cmd = [
            sys.executable, "-m", "app.training.train_gnn_policy",
            "--data-path", data_path,
            "--board-type", board_type_str,
            "--output-dir", save_dir,
            "--epochs", str(config.epochs_per_iter),
            "--batch-size", str(config.batch_size),
            "--lr", str(config.learning_rate),
        ]

        logger.info(f"[TrainCLI] Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            logger.error(f"[TrainCLI] GNN training failed with exit code {result.returncode}")
            sys.exit(result.returncode)

        logger.info("[TrainCLI] GNN training completed successfully")
        return

    # Run CNN training
    train_model(
        config=config,
        data_path=data_path,
        save_path=save_path,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        warmup_epochs=args.warmup_epochs,
        lr_scheduler=args.lr_scheduler,
        lr_min=args.lr_min,
        lr_t0=args.lr_t0,
        lr_t_mult=args.lr_t_mult,
        resume_path=args.resume,
        init_weights_path=getattr(args, 'init_weights', None),
        init_weights_strict=getattr(args, 'init_weights_strict', False),
        freeze_policy=getattr(args, 'freeze_policy', False),
        augment_hex_symmetry=args.augment_hex_symmetry,
        distributed=args.distributed,
        local_rank=args.local_rank,
        scale_lr=args.scale_lr,
        lr_scale_mode=args.lr_scale_mode,
        find_unused_parameters=args.find_unused_parameters,
        sampling_weights=args.sampling_weights,
        multi_player=args.multi_player,
        num_players=args.num_players,
        model_version=model_version,
        num_res_blocks=getattr(args, 'num_res_blocks', None),
        num_filters=getattr(args, 'num_filters', None),
        auto_tune_batch_size=getattr(args, 'auto_tune_batch_size', False),
        track_calibration=getattr(args, 'track_calibration', False),
        # 2024-12 Hot Data Buffer and Integrated Enhancements
        use_hot_data_buffer=getattr(args, 'use_hot_data_buffer', False),
        hot_buffer_size=getattr(args, 'hot_buffer_size', 10000),
        hot_buffer_mix_ratio=getattr(args, 'hot_buffer_mix_ratio', 0.3),
        use_integrated_enhancements=getattr(args, 'use_integrated_enhancements', False),
        enable_curriculum=getattr(args, 'enable_curriculum', False),
        enable_augmentation=getattr(args, 'enable_augmentation', False),
        enable_elo_weighting=getattr(args, 'enable_elo_weighting', False),
        enable_auxiliary_tasks=getattr(args, 'enable_auxiliary_tasks', False),
        enable_batch_scheduling=getattr(args, 'enable_batch_scheduling', False),
        enable_background_eval=getattr(args, 'enable_background_eval', False),
        # Fault tolerance (2025-12)
        enable_circuit_breaker=not getattr(args, 'disable_circuit_breaker', False),
        enable_anomaly_detection=not getattr(args, 'disable_anomaly_detection', False),
        gradient_clip_mode=getattr(args, 'gradient_clip_mode', 'adaptive'),
        gradient_clip_max_norm=getattr(args, 'gradient_clip_max_norm', 1.0),
        anomaly_spike_threshold=getattr(args, 'anomaly_spike_threshold', 3.0),
        anomaly_gradient_threshold=getattr(args, 'anomaly_gradient_threshold', 100.0),
        enable_graceful_shutdown=not getattr(args, 'disable_graceful_shutdown', False),
        # Regularization (2025-12)
        dropout=getattr(args, 'dropout', 0.08),
        # GNN support (2025-12)
        model_type=getattr(args, 'model_type', 'cnn'),
    )


if __name__ == "__main__":
    main()
