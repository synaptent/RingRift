"""Training Loop - Self-play, training, and evaluation iteration.

This module provides the main training loop that iterates between:
1. **Self-play**: Generate games using current best model
2. **Training**: Train on accumulated game data
3. **Evaluation**: Test against baselines and promote if better

The loop integrates with the OptimizedTrainingPipeline (when available) for:
- Export caching to avoid redundant NPZ generation
- Curriculum feedback to adjust data selection
- Health monitoring for pipeline state
- Distributed locks for cluster coordination

Usage:
    from app.training.train_loop import run_training_loop
    from app.training.config import TrainConfig

    # Basic usage with defaults
    run_training_loop()

    # Custom configuration
    config = TrainConfig(
        board_type="hex8",
        num_players=2,
        epochs_per_iteration=20,
        games_per_iteration=500,
    )
    run_training_loop(config=config)

Integration:
    - Called by scripts/run_training_loop.py CLI
    - Emits training events for feedback loop
    - Supports GPU-parallel selfplay via generate_dataset_gpu_parallel

See Also:
    - scripts/run_training_loop.py for CLI wrapper
    - app.training.optimized_pipeline for enhanced pipeline features
    - app.training.train for core training logic
"""

import json
import os
import shutil

from app.ai.descent_ai import DescentAI
from app.ai.heuristic_weights import HEURISTIC_WEIGHT_PROFILES
from app.config.thresholds import WIN_RATE_BEAT_BEST
from app.models import AIConfig
from app.training.config import TrainConfig, get_model_version_for_board
from app.training.generate_data import generate_dataset, generate_dataset_gpu_parallel
from app.training.tournament import Tournament
from app.training.train import train_model

# Optional integration with OptimizedTrainingPipeline (December 2025)
# Provides: export caching, curriculum feedback, health monitoring, distributed locks
try:
    from app.training.optimized_pipeline import get_optimized_pipeline
    HAS_OPTIMIZED_PIPELINE = True
except ImportError:
    HAS_OPTIMIZED_PIPELINE = False
    get_optimized_pipeline = None


def export_heuristic_profiles(log_dir: str) -> None:
    """
    Export the current heuristic weight profiles to a JSON file in the
    configured training log directory. This keeps heuristic tuning as an
    offline concern without changing runtime defaults.
    """
    try:
        base_dir = os.getcwd()
        out_path = os.path.join(
            base_dir,
            log_dir,
            "heuristic_profiles.v1.json",
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(HEURISTIC_WEIGHT_PROFILES, f, indent=2, sort_keys=True)
    except Exception as exc:
        print("Warning: failed to export heuristic profiles:", str(exc))


def run_training_loop(config: TrainConfig | None = None, use_optimized_pipeline: bool = True):
    """Run the training loop with self-play, training, and evaluation.

    Args:
        config: Training configuration. If None, uses defaults.
        use_optimized_pipeline: If True and available, uses OptimizedTrainingPipeline
            for enhanced features (export caching, health monitoring, distributed locks).
    """
    if config is None:
        config = TrainConfig()

    # Use optimized pipeline if available and requested
    if use_optimized_pipeline and HAS_OPTIMIZED_PIPELINE:
        return _run_training_loop_optimized(config)

    print(
        f"Starting training loop: {config.epochs_per_iter} iterations, "
        f"{config.episodes_per_iter} games/iter"
    )

    # Use absolute paths
    base_dir = os.getcwd()
    data_file = os.path.join(
        base_dir,
        config.data_dir,
        "self_play_data.npz",
    )
    model_file = os.path.join(
        base_dir,
        config.model_dir,
        f"{config.model_id}.pth",
    )
    best_model_file = os.path.join(
        base_dir,
        config.model_dir,
        f"{config.model_id}_best.pth",
    )

    # Initialize best model if not exists
    if not os.path.exists(best_model_file) and os.path.exists(model_file):
        shutil.copy(model_file, best_model_file)

    # Use the configured number of training iterations so callers can control
    # how many self-play / training / evaluation cycles are run.
    num_loops = config.iterations

    for i in range(num_loops):
        print(f"\n=== Iteration {i+1}/{num_loops} ===")

        # 1. Self-Play Data Generation
        # Choose between GPU parallel (fast, heuristic-based) or CPU sequential
        # (slower but uses tree search via DescentAI for higher quality)
        if config.use_gpu_parallel_datagen:
            print("Generating self-play data (GPU parallel mode)...")
            generate_dataset_gpu_parallel(
                num_games=config.episodes_per_iter,
                output_file=data_file,
                board_type=config.board_type,
                seed=config.seed + i,  # Vary seed per iteration
                max_moves=config.max_moves_per_game,
                num_players=2,
                history_length=config.history_length,
                feature_version=config.feature_version,
                gpu_batch_size=config.gpu_batch_size,
            )
        else:
            print("Generating self-play data (CPU sequential mode)...")

            # Initialize Descent AIs
            # They will use the current neural net (if available) for evaluation.
            # Use rngSeed so that DescentAI/BaseAI derive a deterministic
            # per-instance RNG for self-play games in the training loop.
            ai1 = DescentAI(
                1,
                AIConfig(
                    difficulty=5,
                    think_time=500,
                    randomness=0.1,
                    rngSeed=config.seed,
                ),
            )
            ai2 = DescentAI(
                2,
                AIConfig(
                    difficulty=5,
                    think_time=500,
                    randomness=0.1,
                    # Use a different per-instance seed from player 1 to avoid
                    # correlated exploration in self-play.
                    rngSeed=(config.seed + 1),
                ),
            )

            # Generate data
            generate_dataset(
                num_games=config.episodes_per_iter,
                output_file=data_file,
                ai1=ai1,
                ai2=ai2,
                board_type=config.board_type,
                seed=config.seed,
                history_length=config.history_length,
                feature_version=config.feature_version,
            )

        # 2. Train Neural Net
        print("Training neural network...")
        # Train on current data, saving to candidate model file
        candidate_model_file = os.path.join(
            base_dir,
            config.model_dir,
            f"{config.model_id}_candidate.pth",
        )
        # Board-aware model version (centralized in config.py)
        model_version = get_model_version_for_board(config.board_type)

        # Create checkpoint directory for this iteration
        checkpoint_dir = os.path.join(base_dir, config.log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # train_model now extracts warmup/scheduler/stopping params from config
        # Enable safety features for production training loops
        train_model(
            config=config,
            data_path=data_file,
            save_path=candidate_model_file,
            model_version=model_version,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=max(1, config.epochs_per_iter // 5),  # ~5 checkpoints per iter
            enable_circuit_breaker=True,
            enable_anomaly_detection=True,
            enable_graceful_shutdown=True,
        )

        # 3. Evaluation (Tournament)
        if os.path.exists(best_model_file):
            print("Running tournament: Candidate vs Best...")
            tournament = Tournament(
                candidate_model_file, best_model_file, num_games=50  # Increased from 10 for statistical validity
            )
            results = tournament.run()

            # Promotion logic: Candidate must win > 55% of games
            # (excluding draws)
            total_decisive = results["A"] + results["B"]
            if total_decisive > 0:
                win_rate = results["A"] / total_decisive
                print(f"Candidate win rate: {win_rate:.2f}")

                if win_rate > WIN_RATE_BEAT_BEST:
                    print("Candidate promoted to Best Model!")
                    shutil.copy(candidate_model_file, best_model_file)
                    # Also update the main model file used for inference
                    shutil.copy(candidate_model_file, model_file)
                else:
                    print("Candidate failed to beat Best Model.")
            else:
                print(
                    "Tournament inconclusive (all draws). Keeping Best Model."
                )
        else:
            # First iteration, promote candidate immediately
            print("First model generated. Promoting to Best Model.")
            shutil.copy(candidate_model_file, best_model_file)
            shutil.copy(candidate_model_file, model_file)

        # Export the current heuristic weight profiles after each iteration so
        # that offline tools can inspect/tune them without changing runtime
        # defaults.
        export_heuristic_profiles(config.log_dir)

        print("Iteration complete.")


def _run_training_loop_optimized(config: TrainConfig) -> None:
    """Run training loop using OptimizedTrainingPipeline.

    This version provides:
    - Export caching (skip unchanged data)
    - Distributed locks (prevent concurrent training)
    - Health monitoring (track status and alerts)
    - Curriculum feedback (adaptive weights)
    - Model registry (full traceability)
    """
    pipeline = get_optimized_pipeline()

    # Determine config key from board type
    board_type = config.board_type.value if hasattr(config.board_type, 'value') else str(config.board_type)
    num_players = getattr(config, 'num_players', 2)
    config_key = f"{board_type}_{num_players}p"

    print(
        f"Starting optimized training loop for {config_key}: "
        f"{config.epochs_per_iter} iterations, {config.episodes_per_iter} games/iter"
    )
    print(f"  Pipeline status: {pipeline.get_status().available_features}")

    base_dir = os.getcwd()
    model_file = os.path.join(
        base_dir,
        config.model_dir,
        f"{config.model_id}.pth",
    )
    best_model_file = os.path.join(
        base_dir,
        config.model_dir,
        f"{config.model_id}_best.pth",
    )

    # Initialize best model if not exists
    if not os.path.exists(best_model_file) and os.path.exists(model_file):
        shutil.copy(model_file, best_model_file)

    num_loops = config.iterations

    for i in range(num_loops):
        print(f"\n=== Iteration {i+1}/{num_loops} ===")

        # Check if training should run (using pipeline signals)
        should_train, reason = pipeline.should_train(config_key)
        if not should_train:
            print(f"Skipping training: {reason}")
            continue

        # Get curriculum weight for adaptive training
        curriculum_weight = pipeline.get_curriculum_weight(config_key)
        effective_games = int(config.episodes_per_iter * curriculum_weight)
        print(f"Curriculum weight: {curriculum_weight:.2f} -> {effective_games} games")

        # 1. Self-Play Data Generation (same as before)
        data_file = os.path.join(base_dir, config.data_dir, "self_play_data.npz")

        if config.use_gpu_parallel_datagen:
            print("Generating self-play data (GPU parallel mode)...")
            generate_dataset_gpu_parallel(
                num_games=effective_games,
                output_file=data_file,
                board_type=config.board_type,
                seed=config.seed + i,
                max_moves=config.max_moves_per_game,
                num_players=2,
                history_length=config.history_length,
                feature_version=config.feature_version,
                gpu_batch_size=config.gpu_batch_size,
            )
        else:
            print("Generating self-play data (CPU sequential mode)...")
            ai1 = DescentAI(
                1,
                AIConfig(difficulty=5, think_time=500, randomness=0.1, rngSeed=config.seed),
            )
            ai2 = DescentAI(
                2,
                AIConfig(difficulty=5, think_time=500, randomness=0.1, rngSeed=config.seed + 1),
            )
            generate_dataset(
                num_games=effective_games,
                output_file=data_file,
                ai1=ai1,
                ai2=ai2,
                board_type=config.board_type,
                seed=config.seed,
                history_length=config.history_length,
                feature_version=config.feature_version,
            )

        # 2. Training via OptimizedPipeline
        print("Training via optimized pipeline...")
        result = pipeline.run_training(
            config_key=config_key,
            npz_path=data_file,
            skip_export=True,  # Data already in NPZ format
        )

        if result.success:
            print(f"Training complete: {result.message}")
            if result.model_path and os.path.exists(result.model_path):
                # 3. Evaluation Tournament
                print("Running tournament: Candidate vs Best...")
                tournament = Tournament(result.model_path, best_model_file, num_games=50)  # Increased from 10 for statistical validity
                results = tournament.run()

                total_decisive = results["A"] + results["B"]
                if total_decisive > 0:
                    win_rate = results["A"] / total_decisive
                    print(f"Candidate win rate: {win_rate:.2f}")

                    if win_rate > WIN_RATE_BEAT_BEST:
                        print("Candidate promoted to Best Model!")
                        shutil.copy(result.model_path, best_model_file)
                        shutil.copy(result.model_path, model_file)
                    else:
                        print("Candidate failed to beat Best Model.")
        else:
            print(f"Training failed: {result.message}")

        export_heuristic_profiles(config.log_dir)
        print("Iteration complete.")


if __name__ == "__main__":
    run_training_loop()
