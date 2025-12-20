import json
import os
import shutil

from app.ai.descent_ai import DescentAI
from app.ai.heuristic_weights import HEURISTIC_WEIGHT_PROFILES
from app.models import AIConfig
from app.training.config import TrainConfig, get_model_version_for_board
from app.training.generate_data import generate_dataset, generate_dataset_gpu_parallel
from app.training.tournament import Tournament
from app.training.train import train_model


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


def run_training_loop(config: TrainConfig | None = None):
    if config is None:
        config = TrainConfig()

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
                candidate_model_file, best_model_file, num_games=10
            )
            results = tournament.run()

            # Promotion logic: Candidate must win > 55% of games
            # (excluding draws)
            total_decisive = results["A"] + results["B"]
            if total_decisive > 0:
                win_rate = results["A"] / total_decisive
                print(f"Candidate win rate: {win_rate:.2f}")

                if win_rate > 0.55:
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


if __name__ == "__main__":
    run_training_loop()
