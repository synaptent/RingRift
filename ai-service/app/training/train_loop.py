import os
import shutil
import json
from typing import Optional

from app.training.generate_data import generate_dataset  # noqa: E402
from app.training.train import train_model  # noqa: E402
from app.models import AIConfig  # noqa: E402
from app.ai.descent_ai import DescentAI  # noqa: E402
from app.training.tournament import Tournament  # noqa: E402
from app.training.config import TrainConfig  # noqa: E402
from app.ai.heuristic_weights import HEURISTIC_WEIGHT_PROFILES  # noqa: E402


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


def run_training_loop(config: Optional[TrainConfig] = None):
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

        # 1. Self-Play (Descent vs Descent)
        print("Generating self-play data...")

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
        )

        # 2. Train Neural Net
        print("Training neural network...")
        # Train on current data, saving to candidate model file
        candidate_model_file = os.path.join(
            base_dir,
            config.model_dir,
            f"{config.model_id}_candidate.pth",
        )
        train_model(
            config=config,
            data_path=data_file,
            save_path=candidate_model_file,
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
