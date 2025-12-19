#!/usr/bin/env python3
"""
Continuous Training Pipeline

Fully automated training loop that:
1. Monitors selfplay data generation
2. Trains model when new data is available
3. Evaluates in gauntlet
4. Promotes best models
5. Restarts selfplay with improved model

Usage:
    python scripts/continuous_training_loop.py --board-type square8 --num-players 2

Environment:
    RINGRIFT_CONTINUOUS_ENABLED=1  # Enable loop (otherwise single iteration)
    RINGRIFT_TRAINING_EPOCHS=20    # Epochs per training cycle
    RINGRIFT_SELFPLAY_GAMES=200    # Games per selfplay cycle
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("continuous_training")


class ContinuousTrainingPipeline:
    """Automated continuous training pipeline."""

    def __init__(
        self,
        board_type: str,
        num_players: int,
        base_dir: str = ".",
        epochs_per_cycle: int = 20,
        games_per_cycle: int = 200,
        min_samples_for_training: int = 5000,
        gauntlet_games: int = 80,
        elo_improvement_threshold: float = 20.0,
    ):
        self.board_type = board_type
        self.num_players = num_players
        self.base_dir = Path(base_dir)
        self.epochs_per_cycle = epochs_per_cycle
        self.games_per_cycle = games_per_cycle
        self.min_samples_for_training = min_samples_for_training
        self.gauntlet_games = gauntlet_games
        self.elo_improvement_threshold = elo_improvement_threshold

        # Directories
        self.config_key = f"{board_type}_{num_players}p"
        self.data_dir = self.base_dir / "data" / "selfplay" / "continuous" / self.config_key
        self.model_dir = self.base_dir / "models" / "continuous" / self.config_key
        self.log_dir = self.base_dir / "logs" / "continuous" / self.config_key
        self.state_file = self.log_dir / "pipeline_state.json"

        # Create directories
        for d in [self.data_dir, self.model_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # State
        self.state = self._load_state()

        # Add file handler
        file_handler = logging.FileHandler(self.log_dir / "pipeline.log")
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)

    def _load_state(self) -> Dict[str, Any]:
        """Load pipeline state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

        return {
            "iteration": 0,
            "best_model": None,
            "best_elo": 400,  # Random baseline
            "total_samples_trained": 0,
            "last_training": None,
            "history": [],
        }

    def _save_state(self):
        """Save pipeline state to disk."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def run_selfplay(self) -> Optional[Path]:
        """Run selfplay data generation."""
        logger.info(f"Starting selfplay: {self.games_per_cycle} games")

        iteration = self.state["iteration"]
        output_path = self.data_dir / f"selfplay_iter{iteration}.npz"

        cmd = [
            sys.executable, "-m", "app.training.generate_data",
            "--board-type", self.board_type,
            "--num-games", str(self.games_per_cycle),
            "--num-players", str(self.num_players),
            "--output", str(output_path),
        ]

        # Use best model if available
        if self.state["best_model"]:
            cmd.extend(["--model-path", self.state["best_model"]])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
            )
            if result.returncode != 0:
                logger.error(f"Selfplay failed: {result.stderr}")
                return None

            logger.info(f"Selfplay complete: {output_path}")
            return output_path

        except subprocess.TimeoutExpired:
            logger.error("Selfplay timed out")
            return None
        except Exception as e:
            logger.error(f"Selfplay error: {e}")
            return None

    def aggregate_data(self) -> Optional[Path]:
        """Aggregate all selfplay data for training."""
        data_files = list(self.data_dir.glob("selfplay_iter*.npz"))
        if not data_files:
            logger.warning("No selfplay data found")
            return None

        # Load and merge all data
        all_states = []
        all_policies = []
        all_values = []

        for f in sorted(data_files):
            try:
                data = np.load(f, allow_pickle=True)
                all_states.append(data['states'])
                all_policies.append(data['policies'])
                all_values.append(data['values'])
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")

        if not all_states:
            return None

        # Concatenate
        merged_states = np.concatenate(all_states, axis=0)
        merged_policies = np.concatenate(all_policies, axis=0)
        merged_values = np.concatenate(all_values, axis=0)

        # Save merged data
        merged_path = self.data_dir / "merged_training.npz"
        np.savez_compressed(
            merged_path,
            states=merged_states,
            policies=merged_policies,
            values=merged_values,
        )

        total_samples = len(merged_states)
        logger.info(f"Aggregated {total_samples} samples from {len(data_files)} files")

        return merged_path if total_samples >= self.min_samples_for_training else None

    def train_model(self, data_path: Path) -> Optional[Path]:
        """Train a new model on aggregated data."""
        logger.info(f"Starting training: {self.epochs_per_cycle} epochs")

        iteration = self.state["iteration"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{self.config_key}_iter{iteration}_{timestamp}"
        output_path = self.model_dir / f"{model_name}.pth"

        cmd = [
            sys.executable, "-m", "app.training.train",
            "--data", str(data_path),
            "--output", str(output_path),
            "--board-type", self.board_type,
            "--num-players", str(self.num_players),
            "--epochs", str(self.epochs_per_cycle),
            "--batch-size", "512",
            "--lr", "1e-3",
            "--warmup-epochs", "1",
            "--lr-scheduler", "cosine",
        ]

        # Resume from best model if available
        if self.state["best_model"] and Path(self.state["best_model"]).exists():
            cmd.extend(["--resume", self.state["best_model"]])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            if result.returncode != 0:
                logger.error(f"Training failed: {result.stderr}")
                return None

            logger.info(f"Training complete: {output_path}")
            return output_path

        except subprocess.TimeoutExpired:
            logger.error("Training timed out")
            return None
        except Exception as e:
            logger.error(f"Training error: {e}")
            return None

    def evaluate_model(self, model_path: Path) -> Optional[float]:
        """Evaluate model in gauntlet and return Elo."""
        logger.info(f"Starting gauntlet evaluation: {self.gauntlet_games} games")

        # Copy model to models directory for gauntlet
        model_name = model_path.stem
        gauntlet_path = self.base_dir / "models" / f"{model_name}.pth"
        shutil.copy(model_path, gauntlet_path)

        cmd = [
            sys.executable, "scripts/run_gauntlet.py",
            "--config", self.config_key,
            "--local",
            "--games-per-matchup", str(self.gauntlet_games // 4),  # 4 baselines
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min timeout
            )

            # Parse Elo from output
            for line in result.stdout.split('\n'):
                if model_name in line and 'Elo' in line:
                    # Extract Elo value
                    import re
                    match = re.search(r'(\d+)\s*Elo', line)
                    if match:
                        elo = float(match.group(1))
                        logger.info(f"Model achieved {elo:.0f} Elo")
                        return elo

            logger.warning("Could not parse Elo from gauntlet output")
            return None

        except subprocess.TimeoutExpired:
            logger.error("Gauntlet timed out")
            return None
        except Exception as e:
            logger.error(f"Gauntlet error: {e}")
            return None

    def maybe_promote_model(self, model_path: Path, elo: float) -> bool:
        """Promote model if it's better than current best."""
        current_best = self.state["best_elo"]
        improvement = elo - current_best

        if improvement >= self.elo_improvement_threshold:
            logger.info(
                f"Promoting model: {elo:.0f} Elo "
                f"(+{improvement:.0f} improvement)"
            )

            # Copy to production location
            production_path = self.base_dir / "models" / f"best_{self.config_key}.pth"
            shutil.copy(model_path, production_path)

            # Update state
            self.state["best_model"] = str(production_path)
            self.state["best_elo"] = elo
            self._save_state()

            return True
        else:
            logger.info(
                f"Model not promoted: {elo:.0f} Elo "
                f"(need +{self.elo_improvement_threshold:.0f}, got +{improvement:.0f})"
            )
            return False

    def run_iteration(self) -> bool:
        """Run a single training iteration."""
        iteration = self.state["iteration"]
        logger.info(f"=== Starting iteration {iteration} ===")

        # Step 1: Run selfplay
        selfplay_path = self.run_selfplay()
        if selfplay_path is None:
            logger.error("Selfplay failed, skipping iteration")
            return False

        # Step 2: Aggregate data
        data_path = self.aggregate_data()
        if data_path is None:
            logger.warning("Not enough data for training")
            return False

        # Step 3: Train model
        model_path = self.train_model(data_path)
        if model_path is None:
            logger.error("Training failed")
            return False

        # Step 4: Evaluate
        elo = self.evaluate_model(model_path)
        if elo is None:
            logger.error("Evaluation failed")
            return False

        # Step 5: Maybe promote
        promoted = self.maybe_promote_model(model_path, elo)

        # Update state
        self.state["iteration"] += 1
        self.state["last_training"] = datetime.now(timezone.utc).isoformat()
        self.state["history"].append({
            "iteration": iteration,
            "elo": elo,
            "promoted": promoted,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._save_state()

        logger.info(f"=== Iteration {iteration} complete ===")
        return True

    def run_loop(self, max_iterations: int = 0):
        """Run continuous training loop."""
        logger.info(f"Starting continuous training for {self.config_key}")
        logger.info(f"  Epochs per cycle: {self.epochs_per_cycle}")
        logger.info(f"  Games per cycle: {self.games_per_cycle}")
        logger.info(f"  Min samples: {self.min_samples_for_training}")

        iterations = 0
        while True:
            try:
                success = self.run_iteration()
                iterations += 1

                if max_iterations > 0 and iterations >= max_iterations:
                    logger.info(f"Reached max iterations ({max_iterations})")
                    break

                if not success:
                    logger.info("Iteration failed, waiting before retry...")
                    time.sleep(300)  # 5 min wait on failure

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                time.sleep(300)


def main():
    parser = argparse.ArgumentParser(description="Continuous training pipeline")
    parser.add_argument("--board-type", required=True, choices=["square8", "square19", "hex8", "hexagonal"])
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--max-iterations", type=int, default=0, help="Max iterations (0=infinite)")
    parser.add_argument("--min-samples", type=int, default=5000)
    parser.add_argument("--gauntlet-games", type=int, default=80)
    parser.add_argument("--elo-threshold", type=float, default=20.0)

    args = parser.parse_args()

    pipeline = ContinuousTrainingPipeline(
        board_type=args.board_type,
        num_players=args.num_players,
        epochs_per_cycle=args.epochs,
        games_per_cycle=args.games,
        min_samples_for_training=args.min_samples,
        gauntlet_games=args.gauntlet_games,
        elo_improvement_threshold=args.elo_threshold,
    )

    pipeline.run_loop(max_iterations=args.max_iterations)


if __name__ == "__main__":
    main()
