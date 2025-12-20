#!/usr/bin/env python3
"""Hot Data Path - Fast iteration training without full reanalysis.

This module implements a fast training path that:
1. Uses 'played' policy targets instead of MCTS/Descent reanalysis
2. Incrementally trains on new games as they arrive
3. Maintains a rolling window of recent training data
4. Triggers promotion evaluation at configurable intervals

The hot path reduces iteration time from 12 hours to 1-2 hours by:
- Skipping expensive policy reanalysis (saves 2+ hours)
- Training incrementally on smaller batches (15-20 epochs vs 50)
- Using adaptive learning rate based on data freshness

Usage:
    # Start hot path training
    python scripts/hot_data_path.py --board square8 --players 2 --continuous

    # Single batch update
    python scripts/hot_data_path.py --board square8 --players 2 --batch-size 1000

    # With promotion evaluation
    python scripts/hot_data_path.py --board square8 --players 2 --eval-every 5
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("hot_data_path")
AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class HotPathConfig:
    """Configuration for hot data path training."""
    board_type: str = "square8"
    num_players: int = 2

    # Data settings
    rolling_window_games: int = 50000  # Keep last N games for training
    min_new_games_for_update: int = 500  # Minimum new games before training
    max_games_per_batch: int = 5000  # Max games to train on per batch

    # Training settings
    epochs_per_batch: int = 15
    batch_size: int = 256
    learning_rate: float = 0.001
    lr_warmup_games: int = 5000  # Reduce LR after this many games
    lr_decay_factor: float = 0.5

    # Evaluation settings
    eval_interval_batches: int = 5  # Evaluate every N batches
    eval_games: int = 50
    promotion_threshold: float = 0.53  # Lower threshold for hot path

    # Paths
    data_dir: Path = field(default_factory=lambda: AI_SERVICE_ROOT / "data" / "hot_path")
    checkpoint_dir: Path = field(default_factory=lambda: AI_SERVICE_ROOT / "models" / "hot_path")

    @property
    def state_file(self) -> Path:
        return self.data_dir / f"hot_path_state_{self.board_type}_{self.num_players}p.json"


@dataclass
class HotPathState:
    """Persistent state for hot path training."""
    last_processed_game_id: Optional[str] = None
    total_games_processed: int = 0
    batches_since_eval: int = 0
    current_model_version: int = 0
    promotions: int = 0
    last_update_time: Optional[str] = None
    training_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "last_processed_game_id": self.last_processed_game_id,
            "total_games_processed": self.total_games_processed,
            "batches_since_eval": self.batches_since_eval,
            "current_model_version": self.current_model_version,
            "promotions": self.promotions,
            "last_update_time": self.last_update_time,
            "training_history": self.training_history[-100:],  # Keep last 100 entries
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HotPathState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class HotDataPath:
    """Fast iteration training with played policy targets."""

    def __init__(self, config: HotPathConfig):
        self.config = config
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.state = self._load_state()
        logger.info(f"HotDataPath initialized: {config.board_type} {config.num_players}p")
        logger.info(f"State: {self.state.total_games_processed} games processed, "
                   f"version {self.state.current_model_version}")

    def _load_state(self) -> HotPathState:
        """Load persistent state from disk."""
        if self.config.state_file.exists():
            try:
                with open(self.config.state_file) as f:
                    return HotPathState.from_dict(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
        return HotPathState()

    def _save_state(self):
        """Save persistent state to disk."""
        with open(self.config.state_file, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def get_new_games(self) -> List[Path]:
        """Find selfplay databases with new games."""
        game_dbs = []

        # Look for selfplay databases
        selfplay_dirs = [
            AI_SERVICE_ROOT / "data" / "gpu_selfplay",
            AI_SERVICE_ROOT / "data" / "selfplay",
            AI_SERVICE_ROOT / "data" / "games",
        ]

        pattern = f"*{self.config.board_type}*{self.config.num_players}p*.db"

        for dir_path in selfplay_dirs:
            if dir_path.exists():
                for db_path in dir_path.glob(pattern):
                    if db_path.stat().st_size > 1024:  # Skip empty DBs
                        game_dbs.append(db_path)

        return sorted(game_dbs, key=lambda p: p.stat().st_mtime, reverse=True)

    def count_new_games(self, db_paths: List[Path]) -> int:
        """Count total games available for training."""
        total = 0
        for db_path in db_paths:
            try:
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM games")
                    total += cursor.fetchone()[0]
            except Exception:
                pass
        return total

    def export_training_batch(
        self,
        db_paths: List[Path],
        max_games: int
    ) -> Tuple[bool, Path]:
        """Export a batch of games for training using 'played' policy."""
        output_dir = self.config.data_dir / "batches"
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_name = f"batch_{self.state.current_model_version}_{int(time.time())}"
        output_path = output_dir / f"{batch_name}.npz"

        # Build export command
        cmd = [
            sys.executable,
            str(AI_SERVICE_ROOT / "scripts" / "export_replay_dataset.py"),
            "--output", str(output_path),
            "--board-type", self.config.board_type,
            "--num-players", str(self.config.num_players),
            "--policy-target", "played",  # Fast path - no reanalysis
            "--max-games", str(max_games),
        ]

        for db_path in db_paths[:10]:  # Limit number of DBs
            cmd.extend(["--db", str(db_path)])

        logger.info(f"Exporting batch: {batch_name}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        if result.returncode != 0:
            logger.error(f"Export failed: {result.stderr}")
            return False, output_path

        return output_path.exists(), output_path

    def get_current_model(self) -> Optional[Path]:
        """Get the current best model for this config."""
        # Check for hot path model first
        hot_model = self.config.checkpoint_dir / f"hot_{self.config.board_type}_{self.config.num_players}p_v{self.state.current_model_version}.pth"
        if hot_model.exists():
            return hot_model

        # Fall back to production model
        token = {"square8": "sq8", "square19": "sq19", "hexagonal": "hex"}.get(
            self.config.board_type, "sq8"
        )
        prod_model = AI_SERVICE_ROOT / "models" / f"ringrift_best_{token}_{self.config.num_players}p.pth"
        if prod_model.exists():
            return prod_model

        return None

    def get_adaptive_lr(self) -> float:
        """Get adaptive learning rate based on training progress."""
        base_lr = self.config.learning_rate

        if self.state.total_games_processed < self.config.lr_warmup_games:
            return base_lr

        # Decay LR after warmup
        decay_steps = (self.state.total_games_processed - self.config.lr_warmup_games) // 10000
        return base_lr * (self.config.lr_decay_factor ** decay_steps)

    def train_on_batch(self, batch_path: Path) -> Tuple[bool, Optional[Path]]:
        """Train model on a batch of data."""
        current_model = self.get_current_model()
        new_version = self.state.current_model_version + 1

        output_model = self.config.checkpoint_dir / f"hot_{self.config.board_type}_{self.config.num_players}p_v{new_version}.pth"

        lr = self.get_adaptive_lr()

        cmd = [
            sys.executable,
            str(AI_SERVICE_ROOT / "scripts" / "train_nnue.py"),
            "--npz", str(batch_path),
            "--board-type", self.config.board_type,
            "--num-players", str(self.config.num_players),
            "--epochs", str(self.config.epochs_per_batch),
            "--batch-size", str(self.config.batch_size),
            "--learning-rate", str(lr),
            "--save-path", str(output_model),
            "--early-stopping-patience", "5",
        ]

        if current_model and current_model.exists():
            cmd.extend(["--resume-from", str(current_model)])

        logger.info(f"Training batch: v{new_version}, lr={lr:.6f}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return False, None

        if output_model.exists():
            self.state.current_model_version = new_version
            self.state.batches_since_eval += 1
            self.state.last_update_time = datetime.now(timezone.utc).isoformat()
            self._save_state()
            return True, output_model

        return False, None

    def evaluate_and_promote(self, new_model: Path) -> Tuple[bool, float]:
        """Evaluate new model against baseline and promote if improved."""
        baseline = self._get_baseline_model()
        if not baseline or not baseline.exists():
            logger.info("No baseline for comparison, promoting by default")
            self._promote_model(new_model)
            return True, 1.0

        # Run evaluation
        cmd = [
            sys.executable,
            str(AI_SERVICE_ROOT / "scripts" / "evaluate_ai_models.py"),
            "--board", self.config.board_type,
            "--players", str(self.config.num_players),
            "--model-a", str(new_model),
            "--model-b", str(baseline),
            "--games", str(self.config.eval_games),
            "--output-json", str(self.config.data_dir / "eval_result.json"),
        ]

        logger.info(f"Evaluating {new_model.name} vs {baseline.name}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            logger.error(f"Evaluation failed: {result.stderr}")
            return False, 0.0

        # Parse results
        eval_json = self.config.data_dir / "eval_result.json"
        if eval_json.exists():
            with open(eval_json) as f:
                data = json.load(f)

            wins = data.get("wins", 0)
            losses = data.get("losses", 0)
            draws = data.get("draws", 0)
            total = wins + losses + draws

            if total > 0:
                win_rate = (wins + 0.5 * draws) / total
                logger.info(f"Win rate: {win_rate:.2%} ({wins}W/{losses}L/{draws}D)")

                if win_rate >= self.config.promotion_threshold:
                    self._promote_model(new_model)
                    return True, win_rate
                else:
                    logger.info("Below promotion threshold, keeping baseline")
                    return False, win_rate

        return False, 0.0

    def _get_baseline_model(self) -> Optional[Path]:
        """Get the current production baseline model."""
        token = {"square8": "sq8", "square19": "sq19", "hexagonal": "hex"}.get(
            self.config.board_type, "sq8"
        )
        return AI_SERVICE_ROOT / "models" / f"ringrift_best_{token}_{self.config.num_players}p.pth"

    def _promote_model(self, model: Path):
        """Promote model to production."""
        token = {"square8": "sq8", "square19": "sq19", "hexagonal": "hex"}.get(
            self.config.board_type, "sq8"
        )
        target = AI_SERVICE_ROOT / "models" / f"ringrift_best_{token}_{self.config.num_players}p.pth"

        # Backup current
        if target.exists():
            backup = target.with_suffix(".pth.backup")
            target.rename(backup)

        # Copy new model
        import shutil
        shutil.copy2(model, target)

        self.state.promotions += 1
        self._save_state()

        logger.info(f"Promoted {model.name} to {target.name}")

    def run_batch(self) -> bool:
        """Run a single training batch."""
        # Find new games
        db_paths = self.get_new_games()
        if not db_paths:
            logger.info("No new games found")
            return False

        total_games = self.count_new_games(db_paths)
        if total_games < self.config.min_new_games_for_update:
            logger.info(f"Not enough new games: {total_games} < {self.config.min_new_games_for_update}")
            return False

        # Export batch
        success, batch_path = self.export_training_batch(
            db_paths,
            self.config.max_games_per_batch
        )
        if not success:
            return False

        # Train
        success, new_model = self.train_on_batch(batch_path)
        if not success:
            return False

        self.state.total_games_processed += min(total_games, self.config.max_games_per_batch)

        # Record history
        self.state.training_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": self.state.current_model_version,
            "games_in_batch": min(total_games, self.config.max_games_per_batch),
            "total_games": self.state.total_games_processed,
        })
        self._save_state()

        # Evaluate if interval reached
        if self.state.batches_since_eval >= self.config.eval_interval_batches:
            promoted, win_rate = self.evaluate_and_promote(new_model)
            self.state.batches_since_eval = 0
            self._save_state()

            if promoted:
                logger.info(f"Model promoted! Win rate: {win_rate:.2%}")

        return True

    def run_continuous(self, interval_seconds: int = 300):
        """Run continuous hot path training."""
        logger.info(f"Starting continuous hot path training, checking every {interval_seconds}s")

        try:
            while True:
                try:
                    had_work = self.run_batch()
                    if had_work:
                        logger.info(f"Batch complete. Total: {self.state.total_games_processed} games, "
                                   f"v{self.state.current_model_version}")
                    else:
                        logger.debug("No work available, sleeping...")
                except Exception as e:
                    logger.error(f"Batch error: {e}")

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Interrupted, saving state...")
            self._save_state()


def main():
    parser = argparse.ArgumentParser(description="Hot Data Path - Fast iteration training")

    parser.add_argument("--board", "-b", type=str, default="square8")
    parser.add_argument("--players", "-p", type=int, default=2)
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds")
    parser.add_argument("--batch-size", type=int, default=5000, help="Max games per batch")
    parser.add_argument("--eval-every", type=int, default=5, help="Evaluate every N batches")
    parser.add_argument("--status", action="store_true", help="Show status and exit")

    args = parser.parse_args()

    config = HotPathConfig(
        board_type=args.board,
        num_players=args.players,
        max_games_per_batch=args.batch_size,
        eval_interval_batches=args.eval_every,
    )

    hot_path = HotDataPath(config)

    if args.status:
        print(json.dumps(hot_path.state.to_dict(), indent=2))
        return

    if args.continuous:
        hot_path.run_continuous(args.interval)
    else:
        hot_path.run_batch()


if __name__ == "__main__":
    main()
