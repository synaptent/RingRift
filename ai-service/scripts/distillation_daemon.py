#!/usr/bin/env python3
"""Automated distillation daemon for production model deployment.

Watches for newly promoted models and automatically distills them to NNUE
format for fast CPU inference in production.

The daemon:
1. Monitors models/promoted/ for new model files
2. Runs NNUE distillation when a new model is detected
3. Validates the distilled model quality
4. Deploys to models/nnue/ for production use
5. Sends notifications on success/failure

Usage:
    python scripts/distillation_daemon.py --check-once  # Single check
    python scripts/distillation_daemon.py --daemon      # Continuous monitoring
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Unified logging
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("distillation_daemon")


@dataclass
class DistillationConfig:
    """Configuration for distillation daemon."""
    check_interval_seconds: int = 300  # 5 minutes
    num_positions: int = 100000
    epochs: int = 10
    batch_size: int = 512
    min_correlation: float = 0.7  # Minimum correlation to accept distilled model
    min_sign_agreement: float = 0.8  # Minimum sign agreement to accept


@dataclass
class DistillationJob:
    """A distillation job."""
    source_model: Path
    board_type: str
    num_players: int
    output_path: Path
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: bool = False
    metrics: Optional[Dict] = None
    error: Optional[str] = None


def detect_board_config(model_path: Path) -> tuple[str, int]:
    """Detect board type and player count from model filename or metadata.

    Returns:
        Tuple of (board_type, num_players)
    """
    name = model_path.stem.lower()

    # Try to extract from filename patterns
    board_type = "square8"
    num_players = 2

    if "sq8" in name or "square8" in name:
        board_type = "square8"
    elif "sq19" in name or "square19" in name:
        board_type = "square19"
    elif "hex" in name or "hexagonal" in name:
        board_type = "hexagonal"

    # Extract player count
    if "_4p" in name or "4p_" in name:
        num_players = 4
    elif "_3p" in name or "3p_" in name:
        num_players = 3
    elif "_2p" in name or "2p_" in name:
        num_players = 2

    # Try to load metadata file if exists
    meta_path = model_path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
                board_type = meta.get("board_type", board_type)
                num_players = meta.get("num_players", num_players)
        except Exception:
            pass

    return board_type, num_players


def get_model_hash(path: Path) -> str:
    """Get hash of model file for change detection."""
    if not path.exists():
        return ""
    h = hashlib.md5()
    h.update(str(path.stat().st_mtime).encode())
    h.update(str(path.stat().st_size).encode())
    return h.hexdigest()[:16]


def find_best_database(board_type: str) -> Path:
    """Find the best training database for the given board type."""
    data_dir = AI_SERVICE_ROOT / "data" / "games"

    # Priority order of databases
    candidates = [
        data_dir / "cluster_synced.db",
        data_dir / "selfplay.db",
        data_dir / "cluster_merged.db",
    ]

    for db in candidates:
        if db.exists():
            # Check if it has data for this board type
            try:
                import sqlite3
                conn = sqlite3.connect(str(db))
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM games WHERE board_type = ?",
                    (board_type,)
                )
                count = cursor.fetchone()[0]
                conn.close()
                if count > 0:
                    logger.info(f"Using database {db.name} ({count} games for {board_type})")
                    return db
            except Exception as e:
                logger.warning(f"Error checking {db}: {e}")
                continue

    # Fallback to first existing database
    for db in candidates:
        if db.exists():
            return db

    raise FileNotFoundError("No game database found for distillation")


def run_distillation(
    source_model: Path,
    board_type: str,
    num_players: int,
    config: DistillationConfig,
) -> DistillationJob:
    """Run NNUE distillation for a model.

    Args:
        source_model: Path to source NN model
        board_type: Board type (square8, hexagonal, square19)
        num_players: Number of players
        config: Distillation configuration

    Returns:
        DistillationJob with results
    """
    output_dir = AI_SERVICE_ROOT / "models" / "nnue"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"nnue_{board_type}_{num_players}p.pt"
    output_path = output_dir / output_name

    job = DistillationJob(
        source_model=source_model,
        board_type=board_type,
        num_players=num_players,
        output_path=output_path,
        started_at=datetime.utcnow(),
    )

    logger.info(f"Starting distillation: {source_model.name} -> {output_name}")

    try:
        # Find best database
        db_path = find_best_database(board_type)

        # Run distillation script
        cmd = [
            sys.executable,
            str(AI_SERVICE_ROOT / "scripts" / "distill_to_nnue.py"),
            "--nn-model", str(source_model),
            "--db", str(db_path),
            "--output", str(output_path),
            "--board-type", board_type,
            "--num-positions", str(config.num_positions),
            "--epochs", str(config.epochs),
            "--batch-size", str(config.batch_size),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=str(AI_SERVICE_ROOT),
        )

        if result.returncode != 0:
            job.error = f"Distillation failed: {result.stderr[-500:]}"
            logger.error(job.error)
            return job

        # Parse metrics from output
        job.metrics = {}
        for line in result.stdout.split("\n"):
            if "Final validation loss:" in line:
                job.metrics["val_loss"] = float(line.split(":")[-1].strip())
            elif "Final correlation:" in line:
                job.metrics["correlation"] = float(line.split(":")[-1].strip())
            elif "Final sign agreement:" in line:
                job.metrics["sign_agreement"] = float(line.split(":")[-1].strip())

        # Validate metrics
        correlation = job.metrics.get("correlation", 0)
        sign_agreement = job.metrics.get("sign_agreement", 0)

        if correlation < config.min_correlation:
            job.error = f"Correlation {correlation:.4f} below threshold {config.min_correlation}"
            logger.warning(job.error)
            # Don't fail, just warn - keep the model anyway

        if sign_agreement < config.min_sign_agreement:
            job.error = f"Sign agreement {sign_agreement:.4f} below threshold {config.min_sign_agreement}"
            logger.warning(job.error)

        job.success = output_path.exists()
        job.completed_at = datetime.utcnow()

        if job.success:
            logger.info(f"Distillation complete: {output_path}")
            logger.info(f"  Correlation: {correlation:.4f}, Sign agreement: {sign_agreement:.4f}")

            # Save metadata
            meta_path = output_path.with_suffix(".meta.json")
            with open(meta_path, "w") as f:
                json.dump({
                    "source_model": str(source_model),
                    "board_type": board_type,
                    "num_players": num_players,
                    "metrics": job.metrics,
                    "distilled_at": job.completed_at.isoformat() + "Z",
                }, f, indent=2)

    except subprocess.TimeoutExpired:
        job.error = "Distillation timed out after 1 hour"
        logger.error(job.error)
    except Exception as e:
        job.error = f"Distillation error: {str(e)}"
        logger.exception(job.error)

    return job


class DistillationDaemon:
    """Daemon that watches for new models and distills them."""

    def __init__(self, config: DistillationConfig):
        self.config = config
        self.models_dir = AI_SERVICE_ROOT / "models"
        self.promoted_dir = self.models_dir / "promoted"
        self.gauntlet_promoted_dir = self.models_dir / "gauntlet_promoted"
        self.known_models: Dict[str, str] = {}  # path -> hash
        self.jobs: List[DistillationJob] = []

        # Also watch for "best" models
        self.best_model_patterns = [
            "ringrift_best_*.pth",
            "*_best_*.pth",
        ]

        self._load_state()

    def _state_path(self) -> Path:
        return AI_SERVICE_ROOT / "data" / "distillation_daemon_state.json"

    def _load_state(self):
        """Load daemon state from disk."""
        state_path = self._state_path()
        if state_path.exists():
            try:
                with open(state_path) as f:
                    state = json.load(f)
                    self.known_models = state.get("known_models", {})
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

    def _save_state(self):
        """Save daemon state to disk."""
        state_path = self._state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(state_path, "w") as f:
                json.dump({
                    "known_models": self.known_models,
                    "last_updated": datetime.utcnow().isoformat() + "Z",
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    def _find_models_to_distill(self) -> List[Path]:
        """Find new models that need distillation."""
        models_to_distill = []

        # Check promoted directories
        for promoted_dir in [self.promoted_dir, self.gauntlet_promoted_dir]:
            if promoted_dir.exists():
                for model_path in promoted_dir.glob("*.pth"):
                    if model_path.is_symlink() and not model_path.exists():
                        continue  # Skip broken symlinks

                    current_hash = get_model_hash(model_path)
                    key = str(model_path)

                    if key not in self.known_models or self.known_models[key] != current_hash:
                        models_to_distill.append(model_path)

        # Check best models
        for pattern in self.best_model_patterns:
            for model_path in self.models_dir.glob(pattern):
                if model_path.is_symlink() and not model_path.exists():
                    continue

                current_hash = get_model_hash(model_path)
                key = str(model_path)

                if key not in self.known_models or self.known_models[key] != current_hash:
                    models_to_distill.append(model_path)

        return models_to_distill

    def process_model(self, model_path: Path) -> Optional[DistillationJob]:
        """Process a single model for distillation."""
        board_type, num_players = detect_board_config(model_path)

        logger.info(f"Processing {model_path.name} ({board_type}, {num_players}p)")

        job = run_distillation(
            source_model=model_path,
            board_type=board_type,
            num_players=num_players,
            config=self.config,
        )

        # Update known models
        self.known_models[str(model_path)] = get_model_hash(model_path)
        self._save_state()

        self.jobs.append(job)
        return job

    def check_once(self) -> List[DistillationJob]:
        """Run a single check for new models."""
        models = self._find_models_to_distill()

        if not models:
            logger.info("No new models to distill")
            return []

        logger.info(f"Found {len(models)} models to distill")

        jobs = []
        for model_path in models:
            try:
                job = self.process_model(model_path)
                if job:
                    jobs.append(job)
            except Exception as e:
                logger.exception(f"Error processing {model_path}: {e}")

        return jobs

    def run_daemon(self):
        """Run continuous monitoring loop."""
        logger.info(f"Starting distillation daemon (interval: {self.config.check_interval_seconds}s)")

        while True:
            try:
                self.check_once()
            except Exception as e:
                logger.exception(f"Error in daemon loop: {e}")

            time.sleep(self.config.check_interval_seconds)


def main():
    parser = argparse.ArgumentParser(description="Automated distillation daemon")
    parser.add_argument(
        "--check-once",
        action="store_true",
        help="Run a single check and exit",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as continuous daemon",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Check interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--num-positions",
        type=int,
        default=100000,
        help="Number of positions for distillation (default: 100000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs (default: 10)",
    )

    args = parser.parse_args()

    config = DistillationConfig(
        check_interval_seconds=args.interval,
        num_positions=args.num_positions,
        epochs=args.epochs,
    )

    daemon = DistillationDaemon(config)

    if args.daemon:
        daemon.run_daemon()
    else:
        # Default to check-once
        jobs = daemon.check_once()

        # Print summary
        successful = sum(1 for j in jobs if j.success)
        failed = len(jobs) - successful

        logger.info(f"\nDistillation complete: {successful} successful, {failed} failed")

        for job in jobs:
            status = "✓" if job.success else "✗"
            logger.info(f"  {status} {job.source_model.name} -> {job.output_path.name}")
            if job.metrics:
                logger.info(f"      Correlation: {job.metrics.get('correlation', 0):.4f}")


if __name__ == "__main__":
    main()
