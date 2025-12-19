#!/usr/bin/env python3
"""Real hyperparameter tuning with actual model training and Elo evaluation.

Unlike the simulated hyperparameter_tuning.py, this script:
1. Actually trains models (short runs) for each trial
2. Evaluates using validation loss AND Elo games
3. Stores results persistently per board/player config
4. Supports resuming interrupted tuning sessions

Usage:
    # Tune square8 2p with 30 trials
    python scripts/tune_hyperparameters.py --board square8 --players 2 --trials 30

    # Tune all configs with 20 trials each
    python scripts/tune_hyperparameters.py --all --trials 20

    # Resume interrupted tuning
    python scripts/tune_hyperparameters.py --board square8 --players 2 --resume
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
HYPERPARAMS_PATH = AI_SERVICE_ROOT / "config" / "hyperparameters.json"

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("tune_hyperparameters")


@dataclass
class TrialResult:
    """Result of a single hyperparameter trial."""
    trial_id: int
    params: Dict[str, Any]
    val_loss: Optional[float] = None
    elo_score: Optional[float] = None
    combined_score: float = -float("inf")
    training_time_sec: float = 0.0
    elo_games: int = 0
    timestamp: str = ""
    error: Optional[str] = None


@dataclass
class TuningSession:
    """State of a hyperparameter tuning session."""
    board_type: str
    num_players: int
    config_key: str
    trials: List[TrialResult] = field(default_factory=list)
    best_trial_id: int = -1
    best_score: float = -float("inf")
    best_params: Dict[str, Any] = field(default_factory=dict)
    total_trials: int = 0
    start_time: str = ""
    last_updated: str = ""


def load_hyperparams_config() -> Dict[str, Any]:
    """Load the hyperparameters configuration file."""
    if HYPERPARAMS_PATH.exists():
        with open(HYPERPARAMS_PATH) as f:
            return json.load(f)
    return {"defaults": {}, "configs": {}, "tuning_config": {}}


def save_hyperparams_config(config: Dict[str, Any]) -> None:
    """Save the hyperparameters configuration file."""
    config["last_updated"] = datetime.utcnow().isoformat() + "Z"
    HYPERPARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HYPERPARAMS_PATH, "w") as f:
        json.dump(config, f, indent=2)


def get_config_key(board_type: str, num_players: int) -> str:
    """Get the config key for a board/player combination."""
    return f"{board_type}_{num_players}p"


def sample_hyperparams(
    search_space: Dict[str, Any],
    best_params: Optional[Dict[str, Any]] = None,
    exploitation_prob: float = 0.3,
) -> Dict[str, Any]:
    """Sample hyperparameters from the search space.

    With exploitation_prob probability, perturb the best known params.
    Otherwise, sample uniformly from the space.
    """
    params = {}

    # Decide whether to explore or exploit
    exploit = best_params and random.random() < exploitation_prob

    for name, spec in search_space.items():
        if exploit and name in best_params and random.random() > 0.3:
            # Keep the best value for this param
            params[name] = best_params[name]
            continue

        param_type = spec.get("type", "float")

        if param_type == "categorical":
            params[name] = random.choice(spec["choices"])
        elif param_type == "int":
            params[name] = random.randint(int(spec["low"]), int(spec["high"]))
        elif param_type == "float":
            if spec.get("log_scale", False):
                log_low = np.log(spec["low"])
                log_high = np.log(spec["high"])
                params[name] = float(np.exp(random.uniform(log_low, log_high)))
            else:
                params[name] = random.uniform(spec["low"], spec["high"])

    return params


def train_model_with_params(
    params: Dict[str, Any],
    board_type: str,
    num_players: int,
    db_paths: List[Path],
    epochs: int = 10,
    output_dir: Optional[Path] = None,
) -> Tuple[Optional[float], Optional[Path], Dict[str, Any]]:
    """Train a model with given hyperparameters.

    Returns:
        (validation_loss, model_path, training_report)
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="hp_tune_"))

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"

    # Build training command
    cmd = [
        sys.executable, str(AI_SERVICE_ROOT / "scripts" / "train_nnue.py"),
        "--db", *[str(db) for db in db_paths[:3]],  # Limit DBs for speed
        "--board-type", board_type,
        "--num-players", str(num_players),
        "--epochs", str(epochs),
        "--batch-size", str(int(params.get("batch_size", 256))),
        "--learning-rate", str(params.get("learning_rate", 0.001)),
        "--weight-decay", str(params.get("weight_decay", 0.0001)),
        "--hidden-dim", str(int(params.get("hidden_dim", 256))),
        "--num-hidden-layers", str(int(params.get("num_hidden_layers", 2))),
        "--early-stopping-patience", str(min(epochs, 10)),
        "--run-dir", str(output_dir),
        "--save-path", str(model_path),
        "--max-samples", "50000",  # Limit samples for faster trials
    ]

    logger.info(f"Training with params: lr={params.get('learning_rate', 0.001):.6f}, "
                f"bs={params.get('batch_size', 256)}, "
                f"hidden={params.get('hidden_dim', 256)}, "
                f"layers={params.get('num_hidden_layers', 2)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per trial
            cwd=str(AI_SERVICE_ROOT),
        )

        if result.returncode != 0:
            logger.warning(f"Training failed: {result.stderr[:500]}")
            return None, None, {"error": result.stderr[:500]}

    except subprocess.TimeoutExpired:
        logger.warning("Training timed out")
        return None, None, {"error": "timeout"}
    except Exception as e:
        logger.warning(f"Training error: {e}")
        return None, None, {"error": str(e)}

    # Parse training report
    report_path = output_dir / "nnue_training_report.json"
    report = {}
    if report_path.exists():
        try:
            with open(report_path) as f:
                report = json.load(f)
        except Exception:
            pass

    val_loss = report.get("best_val_loss")
    if val_loss is not None:
        try:
            val_loss = float(val_loss)
        except (TypeError, ValueError):
            val_loss = None

    if not model_path.exists():
        return None, None, report

    return val_loss, model_path, report


def evaluate_model_elo(
    model_path: Path,
    board_type: str,
    num_players: int,
    num_games: int = 20,
) -> Optional[float]:
    """Evaluate a model's strength via Elo games against baseline.

    Returns win rate against baseline (0.0 to 1.0).
    """
    # Check if we have the tournament infrastructure
    tournament_script = AI_SERVICE_ROOT / "scripts" / "run_nnue_tournament.py"
    if not tournament_script.exists():
        logger.info("Tournament script not found, skipping Elo evaluation")
        return None

    try:
        # Run quick tournament against baseline
        cmd = [
            sys.executable, str(tournament_script),
            "--candidate", str(model_path),
            "--board", board_type,
            "--players", str(num_players),
            "--games", str(num_games),
            "--quick",  # Quick mode for HP tuning
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout
            cwd=str(AI_SERVICE_ROOT),
        )

        if result.returncode != 0:
            return None

        # Parse win rate from output
        for line in result.stdout.split("\n"):
            if "win_rate" in line.lower():
                try:
                    # Extract number from line like "Win rate: 0.65"
                    parts = line.split(":")
                    if len(parts) >= 2:
                        return float(parts[-1].strip())
                except (ValueError, IndexError):
                    pass

        return None

    except Exception as e:
        logger.warning(f"Elo evaluation error: {e}")
        return None


def compute_combined_score(
    val_loss: Optional[float],
    elo_score: Optional[float],
    val_weight: float = 0.6,
    elo_weight: float = 0.4,
) -> float:
    """Compute combined score from validation loss and Elo.

    Higher is better. Val loss is inverted (lower loss = higher score).
    """
    score = 0.0

    if val_loss is not None:
        # Normalize val loss to 0-1 range (assume loss typically 0.5-2.0)
        normalized_loss = max(0, min(1, (2.0 - val_loss) / 1.5))
        score += val_weight * normalized_loss

    if elo_score is not None:
        score += elo_weight * elo_score
    elif val_loss is not None:
        # If no Elo, use full weight on val loss
        normalized_loss = max(0, min(1, (2.0 - val_loss) / 1.5))
        score += elo_weight * normalized_loss

    return score


def run_tuning_session(
    board_type: str,
    num_players: int,
    max_trials: int = 30,
    db_paths: Optional[List[Path]] = None,
    epochs_per_trial: int = 10,
    elo_games_per_trial: int = 20,
    resume: bool = False,
    output_dir: Optional[Path] = None,
) -> TuningSession:
    """Run a hyperparameter tuning session.

    Args:
        board_type: Board type to tune for
        num_players: Number of players
        max_trials: Maximum number of trials
        db_paths: Training database paths
        epochs_per_trial: Training epochs per trial
        elo_games_per_trial: Elo games per trial (0 to skip)
        resume: Resume from previous session
        output_dir: Output directory for logs

    Returns:
        TuningSession with results
    """
    config_key = get_config_key(board_type, num_players)
    hp_config = load_hyperparams_config()
    search_space = hp_config.get("tuning_config", {}).get("search_space", {})

    if output_dir is None:
        output_dir = AI_SERVICE_ROOT / "logs" / "hp_tuning" / config_key
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create session
    session_path = output_dir / "tuning_session.json"
    if resume and session_path.exists():
        with open(session_path) as f:
            session_data = json.load(f)
            session = TuningSession(
                board_type=board_type,
                num_players=num_players,
                config_key=config_key,
                trials=[TrialResult(**t) for t in session_data.get("trials", [])],
                best_trial_id=session_data.get("best_trial_id", -1),
                best_score=session_data.get("best_score", -float("inf")),
                best_params=session_data.get("best_params", {}),
                total_trials=session_data.get("total_trials", 0),
                start_time=session_data.get("start_time", ""),
            )
        logger.info(f"Resumed session with {len(session.trials)} existing trials")
    else:
        session = TuningSession(
            board_type=board_type,
            num_players=num_players,
            config_key=config_key,
            start_time=datetime.utcnow().isoformat() + "Z",
        )

    # Find training databases
    if db_paths is None:
        db_paths = list((AI_SERVICE_ROOT / "data" / "games").glob("*.db"))

    if not db_paths:
        logger.error("No training databases found")
        return session

    logger.info("=" * 60)
    logger.info(f"HYPERPARAMETER TUNING: {config_key}")
    logger.info("=" * 60)
    logger.info(f"Trials: {len(session.trials)}/{max_trials}")
    logger.info(f"Epochs per trial: {epochs_per_trial}")
    logger.info(f"Elo games per trial: {elo_games_per_trial}")
    logger.info(f"Databases: {len(db_paths)}")

    # Run trials
    while len(session.trials) < max_trials:
        trial_id = len(session.trials)
        trial_dir = output_dir / f"trial_{trial_id:03d}"

        # Sample hyperparameters
        params = sample_hyperparams(
            search_space,
            best_params=session.best_params if session.best_params else None,
            exploitation_prob=0.3 if len(session.trials) > 5 else 0.0,
        )

        logger.info(f"\n--- Trial {trial_id + 1}/{max_trials} ---")

        start_time = time.time()

        # Train model
        val_loss, model_path, report = train_model_with_params(
            params=params,
            board_type=board_type,
            num_players=num_players,
            db_paths=db_paths,
            epochs=epochs_per_trial,
            output_dir=trial_dir,
        )

        training_time = time.time() - start_time

        # Evaluate Elo if we have a model
        elo_score = None
        if model_path and elo_games_per_trial > 0:
            elo_score = evaluate_model_elo(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
                num_games=elo_games_per_trial,
            )

        # Compute combined score
        combined_score = compute_combined_score(val_loss, elo_score)

        # Record trial
        trial = TrialResult(
            trial_id=trial_id,
            params=params,
            val_loss=val_loss,
            elo_score=elo_score,
            combined_score=combined_score,
            training_time_sec=training_time,
            elo_games=elo_games_per_trial if elo_score else 0,
            timestamp=datetime.utcnow().isoformat() + "Z",
            error=report.get("error"),
        )
        session.trials.append(trial)
        session.total_trials += 1

        # Update best
        if combined_score > session.best_score:
            session.best_score = combined_score
            session.best_trial_id = trial_id
            session.best_params = params.copy()
            logger.info(f"NEW BEST! score={combined_score:.4f}, val_loss={val_loss}, elo={elo_score}")
        else:
            logger.info(f"Trial score={combined_score:.4f} (best={session.best_score:.4f})")

        # Save session state
        session.last_updated = datetime.utcnow().isoformat() + "Z"
        with open(session_path, "w") as f:
            json.dump({
                "board_type": session.board_type,
                "num_players": session.num_players,
                "config_key": session.config_key,
                "trials": [asdict(t) for t in session.trials],
                "best_trial_id": session.best_trial_id,
                "best_score": session.best_score,
                "best_params": session.best_params,
                "total_trials": session.total_trials,
                "start_time": session.start_time,
                "last_updated": session.last_updated,
            }, f, indent=2)

        # Cleanup trial directory to save space (keep best)
        if trial_id != session.best_trial_id and trial_dir.exists():
            shutil.rmtree(trial_dir, ignore_errors=True)

    # Update hyperparameters.json with best results
    if session.best_params:
        hp_config = load_hyperparams_config()
        if config_key not in hp_config.get("configs", {}):
            hp_config.setdefault("configs", {})[config_key] = {}

        config_entry = hp_config["configs"][config_key]
        config_entry["optimized"] = True
        config_entry["confidence"] = (
            "high" if session.total_trials >= 50 else
            "medium" if session.total_trials >= 20 else
            "low"
        )
        config_entry["tuning_method"] = "real_training"
        config_entry["last_tuned"] = session.last_updated
        config_entry["tuning_trials"] = session.total_trials
        config_entry["hyperparameters"] = session.best_params
        config_entry["best_val_loss"] = session.trials[session.best_trial_id].val_loss if session.best_trial_id >= 0 else None
        config_entry["best_elo_score"] = session.trials[session.best_trial_id].elo_score if session.best_trial_id >= 0 else None
        config_entry["notes"] = f"Tuned with {session.total_trials} trials on {datetime.utcnow().strftime('%Y-%m-%d')}"

        save_hyperparams_config(hp_config)
        logger.info(f"Updated {HYPERPARAMS_PATH} with best params")

    logger.info("\n" + "=" * 60)
    logger.info("TUNING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best score: {session.best_score:.4f}")
    logger.info(f"Best params: {session.best_params}")
    logger.info(f"Trials completed: {session.total_trials}")

    return session


def main():
    parser = argparse.ArgumentParser(
        description="Real hyperparameter tuning with actual model training"
    )

    parser.add_argument(
        "--board",
        type=str,
        default="square8",
        choices=["square8", "square19", "hexagonal"],
        help="Board type to tune",
    )
    parser.add_argument(
        "--players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Tune all board/player configurations",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of trials per configuration",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs per trial",
    )
    parser.add_argument(
        "--elo-games",
        type=int,
        default=0,
        help="Elo evaluation games per trial (0 to skip)",
    )
    parser.add_argument(
        "--db",
        type=str,
        nargs="+",
        help="Training database paths",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted tuning session",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for logs",
    )

    args = parser.parse_args()

    # Expand database paths
    db_paths = None
    if args.db:
        import glob
        db_paths = []
        for pattern in args.db:
            matches = glob.glob(pattern)
            db_paths.extend(Path(m) for m in matches)

    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.all:
        # Tune all configurations
        configs = [
            ("square8", 2), ("square8", 3), ("square8", 4),
            ("square19", 2), ("square19", 3), ("square19", 4),
            ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
        ]

        for board_type, num_players in configs:
            logger.info(f"\n\n{'#' * 60}")
            logger.info(f"# TUNING {board_type} {num_players}p")
            logger.info(f"{'#' * 60}\n")

            run_tuning_session(
                board_type=board_type,
                num_players=num_players,
                max_trials=args.trials,
                db_paths=db_paths,
                epochs_per_trial=args.epochs,
                elo_games_per_trial=args.elo_games,
                resume=args.resume,
                output_dir=output_dir / f"{board_type}_{num_players}p" if output_dir else None,
            )
    else:
        # Tune single configuration
        run_tuning_session(
            board_type=args.board,
            num_players=args.players,
            max_trials=args.trials,
            db_paths=db_paths,
            epochs_per_trial=args.epochs,
            elo_games_per_trial=args.elo_games,
            resume=args.resume,
            output_dir=output_dir,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
