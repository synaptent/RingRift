#!/usr/bin/env python
"""Baseline neural-network training experiment for all board types and player counts.

This script provides a small, reproducible NN training entrypoint
for RingRift games using the existing training stack:

- TrainConfig / get_training_config_for_board from
  app.training.config
- train_model from app.training.train

It is intentionally conservative and exposes two primary modes:

1. Demo / tiny mode (for CI and smoke tests):

   - Enabled via --demo.
   - Uses a non-existent NPZ path so RingRiftDataset generates a
     tiny in-memory dummy dataset (see
     app.training.train.RingRiftDataset.__init__).
   - Runs a very small number of epochs (default: 1) on the
     default device.
   - Writes nn_training_report.json into --run-dir with basic
     metadata and stubbed metrics (final_loss is not currently
     plumbed through train_model).

2. Full mode (experimental, non-CI):

   - Requires an explicit --data-path argument pointing at a
     Square-8 2-player NPZ dataset derived from a canonical DB
     listed in TRAINING_DATA_REGISTRY.md.
   - Uses the Square-8 TrainConfig preset and writes a checkpoint
     under ai-service/models.

This script does NOT change rules semantics or the AI ladder. It
only provides a narrow, documented way to exercise the neural
training stack for Square-8 2-player with a cheap demo
configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.models import BoardType  # noqa: E402
from app.training.config import TrainConfig  # noqa: E402
from app.training.config import (  # noqa: E402
    get_training_config_for_board,
)
from app.training.seed_utils import seed_all  # noqa: E402
from app.training.train import train_model  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np

# Model hygiene: validation at startup
try:
    from scripts.validate_models import run_startup_validation
    HAS_MODEL_VALIDATION = True
except ImportError:
    HAS_MODEL_VALIDATION = False


def _dataset_has_multi_player_values(data_path: str) -> bool:
    """Check if an NPZ dataset contains multi-player value targets."""
    try:
        with np.load(data_path, mmap_mode='r') as f:
            return 'values_mp' in f and 'num_players' in f
    except Exception:
        return False


def _parse_board(board: str) -> BoardType:
    """Map CLI board string to BoardType."""
    b = board.lower()
    if b in {"square8", "sq8"}:
        return BoardType.SQUARE8
    if b in {"square19", "sq19"}:
        return BoardType.SQUARE19
    if b in {"hexagonal", "hex"}:
        return BoardType.HEXAGONAL
    raise SystemExit(f"Unsupported board {board!r}; valid options: square8, square19, hexagonal")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the NN baseline training script."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a baseline neural-network training experiment for "
            "Square-8 2-player games and emit nn_training_report.json "
            "in --run-dir."
        ),
    )
    parser.add_argument(
        "--board",
        default="square8",
        choices=["square8", "sq8", "square19", "sq19", "hexagonal", "hex"],
        help=("Board identifier: square8, square19, or hexagonal."),
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help=("Number of players (2, 3, or 4)."),
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help=("Output dir for nn_training_report.json and checkpoints. " "Will be created if needed."),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Run a tiny, CI-safe demo training loop using the synthetic "
            "dummy dataset path. Intended for smoke tests only."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible runs (default: 42).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help=(
            "Optional explicit nn_model_id / checkpoint stem. When omitted, " "a timestamped baseline id is generated."
        ),
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help=(
            "Path to Square-8 2-player NPZ dataset for non-demo runs. "
            "MUST be derived from a canonical DB listed in "
            "TRAINING_DATA_REGISTRY.md. Ignored in --demo mode."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Optional override for number of training epochs. When omitted, "
            "the board-specific TrainConfig preset is used; in --demo mode "
            "this is clamped to 1 epoch."
        ),
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v2",
        choices=["v2", "v3", "v4"],
        help=(
            "Model architecture version to use. 'v2' is the standard "
            "RingRiftCNN_v2 with global average pooling. 'v3' is the new "
            "architecture with spatial policy heads and rank distribution "
            "output. 'v4' is the NAS-optimized architecture with multi-head "
            "attention (square boards only). Default: v2."
        ),
    )
    # Hyperparameter overrides (from config/hyperparameters.json or CLI)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate override (default: from config preset).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size override (default: from config preset).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay override (default: from config preset).",
    )
    parser.add_argument(
        "--policy-weight",
        type=float,
        default=None,
        help="Policy loss weight override (default: 1.0).",
    )
    parser.add_argument(
        "--value-weight",
        type=float,
        default=None,
        help="Value loss weight override (default: 1.0).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Early stopping patience in epochs (0=disabled).",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=None,
        help="Number of warmup epochs for learning rate.",
    )
    parser.add_argument(
        "--use-optimized-hyperparams",
        action="store_true",
        help="Load optimized hyperparameters from config/hyperparameters.json.",
    )
    parser.add_argument(
        "--sampling-weights",
        type=str,
        default="uniform",
        choices=["uniform", "late_game", "phase_emphasis", "combined", "victory_type"],
        help=(
            "Position sampling strategy: 'uniform' (default), 'late_game' "
            "(bias toward endgame), 'phase_emphasis' (weight by game phase), "
            "'combined' (late_game + phase_emphasis), 'victory_type' "
            "(balance across victory types like territory, elimination, etc.)."
        ),
    )
    parser.add_argument(
        "--use-streaming",
        action="store_true",
        help="Use StreamingDataLoader for memory-efficient large dataset training.",
    )
    return parser.parse_args(argv)


def _build_model_id(
    board_type: BoardType,
    num_players: int,
    explicit: Optional[str],
) -> str:
    """Derive a simple model_id for the baseline run."""
    if explicit:
        return explicit

    if board_type == BoardType.SQUARE8:
        prefix = "sq8"
    elif board_type == BoardType.SQUARE19:
        prefix = "sq19"
    elif board_type == BoardType.HEXAGONAL:
        prefix = "hex"
    else:
        prefix = "unknown"

    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{num_players}p_nn_baseline_{ts}"


def _write_report(path: str, payload: Dict[str, Any]) -> None:
    """Write nn_training_report.json with pretty formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # Model hygiene: validate and clean up corrupted models at startup
    if HAS_MODEL_VALIDATION:
        models_dir = Path(PROJECT_ROOT) / "models"
        if models_dir.exists():
            run_startup_validation(models_dir, cleanup=True)

    board_enum = _parse_board(args.board)
    num_players = args.num_players

    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)

    seed = args.seed or 42

    # Start from the board-specific training preset so we stay consistent with
    # other tooling (tier pipeline, future curriculum loops, etc.).
    train_cfg = get_training_config_for_board(board_enum, TrainConfig())
    train_cfg.seed = seed
    train_cfg.board_type = board_enum

    model_id = _build_model_id(board_enum, args.num_players, args.model_id)
    train_cfg.model_id = model_id

    # Optionally clamp epochs from CLI; in demo mode we always cap at 1.
    if args.epochs is not None:
        train_cfg.epochs_per_iter = max(1, int(args.epochs))

    # Load optimized hyperparameters from config file if requested
    hp_source = "preset"
    if args.use_optimized_hyperparams:
        try:
            from app.config.hyperparameters import get_hyperparameters, is_optimized
            board_name = args.board.replace("sq8", "square8").replace("sq19", "square19").replace("hex", "hexagonal")
            hp = get_hyperparameters(board_name, num_players)
            optimized = is_optimized(board_name, num_players)
            hp_source = "optimized" if optimized else "defaults"

            # Apply hyperparameters from config
            train_cfg.learning_rate = hp.get("learning_rate", train_cfg.learning_rate)
            train_cfg.batch_size = hp.get("batch_size", train_cfg.batch_size)
            train_cfg.weight_decay = hp.get("weight_decay", train_cfg.weight_decay)
            train_cfg.policy_weight = hp.get("policy_weight", train_cfg.policy_weight)
            train_cfg.early_stopping_patience = hp.get("early_stopping_patience", 5)
            train_cfg.warmup_epochs = hp.get("warmup_epochs", train_cfg.warmup_epochs)

            print(f"[HP] Loaded {hp_source} hyperparameters for {board_name}_{num_players}p")
            print(f"[HP]   lr={train_cfg.learning_rate:.6f}, batch={train_cfg.batch_size}, "
                  f"wd={train_cfg.weight_decay:.2e}, policy_w={train_cfg.policy_weight:.2f}")
        except Exception as e:
            print(f"[HP] Warning: Failed to load hyperparameters: {e}")

    # Apply CLI hyperparameter overrides (highest priority)
    if args.learning_rate is not None:
        train_cfg.learning_rate = args.learning_rate
        hp_source = "cli"
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
        hp_source = "cli"
    if args.weight_decay is not None:
        train_cfg.weight_decay = args.weight_decay
        hp_source = "cli"
    if args.policy_weight is not None:
        train_cfg.policy_weight = args.policy_weight
    if args.early_stopping_patience is not None:
        train_cfg.early_stopping_patience = args.early_stopping_patience
    if args.warmup_epochs is not None:
        train_cfg.warmup_epochs = args.warmup_epochs

    # Store value_weight for later use (not in TrainConfig by default)
    value_weight = args.value_weight if args.value_weight is not None else 1.0

    if args.demo:
        # Demo / tiny mode:
        #
        # - Use a dummy NPZ path so RingRiftDataset creates a small synthetic
        #   dataset in memory (see RingRiftDataset.__init__).
        # - Run a single epoch with minimal early stopping to keep CI cheap.
        train_cfg.epochs_per_iter = 1

        data_path = os.path.join(
            train_cfg.data_dir,
            "square8_nn_baseline_demo_dummy.npz",
        )
        mode = "demo"
    else:
        # Full mode requires an explicit NPZ path derived from a canonical DB.
        if not args.data_path:
            print(
                "Error: non-demo mode requires --data-path pointing at a "
                "Square-8 2-player NPZ dataset generated from a DB with "
                "Status = canonical in TRAINING_DATA_REGISTRY.md.",
                file=sys.stderr,
            )
            return 1
        data_path = os.path.abspath(args.data_path)
        mode = "full"

    save_path = os.path.join(train_cfg.model_dir, f"{model_id}.pth")
    checkpoint_dir = os.path.join(
        train_cfg.model_dir,
        "checkpoints",
        model_id,
    )

    # Seed all relevant RNGs for reproducibility.
    seed_all(seed)

    # Auto-detect multi-player value targets in dataset (not in demo mode).
    use_multi_player = False
    if not args.demo and os.path.exists(data_path):
        use_multi_player = _dataset_has_multi_player_values(data_path)
        if use_multi_player:
            print(f"Detected multi-player value targets in {data_path}, enabling multi_player mode.")

    # Run training. The current train_model implementation does not surface
    # final loss/metrics directly, so we record structural metadata and stub
    # metrics. Future tasks (A3+) can thread richer metrics through.
    early_stop = 1 if args.demo else train_cfg.early_stopping_patience
    warmup = 0 if args.demo else train_cfg.warmup_epochs
    lr_sched = "none" if args.demo else train_cfg.lr_scheduler

    sampling_weights = args.sampling_weights if not args.demo else 'uniform'
    use_streaming = args.use_streaming if not args.demo else False

    print(f"[Training] Starting with lr={train_cfg.learning_rate:.6f}, "
          f"batch={train_cfg.batch_size}, epochs={train_cfg.epochs_per_iter}, "
          f"early_stop={early_stop}, warmup={warmup}, scheduler={lr_sched}, "
          f"sampling={sampling_weights}")

    train_model(
        config=train_cfg,
        data_path=data_path,
        save_path=save_path,
        early_stopping_patience=early_stop,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=train_cfg.epochs_per_iter,
        warmup_epochs=warmup,
        lr_scheduler=lr_sched,
        multi_player=use_multi_player,
        num_players=num_players,
        model_version=args.model_version,
        sampling_weights=sampling_weights,
        use_streaming=use_streaming,
    )

    created_at = datetime.now(timezone.utc).isoformat()
    report: Dict[str, Any] = {
        "board": args.board,
        "num_players": num_players,
        "mode": mode,
        "model_id": model_id,
        "model_version": args.model_version,
        "data_path": data_path,
        "model_path": save_path,
        "training_params": {
            "board_type": (
                train_cfg.board_type.name if hasattr(train_cfg.board_type, "name") else str(train_cfg.board_type)
            ),
            "epochs_per_iter": train_cfg.epochs_per_iter,
            "batch_size": train_cfg.batch_size,
            "learning_rate": train_cfg.learning_rate,
            "weight_decay": train_cfg.weight_decay,
            "policy_weight": train_cfg.policy_weight,
            "early_stopping_patience": early_stop,
            "warmup_epochs": warmup,
            "lr_scheduler": lr_sched,
            "seed": train_cfg.seed,
            "model_version": args.model_version,
            "hyperparameter_source": hp_source,
            "sampling_weights": sampling_weights,
            "use_streaming": use_streaming,
        },
        # Metrics are intentionally minimal for this baseline demo. We do not
        # currently plumb the final validation loss out of train_model; this
        # stub makes the JSON schema explicit without committing to a specific
        # metric set. Future work can extend this when richer hooks exist.
        "metrics": {
            "final_loss": None,
            "train_epochs_run": train_cfg.epochs_per_iter,
        },
        "created_at": created_at,
    }

    report_path = os.path.join(run_dir, "nn_training_report.json")
    _write_report(report_path, report)

    # For future tooling, return 0 on success so tests can assert on main().
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
