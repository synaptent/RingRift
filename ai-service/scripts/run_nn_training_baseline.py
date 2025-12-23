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
from typing import Any, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path

import numpy as np

from app.models import BoardType
from app.training.config import (
    TrainConfig,
    get_training_config_for_board,
)
from app.training.seed_utils import seed_all
from app.training.train import train_model

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


def _detect_dataset_feature_version(data_path: str) -> int | None:
    """Auto-detect feature_version from NPZ dataset metadata.

    Returns:
        Feature version (1, 2, or 3) if found in dataset, None otherwise.
        None signals legacy data without version metadata (treated as v1).
    """
    try:
        with np.load(data_path, mmap_mode='r', allow_pickle=True) as f:
            if 'feature_version' in f.files:
                return int(np.asarray(f['feature_version']).item())
    except Exception:
        pass
    return None


def _parse_board(board: str) -> BoardType:
    """Map CLI board string to BoardType."""
    b = board.lower()
    if b in {"square8", "sq8"}:
        return BoardType.SQUARE8
    if b in {"square19", "sq19"}:
        return BoardType.SQUARE19
    if b in {"hex8"}:
        return BoardType.HEX8
    if b in {"hexagonal", "hex"}:
        return BoardType.HEXAGONAL
    raise SystemExit(f"Unsupported board {board!r}; valid options: square8, square19, hex8, hexagonal")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        choices=["square8", "sq8", "square19", "sq19", "hex8", "hexagonal", "hex"],
        help=("Board identifier: square8, square19, hex8, or hexagonal."),
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
    # 2024-12 Training Improvements
    parser.add_argument(
        "--spectral-norm",
        action="store_true",
        help="Enable spectral normalization for gradient stability.",
    )
    parser.add_argument(
        "--cyclic-lr",
        action="store_true",
        help="Enable cyclic learning rate with triangular waves.",
    )
    parser.add_argument(
        "--cyclic-lr-period",
        type=int,
        default=5,
        help="Cyclic LR period in epochs (default: 5).",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training (FP16/BF16).",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="AMP dtype (default: bfloat16).",
    )
    parser.add_argument(
        "--value-whitening",
        action="store_true",
        help="Enable value head whitening for stable training.",
    )
    parser.add_argument(
        "--value-whitening-momentum",
        type=float,
        default=0.99,
        help="Momentum for value whitening running stats (default: 0.99).",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help="Enable Model EMA for better generalization.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA decay factor (default: 0.999).",
    )
    parser.add_argument(
        "--stochastic-depth",
        action="store_true",
        help="Enable stochastic depth regularization.",
    )
    parser.add_argument(
        "--stochastic-depth-prob",
        type=float,
        default=0.1,
        help="Drop probability for stochastic depth (default: 0.1).",
    )
    parser.add_argument(
        "--adaptive-warmup",
        action="store_true",
        help="Use adaptive warmup based on dataset size.",
    )
    parser.add_argument(
        "--hard-example-mining",
        action="store_true",
        help="Enable hard example mining for focused training.",
    )
    parser.add_argument(
        "--hard-example-top-k",
        type=float,
        default=0.3,
        help="Top K percent of hardest examples to upweight (default: 0.3).",
    )
    # 2025-12 Training Improvements
    parser.add_argument(
        "--policy-label-smoothing",
        type=float,
        default=0.0,
        help="Policy label smoothing factor (0=disabled, typical: 0.05-0.1).",
    )
    parser.add_argument(
        "--augment-hex-symmetry",
        action="store_true",
        help="Enable D6 symmetry augmentation for hex boards (12x effective data).",
    )
    parser.add_argument(
        "--no-gauntlet",
        action="store_true",
        help=(
            "Disable baseline gauntlet evaluation during training. "
            "Useful when training new architectures that may have incompatible "
            "checkpoints, or for faster experimental runs."
        ),
    )
    return parser.parse_args(argv)


def _build_model_id(
    board_type: BoardType,
    num_players: int,
    explicit: str | None,
) -> str:
    """Derive a simple model_id for the baseline run."""
    if explicit:
        return explicit

    if board_type == BoardType.SQUARE8:
        prefix = "sq8"
    elif board_type == BoardType.SQUARE19:
        prefix = "sq19"
    elif board_type == BoardType.HEX8:
        prefix = "hex8"
    elif board_type == BoardType.HEXAGONAL:
        prefix = "hexagonal"
    else:
        prefix = "unknown"

    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{num_players}p_nn_baseline_{ts}"


def _write_report(path: str, payload: dict[str, Any]) -> None:
    """Write nn_training_report.json with pretty formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
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
            # Map short names to canonical names, but keep hex8 as-is
            board_name = args.board
            if board_name == "sq8":
                board_name = "square8"
            elif board_name == "sq19":
                board_name = "square19"
            elif board_name == "hex":
                board_name = "hexagonal"
            # hex8 stays as hex8
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

    # Auto-detect feature_version from dataset metadata (not in demo mode).
    # This allows training to work with datasets of different feature versions
    # without requiring explicit --feature-version flags.
    if not args.demo and os.path.exists(data_path):
        detected_fv = _detect_dataset_feature_version(data_path)
        if detected_fv is not None:
            if detected_fv != train_cfg.feature_version:
                print(f"[Auto-detect] Dataset has feature_version={detected_fv}, "
                      f"adjusting config from {train_cfg.feature_version}")
                train_cfg.feature_version = detected_fv
        else:
            # Legacy dataset without feature_version metadata - use v1
            if train_cfg.feature_version != 1:
                print(f"[Auto-detect] Dataset missing feature_version (legacy), "
                      f"adjusting config from {train_cfg.feature_version} to 1")
                train_cfg.feature_version = 1

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

    # Collect 2024-12 training improvements from args
    training_improvements = {
        'spectral_norm': getattr(args, 'spectral_norm', False),
        'cyclic_lr': getattr(args, 'cyclic_lr', False),
        'cyclic_lr_period': getattr(args, 'cyclic_lr_period', 5),
        'mixed_precision': getattr(args, 'mixed_precision', False),
        'amp_dtype': getattr(args, 'amp_dtype', 'bfloat16'),
        'value_whitening': getattr(args, 'value_whitening', False),
        'value_whitening_momentum': getattr(args, 'value_whitening_momentum', 0.99),
        'ema': getattr(args, 'ema', False),
        'ema_decay': getattr(args, 'ema_decay', 0.999),
        'stochastic_depth': getattr(args, 'stochastic_depth', False),
        'stochastic_depth_prob': getattr(args, 'stochastic_depth_prob', 0.1),
        'adaptive_warmup': getattr(args, 'adaptive_warmup', False),
        'hard_example_mining': getattr(args, 'hard_example_mining', False),
        'hard_example_top_k': getattr(args, 'hard_example_top_k', 0.3),
        # 2025-12 Training Improvements
        'policy_label_smoothing': getattr(args, 'policy_label_smoothing', 0.0),
        'augment_hex_symmetry': getattr(args, 'augment_hex_symmetry', False),
    }

    # Log enabled improvements
    enabled = [k for k, v in training_improvements.items() if v and k not in ['cyclic_lr_period', 'amp_dtype', 'value_whitening_momentum', 'ema_decay', 'stochastic_depth_prob', 'hard_example_top_k', 'policy_label_smoothing']]
    if training_improvements.get('policy_label_smoothing', 0) > 0:
        enabled.append(f"policy_label_smoothing={training_improvements['policy_label_smoothing']}")
    if enabled:
        print(f"[Training] 2024-12 improvements enabled: {', '.join(enabled)}")

    # Handle --no-gauntlet flag to disable baseline evaluation
    if args.no_gauntlet:
        training_improvements['enable_background_eval'] = False
        print("[Training] Baseline gauntlet evaluation DISABLED (--no-gauntlet)")

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
        **training_improvements,
    )

    created_at = datetime.now(timezone.utc).isoformat()
    report: dict[str, Any] = {
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

    # Register model in the model registry for tracking
    try:
        from app.training.training_registry import register_trained_model
        registered_id = register_trained_model(
            model_path=save_path,
            board_type=args.board,
            num_players=num_players,
            training_config=report["training_params"],
            metrics=report.get("metrics"),
            description=f"Baseline training {mode} mode",
            tags=["baseline", mode],
            source="nn_training_baseline",
            data_path=data_path,
        )
        if registered_id:
            print(f"[Registry] Registered model as {registered_id}")
    except Exception as e:
        print(f"[Registry] Warning: Could not register model: {e}")

    # For future tooling, return 0 on success so tests can assert on main().
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
