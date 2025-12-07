#!/usr/bin/env python
"""Baseline neural-network training experiment for Square-8 2-player.

This script provides a small, reproducible NN training entrypoint
for Square-8 2-player games using the existing training stack:

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


def _parse_board(board: str) -> BoardType:
    """Map CLI board string to BoardType (Square-8 only for A2)."""
    b = board.lower()
    if b in {"square8", "sq8"}:
        return BoardType.SQUARE8
    raise SystemExit(
        f"Unsupported board {board!r}; this baseline script currently "
        "supports only square8 2-player."
    )


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
        help=(
            "Board identifier (currently only 'square8' is supported)."
        ),
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help=(
            "Number of players (baseline is designed for 2-player only)."
        ),
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help=(
            "Output dir for nn_training_report.json and checkpoints. "
            "Will be created if needed."
        ),
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
            "Optional explicit nn_model_id / checkpoint stem. When omitted, "
            "a timestamped baseline id is generated."
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

    board_enum = _parse_board(args.board)
    if args.num_players != 2:
        print(
            f"Warning: num_players={args.num_players} is not the canonical "
            "2-player baseline; proceeding but this script is tuned for 2p.",
            file=sys.stderr,
        )

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

    # Run training. The current train_model implementation does not surface
    # final loss/metrics directly, so we record structural metadata and stub
    # metrics. Future tasks (A3+) can thread richer metrics through.
    train_model(
        config=train_cfg,
        data_path=data_path,
        save_path=save_path,
        early_stopping_patience=1 if args.demo else 5,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=train_cfg.epochs_per_iter,
        warmup_epochs=0,
        lr_scheduler="none",
    )

    created_at = datetime.now(timezone.utc).isoformat()
    report: Dict[str, Any] = {
        "board": "square8",
        "num_players": args.num_players,
        "mode": mode,
        "model_id": model_id,
        "data_path": data_path,
        "model_path": save_path,
        "training_params": {
            "board_type": train_cfg.board_type.name
            if hasattr(train_cfg.board_type, "name")
            else str(train_cfg.board_type),
            "epochs_per_iter": train_cfg.epochs_per_iter,
            "batch_size": train_cfg.batch_size,
            "learning_rate": train_cfg.learning_rate,
            "seed": train_cfg.seed,
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