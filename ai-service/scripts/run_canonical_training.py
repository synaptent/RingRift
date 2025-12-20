#!/usr/bin/env python
"""
Canonical training entrypoint: export a dataset from a canonical DB and train.

This consolidates the canonical training workflow into a single script:
  1) Export NPZ data from a canonical GameReplayDB
  2) Train a neural model with aligned history_length + feature_version

Usage (from ai-service/):
  PYTHONPATH=. python scripts/run_canonical_training.py \
    --db data/games/canonical_square8.db \
    --board-type square8 \
    --num-players 2 \
    --model-version v3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models import BoardType  # type: ignore[import]
from app.training.config import TrainConfig, get_training_config_for_board  # type: ignore[import]
from app.training.generate_data import (  # type: ignore[import]
    _assert_db_is_canonical_if_summary_exists,
)
from app.training.train import train_model  # type: ignore[import]
from scripts.export_replay_dataset import export_replay_dataset  # type: ignore[import]
from scripts.lib.cli import BOARD_TYPE_MAP


def _default_model_version(board_type: BoardType) -> str:
    if board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
        return "v3"
    if board_type == BoardType.SQUARE8:
        return "v3"
    return "v2"


def _validate_canonical_db(db_path: Path, allow_noncanonical: bool) -> None:
    if allow_noncanonical:
        return
    if not db_path.name.startswith("canonical_"):
        raise SystemExit(
            f"[canonical-training] Refusing to train from non-canonical DB: {db_path}\n"
            "Expected basename starting with 'canonical_'. Pass --allow-noncanonical "
            "to override for ad-hoc experiments."
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export canonical training data and train a model in one step.",
    )
    parser.add_argument("--db", required=True, help="Path to canonical GameReplayDB.")
    parser.add_argument(
        "--board-type",
        required=True,
        choices=["square8", "square19", "hex8", "hexagonal"],
        help="Board type for export/training.",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        required=True,
        choices=[2, 3, 4],
        help="Number of players to export/train for.",
    )
    parser.add_argument(
        "--model-version",
        choices=["v2", "v3", "v4", "hex"],
        default=None,
        help="Model architecture version (default: board-aware).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override nn_model_id / checkpoint stem (optional).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Explicit checkpoint output path (optional).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/training",
        help="Directory to write exported NPZ dataset (default: data/training).",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=3,
        help="Number of history frames to stack (default: 3).",
    )
    parser.add_argument(
        "--feature-version",
        type=int,
        default=2,
        help=(
            "Feature encoding version for global feature layout (default: 2). "
            "Use 1 to keep legacy hex globals without chain/FE flags."
        ),
    )
    parser.add_argument(
        "--encoder-version",
        choices=["default", "v2", "v3"],
        default="default",
        help="Hex encoder version to use when exporting (default: v3).",
    )
    parser.add_argument(
        "--legacy-policy-encoding",
        action="store_true",
        help="Use legacy MAX_N policy encoding (not recommended for v3/v4).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs per iter (optional).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (optional).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed (default: 42).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a tiny demo configuration (fast smoke test).",
    )
    parser.add_argument(
        "--allow-noncanonical",
        action="store_true",
        help="Allow non-canonical DBs (use for experiments only).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    board_type = BOARD_TYPE_MAP[args.board_type]
    db_path = Path(args.db)

    _validate_canonical_db(db_path, args.allow_noncanonical)
    try:
        _assert_db_is_canonical_if_summary_exists(
            db_path,
            allow_noncanonical=args.allow_noncanonical,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    model_version = args.model_version or _default_model_version(board_type)
    use_board_aware_encoding = not args.legacy_policy_encoding

    if model_version in ("v3", "v4") and not use_board_aware_encoding:
        raise SystemExit(
            "[canonical-training] v3/v4 models require board-aware policy encoding. "
            "Remove --legacy-policy-encoding or use --model-version v2."
        )

    dataset_dir = Path(args.output_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = (
        f"{db_path.stem}.{args.board_type}.{args.num_players}p."
        f"hl{args.history_length}."
        f"fv{args.feature_version}.npz"
    )
    dataset_path = dataset_dir / dataset_name

    export_replay_dataset(
        db_path=str(db_path),
        board_type=board_type,
        num_players=args.num_players,
        output_path=str(dataset_path),
        history_length=args.history_length,
        feature_version=args.feature_version,
        use_board_aware_encoding=use_board_aware_encoding,
        encoder_version=args.encoder_version,
    )

    config = get_training_config_for_board(board_type, TrainConfig())
    config.feature_version = args.feature_version
    config.history_length = args.history_length
    config.seed = args.seed
    if args.model_id:
        config.model_id = args.model_id
    if args.epochs is not None:
        config.epochs_per_iter = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    if args.demo:
        config.epochs_per_iter = max(1, min(config.epochs_per_iter, 2))
        config.batch_size = max(32, min(config.batch_size, 128))

    save_path = args.save_path or os.path.join(
        config.model_dir,
        f"{config.model_id}.pth",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_kwargs = {
        "config": config,
        "data_path": str(dataset_path),
        "save_path": save_path,
        "checkpoint_dir": os.path.dirname(save_path),
        "checkpoint_interval": max(1, config.epochs_per_iter // 2),
        "multi_player": args.num_players > 2,
        "num_players": args.num_players,
        "model_version": model_version,
    }
    if args.demo:
        train_kwargs.update(
            early_stopping_patience=0,
            warmup_epochs=0,
            lr_scheduler="none",
        )
    train_model(**train_kwargs)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
