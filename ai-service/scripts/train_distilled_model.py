#!/usr/bin/env python3
"""Train a smaller distilled model using soft policy targets from MCTS.

This script trains a compact model (6 residual blocks, 96 filters) using
the NPZ dataset exported from Gumbel MCTS selfplay with soft policy targets.

Usage:
    python scripts/train_distilled_model.py \
        --data data/training/sq8_kl_distill.npz \
        --output models/distilled_sq8_2p.pth \
        --board-type square8 \
        --num-players 2
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

from app.models import BoardType
from app.training.config import TrainConfig, get_training_config_for_board
from app.training.train import train_model
from scripts.lib.cli import BOARD_TYPE_MAP


def main():
    parser = argparse.ArgumentParser(
        description="Train distilled model from MCTS soft targets"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to NPZ dataset with soft policy targets"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output model path"
    )
    parser.add_argument(
        "--board-type",
        choices=["square8", "square19", "hexagonal"],
        default="square8",
    )
    parser.add_argument(
        "--num-players",
        type=int, choices=[2, 3, 4], default=2,
    )
    parser.add_argument(
        "--model-version",
        choices=["v2", "v3"],
        default="v3",
        help="Model architecture version"
    )
    parser.add_argument(
        "--num-res-blocks",
        type=int, default=6,
        help="Number of residual blocks (default: 6 for compact model)"
    )
    parser.add_argument(
        "--num-filters",
        type=int, default=96,
        help="Number of filters (default: 96 for compact model)"
    )
    parser.add_argument(
        "--epochs",
        type=int, default=50,
        help="Training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int, default=128,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float, default=2e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--model-id",
        type=str, default="distilled_sq8_2p",
        help="Model identifier"
    )
    parser.add_argument(
        "--resume",
        type=str, default=None,
        help="Resume training from checkpoint (for fine-tuning)"
    )

    args = parser.parse_args()

    board_type = BOARD_TYPE_MAP[args.board_type]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create training config
    config = get_training_config_for_board(board_type, TrainConfig())
    config.model_id = args.model_id
    config.epochs_per_iter = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.num_res_blocks = args.num_res_blocks
    config.num_filters = args.num_filters

    print("=" * 60)
    print("DISTILLED MODEL TRAINING")
    print("=" * 60)
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    if args.resume:
        print(f"Fine-tuning from: {args.resume}")
    print(f"Architecture: {args.num_res_blocks} blocks, {args.num_filters} filters")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 60)

    # Train the model
    train_model(
        config=config,
        data_path=args.data,
        save_path=str(output_path),
        checkpoint_dir=str(output_path.parent),
        checkpoint_interval=max(1, args.epochs // 5),
        multi_player=args.num_players > 2,
        num_players=args.num_players,
        model_version=args.model_version,
        num_res_blocks=args.num_res_blocks,
        num_filters=args.num_filters,
        resume_path=args.resume,
    )

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
