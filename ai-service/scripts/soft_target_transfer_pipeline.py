#!/usr/bin/env python3
"""Soft-target transfer learning pipeline.

Combines GPU MCTS soft-target data with transfer learning for efficient
model improvement:

1. Generate soft-target training data with GPU MCTS
2. (Optional) Transfer weights from existing model
3. Fine-tune with soft policy targets
4. Validate and export

Usage:
    # Full pipeline: generate data + train from scratch
    python scripts/soft_target_transfer_pipeline.py \
        --board-type hex8 --num-players 4 \
        --num-games 1000 \
        --output models/hex8_4p_soft.pth

    # Transfer learning: use existing 2p model
    python scripts/soft_target_transfer_pipeline.py \
        --board-type hex8 --num-players 4 \
        --num-games 1000 \
        --init-model models/canonical_hex8_2p.pth \
        --output models/hex8_4p_transfer.pth

    # Fine-tune existing model with fresh soft targets
    python scripts/soft_target_transfer_pipeline.py \
        --board-type hex8 --num-players 2 \
        --num-games 2000 \
        --init-model models/canonical_hex8_2p.pth \
        --output models/hex8_2p_softfine.pth
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_soft_targets(
    board_type: str,
    num_players: int,
    num_games: int,
    output_path: str,
    device: str = "cuda",
    simulation_budget: int = 64,
) -> str:
    """Generate soft-target training data with GPU MCTS."""
    from app.training.gpu_mcts_selfplay import run_gpu_mcts_selfplay

    logger.info(f"Generating {num_games} games with GPU MCTS...")
    logger.info(f"  Board: {board_type}, Players: {num_players}")
    logger.info(f"  Simulations: {simulation_budget}, Device: {device}")

    games = run_gpu_mcts_selfplay(
        board_type=board_type,
        num_players=num_players,
        num_games=num_games,
        output_path=output_path,
        device=device,
        encoder_version="v3",
    )

    total_samples = sum(len(g.samples) for g in games)
    completed = sum(1 for g in games if g.termination_reason == "normal")

    logger.info(f"Generated {total_samples} samples from {completed}/{num_games} completed games")
    return output_path


def transfer_weights(
    source_model: str,
    target_num_players: int,
    board_type: str,
) -> dict:
    """Load and adapt model weights for transfer learning.

    Handles:
    - Same player count: direct copy
    - 2p -> 4p: resize value head
    - Different board: copy compatible layers only
    """
    from app.utils.torch_utils import safe_load_checkpoint
    logger.info(f"Loading source model: {source_model}")
    checkpoint = safe_load_checkpoint(source_model, map_location="cpu")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    else:
        state_dict = checkpoint
        metadata = {}

    # Check source model configuration
    source_num_players = metadata.get("num_players", 2)

    if source_num_players == target_num_players:
        logger.info(f"Same player count ({source_num_players}), using direct transfer")
        return state_dict

    # Need to resize value head
    logger.info(f"Resizing value head: {source_num_players}p -> {target_num_players}p")

    new_state_dict = {}
    for key, value in state_dict.items():
        if "value_fc2" in key:
            if "weight" in key and value.shape[0] == source_num_players:
                # Extend value head weights
                new_weight = torch.zeros(target_num_players, value.shape[1])
                new_weight[:source_num_players] = value
                # Initialize new player outputs randomly
                torch.nn.init.xavier_uniform_(new_weight[source_num_players:])
                new_state_dict[key] = new_weight
                logger.info(f"  Resized {key}: {value.shape} -> {new_weight.shape}")
            elif "bias" in key and value.shape[0] == source_num_players:
                # Extend bias
                new_bias = torch.zeros(target_num_players)
                new_bias[:source_num_players] = value
                new_state_dict[key] = new_bias
                logger.info(f"  Resized {key}: {value.shape} -> {new_bias.shape}")
            else:
                new_state_dict[key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict


def train_with_soft_targets(
    data_path: str,
    output_path: str,
    board_type: str,
    num_players: int,
    init_weights: dict | None = None,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
) -> dict:
    """Train model with soft policy targets."""
    from app.models import BoardType
    from app.training.train import TrainConfig, train_model

    board_type_enum = getattr(BoardType, board_type.upper())

    config = TrainConfig(
        board_type=board_type_enum,
        num_players=num_players,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        model_version="v2",
    )

    logger.info(f"Training with soft targets: {data_path}")
    logger.info(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")

    # Train the model
    metrics = train_model(
        config=config,
        data_path=data_path,
        save_path=output_path,
        early_stopping_patience=5,
    )

    logger.info(f"Training complete: {output_path}")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")

    return metrics


def validate_output(model_path: str, data_path: str) -> bool:
    """Quick validation of the trained model."""
    logger.info("Validating output model...")

    # Check model file exists and can be loaded
    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        return False

    try:
        from app.utils.torch_utils import safe_load_checkpoint
        checkpoint = safe_load_checkpoint(model_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        n_params = sum(p.numel() for p in state_dict.values())
        logger.info(f"  Model parameters: {n_params:,}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

    # Validate training data
    try:
        data = np.load(data_path)
        n_samples = len(data["features"])
        logger.info(f"  Training samples: {n_samples:,}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Soft-target transfer learning pipeline")

    # Required arguments
    parser.add_argument("--board-type", required=True, help="Board type")
    parser.add_argument("--num-players", type=int, required=True, help="Number of players")
    parser.add_argument("--output", required=True, help="Output model path")

    # Data generation
    parser.add_argument("--num-games", type=int, default=1000, help="Games to generate")
    parser.add_argument("--data-path", help="Use existing data instead of generating")
    parser.add_argument("--simulation-budget", type=int, default=64, help="MCTS simulations")
    parser.add_argument("--device", default="cuda", help="Device for GPU MCTS")

    # Transfer learning
    parser.add_argument("--init-model", help="Initialize from existing model")

    # Training
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")

    # Options
    parser.add_argument("--keep-data", action="store_true", help="Keep generated data")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation")

    args = parser.parse_args()

    # Determine data path
    if args.data_path:
        data_path = args.data_path
        logger.info(f"Using existing data: {data_path}")
    else:
        # Generate soft-target data
        if args.keep_data:
            data_dir = Path("data/training/pipeline")
            data_dir.mkdir(parents=True, exist_ok=True)
            data_path = str(data_dir / f"soft_{args.board_type}_{args.num_players}p.npz")
        else:
            data_path = tempfile.mktemp(suffix=".npz")

        generate_soft_targets(
            board_type=args.board_type,
            num_players=args.num_players,
            num_games=args.num_games,
            output_path=data_path,
            device=args.device,
            simulation_budget=args.simulation_budget,
        )

    # Load/adapt initial weights if specified
    init_weights = None
    if args.init_model:
        init_weights = transfer_weights(
            source_model=args.init_model,
            target_num_players=args.num_players,
            board_type=args.board_type,
        )

    # Train with soft targets
    metrics = train_with_soft_targets(
        data_path=data_path,
        output_path=args.output,
        board_type=args.board_type,
        num_players=args.num_players,
        init_weights=init_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    # Validate output
    if not args.skip_validation:
        valid = validate_output(args.output, data_path)
        if not valid:
            logger.error("Validation failed!")
            sys.exit(1)

    # Cleanup temporary data
    if not args.data_path and not args.keep_data:
        try:
            os.unlink(data_path)
            logger.info("Cleaned up temporary data file")
        except Exception:
            pass

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"  Model: {args.output}")
    if "policy_accuracy" in metrics:
        logger.info(f"  Policy accuracy: {metrics['policy_accuracy']:.2%}")
    if "value_loss" in metrics:
        logger.info(f"  Value loss: {metrics['value_loss']:.4f}")


if __name__ == "__main__":
    main()
