#!/usr/bin/env python3
"""Cross-board transfer learning for RingRift neural networks.

Pre-trains on square8 (abundant data), then fine-tunes on square19/hexagonal.
This leverages the shared game mechanics across board types.

Usage:
    python scripts/cross_board_transfer.py --source square8 --target square19
    python scripts/cross_board_transfer.py --source square8 --target hexagonal --epochs 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


def find_best_source_model(
    board_type: str,
    num_players: int = 2,
) -> Path | None:
    """Find the best model for a given board type."""
    models_dir = AI_SERVICE_ROOT / "models"

    # Try various naming patterns
    # Handle common abbreviations: square8->sq8, square19->sq19, hexagonal->hex
    short_name = board_type.replace("square", "sq").replace("hexagonal", "hex")
    patterns = [
        f"ringrift_best_{short_name}_{num_players}p.pth",
        f"ringrift_best_{board_type}_{num_players}p.pth",
        f"ringrift_v*_{short_name}_{num_players}p*.pth",
        f"ringrift_v*_{board_type}_{num_players}p*.pth",
        f"*{short_name}_{num_players}p*.pth",
    ]

    for pattern in patterns:
        matches = list(models_dir.glob(pattern))
        if matches:
            # Return most recent
            return max(matches, key=lambda p: p.stat().st_mtime)

    return None


def transfer_and_finetune(
    source_board: str,
    target_board: str,
    num_players: int = 2,
    epochs: int = 30,
    learning_rate: float = 1e-4,  # Lower LR for fine-tuning
    freeze_early_layers: bool = True,
) -> dict:
    """Transfer weights from source model and fine-tune on target data.

    Args:
        source_board: Source board type (e.g., 'square8')
        target_board: Target board type (e.g., 'square19')
        num_players: Number of players
        epochs: Fine-tuning epochs
        learning_rate: Learning rate for fine-tuning
        freeze_early_layers: Whether to freeze early layers

    Returns:
        Training results dict
    """
    import torch

    from app.models import BoardType
    from app.training.config import TrainConfig
    from app.training.train import train_from_file

    # Find source model
    source_model_path = find_best_source_model(source_board, num_players)
    if not source_model_path:
        raise ValueError(f"No source model found for {source_board} {num_players}p")

    print(f"Source model: {source_model_path}")

    # Find target training data
    target_data_patterns = [
        AI_SERVICE_ROOT / "data" / "training" / f"daemon_{target_board}_{num_players}p.npz",
        AI_SERVICE_ROOT / "data" / "training" / f"{target_board}_{num_players}p.npz",
    ]

    target_data_path = None
    for p in target_data_patterns:
        if p.exists():
            target_data_path = p
            break

    if not target_data_path:
        raise ValueError(f"No training data found for {target_board} {num_players}p")

    print(f"Target data: {target_data_path}")

    # Load source model
    from app.utils.torch_utils import safe_load_checkpoint
    safe_load_checkpoint(source_model_path, map_location="cpu")

    # Create output path
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = AI_SERVICE_ROOT / "models" / "transfer"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"transfer_{source_board}_to_{target_board}_{num_players}p_{timestamp}.pth"

    # Configure training
    board_type_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
        "hex8": BoardType.HEX8,
    }

    config = TrainConfig(
        board_type=board_type_map.get(target_board, BoardType.SQUARE8),
        epochs_per_iter=epochs,
        learning_rate=learning_rate,
        batch_size=64,
        weight_decay=1e-5,  # Lower weight decay for fine-tuning
        model_id=f"transfer_{source_board}_to_{target_board}_{num_players}p",
    )

    # Train with transfer
    print(f"\nFine-tuning on {target_board} data for {epochs} epochs...")
    print(f"Learning rate: {learning_rate}, Freeze early: {freeze_early_layers}")

    start_time = time.time()

    losses = train_from_file(
        data_path=str(target_data_path),
        output_path=str(output_path),
        config=config,
        initial_model_path=str(source_model_path),
    )

    duration = time.time() - start_time

    results = {
        "source_board": source_board,
        "target_board": target_board,
        "num_players": num_players,
        "source_model": str(source_model_path),
        "output_model": str(output_path),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "duration_seconds": round(duration, 1),
        "final_loss": losses.get("total", 0),
        "policy_loss": losses.get("policy", 0),
        "value_loss": losses.get("value", 0),
    }

    print(f"\nTransfer learning complete in {duration:.1f}s")
    print(f"Output: {output_path}")
    print(f"Final loss: {losses.get('total', 0):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-board transfer learning")
    parser.add_argument("--source", default="square8", help="Source board type")
    parser.add_argument("--target", required=True, help="Target board type")
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    parser.add_argument("--epochs", type=int, default=30, help="Fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--no-freeze", action="store_true", help="Don't freeze early layers")
    parser.add_argument("--output", help="Output JSON for results")

    args = parser.parse_args()

    print("Cross-Board Transfer Learning")
    print("=" * 60)
    print(f"Source: {args.source} -> Target: {args.target}")
    print(f"Players: {args.players}")
    print()

    try:
        results = transfer_and_finetune(
            source_board=args.source,
            target_board=args.target,
            num_players=args.players,
            epochs=args.epochs,
            learning_rate=args.lr,
            freeze_early_layers=not args.no_freeze,
        )

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
