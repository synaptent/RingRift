#!/usr/bin/env python
"""Distill a trained neural network model to NNUE for fast CPU inference.

This script takes a trained RingRift neural network model and distills its
value predictions into a smaller NNUE (Efficiently Updatable Neural Network)
model optimized for alpha-beta search in Minimax at difficulty 4+.

The distillation process:
1. Load the teacher NN model and student NNUE model
2. Sample positions from game replay databases
3. For each position:
   - Get NN value prediction (teacher signal)
   - Train NNUE to match the NN value (MSE loss)
4. Validate distillation quality (MSE, correlation, sign agreement)
5. Save the trained NNUE checkpoint

The NNUE model is ~100x smaller than the full NN and designed for CPU inference,
making it suitable for real-time alpha-beta search during gameplay.

Usage examples
--------------

From the ``ai-service`` root::

    # Basic distillation from a trained NN model
    python scripts/distill_to_nnue.py \\
        --nn-model models/nn_v2_square8.pt \\
        --db data/games/selfplay.db \\
        --output models/nnue/nnue_square8.pt \\
        --board-type square8

    # Distillation with custom training parameters
    python scripts/distill_to_nnue.py \\
        --nn-model models/nn_best.pt \\
        --db data/games/corpus.db \\
        --db data/games/cmaes_corpus.db \\
        --output models/nnue/nnue_distilled.pt \\
        --board-type square8 \\
        --num-positions 200000 \\
        --batch-size 512 \\
        --epochs 20 \\
        --lr 0.001

    # Dry run to check position count and NN loading
    python scripts/distill_to_nnue.py \\
        --nn-model models/nn_v2_square8.pt \\
        --db data/games/selfplay.db \\
        --output models/nnue/nnue_test.pt \\
        --board-type square8 \\
        --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

# Allow imports from app/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

from app.ai.nnue import (
    RingRiftNNUE,
    extract_features_from_gamestate,
    get_feature_dim,
)
from app.models import BoardType, GameState
from app.db import GameReplayDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("distill_to_nnue")


@dataclass
class DistillationConfig:
    """Configuration for NNUE distillation."""
    board_type: BoardType = BoardType.SQUARE8
    num_positions: int = 100_000
    batch_size: int = 256
    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    val_split: float = 0.1
    sample_every_n_moves: int = 1
    min_game_length: int = 10
    seed: int = 42


# =============================================================================
# Teacher Model Loading
# =============================================================================

def load_teacher_model(
    model_path: str,
    board_type: BoardType,
    device: str = "cpu",
) -> nn.Module:
    """Load a trained NN model to use as the teacher for distillation.

    Supports loading from various checkpoint formats:
    - Direct state_dict
    - Dict with 'model_state_dict' key
    - Full checkpoint with architecture info

    Args:
        model_path: Path to the NN checkpoint
        board_type: Board type for architecture selection
        device: Device to load model on

    Returns:
        Loaded and eval-mode NN model
    """
    # Import here to avoid circular dependencies
    from app.ai.neural_net import (
        RingRiftCNN_v2,
        RingRiftCNN_v2_Lite,
        RingRiftCNN_v3,
        RingRiftCNN_v3_12f,
        BOARD_SPATIAL_SIZES,
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Teacher model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    board_size = BOARD_SPATIAL_SIZES.get(board_type, 8)

    # Try to detect architecture from checkpoint
    arch_version = None
    if isinstance(checkpoint, dict):
        arch_version = checkpoint.get("architecture_version")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    else:
        state_dict = checkpoint

    # Infer architecture from state_dict keys or explicit version
    model: nn.Module

    if arch_version == "v3.0.0" or "spatial_policy" in str(state_dict.keys()):
        logger.info(f"Loading RingRiftCNN_v3 architecture (board_size={board_size})")
        model = RingRiftCNN_v3(board_size=board_size)
    elif arch_version == "v3.0.0-12f":
        logger.info(f"Loading RingRiftCNN_v3_12f architecture (board_size={board_size})")
        model = RingRiftCNN_v3_12f(board_size=board_size)
    elif arch_version == "v2.0.0-lite" or "num_filters: 96" in str(checkpoint):
        logger.info(f"Loading RingRiftCNN_v2_Lite architecture (board_size={board_size})")
        model = RingRiftCNN_v2_Lite(board_size=board_size)
    else:
        # Default to v2 architecture
        logger.info(f"Loading RingRiftCNN_v2 architecture (board_size={board_size})")
        model = RingRiftCNN_v2(board_size=board_size)

    # Load weights
    try:
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded teacher model from {model_path}")
    except Exception as e:
        logger.warning(f"Partial state_dict load: {e}")
        # Try loading with strict=False to handle architecture mismatches
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model


def get_nn_value(
    model: nn.Module,
    game_state: GameState,
    player_number: int,
    device: str = "cpu",
) -> float:
    """Get the value prediction from the teacher NN model.

    Args:
        model: Teacher NN model
        game_state: Game state to evaluate
        player_number: Player perspective for value
        device: Device model is on

    Returns:
        Value in [-1, 1] from the NN
    """
    # Import feature extraction from neural_net module
    from app.ai.neural_net import (
        encode_state_to_tensor,
        encode_global_features,
        BOARD_SPATIAL_SIZES,
    )

    board_type = game_state.board_type
    board_size = BOARD_SPATIAL_SIZES.get(board_type, 8)

    # Encode features for NN
    features = encode_state_to_tensor(game_state, board_size=board_size)
    globals_vec = encode_global_features(game_state)

    # Get value prediction
    with torch.no_grad():
        x = torch.from_numpy(features[None, ...]).float().to(device)
        g = torch.from_numpy(globals_vec[None, ...]).float().to(device)
        output = model(x, g)

        # Handle different output formats (v2 vs v3)
        if isinstance(output, tuple):
            if len(output) >= 2:
                value_tensor = output[0]  # (batch, num_players)
            else:
                value_tensor = output[0]
        else:
            value_tensor = output

        # Get value for the specified player (0-indexed)
        player_idx = player_number - 1
        if value_tensor.dim() > 1 and value_tensor.size(1) > player_idx:
            value = float(value_tensor[0, player_idx].item())
        else:
            value = float(value_tensor[0].item())

    return value


# =============================================================================
# Distillation Dataset
# =============================================================================

class DistillationDataset(IterableDataset):
    """Dataset that generates (NNUE_features, NN_value) pairs for distillation.

    Streams positions from game databases, extracts features for NNUE,
    and generates teacher labels from the NN model.
    """

    def __init__(
        self,
        db_paths: List[str],
        teacher_model: nn.Module,
        config: DistillationConfig,
        device: str = "cpu",
        max_positions: Optional[int] = None,
    ):
        self.db_paths = db_paths
        self.teacher_model = teacher_model
        self.config = config
        self.device = device
        self.max_positions = max_positions or config.num_positions
        self.feature_dim = get_feature_dim(config.board_type)
        self.rng = np.random.default_rng(config.seed)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over (features, value) pairs."""
        positions_yielded = 0
        db_paths = self.db_paths.copy()
        self.rng.shuffle(db_paths)

        for db_path in db_paths:
            if positions_yielded >= self.max_positions:
                break

            if not os.path.exists(db_path):
                logger.warning(f"Database not found: {db_path}")
                continue

            try:
                db = GameReplayDB(db_path)
                for game_meta, initial_state, moves in db.iterate_games():
                    if positions_yielded >= self.max_positions:
                        break

                    game_id = game_meta.get("game_id")
                    total_moves = game_meta.get("total_moves", 0)

                    # Skip short games
                    if total_moves < self.config.min_game_length:
                        continue

                    # Sample positions from this game
                    for sample in self._sample_from_game(
                        db, game_id, initial_state, moves, total_moves
                    ):
                        if positions_yielded >= self.max_positions:
                            break
                        yield sample
                        positions_yielded += 1

            except Exception as e:
                logger.error(f"Error processing {db_path}: {e}")
                continue

        logger.info(f"Dataset iteration complete: {positions_yielded} positions")

    def _sample_from_game(
        self,
        db: GameReplayDB,
        game_id: str,
        initial_state: GameState,
        moves: list,
        total_moves: int,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Sample positions from a single game and generate distillation pairs."""
        # Try to get positions from state snapshots first
        for move_num in range(0, total_moves, self.config.sample_every_n_moves):
            try:
                state = db.get_state_at_move(game_id, move_num)
                if state is None:
                    continue

                # Get current player
                current_player = state.current_player
                if current_player is None or current_player < 1:
                    current_player = 1

                # Get teacher NN value
                nn_value = get_nn_value(
                    self.teacher_model,
                    state,
                    current_player,
                    self.device,
                )

                # Get NNUE features
                nnue_features = extract_features_from_gamestate(state, current_player)

                # Convert to tensors
                features_tensor = torch.from_numpy(nnue_features).float()
                value_tensor = torch.tensor([nn_value], dtype=torch.float32)

                yield features_tensor, value_tensor

            except Exception as e:
                logger.debug(f"Error sampling {game_id}:{move_num}: {e}")
                continue


# =============================================================================
# Training Loop
# =============================================================================

def train_nnue_distillation(
    teacher_model: nn.Module,
    student_model: RingRiftNNUE,
    db_paths: List[str],
    config: DistillationConfig,
    device: str = "cpu",
) -> Tuple[RingRiftNNUE, dict]:
    """Train NNUE via distillation from NN teacher.

    Args:
        teacher_model: Trained NN model (frozen)
        student_model: NNUE model to train
        db_paths: Paths to game databases
        config: Training configuration
        device: Device for training

    Returns:
        Tuple of (trained NNUE model, training metrics dict)
    """
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    student_model = student_model.to(device)
    student_model.train()

    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.learning_rate * 0.01,
    )

    criterion = nn.MSELoss()

    # Split positions for train/val
    num_train = int(config.num_positions * (1 - config.val_split))
    num_val = config.num_positions - num_train

    metrics = {
        "train_losses": [],
        "val_losses": [],
        "val_correlations": [],
        "val_sign_agreements": [],
    }

    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(config.epochs):
        # Training
        train_dataset = DistillationDataset(
            db_paths=db_paths,
            teacher_model=teacher_model,
            config=config,
            device=device,
            max_positions=num_train,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=0,  # IterableDataset doesn't support multi-process
        )

        student_model.train()
        train_loss = 0.0
        train_batches = 0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = student_model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)
        metrics["train_losses"].append(avg_train_loss)

        # Validation
        val_dataset = DistillationDataset(
            db_paths=db_paths,
            teacher_model=teacher_model,
            config=config,
            device=device,
            max_positions=num_val,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            num_workers=0,
        )

        student_model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)

                outputs = student_model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_batches += 1

                all_preds.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())

        avg_val_loss = val_loss / max(val_batches, 1)
        metrics["val_losses"].append(avg_val_loss)

        # Calculate correlation and sign agreement
        if all_preds and all_targets:
            preds_arr = np.array(all_preds)
            targets_arr = np.array(all_targets)

            correlation = np.corrcoef(preds_arr, targets_arr)[0, 1]
            sign_agreement = np.mean(np.sign(preds_arr) == np.sign(targets_arr))

            metrics["val_correlations"].append(float(correlation))
            metrics["val_sign_agreements"].append(float(sign_agreement))

            logger.info(
                f"Epoch {epoch + 1}/{config.epochs} - "
                f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
                f"Corr: {correlation:.4f}, Sign Agree: {sign_agreement:.4f}"
            )
        else:
            metrics["val_correlations"].append(0.0)
            metrics["val_sign_agreements"].append(0.0)
            logger.info(
                f"Epoch {epoch + 1}/{config.epochs} - "
                f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
            )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = student_model.state_dict().copy()

        scheduler.step()

    # Restore best model
    if best_state_dict is not None:
        student_model.load_state_dict(best_state_dict)

    metrics["final_val_loss"] = best_val_loss
    metrics["final_correlation"] = metrics["val_correlations"][-1] if metrics["val_correlations"] else 0.0
    metrics["final_sign_agreement"] = metrics["val_sign_agreements"][-1] if metrics["val_sign_agreements"] else 0.0

    return student_model, metrics


def save_nnue_checkpoint(
    model: RingRiftNNUE,
    output_path: str,
    config: DistillationConfig,
    metrics: dict,
    teacher_model_path: str,
) -> None:
    """Save NNUE checkpoint with metadata."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture_version": model.ARCHITECTURE_VERSION,
        "board_type": config.board_type.value,
        "distillation_config": {
            "num_positions": config.num_positions,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
        },
        "metrics": metrics,
        "teacher_model_path": teacher_model_path,
    }

    torch.save(checkpoint, output_path)
    logger.info(f"Saved NNUE checkpoint to {output_path}")


# =============================================================================
# CLI Interface
# =============================================================================

def count_available_positions(db_paths: List[str], board_type: BoardType) -> int:
    """Count total available positions across databases."""
    total = 0
    for db_path in db_paths:
        if not os.path.exists(db_path):
            continue
        try:
            db = GameReplayDB(db_path)
            # Count completed games
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT SUM(total_moves) FROM games
                WHERE game_status = 'completed'
                  AND board_type = ?
            """, (board_type.value.lower(),))
            result = cursor.fetchone()
            if result and result[0]:
                total += int(result[0])
            conn.close()
        except Exception as e:
            logger.warning(f"Error counting positions in {db_path}: {e}")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill a trained NN model to NNUE for fast CPU inference."
    )
    parser.add_argument(
        "--nn-model",
        type=str,
        required=True,
        help="Path to the trained NN model checkpoint (teacher).",
    )
    parser.add_argument(
        "--db",
        type=str,
        action="append",
        required=True,
        help="Path to a game replay database. Can be specified multiple times.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the trained NNUE checkpoint.",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        choices=["square8", "square19", "hexagonal"],
        default="square8",
        help="Board type for the models (default: square8).",
    )
    parser.add_argument(
        "--num-positions",
        type=int,
        default=100_000,
        help="Number of positions to use for distillation (default: 100000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size (default: 256).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for training (cpu, cuda, mps). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check configuration and position counts without training.",
    )

    args = parser.parse_args()

    # Parse board type
    board_type_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
    }
    board_type = board_type_map[args.board_type]

    # Determine device
    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    logger.info(f"Using device: {device}")
    logger.info(f"Board type: {board_type.value}")
    logger.info(f"Teacher model: {args.nn_model}")
    logger.info(f"Databases: {args.db}")

    # Check available positions
    available = count_available_positions(args.db, board_type)
    logger.info(f"Available positions in databases: {available}")

    if available < args.num_positions:
        logger.warning(
            f"Requested {args.num_positions} positions but only {available} available. "
            f"Will use all available positions."
        )

    if args.dry_run:
        logger.info("Dry run complete. Exiting.")
        return

    # Create config
    config = DistillationConfig(
        board_type=board_type,
        num_positions=min(args.num_positions, available) if available > 0 else args.num_positions,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
    )

    # Load teacher model
    logger.info("Loading teacher NN model...")
    teacher_model = load_teacher_model(args.nn_model, board_type, device)

    # Create student NNUE model
    logger.info("Creating student NNUE model...")
    student_model = RingRiftNNUE(board_type=board_type)
    logger.info(f"NNUE feature dim: {get_feature_dim(board_type)}")

    # Run distillation
    logger.info("Starting distillation training...")
    trained_model, metrics = train_nnue_distillation(
        teacher_model=teacher_model,
        student_model=student_model,
        db_paths=args.db,
        config=config,
        device=device,
    )

    # Save checkpoint
    save_nnue_checkpoint(
        model=trained_model,
        output_path=args.output,
        config=config,
        metrics=metrics,
        teacher_model_path=args.nn_model,
    )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("DISTILLATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final validation loss: {metrics['final_val_loss']:.6f}")
    logger.info(f"Final correlation: {metrics['final_correlation']:.4f}")
    logger.info(f"Final sign agreement: {metrics['final_sign_agreement']:.4f}")
    logger.info(f"NNUE checkpoint saved to: {args.output}")


if __name__ == "__main__":
    main()
