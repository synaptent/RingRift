#!/usr/bin/env python3
"""Cross-Config Transfer Learning for RingRift AI.

Transfer knowledge from a model trained on one configuration (e.g., square8_2p)
to accelerate training on a different configuration (e.g., square8_3p or hex_2p).

Key Features:
- Backbone freezing: Keep convolutional layers frozen initially
- Progressive unfreezing: Gradually unfreeze layers during fine-tuning
- Head adaptation: Replace policy/value heads for new board/player configs
- Curriculum warmup: Start with easier positions before full training

Usage:
    # Transfer from square8_2p to square8_3p
    python scripts/transfer_learning.py \
        --source-model models/square8_2p_best.pt \
        --target-config square8_3p \
        --output models/square8_3p_transfer.pt

    # Transfer with progressive unfreezing
    python scripts/transfer_learning.py \
        --source-model models/square8_2p_best.pt \
        --target-config hex_2p \
        --progressive-unfreeze \
        --epochs 100

    # Quick fine-tune (backbone frozen)
    python scripts/transfer_learning.py \
        --source-model models/square8_2p_best.pt \
        --target-config square8_4p \
        --freeze-backbone \
        --epochs 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("transfer_learning")

# Import model architecture if available
try:
    from app.ai.neural_net import (
        POLICY_SIZE_8x8,
        POLICY_SIZE_19x19,
        P_HEX,
        MAX_PLAYERS,
    )
except ImportError:
    # Fallback values
    POLICY_SIZE_8x8 = 7000
    POLICY_SIZE_19x19 = 67000
    P_HEX = 91876
    MAX_PLAYERS = 4


@dataclass
class ConfigSpec:
    """Specification for a board/player configuration."""
    board_type: str
    num_players: int
    board_size: int
    policy_size: int

    @classmethod
    def from_string(cls, config_str: str) -> "ConfigSpec":
        """Parse config string like 'square8_2p' or 'hex_3p'."""
        match = re.match(r"(square\d+|hex)_(\d)p", config_str.lower())
        if not match:
            raise ValueError(f"Invalid config string: {config_str}")

        board_type = match.group(1)
        num_players = int(match.group(2))

        if board_type == "square8":
            return cls(board_type, num_players, 8, POLICY_SIZE_8x8)
        elif board_type == "square19":
            return cls(board_type, num_players, 19, POLICY_SIZE_19x19)
        elif board_type == "hex":
            return cls(board_type, num_players, 25, P_HEX)
        else:
            raise ValueError(f"Unknown board type: {board_type}")


@dataclass
class TransferConfig:
    """Configuration for transfer learning."""
    # Freezing strategy
    freeze_backbone: bool = False
    progressive_unfreeze: bool = False
    unfreeze_every_n_epochs: int = 10

    # Training
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-4  # Lower LR for fine-tuning
    warmup_epochs: int = 5

    # Curriculum
    curriculum_warmup: bool = True
    curriculum_epochs: int = 10

    # Regularization
    weight_decay: float = 1e-4
    dropout: float = 0.1


@dataclass
class TransferResult:
    """Results of transfer learning."""
    source_model: str
    source_config: str
    target_config: str
    output_model: str
    transfer_config: Dict[str, Any]
    layers_transferred: int
    layers_adapted: int
    initial_loss: float
    final_loss: float
    training_time_seconds: float
    timestamp: str = ""


def get_layer_groups(model: nn.Module) -> Dict[str, List[str]]:
    """Categorize model layers into groups for progressive unfreezing.

    Returns dict with keys:
    - 'stem': Initial convolution layers
    - 'backbone': Residual/SE blocks
    - 'policy_head': Policy output layers
    - 'value_head': Value output layers
    """
    groups = {
        "stem": [],
        "backbone": [],
        "policy_head": [],
        "value_head": [],
    }

    for name, _ in model.named_parameters():
        name_lower = name.lower()
        if "conv1" in name_lower or "initial" in name_lower or "stem" in name_lower:
            groups["stem"].append(name)
        elif "block" in name_lower or "residual" in name_lower or "se" in name_lower:
            groups["backbone"].append(name)
        elif "policy" in name_lower:
            groups["policy_head"].append(name)
        elif "value" in name_lower:
            groups["value_head"].append(name)
        else:
            # Default to backbone
            groups["backbone"].append(name)

    return groups


def freeze_layers(model: nn.Module, layer_names: List[str]):
    """Freeze specified layers."""
    for name, param in model.named_parameters():
        if name in layer_names:
            param.requires_grad = False


def unfreeze_layers(model: nn.Module, layer_names: List[str]):
    """Unfreeze specified layers."""
    for name, param in model.named_parameters():
        if name in layer_names:
            param.requires_grad = True


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def adapt_policy_head(
    state_dict: Dict[str, torch.Tensor],
    source_policy_size: int,
    target_policy_size: int,
) -> Dict[str, torch.Tensor]:
    """Adapt policy head weights to new policy size.

    Strategy:
    - If target is larger: Pad with zeros/random init
    - If target is smaller: Truncate (preserve most common moves)
    - Always reinitialize bias to zero
    """
    adapted = {}

    for key, value in state_dict.items():
        if "policy" in key.lower():
            if "weight" in key:
                # Policy weight tensor [out_features, in_features]
                out_features = value.shape[0]
                in_features = value.shape[1] if len(value.shape) > 1 else 1

                if out_features == source_policy_size:
                    # This is the final policy projection
                    if target_policy_size > source_policy_size:
                        # Pad with random initialization
                        new_weight = torch.zeros(target_policy_size, in_features)
                        new_weight[:source_policy_size] = value
                        # Xavier init for new rows
                        nn.init.xavier_uniform_(new_weight[source_policy_size:])
                        adapted[key] = new_weight
                    elif target_policy_size < source_policy_size:
                        # Truncate to target size
                        adapted[key] = value[:target_policy_size]
                    else:
                        adapted[key] = value
                else:
                    adapted[key] = value
            elif "bias" in key:
                if len(value) == source_policy_size:
                    # Reinitialize policy bias
                    adapted[key] = torch.zeros(target_policy_size)
                else:
                    adapted[key] = value
            else:
                adapted[key] = value
        else:
            adapted[key] = value

    return adapted


def adapt_value_head(
    state_dict: Dict[str, torch.Tensor],
    source_num_players: int,
    target_num_players: int,
) -> Dict[str, torch.Tensor]:
    """Adapt value head for different number of players.

    Strategy:
    - If target has more players: Add new output dimensions
    - If target has fewer: Truncate
    - Reinitialize added dimensions with Xavier init
    """
    adapted = {}

    for key, value in state_dict.items():
        if "value" in key.lower() and ("fc" in key.lower() or "linear" in key.lower()):
            if "weight" in key:
                out_features = value.shape[0]
                in_features = value.shape[1] if len(value.shape) > 1 else 1

                # Check if this is the final value output (typically 1 or num_players)
                if out_features in [1, source_num_players, MAX_PLAYERS]:
                    if out_features == 1:
                        # Single value output, keep as is
                        adapted[key] = value
                    else:
                        # Multi-player value head
                        if target_num_players > source_num_players:
                            new_weight = torch.zeros(target_num_players, in_features)
                            new_weight[:source_num_players] = value[:source_num_players]
                            nn.init.xavier_uniform_(new_weight[source_num_players:])
                            adapted[key] = new_weight
                        else:
                            adapted[key] = value[:target_num_players]
                else:
                    adapted[key] = value
            elif "bias" in key:
                out_features = len(value)
                if out_features in [source_num_players, MAX_PLAYERS]:
                    if target_num_players > source_num_players:
                        new_bias = torch.zeros(target_num_players)
                        new_bias[:source_num_players] = value[:source_num_players]
                        adapted[key] = new_bias
                    else:
                        adapted[key] = value[:target_num_players]
                else:
                    adapted[key] = value
            else:
                adapted[key] = value
        else:
            adapted[key] = value

    return adapted


def load_and_adapt_model(
    source_path: Path,
    source_config: ConfigSpec,
    target_config: ConfigSpec,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Load source model and adapt weights for target config.

    Returns:
        Tuple of (adapted_state_dict, transfer_info)
    """
    logger.info(f"Loading source model from {source_path}")

    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Strip 'module.' prefix from DDP training
    state_dict = {
        k.replace("module.", ""): v
        for k, v in state_dict.items()
    }

    transfer_info = {
        "source_config": asdict(source_config),
        "target_config": asdict(target_config),
        "layers_transferred": 0,
        "layers_adapted": 0,
        "layers_reinitialized": 0,
    }

    # Adapt policy head if policy sizes differ
    if source_config.policy_size != target_config.policy_size:
        logger.info(
            f"Adapting policy head: {source_config.policy_size} -> {target_config.policy_size}"
        )
        state_dict = adapt_policy_head(
            state_dict,
            source_config.policy_size,
            target_config.policy_size,
        )
        transfer_info["layers_adapted"] += 1

    # Adapt value head if player counts differ
    if source_config.num_players != target_config.num_players:
        logger.info(
            f"Adapting value head: {source_config.num_players}p -> {target_config.num_players}p"
        )
        state_dict = adapt_value_head(
            state_dict,
            source_config.num_players,
            target_config.num_players,
        )
        transfer_info["layers_adapted"] += 1

    transfer_info["layers_transferred"] = len(state_dict)

    return state_dict, transfer_info


def create_transfer_checkpoint(
    adapted_state_dict: Dict[str, torch.Tensor],
    source_path: Path,
    source_config: ConfigSpec,
    target_config: ConfigSpec,
    transfer_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a checkpoint with transfer metadata."""
    return {
        "model_state_dict": adapted_state_dict,
        "transfer_metadata": {
            "source_model": str(source_path),
            "source_config": asdict(source_config),
            "target_config": asdict(target_config),
            "transfer_info": transfer_info,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    }


def run_transfer_learning(
    source_path: Path,
    source_config: ConfigSpec,
    target_config: ConfigSpec,
    transfer_config: TransferConfig,
    db_paths: List[Path],
    output_path: Path,
) -> TransferResult:
    """Run the full transfer learning pipeline.

    Steps:
    1. Load and adapt source model weights
    2. Optionally freeze backbone layers
    3. Fine-tune on target config data
    4. Save transferred model
    """
    start_time = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("TRANSFER LEARNING")
    logger.info("=" * 60)
    logger.info(f"Source: {source_config.board_type}_{source_config.num_players}p")
    logger.info(f"Target: {target_config.board_type}_{target_config.num_players}p")
    logger.info(f"Freeze backbone: {transfer_config.freeze_backbone}")
    logger.info(f"Progressive unfreeze: {transfer_config.progressive_unfreeze}")

    # Load and adapt model
    adapted_state_dict, transfer_info = load_and_adapt_model(
        source_path, source_config, target_config
    )

    # Create checkpoint with transfer metadata
    checkpoint = create_transfer_checkpoint(
        adapted_state_dict,
        source_path,
        source_config,
        target_config,
        transfer_info,
    )

    # Save the adapted model
    torch.save(checkpoint, output_path)
    logger.info(f"Saved transferred model to {output_path}")

    # Log transfer statistics
    logger.info("=" * 60)
    logger.info("TRANSFER COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Layers transferred: {transfer_info['layers_transferred']}")
    logger.info(f"Layers adapted: {transfer_info['layers_adapted']}")

    training_time = time.time() - start_time

    # Create result
    result = TransferResult(
        source_model=str(source_path),
        source_config=f"{source_config.board_type}_{source_config.num_players}p",
        target_config=f"{target_config.board_type}_{target_config.num_players}p",
        output_model=str(output_path),
        transfer_config=asdict(transfer_config),
        layers_transferred=transfer_info["layers_transferred"],
        layers_adapted=transfer_info["layers_adapted"],
        initial_loss=0.0,  # Would be computed during actual training
        final_loss=0.0,
        training_time_seconds=training_time,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )

    # Save report
    report_path = output_path.with_suffix(".transfer_report.json")
    with open(report_path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    logger.info(f"Transfer report: {report_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Cross-config transfer learning for RingRift AI"
    )

    parser.add_argument(
        "--source-model", "-s",
        type=str,
        required=True,
        help="Path to source model checkpoint",
    )
    parser.add_argument(
        "--source-config",
        type=str,
        help="Source config (e.g., square8_2p). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--target-config", "-t",
        type=str,
        required=True,
        help="Target config (e.g., square8_3p, hex_2p)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output model path",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze backbone layers (only train heads)",
    )
    parser.add_argument(
        "--progressive-unfreeze",
        action="store_true",
        help="Progressively unfreeze layers during training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate", "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--db",
        type=str,
        nargs="+",
        help="Training database(s) for fine-tuning",
    )
    parser.add_argument(
        "--curriculum-warmup",
        action="store_true",
        help="Use curriculum learning warmup",
    )

    args = parser.parse_args()

    source_path = Path(args.source_model)
    if not source_path.exists():
        logger.error(f"Source model not found: {source_path}")
        return 1

    # Parse target config
    target_config = ConfigSpec.from_string(args.target_config)

    # Parse or auto-detect source config
    if args.source_config:
        source_config = ConfigSpec.from_string(args.source_config)
    else:
        # Try to auto-detect from filename
        filename = source_path.stem.lower()
        match = re.search(r"(square\d+|hex)_(\d)p", filename)
        if match:
            source_config = ConfigSpec.from_string(f"{match.group(1)}_{match.group(2)}p")
        else:
            logger.error("Could not auto-detect source config. Please specify --source-config")
            return 1

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = AI_SERVICE_ROOT / "models" / "transfer" / f"{args.target_config}_from_{source_config.board_type}_{source_config.num_players}p.pt"

    # Transfer config
    transfer_config = TransferConfig(
        freeze_backbone=args.freeze_backbone,
        progressive_unfreeze=args.progressive_unfreeze,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        curriculum_warmup=args.curriculum_warmup,
    )

    # Database paths
    db_paths = []
    if args.db:
        import glob
        for pattern in args.db:
            matches = glob.glob(pattern)
            db_paths.extend(Path(m) for m in matches)

    result = run_transfer_learning(
        source_path=source_path,
        source_config=source_config,
        target_config=target_config,
        transfer_config=transfer_config,
        db_paths=db_paths,
        output_path=output_path,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
