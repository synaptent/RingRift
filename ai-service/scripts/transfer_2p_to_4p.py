#!/usr/bin/env python3
"""Transfer learning: adapt same-board models across player counts.

This historical script name is kept for backwards compatibility, but the
implementation now supports any same-board transfer across 2/3/4 players.

It adapts a checkpoint by:
1. Loading all shared weights (conv layers, residual blocks, policy head)
2. Resizing only the per-player value head to the requested player count
3. Keeping transfer metadata so fine-tuning can track where it came from

Usage:
    # Transfer 2p -> 4p (backwards-compatible path)
    PYTHONPATH=. python scripts/transfer_2p_to_4p.py \
        --source models/canonical_sq8_2p.pth \
        --output models/transfer_sq8_4p_init.pth \
        --board-type square8

    # Transfer 2p -> 3p
    PYTHONPATH=. python scripts/transfer_2p_to_4p.py \
        --source models/canonical_hex8_2p.pth \
        --output models/transfer_hex8_3p_init.pth \
        --board-type hex8 \
        --target-players 3

    # Transfer 4p -> 2p for large-board bootstrap
    PYTHONPATH=. python scripts/transfer_2p_to_4p.py \
        --source models/ringrift_best_square19_4p.pth \
        --output models/transfer_square19_2p_init.pth \
        --board-type square19 \
        --target-players 2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def infer_source_players(metadata: dict, state_dict: dict[str, torch.Tensor]) -> int:
    """Infer source player count from metadata or value-head shapes."""
    metadata_players = metadata.get("num_players")
    if isinstance(metadata_players, int) and metadata_players in (2, 3, 4):
        return metadata_players

    config_players = metadata.get("_versioning_metadata", {}).get("config", {}).get("num_players")
    if isinstance(config_players, int) and config_players in (2, 3, 4):
        return config_players

    for key, value in state_dict.items():
        if "value_fc2" in key and value.ndim in (1, 2) and value.shape[0] in (2, 3, 4):
            return int(value.shape[0])

    raise ValueError("Could not infer source player count from checkpoint")


def resize_value_head_weight(
    old_weight: torch.Tensor,
    source_players: int,
    target_players: int,
) -> torch.Tensor:
    """Resize value-head weights for the target player count.

    Expansion preserves the original outputs and initializes new rows from the
    mean source row plus noise. Shrinking retains the leading rows as a
    fine-tuning bootstrap; the target loop is expected to adapt them further.
    """
    if source_players == target_players:
        return old_weight.clone()
    if old_weight.ndim != 2:
        raise ValueError(f"Expected 2D value head weight, got {old_weight.shape}")

    if target_players < source_players:
        return old_weight[:target_players, :].clone()

    new_weight = torch.randn((target_players, old_weight.shape[1])) * 0.1
    new_weight[:source_players, :] = old_weight[:source_players, :]
    avg = old_weight[:source_players, :].mean(dim=0)
    for p in range(source_players, target_players):
        new_weight[p, :] = avg + torch.randn_like(avg) * 0.1
    return new_weight


def resize_value_head_bias(
    old_bias: torch.Tensor,
    source_players: int,
    target_players: int,
) -> torch.Tensor:
    """Resize value-head bias vector for the target player count."""
    if source_players == target_players:
        return old_bias.clone()
    if old_bias.ndim != 1:
        raise ValueError(f"Expected 1D value head bias, got {old_bias.shape}")

    if target_players < source_players:
        return old_bias[:target_players].clone()

    new_bias = torch.zeros(target_players)
    new_bias[:source_players] = old_bias[:source_players]
    avg = old_bias[:source_players].mean()
    for p in range(source_players, target_players):
        new_bias[p] = avg + torch.randn(1).item() * 0.01
    return new_bias


def transfer_model_players(
    source_path: str,
    output_path: str,
    board_type: str,
    target_players: int = 4,
    source_players: int | None = None,
) -> None:
    """Transfer same-board model weights to a different player count."""
    if target_players not in (2, 3, 4):
        raise ValueError(f"target_players must be 2, 3, or 4, got {target_players}")
    if source_players is not None and source_players not in (2, 3, 4):
        raise ValueError(f"source_players must be 2, 3, or 4, got {source_players}")

    from app.utils.torch_utils import safe_load_checkpoint

    logger.info(f"Loading source model: {source_path}")
    checkpoint = safe_load_checkpoint(source_path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    elif isinstance(checkpoint, dict) and any("conv" in k or "res_blocks" in k for k in checkpoint.keys()):
        state_dict = checkpoint
        metadata = {}
    else:
        state_dict = checkpoint
        metadata = {}

    source_players = source_players or infer_source_players(metadata, state_dict)
    logger.info(f"Transfer: {source_players}-player -> {target_players}-player")

    value_keys_to_resize: list[str] = []
    for key, value in state_dict.items():
        if "value_fc2" in key and "weight" in key and value.shape[0] == source_players:
            value_keys_to_resize.append(key)
            logger.info(f"  Found value head: {key} {value.shape}")

    for key in value_keys_to_resize:
        old_weight = state_dict[key]
        new_weight = resize_value_head_weight(old_weight, source_players, target_players)
        for p in range(target_players):
            if new_weight[p, :].abs().sum() < 1e-6:
                logger.warning(f"  Player {p+1} weights near zero - adding noise")
                new_weight[p, :] = torch.randn(new_weight.shape[1]) * 0.1
        state_dict[key] = new_weight
        logger.info(f"  Resized {key}: {old_weight.shape} -> {new_weight.shape}")
        for p in range(target_players):
            mag = new_weight[p, :].abs().mean().item()
            logger.info(f"    Player {p+1} weight magnitude: {mag:.4f}")

    for key in list(state_dict.keys()):
        if "value_fc2" in key and "bias" in key:
            old_bias = state_dict[key]
            if old_bias.shape[0] == source_players:
                new_bias = resize_value_head_bias(old_bias, source_players, target_players)
                state_dict[key] = new_bias
                logger.info(f"  Resized {key}: {old_bias.shape} -> {new_bias.shape}")

    new_checkpoint = {
        "model_state_dict": state_dict,
        "transfer_from": source_path,
        "transfer_type": f"{source_players}p_to_{target_players}p",
        "source_num_players": source_players,
        "num_players": target_players,
        "_versioning_metadata": {
            "config": {
                "num_players": target_players,
                "board_type": board_type,
                "transfer_learning": True,
            }
        },
    }

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(new_checkpoint, output_path)
    logger.info(f"Saved {target_players}-player model to: {output_path}")

    verify = safe_load_checkpoint(output_path, map_location="cpu")
    verify_sd = verify["model_state_dict"]
    for key in verify_sd:
        if "value_fc2" in key:
            logger.info(f"  Verified {key}: {verify_sd[key].shape}")


def transfer_2p_to_np(
    source_path: str,
    output_path: str,
    board_type: str,
    target_players: int = 4,
) -> None:
    """Backwards-compatible wrapper for the historical 2p -> Np helper."""
    return transfer_model_players(
        source_path,
        output_path,
        board_type,
        target_players=target_players,
        source_players=2,
    )


def transfer_2p_to_4p(source_path: str, output_path: str, board_type: str) -> None:
    """Transfer 2-player model weights to 4-player architecture (backwards compat)."""
    return transfer_2p_to_np(source_path, output_path, board_type, target_players=4)


def main():
    parser = argparse.ArgumentParser(description="Transfer same-board model to a different player count")
    parser.add_argument("--source", required=True, help="Source model path")
    parser.add_argument("--output", required=True, help="Output model path")
    parser.add_argument("--board-type", required=True, choices=["square8", "square19", "hex8", "hexagonal"])
    parser.add_argument("--target-players", type=int, default=4, choices=[2, 3, 4],
                        help="Target number of players (default: 4)")
    parser.add_argument("--source-players", type=int, default=None, choices=[2, 3, 4],
                        help="Optional source player-count override")

    args = parser.parse_args()
    transfer_model_players(
        args.source,
        args.output,
        args.board_type,
        target_players=args.target_players,
        source_players=args.source_players,
    )


if __name__ == "__main__":
    main()
