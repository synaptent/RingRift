#!/usr/bin/env python3
"""Transfer learning: adapt same-board models across player counts.

The implementation supports any same-board transfer across 2/3/4 players.

It adapts a checkpoint by:
1. Loading all shared weights (conv layers, residual blocks, policy head)
2. Resizing per-player output heads to the requested player count
3. Keeping transfer metadata so fine-tuning can track where it came from
4. Strict-loading the result into a fresh target model when the architecture is
   recognized, so incompatible transfer artifacts fail at generation time

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
from typing import Any

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

    for final_value_key in ("value_fc3", "value_fc2"):
        for key, value in state_dict.items():
            if (
                final_value_key in key
                and value.ndim in (1, 2)
                and value.shape[0] in (2, 3, 4)
            ):
                return int(value.shape[0])

    for final_rank_key in ("rank_dist_fc3", "rank_dist_fc2"):
        for key, value in state_dict.items():
            if final_rank_key not in key or value.ndim not in (1, 2):
                continue
            for players in (2, 3, 4):
                if value.shape[0] == players * players:
                    return players

    raise ValueError("Could not infer source player count from checkpoint")


def _is_final_value_head_key(key: str, value: torch.Tensor, source_players: int) -> bool:
    return (
        key.endswith((".weight", ".bias"))
        and any(name in key for name in ("value_fc3", "value_fc2", "value_head"))
        and value.ndim in (1, 2)
        and value.shape[0] == source_players
    )


def _is_rank_distribution_head_key(key: str, value: torch.Tensor, source_players: int) -> bool:
    return (
        key.endswith((".weight", ".bias"))
        and any(name in key for name in ("rank_dist_fc3", "rank_dist_fc2"))
        and value.ndim in (1, 2)
        and value.shape[0] == source_players * source_players
    )


def _infer_board_shape(board_type: str) -> tuple[int, int | None]:
    if board_type == "hex8":
        return 9, 4
    if board_type == "hexagonal":
        return 25, 12
    if board_type == "square8":
        return 8, None
    if board_type == "square19":
        return 19, None
    raise ValueError(f"Unsupported board_type={board_type!r}")


def _infer_num_res_blocks(state_dict: dict[str, torch.Tensor]) -> int:
    block_indices: set[int] = set()
    for key in state_dict:
        if not key.startswith("res_blocks."):
            continue
        parts = key.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            block_indices.add(int(parts[1]))
    return max(block_indices) + 1 if block_indices else 13


def _metadata_config(metadata: dict[str, Any]) -> dict[str, Any]:
    versioning = metadata.get("_versioning_metadata")
    if isinstance(versioning, dict):
        config = versioning.get("config", {})
        if isinstance(config, dict):
            return dict(config)
    return {}


def _strict_verify_target_model(
    output_path: str,
    board_type: str,
    target_players: int,
) -> None:
    """Build the target architecture and strict-load the transferred state."""
    from app.utils.torch_utils import safe_load_checkpoint

    verify = safe_load_checkpoint(output_path, map_location="cpu")
    verify_sd = verify["model_state_dict"]
    config = _metadata_config(verify)

    conv1 = verify_sd.get("conv1.weight")
    has_v4_heads = (
        "value_fc3.weight" in verify_sd
        and "rank_dist_fc3.weight" in verify_sd
        and conv1 is not None
        and tuple(conv1.shape[2:]) == (5, 5)
    )
    if not has_v4_heads:
        logger.info("Skipping strict target-model verify for unrecognized architecture")
        return

    board_size, hex_radius = _infer_board_shape(board_type)
    num_filters = int(conv1.shape[0])
    total_in_channels = int(conv1.shape[1])
    num_res_blocks = int(config.get("num_res_blocks") or _infer_num_res_blocks(verify_sd))
    global_features = int(config.get("global_features") or 20)

    if board_type in ("hex8", "hexagonal"):
        from app.ai.neural_net.hex_architectures import HexNeuralNet_v4

        model = HexNeuralNet_v4(
            in_channels=total_in_channels,
            global_features=global_features,
            num_res_blocks=num_res_blocks,
            num_filters=num_filters,
            board_size=board_size,
            hex_radius=hex_radius,
            policy_size=None,
            num_players=target_players,
        )
    else:
        from app.ai.neural_net.square_architectures import RingRiftCNN_v4

        history_length = int(config.get("history_length") or 3)
        if total_in_channels % (history_length + 1) != 0:
            raise ValueError(
                f"Cannot infer square base channels from conv1 in_channels={total_in_channels} "
                f"and history_length={history_length}"
            )
        model = RingRiftCNN_v4(
            board_size=board_size,
            in_channels=total_in_channels // (history_length + 1),
            global_features=global_features,
            num_res_blocks=num_res_blocks,
            num_filters=num_filters,
            history_length=history_length,
            policy_size=None,
            num_players=target_players,
        )

    model.load_state_dict(verify_sd, strict=True)
    logger.info(
        "Strict target-model verification passed: %s %sp",
        type(model).__name__,
        target_players,
    )


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

    new_weight = torch.randn(
        (target_players, old_weight.shape[1]),
        dtype=old_weight.dtype,
        device=old_weight.device,
    ) * 0.1
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

    new_bias = torch.zeros(target_players, dtype=old_bias.dtype, device=old_bias.device)
    new_bias[:source_players] = old_bias[:source_players]
    avg = old_bias[:source_players].mean()
    for p in range(source_players, target_players):
        new_bias[p] = avg + torch.randn((), dtype=old_bias.dtype, device=old_bias.device) * 0.01
    return new_bias


def resize_rank_distribution_weight(
    old_weight: torch.Tensor,
    source_players: int,
    target_players: int,
) -> torch.Tensor:
    """Resize flattened rank-distribution weights from N² to target N²."""
    if source_players == target_players:
        return old_weight.clone()
    if old_weight.ndim != 2:
        raise ValueError(f"Expected 2D rank head weight, got {old_weight.shape}")

    src = old_weight.view(source_players, source_players, old_weight.shape[1])
    new_weight = torch.empty(
        (target_players, target_players, old_weight.shape[1]),
        dtype=old_weight.dtype,
        device=old_weight.device,
    )
    avg = src.mean(dim=(0, 1))
    new_weight[:] = avg + torch.randn_like(new_weight) * 0.1
    common = min(source_players, target_players)
    new_weight[:common, :common, :] = src[:common, :common, :]
    return new_weight.view(target_players * target_players, old_weight.shape[1])


def resize_rank_distribution_bias(
    old_bias: torch.Tensor,
    source_players: int,
    target_players: int,
) -> torch.Tensor:
    """Resize flattened rank-distribution bias from N² to target N²."""
    if source_players == target_players:
        return old_bias.clone()
    if old_bias.ndim != 1:
        raise ValueError(f"Expected 1D rank head bias, got {old_bias.shape}")

    src = old_bias.view(source_players, source_players)
    new_bias = torch.empty(
        (target_players, target_players),
        dtype=old_bias.dtype,
        device=old_bias.device,
    )
    avg = src.mean()
    new_bias[:] = avg + torch.randn_like(new_bias) * 0.01
    common = min(source_players, target_players)
    new_bias[:common, :common] = src[:common, :common]
    return new_bias.view(target_players * target_players)


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

    resized_keys: list[str] = []
    for key, value in list(state_dict.items()):
        if _is_final_value_head_key(key, value, source_players):
            if key.endswith(".weight"):
                new_value = resize_value_head_weight(value, source_players, target_players)
            else:
                new_value = resize_value_head_bias(value, source_players, target_players)
            state_dict[key] = new_value
            resized_keys.append(key)
            logger.info(f"  Resized value head {key}: {value.shape} -> {new_value.shape}")
        elif _is_rank_distribution_head_key(key, value, source_players):
            if key.endswith(".weight"):
                new_value = resize_rank_distribution_weight(value, source_players, target_players)
            else:
                new_value = resize_rank_distribution_bias(value, source_players, target_players)
            state_dict[key] = new_value
            resized_keys.append(key)
            logger.info(f"  Resized rank head {key}: {value.shape} -> {new_value.shape}")

    if not resized_keys:
        raise ValueError(
            f"No player-count output heads found for source_players={source_players}; "
            "refusing to save an unmodified transfer checkpoint."
        )

    previous_config = _metadata_config(metadata)
    transfer_config = {
        **previous_config,
        "num_players": target_players,
        "board_type": board_type,
        "transfer_learning": True,
    }

    new_checkpoint = {
        "model_state_dict": state_dict,
        "transfer_from": source_path,
        "transfer_type": f"{source_players}p_to_{target_players}p",
        "source_num_players": source_players,
        "num_players": target_players,
        "_versioning_metadata": {
            "config": transfer_config,
        },
    }

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(new_checkpoint, output_path)
    logger.info(f"Saved {target_players}-player model to: {output_path}")

    verify = safe_load_checkpoint(output_path, map_location="cpu")
    verify_sd = verify["model_state_dict"]
    for key in resized_keys:
        logger.info(f"  Verified resized {key}: {verify_sd[key].shape}")
    _strict_verify_target_model(output_path, board_type, target_players)


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
