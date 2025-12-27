#!/usr/bin/env python3
"""Transfer learning: adapt 2-player model to N-player.

This script takes a trained 2-player model and adapts it for 3 or 4-player games by:
1. Loading all shared weights (conv layers, residual blocks, policy head)
2. Reinitializing the value head for N players (extends from 2 outputs to N)

Usage:
    # Transfer to 4-player (default)
    PYTHONPATH=. python scripts/transfer_2p_to_4p.py \
        --source models/canonical_sq8_2p.pth \
        --output models/transfer_sq8_4p_init.pth \
        --board-type square8

    # Transfer to 3-player
    PYTHONPATH=. python scripts/transfer_2p_to_4p.py \
        --source models/canonical_hex8_2p.pth \
        --output models/transfer_hex8_3p_init.pth \
        --board-type hex8 \
        --target-players 3
"""

import argparse
import logging
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def transfer_2p_to_np(source_path: str, output_path: str, board_type: str, target_players: int = 4) -> None:
    """Transfer 2-player model weights to N-player architecture.

    Args:
        source_path: Path to source 2-player model
        output_path: Path to save target model
        board_type: Board type (square8, hex8, etc.)
        target_players: Target number of players (3 or 4)
    """
    if target_players not in (3, 4):
        raise ValueError(f"target_players must be 3 or 4, got {target_players}")

    from app.utils.torch_utils import safe_load_checkpoint
    logger.info(f"Loading source model: {source_path}")
    logger.info(f"Transfer: 2-player -> {target_players}-player")
    checkpoint = safe_load_checkpoint(source_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and any('conv' in k or 'res_blocks' in k for k in checkpoint.keys()):
        state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Find value head weights that need resizing
    value_keys_to_resize = []
    for key in state_dict.keys():
        if 'value_fc2' in key and 'weight' in key:
            shape = state_dict[key].shape
            if shape[0] == 2:  # 2-player output
                value_keys_to_resize.append(key)
                logger.info(f"  Found 2-player value head: {key} {shape}")

    # Resize value head from 2 to target_players
    # Dec 27, 2025: Fixed initialization - was using 0.01 noise (too small),
    # and Player 2 could end up all zeros if weight copy failed
    for key in value_keys_to_resize:
        old_weight = state_dict[key]
        new_shape = (target_players, old_weight.shape[1]) if len(old_weight.shape) == 2 else (target_players,)

        # Initialize new weights with small random values (not zeros!)
        # This ensures no player starts with all-zero weights
        new_weight = torch.randn(new_shape) * 0.1

        # Copy existing weights for players 1-2
        if len(old_weight.shape) == 2:
            new_weight[0, :] = old_weight[0, :]  # Player 1 - explicit copy
            new_weight[1, :] = old_weight[1, :]  # Player 2 - explicit copy
            # Initialize additional players by averaging players 1-2 with LARGER noise
            avg = old_weight.mean(dim=0)
            for p in range(2, target_players):
                # Use 0.1 scale instead of 0.01 for better learning dynamics
                new_weight[p, :] = avg + torch.randn_like(avg) * 0.1
        else:
            new_weight[0] = old_weight[0]  # Player 1
            new_weight[1] = old_weight[1]  # Player 2
            avg = old_weight.mean()
            for p in range(2, target_players):
                new_weight[p] = avg + torch.randn(1).item() * 0.1

        # Validate: ensure no player has all-zero weights
        for p in range(target_players):
            if len(new_weight.shape) == 2:
                if new_weight[p, :].abs().sum() < 1e-6:
                    logger.warning(f"  Player {p+1} weights near zero - adding noise")
                    new_weight[p, :] = torch.randn(new_weight.shape[1]) * 0.1
            else:
                if abs(new_weight[p].item()) < 1e-6:
                    logger.warning(f"  Player {p+1} weight near zero - adding noise")
                    new_weight[p] = torch.randn(1).item() * 0.1

        state_dict[key] = new_weight
        logger.info(f"  Resized {key}: {old_weight.shape} -> {new_weight.shape}")
        # Log weight magnitudes for each player
        for p in range(target_players):
            if len(new_weight.shape) == 2:
                mag = new_weight[p, :].abs().mean().item()
            else:
                mag = abs(new_weight[p].item())
            logger.info(f"    Player {p+1} weight magnitude: {mag:.4f}")

    # Also resize bias if present
    for key in list(state_dict.keys()):
        if 'value_fc2' in key and 'bias' in key:
            old_bias = state_dict[key]
            if old_bias.shape[0] == 2:
                new_bias = torch.zeros(target_players)
                new_bias[:2] = old_bias
                for p in range(2, target_players):
                    new_bias[p] = old_bias.mean() + torch.randn(1).item() * 0.01
                state_dict[key] = new_bias
                logger.info(f"  Resized {key}: {old_bias.shape} -> {new_bias.shape}")

    # Create new checkpoint
    new_checkpoint = {
        'model_state_dict': state_dict,
        'transfer_from': source_path,
        'transfer_type': f'2p_to_{target_players}p',
        'num_players': target_players,
        '_versioning_metadata': {
            'config': {
                'num_players': target_players,
                'board_type': board_type,
                'transfer_learning': True,
            }
        }
    }

    # Save
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(new_checkpoint, output_path)
    logger.info(f"Saved {target_players}-player model to: {output_path}")

    # Verify
    verify = safe_load_checkpoint(output_path, map_location='cpu')
    verify_sd = verify['model_state_dict']
    for key in verify_sd:
        if 'value_fc2' in key:
            logger.info(f"  Verified {key}: {verify_sd[key].shape}")


# Keep the old function name for backwards compatibility
def transfer_2p_to_4p(source_path: str, output_path: str, board_type: str) -> None:
    """Transfer 2-player model weights to 4-player architecture (backwards compat)."""
    return transfer_2p_to_np(source_path, output_path, board_type, target_players=4)


def main():
    parser = argparse.ArgumentParser(description="Transfer 2-player model to N-player")
    parser.add_argument('--source', required=True, help='Source 2-player model path')
    parser.add_argument('--output', required=True, help='Output model path')
    parser.add_argument('--board-type', required=True, choices=['square8', 'square19', 'hex8', 'hexagonal'])
    parser.add_argument('--target-players', type=int, default=4, choices=[3, 4],
                        help='Target number of players (default: 4)')

    args = parser.parse_args()
    transfer_2p_to_np(args.source, args.output, args.board_type, args.target_players)


if __name__ == '__main__':
    main()
