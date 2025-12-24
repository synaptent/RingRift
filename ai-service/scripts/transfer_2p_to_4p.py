#!/usr/bin/env python3
"""Transfer learning: adapt 2-player model to 4-player.

This script takes a trained 2-player model and adapts it for 4-player games by:
1. Loading all shared weights (conv layers, residual blocks, policy head)
2. Reinitializing the value head for 4 players (extends from 2 outputs to 4)

Usage:
    PYTHONPATH=. python scripts/transfer_2p_to_4p.py \
        --source models/canonical_sq8_2p.pth \
        --output models/transfer_sq8_4p_init.pth \
        --board-type square8
"""

import argparse
import logging
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def transfer_2p_to_4p(source_path: str, output_path: str, board_type: str) -> None:
    """Transfer 2-player model weights to 4-player architecture."""

    logger.info(f"Loading source model: {source_path}")
    checkpoint = torch.load(source_path, map_location='cpu')

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

    # Resize value head from 2 to 4 players
    for key in value_keys_to_resize:
        old_weight = state_dict[key]
        new_shape = (4, old_weight.shape[1]) if len(old_weight.shape) == 2 else (4,)

        # Initialize new weights
        new_weight = torch.zeros(new_shape)

        # Copy existing weights for players 1-2
        if len(old_weight.shape) == 2:
            new_weight[:2, :] = old_weight
            # Initialize players 3-4 by averaging players 1-2 with noise
            avg = old_weight.mean(dim=0)
            new_weight[2, :] = avg + torch.randn_like(avg) * 0.01
            new_weight[3, :] = avg + torch.randn_like(avg) * 0.01
        else:
            new_weight[:2] = old_weight
            avg = old_weight.mean()
            new_weight[2] = avg + torch.randn(1).item() * 0.01
            new_weight[3] = avg + torch.randn(1).item() * 0.01

        state_dict[key] = new_weight
        logger.info(f"  Resized {key}: {old_weight.shape} -> {new_weight.shape}")

    # Also resize bias if present
    for key in list(state_dict.keys()):
        if 'value_fc2' in key and 'bias' in key:
            old_bias = state_dict[key]
            if old_bias.shape[0] == 2:
                new_bias = torch.zeros(4)
                new_bias[:2] = old_bias
                new_bias[2] = old_bias.mean() + torch.randn(1).item() * 0.01
                new_bias[3] = old_bias.mean() + torch.randn(1).item() * 0.01
                state_dict[key] = new_bias
                logger.info(f"  Resized {key}: {old_bias.shape} -> {new_bias.shape}")

    # Create new checkpoint
    new_checkpoint = {
        'model_state_dict': state_dict,
        'transfer_from': source_path,
        'transfer_type': '2p_to_4p',
        'num_players': 4,
        '_versioning_metadata': {
            'config': {
                'num_players': 4,
                'board_type': board_type,
                'transfer_learning': True,
            }
        }
    }

    # Save
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(new_checkpoint, output_path)
    logger.info(f"Saved 4-player model to: {output_path}")

    # Verify
    verify = torch.load(output_path, map_location='cpu')
    verify_sd = verify['model_state_dict']
    for key in verify_sd:
        if 'value_fc2' in key:
            logger.info(f"  Verified {key}: {verify_sd[key].shape}")


def main():
    parser = argparse.ArgumentParser(description="Transfer 2-player model to 4-player")
    parser.add_argument('--source', required=True, help='Source 2-player model path')
    parser.add_argument('--output', required=True, help='Output 4-player model path')
    parser.add_argument('--board-type', required=True, choices=['square8', 'square19', 'hex8', 'hexagonal'])

    args = parser.parse_args()
    transfer_2p_to_4p(args.source, args.output, args.board_type)


if __name__ == '__main__':
    main()
