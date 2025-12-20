#!/usr/bin/env python3
"""Merge original Gumbel MCTS policy data with balanced value targets.

Takes the original Gumbel MCTS dataset (with soft policy targets) and
balances the value distribution by reweighting or subsampling.

Usage:
    python scripts/merge_balanced_values.py \
        --input data/training/sq8_kl_distill.npz \
        --output data/training/sq8_balanced_policy.npz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def balance_values(
    features: np.ndarray,
    globals_arr: np.ndarray,
    values: np.ndarray,
    policy_indices: np.ndarray,
    policy_values: np.ndarray,
    target_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Balance value distribution by subsampling majority class.

    Args:
        features: Feature arrays [N, C, H, W]
        globals_arr: Global feature arrays [N, G]
        values: Value targets [N]
        policy_indices: Policy index arrays [N] (ragged)
        policy_values: Policy value arrays [N] (ragged)
        target_ratio: Target ratio for positive class (default 0.5)

    Returns:
        Balanced versions of all arrays
    """
    # Separate by value sign (positive = P1 wins, negative = P2 wins)
    p1_mask = values > 0
    p2_mask = values < 0
    draw_mask = values == 0

    n_p1 = p1_mask.sum()
    n_p2 = p2_mask.sum()
    n_draw = draw_mask.sum()

    logger.info(f"Original distribution: P1={n_p1}, P2={n_p2}, Draw={n_draw}")

    # Determine target counts for balanced dataset
    n_minority = min(n_p1, n_p2)
    n_target = n_minority  # Match minority class

    # Get indices
    p1_indices = np.where(p1_mask)[0]
    p2_indices = np.where(p2_mask)[0]
    draw_indices = np.where(draw_mask)[0]

    # Subsample majority class
    rng = np.random.default_rng(42)

    if n_p1 > n_target:
        p1_indices = rng.choice(p1_indices, n_target, replace=False)
    if n_p2 > n_target:
        p2_indices = rng.choice(p2_indices, n_target, replace=False)

    # Combine indices and shuffle
    balanced_indices = np.concatenate([p1_indices, p2_indices, draw_indices])
    rng.shuffle(balanced_indices)

    logger.info(f"Balanced to: P1={len(p1_indices)}, P2={len(p2_indices)}, Draw={len(draw_indices)}")
    logger.info(f"Total samples: {len(balanced_indices)}")

    # Extract balanced data
    return (
        features[balanced_indices],
        globals_arr[balanced_indices],
        values[balanced_indices],
        policy_indices[balanced_indices],
        policy_values[balanced_indices],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Balance value distribution in training data"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input NPZ file with original Gumbel MCTS data"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output NPZ file with balanced values"
    )
    parser.add_argument(
        "--target-ratio", type=float, default=0.5,
        help="Target ratio for P1 wins (default: 0.5)"
    )

    args = parser.parse_args()

    # Load original data
    logger.info(f"Loading {args.input}...")
    data = np.load(args.input, allow_pickle=True)

    features = data["features"]
    globals_arr = data["globals"]
    values = data["values"]
    policy_indices = data["policy_indices"]
    policy_values = data["policy_values"]

    logger.info(f"Loaded {len(features)} samples")

    # Balance
    balanced = balance_values(
        features, globals_arr, values, policy_indices, policy_values,
        target_ratio=args.target_ratio,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        args.output,
        features=balanced[0],
        globals=balanced[1],
        values=balanced[2],
        policy_indices=balanced[3],
        policy_values=balanced[4],
        # Copy metadata
        board_type=data.get("board_type", np.asarray("square8")),
        board_size=data.get("board_size", np.asarray(8)),
        history_length=data.get("history_length", np.asarray(3)),
        feature_version=data.get("feature_version", np.asarray(2)),
        policy_encoding=data.get("policy_encoding", np.asarray("soft_kl")),
    )

    logger.info(f"Saved balanced data to {args.output}")


if __name__ == "__main__":
    main()
