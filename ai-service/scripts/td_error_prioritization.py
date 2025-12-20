#!/usr/bin/env python3
"""TD-Error based prioritized replay for RingRift training.

Implements Prioritized Experience Replay (PER) that weights training samples
by their temporal difference error. Samples with high prediction error are
sampled more frequently, leading to more efficient learning.

Reference: Schaul et al. "Prioritized Experience Replay" (2015)

Usage:
    python scripts/td_error_prioritization.py --data training.npz --output prioritized.npz
    python scripts/td_error_prioritization.py --compute-priorities --model models/best.pth
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


class PrioritizedReplayBuffer:
    """Prioritized replay buffer using TD-error priorities.

    Uses a sum-tree for efficient O(log n) sampling and updates.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,  # Priority exponent (0 = uniform, 1 = full prioritization)
        beta: float = 0.4,   # Importance sampling exponent
        beta_increment: float = 0.001,  # Beta annealing
        epsilon: float = 1e-6,  # Small constant for numerical stability
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        # Sum-tree for efficient priority sampling
        self.tree_size = 2 * capacity - 1
        self.tree = np.zeros(self.tree_size, dtype=np.float64)

        # Data storage
        self.data_index = 0
        self.size = 0
        self.max_priority = 1.0

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent > 0:
            self._propagate(parent, change)

    def _get_leaf(self, value: float) -> int:
        """Find leaf node for given priority value."""
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx - (self.capacity - 1)

    def add(self, priority: Optional[float] = None) -> int:
        """Add sample with priority, return data index."""
        if priority is None:
            priority = self.max_priority

        priority = (priority + self.epsilon) ** self.alpha

        data_idx = self.data_index
        tree_idx = data_idx + self.capacity - 1

        # Update tree
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

        self.data_index = (self.data_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        return data_idx

    def update(self, data_idx: int, priority: float) -> None:
        """Update priority for sample."""
        priority = (priority + self.epsilon) ** self.alpha
        self.max_priority = max(self.max_priority, priority)

        tree_idx = data_idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample batch with prioritized sampling.

        Returns:
            indices: Data indices
            priorities: Sample priorities
            weights: Importance sampling weights
        """
        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        # Segment the priority range
        segment = self.tree[0] / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            data_idx = self._get_leaf(value)
            indices[i] = data_idx
            priorities[i] = self.tree[data_idx + self.capacity - 1]

        # Importance sampling weights
        probs = priorities / self.tree[0]
        min_prob = np.min(probs)
        weights = (self.size * probs) ** (-self.beta)
        weights /= weights.max()  # Normalize

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return indices, priorities, weights

    @property
    def total_priority(self) -> float:
        return self.tree[0]


def compute_td_errors(
    model_path: str,
    data_path: str,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute TD-errors for all samples in dataset.

    TD-error = |predicted_value - actual_outcome|

    Args:
        model_path: Path to model checkpoint
        data_path: Path to training data NPZ

    Returns:
        Array of TD-errors for each sample
    """
    import torch
    from app.training.network import RingRiftNet

    # Load data
    data = np.load(data_path, allow_pickle=True)
    features = data["features"]
    globals_data = data["globals"]
    values = data["values"]
    n_samples = len(values)

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu")
    board_type = checkpoint.get("board_type", "square8")

    # Determine input dimensions
    feature_dim = features.shape[1] if len(features.shape) > 1 else features[0].shape[0]
    global_dim = globals_data.shape[1] if len(globals_data.shape) > 1 else globals_data[0].shape[0]

    # Create and load model
    model = RingRiftNet(feature_dim=feature_dim, global_dim=global_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Compute predictions in batches
    td_errors = np.zeros(n_samples, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)

            feat_batch = torch.tensor(features[i:end], dtype=torch.float32)
            glob_batch = torch.tensor(globals_data[i:end], dtype=torch.float32)
            true_values = values[i:end]

            # Get predictions
            _, pred_values = model(feat_batch, glob_batch)
            pred_values = pred_values.squeeze().numpy()

            # TD-error is absolute difference
            td_errors[i:end] = np.abs(pred_values - true_values)

    return td_errors


def create_prioritized_dataset(
    data_path: str,
    td_errors: np.ndarray,
    output_path: str,
    alpha: float = 0.6,
) -> dict:
    """Create dataset with priority weights based on TD-errors.

    Args:
        data_path: Original data path
        td_errors: TD-errors for each sample
        output_path: Output path for prioritized data
        alpha: Priority exponent

    Returns:
        Statistics dict
    """
    data = np.load(data_path, allow_pickle=True)

    # Compute priorities: higher TD-error = higher priority
    priorities = (td_errors + 1e-6) ** alpha

    # Normalize to sampling probabilities
    priorities = priorities / priorities.sum()

    # Save with priorities
    np.savez_compressed(
        output_path,
        features=data["features"],
        globals=data["globals"],
        values=data["values"],
        policy_indices=data.get("policy_indices", np.array([])),
        policy_values=data.get("policy_values", np.array([])),
        td_priorities=priorities,
        td_errors=td_errors,
    )

    return {
        "n_samples": len(td_errors),
        "mean_td_error": float(np.mean(td_errors)),
        "max_td_error": float(np.max(td_errors)),
        "min_td_error": float(np.min(td_errors)),
        "std_td_error": float(np.std(td_errors)),
        "alpha": alpha,
    }


def main():
    parser = argparse.ArgumentParser(description="TD-Error prioritized replay")
    parser.add_argument("--data", required=True, help="Training data NPZ path")
    parser.add_argument("--model", help="Model path for computing TD-errors")
    parser.add_argument("--output", help="Output prioritized data path")
    parser.add_argument("--alpha", type=float, default=0.6, help="Priority exponent")
    parser.add_argument("--compute-only", action="store_true",
                        help="Only compute and print TD-error stats")

    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"Data file not found: {args.data}")
        return 1

    # Find model if not specified
    if not args.model:
        # Look for best model
        best_models = list((AI_SERVICE_ROOT / "models").glob("ringrift_best_*.pth"))
        if best_models:
            args.model = str(max(best_models, key=lambda p: p.stat().st_mtime))
            print(f"Using model: {args.model}")
        else:
            print("No model found. Specify --model path.")
            return 1

    print("Computing TD-errors...")
    td_errors = compute_td_errors(args.model, args.data)

    print(f"\nTD-Error Statistics:")
    print(f"  Mean: {np.mean(td_errors):.4f}")
    print(f"  Std:  {np.std(td_errors):.4f}")
    print(f"  Min:  {np.min(td_errors):.4f}")
    print(f"  Max:  {np.max(td_errors):.4f}")

    # Distribution
    percentiles = [25, 50, 75, 90, 95, 99]
    print(f"\n  Percentiles:")
    for p in percentiles:
        val = np.percentile(td_errors, p)
        print(f"    P{p}: {val:.4f}")

    if args.compute_only:
        return 0

    # Create prioritized dataset
    output_path = args.output or args.data.replace(".npz", "_prioritized.npz")
    print(f"\nCreating prioritized dataset: {output_path}")

    stats = create_prioritized_dataset(
        args.data,
        td_errors,
        output_path,
        alpha=args.alpha,
    )

    print(f"Saved {stats['n_samples']} samples with TD priorities")

    return 0


if __name__ == "__main__":
    sys.exit(main())
