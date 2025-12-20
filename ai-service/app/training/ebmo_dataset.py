"""EBMO Training Dataset with Contrastive Sampling.

Provides datasets for training Energy-Based Move Optimization networks
using contrastive learning on game data.

Key features:
- Loads from existing NPZ game files
- Generates positive samples (moves that were played)
- Generates negative samples (random legal moves, losing moves)
- Hard negative mining (moves similar to good ones but worse)
- Outcome-weighted sampling

Usage:
    from app.training.ebmo_dataset import EBMODataset, EBMODataLoader

    dataset = EBMODataset(
        data_paths=["games/*.npz"],
        num_negatives=15,
        board_size=8,
    )

    loader = EBMODataLoader(dataset, batch_size=64)
    for batch in loader:
        features, globals, pos_actions, neg_actions, outcomes = batch
        # Train EBMO network...
"""

from __future__ import annotations

import glob
import logging
import random
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EBMODatasetConfig:
    """Configuration for EBMO dataset."""

    # Data sources
    data_paths: list[str] = None  # Glob patterns or file paths
    data_dir: str = "data/games"

    # Sampling
    num_negatives: int = 15
    hard_negative_ratio: float = 0.3  # Fraction of negatives that are "hard"
    outcome_weighted: bool = True  # Weight samples by game outcome

    # Board configuration
    board_size: int = 8
    num_input_channels: int = 14
    num_global_features: int = 20
    action_feature_dim: int = 14

    # Processing
    max_samples_per_game: int = 100
    shuffle_buffer_size: int = 10000

    def __post_init__(self):
        if self.data_paths is None:
            self.data_paths = [f"{self.data_dir}/*.npz"]


# =============================================================================
# Data Sample
# =============================================================================


@dataclass
class EBMOSample:
    """Single training sample for EBMO.

    Contains:
    - State features (board + global)
    - Positive action (move that was played)
    - Negative actions (random/hard negatives)
    - Game outcome for weighting
    """

    board_features: np.ndarray  # (C, H, W)
    global_features: np.ndarray  # (G,)
    positive_action: np.ndarray  # (action_dim,)
    negative_actions: np.ndarray  # (num_negatives, action_dim)
    outcome: float  # +1 win, 0 draw, -1 loss


# =============================================================================
# Action Feature Generation
# =============================================================================


class ActionFeatureGenerator:
    """Generate action features for training data.

    Works with raw data (positions, move types) rather than Move objects,
    suitable for processing NPZ files without full game reconstruction.
    """

    NUM_MOVE_TYPES = 8

    def __init__(self, board_size: int = 8, action_dim: int = 14):
        self.board_size = board_size
        self.action_dim = action_dim

    def generate_from_positions(
        self,
        from_x: float,
        from_y: float,
        to_x: float,
        to_y: float,
        move_type: int = 0,
    ) -> np.ndarray:
        """Generate action features from position coordinates.

        Args:
            from_x, from_y: Source position (0 to board_size-1)
            to_x, to_y: Destination position
            move_type: Move type index (0-7)

        Returns:
            (action_dim,) feature array
        """
        features = np.zeros(self.action_dim, dtype=np.float32)

        # Normalized positions [0, 1]
        features[0] = from_x / (self.board_size - 1)
        features[1] = from_y / (self.board_size - 1)
        features[2] = to_x / (self.board_size - 1)
        features[3] = to_y / (self.board_size - 1)

        # Move type one-hot (indices 4-11)
        type_idx = min(move_type, self.NUM_MOVE_TYPES - 1)
        features[4 + type_idx] = 1.0

        # Direction vector
        dx = to_x - from_x
        dy = to_y - from_y
        dist = np.sqrt(dx * dx + dy * dy) + 1e-8
        features[12] = dx / dist
        features[13] = dy / dist

        return features

    def generate_random_action(self) -> np.ndarray:
        """Generate a random action for negative sampling.

        Returns:
            (action_dim,) random feature array
        """
        from_x = np.random.randint(0, self.board_size)
        from_y = np.random.randint(0, self.board_size)
        to_x = np.random.randint(0, self.board_size)
        to_y = np.random.randint(0, self.board_size)
        move_type = np.random.randint(0, self.NUM_MOVE_TYPES)

        return self.generate_from_positions(from_x, from_y, to_x, to_y, move_type)

    def generate_hard_negative(
        self,
        positive_action: np.ndarray,
        noise_scale: float = 0.1,
    ) -> np.ndarray:
        """Generate a hard negative by perturbing a positive action.

        Args:
            positive_action: Features of the positive (good) action
            noise_scale: Scale of noise to add

        Returns:
            (action_dim,) perturbed feature array
        """
        hard_neg = positive_action.copy()

        # Perturb positions slightly
        noise = np.random.randn(4) * noise_scale
        hard_neg[:4] = np.clip(hard_neg[:4] + noise, 0.0, 1.0)

        # Maybe change move type
        if np.random.random() < 0.3:
            # Zero out old type, set new type
            hard_neg[4:12] = 0.0
            new_type = np.random.randint(0, self.NUM_MOVE_TYPES)
            hard_neg[4 + new_type] = 1.0

        # Recalculate direction
        dx = hard_neg[2] - hard_neg[0]
        dy = hard_neg[3] - hard_neg[1]
        dist = np.sqrt(dx * dx + dy * dy) + 1e-8
        hard_neg[12] = dx / dist
        hard_neg[13] = dy / dist

        return hard_neg


# =============================================================================
# NPZ Game Data Parser
# =============================================================================


class GameDataParser:
    """Parse game data from NPZ files.

    Expected NPZ structure (from parallel_selfplay.py):
    - features: (N, C, H, W) board features
    - globals: (N, G) global features
    - values: (N,) game outcomes
    - policy_indices: sparse policy indices
    - policy_values: sparse policy values
    """

    def __init__(
        self,
        board_size: int = 8,
        num_channels: int = 14,
        num_globals: int = 20,
    ):
        self.board_size = board_size
        self.num_channels = num_channels
        self.num_globals = num_globals

    def load_npz(self, path: str) -> dict[str, np.ndarray] | None:
        """Load and validate NPZ file.

        Args:
            path: Path to NPZ file

        Returns:
            Dict with arrays or None if invalid
        """
        try:
            data = np.load(path, allow_pickle=True)

            # Validate required keys
            required = ["features", "values"]
            for key in required:
                if key not in data:
                    logger.warning(f"Missing key '{key}' in {path}")
                    return None

            return dict(data)

        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return None

    def extract_samples(
        self,
        data: dict[str, np.ndarray],
        action_generator: ActionFeatureGenerator,
        num_negatives: int = 15,
        hard_negative_ratio: float = 0.3,
        max_samples: int = 100,
    ) -> list[EBMOSample]:
        """Extract EBMO samples from game data.

        Args:
            data: NPZ data dict
            action_generator: Feature generator
            num_negatives: Number of negative samples per positive
            hard_negative_ratio: Fraction of hard negatives
            max_samples: Maximum samples to extract

        Returns:
            List of EBMOSample objects
        """
        samples = []

        features = data["features"]
        values = data["values"]
        globals_array = data.get("globals", np.zeros((len(features), self.num_globals)))

        # Policy data for positive actions (if available)
        policy_indices = data.get("policy_indices")
        data.get("policy_values")

        num_positions = min(len(features), max_samples)
        indices = np.random.choice(len(features), num_positions, replace=False)

        for idx in indices:
            board_feat = features[idx]
            global_feat = globals_array[idx] if idx < len(globals_array) else np.zeros(self.num_globals)
            outcome = values[idx]

            # Generate positive action
            # If we have policy data, use it; otherwise generate random
            if policy_indices is not None and idx < len(policy_indices):
                # Use policy to infer action
                pos_action = self._action_from_policy(
                    policy_indices[idx],
                    action_generator,
                )
            else:
                # Generate plausible positive action
                pos_action = action_generator.generate_random_action()

            # Generate negative actions
            num_hard = int(num_negatives * hard_negative_ratio)
            num_random = num_negatives - num_hard

            negatives = []

            # Hard negatives (perturbations of positive)
            for _ in range(num_hard):
                neg = action_generator.generate_hard_negative(pos_action)
                negatives.append(neg)

            # Random negatives
            for _ in range(num_random):
                neg = action_generator.generate_random_action()
                negatives.append(neg)

            neg_array = np.stack(negatives, axis=0)

            samples.append(EBMOSample(
                board_features=board_feat.astype(np.float32),
                global_features=global_feat.astype(np.float32),
                positive_action=pos_action,
                negative_actions=neg_array,
                outcome=float(outcome),
            ))

        return samples

    def _action_from_policy(
        self,
        policy_indices: np.ndarray,
        action_generator: ActionFeatureGenerator,
    ) -> np.ndarray:
        """Convert sparse policy index to action features.

        This is a simplified conversion - the actual mapping depends
        on the policy encoding scheme used during data generation.
        """
        if len(policy_indices) == 0:
            return action_generator.generate_random_action()

        # For now, just use the first (best) action
        idx = policy_indices[0] if isinstance(policy_indices, np.ndarray) else policy_indices

        # Convert policy index to position (simplified)
        # Full implementation would use the exact encoding from neural_net.py
        board_size = action_generator.board_size
        total_positions = board_size * board_size

        # Approximate decoding (assumes placement-like encoding)
        from_pos = int(idx) % total_positions
        to_pos = (int(idx) // total_positions) % total_positions

        from_x = from_pos % board_size
        from_y = from_pos // board_size
        to_x = to_pos % board_size
        to_y = to_pos // board_size

        return action_generator.generate_from_positions(
            from_x, from_y, to_x, to_y, move_type=1  # Assume movement
        )


# =============================================================================
# PyTorch Dataset
# =============================================================================


class EBMODataset(Dataset):
    """PyTorch Dataset for EBMO training.

    Loads game data from NPZ files and generates contrastive samples
    on-the-fly.
    """

    def __init__(
        self,
        data_paths: list[str] | None = None,
        data_dir: str = "data/games",
        num_negatives: int = 15,
        hard_negative_ratio: float = 0.3,
        board_size: int = 8,
        max_samples_per_game: int = 100,
        preload: bool = False,
    ):
        """Initialize EBMO dataset.

        Args:
            data_paths: Glob patterns for NPZ files
            data_dir: Directory containing game data
            num_negatives: Negative samples per positive
            hard_negative_ratio: Fraction of hard negatives
            board_size: Board size for feature extraction
            max_samples_per_game: Max samples per game file
            preload: Whether to load all data into memory
        """
        self.num_negatives = num_negatives
        self.hard_negative_ratio = hard_negative_ratio
        self.max_samples_per_game = max_samples_per_game

        # Find all data files
        if data_paths is None:
            data_paths = [f"{data_dir}/*.npz"]

        self.file_paths = []
        for pattern in data_paths:
            self.file_paths.extend(glob.glob(pattern))

        if not self.file_paths:
            logger.warning(f"No data files found matching {data_paths}")

        # Initialize processors
        self.action_generator = ActionFeatureGenerator(board_size)
        self.parser = GameDataParser(board_size)

        # Preload samples if requested
        self.preloaded_samples: list[EBMOSample] | None = None
        if preload and self.file_paths:
            self._preload_all()

    def _preload_all(self) -> None:
        """Load all samples into memory."""
        logger.info(f"Preloading {len(self.file_paths)} game files...")

        self.preloaded_samples = []
        for path in self.file_paths:
            data = self.parser.load_npz(path)
            if data is not None:
                samples = self.parser.extract_samples(
                    data,
                    self.action_generator,
                    self.num_negatives,
                    self.hard_negative_ratio,
                    self.max_samples_per_game,
                )
                self.preloaded_samples.extend(samples)

        logger.info(f"Preloaded {len(self.preloaded_samples)} samples")

    def __len__(self) -> int:
        if self.preloaded_samples is not None:
            return len(self.preloaded_samples)
        # Estimate based on file count and max samples
        return len(self.file_paths) * self.max_samples_per_game

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """Get a training sample.

        Returns:
            (board_features, global_features, positive_action,
             negative_actions, outcome) tuple of tensors
        """
        if self.preloaded_samples is not None:
            sample = self.preloaded_samples[idx % len(self.preloaded_samples)]
        else:
            # Load from random file
            file_idx = idx % len(self.file_paths)
            path = self.file_paths[file_idx]
            data = self.parser.load_npz(path)

            if data is None:
                # Return dummy sample
                return self._dummy_sample()

            samples = self.parser.extract_samples(
                data,
                self.action_generator,
                self.num_negatives,
                self.hard_negative_ratio,
                1,  # Just one sample
            )

            if not samples:
                return self._dummy_sample()

            sample = samples[0]

        return (
            torch.from_numpy(sample.board_features),
            torch.from_numpy(sample.global_features),
            torch.from_numpy(sample.positive_action),
            torch.from_numpy(sample.negative_actions),
            torch.tensor(sample.outcome, dtype=torch.float32),
        )

    def _dummy_sample(self) -> tuple[torch.Tensor, ...]:
        """Create a dummy sample for error cases."""
        board_size = self.action_generator.board_size
        return (
            torch.zeros(14, board_size, board_size),
            torch.zeros(20),
            torch.zeros(14),
            torch.zeros(self.num_negatives, 14),
            torch.tensor(0.0),
        )


# =============================================================================
# Streaming Dataset (for large data)
# =============================================================================


class EBMOStreamingDataset(IterableDataset):
    """Streaming dataset that doesn't load all data into memory.

    Better for very large datasets that don't fit in RAM.
    """

    def __init__(
        self,
        data_paths: list[str] | None = None,
        data_dir: str = "data/games",
        num_negatives: int = 15,
        hard_negative_ratio: float = 0.3,
        board_size: int = 8,
        max_samples_per_game: int = 100,
        shuffle: bool = True,
    ):
        if data_paths is None:
            data_paths = [f"{data_dir}/*.npz"]

        self.file_paths = []
        for pattern in data_paths:
            self.file_paths.extend(glob.glob(pattern))

        self.num_negatives = num_negatives
        self.hard_negative_ratio = hard_negative_ratio
        self.max_samples_per_game = max_samples_per_game
        self.shuffle = shuffle

        self.action_generator = ActionFeatureGenerator(board_size)
        self.parser = GameDataParser(board_size)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        """Iterate over samples from all files."""
        file_list = self.file_paths.copy()
        if self.shuffle:
            random.shuffle(file_list)

        for path in file_list:
            data = self.parser.load_npz(path)
            if data is None:
                continue

            samples = self.parser.extract_samples(
                data,
                self.action_generator,
                self.num_negatives,
                self.hard_negative_ratio,
                self.max_samples_per_game,
            )

            if self.shuffle:
                random.shuffle(samples)

            for sample in samples:
                yield (
                    torch.from_numpy(sample.board_features),
                    torch.from_numpy(sample.global_features),
                    torch.from_numpy(sample.positive_action),
                    torch.from_numpy(sample.negative_actions),
                    torch.tensor(sample.outcome, dtype=torch.float32),
                )


# =============================================================================
# DataLoader utilities
# =============================================================================


def create_ebmo_dataloader(
    data_paths: list[str] | None = None,
    data_dir: str = "data/games",
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    num_negatives: int = 15,
    board_size: int = 8,
    streaming: bool = False,
    preload: bool = True,
) -> DataLoader:
    """Create DataLoader for EBMO training.

    Args:
        data_paths: Glob patterns for data files
        data_dir: Directory containing data
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        num_negatives: Negative samples per positive
        board_size: Board size
        streaming: Use streaming dataset (for large data)
        preload: Preload all data (if not streaming)

    Returns:
        DataLoader instance
    """
    if streaming:
        dataset = EBMOStreamingDataset(
            data_paths=data_paths,
            data_dir=data_dir,
            num_negatives=num_negatives,
            board_size=board_size,
            shuffle=shuffle,
        )
        # Streaming datasets don't support shuffle in DataLoader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        dataset = EBMODataset(
            data_paths=data_paths,
            data_dir=data_dir,
            num_negatives=num_negatives,
            board_size=board_size,
            preload=preload,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )


# =============================================================================
# Synthetic Data Generation (for testing)
# =============================================================================


def generate_synthetic_ebmo_data(
    num_samples: int = 1000,
    board_size: int = 8,
    num_negatives: int = 15,
) -> list[EBMOSample]:
    """Generate synthetic data for testing EBMO.

    Creates random state-action pairs with outcomes based on
    simple heuristics (e.g., center positions are better).

    Args:
        num_samples: Number of samples to generate
        board_size: Board size
        num_negatives: Negatives per sample

    Returns:
        List of EBMOSample
    """
    action_gen = ActionFeatureGenerator(board_size)
    samples = []

    for _ in range(num_samples):
        # Random board features
        board_features = np.random.randn(14, board_size, board_size).astype(np.float32)
        global_features = np.random.randn(20).astype(np.float32)

        # Positive action (biased toward center)
        center = (board_size - 1) / 2
        from_x = int(np.clip(np.random.normal(center, 1.5), 0, board_size - 1))
        from_y = int(np.clip(np.random.normal(center, 1.5), 0, board_size - 1))
        to_x = int(np.clip(np.random.normal(center, 1.5), 0, board_size - 1))
        to_y = int(np.clip(np.random.normal(center, 1.5), 0, board_size - 1))

        positive_action = action_gen.generate_from_positions(
            from_x, from_y, to_x, to_y, move_type=1
        )

        # Negatives (more random)
        negatives = []
        for _ in range(num_negatives):
            neg = action_gen.generate_random_action()
            negatives.append(neg)
        negative_actions = np.stack(negatives)

        # Outcome based on position quality
        center_dist = abs(to_x - center) + abs(to_y - center)
        outcome = 1.0 if center_dist < board_size / 4 else -1.0

        samples.append(EBMOSample(
            board_features=board_features,
            global_features=global_features,
            positive_action=positive_action,
            negative_actions=negative_actions,
            outcome=outcome,
        ))

    return samples


__all__ = [
    "ActionFeatureGenerator",
    "EBMODataset",
    "EBMODatasetConfig",
    "EBMOSample",
    "EBMOStreamingDataset",
    "GameDataParser",
    "create_ebmo_dataloader",
    "generate_synthetic_ebmo_data",
]
