"""Data Augmentation for Board Games.

Leverages board symmetry to multiply training data without additional
storage cost. Applied at data loading time.

Symmetries:
- Square boards: 8-fold (4 rotations × 2 reflections)
- Hexagonal boards: 6-fold (6 rotations)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BoardSymmetry(Enum):
    """Types of board symmetry."""
    NONE = 0
    SQUARE_8FOLD = 8  # 4 rotations × 2 reflections
    HEX_6FOLD = 6  # 6 rotations


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    enabled: bool = True
    symmetry: BoardSymmetry = BoardSymmetry.SQUARE_8FOLD
    random_transform: bool = True  # Random vs all transforms
    augment_policy: bool = True  # Also transform policy targets


class BoardAugmenter:
    """Applies symmetry-based augmentation to board game data."""

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()

    def augment_features(
        self,
        features: np.ndarray,
        transform_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Augment feature tensor using board symmetry.

        Args:
            features: Feature tensor (C, H, W) or (N, C, H, W)
            transform_idx: Specific transform index (0-7 for square)

        Returns:
            Augmented features
        """
        if not self.config.enabled:
            return features

        # Pick random transform if not specified
        if transform_idx is None and self.config.random_transform:
            transform_idx = np.random.randint(0, self.config.symmetry.value)
        elif transform_idx is None:
            transform_idx = 0

        if self.config.symmetry == BoardSymmetry.SQUARE_8FOLD:
            return self._apply_square_transform(features, transform_idx)
        elif self.config.symmetry == BoardSymmetry.HEX_6FOLD:
            return self._apply_hex_transform(features, transform_idx)
        return features

    def _apply_square_transform(
        self,
        features: np.ndarray,
        transform_idx: int,
    ) -> np.ndarray:
        """Apply one of 8 square board symmetries.

        Transforms:
        0: Identity
        1: Rotate 90°
        2: Rotate 180°
        3: Rotate 270°
        4: Horizontal flip
        5: Rotate 90° + flip
        6: Rotate 180° + flip
        7: Rotate 270° + flip
        """
        is_batched = features.ndim == 4
        if not is_batched:
            features = features[np.newaxis, ...]

        result = features.copy()

        # Apply rotation
        k = transform_idx % 4
        if k > 0:
            result = np.rot90(result, k=k, axes=(-2, -1))

        # Apply flip
        if transform_idx >= 4:
            result = np.flip(result, axis=-1)

        return result[0] if not is_batched else result

    def _apply_hex_transform(
        self,
        features: np.ndarray,
        transform_idx: int,
    ) -> np.ndarray:
        """Apply one of 6 hexagonal board symmetries (60° rotations)."""
        # Hex rotation is more complex - simplified version here
        is_batched = features.ndim == 4
        if not is_batched:
            features = features[np.newaxis, ...]

        # For hex boards, use approximate rotation
        k = transform_idx % 3
        result = np.rot90(features, k=k * 2, axes=(-2, -1))

        # Mirror for transforms 3-5
        if transform_idx >= 3:
            result = np.flip(result, axis=-1)

        return result[0] if not is_batched else result

    def augment_policy(
        self,
        policy: np.ndarray,
        board_size: int,
        transform_idx: int,
    ) -> np.ndarray:
        """Augment policy vector to match feature transformation.

        Args:
            policy: Policy vector (sparse or dense)
            board_size: Board size (e.g., 8 for 8x8)
            transform_idx: Same transform used on features

        Returns:
            Transformed policy
        """
        if not self.config.augment_policy:
            return policy

        # For dense policy, reshape and transform like features
        if len(policy) == board_size * board_size:
            policy_2d = policy.reshape(board_size, board_size)
            if self.config.symmetry == BoardSymmetry.SQUARE_8FOLD:
                k = transform_idx % 4
                if k > 0:
                    policy_2d = np.rot90(policy_2d, k=k)
                if transform_idx >= 4:
                    policy_2d = np.flip(policy_2d, axis=1)
            return policy_2d.flatten()

        return policy

    def get_all_transforms(
        self,
        features: np.ndarray,
    ) -> List[np.ndarray]:
        """Get all symmetry transforms of features.

        Args:
            features: Original features (C, H, W)

        Returns:
            List of all transformed versions
        """
        transforms = []
        n_transforms = self.config.symmetry.value
        for i in range(n_transforms):
            transforms.append(self.augment_features(features, i))
        return transforms


class AugmentedDataset:
    """Dataset wrapper that applies augmentation on-the-fly."""

    def __init__(
        self,
        base_dataset,
        augmenter: Optional[BoardAugmenter] = None,
        expand_dataset: bool = False,
    ):
        """Initialize augmented dataset.

        Args:
            base_dataset: Underlying dataset
            augmenter: Board augmenter (created if None)
            expand_dataset: If True, multiply dataset by symmetry count
        """
        self.base_dataset = base_dataset
        self.augmenter = augmenter or BoardAugmenter()
        self.expand_dataset = expand_dataset
        self.n_transforms = self.augmenter.config.symmetry.value if expand_dataset else 1

    def __len__(self):
        return len(self.base_dataset) * self.n_transforms

    def __getitem__(self, idx: int):
        if self.expand_dataset:
            base_idx = idx // self.n_transforms
            transform_idx = idx % self.n_transforms
        else:
            base_idx = idx
            transform_idx = None  # Random

        item = self.base_dataset[base_idx]

        if isinstance(item, tuple):
            features = item[0]
            rest = item[1:]
            aug_features = self.augmenter.augment_features(features, transform_idx)
            return (aug_features, *rest)

        return self.augmenter.augment_features(item, transform_idx)


def create_augmenter(board_type: str = "square8") -> BoardAugmenter:
    """Create an augmenter for the specified board type."""
    if "hex" in board_type.lower():
        symmetry = BoardSymmetry.HEX_6FOLD
    else:
        symmetry = BoardSymmetry.SQUARE_8FOLD

    config = AugmentationConfig(symmetry=symmetry)
    return BoardAugmenter(config)
