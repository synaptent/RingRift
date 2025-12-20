"""
Unified Data Augmentation Pipeline for RingRift AI Training.

Provides a unified interface for augmenting training data using board
symmetries. Automatically selects the appropriate augmentation strategy
based on board type:
- Square boards (8x8, 19x19): D4 symmetry (8 transformations)
- Hexagonal boards: D6 symmetry (12 transformations)

Usage:
    from app.training.data_augmentation import DataAugmentor

    # Create augmentor for specific board type
    augmentor = DataAugmentor(board_type="square8")

    # Augment a single sample
    augmented_samples = augmentor.augment_sample(features, globals, policy_idx, policy_val)

    # Or use random augmentation for online training
    aug_features, aug_policy_idx, aug_policy_val = augmentor.random_augment(
        features, policy_idx, policy_val
    )
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Union

import numpy as np

from app.models.core import BoardType

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    enabled: bool = True
    # Probability of applying augmentation during online training
    augment_probability: float = 1.0
    # Whether to use all transforms or sample randomly
    all_transforms: bool = True
    # Number of random transforms to sample if not using all
    num_random_transforms: int = 4
    # Exclude identity transform when sampling
    exclude_identity: bool = False


class DataAugmentor:
    """
    Unified data augmentation for RingRift training.

    Automatically selects appropriate symmetry group based on board type:
    - Square boards: D4 (8 symmetries)
    - Hexagonal boards: D6 (12 symmetries)
    """

    def __init__(
        self,
        board_type: Union[str, BoardType],
        config: AugmentationConfig | None = None,
    ):
        """
        Initialize augmentor for specific board type.

        Args:
            board_type: Board type string or enum
            config: Augmentation configuration
        """
        if isinstance(board_type, str):
            board_type = BoardType(board_type)

        self.board_type = board_type
        self.config = config or AugmentationConfig()

        # Initialize appropriate transformer
        self._init_transformer()

    def _init_transformer(self):
        """Initialize the appropriate symmetry transformer."""
        if self.board_type == BoardType.HEXAGONAL:
            from app.training.hex_augmentation import HexSymmetryTransform
            self.transformer = HexSymmetryTransform()
            self.num_transforms = 12
            self.board_size = 25
        elif self.board_type == BoardType.SQUARE8:
            from app.training.square_augmentation import SquareSymmetryTransform
            self.transformer = SquareSymmetryTransform(board_size=8)
            self.num_transforms = 8
            self.board_size = 8
        elif self.board_type == BoardType.SQUARE19:
            from app.training.square_augmentation import SquareSymmetryTransform
            self.transformer = SquareSymmetryTransform(board_size=19)
            self.num_transforms = 8
            self.board_size = 19
        else:
            raise ValueError(f"Unknown board type: {self.board_type}")

        logger.debug(
            f"Initialized {self.board_type.value} augmentor with "
            f"{self.num_transforms} transforms"
        )

    def get_all_transforms(self) -> list[int]:
        """Get list of all transform IDs."""
        return list(range(self.num_transforms))

    def get_random_transform(self, exclude_identity: bool = False) -> int:
        """Get a random transform ID."""
        transforms = self.get_all_transforms()
        if exclude_identity:
            transforms = [t for t in transforms if t != 0]
        return random.choice(transforms)

    def transform_board(
        self,
        features: np.ndarray,
        transform_id: int,
    ) -> np.ndarray:
        """Transform board features."""
        return self.transformer.transform_board(features, transform_id)

    def transform_sparse_policy(
        self,
        policy_indices: np.ndarray,
        policy_values: np.ndarray,
        transform_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform sparse policy."""
        return self.transformer.transform_sparse_policy(
            policy_indices, policy_values, transform_id
        )

    def transform_dense_policy(
        self,
        policy: np.ndarray,
        transform_id: int,
    ) -> np.ndarray:
        """Transform dense policy vector."""
        return self.transformer.transform_policy(policy, transform_id)

    def augment_sample(
        self,
        features: np.ndarray,
        globals_vec: np.ndarray,
        policy_indices: np.ndarray,
        policy_values: np.ndarray,
        transforms: list[int] | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Augment a single training sample with multiple transforms.

        Args:
            features: Board features (C, H, W)
            globals_vec: Global features vector
            policy_indices: Sparse policy indices
            policy_values: Sparse policy values
            transforms: Optional list of transform IDs to apply.
                        If None, uses config settings.

        Returns:
            List of augmented samples (features, globals, policy_idx, policy_val)
        """
        if not self.config.enabled:
            return [(features.copy(), globals_vec.copy(),
                    policy_indices.copy(), policy_values.copy())]

        if transforms is None:
            if self.config.all_transforms:
                transforms = self.get_all_transforms()
            else:
                available = self.get_all_transforms()
                if self.config.exclude_identity:
                    available = [t for t in available if t != 0]
                k = min(self.config.num_random_transforms, len(available))
                transforms = random.sample(available, k)
                # Always include identity if not excluded
                if not self.config.exclude_identity and 0 not in transforms:
                    transforms = [0, *transforms[:k - 1]]

        augmented = []
        for t in transforms:
            aug_features = self.transform_board(features, t)
            aug_indices, aug_values = self.transform_sparse_policy(
                policy_indices, policy_values, t
            )
            augmented.append((
                aug_features,
                globals_vec.copy(),  # Globals unchanged
                aug_indices,
                aug_values,
            ))

        return augmented

    def random_augment(
        self,
        features: np.ndarray,
        policy_indices: np.ndarray,
        policy_values: np.ndarray,
        globals_vec: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Apply a single random augmentation to a sample.

        Useful for online augmentation during training where you want
        one random transform per sample.

        Args:
            features: Board features (C, H, W)
            policy_indices: Sparse policy indices
            policy_values: Sparse policy values
            globals_vec: Optional global features

        Returns:
            Tuple of (aug_features, aug_policy_idx, aug_policy_val, globals)
        """
        if not self.config.enabled:
            return (features.copy(), policy_indices.copy(),
                    policy_values.copy(), globals_vec)

        if random.random() > self.config.augment_probability:
            return (features.copy(), policy_indices.copy(),
                    policy_values.copy(), globals_vec)

        t = self.get_random_transform(exclude_identity=self.config.exclude_identity)

        aug_features = self.transform_board(features, t)
        aug_indices, aug_values = self.transform_sparse_policy(
            policy_indices, policy_values, t
        )

        return (aug_features, aug_indices, aug_values, globals_vec)

    def augment_batch(
        self,
        features_batch: np.ndarray,
        globals_batch: np.ndarray,
        policy_indices_batch: list[np.ndarray],
        policy_values_batch: list[np.ndarray],
        same_transform: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """
        Apply random augmentation to a batch of samples.

        Args:
            features_batch: Batch of features (B, C, H, W)
            globals_batch: Batch of globals (B, G)
            policy_indices_batch: List of sparse policy indices
            policy_values_batch: List of sparse policy values
            same_transform: If True, apply same transform to all samples

        Returns:
            Tuple of augmented (features, globals, policy_indices, policy_values)
        """
        batch_size = features_batch.shape[0]

        if same_transform:
            t = self.get_random_transform()
            transforms = [t] * batch_size
        else:
            transforms = [self.get_random_transform() for _ in range(batch_size)]

        aug_features = np.zeros_like(features_batch)
        aug_indices_list = []
        aug_values_list = []

        for i in range(batch_size):
            aug_features[i] = self.transform_board(features_batch[i], transforms[i])
            aug_idx, aug_val = self.transform_sparse_policy(
                policy_indices_batch[i],
                policy_values_batch[i],
                transforms[i],
            )
            aug_indices_list.append(aug_idx)
            aug_values_list.append(aug_val)

        return (aug_features, globals_batch.copy(), aug_indices_list, aug_values_list)


class AugmentedDataLoader:
    """
    Wrapper that adds augmentation to an existing data loader.

    Multiplies the effective dataset size by the number of symmetry
    transforms while keeping memory usage constant.
    """

    def __init__(
        self,
        base_loader,
        augmentor: DataAugmentor,
        mode: str = "expand",
    ):
        """
        Initialize augmented loader.

        Args:
            base_loader: Base data loader to wrap
            augmentor: DataAugmentor instance
            mode: Augmentation mode:
                - "expand": Each sample generates N augmented versions
                - "random": Each sample gets one random transform
                - "online": Apply random transform during iteration
        """
        self.base_loader = base_loader
        self.augmentor = augmentor
        self.mode = mode

    def __iter__(self):
        """Iterate with augmentation."""
        for batch in self.base_loader:
            if self.mode == "random" or self.mode == "online":
                # Apply single random transform per sample
                yield self._augment_batch_random(batch)
            elif self.mode == "expand":
                # Yield all augmented versions
                yield from self._augment_batch_expand(batch)
            else:
                yield batch

    def _augment_batch_random(self, batch):
        """Apply random augmentation to batch."""
        features, globals_vec, policy_idx, policy_val, values = batch

        aug_features, aug_globals, aug_idx, aug_val = self.augmentor.augment_batch(
            features, globals_vec, policy_idx, policy_val
        )

        return (aug_features, aug_globals, aug_idx, aug_val, values)

    def _augment_batch_expand(self, batch):
        """Expand batch with all augmentations."""
        features, globals_vec, policy_idx, policy_val, values = batch

        for t in self.augmentor.get_all_transforms():
            aug_features = np.zeros_like(features)
            aug_idx_list = []
            aug_val_list = []

            for i in range(features.shape[0]):
                aug_features[i] = self.augmentor.transform_board(features[i], t)
                aug_i, aug_v = self.augmentor.transform_sparse_policy(
                    policy_idx[i], policy_val[i], t
                )
                aug_idx_list.append(aug_i)
                aug_val_list.append(aug_v)

            yield (aug_features, globals_vec.copy(), aug_idx_list, aug_val_list, values)

    def __len__(self):
        """Return effective length."""
        base_len = len(self.base_loader)
        if self.mode == "expand":
            return base_len * self.augmentor.num_transforms
        return base_len


def create_augmentor(
    board_type: Union[str, BoardType],
    enabled: bool = True,
    mode: str = "all",
) -> DataAugmentor:
    """
    Factory function to create appropriately configured augmentor.

    Args:
        board_type: Board type
        enabled: Whether augmentation is enabled
        mode: Augmentation mode:
            - "all": Use all symmetry transforms
            - "random": Sample random transforms
            - "light": Use only rotations (no reflections)

    Returns:
        Configured DataAugmentor
    """
    if mode == "all":
        config = AugmentationConfig(
            enabled=enabled,
            all_transforms=True,
        )
    elif mode == "random":
        config = AugmentationConfig(
            enabled=enabled,
            all_transforms=False,
            num_random_transforms=4,
        )
    elif mode == "light":
        # Only rotations
        config = AugmentationConfig(
            enabled=enabled,
            all_transforms=False,
            num_random_transforms=4,
            exclude_identity=False,
        )
    else:
        config = AugmentationConfig(enabled=enabled)

    return DataAugmentor(board_type, config)


def get_augmentation_factor(board_type: Union[str, BoardType]) -> int:
    """
    Get the number of symmetry transforms for a board type.

    Args:
        board_type: Board type

    Returns:
        Number of symmetry transforms (8 for square, 12 for hex)
    """
    if isinstance(board_type, str):
        board_type = BoardType(board_type)

    if board_type == BoardType.HEXAGONAL:
        return 12
    else:
        return 8
