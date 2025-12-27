"""Training data loading utilities for RingRift AI.

This module provides factory functions to consolidate data loading logic from train.py.
It handles streaming vs memory-based loading, data validation, and distributed sampling.

December 2025: Extracted from train.py to improve modularity (Wave 4).
"""

from __future__ import annotations

import glob
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

if TYPE_CHECKING:
    from app.training.train_config import BoardType, FullTrainingConfig

logger = logging.getLogger(__name__)

# Auto-streaming threshold: 2GB by default, configurable via env var
AUTO_STREAMING_THRESHOLD_BYTES = int(
    os.environ.get("RINGRIFT_AUTO_STREAMING_THRESHOLD_GB", "2")
) * (1024 ** 3)


# =============================================================================
# Lazy imports for optional dependencies
# =============================================================================

def _try_import_streaming_loader():
    """Lazy import StreamingDataLoader."""
    try:
        from app.training.streaming_data_loader import StreamingDataLoader
        return StreamingDataLoader
    except ImportError:
        return None


def _try_import_weighted_streaming_loader():
    """Lazy import WeightedStreamingDataLoader."""
    try:
        from app.training.streaming_data_loader import WeightedStreamingDataLoader
        return WeightedStreamingDataLoader
    except ImportError:
        return None


def _try_import_dataset():
    """Lazy import RingRiftDataset."""
    try:
        from app.training.dataset import RingRiftDataset
        return RingRiftDataset
    except ImportError:
        return None


def _try_import_weighted_dataset():
    """Lazy import WeightedRingRiftDataset."""
    try:
        from app.training.dataset import WeightedRingRiftDataset
        return WeightedRingRiftDataset
    except ImportError:
        return None


def _try_import_data_catalog():
    """Lazy import DataCatalog."""
    try:
        from app.distributed.data_catalog import get_data_catalog
        return get_data_catalog
    except ImportError:
        return None


def _try_import_safe_npz():
    """Lazy import safe_load_npz."""
    try:
        from app.utils.npz_utils import safe_load_npz
        return safe_load_npz
    except ImportError:
        return None


def _try_import_distributed():
    """Lazy import distributed utilities."""
    try:
        from app.training.distributed_utils import (
            get_distributed_sampler,
            get_rank,
            get_world_size,
        )
        return get_distributed_sampler, get_rank, get_world_size
    except ImportError:
        return None, None, None


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class DataLoaderConfig:
    """Configuration for data loading."""

    data_path: str | list[str] | None = None
    data_dir: str | None = None
    use_streaming: bool = False
    batch_size: int = 512
    sampling_weights: str = "uniform"
    val_split: float = 0.2
    num_workers: int = 0
    discover_synced_data: bool = False
    filter_empty_policies: bool = True
    enable_elo_weighting: bool = False
    min_quality_score: float = 0.0
    seed: int = 42
    policy_size: int = 64

    @classmethod
    def from_config(cls, config: FullTrainingConfig) -> DataLoaderConfig:
        """Create from FullTrainingConfig."""
        return cls(
            data_path=getattr(config, 'data_path', None),
            data_dir=getattr(config, 'data_dir', None),
            use_streaming=getattr(config, 'use_streaming', False),
            batch_size=config.batch_size,
            sampling_weights=getattr(config, 'sampling_weights', 'uniform'),
            val_split=getattr(config, 'val_split', 0.2),
            seed=config.seed,
        )


@dataclass
class DatasetMetadata:
    """Metadata extracted from NPZ dataset files."""

    in_channels: int | None = None
    globals_dim: int | None = None
    history_length: int | None = None
    feature_version: int | None = None
    policy_encoding: str | None = None
    encoder_type: str | None = None
    encoder_version: str | None = None
    base_channels: int | None = None
    board_type: str | None = None
    in_channels_from_metadata: int | None = None


@dataclass
class DataLoaderResult:
    """Result of data loader setup."""

    train_loader: DataLoader | Any | None = None
    val_loader: DataLoader | Any | None = None
    train_streaming_loader: Any | None = None
    val_streaming_loader: Any | None = None
    train_sampler: Any | None = None
    train_size: int = 0
    val_size: int = 0
    total_samples: int = 0
    feature_shape: tuple | None = None
    use_streaming: bool = False
    value_only_training: bool = False
    has_multi_player_values: bool = False
    full_dataset: Any | None = None


# =============================================================================
# Data Path Collection
# =============================================================================

def collect_data_paths(
    data_path: str | list[str] | None,
    data_dir: str | None,
    discover_synced_data: bool = False,
    board_type: str | None = None,
    num_players: int | None = None,
) -> list[str]:
    """Collect all data paths for training.

    Args:
        data_path: Single path or list of paths to NPZ files.
        data_dir: Directory containing NPZ files.
        discover_synced_data: Use DataCatalog to discover additional sources.
        board_type: Board type for filtering discovered data.
        num_players: Player count for filtering discovered data.

    Returns:
        Deduplicated list of data paths.
    """
    data_paths: list[str] = []

    # Quality-aware data discovery from synced sources
    if discover_synced_data:
        get_data_catalog = _try_import_data_catalog()
        if get_data_catalog:
            try:
                catalog = get_data_catalog()
                discovered_paths = catalog.get_recommended_training_sources(
                    target_games=100000,
                    board_type=board_type,
                    num_players=num_players,
                )
                if discovered_paths:
                    data_paths.extend([str(p) for p in discovered_paths])
                    stats = catalog.get_stats()
                    logger.info(
                        f"DataCatalog discovered {len(discovered_paths)} sources "
                        f"with {stats.total_games} total games "
                        f"(avg quality: {stats.avg_quality_score:.3f})"
                    )
            except (ImportError, AttributeError, OSError, ConnectionError) as e:
                logger.warning(f"DataCatalog discovery failed: {e}")

    # Collect from data_dir
    if data_dir is not None:
        npz_pattern = os.path.join(data_dir, "*.npz")
        dir_paths = sorted(glob.glob(npz_pattern))
        data_paths.extend(dir_paths)
        if dir_paths:
            logger.info(f"Found {len(dir_paths)} .npz files in {data_dir}")

    # Collect from data_path
    if isinstance(data_path, list):
        data_paths.extend(data_path)
    elif data_path:
        data_paths.append(data_path)

    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique_paths: list[str] = []
    for p in data_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    return unique_paths


def get_total_data_size(paths: list[str]) -> int:
    """Calculate total size of data files in bytes."""
    total_size = 0
    for p in paths:
        if os.path.exists(p):
            total_size += os.path.getsize(p)
    return total_size


def should_use_streaming(
    paths: list[str],
    force_streaming: bool = False,
    threshold_bytes: int = AUTO_STREAMING_THRESHOLD_BYTES,
) -> bool:
    """Determine if streaming mode should be used.

    Args:
        paths: List of data file paths.
        force_streaming: If True, always use streaming.
        threshold_bytes: Size threshold for auto-enabling streaming.

    Returns:
        True if streaming mode should be used.
    """
    if force_streaming:
        return True

    total_size = get_total_data_size(paths)
    if total_size > threshold_bytes:
        size_gb = total_size / (1024 ** 3)
        threshold_gb = threshold_bytes / (1024 ** 3)
        logger.warning(
            f"Auto-enabling streaming mode: dataset size {size_gb:.1f}GB "
            f"exceeds threshold {threshold_gb:.0f}GB. "
            f"Set RINGRIFT_AUTO_STREAMING_THRESHOLD_GB to adjust or "
            f"use --use-streaming explicitly."
        )
        return True

    return False


# =============================================================================
# Dataset Metadata Extraction
# =============================================================================

def extract_dataset_metadata(npz_path: str) -> DatasetMetadata:
    """Extract metadata from an NPZ dataset file.

    Args:
        npz_path: Path to the NPZ file.

    Returns:
        DatasetMetadata with extracted values.
    """
    metadata = DatasetMetadata()
    safe_load_npz = _try_import_safe_npz()

    if not safe_load_npz or not os.path.exists(npz_path):
        return metadata

    try:
        with safe_load_npz(npz_path, mmap_mode="r") as d:
            # Feature shape
            if "features" in d:
                feat_shape = d["features"].shape
                if len(feat_shape) >= 2:
                    metadata.in_channels = int(feat_shape[1])

            # Globals dimension
            if "globals" in d:
                glob_shape = d["globals"].shape
                if len(glob_shape) >= 2:
                    metadata.globals_dim = int(glob_shape[1])

            # String/scalar metadata fields
            for field_name, attr_name in [
                ("policy_encoding", "policy_encoding"),
                ("encoder_type", "encoder_type"),
                ("encoder_version", "encoder_version"),
                ("board_type", "board_type"),
            ]:
                if field_name in d:
                    try:
                        setattr(metadata, attr_name, str(np.asarray(d[field_name]).item()))
                    except (ValueError, TypeError, AttributeError):
                        pass

            # Integer metadata fields
            for field_name, attr_name in [
                ("history_length", "history_length"),
                ("feature_version", "feature_version"),
                ("base_channels", "base_channels"),
                ("in_channels", "in_channels_from_metadata"),
            ]:
                if field_name in d:
                    try:
                        setattr(metadata, attr_name, int(np.asarray(d[field_name]).item()))
                    except (ValueError, TypeError, AttributeError):
                        pass

    except (OSError, KeyError, ValueError) as exc:
        logger.warning(f"Failed to read dataset metadata from {npz_path}: {exc}")

    return metadata


def validate_dataset_metadata(
    metadata: DatasetMetadata,
    config_history_length: int,
    config_feature_version: int,
    model_version: str,
    use_hex_model: bool = False,
    use_hex_v3: bool = False,
    npz_path: str = "",
) -> None:
    """Validate dataset metadata against training configuration.

    Raises:
        ValueError: If metadata doesn't match configuration.
    """
    # History length validation
    if metadata.history_length is not None:
        if metadata.history_length != config_history_length:
            raise ValueError(
                f"Training history_length does not match dataset metadata.\n"
                f"  dataset={npz_path}\n"
                f"  dataset_history_length={metadata.history_length}\n"
                f"  config.history_length={config_history_length}\n"
                "Regenerate the dataset with matching --history-length or "
                "update the training config."
            )
    elif config_history_length != 3:
        logger.warning(
            f"Dataset {npz_path} missing history_length metadata; using "
            f"config.history_length={config_history_length}. Ensure the dataset "
            "was built with matching history frames."
        )

    # Feature version validation
    if metadata.feature_version is not None:
        if metadata.feature_version != config_feature_version:
            raise ValueError(
                f"Training feature_version does not match dataset metadata.\n"
                f"  dataset={npz_path}\n"
                f"  dataset_feature_version={metadata.feature_version}\n"
                f"  config_feature_version={config_feature_version}\n"
                "Regenerate the dataset with matching --feature-version or "
                "update the training config."
            )
    elif config_feature_version != 1:
        raise ValueError(
            f"Dataset is missing feature_version metadata but training "
            f"was configured for feature_version={config_feature_version}.\n"
            f"  dataset={npz_path}\n"
            "Regenerate the dataset with --feature-version or "
            "set feature_version=1 to use legacy features."
        )

    # Globals dimension validation
    if metadata.globals_dim is None:
        if npz_path.endswith(".npz"):
            raise ValueError(
                f"Dataset is missing globals features required for training.\n"
                f"  dataset={npz_path}\n"
                "Regenerate the dataset with scripts/export_replay_dataset.py."
            )
    elif metadata.globals_dim != 20:
        raise ValueError(
            f"Dataset globals feature dimension does not match the CNN encoder.\n"
            f"  dataset={npz_path}\n"
            f"  dataset_globals_dim={metadata.globals_dim}\n"
            "Regenerate the dataset with scripts/export_replay_dataset.py "
            "to produce 20 global features."
        )

    # Channel validation
    if metadata.in_channels is not None:
        if use_hex_model:
            hex_base = 16 if use_hex_v3 else 10
            expected_in_channels = hex_base * (config_history_length + 1)
            expected_encoder = "hex_v3" if use_hex_v3 else "hex_v2"
        else:
            expected_in_channels = 14 * (config_history_length + 1)
            expected_encoder = "square"

        # Cross-validate in_channels from metadata against actual feature shape
        if metadata.in_channels_from_metadata is not None:
            if metadata.in_channels_from_metadata != metadata.in_channels:
                raise ValueError(
                    "========================================\n"
                    "DATA INTEGRITY ERROR - METADATA MISMATCH\n"
                    "========================================\n"
                    f"Dataset in_channels metadata: {metadata.in_channels_from_metadata}\n"
                    f"Actual feature shape:         {metadata.in_channels} channels\n"
                    f"Dataset:                      {npz_path}\n"
                    "\n"
                    "The export script recorded a channel count that doesn't match\n"
                    "the actual feature tensor shape. This indicates a bug in the\n"
                    "export pipeline.\n"
                    "\n"
                    "SOLUTION: Re-export the data with a fixed export script.\n"
                    "========================================"
                )

        # Encoder type validation
        if metadata.encoder_type and metadata.encoder_type != expected_encoder:
            raise ValueError(
                "========================================\n"
                "ENCODER TYPE MISMATCH - CANNOT TRAIN\n"
                "========================================\n"
                f"Dataset encoded with: {metadata.encoder_type}\n"
                f"Model expects:        {expected_encoder}\n"
                f"Model version:        {model_version}\n"
                f"Dataset:              {npz_path}\n"
                "\n"
                "SOLUTION: Re-export data with --encoder-version matching model version\n"
                f"  For v3 model: use --encoder-version v3\n"
                f"  For v2 model: use --encoder-version v2\n"
                "========================================"
            )

        # Channel count validation
        if metadata.in_channels != expected_in_channels:
            encoder_info = ""
            if metadata.encoder_type:
                encoder_info = f"  dataset_encoder_type={metadata.encoder_type}\n"
                encoder_info += f"  dataset_base_channels={metadata.base_channels}\n"
                if metadata.board_type:
                    encoder_info += f"  dataset_board_type={metadata.board_type}\n"

            raise ValueError(
                f"Dataset feature channels do not match the expected encoder.\n"
                f"  dataset={npz_path}\n"
                f"  dataset_in_channels={metadata.in_channels}\n"
                f"  expected_in_channels={expected_in_channels} ({expected_encoder})\n"
                f"{encoder_info}"
                f"Model expects {expected_encoder} encoder ({expected_in_channels} channels).\n"
                "Solutions:\n"
                f"  1. Regenerate dataset with matching encoder version:\n"
                f"     --encoder-version {'v3' if use_hex_v3 else 'v2'}\n"
                f"  2. Or use matching model version for your data:\n"
                f"     --model-version {'v2' if metadata.in_channels == 40 else 'v3' if metadata.in_channels == 64 else 'unknown'}"
            )


# =============================================================================
# Sample Count and Weighting
# =============================================================================

def get_sample_count(npz_path: str) -> int:
    """Get the number of samples in an NPZ file."""
    safe_load_npz = _try_import_safe_npz()
    if not safe_load_npz or not os.path.exists(npz_path):
        return 0

    try:
        with safe_load_npz(npz_path, mmap_mode="r") as d:
            if "features" in d:
                return d["features"].shape[0]
    except (OSError, KeyError):
        pass

    return 0


def load_elo_weights(
    npz_path: str,
    model_elo: float = 1500.0,
    elo_scale: float = 400.0,
    min_weight: float = 0.2,
    max_weight: float = 3.0,
) -> np.ndarray | None:
    """Load ELO-based sample weights from NPZ file.

    Higher-rated opponents yield higher sample weights.

    Returns:
        Array of sample weights, or None if not available.
    """
    safe_load_npz = _try_import_safe_npz()
    if not safe_load_npz or not os.path.exists(npz_path):
        return None

    try:
        with safe_load_npz(npz_path, mmap_mode="r") as npz_data:
            if "opponent_elo" not in npz_data:
                return None

            from app.training.elo_weighting import compute_elo_weights

            opponent_elos = np.array(npz_data["opponent_elo"])
            weights = compute_elo_weights(
                opponent_elos,
                model_elo=model_elo,
                elo_scale=elo_scale,
                min_weight=min_weight,
                max_weight=max_weight,
            )
            logger.info(
                f"ELO weighting enabled: {len(weights)} samples, "
                f"weight range [{weights.min():.3f}, {weights.max():.3f}]"
            )
            return weights

    except (OSError, KeyError, ValueError, ImportError) as e:
        logger.warning(f"Failed to load ELO weights: {e}")
        return None


def load_quality_weights(
    npz_path: str,
    min_quality_score: float = 0.0,
) -> np.ndarray | None:
    """Load quality-based sample weights from NPZ file.

    Higher-quality games yield higher sample weights.

    Returns:
        Array of sample weights, or None if not available.
    """
    safe_load_npz = _try_import_safe_npz()
    if not safe_load_npz or not os.path.exists(npz_path):
        return None

    try:
        with safe_load_npz(npz_path, mmap_mode="r") as npz_data:
            if "quality_score" not in npz_data:
                return None

            quality_scores = np.array(npz_data["quality_score"])

            if min_quality_score > 0.0:
                quality_mask = quality_scores >= min_quality_score
                num_filtered = np.sum(~quality_mask)
                logger.info(
                    f"Quality filtering: {num_filtered} samples below threshold "
                    f"({min_quality_score:.2f}) will be weighted to 0"
                )
                weights = np.where(quality_mask, quality_scores, 0.0)
            else:
                weights = quality_scores

            nonzero = weights[weights > 0]
            if len(nonzero) > 0:
                logger.info(
                    f"Quality weighting enabled: {len(weights)} samples, "
                    f"weight range [{nonzero.min():.3f}, {nonzero.max():.3f}]"
                )
            return weights

    except (OSError, KeyError, ValueError) as e:
        logger.warning(f"Failed to load quality scores: {e}")
        return None


# =============================================================================
# DataLoader Creation
# =============================================================================

def get_num_loader_workers(use_streaming: bool = False) -> int:
    """Determine number of DataLoader workers based on platform.

    Returns 0 on macOS (mmap incompatible with multiprocessing).
    On Linux, returns moderate parallelism for non-streaming mode.
    """
    env_workers = os.environ.get("RINGRIFT_DATALOADER_WORKERS")
    if env_workers is not None:
        return int(env_workers)

    if sys.platform == "darwin":
        return 0  # macOS: mmap incompatible with multiprocessing

    if use_streaming:
        return 0  # Streaming mode: single worker for safety

    # Linux/Windows: use moderate parallelism
    import multiprocessing
    return min(4, multiprocessing.cpu_count() // 2)


def create_streaming_loaders(
    data_paths: list[str],
    config: DataLoaderConfig,
    policy_size: int,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[Any, Any, int, int]:
    """Create streaming data loaders for large datasets.

    Returns:
        Tuple of (train_loader, val_loader, train_samples, val_samples).
    """
    StreamingDataLoader = _try_import_streaming_loader()
    WeightedStreamingDataLoader = _try_import_weighted_streaming_loader()

    if not StreamingDataLoader:
        raise ImportError("StreamingDataLoader not available")

    # Get total sample count
    total_samples = sum(
        get_sample_count(p) for p in data_paths if os.path.exists(p)
    )

    if total_samples == 0:
        raise ValueError("No samples found in data files")

    logger.info(
        f"StreamingDataLoader: {total_samples} total samples "
        f"across {len(data_paths)} files"
    )

    # Calculate split
    val_samples = int(total_samples * config.val_split)
    train_samples = total_samples - val_samples

    # Create loaders based on sampling strategy
    filter_empty = config.filter_empty_policies

    if config.sampling_weights != 'uniform' and WeightedStreamingDataLoader:
        train_loader = WeightedStreamingDataLoader(
            data_paths=data_paths,
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.seed,
            drop_last=False,
            policy_size=policy_size,
            rank=rank,
            world_size=world_size,
            filter_empty_policies=filter_empty,
            sampling_weights=config.sampling_weights,
        )
        logger.info(
            f"Using WeightedStreamingDataLoader with "
            f"sampling_weights={config.sampling_weights}"
        )
    else:
        train_loader = StreamingDataLoader(
            data_paths=data_paths,
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.seed,
            drop_last=False,
            policy_size=policy_size,
            rank=rank,
            world_size=world_size,
            filter_empty_policies=filter_empty,
        )

    # Validation loader always uses uniform sampling
    val_loader = StreamingDataLoader(
        data_paths=data_paths,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed + 1000,
        drop_last=False,
        policy_size=policy_size,
        rank=rank,
        world_size=world_size,
        filter_empty_policies=filter_empty,
    )

    return train_loader, val_loader, train_samples, val_samples


def create_memory_loaders(
    data_path: str,
    config: DataLoaderConfig,
    board_type: Any,
    augment_hex: bool = False,
    multi_player: bool = False,
    use_heuristics: bool = False,
    elo_weights: np.ndarray | None = None,
    quality_weights: np.ndarray | None = None,
    distributed: bool = False,
) -> tuple[DataLoader, DataLoader, Any, int, int, Any]:
    """Create memory-based data loaders for smaller datasets.

    Returns:
        Tuple of (train_loader, val_loader, train_sampler, train_size, val_size, full_dataset).
    """
    RingRiftDataset = _try_import_dataset()
    WeightedRingRiftDataset = _try_import_weighted_dataset()

    if not RingRiftDataset:
        raise ImportError("RingRiftDataset not available")

    filter_empty = config.filter_empty_policies

    # Create dataset based on sampling strategy
    if config.sampling_weights == 'uniform':
        full_dataset = RingRiftDataset(
            data_path,
            board_type=board_type,
            augment_hex=augment_hex,
            use_multi_player_values=multi_player,
            filter_empty_policies=filter_empty,
            return_num_players=multi_player,
            return_heuristics=use_heuristics,
        )
        use_weighted_sampling = False
    else:
        if not WeightedRingRiftDataset:
            raise ImportError("WeightedRingRiftDataset not available")
        full_dataset = WeightedRingRiftDataset(
            data_path,
            board_type=board_type,
            augment_hex=augment_hex,
            weighting=config.sampling_weights,
            use_multi_player_values=multi_player,
            filter_empty_policies=filter_empty,
            return_num_players=multi_player,
            return_heuristics=use_heuristics,
        )
        use_weighted_sampling = True

    if len(full_dataset) == 0:
        raise ValueError(f"Training dataset at {data_path} is empty")

    # Split into train/val
    train_size = int((1.0 - config.val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )

    num_workers = get_num_loader_workers(use_streaming=False)
    train_sampler = None

    if distributed:
        get_distributed_sampler, _, _ = _try_import_distributed()
        if get_distributed_sampler:
            train_sampler = get_distributed_sampler(train_dataset, shuffle=True)
            val_sampler = get_distributed_sampler(val_dataset, shuffle=False)
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=num_workers,
                pin_memory=True,
            )
            return train_loader, val_loader, train_sampler, train_size, val_size, full_dataset

    # Non-distributed: optionally use weighted sampling
    use_any_weighting = (
        use_weighted_sampling
        or (elo_weights is not None)
        or (quality_weights is not None)
    )

    if use_any_weighting and isinstance(train_dataset, torch.utils.data.Subset):
        subset_indices = np.array(train_dataset.indices, dtype=np.int64)

        # Start with position-based weights if available
        if use_weighted_sampling and hasattr(full_dataset, 'sample_weights'):
            if full_dataset.sample_weights is None:
                train_weights_np = np.ones(len(train_dataset), dtype=np.float32)
            else:
                train_weights_np = full_dataset.sample_weights[subset_indices].astype(np.float32)
        else:
            train_weights_np = np.ones(len(train_dataset), dtype=np.float32)

        # Apply ELO weights multiplicatively
        if elo_weights is not None:
            elo_weights_subset = elo_weights[subset_indices].astype(np.float32)
            train_weights_np = train_weights_np * elo_weights_subset

        # Apply quality weights multiplicatively
        if quality_weights is not None:
            quality_weights_subset = quality_weights[subset_indices].astype(np.float32)
            train_weights_np = train_weights_np * quality_weights_subset

        # Log combined weights
        weight_sources = []
        if use_weighted_sampling:
            weight_sources.append("position")
        if elo_weights is not None:
            weight_sources.append("ELO")
        if quality_weights is not None:
            weight_sources.append("quality")

        nonzero = train_weights_np[train_weights_np > 0]
        if len(nonzero) > 0:
            logger.info(
                f"Combined weights ({' * '.join(weight_sources)}): "
                f"{len(nonzero)}/{len(train_weights_np)} samples with weight > 0, "
                f"range [{nonzero.min():.3f}, {nonzero.max():.3f}]"
            )

        train_weights = torch.from_numpy(train_weights_np)
        train_sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=train_sampler,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, train_sampler, train_size, val_size, full_dataset


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    'DataLoaderConfig',
    'DataLoaderResult',
    'DatasetMetadata',
    # Path collection
    'collect_data_paths',
    'get_total_data_size',
    'should_use_streaming',
    # Metadata
    'extract_dataset_metadata',
    'validate_dataset_metadata',
    # Weights
    'get_sample_count',
    'load_elo_weights',
    'load_quality_weights',
    # Loaders
    'create_streaming_loaders',
    'create_memory_loaders',
    'get_num_loader_workers',
    # Constants
    'AUTO_STREAMING_THRESHOLD_BYTES',
]
